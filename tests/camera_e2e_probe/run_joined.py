"""One synthetic Telegent -> HTTP -> Ohmo Camera admission probe.

Run with OpenHarness's locked test environment; Telegent imports resolve from
CAMERA_TELEGENT_WORKTREE. Dropbox and Telegram use fixtures. The classifier
and its route attestation are synthetic fixture data, so this is a partial E2E.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
TELEGENT = Path(os.environ["CAMERA_TELEGENT_WORKTREE"]).resolve()
sys.path[:0] = [str(ROOT / "tests/test_ohmo"), str(TELEGENT / "telegent/health_advisor")]
sys.path.insert(0, str(TELEGENT))

from test_camera_ingress import FakeTelegram, _deepseek_route_json  # noqa: E402
from probe_support import (  # noqa: E402
    create_storage_run_dir,
    require_bound_answer,
    select_finalizer_event,
    source_jpeg,
    unique_honcho_scope,
)
from telegent.health_advisor import test_dropbox_camera_pipeline as fixtures  # noqa: E402
from telegent.health_advisor.dropbox_camera.camera_submission import CameraSubmissionClient  # noqa: E402
from telegent.health_advisor.dropbox_camera.clip_prefilter import (  # noqa: E402
    ClipFoodPrefilter,
    ClipPrefilterRelease,
)
from ohmo.gateway.camera import CameraIngress, serve_camera_http  # noqa: E402
from ohmo.gateway.config import save_gateway_config  # noqa: E402
from ohmo.gateway.models import CameraIngressConfig, GatewayConfig  # noqa: E402
from ohmo.gateway.runtime import OhmoSessionRuntimePool  # noqa: E402
from ohmo.evals.nutrition_trace import NutritionAnnotationV2  # noqa: E402
from ohmo.memory_service.honcho_client import HonchoClient  # noqa: E402
from openharness.channels.bus.events import InboundMessage  # noqa: E402
from openharness.channels.bus.queue import MessageBus  # noqa: E402


async def verify_finalizer_event(
    *,
    url: str,
    workspace: str,
    session: str,
    candidate_id: str,
    answer_message_id: str,
    started: datetime,
    expected_capture_time: datetime,
) -> None:
    """Read the real finalizer's event, then verify Telegent's exact projection."""
    from mcp.server.fastmcp.server import FastMCP
    from mcp.types import ToolAnnotations
    from telegent.health_advisor.nutrition.config import NutritionHonchoCredentials
    from telegent.health_advisor.nutrition.store import NutritionDataStore
    from telegent.health_advisor.nutrition.sync import sync_nutrition_honcho
    from telegent.health_advisor.storage import HealthDataStore
    from telegent.mcp_simple_auth.wellness import register_wellness_tools

    async with HonchoClient(url, "local-auth-disabled", workspace) as client:
        messages = await client.list_recent_message_metadata(
            session,
            expected_peer_id="ohmo",
            since=started - timedelta(minutes=1),
            until=datetime.now(UTC) + timedelta(minutes=1),
        )
    event = select_finalizer_event(messages, candidate_id, answer_message_id)
    trace = event.metadata.get("decision_trace")
    annotation = trace.get("annotations", {}).get("nutrition") if isinstance(trace, dict) else None
    try:
        validated = NutritionAnnotationV2.model_validate(annotation)
    except ValidationError:
        raise AssertionError("finalizer nutrition metadata is invalid") from None
    if not (
        validated.record_type == "meal_observation"
        and validated.consumption_status == "consumed"
        and "image" in validated.basis
        and validated.energy_kcal_best is not None
    ):
        raise AssertionError("finalizer event has no consumed image meal with kcal")
    if validated.meal_at != expected_capture_time or validated.meal_date is not None:
        raise AssertionError("finalizer meal date differs from admitted Camera capture time")

    credentials = NutritionHonchoCredentials.model_validate(
        {
            "schema_version": 1,
            "base_url": url,
            "sources": [
                {
                    "user_id": "synthetic_owner",
                    "workspace": workspace,
                    "session": session,
                    "workspace_jwt": "local-auth-disabled",
                    "start_at": (started - timedelta(minutes=1)).isoformat(),
                    "device_ids": [],
                    "timezone": "UTC",
                }
            ],
        }
    )
    with nullcontext(create_storage_run_dir(ROOT)) as directory:
        print(f"Finalizer projection retained at {directory}", flush=True)
        db = NutritionDataStore(directory / "nutrition.db")
        health = HealthDataStore(directory / "health.db")
        try:
            app = FastMCP(name="camera-finalizer-probe")

            async def get_health_store():
                return health

            async def get_nutrition_store():
                return db

            register_wellness_tools(
                app,
                read_only_annotations=ToolAnnotations(readOnlyHint=True),
                get_health_store=get_health_store,
                get_nutrition_store=get_nutrition_store,
                wellness_user_id="synthetic_owner",
            )

            async def balance():
                result = await app.call_tool(
                    "get_wellness_data",
                    {
                        "params": {
                            "start": (expected_capture_time - timedelta(days=1)).isoformat(),
                            "end": (expected_capture_time + timedelta(days=1)).isoformat(),
                        }
                    },
                )
                meals = result[1]["nutrition_records"]
                return meals, sum(meal["energy_kcal_best"] or 0 for meal in meals)

            await sync_nutrition_honcho(credentials=credentials, db=db)
            record = db.get_record_by_event_id("synthetic_owner", event.id)
            if record is None:
                raise AssertionError("finalizer event did not enter Telegent projection")
            meal = db.get_current_meal("synthetic_owner", record.meal_id)
            meals, kcal = await balance()
            if not (
                meal is not None
                and meal.event_ids == [event.id]
                and meal.latest_event_id == event.id
                and len(meals) == 1
                and meals[0]["latest_event_id"] == event.id
                and kcal == validated.energy_kcal_best
            ):
                raise AssertionError("exact finalizer event or wellness kcal mismatch")
            snapshot = meal.model_dump(mode="json")
            await sync_nutrition_honcho(credentials=credentials, db=db)
            replay_meals, replay_kcal = await balance()
            if not (
                db.get_current_meal("synthetic_owner", record.meal_id).model_dump(mode="json")
                == snapshot
                and replay_meals == meals
                and replay_kcal == kcal
            ):
                raise AssertionError("replay changed the finalizer meal balance")
            print(f"PASS finalizer event={event.id} kcal={kcal} replay=stable")
        finally:
            health.close()
            db.close()


class LocalCameraTransport(httpx.BaseTransport):
    def __init__(self, port: int) -> None:
        self._transport = httpx.HTTPTransport(proxy=None)
        self._port = port

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # The production client requires its fixed private VPN origin. Keep
        # its exact multipart bytes while routing this test to loopback.
        target = request.url.copy_with(host="127.0.0.1", port=self._port)
        forwarded = httpx.Request(
            request.method, target, headers=request.headers, content=request.read()
        )
        return self._transport.handle_request(forwarded)

    def close(self) -> None:
        self._transport.close()


class FixtureClassifier(fixtures._Classifier):
    def __init__(self) -> None:
        super().__init__(
            model="deepseek/deepseek-v4.1-flash",
            prompt_version="v1",
            policy_version="deepseek-camera-production-v1",
        )

    def classify(self, *, candidate_id: str, image_path: str | Path):
        decision = super().classify(candidate_id=candidate_id, image_path=image_path)
        return decision.model_copy(
            update={"classifier_route_attestation_json": _deepseek_route_json()}
        )


async def main() -> None:
    positive = os.environ.get("CAMERA_RUN_POSITIVE") == "1"
    private_bytes = source_jpeg(
        os.environ.get("CAMERA_SOURCE_JPEG"), os.environ.get("CAMERA_SOURCE_SHA256"), ROOT
    )
    if positive and private_bytes is None:
        raise ValueError("positive path requires a validated private JPEG")
    with nullcontext(create_storage_run_dir(ROOT)) as root:
        print(f"Joined Camera run retained at {root}", flush=True)
        token = root / "token"
        token.write_text("s" * 40 + "\n", encoding="ascii")
        token.chmod(0o600)
        bus = MessageBus()
        telegram = FakeTelegram()
        ingress = CameraIngress(
            CameraIngressConfig(
                enabled=True,
                listen_port=18751,
                bearer_token_file=token,
                principal="123",
                tenant_id="synthetic_owner",
                chat_id="123",
                session_key="telegram:123",
            ),
            workspace=root,
            bus=bus,
            telegram=telegram,
        )
        server = await asyncio.start_server(
            lambda reader, writer: serve_camera_http(ingress, reader, writer),
            host="127.0.0.1",
            port=0,
        )
        port = server.sockets[0].getsockname()[1]
        source_bytes = private_bytes if private_bytes is not None else fixtures._image_bytes(root)
        entry = fixtures._entry("id:camera-joined", "rev:1").model_copy(
            update={"size": len(source_bytes)}
        )
        candidate_id = fixtures.candidate_id_for(entry.file_id, entry.rev)
        artifacts = root / "artifacts"
        store = fixtures._test_store(artifacts)
        http_client = httpx.Client(transport=LocalCameraTransport(port), timeout=10)
        client = CameraSubmissionClient(
            store=store,
            base_url="http://10.8.0.8:18751",
            service_token="s" * 40,
            http_client=http_client,
        )
        clip_release = ClipPrefilterRelease.model_validate(
            json.loads(
                (TELEGENT / "telegent/data/dropbox_camera_clip_prefilter_release.json").read_text()
            )["release"]
        )
        pipeline = fixtures._pipeline(
            fixtures._Source([entry], source_bytes),
            FixtureClassifier(),
            artifacts,
            camera_submission_client=client,
            clip_prefilter=ClipFoodPrefilter(clip_release, scorer=lambda *_: 0.0),
            release=fixtures._active_release(
                model="deepseek/deepseek-v4.1-flash",
                prompt_version="v1",
                policy_version="deepseek-camera-production-v1",
            ),
        )
        try:
            first = await asyncio.to_thread(pipeline.run_once)
            assert first.published == 1, first
            assert first.results[0].submission_outcome == "accepted", first.results[0]
            inbound = await asyncio.wait_for(bus.consume_inbound(), timeout=5)
            assert inbound.metadata["_camera_candidate_id"] == candidate_id
            assert len(telegram.calls) == 1
            snapshot = Path(telegram.calls[0][1])
            assert snapshot.read_bytes() == source_bytes
            assert ingress._attempts[candidate_id]["photo_id"] == 77
            second = await asyncio.to_thread(pipeline.run_once)
            assert second.duplicate_suppression == 1, second
            assert len(telegram.calls) == 1
            if os.environ.get("CAMERA_RUN_MODEL") == "1" or positive:
                # Luna now accepts the image natively through OpenRouter.
                honcho_url = os.environ["CAMERA_HONCHO_URL"]
                workspace, session = unique_honcho_scope()
                save_gateway_config(
                    GatewayConfig(
                        provider_profile="openrouter",
                        enabled_channels=["telegram"],
                        family_principals={"123": "synthetic_owner"},
                        enabled_memory_tenants=("synthetic_owner",),
                        conversation_learning=True,
                        memory_backend="shadow",
                        honcho_base_url=honcho_url,
                        tenant_honcho={
                            "synthetic_owner": {
                                "workspace": workspace,
                                "api_key": "local-auth-disabled",
                                "observed_peer": "owner",
                                "session": session,
                            }
                        },
                        camera_ingress=ingress.config,
                    ),
                    root,
                )
                pool = OhmoSessionRuntimePool(
                    cwd=root,
                    workspace=root,
                    provider_profile="openrouter",
                    model="openai/gpt-6-luna",
                    max_turns=4,
                    effort="none",
                )
                pool._camera_ingress = ingress
                try:

                    async def model_turn(message: InboundMessage) -> None:
                        try:
                            updates = [
                                item
                                async for item in pool.stream_message(message, message.session_key)
                            ]
                        except Exception as exc:
                            raise AssertionError(
                                f"Model turn failed; exception class: {type(exc).__name__}"
                            ) from None
                        kind_counts = Counter(item.kind for item in updates)
                        if not updates or updates[-1].kind != "final":
                            raise AssertionError(
                                "Model turn ended without final; "
                                f"update kinds/counts: {dict(kind_counts)}"
                            )
                        print(f"PASS Ohmo runtime model turn: updates={len(updates)}")

                    await model_turn(inbound)
                    if positive:
                        answer_text = os.environ.get("CAMERA_OWNER_REPLY", "Я съела 4 сливы")
                        if not answer_text.strip():
                            raise ValueError("positive path requires a nonempty owner reply")
                        answer = InboundMessage(
                            channel="telegram",
                            sender_id="123",
                            chat_id="123",
                            content=answer_text,
                            metadata={
                                "is_group": False,
                                "chat_type": "private",
                                "message_id": 78,
                                "reply_to_message_id": 77,
                                "_telegram_raw_text": answer_text,
                            },
                        )
                        ingress.process_real_inbound(answer)
                        require_bound_answer(answer, candidate_id)
                        expected_capture_time = ingress.trusted_capture_time_for_answer(answer)
                        if expected_capture_time is None:
                            raise AssertionError(
                                "bound answer has no validated Camera capture time"
                            )
                        if answer.session_key != inbound.session_key:
                            raise AssertionError("owner answer escaped the Camera session")
                        started = datetime.now(UTC)
                        await model_turn(answer)
                        await verify_finalizer_event(
                            url=honcho_url,
                            workspace=workspace,
                            session=session,
                            candidate_id=candidate_id,
                            answer_message_id="78",
                            started=started,
                            expected_capture_time=expected_capture_time,
                        )
                finally:
                    await pool.aclose()
            print(
                "PASS joined Telegent HTTP -> Ohmo admission, "
                f"candidate={candidate_id}, image_sha256={hashlib.sha256(source_bytes).hexdigest()}, "
                "native_photo_id=77, replay_sends=0"
            )
        finally:
            server.close()
            await server.wait_closed()
            await ingress.close()
            client.close()
            http_client.close()
            for test_store in fixtures._TEST_STORES.values():
                test_store.close()
            fixtures._TEST_STORES.clear()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        if os.environ.get("CAMERA_RUN_POSITIVE") == "1":
            print(f"FAIL positive Camera probe: {type(exc).__name__}", file=sys.stderr)
            raise SystemExit(1) from None
        raise
