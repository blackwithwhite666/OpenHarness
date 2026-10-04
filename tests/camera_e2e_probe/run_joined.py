"""One synthetic Telegent -> HTTP -> Ohmo Camera admission probe.

Run with OpenHarness's locked test environment; Telegent imports resolve from
CAMERA_TELEGENT_WORKTREE. Dropbox and Telegram use fixtures. The classifier
and its route attestation are synthetic fixture data, so this is a partial E2E.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
TELEGENT = Path(os.environ["CAMERA_TELEGENT_WORKTREE"]).resolve()
sys.path[:0] = [str(ROOT / "tests/test_ohmo"), str(TELEGENT / "telegent/health_advisor")]
sys.path.insert(0, str(TELEGENT))

from test_camera_ingress import _deepseek_route_json  # noqa: E402
from probe_support import (  # noqa: E402
    create_storage_run_dir,
    select_finalizer_event,
    verify_source_worktree,
    unique_honcho_scope,
    native_preflight_and_clients,
)
from telegent.health_advisor import test_dropbox_camera_pipeline as fixtures  # noqa: E402
from telegent.health_advisor.dropbox_camera.camera_submission import CameraSubmissionClient  # noqa: E402
from telegent.health_advisor.dropbox_camera.clip_prefilter import ClipFoodPrefilter  # noqa: E402
from ohmo.gateway.camera import CameraIngress, serve_camera_http  # noqa: E402
from ohmo.gateway.models import CameraIngressConfig  # noqa: E402
from ohmo.evals.nutrition_trace import NutritionAnnotationV2  # noqa: E402
from ohmo.memory_service.honcho_client import HonchoClient  # noqa: E402
from openharness.channels.bus.queue import MessageBus  # noqa: E402
from openharness.channels.impl.telegram import TelegramChannel  # noqa: E402
from openharness.config.schema import TelegramConfig  # noqa: E402
from camera_runtime_support import (  # noqa: E402
    distinct_offline_clients,
    OfflineTelegramBot,
    run_camera_runtime_trajectory,
)


async def verify_finalizer_event(
    *,
    url: str,
    workspace: str,
    session: str,
    candidate_id: str,
    answer_message_id: str,
    native_photo_id: int,
    expected_event_id: str,
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
            until=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
    event = select_finalizer_event(messages, candidate_id, answer_message_id, native_photo_id)
    consumed_events = []
    for item in messages:
        trace_item = item.metadata.get("decision_trace")
        nutrition_item = (
            trace_item.get("annotations", {}).get("nutrition")
            if isinstance(trace_item, dict)
            else None
        )
        if (
            isinstance(nutrition_item, dict)
            and nutrition_item.get("record_type") == "meal_observation"
            and nutrition_item.get("consumption_status") == "consumed"
        ):
            consumed_events.append(item.id)
    if consumed_events != [event.id]:
        raise AssertionError("Honcho full read does not contain exactly one consumed Camera event")
    if event.id != expected_event_id:
        raise AssertionError("Honcho event ID differs from the runtime and Camera receipt IDs")
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
    expected_bindings = {
        "source_principal": "telegram:123",
        "tenant_id": "synthetic_owner",
        "camera_candidate_id": candidate_id,
        "source_message_id": answer_message_id,
        "camera_operation_id": candidate_id,
        "camera_reply_to_native_message_id": str(native_photo_id),
        "source_image_attachment_count": 1,
    }
    binding_mismatches = [
        key for key, expected in expected_bindings.items() if event.metadata.get(key) != expected
    ]
    if (
        not isinstance(event.metadata.get("gateway_session_id"), str)
        or not event.metadata["gateway_session_id"]
    ):
        binding_mismatches.append("gateway_session_id")
    if binding_mismatches:
        raise AssertionError(
            "finalizer event owner/source/operation binding differs: "
            + ",".join(binding_mismatches)
        )

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
            checks = {
                "current_user": meal is not None and meal.user_id == "synthetic_owner",
                "event_ids": meal is not None and meal.event_ids == [event.id],
                "latest_event_id": meal is not None and meal.latest_event_id == event.id,
                "active_current_meal": meal is not None and meal.status.value == "active",
                "consumption_status": meal is not None and meal.consumption_status == "consumed",
                "meal_date": meal is not None
                and meal.local_day(ZoneInfo("UTC")) == expected_capture_time.date().isoformat(),
                "wellness_meal_count": len(meals) == 1,
                "wellness_latest_event_id": len(meals) == 1
                and meals[0]["latest_event_id"] == event.id,
                "wellness_kcal": len(meals) == 1
                and meals[0]["energy_kcal_best"] == validated.energy_kcal_best,
                "effective_balance_kcal": kcal == validated.energy_kcal_best,
            }
            failed_checks = [name for name, passed in checks.items() if not passed]
            if failed_checks:
                raise AssertionError(
                    "exact finalizer event or wellness projection mismatch: "
                    + ",".join(failed_checks)
                )
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


async def verify_zero_meals_before_answer(
    *, url: str, workspace: str, session: str, started: datetime, capture_time: datetime
) -> None:
    """Full Honcho read and real Telegent sync before virtual-owner confirmation."""
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
            until=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
    consumed_events = []
    for item in messages:
        trace = item.metadata.get("decision_trace")
        annotations = trace.get("annotations") if isinstance(trace, dict) else None
        nutrition = annotations.get("nutrition") if isinstance(annotations, dict) else None
        if (
            isinstance(nutrition, dict)
            and nutrition.get("record_type") == "meal_observation"
            and nutrition.get("consumption_status") == "consumed"
        ):
            consumed_events.append(item.id)
    if consumed_events:
        raise AssertionError("analysis-only Camera turn persisted a consumed meal")
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
    directory = create_storage_run_dir(ROOT)
    db = NutritionDataStore(directory / "nutrition.db")
    health = HealthDataStore(directory / "health.db")
    try:
        app = FastMCP(name="camera-before-answer-probe")

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
        await sync_nutrition_honcho(credentials=credentials, db=db)
        result = await app.call_tool(
            "get_wellness_data",
            {
                "params": {
                    "start": (capture_time - timedelta(days=1)).isoformat(),
                    "end": (capture_time + timedelta(days=1)).isoformat(),
                }
            },
        )
        meals = result[1]["nutrition_records"]
        if meals or sum(meal["energy_kcal_best"] or 0 for meal in meals) != 0:
            raise AssertionError("Telegent shows a meal before owner confirmation")
        print(
            "PASS analysis-only Honcho full read + Telegent sync: "
            "consumed_events=0 meals=0 balance_kcal=0",
            flush=True,
        )
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
    logging.basicConfig(level=logging.INFO)
    acceptance = os.environ.get("CAMERA_ACCEPTANCE") == "1"
    expected_openharness = os.environ["CAMERA_OPENHARNESS_SHA"]
    expected_telegent = os.environ["CAMERA_TELEGENT_SHA"]
    openharness_state = verify_source_worktree(ROOT, expected_openharness, require_clean=acceptance)
    telegent_state = verify_source_worktree(TELEGENT, expected_telegent, require_clean=acceptance)
    if acceptance:
        print("SOURCE ACCEPTANCE: clean pinned worktree pair", flush=True)
    else:
        print(
            "OFFLINE SMOKE ONLY: source cleanliness is not an acceptance claim; "
            f"OpenHarness state={openharness_state[1]!r}, Telegent state={telegent_state[1]!r}",
            flush=True,
        )
    mode = os.environ.get("CAMERA_RUN_MODE", "offline")
    if mode not in {"offline", "native"}:
        raise RuntimeError("CAMERA_RUN_MODE must be offline or native")
    if mode == "native" and not acceptance:
        raise RuntimeError(
            "native Camera requires CAMERA_ACCEPTANCE=1 on a clean exact source pair"
        )
    if os.environ.get("CAMERA_RUN_POSITIVE") == "1" or os.environ.get("CAMERA_RUN_MODEL") == "1":
        raise RuntimeError(
            "legacy fixed-reply/OpenRouter model path is disabled; offline smoke does not test Luna"
        )
    native_config = None
    native_source_bytes = None
    if mode == "native":
        from openharness.config.paths import get_config_file_path
        from openharness.config.settings import load_settings

        user_scenario = os.environ.get("CAMERA_USER_SCENARIO", "").strip()
        if not user_scenario:
            raise RuntimeError("native precondition missing: CAMERA_USER_SCENARIO")
        native_config_text = os.environ.get("CAMERA_NATIVE_CONFIG_DIR", "").strip()
        if not native_config_text:
            raise RuntimeError(
                "native precondition missing: lead must bind CAMERA_NATIVE_CONFIG_DIR read-only"
            )
        native_config = Path(native_config_text).resolve(strict=True)
        if not native_config.is_dir():
            raise RuntimeError("native precondition missing: settings binding is not a directory")
        os.environ["OPENHARNESS_CONFIG_DIR"] = str(native_config)
        os.environ["OPENHARNESS_PROFILE"] = "codex"
        settings_path = get_config_file_path()
        if not settings_path.is_file():
            raise RuntimeError(
                "native precondition missing: lead must bind an existing native settings directory"
            )
        settings = load_settings(settings_path)
        (bot_client, user_client), native_source_bytes = native_preflight_and_clients(
            settings,
            scenario=user_scenario,
            source_path=os.environ.get("CAMERA_SOURCE_JPEG"),
            source_sha256=os.environ.get("CAMERA_SOURCE_SHA256"),
            root=ROOT,
        )
    else:
        bot_client, user_client = distinct_offline_clients()
        user_scenario = "synthetic offline owner selects the exact offered confirmation"

    with nullcontext(create_storage_run_dir(ROOT)) as root:
        print(f"Joined Camera run retained at {root}", flush=True)
        run_config = root / "openharness-config"
        run_data = root / "openharness-data"
        run_logs = root / "openharness-logs"
        if mode == "native":
            run_config = native_config
        for directory in (run_data, run_logs):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if mode == "offline":
            run_config.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.environ["OPENHARNESS_CONFIG_DIR"] = str(run_config)
        os.environ["OPENHARNESS_DATA_DIR"] = str(run_data)
        os.environ["OPENHARNESS_LOGS_DIR"] = str(run_logs)
        os.environ["OHMO_MEMORY_AUTOINDEX"] = "0"
        os.environ["OHMO_MEMORY_JUDGE"] = "0"
        token = root / "token"
        token.write_text("s" * 40 + "\n", encoding="ascii")
        token.chmod(0o600)
        bus = MessageBus()
        fake_bot = OfflineTelegramBot()
        telegram = TelegramChannel(
            TelegramConfig(token="offline-no-network", allow_from=["123"]), bus
        )
        from types import SimpleNamespace

        telegram._app = SimpleNamespace(bot=fake_bot)
        telegram.polling_started = True
        telegram._start_typing = lambda _chat_id: None
        telegram._stop_typing = lambda _chat_id: None
        workspace, session = unique_honcho_scope()
        honcho_url = os.environ.get("CAMERA_HONCHO_URL", "http://api:8000")
        ingress = None
        server = None
        client = None
        http_client = None
        try:
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
            source_bytes = (
                native_source_bytes
                if native_source_bytes is not None
                else fixtures._image_at(datetime.now(timezone.utc), "orange")
            )
            if mode == "offline" and native_source_bytes is not None:
                raise AssertionError("offline mode must use only its synthetic fixture image")
            entry = fixtures._entry("id:camera-joined", "rev:1").model_copy(
                update={"size": len(source_bytes), "server_modified": datetime.now(timezone.utc)}
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
            from telegent.health_advisor.dropbox_camera.deepseek_runtime import (
                load_clip_runtime_descriptor,
            )

            clip_release = load_clip_runtime_descriptor(
                TELEGENT / "telegent/data/dropbox_camera_clip_prefilter_release.json"
            )
            pipeline = fixtures._pipeline(
                fixtures._Source([entry], source_bytes),
                FixtureClassifier(),
                artifacts,
                camera_submission_client=client,
                clip_prefilter=ClipFoodPrefilter(clip_release, scorer=lambda *_: 0.0),
                clock=lambda: datetime.now(timezone.utc),
                release=fixtures._active_release(
                    model="deepseek/deepseek-v4.1-flash",
                    prompt_version="v1",
                    policy_version="deepseek-camera-production-v1",
                ),
            )
            first = await asyncio.to_thread(pipeline.run_once)
            assert first.published == 1, first
            assert first.results[0].submission_outcome == "accepted", first.results[0]
            inbound = await asyncio.wait_for(bus.consume_inbound(), timeout=5)
            assert inbound.metadata["_camera_candidate_id"] == candidate_id
            assert len(fake_bot.calls) == 1
            assert fake_bot.photo_bodies == [source_bytes]
            snapshot = Path(ingress._attempts[candidate_id]["snapshot"])
            assert snapshot.read_bytes() == source_bytes
            second = await asyncio.to_thread(pipeline.run_once)
            assert second.duplicate_suppression == 1, second
            assert len(fake_bot.calls) == 1
            capture_time = ingress._attempt_capture_time(ingress._attempts[candidate_id])
            if capture_time is None:
                raise AssertionError("admitted Camera photo has no trusted capture time")

            async def before_answer(started):
                await verify_zero_meals_before_answer(
                    url=honcho_url,
                    workspace=workspace,
                    session=session,
                    started=started,
                    capture_time=capture_time,
                )

            trajectory = await run_camera_runtime_trajectory(
                root=root,
                bus=bus,
                ingress=ingress,
                initial_message=inbound,
                channel=telegram,
                fake_bot=fake_bot,
                candidate_id=candidate_id,
                bot_client=bot_client,
                user_client=user_client,
                honcho_url=honcho_url,
                workspace=workspace,
                session=session,
                before_answer=before_answer,
                config_dir=run_config,
                user_scenario=user_scenario,
            )
            event_id = trajectory["event_id"]
            if event_id != trajectory["receipt"].get("event_id"):
                raise AssertionError("runtime, Camera commit, and trajectory event IDs differ")
            producer_replay = await asyncio.to_thread(pipeline.run_once)
            if producer_replay.duplicate_suppression != 1:
                raise AssertionError("post-finalization producer replay was not suppressed")
            if (
                not trajectory["owner_replay_unbound"]
                or trajectory["owner_replay_event_id"] is not None
            ):
                raise AssertionError("completed-photo owner replay was not consumed as unbound")
            await verify_finalizer_event(
                url=honcho_url,
                workspace=workspace,
                session=session,
                candidate_id=candidate_id,
                answer_message_id=str(trajectory["answer"].metadata["message_id"]),
                native_photo_id=trajectory["native_photo_id"],
                expected_event_id=event_id,
                started=trajectory["started"],
                expected_capture_time=trajectory["capture_time"],
            )
            if ingress._attempts[candidate_id].get("camera_commit") != trajectory["receipt"]:
                raise AssertionError("owner replay changed the immutable Camera commit receipt")
            if sum(name == "send_photo" for name, _ in fake_bot.calls) != 1:
                raise AssertionError("producer or owner replay emitted an extra photo")
            label = "OFFLINE SYNTHETIC" if mode == "offline" else "NATIVE OPT-IN"
            print(
                f"PASS {label} full functional Camera trajectory; candidate={candidate_id} "
                f"event={event_id} capture_date={trajectory['capture_time'].date().isoformat()} "
                f"kcal={125 if mode == 'offline' else 'native'} producer+owner replay stable "
                f"producer_image_sha256={hashlib.sha256(source_bytes).hexdigest()} "
                f"photo_id={trajectory['native_photo_id']} bot_api_calls="
                f"{getattr(bot_client, 'calls', 'native')} user_api_calls="
                f"{getattr(user_client, 'calls', 'native')}",
                flush=True,
            )
        finally:
            if server is not None:
                server.close()
                await server.wait_closed()
            if ingress is not None:
                await ingress.close()
            if client is not None:
                client.close()
            if http_client is not None:
                http_client.close()
            for test_store in fixtures._TEST_STORES.values():
                test_store.close()
            fixtures._TEST_STORES.clear()
            if (
                verify_source_worktree(ROOT, expected_openharness, require_clean=acceptance)[0]
                != openharness_state[0]
            ):
                raise AssertionError("OpenHarness source pair changed during Camera run")
            if (
                verify_source_worktree(TELEGENT, expected_telegent, require_clean=acceptance)[0]
                != telegent_state[0]
            ):
                raise AssertionError("Telegent source pair changed during Camera run")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        if (
            os.environ.get("CAMERA_RUN_POSITIVE") == "1"
            or os.environ.get("CAMERA_RUN_MODE") == "native"
        ):
            print(f"FAIL Camera trajectory: {type(exc).__name__}: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        raise
