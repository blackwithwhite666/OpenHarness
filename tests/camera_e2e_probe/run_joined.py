"""One synthetic Telegent -> HTTP -> Ohmo Camera admission probe.

Run with OpenHarness's locked test environment; Telegent imports resolve from
CAMERA_TELEGENT_WORKTREE. Dropbox and Telegram use fixtures. The classifier
and its route attestation are synthetic fixture data, so this is a partial E2E.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
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
    e5_raw_honcho_row_fingerprint,
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
    signed_wire: bool = False,
    runtime_principal: str = "123",
    verify_projection: bool = True,
) -> None:
    """Read the real finalizer's event, then verify Telegent's exact projection."""
    from mcp.server.fastmcp.server import FastMCP
    from mcp.types import ToolAnnotations
    from telegent.health_advisor.nutrition.config import NutritionHonchoCredentials
    from telegent.health_advisor.nutrition.store import NutritionDataStore
    from telegent.health_advisor.nutrition.sync import sync_nutrition_honcho
    from telegent.health_advisor.storage import HealthDataStore
    from telegent.mcp_simple_auth.wellness import register_wellness_tools
    from probe_support import (
        call_wellness_with_synthetic_self,
        synthetic_wellness_self_scope,
    )

    async with HonchoClient(url, "local-auth-disabled", workspace) as client:
        messages = await client.list_recent_message_metadata(
            session,
            expected_peer_id="ohmo",
            since=started - timedelta(minutes=1),
            until=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
    event = select_finalizer_event(
        messages, candidate_id, answer_message_id, native_photo_id,
        expected_capture_time=expected_capture_time,
        expected_event_id=expected_event_id,
    )
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
    occurrence_source = event.metadata.get("photo_occurrence_source")
    expected_bindings = {
        "source_principal": "telegram:123",
        "tenant_id": "synthetic_owner",
        "source_message_id": answer_message_id,
        "ingest_source": "dropbox_camera",
        "source_image_attachment_count": 0,
    }
    binding_mismatches = [
        key for key, expected in expected_bindings.items() if event.metadata.get(key) != expected
    ]
    if not isinstance(occurrence_source, dict) or any(
        occurrence_source.get(key) != value
        for key, value in {
            "source_origin": "dropbox_camera",
            "origin_principal": "telegram:__camera__",
            "camera_candidate_id": candidate_id,
            "native_photo_message_id": str(native_photo_id),
        }.items()
    ):
        binding_mismatches.append("photo_occurrence_source")
    if any(key in event.metadata for key in ("camera_answer_bound", "camera_route", "camera_operation_id")):
        binding_mismatches.append("camera_answer_authority")
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
    if not verify_projection:
        return

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
        participant_registry, authorization_context, authorized_read = (
            synthetic_wellness_self_scope("synthetic_owner")
        )
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
                participant_registry=participant_registry,
                authorization_context=authorization_context,
            )

            async def balance():
                arguments = {
                    "params": {
                        "start": (expected_capture_time - timedelta(days=1)).isoformat(),
                        "end": (expected_capture_time + timedelta(days=1)).isoformat(),
                    }
                }
                result = await call_wellness_with_synthetic_self(
                    app, authorization_context, authorized_read, arguments
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
            expected_meal_id = record.meal_id
            health.close()
            db.close()
            db = NutritionDataStore(directory / "nutrition.db")
            health = HealthDataStore(directory / "health.db")
            reopened_record = db.get_record_by_event_id("synthetic_owner", event.id)
            reopened_meal = db.get_current_meal("synthetic_owner", expected_meal_id)
            reopened_meals, reopened_kcal = await balance()
            if not (
                reopened_record is not None
                and reopened_meal is not None
                and reopened_meal.model_dump(mode="json") == snapshot
                and reopened_meal.event_ids == [event.id]
                and reopened_meal.latest_event_id == event.id
                and reopened_meal.local_day(ZoneInfo("UTC"))
                == expected_capture_time.date().isoformat()
                and len(reopened_meals) == 1
                and reopened_meals[0]["latest_event_id"] == event.id
                and reopened_kcal == validated.energy_kcal_best
            ):
                raise AssertionError(
                    "reopened RocksDB projection differs from the runtime event/current meal"
                )
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
            if signed_wire:
                health.close()
                db.close()
                await verify_signed_wellness_wire(
                    database_directory=directory,
                    event_id=event.id,
                    started=started,
                    capture_time=expected_capture_time,
                    runtime_principal=runtime_principal,
                    expected_kcal=validated.energy_kcal_best,
                )
                db = NutritionDataStore(directory / "nutrition.db")
                health = HealthDataStore(directory / "health.db")
        finally:
            health.close()
            db.close()


async def verify_e5_projection_stage(
    *, url: str, workspace: str, session: str, started: datetime,
    original_event_id: str, latest_event_id: str, source_day: date,
    corrected_day: date, expected_day: date | None, expected_kcal: float,
    expected_status: str, expected_meal_at: datetime | None,
    expected_items: list[tuple[str, str]] | None = None,
) -> dict[str, object]:
    """Sync actual Honcho events into fresh/reopened RocksDB and query wellness windows."""
    from mcp.server.fastmcp.server import FastMCP
    from mcp.types import ToolAnnotations
    from telegent.health_advisor.nutrition.config import NutritionHonchoCredentials
    from telegent.health_advisor.nutrition.store import NutritionDataStore
    from telegent.health_advisor.nutrition.sync import sync_nutrition_honcho
    from telegent.health_advisor.storage import HealthDataStore
    from telegent.mcp_simple_auth.wellness import register_wellness_tools
    from probe_support import call_wellness_with_synthetic_self, synthetic_wellness_self_scope

    credentials = NutritionHonchoCredentials.model_validate({
        "schema_version": 1, "base_url": url,
        "sources": [{
            "user_id": "synthetic_owner", "workspace": workspace, "session": session,
            "workspace_jwt": "local-auth-disabled",
            "start_at": (started - timedelta(minutes=1)).isoformat(),
            "device_ids": [], "timezone": "UTC",
        }],
    })
    directory = create_storage_run_dir(ROOT)
    print(f"E5 {latest_event_id} projection retained at {directory}", flush=True)
    db = NutritionDataStore(directory / "nutrition.db")
    health = HealthDataStore(directory / "health.db")
    try:
        await sync_nutrition_honcho(credentials=credentials, db=db)
        original = db.get_record_by_event_id("synthetic_owner", original_event_id)
        latest = db.get_record_by_event_id("synthetic_owner", latest_event_id)
        if original is None or latest is None:
            raise AssertionError("E5 projection omitted a runtime receipt event")
        current = db.get_current_meal("synthetic_owner", original.meal_id)
        if (
            current is None or current.latest_event_id != latest_event_id
            or current.energy_kcal_best != expected_kcal
            or current.consumption_status != expected_status
        ):
            raise AssertionError("E5 current RocksDB meal does not match the latest runtime correction")
        current_day = current.local_day(ZoneInfo("UTC"))
        if current_day != (expected_day.isoformat() if expected_day else None):
            raise AssertionError("E5 current RocksDB meal date differs from expected correction")
        if expected_meal_at is not None and current.meal_at != expected_meal_at:
            raise AssertionError("E5 current RocksDB meal_at differs from explicit owner correction")
        current_items = [
            (
                str(item.get("name", "")).casefold(),
                str(item.get("quantity_text", "")),
            )
            if isinstance(item, dict)
            else (str(item.name).casefold(), str(item.quantity_text))
            for item in (current.items or [])
        ]
        if expected_items is not None and current_items != [
            (name.casefold(), quantity) for name, quantity in expected_items
        ]:
            raise AssertionError("E5 current RocksDB meal items or quantities differ")
        snapshot = current.model_dump(mode="json")
        meal_id = current.meal_id
        health.close()
        db.close()
        db = NutritionDataStore(directory / "nutrition.db")
        health = HealthDataStore(directory / "health.db")
        reopened = db.get_current_meal("synthetic_owner", meal_id)
        if reopened is None or reopened.model_dump(mode="json") != snapshot:
            raise AssertionError("E5 reopened RocksDB current meal changed")

        registry, auth_context, authorized = synthetic_wellness_self_scope("synthetic_owner")
        app = FastMCP(name="camera-e5-projection")

        async def health_store():
            return health

        async def nutrition_store():
            return db

        register_wellness_tools(
            app, read_only_annotations=ToolAnnotations(readOnlyHint=True),
            get_health_store=health_store, get_nutrition_store=nutrition_store,
            participant_registry=registry, authorization_context=auth_context,
        )

        async def day_records(day: date) -> list[dict[str, object]]:
            args = {"params": {
                "start": datetime.combine(day, datetime.min.time(), timezone.utc).isoformat(),
                "end": datetime.combine(day + timedelta(days=1), datetime.min.time(), timezone.utc).isoformat(),
            }}
            result = await call_wellness_with_synthetic_self(app, auth_context, authorized, args)
            return result[1]["nutrition_records"]

        source_records = await day_records(source_day)
        corrected_records = await day_records(corrected_day)
        source_intake = sum(row["energy_kcal_best"] or 0 for row in source_records)
        corrected_intake = sum(row["energy_kcal_best"] or 0 for row in corrected_records)
        expected_source = expected_kcal if expected_day == source_day else 0
        expected_corrected = expected_kcal if expected_day == corrected_day else 0
        if (source_intake, corrected_intake) != (expected_source, expected_corrected):
            raise AssertionError(
                f"E5 exact-window intake mismatch source={source_intake} "
                f"corrected={corrected_intake} expected={expected_source}/{expected_corrected}"
            )
        await sync_nutrition_honcho(credentials=credentials, db=db)
        replayed = db.get_current_meal("synthetic_owner", meal_id)
        if replayed is None or replayed.model_dump(mode="json") != snapshot:
            raise AssertionError("E5 sync replay changed current meal or immutable event selection")
        if expected_day is None:
            raise AssertionError("E5 current meal has no expected day for wellness projection")
        current_records = (
            source_records if expected_day == source_day
            else corrected_records if expected_day == corrected_day
            else await day_records(expected_day)
        )
        public_meals = [
            row for row in current_records
            if row.get("latest_event_id") == latest_event_id
        ]
        if expected_status == "not_consumed":
            if public_meals:
                raise AssertionError("E5 denial event unexpectedly appears in public wellness output")
        elif len(public_meals) != 1:
            raise AssertionError("Telegent wellness output omitted the exact current correction event")
        return {
            "latest_event_id": latest_event_id, "meal_id": meal_id,
            "meal_day": current_day, "meal_at": current.meal_at.isoformat() if current.meal_at else None,
            "kcal": current.energy_kcal_best, "status": current.consumption_status,
            "items": current_items,
            "canonical_meal": public_meals[0] if public_meals else None,
            "source_day_intake": source_intake, "corrected_day_intake": corrected_intake,
            "projection": str(directory), "reopened": True, "sync_replay_stable": True,
        }
    finally:
        health.close()
        db.close()


async def verify_signed_wellness_wire(
    *, database_directory: Path, event_id: str,
    started: datetime, capture_time: datetime, runtime_principal: str,
    expected_kcal: float,
) -> None:
    """Read this runtime event through Telegent's signed SDK HTTP guard."""
    import time

    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.auth.provider import AccessToken
    from pydantic import AnyHttpUrl

    from openharness.mcp.wellness_delegation import (
        TrustedWellnessActor,
        WellnessDelegationConfig as OhWellnessDelegationConfig,
        sign_wellness_call,
    )

    from telegent.health_advisor.nutrition.store import NutritionDataStore
    from telegent.health_advisor.storage import HealthDataStore
    from telegent.mcp_simple_auth.auth_server import (
        AuthServerSettings,
        SimpleAuthSettings,
        create_authorization_server,
    )
    from telegent.mcp_simple_auth.server import (
        ResourceServerSettings,
        ToolGroupSelection,
        create_resource_server,
    )
    from telegent.mcp_simple_auth.wellness_identity import WellnessParticipantRegistry
    import telegent.mcp_simple_auth.server as server_module

    resource_url = "https://camera-storage-probe.invalid/mcp"
    signing_config = OhWellnessDelegationConfig(
        key=b"synthetic-camera-wire-key-0123456789",
        kid="camera-wire-test",
        issuer="camera-offline-test",
        audience=resource_url,
        client_id="camera-ohmo-test-client",
    )
    if not runtime_principal.isdecimal():
        raise AssertionError("runtime admitted principal is not a numeric Telegram actor")
    import os

    delegation_env = {
        "WELLNESS_DELEGATION_SIGNING_KEY": signing_config.key.decode(),
        "WELLNESS_DELEGATION_KID": signing_config.kid,
        "WELLNESS_DELEGATION_ISSUER": signing_config.issuer,
        "WELLNESS_DELEGATION_AUDIENCE": signing_config.audience,
        "WELLNESS_DELEGATION_CLIENT_ID": signing_config.client_id,
    }
    prior_delegation_env = {key: os.environ.get(key) for key in delegation_env}
    os.environ.update(delegation_env)
    auth_settings = SimpleAuthSettings(
        auth_db_path=str(database_directory / "wire-oauth.db")
    )
    auth_server_settings = AuthServerSettings(
        server_url=AnyHttpUrl("http://localhost:9000")
    )
    auth_app = create_authorization_server(auth_server_settings, auth_settings)
    owner_bearer = "mcp_synthetic_camera_owner"
    reader_bearer = "mcp_synthetic_camera_reader"
    wrong_client_bearer = "mcp_synthetic_camera_wrong_client"
    for bearer, client_id in (
        (owner_bearer, signing_config.client_id),
        (reader_bearer, signing_config.client_id),
        (wrong_client_bearer, "camera-other-client"),
    ):
        auth_app.state.oauth_provider.store.save_access_token(
            AccessToken(
                token=bearer,
                client_id=client_id,
                scopes=[auth_settings.mcp_scope],
                expires_at=int(time.time()) + 3600,
                resource=resource_url,
            ),
            {"user": "synthetic-camera-test"},
        )

    registry = WellnessParticipantRegistry.from_payload(
        {
            "schema_version": 2,
            "default_participant_id": int(runtime_principal),
            "participants": {
                runtime_principal: {"user_id": "synthetic_owner", "login": "owner"},
                f"{int(runtime_principal) + 1}": {
                    "user_id": "synthetic_owner", "login": "owner_alt"
                },
                "202": {"user_id": "synthetic_reader", "login": "reader"},
                "303": {"user_id": "synthetic_ungranted", "login": "ungranted"},
            },
            "nutrition_read_grants": [
                {
                    "owner_participant_id": int(runtime_principal),
                    "reader_participant_id": 202,
                    "scope": "nutrition.read",
                    "consent_ref": "synthetic-camera-test-consent",
                }
            ],
        }
    )
    resource_settings = ResourceServerSettings(
        server_url=AnyHttpUrl(resource_url),
        auth_server_url=auth_server_settings.server_url,
        auth_server_introspection_endpoint="http://localhost:9000/introspect",
        mcp_scope=auth_settings.mcp_scope,
        json_response=True,
        stateless_http=False,
        event_store_db_path=str(database_directory / "wire-events.db"),
    )
    server = create_resource_server(
        resource_settings,
        health_db_path=str(database_directory / "health.db"),
        nutrition_db_path=str(database_directory / "nutrition.db"),
        wellness_participant_registry=registry,
        tool_groups=ToolGroupSelection(
            time=False, messenger=False, twitter=False, calendar=False,
            wellness=True, knowledge=False,
        ),
    )
    resource_app = server.streamable_http_app()
    accesses: list[str] = []
    created_nutrition: list[NutritionDataStore] = []
    created_health: list[HealthDataStore] = []
    original_nutrition_ctor = server_module.NutritionDataStore
    original_health_ctor = server_module.HealthDataStore
    original_get_binding = NutritionDataStore.get_binding

    def tracked_nutrition(*args, **kwargs):
        store = original_nutrition_ctor(*args, **kwargs)
        created_nutrition.append(store)
        return store

    def tracked_health(*args, **kwargs):
        store = original_health_ctor(*args, **kwargs)
        created_health.append(store)
        return store

    def tracked_binding(store, user_id):
        accesses.append(user_id)
        return original_get_binding(store, user_id)

    server_module.NutritionDataStore = tracked_nutrition
    server_module.HealthDataStore = tracked_health
    NutritionDataStore.get_binding = tracked_binding

    class AuthASGIClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("transport", httpx.ASGITransport(app=auth_app))
            kwargs.setdefault("base_url", "http://localhost:9000")
            super().__init__(*args, **kwargs)

    original_async_client = httpx.AsyncClient
    resource_client = original_async_client(
        transport=httpx.ASGITransport(app=resource_app),
        base_url="https://camera-storage-probe.invalid",
        headers={"Authorization": f"Bearer {owner_bearer}"},
    )
    httpx.AsyncClient = AuthASGIClient
    request = {
        "params": {
            "start": (capture_time - timedelta(days=1)).isoformat(),
            "end": (capture_time + timedelta(days=1)).isoformat(),
        }
    }
    def storage_counts():
        return (len(created_nutrition), len(created_health), len(accesses))

    def require_denied_without_store(result, label, before):
        if not result.isError or storage_counts() != before:
            raise AssertionError(
                f"{label} was not denied before store construction/access: "
                f"before={before}, after={storage_counts()}"
            )

    try:
        async with server.session_manager.run():
            async with resource_client:
                async with streamable_http_client(
                    resource_url, http_client=resource_client
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        unsigned = await session.call_tool("get_wellness_data", request)
                        require_denied_without_store(unsigned, "unsigned wellness request", (0, 0, 0))

                        trusted_actor = TrustedWellnessActor(runtime_principal)
                        owner = await session.call_tool(
                            "get_wellness_data", request,
                            meta=sign_wellness_call(signing_config, trusted_actor, request),
                        )
                        if owner.isError or len(owner.structuredContent["nutrition_records"]) != 1:
                            raise AssertionError("trusted owner could not read the projected Camera meal")
                        if owner.structuredContent["nutrition_records"][0]["latest_event_id"] != event_id:
                            raise AssertionError("signed owner read returned a different Camera event")

                        alias_123 = {"params": {**request["params"], "login": "owner"}}
                        alias_124 = {
                            "params": {**request["params"], "login": "owner_alt"}
                        }
                        own_reads = (
                            (runtime_principal, {"params": dict(request["params"])}),
                            (runtime_principal, alias_123),
                            (str(int(runtime_principal) + 1), {"params": dict(request["params"])}),
                            (str(int(runtime_principal) + 1), alias_124),
                        )
                        for alias_actor, alias_request in own_reads:
                            own_alias = await session.call_tool(
                                "get_wellness_data", alias_request,
                                meta=sign_wellness_call(
                                    signing_config, TrustedWellnessActor(alias_actor), alias_request
                                ),
                            )
                            if (
                                own_alias.isError
                                or len(own_alias.structuredContent["nutrition_records"]) != 1
                                or own_alias.structuredContent["nutrition_records"][0]["latest_event_id"] != event_id
                                or own_alias.structuredContent["nutrition_records"][0]["energy_kcal_best"] != expected_kcal
                                or own_alias.structuredContent["nutrition_records"][0]["day"] != capture_time.date().isoformat()
                            ):
                                raise AssertionError(
                                    f"runtime owner account {alias_actor} could not read the same projected meal"
                                )

                        foreign_request = {"params": {**request["params"], "login": "reader"}}
                        before = storage_counts()
                        foreign = await session.call_tool(
                            "get_wellness_data", foreign_request,
                            meta=sign_wellness_call(signing_config, trusted_actor, foreign_request),
                        )
                        require_denied_without_store(foreign, "explicit foreign login", before)

                        missing_request = {
                            "params": {**request["params"], "login": "missing-owner-alias"}
                        }
                        before = storage_counts()
                        missing = await session.call_tool(
                            "get_wellness_data", missing_request,
                            meta=sign_wellness_call(signing_config, trusted_actor, missing_request),
                        )
                        require_denied_without_store(missing, "missing target alias", before)

                        reader_request = {
                            "params": {**request["params"], "login": "owner"}
                        }
                        reader_token = sign_wellness_call(
                            signing_config, TrustedWellnessActor("202"), reader_request
                        )
                        reader = await session.call_tool(
                            "get_wellness_data", reader_request, meta=reader_token
                        )
                        if reader.isError:
                            raise AssertionError("directed nutrition-only reader was denied")
                        payload = reader.structuredContent
                        if (
                            len(payload["nutrition_records"]) != 1
                            or payload["nutrition_records"][0]["latest_event_id"] != event_id
                            or payload["nutrition_records"][0]["energy_kcal_best"] != expected_kcal
                            or payload["nutrition_records"][0]["day"] != capture_time.date().isoformat()
                            or payload["health_authorized"] is not False
                            or payload["energy_intervals"]
                            or payload["raw_samples"]
                        ):
                            raise AssertionError("reader response did not expose only the granted meal")

                        before = storage_counts()
                        ungranted_request = {
                            "params": {**reader_request["params"]}
                        }
                        ungranted = await session.call_tool(
                            "get_wellness_data",
                            ungranted_request,
                            meta=sign_wellness_call(
                                signing_config, TrustedWellnessActor("303"), ungranted_request
                            ),
                        )
                        require_denied_without_store(ungranted, "ungranted target read", before)

                        changed = {"params": {**reader_request["params"], "end": started.isoformat()}}
                        before = storage_counts()
                        tampered = await session.call_tool(
                            "get_wellness_data", changed, meta=reader_token
                        )
                        require_denied_without_store(tampered, "tampered signed request body", before)

            async with original_async_client(
                transport=httpx.ASGITransport(app=resource_app),
                base_url="https://camera-storage-probe.invalid",
                headers={"Authorization": f"Bearer {wrong_client_bearer}"},
            ) as wrong_client:
                async with streamable_http_client(
                    resource_url, http_client=wrong_client
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        before = storage_counts()
                        wrong_client_token = sign_wellness_call(
                            signing_config, trusted_actor, request
                        )
                        denied = await session.call_tool(
                            "get_wellness_data", request, meta=wrong_client_token
                        )
                        require_denied_without_store(denied, "wrong OAuth client", before)
        if not created_nutrition or not accesses or set(accesses) != {"synthetic_owner"}:
            raise AssertionError("valid signed reads did not use the projected owner binding")
        print(
            f"PASS signed SDK HTTP guard runtime_event={event_id} "
            f"runtime_owner={runtime_principal} reader=202 "
            "ungranted/tamper/wrong-client denied before read; reader health absent",
            flush=True,
        )
    finally:
        for key, value in prior_delegation_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        httpx.AsyncClient = original_async_client
        server_module.NutritionDataStore = original_nutrition_ctor
        server_module.HealthDataStore = original_health_ctor
        NutritionDataStore.get_binding = original_get_binding
        await resource_client.aclose()
        for store in (*created_nutrition, *created_health):
            store.close()


async def verify_failed_append_absence(
    *, url: str, workspace: str, session: str, started: datetime,
) -> None:
    """Verify a pre-transport failure leaves no Camera event or current meal."""
    from mcp.server.fastmcp.server import FastMCP
    from mcp.types import ToolAnnotations

    from ohmo.memory_service.honcho_client import HonchoClient
    from telegent.health_advisor.nutrition.config import NutritionHonchoCredentials
    from telegent.health_advisor.nutrition.store import NutritionDataStore
    from telegent.health_advisor.nutrition.sync import sync_nutrition_honcho
    from telegent.health_advisor.storage import HealthDataStore
    from telegent.mcp_simple_auth.wellness import register_wellness_tools
    from probe_support import (
        call_wellness_with_synthetic_self,
        synthetic_wellness_self_scope,
    )

    async with HonchoClient(url, "local-auth-disabled", workspace) as client:
        rows = await client.list_recent_message_metadata(
            session,
            expected_peer_id="ohmo",
            since=started - timedelta(minutes=1),
            until=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
    meal_events = []
    for row in rows:
        trace = row.metadata.get("decision_trace")
        annotation = (
            trace.get("annotations", {}).get("nutrition")
            if isinstance(trace, dict) else None
        )
        if isinstance(annotation, dict) and annotation.get("record_type") in {
            "meal_observation", "meal_correction"
        }:
            meal_events.append(row.id)
    if meal_events:
        raise AssertionError(f"failed append left Honcho nutrition events: {meal_events}")

    root = create_storage_run_dir(ROOT)
    database = NutritionDataStore(root / "nutrition.db")
    health = HealthDataStore(root / "health.db")
    owner = "synthetic_owner"
    credentials = NutritionHonchoCredentials.model_validate(
        {
            "schema_version": 1,
            "base_url": url,
            "sources": [
                {
                    "user_id": owner,
                    "workspace": workspace,
                    "session": session,
                    "workspace_jwt": "local-auth-disabled",
                    "start_at": (started - timedelta(days=1)).isoformat(),
                    "device_ids": [],
                    "timezone": "UTC",
                }
            ],
        }
    )
    participant_registry, authorization_context, authorized_read = (
        synthetic_wellness_self_scope(owner)
    )
    app = FastMCP(name="camera-failed-append-probe")

    async def get_health_store():
        return health

    async def get_nutrition_store():
        return database

    register_wellness_tools(
        app,
        read_only_annotations=ToolAnnotations(readOnlyHint=True),
        get_health_store=get_health_store,
        get_nutrition_store=get_nutrition_store,
        participant_registry=participant_registry,
        authorization_context=authorization_context,
    )
    try:
        await sync_nutrition_honcho(credentials=credentials, db=database)
        event_rows = list(database.iter_records(owner))
        if event_rows:
            raise AssertionError(f"failed append created projection records: {event_rows}")
        arguments = {
            "params": {
                "start": (started - timedelta(days=1)).isoformat(),
                "end": (started + timedelta(days=1)).isoformat(),
            }
        }
        result = await call_wellness_with_synthetic_self(
            app, authorization_context, authorized_read, arguments
        )
        if result[1]["nutrition_records"]:
            raise AssertionError("failed append appeared in the current Wellness meal read")
        print("PASS injected pre-write failure: full Honcho read and current projection contain no meal")
    finally:
        health.close()
        database.close()


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
    from probe_support import (
        call_wellness_with_synthetic_self,
        synthetic_wellness_self_scope,
    )

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
    participant_registry, authorization_context, authorized_read = (
        synthetic_wellness_self_scope("synthetic_owner")
    )
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
            participant_registry=participant_registry,
            authorization_context=authorization_context,
        )
        await sync_nutrition_honcho(credentials=credentials, db=db)
        arguments = {
            "params": {
                "start": (capture_time - timedelta(days=1)).isoformat(),
                "end": (capture_time + timedelta(days=1)).isoformat(),
            }
        }
        result = await call_wellness_with_synthetic_self(
            app, authorization_context, authorized_read, arguments
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


async def verify_actual_camera_grade(
    *, root: Path, url: str, workspace: str, session: str,
    started: datetime, answer, capture_time: datetime,
    expected_consumed: bool, expected_kcal: float | None,
    expected_latest_event_id: str | None,
    meal_day: date,
) -> dict[str, object]:
    """Grade the recorded Camera owner turn against fresh complete service reads."""
    import sqlite3

    from mcp.server.fastmcp.server import FastMCP
    from mcp.types import ToolAnnotations
    from ohmo.evals.nutrition_persistence import (
        Goal, Manifest, bind_wellness_snapshot, derive_meal_id,
        export_eval_dialogue, grade_manifest, read_honcho_messages,
        validate_dialogue_binding,
    )
    from telegent.health_advisor.nutrition.config import NutritionHonchoCredentials
    from telegent.health_advisor.nutrition.store import NutritionDataStore
    from telegent.health_advisor.nutrition.sync import sync_nutrition_honcho
    from telegent.health_advisor.storage import HealthDataStore
    from telegent.mcp_simple_auth.wellness import register_wellness_tools
    from probe_support import call_wellness_with_synthetic_self, synthetic_wellness_self_scope

    eval_root = root / "evals"
    with sqlite3.connect(f"{(eval_root / 'evals.sqlite').as_uri()}?mode=ro", uri=True) as connection:
        episode_ids = [row[0] for row in connection.execute(
            "SELECT episode_id FROM episodes ORDER BY created_at, episode_id"
        ).fetchall()]
    exported = export_eval_dialogue(eval_root, episode_ids=episode_ids)
    episode_ids = [item["episode"]["episode_id"] for item in exported["episodes"]
                   if item.get("trusted_camera_context", {}).get("kind") == "initial_context"
                   or any("gateway_final_metadata" in turn
                          for turn in item["turn_provenance"])]
    exported = export_eval_dialogue(eval_root, episode_ids=episode_ids)
    source_id = str(answer.metadata["message_id"])
    owner_episodes = [item for item in exported["episodes"]
                      if source_id in item["source_message_ids"]
                      and any(turn.get("source_message_id") == source_id
                              for turn in item["turn_provenance"])]
    if not owner_episodes:
        raise AssertionError("recorder export lacks the ordinary owner turn")
    owner_episodes.sort(key=lambda item: item["episode"]["created_at"])
    root_episode = owner_episodes[0]
    root_turn = next(turn for turn in root_episode["turn_provenance"]
                     if turn.get("source_message_id") == source_id)
    gateway_session = root_episode["episode"]["session_id"]
    as_of = datetime.now(timezone.utc)
    goal = Goal(
        case_id="actual-camera-owner", episode_ids=episode_ids,
        owner_id="synthetic_owner", principal_id="telegram:123",
        workspace_id=workspace, eval_workspace=str(root), peer_id="ohmo",
        canonical_owner_id="synthetic_owner", canonical_login="synthetic_owner",
        session_id=session, gateway_session_id=gateway_session,
        source_message_id=source_id, meal_date=meal_day, meal_timezone="UTC",
        trajectory_started_at=started, trajectory_as_of=as_of,
        logical_turn_id=root_turn["logical_turn_id"],
        trace_episode_id=root_episode["episode"]["episode_id"],
        operation_id=root_turn["operation_id"],
        canonical_meal_id=derive_meal_id(
            tenant_id="synthetic_owner", source_principal="telegram:123",
            gateway_session_id=gateway_session, source_message_id=source_id,
        ),
        expected_consumed=expected_consumed, expected_kcal=expected_kcal,
        expectation_origin="explicit_fixture",
        expectation_source="offline-scripted-owner-and-photo-fixture",
    )
    manifest = Manifest(schema_version=1, goals=[goal])
    binding = validate_dialogue_binding(manifest, exported)[goal.case_id]
    if not binding["complete"]:
        raise AssertionError(f"actual recorder source binding incomplete: {binding['reason']}")
    honcho = await read_honcho_messages(
        base_url=url, api_key="local-auth-disabled", workspace=workspace,
        session=session, owner_id=goal.owner_id,
        since=started - timedelta(minutes=1),
        until=datetime.now(timezone.utc), peer_id="ohmo",
    )
    if honcho.get("complete") is not True:
        raise AssertionError(f"actual Honcho read incomplete: {honcho}")
    directory = create_storage_run_dir(ROOT)
    db = NutritionDataStore(directory / "nutrition.db")
    health = HealthDataStore(directory / "health.db")
    credentials = NutritionHonchoCredentials.model_validate({
        "schema_version": 1, "base_url": url,
        "sources": [{"user_id": "synthetic_owner", "workspace": workspace,
                     "session": session, "workspace_jwt": "local-auth-disabled",
                     "start_at": (started - timedelta(days=2)).isoformat(),
                     "device_ids": [], "timezone": "UTC"}],
    })
    registry, auth_context, authorized = synthetic_wellness_self_scope("synthetic_owner")
    app = FastMCP(name="camera-actual-grade-probe")

    async def get_health_store():
        return health

    async def get_nutrition_store():
        return db

    register_wellness_tools(
        app, read_only_annotations=ToolAnnotations(readOnlyHint=True),
        get_health_store=get_health_store, get_nutrition_store=get_nutrition_store,
        participant_registry=registry, authorization_context=auth_context,
    )
    try:
        await sync_nutrition_honcho(credentials=credentials, db=db)
        read_start = datetime.combine(min(meal_day, capture_time.date()),
                                      datetime.min.time(), timezone.utc)
        read_end = datetime.now(timezone.utc)
        result = await call_wellness_with_synthetic_self(
            app, auth_context, authorized,
            {"params": {"start": read_start.isoformat(), "end": read_end.isoformat()}},
        )
        body = result[1]
        if body.get("nutrition_status") != "complete":
            raise AssertionError(f"actual Telegent nutrition read incomplete: {body.get('nutrition_status')}")
        queried = datetime.now(timezone.utc)
        wellness = bind_wellness_snapshot({
            "complete": True, "user_id": goal.canonical_owner_id,
            "login": goal.canonical_login, "start": read_start.isoformat(),
            "end": read_end.isoformat(), "queried_at": queried.isoformat(),
            "meals": body["nutrition_records"],
            "unassigned": body["nutrition_unassigned_records"],
        }, goal=goal)
        grade = grade_manifest(
            manifest, honcho, wellness, now=queried,
            reviewed_turn_sources=binding["reviewed_turn_sources"],
            reviewed_turn_provenance=binding["reviewed_turn_provenance"],
        )[0]
        expected = (("FAIL", "HONCHO_MISSING_WRITE") if expected_consumed
                    and expected_latest_event_id is None else
                    ("PASS", "SAME_EVENT_PROJECTED") if expected_consumed
                    else ("PASS", "COMPLETE_ABSENCE"))
        if (grade["a1"], grade["stage"]) != expected:
            raise AssertionError(f"actual Camera grader mismatch: {grade}")
        if expected_latest_event_id is not None and grade.get("actual_latest_event_id") != expected_latest_event_id:
            raise AssertionError(f"grader selected wrong latest event: {grade}")
        print(f"PASS actual Camera grader a1={grade['a1']} stage={grade['stage']}", flush=True)
        return grade
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
    restart_join = os.environ.get("CAMERA_RESTART_JOIN", "")
    correction_join = os.environ.get("CAMERA_CORRECTION_JOIN", "")
    missing_write_review = os.environ.get("CAMERA_MISSING_WRITE_REVIEW") == "1"
    if restart_join not in {"", "two-photo"}:
        raise RuntimeError("CAMERA_RESTART_JOIN must be two-photo")
    if restart_join and mode != "offline":
        raise RuntimeError("Camera restart join is deterministic and offline-only")
    if correction_join not in {"", "portion-denial", "context-items-date"}:
        raise RuntimeError("CAMERA_CORRECTION_JOIN must be portion-denial or context-items-date")
    if correction_join and mode != "offline":
        raise RuntimeError("Camera correction join is deterministic and offline-only")
    if missing_write_review and (mode != "offline" or correction_join or restart_join):
        raise RuntimeError("missing-write review requires standalone offline mode")
    append_fault = os.environ.get("CAMERA_HONCHO_APPEND_FAULT", "")
    if append_fault not in {"", "before", "timeout-after"}:
        raise RuntimeError("CAMERA_HONCHO_APPEND_FAULT must be before or timeout-after")
    if append_fault and mode != "offline":
        raise RuntimeError("Camera append fault injection is deterministic and offline-only")
    if append_fault and missing_write_review:
        raise RuntimeError("missing-write review and append transport fault are separate fixtures")
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
        bot_client, user_client = distinct_offline_clients(
            context_items_date=correction_join == "context-items-date",
            omit_nutrition_annotation=missing_write_review,
        )
        user_scenario = "synthetic offline owner selects the exact offered confirmation"

    original_honcho_create_messages = None
    fault_injected = [False]

    def inject_append_transport_fault():
        nonlocal original_honcho_create_messages
        from ohmo.memory_service.honcho_client import HonchoClient

        original_honcho_create_messages = HonchoClient.create_messages

        async def create_messages_with_fault(honcho, target_session, messages):
            is_meal_event = any(
                isinstance(message, dict)
                and isinstance(message.get("metadata"), dict)
                and isinstance(message["metadata"].get("decision_trace"), dict)
                and isinstance(
                    message["metadata"]["decision_trace"].get("annotations"), dict
                )
                and isinstance(
                    message["metadata"]["decision_trace"]["annotations"].get("nutrition"),
                    dict,
                )
                and message["metadata"]["decision_trace"]["annotations"]["nutrition"].get(
                    "record_type"
                ) == "meal_observation"
                for message in messages
            )
            if is_meal_event and not fault_injected[0]:
                fault_injected[0] = True
                if append_fault == "before":
                    raise httpx.ConnectError("deterministic camera probe transport failure")
                await original_honcho_create_messages(honcho, target_session, messages)
                raise httpx.ReadTimeout("deterministic post-append response timeout")
            return await original_honcho_create_messages(honcho, target_session, messages)

        HonchoClient.create_messages = create_messages_with_fault

    if append_fault:
        inject_append_transport_fault()

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
            if mode == "offline":
                bot_client.candidate_id = candidate_id
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
            camera_source = fixtures._Source([entry], source_bytes)
            pipeline = fixtures._pipeline(
                camera_source,
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

            second_candidate_id = None

            async def admit_intervening_photo(process_and_deliver, _active_ingress, _active_pool):
                nonlocal second_candidate_id
                second_photo_bytes = fixtures._image_at(datetime.now(timezone.utc), "blue")
                second_entry = fixtures._entry(
                    "id:camera-joined-intervening", "rev:1"
                ).model_copy(
                    update={
                        "size": len(second_photo_bytes),
                        "server_modified": datetime.now(timezone.utc),
                    }
                )
                second_candidate_id = fixtures.candidate_id_for(
                    second_entry.file_id, second_entry.rev
                )
                camera_source.entries[:] = [second_entry]
                camera_source.revision_payloads[second_entry.rev] = second_photo_bytes
                admitted = await asyncio.to_thread(pipeline.run_once)
                assert admitted.published == 1, admitted
                assert admitted.results[0].submission_outcome == "accepted", admitted.results[0]
                second_inbound = await asyncio.wait_for(bus.consume_inbound(), timeout=5)
                assert second_inbound.metadata["_camera_candidate_id"] == second_candidate_id
                assert second_inbound.media
                assert len(fake_bot.photo_bodies) == 2
                assert fake_bot.photo_bodies[1] == second_photo_bytes
                assert fake_bot.photo_bodies[1] != fake_bot.photo_bodies[0]
                second_capture_time = _active_ingress._attempt_capture_time(
                    _active_ingress._attempts[second_candidate_id]
                )
                if second_capture_time is None or second_capture_time <= capture_time:
                    raise AssertionError("intervening Camera photo is not newer than the first source")
                await process_and_deliver(second_inbound, second_inbound.session_key)
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

            async def append_runtime_corrections(
                process_and_deliver, active_ingress, active_pool,
                active_candidate_id, answer, original_receipt, restart_runtime,
            ):
                from openharness.channels.bus.events import InboundMessage
                from ohmo.memory_service.honcho_client import HonchoClient as RuntimeHonchoClient

                original_id = original_receipt["event_id"]

                async def rows():
                    async with RuntimeHonchoClient(honcho_url, "local-auth-disabled", workspace) as honcho:
                        return await honcho.list_messages_in_window(
                            session, expected_peer_id="ohmo",
                            since=trajectory_started - timedelta(minutes=1),
                            until=datetime.now(timezone.utc) + timedelta(minutes=1),
                        )

                original_row = next(row for row in await rows() if row["id"] == original_id)
                original_fingerprint = e5_raw_honcho_row_fingerprint(original_row)
                original_operation = original_row["metadata"].get("client_op_id")
                if (
                    not isinstance(original_operation, str)
                    or original_operation != f"{original_row['metadata'].get('logical_turn_id')}:assistant"
                ):
                    raise AssertionError("original runtime append has no exact logical operation")
                original_nutrition = original_row["metadata"]["decision_trace"]["annotations"]["nutrition"]
                original_items = original_nutrition["items"]
                source_day = capture_time.date()
                corrected_at = (capture_time - timedelta(days=1)).replace(
                    hour=18, minute=45, second=0, microsecond=0,
                )
                latest_id = original_id
                correction_ids = []
                correction_operations = set()

                async def correction(text, message_id, annotation):
                    nonlocal latest_id
                    message = InboundMessage(
                        channel="telegram", sender_id="123", chat_id="123",
                        content=text, timestamp=datetime.now(timezone.utc),
                        metadata={"message_id": message_id, "is_group": False,
                                  "chat_type": "private", "_telegram_raw_text": text},
                    )
                    bot_client.queue_correction(annotation)
                    delivered, receipt, event_id = await process_and_deliver(
                        message, message.session_key or "telegram:123",
                    )
                    if not isinstance(event_id, str) or not receipt:
                        raise AssertionError(f"ordinary correction did not append: {text!r}; {delivered!r}")
                    current_rows = await rows()
                    matches = [row for row in current_rows if row["id"] == event_id]
                    if len(matches) != 1:
                        raise AssertionError("correction receipt has no unique durable Honcho row")
                    metadata = matches[0]["metadata"]
                    selected = metadata.get("selected_source")
                    if (
                        metadata.get("source_principal") != "telegram:123"
                        or metadata.get("tenant_id") != "synthetic_owner"
                        or metadata.get("source_message_id") != message_id
                        or not isinstance(metadata.get("client_op_id"), str)
                        or metadata["client_op_id"] != f"{metadata.get('logical_turn_id')}:assistant"
                        or metadata["client_op_id"] == original_operation
                        or metadata["client_op_id"] in correction_operations
                        or not isinstance(selected, dict)
                        or selected.get("original_receipt_event_id") != original_id
                        or selected.get("original_receipt_client_op_id") != original_operation
                        or selected.get("source_message_id") != answer.metadata["message_id"]
                    ):
                        raise AssertionError("correction lacks trusted original/owner/operation receipt")
                    still_original = next(row for row in current_rows if row["id"] == original_id)
                    if e5_raw_honcho_row_fingerprint(still_original) != original_fingerprint:
                        raise AssertionError("correction mutated the original Honcho row")
                    latest_id = event_id
                    correction_ids.append(event_id)
                    correction_operations.add(metadata["client_op_id"])
                    return event_id

                if correction_join == "context-items-date":
                    changed_items = [dict(item) for item in original_items]
                    changed_items[0]["quantity_text"] = "half a serving"
                    item_id = await correction(
                        "Only half the cottage cheese; keep the berries",
                        "offline-context-item", {
                            "schema_version": 2, "record_type": "meal_correction",
                            "changed_fields": ["items"], "items": changed_items,
                        },
                    )
                    await verify_e5_projection_stage(
                        url=honcho_url, workspace=workspace, session=session,
                        started=trajectory_started, original_event_id=original_id,
                        latest_event_id=item_id, source_day=source_day,
                        corrected_day=corrected_at.date(), expected_day=source_day,
                        expected_kcal=225, expected_status="consumed",
                        expected_meal_at=capture_time,
                        expected_items=[(item["name"], item["quantity_text"]) for item in changed_items],
                    )
                    active_ingress, active_pool = await restart_runtime()
                    date_id = await correction(
                        "That was yesterday at 18:45", "offline-context-date", {
                            "schema_version": 2, "record_type": "meal_correction",
                            "changed_fields": ["meal_at"], "meal_at": corrected_at.isoformat(),
                        },
                    )
                    await verify_e5_projection_stage(
                        url=honcho_url, workspace=workspace, session=session,
                        started=trajectory_started, original_event_id=original_id,
                        latest_event_id=date_id, source_day=source_day,
                        corrected_day=corrected_at.date(), expected_day=corrected_at.date(),
                        expected_kcal=225, expected_status="consumed",
                        expected_meal_at=corrected_at,
                        expected_items=[(item["name"], item["quantity_text"]) for item in changed_items],
                    )
                else:
                    denial_id = await correction(
                        "I did not eat it", "offline-context-denial", {
                            "schema_version": 2, "record_type": "meal_correction",
                            "changed_fields": ["consumption_status", "energy_kcal_best"],
                            "consumption_status": "not_consumed", "energy_kcal_best": 0,
                        },
                    )
                    await verify_e5_projection_stage(
                        url=honcho_url, workspace=workspace, session=session,
                        started=trajectory_started, original_event_id=original_id,
                        latest_event_id=denial_id, source_day=source_day,
                        corrected_day=source_day, expected_day=source_day,
                        expected_kcal=0, expected_status="not_consumed",
                        expected_meal_at=capture_time,
                    )
                return {"latest_event_id": latest_id, "correction_event_ids": correction_ids,
                        "original_row_fingerprint": original_fingerprint}

            trajectory_started = datetime.now(timezone.utc)

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
                before_replay=(
                    admit_intervening_photo if restart_join == "two-photo" else None
                ),
                advance_before_owner_action=bool(restart_join),
                restart_before_replay=bool(restart_join or correction_join) or append_fault == "timeout-after",
                expect_append_failure=append_fault == "before",
                expect_missing_nutrition=missing_write_review,
                after_save=(
                    append_runtime_corrections if correction_join else None
                ),
            )
            if append_fault and not fault_injected[0]:
                raise AssertionError("requested Honcho append transport fault was not injected")
            ingress = trajectory["ingress"]
            if restart_join:
                if trajectory["camera_clock_advanced"] != timedelta(minutes=31):
                    raise AssertionError("restart changed the controlled clock by the wrong amount")
                if restart_join == "two-photo":
                    if second_candidate_id is None or ingress._attempts[second_candidate_id].get("attention_active") is not True:
                        raise AssertionError("old source released the newer photo's attention")
            event_id = trajectory["event_id"]
            camera_source.entries[:] = [entry]
            producer_replay = await asyncio.to_thread(pipeline.run_once)
            if producer_replay.duplicate_suppression != 1:
                raise AssertionError("producer replay was not suppressed")
            if append_fault == "before" or missing_write_review:
                if event_id is not None or not trajectory.get("final_status_text"):
                    raise AssertionError("unsaved owner turn returned a receipt or no response")
                await verify_failed_append_absence(
                    url=honcho_url, workspace=workspace, session=session,
                    started=trajectory["started"],
                )
                if missing_write_review:
                    for reviewed_consumed in (True, False):
                        await verify_actual_camera_grade(
                            root=root, url=honcho_url, workspace=workspace,
                            session=session, started=trajectory["started"],
                            answer=trajectory["answer"], capture_time=trajectory["capture_time"],
                            expected_consumed=reviewed_consumed,
                            expected_kcal=125 if reviewed_consumed else None,
                            expected_latest_event_id=None,
                            meal_day=trajectory["capture_time"].date(),
                        )
            else:
                if not isinstance(event_id, str) or event_id != trajectory["receipt"]["event_id"]:
                    raise AssertionError("runtime returned no durable event receipt")
                await verify_finalizer_event(
                    url=honcho_url, workspace=workspace, session=session,
                    candidate_id=candidate_id,
                    answer_message_id=str(trajectory["answer"].metadata["message_id"]),
                    native_photo_id=trajectory["native_photo_id"],
                    expected_event_id=event_id,
                    started=trajectory["started"],
                    expected_capture_time=trajectory["capture_time"],
                    verify_projection=not bool(correction_join),
                    signed_wire=os.environ.get("CAMERA_SIGNED_WIRE") == "1",
                )
                async with HonchoClient(honcho_url, "local-auth-disabled", workspace) as honcho:
                    raw_rows = await honcho.list_messages_in_window(
                        session, expected_peer_id="ohmo",
                        since=trajectory["started"] - timedelta(minutes=1),
                        until=datetime.now(timezone.utc) + timedelta(minutes=1),
                    )
                nutrition_rows = [
                    row for row in raw_rows
                    if isinstance(row.get("metadata", {}).get("decision_trace"), dict)
                    and isinstance(
                        row["metadata"]["decision_trace"].get("annotations", {}).get("nutrition"),
                        dict,
                    )
                    and row["metadata"]["decision_trace"]["annotations"]["nutrition"].get(
                        "record_type"
                    ) in {"meal_observation", "meal_correction", "meal_deletion"}
                ]
                corrections = trajectory["after_save_result"] if correction_join else None
                expected_ids = [event_id] + (
                    corrections["correction_event_ids"] if corrections else []
                )
                if len(nutrition_rows) != len(expected_ids) or {
                    row["id"] for row in nutrition_rows
                } != set(expected_ids):
                    raise AssertionError("replay duplicated or removed a durable nutrition event")
                if corrections:
                    original_row = next(row for row in nutrition_rows if row["id"] == event_id)
                    if e5_raw_honcho_row_fingerprint(original_row) != corrections["original_row_fingerprint"]:
                        raise AssertionError("replay or correction mutated the original Honcho row")
                await verify_actual_camera_grade(
                    root=root, url=honcho_url, workspace=workspace,
                    session=session, started=trajectory["started"],
                    answer=trajectory["answer"], capture_time=trajectory["capture_time"],
                    expected_consumed=correction_join != "portion-denial",
                    expected_kcal=(225 if correction_join == "context-items-date" else 125)
                    if correction_join != "portion-denial" else None,
                    expected_latest_event_id=(corrections["latest_event_id"] if corrections else event_id)
                    if correction_join != "portion-denial" else None,
                    meal_day=(trajectory["capture_time"] - timedelta(days=1)).date()
                    if correction_join == "context-items-date" else trajectory["capture_time"].date(),
                )
            expected_photo_count = 2 if restart_join == "two-photo" else 1
            if sum(name == "send_photo" for name, _ in fake_bot.calls) != expected_photo_count:
                raise AssertionError("replay emitted another photo")
            print(
                f"PASS {'OFFLINE SYNTHETIC' if mode == 'offline' else 'NATIVE OPT-IN'} ordinary Camera trajectory candidate={candidate_id} "
                f"event={event_id} producer_replay_suppressed=True "
                f"owner_replay_appended={trajectory.get('owner_replay_event_id') is not None}",
                flush=True,
            )
        finally:
            if original_honcho_create_messages is not None:
                HonchoClient.create_messages = original_honcho_create_messages
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
