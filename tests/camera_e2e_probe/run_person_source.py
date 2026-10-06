"""Run one person-origin text/photo turn through Ohmo and exact persistence evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/camera_e2e_probe"))

from camera_runtime_support import OfflinePersonSourceApi  # noqa: E402
from person_source_input import person_source_message  # noqa: E402
from person_source_runtime import (  # noqa: E402
    honcho_history_snapshot,
    new_nutrition_event_ids,
    projected_intake_kcal,
    run_person_source_turn,
)
from probe_support import (  # noqa: E402
    create_storage_run_dir,
    native_person_source_clients,
    source_jpeg,
    unique_honcho_scope,
    verify_source_tree_pin,
)


def _aware_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("CAMERA_SOURCE_SENT_AT must include a timezone offset")
    return parsed


async def main() -> None:
    mode = os.environ.get("CAMERA_RUN_MODE", "offline")
    if mode not in {"offline", "native"}:
        raise ValueError("CAMERA_RUN_MODE must be offline or native")
    native = mode == "native"
    if native and os.environ.get("CAMERA_ACCEPTANCE") != "1":
        raise ValueError("native person-source mode requires CAMERA_ACCEPTANCE=1")
    telegent_path = Path(os.environ["CAMERA_TELEGENT_WORKTREE"]).resolve(strict=True)
    telegent_sha = os.environ["CAMERA_TELEGENT_SHA"]
    telegent_path, telegent_head, telegent_tree = verify_source_tree_pin(
        telegent_path, telegent_sha
    )
    telegent_state = (telegent_head, telegent_tree)
    source_pair = {
        "telegent": {
            "worktree": str(telegent_path),
            "head": telegent_head,
            "tree": telegent_tree,
        }
    }
    openharness_state = None
    if native:
        openharness_sha = os.environ["CAMERA_OPENHARNESS_SHA"]
        openharness_state = verify_source_tree_pin(ROOT, openharness_sha)
        source_pair["openharness"] = {
            "worktree": str(openharness_state[0]),
            "head": openharness_state[1],
            "tree": openharness_state[2],
        }

    sender_id = os.environ["CAMERA_SOURCE_SENDER_ID"]
    chat_id = os.environ["CAMERA_SOURCE_CHAT_ID"]
    source_id = os.environ["CAMERA_SOURCE_MESSAGE_ID"]
    sent_at = _aware_time(os.environ["CAMERA_SOURCE_SENT_AT"])
    scenario = os.environ["CAMERA_USER_SCENARIO"].strip()
    if not scenario:
        raise ValueError("CAMERA_USER_SCENARIO must be a public human scenario")
    owner_id = os.environ["CAMERA_OWNER_ID"]
    honcho_url = os.environ.get("CAMERA_HONCHO_URL", "http://api:8000")
    source_kind = os.environ["CAMERA_SOURCE_KIND"]
    text = os.environ.get("CAMERA_SOURCE_TEXT", "")
    source_path = os.environ.get("CAMERA_SOURCE_JPEG") or None
    if source_kind not in {"text", "photo"}:
        raise ValueError("CAMERA_SOURCE_KIND must be text or photo")
    if (source_kind == "text") != (source_path is None):
        raise ValueError("text mode has no JPEG; photo mode requires CAMERA_SOURCE_JPEG")
    if source_kind == "text" and not text.strip():
        raise ValueError("text mode requires CAMERA_SOURCE_TEXT")
    if source_kind == "photo" and not text.strip():
        text = ""
    source_digest = None
    if source_path:
        source_digest = os.environ.get("CAMERA_SOURCE_SHA256")
        original = source_jpeg(source_path, source_digest, ROOT)
        assert original is not None
        if hashlib.sha256(original).hexdigest() != source_digest:
            raise AssertionError("source JPEG changed during bounded preflight")

    native_config = None
    if native:
        from openharness.config.paths import get_config_file_path
        from openharness.config.settings import load_settings

        config_text = os.environ["CAMERA_NATIVE_CONFIG_DIR"]
        native_config = Path(config_text).resolve(strict=True)
        if not native_config.is_dir():
            raise ValueError("CAMERA_NATIVE_CONFIG_DIR must name a read-only settings directory")
        os.environ["OPENHARNESS_CONFIG_DIR"] = str(native_config)
        os.environ["OPENHARNESS_PROFILE"] = "codex"
        settings = load_settings(get_config_file_path())
        bot_client, continuation_client = native_person_source_clients(
            settings, scenario=scenario
        )
        if bot_client is continuation_client:
            raise AssertionError("native clients must be separate")
        from openharness.api.codex_client import CodexApiClient

        if not isinstance(bot_client, CodexApiClient) or not isinstance(
            continuation_client, CodexApiClient
        ):
            raise AssertionError("native clients must both be Codex subscription clients")
    elif os.environ.get("CAMERA_PERSON_SOURCE_CASE"):
        from person_source_acceptance import SyntheticPersonSourceApi

        bot_client = SyntheticPersonSourceApi(os.environ["CAMERA_PERSON_SOURCE_CASE"])
    else:
        bot_client = OfflinePersonSourceApi(
            outcome=os.environ.get("CAMERA_OFFLINE_OUTCOME", "meal"),
            basis="text" if source_kind == "text" else "image",
        )

    if source_path:
        current_source = source_jpeg(source_path, source_digest, ROOT)
        if current_source is None or hashlib.sha256(current_source).hexdigest() != source_digest:
            raise ValueError("person-source JPEG changed after native preflight")
    message = person_source_message(
        sender_id=sender_id,
        chat_id=chat_id,
        source_message_id=source_id,
        sent_at=sent_at,
        text=text,
        photo_path=source_path,
    )
    workspace, session = unique_honcho_scope()
    root = create_storage_run_dir(ROOT)
    print(f"Person-source run artifacts: {root}", flush=True)

    from ohmo.memory_service.honcho_client import HonchoClient

    lower = datetime(1970, 1, 1, tzinfo=timezone.utc)
    async with HonchoClient(honcho_url, "local-auth-disabled", workspace) as honcho:

        async def read_before(until: datetime):
            try:
                return {
                    "status": "complete",
                    "rows": await honcho_history_snapshot(
                        honcho, session=session, since=lower, until=until
                    ),
                }
            except Exception as exc:
                return {"status": "error", "rows": None, "error": f"{type(exc).__name__}: {exc}"}

        trajectory = await run_person_source_turn(
            root=root,
            message=message,
            owner_id=owner_id,
            honcho_url=honcho_url,
            workspace=workspace,
            session=session,
            bot_client=bot_client,
            native_mode=native,
            config_dir=native_config,
            before_turn=read_before,
        )
        replay_trajectory = None
        after_first_result = None
        if os.environ.get("CAMERA_PERSON_SOURCE_CASE") == "send_time_meal":
            first_completed_at = datetime.fromisoformat(trajectory["completed_at"])
            try:
                after_first_result = {
                    "status": "complete",
                    "rows": await honcho_history_snapshot(
                        honcho, session=session, since=lower, until=first_completed_at
                    ),
                }
            except Exception as exc:
                after_first_result = {
                    "status": "error", "rows": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            replay_trajectory = await run_person_source_turn(
                root=root,
                message=message,
                owner_id=owner_id,
                honcho_url=honcho_url,
                workspace=workspace,
                session=session,
                bot_client=bot_client,
                native_mode=native,
                config_dir=native_config,
                before_turn=read_before,
            )
        before_result = trajectory["before_evidence"]
        before = before_result.get("rows") or []
        before_until = datetime.fromisoformat(trajectory["before_until"])
        completed_at = datetime.fromisoformat(
            (replay_trajectory or trajectory)["completed_at"]
        )
        try:
            after_result = {
                "status": "complete",
                "rows": await honcho_history_snapshot(
                    honcho, session=session, since=lower, until=completed_at
                ),
            }
        except Exception as exc:
            after_result = {"status": "error", "rows": None, "error": f"{type(exc).__name__}: {exc}"}
        after = after_result.get("rows") or []

    expected_event = trajectory["save"]["event_id"]
    projection: dict[str, object] = {
        "status": "not_attempted",
        "event_id": expected_event,
        "event": None,
        "effective_meal": None,
        "wellness": None,
    }
    projected_rows: list[dict] = []
    person_case = os.environ.get("CAMERA_PERSON_SOURCE_CASE")
    person_acceptance = person_case is not None
    if expected_event or person_acceptance:
        try:
            from mcp.server.fastmcp.server import FastMCP
            from mcp.types import ToolAnnotations
            sys.path.insert(0, str(telegent_path))
            from telegent.health_advisor.nutrition.config import NutritionHonchoCredentials
            from telegent.health_advisor.nutrition.store import NutritionDataStore
            from telegent.health_advisor.nutrition.sync import sync_nutrition_honcho
            from telegent.health_advisor.storage import HealthDataStore
            from telegent.mcp_simple_auth.wellness import register_wellness_tools

            from probe_support import call_wellness_with_synthetic_self, synthetic_wellness_self_scope

            credentials = NutritionHonchoCredentials.model_validate({
                "schema_version": 1,
                "base_url": honcho_url,
                "sources": [{
                    "user_id": owner_id,
                    "workspace": workspace,
                    "session": session,
                    "workspace_jwt": "local-auth-disabled",
                    "start_at": lower.isoformat(),
                    "device_ids": [],
                    "timezone": "UTC",
                }],
            })
            database = NutritionDataStore(root / "nutrition.db")
            health = HealthDataStore(root / "health.db")
            registry, authorization, authorized_read = synthetic_wellness_self_scope(owner_id)
            app = FastMCP(name="person-source-projection")

            async def get_health_store():
                return health

            async def get_nutrition_store():
                return database

            register_wellness_tools(
                app,
                read_only_annotations=ToolAnnotations(readOnlyHint=True),
                get_health_store=get_health_store,
                get_nutrition_store=get_nutrition_store,
                participant_registry=registry,
                authorization_context=authorization,
            )
            try:
                await sync_nutrition_honcho(credentials=credentials, db=database)
                record = database.get_record_by_event_id(owner_id, expected_event) if expected_event else None
                meal = database.get_current_meal(owner_id, record.meal_id) if record else None
                event_time = (
                    record.nutrition.meal_at or record.capture_time
                    if record else sent_at
                )
                day_start = event_time - timedelta(days=1)
                day_end = event_time + timedelta(days=1)
                wellness_result = await call_wellness_with_synthetic_self(
                    app,
                    authorization,
                    authorized_read,
                    {"params": {"start": day_start.isoformat(), "end": day_end.isoformat()}},
                )
                projected_rows = wellness_result[1]["nutrition_records"]
                projected_row = next(
                    (row for row in projected_rows if row.get("latest_event_id") == expected_event),
                    None,
                )
                records_before_close = list(database.iter_records(owner_id))
                meals_before_close = list(database.iter_current_meals(owner_id))
                reopened_record = None
                reopened_meal = None
                database.close()
                database = NutritionDataStore(root / "nutrition.db")
                reopened_records = list(database.iter_records(owner_id))
                reopened_meals = list(database.iter_current_meals(owner_id))
                reopened_record = (
                    database.get_record_by_event_id(owner_id, expected_event)
                    if expected_event else None
                )
                reopened_meal = (
                    database.get_current_meal(owner_id, reopened_record.meal_id)
                    if reopened_record else None
                )
                projection = {
                    "status": (
                        "projected"
                        if expected_event and reopened_record is not None and reopened_meal is not None
                        else "complete_absence"
                        if person_acceptance and not reopened_records and not reopened_meals
                        and before_result["status"] == after_result["status"] == "complete"
                        and not new_nutrition_event_ids(before, after)
                        else "missing"
                    ),
                    "event_id": expected_event,
                    "store_reopened": True,
                    "event": record.model_dump(mode="json") if record else None,
                    "effective_meal": meal.model_dump(mode="json") if meal else None,
                    "effective_meal_date": meal.local_day(ZoneInfo("UTC")) if meal else None,
                    "intake_kcal": projected_intake_kcal(projected_rows),
                    "honcho_event": next(
                        (row for row in after if row["id"] == expected_event), None
                    ),
                    "wellness": projected_row,
                    "all_wellness_rows": projected_rows,
                    "new_nutrition_event_ids": new_nutrition_event_ids(before, after),
                    "records_before_close": [row.model_dump(mode="json") for row in records_before_close],
                    "current_meals_before_close": [row.model_dump(mode="json") for row in meals_before_close],
                    "reopened_records": [row.model_dump(mode="json") for row in reopened_records],
                    "reopened_current_meals": [row.model_dump(mode="json") for row in reopened_meals],
                    "reopened_event": reopened_record.model_dump(mode="json") if reopened_record else None,
                    "reopened_effective_meal": reopened_meal.model_dump(mode="json") if reopened_meal else None,
                    "reopened_effective_meal_date": (
                        reopened_meal.local_day(ZoneInfo("UTC")) if reopened_meal else None
                    ),
                }
            finally:
                health.close()
                database.close()
        except Exception as exc:
            projection = {
                "status": "error",
                "event_id": expected_event,
                "error": f"{type(exc).__name__}: {exc}",
                "event": None,
                "effective_meal": None,
                "wellness": None,
            }
    elif trajectory["save"]["ambiguous"]:
        projection["status"] = "ambiguous_append_receipts"
    elif source_kind == "text" and os.environ.get("CAMERA_OFFLINE_OUTCOME") == "nonfood":
        observed = new_nutrition_event_ids(before, after)
        projection["status"] = (
            "complete_absence_nonfood"
            if not observed and before_result["status"] == after_result["status"] == "complete"
            else "unexpected_nutrition_events" if observed else "absence_unverified"
        )
        projection["observed_nutrition_event_ids"] = observed
    else:
        observed = new_nutrition_event_ids(before, after)
        projection["status"] = (
            "event_without_runtime_receipt" if observed
            else "no_append_receipt" if before_result["status"] == after_result["status"] == "complete"
            else "history_unavailable"
        )
        projection["observed_nutrition_event_ids"] = observed

    replay_result = None
    if replay_trajectory is not None:
        replay_after = (replay_trajectory.get("before_evidence") or {}).get("rows") or []
        replay_result = {
            "same_source": (
                replay_trajectory.get("source_message_id") == trajectory.get("source_message_id")
                and hashlib.sha256(source_jpeg(source_path, source_digest, ROOT) or b"").hexdigest()
                == source_digest
            ),
            "fixture_saw_known_photo_context": getattr(bot_client, "saw_same_photo_context", False),
            "save": replay_trajectory["save"],
            "new_nutrition_event_ids": new_nutrition_event_ids(
                after_first_result.get("rows") or [], replay_after
            ) if after_first_result else [],
            "event_count_after_replay": len(new_nutrition_event_ids(before, after)),
            "after_first_status": after_first_result["status"] if after_first_result else "missing",
            "after_first": (after_first_result or {}).get("rows"),
            "after_replay": after,
        }

    artifact = {
        "mode": mode,
        "persistence_chain_complete": bool(
            trajectory["save"]["saved"]
            and projection["status"] in {"projected", "projected_and_reopened"}
            and any(row["id"] == expected_event for row in after)
        ),
        "native_persistence_chain_complete": bool(
            native
            and trajectory["save"]["saved"]
            and projection["status"] in {"projected", "projected_and_reopened"}
            and any(row["id"] == expected_event for row in after)
        ),
        "source_pair": source_pair,
        "interaction_boundary": {
            "runtime_turns": 2 if replay_trajectory is not None else 1,
            "virtual_user_continuation": "not_run",
            "continuation_client": "preflighted_unused" if native else "not_created",
            "public_scenario_use": "retained_in_artifact_only",
        },
        "scope": {"owner_id": owner_id, "workspace": workspace, "session": session},
        "cutoffs": {
            "since": lower.isoformat(),
            "before_until": before_until.isoformat(),
            "after_until": completed_at.isoformat(),
            "peer_id": "ohmo",
        },
        "input": {
            "channel": "telegram",
            "sender_id": sender_id,
            "chat_id": chat_id,
            "chat_classification": "caller asserts private Telegram chat; no family grant",
            "source_message_id": source_id,
            "sent_at": sent_at.isoformat(),
            "kind": source_kind,
            "text": text if source_kind == "text" else "",
            "photo_sha256": source_digest,
        },
        "trajectory": {
            key: value for key, value in trajectory.items()
            if key not in {"runtime_episodes", "before_evidence"}
        },
        "replay": replay_result,
        "runtime_episodes": trajectory["runtime_episodes"],
        "honcho": {
            "before_status": before_result["status"],
            "before_error": before_result.get("error"),
            "before": before,
            "after_status": after_result["status"],
            "after_error": after_result.get("error"),
            "after": after,
        },
        "projection": projection,
        "projection_authority": "synthetic in-process self scope; not a wire authorization claim",
        "public_owner_scenario": scenario,
    }
    final_telegent = verify_source_tree_pin(telegent_path, telegent_sha)
    if final_telegent != (telegent_path, *telegent_state):
        raise ValueError("person-source run changed the caller-pinned Telegent checkout")
    if native:
        final_openharness = verify_source_tree_pin(ROOT, openharness_sha)
        if final_openharness != openharness_state:
            raise ValueError("native person-source run changed the OpenHarness checkout")
    if source_path:
        final_source = source_jpeg(source_path, source_digest, ROOT)
        if final_source is None or hashlib.sha256(final_source).hexdigest() != source_digest:
            raise ValueError("person-source JPEG changed during the runtime trajectory")
    output = root / "person-source-evidence.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=str) + "\n")
    output.chmod(0o600)
    print(json.dumps({
        "artifact": str(output),
        "save": trajectory["save"],
        "projection_status": projection["status"],
        "persistence_chain_complete": artifact["persistence_chain_complete"],
        "honcho_before_rows": len(before),
        "honcho_after_rows": len(after),
        "runtime_episode_ids": trajectory["episode_ids"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
