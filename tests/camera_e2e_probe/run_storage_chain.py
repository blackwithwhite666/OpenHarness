"""Deterministic real-Honcho -> real-Telegent exact-event storage probe.

The nutrition finalization metadata is a validated synthetic fixture. This
does not run Ohmo's model finalizer or a Camera owner-answer turn.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from math import isclose
from pathlib import Path
from uuid import uuid4

from mcp.server.fastmcp.server import FastMCP
from mcp.types import ToolAnnotations

from ohmo.evals.nutrition_trace import NutritionAnnotationV2
from ohmo.memory_service.honcho_client import HonchoClient

ROOT = Path(__file__).resolve().parents[2]
TELEGENT = Path(os.environ["CAMERA_TELEGENT_WORKTREE"]).resolve()
sys.path.insert(0, str(TELEGENT))

from telegent.health_advisor.nutrition.config import (  # noqa: E402
    NutritionHonchoCredentials,
)
from telegent.health_advisor.nutrition.store import NutritionDataStore  # noqa: E402
from telegent.health_advisor.nutrition.sync import sync_nutrition_honcho  # noqa: E402
from telegent.health_advisor.storage import HealthDataStore  # noqa: E402
from telegent.mcp_simple_auth.wellness import register_wellness_tools  # noqa: E402


def assert_same_window_bound(actual: str, expected: datetime, *, label: str) -> None:
    """Compare a serialized bound by exact instant, including Z output."""
    parsed = datetime.fromisoformat(actual.replace("Z", "+00:00"))
    assert (
        parsed.tzinfo is not None
        and parsed.utcoffset() is not None
        and expected.tzinfo is not None
        and expected.utcoffset() is not None
    ), (
        f"{label} must remain timezone-aware: actual={actual!r}, expected={expected!r}"
    )
    assert parsed == expected, (
        f"{label} differs: actual={actual!r} parsed={parsed!r}, expected={expected!r}"
    )


async def main() -> None:
    url = os.environ["CAMERA_HONCHO_URL"]
    run_id = uuid4().hex
    owner = f"camera_synthetic_{run_id}"
    workspace = f"camera-storage-{run_id}"
    session = "nutrition"
    source_message_id = f"synthetic-source-{run_id}"
    device_id = f"camera-watch-{run_id}"
    # This fixed window crosses local midnight in Europe/Moscow. The request
    # uses different offsets for the same instants; samples at both endpoints
    # must remain members after conversion to UTC.
    start = datetime.fromisoformat("2026-10-05T23:50:00+03:00")
    end = datetime.fromisoformat("2026-10-06T00:10:00+03:00").astimezone(timezone.utc)
    scratch_parent = ROOT / "tmp" / "camera-native-docker" / "storage-runs"
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="run-", dir=scratch_parent))
    db = NutritionDataStore(scratch / "nutrition.db")
    health = HealthDataStore(scratch / "health.db")
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
                    "start_at": (start - timedelta(days=1)).isoformat(),
                    "device_ids": [device_id],
                    "timezone": "Europe/Moscow",
                }
            ],
        }
    )
    source_id = credentials.sources[0].cursor_source_id
    app = FastMCP(name="camera-storage-probe")

    async def get_health_store() -> HealthDataStore:
        return health

    async def get_nutrition_store() -> NutritionDataStore:
        return db

    register_wellness_tools(
        app,
        read_only_annotations=ToolAnnotations(readOnlyHint=True),
        get_health_store=get_health_store,
        get_nutrition_store=get_nutrition_store,
        wellness_user_id=owner,
    )

    async def balance():
        result = await app.call_tool(
            "get_wellness_data",
            {
                "params": {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                }
            },
        )
        payload = result[1]
        meals = payload["nutrition_records"]
        assert_same_window_bound(
            payload["interval"]["start"], start, label="nutrition interval start"
        )
        assert_same_window_bound(
            payload["interval"]["end"], end, label="nutrition interval end"
        )
        assert payload["nutrition_status"] == "complete", payload["nutrition_status"]
        return payload, meals

    def set_energy_fixture() -> None:
        """Persist deterministic, unit-certified synthetic HAE fixture points."""
        from telegent.health_advisor.models import SampleClass
        from telegent.health_advisor.storage import keys
        from telegent.health_advisor.storage.blocks import pack_block

        basal_type = "HealthAutoExportMetric_basal_energy_burned"
        active_type = "HealthAutoExportMetric_active_energy"
        points = (
            (basal_type, start, 100.0, "kcal", "inside-start"),
            (basal_type, end, 418.4, "kJ", "inside-end"),
            (active_type, end, 41.84, "kJ", "inside-end-active"),
            (basal_type, start - timedelta(seconds=1), 50000.0, "kJ", "before"),
            (active_type, end + timedelta(seconds=1), 50000.0, "kJ", "after"),
        )
        by_type_and_day: dict[
            tuple[str, str], list[tuple[datetime, float, str, str]]
        ] = {}
        for sample_type, when, value, unit, label in points:
            utc_day = when.astimezone(timezone.utc).date().isoformat()
            by_type_and_day.setdefault((sample_type, utc_day), []).append(
                (when, value, unit, label)
            )
        health.save_meta(device_id, {"observed_types": [basal_type, active_type]})
        for (sample_type, utc_day), rows in by_type_and_day.items():
            unit_by_uuid = {
                f"{run_id}-{label}": unit for _, _, unit, label in rows
            }
            health.merge_block(
                device_id,
                sample_type,
                utc_day,
                sample_class=SampleClass.quantity,
                unit=None,
                columns={
                    "start": [int(when.timestamp() * 1000) for when, _, _, _ in rows],
                    "end": [int(when.timestamp() * 1000) for when, _, _, _ in rows],
                    "value": [value for _, value, _, _ in rows],
                    "source": ["synthetic-watch"] * len(rows),
                    "uuid": [f"{run_id}-{label}" for _, _, _, label in rows],
                },
            )
            block = health.get_block(device_id, sample_type, utc_day)
            assert block is not None
            block["point_unit"] = [unit_by_uuid[uuid] for uuid in block["uuid"]]
            health._db.put(
                keys.block_key(device_id, sample_type, utc_day), pack_block(block)
            )

    async def append_food(
        phase: str,
        when: datetime,
        kcal: int,
        *,
        reply: str | None = None,
    ):
        return await append(
            {
                "schema_version": 2,
                "record_type": "meal_observation",
                "basis": ["image"],
                "consumption_status": "consumed",
                "meal_at": when.isoformat(),
                "energy_kcal_min": kcal,
                "energy_kcal_max": kcal,
                "energy_kcal_best": kcal,
            },
            phase=phase,
            reply=reply,
        )

    def metadata(annotation: dict, *, operation: str, reply: str | None = None) -> dict:
        validated = NutritionAnnotationV2.model_validate(annotation)
        data = {
            "client_op_id": operation,
            "role": "assistant",
            "tenant_id": owner,
            "source_principal": "telegram:synthetic-owner",
            "gateway_session_id": session,
            "logical_turn_id": operation,
            "nutrition_annotation_status": "recorded",
            "decision_trace": {
                "episode_id": f"episode-{operation}",
                "annotations": {"nutrition": validated.model_dump(mode="json", exclude_unset=True)},
            },
        }
        if reply is None:
            data["source_message_id"] = f"{source_message_id}-{operation}"
        else:
            data["reply_to_source_message_id"] = reply
        return data

    try:
        async with HonchoClient(url, "local-auth-disabled", workspace) as client:
            await client.get_or_create_workspace()
            await client.get_or_create_peer("ohmo")
            await client.get_or_create_peer("owner")
            await client.get_or_create_session(session, peers={"ohmo": {}, "owner": {}})

            set_energy_fixture()

            async def append(annotation: dict, *, phase: str, reply: str | None = None):
                operation = f"{run_id}-{phase}"
                created = await client.create_messages(
                    session,
                    [
                        {
                            "content": f"Synthetic {phase} fixture",
                            "peer_id": "ohmo",
                            "metadata": metadata(annotation, operation=operation, reply=reply),
                        }
                    ],
                )
                assert len(created) == 1, (phase, created)
                return created[0]

            outside_before = await append_food(
                "outside-before", start - timedelta(seconds=1), 71
            )
            observation = await append_food("observation", start, 320)
            outside_after = await append_food(
                "outside-after", end + timedelta(seconds=1), 83
            )
            observation_source_message_id = f"{source_message_id}-{run_id}-observation"
            first = await sync_nutrition_honcho(credentials=credentials, db=db)
            original = db.get_record_by_event_id(owner, observation.id)
            assert original is not None, (observation.id, first)
            original_snapshot = original.model_dump(mode="json")
            meal = db.get_current_meal(owner, original.meal_id)
            assert meal is not None and meal.event_ids == [observation.id], meal
            assert meal.latest_event_id == observation.id
            honcho_observation = await client.list_messages_in_window(
                session,
                expected_peer_id="ohmo",
                since=observation.created_at - timedelta(seconds=1),
                until=datetime.now(timezone.utc) + timedelta(minutes=1),
            )
            honcho_original = [
                item for item in honcho_observation if item["id"] == observation.id
            ]
            assert len(honcho_original) == 1, honcho_observation
            honcho_original_snapshot = honcho_original[0].copy()
            assert (
                honcho_original_snapshot["content"] == "Synthetic observation fixture"
            )
            payload, meals = await balance()
            intake = sum(meal["energy_kcal_best"] for meal in meals)
            assert len(meals) == 1 and meals[0]["latest_event_id"] == observation.id
            assert intake == 320, (observation.id, intake, meals)
            assert payload["linked_device_ids"] == [device_id]
            interval, = payload["energy_intervals"]
            interval_snapshot = interval.copy()
            assert_same_window_bound(interval["start"], start, label="energy interval start")
            assert_same_window_bound(interval["end"], end, label="energy interval end")
            assert interval["timezone"] == "Europe/Moscow"
            assert interval["basal_sum"] == 836.8 and interval["basal_unit"] == "kJ"
            assert interval["active_sum"] == 41.84 and interval["active_unit"] == "kJ"
            assert interval["basal_points"] == 2 and interval["active_points"] == 1
            assert interval["basal_conflicting_timestamps"] == 0
            assert interval["active_conflicting_timestamps"] == 0
            assert interval["unresolved_key_count"] == 0
            assert interval["legacy_synthetic_count"] == 0
            assert interval["snapshot_revision"] == payload["energy_snapshot_revision"]
            basal_kcal = interval["basal_sum"] / 4.184
            active_kcal = interval["active_sum"] / 4.184
            expenditure = basal_kcal + active_kcal
            net = intake - expenditure
            assert isclose(basal_kcal, 200)
            assert isclose(active_kcal, 10)
            assert isclose(expenditure, 210)
            assert isclose(net, 110), (intake, expenditure, net)
            assert payload["energy_days"]
            assert payload["energy_days"][0]["basal_sum"] > interval["basal_sum"]
            event_ids_after_sync = sorted(
                record.event_id for record in db.iter_records(owner)
            )
            assert event_ids_after_sync == sorted(
                [outside_before.id, observation.id, outside_after.id]
            )
            assert len(event_ids_after_sync) == 3
            print(
                f"PASS exact window intake={intake} kcal basal={basal_kcal} kcal "
                f"active={active_kcal} kcal expenditure={expenditure} kcal "
                f"net={net} kcal"
            )

            first_cursor = db.get_cursor(source_id).cursor
            first_meal = meal.model_dump(mode="json")
            await sync_nutrition_honcho(credentials=credentials, db=db)
            payload, meals = await balance()
            assert db.get_cursor(source_id).cursor == first_cursor
            assert sorted(
                record.event_id for record in db.iter_records(owner)
            ) == event_ids_after_sync
            assert (
                db.get_current_meal(owner, original.meal_id).model_dump(mode="json")
                == first_meal
            )
            assert len(meals) == 1 and sum(m["energy_kcal_best"] for m in meals) == 320
            assert payload["energy_intervals"] == [interval_snapshot]
            replay_intake = sum(m["energy_kcal_best"] for m in meals)
            replay_net = replay_intake - (
                interval["basal_sum"] / 4.184 + interval["active_sum"] / 4.184
            )
            assert replay_intake == 320 and isclose(replay_net, 110)
            print(
                f"PASS observation replay event={observation.id} "
                f"intake={replay_intake} kcal net={replay_net} kcal"
            )

            correction = await append(
                {
                    "schema_version": 2,
                    "record_type": "meal_correction",
                    "changed_fields": [
                        "consumption_status",
                        "energy_kcal_min",
                        "energy_kcal_max",
                        "energy_kcal_best",
                    ],
                    "consumption_status": "not_consumed",
                    "energy_kcal_min": 0,
                    "energy_kcal_max": 0,
                    "energy_kcal_best": 0,
                },
                phase="correction",
                reply=observation_source_message_id,
            )
            await sync_nutrition_honcho(credentials=credentials, db=db)
            honcho_after_correction = await client.list_messages_in_window(
                session,
                expected_peer_id="ohmo",
                since=observation.created_at - timedelta(seconds=1),
                until=datetime.now(timezone.utc) + timedelta(minutes=1),
            )
            honcho_original_after = [
                item for item in honcho_after_correction if item["id"] == observation.id
            ]
            assert honcho_original_after == [honcho_original_snapshot]
            assert (
                db.get_record_by_event_id(owner, observation.id).model_dump(mode="json")
                == original_snapshot
            )
            assert db.get_record_by_event_id(owner, correction.id) is not None
            event_ids_after_correction = sorted(
                record.event_id for record in db.iter_records(owner)
            )
            assert event_ids_after_correction == sorted(
                event_ids_after_sync + [correction.id]
            )
            meal = db.get_current_meal(owner, original.meal_id)
            assert meal is not None
            assert meal.event_ids == [observation.id, correction.id], meal
            assert meal.latest_event_id == correction.id
            assert meal.consumption_status == "not_consumed"
            payload, meals = await balance()
            assert meals == [], meals
            corrected_intake = 0
            corrected_interval, = payload["energy_intervals"]
            assert corrected_interval == interval_snapshot
            corrected_expenditure = (
                corrected_interval["basal_sum"] / 4.184
                + corrected_interval["active_sum"] / 4.184
            )
            corrected_net = corrected_intake - corrected_expenditure
            assert isclose(corrected_net, -210)
            print(
                f"PASS correction event={correction.id} original={observation.id} "
                f"intake={corrected_intake} kcal "
                f"expenditure={corrected_expenditure} kcal "
                f"net={corrected_net} kcal"
            )

            correction_cursor = db.get_cursor(source_id).cursor
            correction_meal = meal.model_dump(mode="json")
            await sync_nutrition_honcho(credentials=credentials, db=db)
            payload, meals = await balance()
            assert db.get_cursor(source_id).cursor == correction_cursor
            assert sorted(
                record.event_id for record in db.iter_records(owner)
            ) == event_ids_after_correction
            assert (
                db.get_current_meal(owner, original.meal_id).model_dump(mode="json")
                == correction_meal
            )
            assert meals == [], meals
            assert payload["energy_intervals"] == [interval_snapshot]
            assert isclose(corrected_net, -210)
            print(
                f"PASS correction replay event={correction.id} "
                f"intake=0 kcal net={corrected_net} kcal"
            )
            print(
                f"PASS exact-event chain: observation={observation.id} "
                f"correction={correction.id} projection={scratch / 'nutrition.db'}"
            )
    finally:
        health.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
