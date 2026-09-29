"""Deterministic real-Honcho -> real-Telegent exact-event storage probe.

The nutrition finalization metadata is a validated synthetic fixture. This
does not run Ohmo's model finalizer or a Camera owner-answer turn.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
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


async def main() -> None:
    url = os.environ["CAMERA_HONCHO_URL"]
    run_id = uuid4().hex
    owner = f"camera_synthetic_{run_id}"
    workspace = f"camera-storage-{run_id}"
    session = "nutrition"
    source_message_id = f"synthetic-source-{run_id}"
    now = datetime.now(UTC)
    day = now.date().isoformat()
    scratch_parent = ROOT / "tmp" / "camera-e2e" / "storage-runs"
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
                    "start_at": (now - timedelta(minutes=5)).isoformat(),
                    "device_ids": [],
                    "timezone": "UTC",
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
                    "start": (now - timedelta(days=1)).isoformat(),
                    "end": (now + timedelta(days=1)).isoformat(),
                }
            },
        )
        payload = result[1]
        meals = payload["nutrition_records"]
        return meals, sum(meal["energy_kcal_best"] or 0 for meal in meals)

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
            data["source_message_id"] = source_message_id
        else:
            data["reply_to_source_message_id"] = reply
        return data

    try:
        async with HonchoClient(url, "local-auth-disabled", workspace) as client:
            await client.get_or_create_workspace()
            await client.get_or_create_peer("ohmo")
            await client.get_or_create_peer("owner")
            await client.get_or_create_session(session, peers={"ohmo": {}, "owner": {}})

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

            observation = await append(
                {
                    "schema_version": 2,
                    "record_type": "meal_observation",
                    "basis": ["image"],
                    "consumption_status": "consumed",
                    "meal_date": day,
                    "energy_kcal_min": 320,
                    "energy_kcal_max": 320,
                    "energy_kcal_best": 320,
                },
                phase="observation",
            )
            first = await sync_nutrition_honcho(credentials=credentials, db=db)
            original = db.get_record_by_event_id(owner, observation.id)
            assert original is not None, (observation.id, first)
            original_snapshot = original.model_dump(mode="json")
            meal = db.get_current_meal(owner, original.meal_id)
            assert meal is not None and meal.event_ids == [observation.id], meal
            assert meal.latest_event_id == observation.id
            meals, kcal = await balance()
            assert len(meals) == 1 and meals[0]["latest_event_id"] == observation.id
            assert kcal == 320, (observation.id, kcal, meals)
            print(f"PASS observation event={observation.id} expected_kcal=320 actual_kcal={kcal}")

            first_cursor = db.get_cursor(source_id).cursor
            first_meal = meal.model_dump(mode="json")
            replay = await sync_nutrition_honcho(credentials=credentials, db=db)
            meals, kcal = await balance()
            assert db.get_cursor(source_id).cursor == first_cursor
            assert (
                db.get_current_meal(owner, original.meal_id).model_dump(mode="json") == first_meal
            )
            assert len(meals) == 1 and kcal == 320, (replay, kcal, meals)
            print(
                f"PASS observation replay event={observation.id} expected_kcal=320 actual_kcal={kcal}"
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
                reply=source_message_id,
            )
            await sync_nutrition_honcho(credentials=credentials, db=db)
            assert (
                db.get_record_by_event_id(owner, observation.id).model_dump(mode="json")
                == original_snapshot
            )
            assert db.get_record_by_event_id(owner, correction.id) is not None
            meal = db.get_current_meal(owner, original.meal_id)
            assert meal is not None and meal.event_ids == [observation.id, correction.id], meal
            assert meal.latest_event_id == correction.id
            assert meal.consumption_status == "not_consumed"
            meals, kcal = await balance()
            assert len(meals) == 1 and meals[0]["latest_event_id"] == correction.id
            assert kcal == 0, (correction.id, kcal, meals)
            print(
                f"PASS correction event={correction.id} original={observation.id} expected_kcal=0 actual_kcal={kcal}"
            )

            correction_cursor = db.get_cursor(source_id).cursor
            correction_meal = meal.model_dump(mode="json")
            final_replay = await sync_nutrition_honcho(credentials=credentials, db=db)
            meals, kcal = await balance()
            assert db.get_cursor(source_id).cursor == correction_cursor
            assert (
                db.get_current_meal(owner, original.meal_id).model_dump(mode="json")
                == correction_meal
            )
            assert len(meals) == 1 and kcal == 0, (final_replay, kcal, meals)
            print(
                f"PASS correction replay event={correction.id} expected_kcal=0 actual_kcal={kcal}"
            )
            print(
                f"PASS exact-event chain: observation={observation.id} correction={correction.id} projection={scratch / 'nutrition.db'}"
            )
    finally:
        health.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
