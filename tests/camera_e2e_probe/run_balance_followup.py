"""Reproject a frozen native Camera event with real Telegent energy facts.

This offline stage makes no model call. A native owner report is a separate
acceptance step; the JSON output is its factual oracle, not a tool stub.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from math import isclose
from pathlib import Path

from mcp.server.fastmcp.server import FastMCP
from mcp.types import ToolAnnotations

ROOT = Path(__file__).resolve().parents[2]
TELEGENT = Path(os.environ["CAMERA_TELEGENT_WORKTREE"]).resolve()
sys.path.insert(0, str(TELEGENT))

from balance_support import assert_fixture_balance, write_energy_fixture  # noqa: E402
from camera_runtime_support import e5_raw_honcho_row_fingerprint  # noqa: E402
from ohmo.evals.nutrition_trace import NutritionAnnotationV2  # noqa: E402
from ohmo.memory_service.honcho_client import HonchoClient  # noqa: E402
from probe_support import (  # noqa: E402
    call_wellness_with_synthetic_self, synthetic_wellness_self_scope,
)
from telegent.health_advisor.nutrition.config import NutritionHonchoCredentials  # noqa: E402
from telegent.health_advisor.nutrition.store import NutritionDataStore  # noqa: E402
from telegent.health_advisor.nutrition.sync import sync_nutrition_honcho  # noqa: E402
from telegent.health_advisor.storage import HealthDataStore  # noqa: E402
from telegent.mcp_simple_auth.wellness import register_wellness_tools  # noqa: E402


async def run(grade_dir: Path, honcho_url: str, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    permitted = (ROOT / "tmp" / "full-chain-balance").resolve()
    if output_dir != permitted and permitted not in output_dir.parents:
        raise ValueError("output must stay under tmp/full-chain-balance")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("fresh output directory required; existing evidence is immutable")
    manifest = json.loads((grade_dir / "manifest.json").read_text())
    snapshot = json.loads((grade_dir / "honcho-snapshot.json").read_text())
    goal, = manifest["goals"]
    a1 = json.loads((grade_dir / "a1-result.json").read_text())
    event_id, = a1["actual_event_ids"]
    if (goal["owner_id"] != "synthetic_owner"
            or goal["principal_id"] != "telegram:123"
            or a1["a1"] != "PASS" or a1["actual_latest_event_id"] != event_id
            or snapshot.get("complete") is not True
            or snapshot.get("workspace_id") != goal["workspace_id"]
            or snapshot.get("session_id") != goal["session_id"]):
        raise AssertionError("frozen native grade identity is inconsistent")
    rows = snapshot["messages"]
    frozen = [row for row in rows if row.get("id") == event_id]
    if len(frozen) != 1:
        raise AssertionError("frozen snapshot lacks exact event")
    frozen_fingerprint = e5_raw_honcho_row_fingerprint(frozen[0])
    started = datetime.fromisoformat(goal["trajectory_started_at"].replace("Z", "+00:00"))
    async with HonchoClient(honcho_url, "local-auth-disabled", goal["workspace_id"]) as client:
        live_rows = await client.list_messages_in_window(
            goal["session_id"], expected_peer_id="ohmo",
            since=started - timedelta(minutes=1), until=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
    live = [row for row in live_rows if row.get("id") == event_id]
    if len(live) != 1 or e5_raw_honcho_row_fingerprint(live[0]) != frozen_fingerprint:
        raise AssertionError("live Honcho event differs from immutable native grade")
    event = live[0]
    metadata = event["metadata"]
    trace = metadata.get("decision_trace")
    annotation = trace.get("annotations", {}).get("nutrition") if isinstance(trace, dict) else None
    nutrition = NutritionAnnotationV2.model_validate(annotation)
    if (nutrition.record_type != "meal_observation"
            or nutrition.consumption_status != "consumed"
            or nutrition.meal_at is None or nutrition.energy_kcal_best != a1["actual_kcal"]
            or metadata.get("tenant_id") != "synthetic_owner"
            or metadata.get("source_principal") != "telegram:123"
            or metadata.get("source_message_id") != goal["source_message_id"]
            or metadata.get("gateway_session_id") != goal["gateway_session_id"]):
        raise AssertionError("live model event differs from trusted Camera binding")
    capture = nutrition.meal_at
    start, end = capture - timedelta(hours=12), capture + timedelta(hours=12)
    device_id = "camera-balance-watch"
    db = NutritionDataStore(output_dir / "nutrition.db")
    health = HealthDataStore(output_dir / "health.db")
    registry, auth, read = synthetic_wellness_self_scope("synthetic_owner")
    credentials = NutritionHonchoCredentials.model_validate({
        "schema_version": 1, "base_url": honcho_url,
        "sources": [{
            "user_id": "synthetic_owner", "workspace": goal["workspace_id"],
            "session": goal["session_id"], "workspace_jwt": "local-auth-disabled",
            "start_at": (started - timedelta(minutes=1)).isoformat(),
            "device_ids": [device_id], "timezone": "UTC",
        }],
    })

    def app_for(current_db, current_health):
        app = FastMCP(name="camera-balance-factual-projection")

        async def get_health_store():
            return current_health

        async def get_nutrition_store():
            return current_db

        register_wellness_tools(
            app, read_only_annotations=ToolAnnotations(readOnlyHint=True),
            get_health_store=get_health_store, get_nutrition_store=get_nutrition_store,
            participant_registry=registry, authorization_context=auth,
        )
        return app

    async def read_payload(app):
        arguments = {"params": {"start": start.isoformat(), "end": end.isoformat()}}
        result = await call_wellness_with_synthetic_self(app, auth, read, arguments)
        return arguments, result[1]

    try:
        write_energy_fixture(health, device_id, start, end)
        await sync_nutrition_honcho(credentials=credentials, db=db)
        record = db.get_record_by_event_id("synthetic_owner", event_id)
        if record is None:
            raise AssertionError("exact native event did not sync")
        current = db.get_current_meal("synthetic_owner", record.meal_id)
        if current is None or current.latest_event_id != event_id:
            raise AssertionError("exact native event is not current")
        current_snapshot = current.model_dump(mode="json")
        arguments, payload = await read_payload(app_for(db, health))
        totals = assert_fixture_balance(
            payload, start=start, end=end, event_id=event_id, device_id=device_id,
        )
        if not all(isclose(actual, expected, abs_tol=1e-8)
                   for actual, expected in zip(totals, (137.0, 210.0, -73.0), strict=True)):
            raise AssertionError(f"unexpected exact native balance: {totals}")
        health.close()
        db.close()
        db = NutritionDataStore(output_dir / "nutrition.db")
        health = HealthDataStore(output_dir / "health.db")
        reopened = db.get_current_meal("synthetic_owner", record.meal_id)
        _, reopened_payload = await read_payload(app_for(db, health))
        if reopened is None or reopened.model_dump(mode="json") != current_snapshot or reopened_payload != payload:
            raise AssertionError("reopened current meal or wellness facts changed")
        await sync_nutrition_honcho(credentials=credentials, db=db)
        _, replay_payload = await read_payload(app_for(db, health))
        replay_meal = db.get_current_meal("synthetic_owner", record.meal_id)
        changed_keys = [key for key in payload
                        if key != "nutrition_last_success_at"
                        and replay_payload.get(key) != payload[key]]
        if (replay_meal is None or replay_meal.model_dump(mode="json") != current_snapshot
                or changed_keys):
            raise AssertionError(f"sync replay changed canonical facts: {changed_keys}")
        async with HonchoClient(honcho_url, "local-auth-disabled", goal["workspace_id"]) as client:
            after_rows = await client.list_messages_in_window(
                goal["session_id"], expected_peer_id="ohmo",
                since=started - timedelta(minutes=1), until=datetime.now(timezone.utc) + timedelta(minutes=1),
            )
        after = [row for row in after_rows if row.get("id") == event_id]
        if len(after) != 1 or e5_raw_honcho_row_fingerprint(after[0]) != frozen_fingerprint:
            raise AssertionError("original Honcho row changed during projection")
        (output_dir / "facts.json").write_text(json.dumps({
            "status": "offline_real_honcho_real_telegent",
            "event_id": event_id, "meal_id": record.meal_id,
            "honcho_fingerprint": frozen_fingerprint,
            "query": arguments, "payload": payload,
            "intake_kcal": totals[0], "expenditure_kcal": totals[1],
            "observed_balance_kcal": totals[2], "reopen_replay_stable": True,
        }, indent=2, ensure_ascii=False) + "\n")
        print(f"PASS exact native event {event_id}: {totals[0]} - {totals[1]} = {totals[2]} kcal")
    finally:
        health.close()
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-grade-dir", type=Path, required=True)
    parser.add_argument("--honcho-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.retained_grade_dir, args.honcho_url, args.output_dir))
