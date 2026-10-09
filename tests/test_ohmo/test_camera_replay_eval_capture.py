"""Recorder and grade checks for ordinary Camera owner turns and corrections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import sqlite3

import pytest

from ohmo.evals.nutrition_persistence import (
    Goal, Manifest, bind_wellness_snapshot, derive_meal_id, export_eval_dialogue,
    grade_manifest, validate_dialogue_binding, _fold_events,
)
from tests.test_ohmo import test_camera_f84_joint_runtime as joint
from tests.test_ohmo.test_camera_correction_replay import (
    _correction, _events, _ordinary_history, _quantity,
)


def _grade_actual_history(pool, server, original, first, latest):
    """Grade actual recorder/append evidence against the matching canonical read shape."""
    recorders = [original[4], first[4], latest[4]]
    exported = _episodes(pool, recorders)
    original_id = original[3].metadata["nutrition_append_event_id"]
    latest_id = latest[3].metadata["nutrition_append_event_id"]
    original_row = next(row for row in server.rows if row["id"] == original_id)
    latest_row = next(row for row in server.rows if row["id"] == latest_id)
    effective = _fold_events(_events(server, "camera-owner-source"))
    started = datetime.fromisoformat(original_row["created_at"])
    as_of = datetime.fromisoformat(latest_row["created_at"])
    queried = as_of + timedelta(days=2)
    goal = Goal(
        case_id="ordinary-quantity-history", episode_ids=[item.episode_id for item in recorders],
        owner_id="review-tenant", principal_id="telegram:123", workspace_id="review-workspace",
        eval_workspace=str(pool._workspace), peer_id="ohmo",
        canonical_owner_id="review-tenant", canonical_login="owner",
        session_id="review-session", gateway_session_id="gateway-session",
        source_message_id="camera-owner-source", meal_date=datetime.fromisoformat(
            effective["meal_date"]
        ).date(), meal_timezone="UTC", trajectory_started_at=started,
        trajectory_as_of=as_of, logical_turn_id=original_row["metadata"]["logical_turn_id"],
        trace_episode_id=original_row["metadata"]["decision_trace_episode_id"],
        operation_id=original_row["metadata"]["client_op_id"],
        canonical_meal_id=derive_meal_id(
            tenant_id="review-tenant", source_principal="telegram:123",
            gateway_session_id="gateway-session", source_message_id="camera-owner-source",
        ), expected_consumed=True, expected_kcal=750,
        expectation_origin="reviewed_user_dialogue", expectation_source="synthetic-joint-runtime-review",
    )
    manifest = Manifest(schema_version=1, goals=[goal])
    binding = validate_dialogue_binding(manifest, exported)[goal.case_id]
    assert binding["complete"] is True, binding
    honcho = {
        "complete": True, "workspace_id": goal.workspace_id, "session_id": goal.session_id,
        "owner_id": goal.owner_id, "since": started.isoformat(), "until": as_of.isoformat(),
        "queried_at": queried.isoformat(), "messages": [
            deepcopy(row) for row in server.rows
            if row["peer_id"] == goal.peer_id
            and row["session_id"] == goal.session_id
            and row["workspace_id"] == goal.workspace_id
            and started <= datetime.fromisoformat(row["created_at"]) <= as_of
        ],
    }
    canonical = {
        "meal_id": goal.canonical_meal_id, "revision": effective["revision"],
        "status": "active", "latest_event_id": latest_id, "day": effective["meal_date"],
        "provisional": True, "capture_time": latest_row["created_at"],
        "meal_at": effective["meal_at"], "meal_date": effective["meal_date_field"],
        "source_message_id": goal.source_message_id, "ingest_source": "telegram",
        "confirmation_required": False, "reply_to_source_message_id": None,
        "received_at": None, "is_forwarded": False, "source_message_at": None,
        "is_estimate": True, "basis": ["image"],
        "consumption_status": "consumed", "energy_kcal_min": 750,
        "energy_kcal_max": 750, "energy_kcal_best": effective["energy_kcal_best"],
        "protein_g": None, "fat_g": None, "carbohydrate_g": None,
        "items": effective["items"], "confidence": "medium",
        "assumptions": [], "warnings": [],
    }
    day_start = datetime.combine(goal.meal_date, datetime.min.time(), goal.trajectory_started_at.tzinfo)
    wellness = bind_wellness_snapshot({
        "complete": True, "user_id": goal.canonical_owner_id, "login": goal.canonical_login,
        "start": min(started, day_start).isoformat(), "end": queried.isoformat(),
        "queried_at": queried.isoformat(), "meals": [canonical], "unassigned": [],
    }, goal=goal)
    return grade_manifest(
        manifest, honcho, wellness, now=queried,
        reviewed_turn_sources=binding["reviewed_turn_sources"],
        reviewed_turn_provenance=binding["reviewed_turn_provenance"],
    )[0]


def _episodes(pool, recorders):
    ids = [recorder.episode_id for recorder in recorders]
    exported = export_eval_dialogue(pool._workspace / "evals", episode_ids=ids)
    assert len(exported["episodes"]) == len(ids)
    assert {item["episode"]["episode_id"] for item in exported["episodes"]} == set(ids)
    return exported


@pytest.mark.asyncio
async def test_complete_quantity_history_reaches_normal_a1_and_rejects_wrong_target(
    tmp_path, monkeypatch,
):
    pool, bundle, server, client, photo_ref, original = await _ordinary_history(tmp_path, monkeypatch)
    try:
        first = await _correction(
            pool, bundle, photo_ref, "quantity-first", "It was one piece",
            _quantity("one piece"), minutes=1,
        )
        latest = await _correction(
            pool, bundle, photo_ref, "quantity-latest", "Actually half a serving",
            _quantity("half a serving"), minutes=2,
        )
        exported = _episodes(pool, [original[4], first[4], latest[4]])
        for final, recorder in ((original[3], original[4]),
                                (first[3], first[4]), (latest[3], latest[4])):
            matching = next(item for item in exported["episodes"]
                            if item["episode"]["episode_id"] == recorder.episode_id)
            provenance = matching["turn_provenance"][0]
            actual = provenance["gateway_final_metadata"]["nutrition_actual_append_receipt"]
            persisted = next(row for row in server.rows if row["id"] == actual["event_id"])
            assert actual["event_id"] == final.metadata["nutrition_append_event_id"]
            assert actual["client_op_id"] == persisted["metadata"]["client_op_id"]
            assert actual["source_message_id"] == persisted["metadata"]["source_message_id"]
            assert actual["annotation"] == persisted["metadata"]["decision_trace"]["annotations"]["nutrition"]

        grade = _grade_actual_history(pool, server, original, first, latest)
        assert (grade["a1"], grade["stage"], grade.get("actual_latest_event_id")) == (
            "PASS", "SAME_EVENT_PROJECTED", latest[3].metadata["nutrition_append_event_id"],
        ), repr(grade)

        row = next(row for row in server.rows
                   if row["id"] == latest[3].metadata["nutrition_append_event_id"])
        original_metadata = deepcopy(row["metadata"])
        row["metadata"] = {**row["metadata"], "target_meal_id": "wrong-meal"}
        try:
            forged = _grade_actual_history(pool, server, original, first, latest)
            assert (forged["a1"], forged["stage"]) == (
                "FAIL", "HONCHO_TARGET_MISMATCH",
            ), forged
        finally:
            row["metadata"] = original_metadata
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_old_exact_replay_after_newer_correction_has_no_false_saved_episode(
    tmp_path, monkeypatch,
):
    pool, bundle, server, client, photo_ref, original = await _ordinary_history(
        tmp_path, monkeypatch, name="recorder-replay",
    )
    try:
        first = await _correction(
            pool, bundle, photo_ref, "quantity-first", "It was one piece",
            _quantity("one piece"), minutes=1,
        )
        latest = await _correction(
            pool, bundle, photo_ref, "quantity-latest", "Actually half a serving",
            _quantity("half a serving"), minutes=2,
        )
        path = pool._workspace / "evals" / "evals.sqlite"
        with sqlite3.connect(path) as connection:
            episode_count = connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        rows_before = deepcopy(server.rows)
        prompts_before = len(bundle.engine.observed_prompts)
        updates = [update async for update in pool._stream_engine_message(
            bundle=bundle, message=first[0], session_key="telegram:123",
            user_prompt=first[2].text, user_message=first[2],
            turn_ctx=replace(first[1], is_owner=True), memory_scope=joint.SCOPE,
            recorder=None, todo_lifecycle=False,
        )]
        assert updates == []
        assert server.rows == rows_before
        assert len(bundle.engine.observed_prompts) == prompts_before
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == episode_count
        exported = _episodes(pool, [original[4], first[4], latest[4]])
        assert len(exported["episodes"]) == episode_count
        assert server.rows[-1]["id"] == latest[3].metadata["nutrition_append_event_id"]
    finally:
        await client.aclose()
