from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import httpx
from pydantic import ValidationError

from ohmo.evals.nutrition_persistence import (
    Goal,
    Manifest,
    bind_wellness_snapshot,
    export_eval_dialogue,
    grade_manifest,
    read_telegent_wellness,
    validate_dialogue_binding,
    derive_meal_id,
)
from openharness.evals import EvalEpisode, EvalEvent, EvalStore
from ohmo.memory_service.honcho_client import HonchoClient, HonchoError

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
DAY = date(2026, 10, 1)


def gateway_turn_metadata(source, *, reply=None, episode="ep-tea", created=NOW, owner="owner-1"):
    from openharness.channels.bus.events import InboundMessage
    from ohmo.gateway.memory_gate import MemoryScope
    from ohmo.gateway.runtime import _build_conversation_turn_metadata
    from ohmo.gateway.turn_context import build_turn_context

    inbound = InboundMessage(channel="telegram", sender_id=f"{owner}|mutable_name", chat_id="chat-1",
        content="tea update", timestamp=created,
        metadata={"message_id": source, "reply_to_message_id": reply})
    context = build_turn_context(inbound, session_id="session-1")
    _, _, metadata = _build_conversation_turn_metadata(
        turn_ctx=context, message=inbound, scope=MemoryScope(owner, ()))
    metadata.update(role="assistant", decision_trace_episode_id=episode,
                    decision_trace={"episode_id": episode, "annotations": {}})
    return metadata


def goal(*, expected=True, kcal=25):
    metadata = gateway_turn_metadata("src-tea")
    return Goal(case_id="tea", episode_ids=["ep-tea"], owner_id="owner-1", canonical_owner_id="owner-1",
                principal_id="telegram:owner-1", workspace_id="workspace-1", eval_workspace="synthetic-evals",
                peer_id="ohmo", canonical_login="owner",
                session_id="session-1", gateway_session_id="session-1",
                source_message_id="src-tea", meal_date=DAY, meal_timezone="UTC",
                trajectory_started_at=NOW.replace(hour=11), trajectory_as_of=NOW,
                logical_turn_id=metadata["logical_turn_id"], trace_episode_id="ep-tea",
                operation_id=metadata["client_op_id"],
                canonical_meal_id=derive_meal_id(tenant_id="owner-1", source_principal="telegram:owner-1",
                                                 gateway_session_id="session-1", source_message_id="src-tea"),
                expected_consumed=expected,
                expected_kcal=kcal if expected else None, expectation_origin="reviewed_user_dialogue",
                expectation_source="review:tea-01", review_notes="user confirmed milk and no sugar")


def raw_event(event_id="evt-1", *, kcal=25, owner="owner-1", session="session-1", source="src-tea",
              day="2026-10-01", created="2026-10-01T12:00:00+00:00", status="consumed", reply=None):
    nutrition = {"schema_version": 2, "record_type": "meal_observation" if reply is None else "meal_correction",
                 "basis": ["user_report"], "consumption_status": status, "meal_date": day,
                 "energy_kcal_min": kcal, "energy_kcal_max": kcal, "energy_kcal_best": kcal,
                 "changed_fields": [] if reply is None else ["basis", "consumption_status", "meal_date",
                     "energy_kcal_best", "energy_kcal_min", "energy_kcal_max"]}
    metadata = gateway_turn_metadata(source, reply=reply, created=datetime.fromisoformat(created), owner=owner)
    metadata["gateway_session_id"] = session
    metadata["decision_trace"]["annotations"]["nutrition"] = nutrition
    return {"id": event_id, "peer_id": "ohmo", "session_id": session, "workspace_id": "workspace-1", "created_at": created,
            "content": "persisted assistant message", "metadata": metadata}


def per_turn_event(event_id, *, source, episode, turn, kcal, kind="meal_observation", reply=None,
                   status="consumed", created="2026-10-01T12:00:00+00:00"):
    row = raw_event(event_id, kcal=kcal, source=source, reply=reply, status=status, created=created)
    metadata = row["metadata"]
    metadata.update(decision_trace_episode_id=episode)
    metadata["decision_trace"]["episode_id"] = episode
    nutrition = metadata["decision_trace"]["annotations"]["nutrition"]
    nutrition["record_type"] = kind
    if kind == "meal_deletion":
        nutrition.update(energy_kcal_min=None, energy_kcal_max=None, energy_kcal_best=None,
                        consumption_status="not_consumed", changed_fields=[])
    elif kind == "meal_correction" and not nutrition["changed_fields"]:
        nutrition["changed_fields"] = ["consumption_status", "energy_kcal_best",
                                        "energy_kcal_max", "energy_kcal_min"]
    return row


def reviewed_turn_map(*events):
    grouped = {}
    for event in events:
        metadata = event["metadata"]
        episode = metadata["decision_trace_episode_id"]
        grouped.setdefault(episode, []).append({
            "source_message_id": metadata["source_message_id"],
            "logical_turn_id": metadata["logical_turn_id"],
            "operation_id": metadata["client_op_id"],
            "principal_id": metadata["source_principal"],
        })
    return grouped


def canonical_record(*, kcal=25, source="src-tea", day="2026-10-01", latest="evt-1"):
    return {"meal_id": derive_meal_id(tenant_id="owner-1", source_principal="telegram:owner-1",
            gateway_session_id="session-1", source_message_id=source), "revision": 1,
            "status": "active", "latest_event_id": latest, "day": day, "provisional": True,
            "capture_time": NOW.isoformat(), "meal_at": None, "meal_date": day,
            "source_message_id": source, "ingest_source": "telegram", "confirmation_required": None,
            "reply_to_source_message_id": None, "received_at": None, "is_forwarded": False,
            "source_message_at": None, "is_estimate": True, "basis": ["user_report"],
            "consumption_status": "consumed", "energy_kcal_min": kcal, "energy_kcal_max": kcal,
            "energy_kcal_best": kcal, "protein_g": None, "fat_g": None, "carbohydrate_g": None,
            "items": [], "confidence": "medium", "assumptions": [], "warnings": []}


def snapshots(events=None, *, canonical=True, status="complete"):
    events = [raw_event()] if events is None else events
    honcho = {"complete": True, "workspace_id": "workspace-1", "session_id": "session-1",
              "owner_id": "owner-1", "since": "2026-10-01T00:00:00+00:00",
              "until": NOW.isoformat(), "queried_at": NOW.isoformat(), "messages": events}
    record = canonical_record(kcal=25, latest=events[-1]["id"] if events else "evt-1")
    telegent = {"complete": True, "user_id": "owner-1", "login": "owner", "start": "2026-10-01T00:00:00+00:00",
                "end": NOW.isoformat(), "queried_at": NOW.isoformat(), "meals": [record], "unassigned": []}
    return honcho, bind_wellness_snapshot(telegent, goal=goal()) if canonical else {
        "complete": True, "user_id": "owner-1", "login": "owner", "start": "2026-10-01T00:00:00+00:00",
        "end": NOW.isoformat(), "queried_at": NOW.isoformat(), "meal": None,
    }


def grade(honcho, telegent, *, kcal=25, now=NOW):
    return grade_manifest(Manifest(schema_version=1, goals=[goal(kcal=kcal)]), honcho, telegent,
                          now=now, grace_seconds=300)[0]


def test_missing_honcho_row_fails_even_when_episode_completed_and_trace_is_missing_or_proposal_only():
    honcho, telegent = snapshots(events=[], canonical=False)
    honcho["episode_status"] = "completed"
    assert grade(honcho, telegent)["a1"] == "FAIL"
    honcho["messages"] = [{"id": "assistant-proposal", "session_id": "session-1", "peer_id": "ohmo",
                           "workspace_id": "workspace-1", "created_at": NOW.isoformat(),
                           "metadata": {"tenant_id": "owner-1", "role": "assistant"}}]
    assert grade(honcho, telegent)["a1"] == "FAIL"


def rotated_gateway_event(event_id="evt-new-session", *, source="src-new-session", annotated=False):
    row = per_turn_event(event_id, source=source, episode="ep-new-session", turn="turn-new-session",
                         kcal=99)
    row["metadata"]["gateway_session_id"] = "gateway-session-new"
    if not annotated:
        row["metadata"]["decision_trace"]["annotations"] = {}
    return row


def reviewed_context_turn(event_id, *, source, episode, logical_turn, created):
    row = per_turn_event(event_id, source=source, episode=episode, turn=logical_turn, kcal=0,
                         created=created)
    row["metadata"]["decision_trace"]["annotations"] = {}
    return row


@pytest.mark.parametrize("annotated", [False, True], ids=["ordinary", "food"])
def test_rotated_gateway_traffic_is_scoped_but_not_mistaken_for_reviewed_goal(annotated):
    unrelated = rotated_gateway_event(annotated=annotated)

    honcho, telegent = snapshots([unrelated], canonical=False)
    missing = grade(honcho, telegent)
    assert missing["a1"] == "FAIL" and missing["stage"] == "HONCHO_GOAL_MISMATCH"

    valid, canonical = snapshots()
    valid["messages"].append(unrelated)
    assert grade(valid, canonical)["a1"] == "PASS"

    negative = Manifest(schema_version=1, goals=[goal(expected=False, kcal=None)])
    absent, canonical_absence = snapshots([unrelated], canonical=False)
    assert grade_manifest(negative, absent, canonical_absence, now=NOW)[0]["a1"] == "PASS"


def test_relevant_gateway_mismatch_and_shared_query_scope_contamination_stay_inconclusive():
    relevant = raw_event()
    relevant["metadata"]["gateway_session_id"] = "gateway-session-new"
    honcho, telegent = snapshots([relevant])
    assert grade(honcho, telegent)["a1"] == "INCONCLUSIVE"

    for field, value in (("tenant_id", "other-owner"), ("session_id", "other-session"),
                         ("workspace_id", "other-workspace"),
                         ("peer_id", "other-peer"), ("created_at", "2025-01-01T00:00:00+00:00")):
        for annotated in (False, True):
            row = rotated_gateway_event(annotated=annotated)
            (row["metadata"] if field == "tenant_id" else row)[field] = value
            honcho, telegent = snapshots([row], canonical=False)
            assert grade(honcho, telegent)["a1"] == "INCONCLUSIVE", (field, annotated)


@pytest.mark.parametrize("edge", ["nested_trace", "logical_turn"])
@pytest.mark.parametrize("foreign_principal", [False, True], ids=["same-principal", "foreign-principal"])
@pytest.mark.parametrize("expected_consumed", [False, True], ids=["negative", "positive"])
def test_annotated_goal_identity_edges_cannot_hide_contradictory_rows(
        edge, foreign_principal, expected_consumed):
    reviewed = goal(expected=expected_consumed, kcal=25 if expected_consumed else None)
    row = per_turn_event("evt-conflict", source="src-other", episode="ep-other", turn="other", kcal=99)
    metadata = row["metadata"]
    if foreign_principal:
        metadata["source_principal"] = "telegram:foreign-principal"
    if edge == "nested_trace":
        metadata["decision_trace"]["episode_id"] = reviewed.trace_episode_id
    else:
        metadata["logical_turn_id"] = reviewed.logical_turn_id
    honcho, telegent = snapshots() if expected_consumed else snapshots([], canonical=False)
    honcho["messages"].append(row)
    result = grade_manifest(Manifest(schema_version=1, goals=[reviewed]), honcho, telegent, now=NOW)[0]
    assert result["a1"] == "INCONCLUSIVE", result


@pytest.mark.parametrize("edge", ["nested_trace", "logical_turn"])
def test_same_principal_contradictory_goal_identity_stays_ambiguous(edge):
    row = per_turn_event("evt-conflict", source="src-other", episode="ep-other", turn="other", kcal=99)
    if edge == "nested_trace":
        row["metadata"]["decision_trace"]["episode_id"] = "ep-tea"
    else:
        row["metadata"]["logical_turn_id"] = goal().logical_turn_id
    honcho, telegent = snapshots()
    honcho["messages"].append(row)
    result = grade(honcho, telegent)
    assert result["a1"] == "INCONCLUSIVE", result


@pytest.mark.parametrize("kind", ["meal_correction", "meal_deletion"])
def test_unresolved_correction_or_deletion_linked_by_nested_trace_is_fail_closed(kind):
    row = per_turn_event("evt-orphan", source="src-correction", episode="ep-other", turn="other",
                         kcal=99, kind=kind, reply="src-unknown")
    row["metadata"]["reply_to_source_message_id"] = None
    row["metadata"]["source_principal"] = "telegram:foreign-principal"
    row["metadata"]["decision_trace"]["episode_id"] = "ep-tea"
    honcho, telegent = snapshots()
    honcho["messages"].append(row)
    result = grade(honcho, telegent)
    assert result["a1"] == "INCONCLUSIVE" and result["stage"] == "HONCHO_TARGET_UNAVAILABLE", result


def reviewed_multiepisode_fixture():
    context = reviewed_context_turn("evt-balance", source="src-balance", episode="ep-balance",
                                    logical_turn="turn-balance", created="2026-10-01T11:20:00+00:00")
    clarification = reviewed_context_turn("evt-clarify", source="src-clarify", episode="ep-clarify",
                                          logical_turn="turn-clarify", created="2026-10-01T11:40:00+00:00")
    root = raw_event(created="2026-10-01T11:55:00+00:00")
    reviewed = Goal.model_validate({**goal().model_dump(mode="json"),
                                    "episode_ids": ["ep-balance", "ep-clarify", "ep-tea"]})
    provenance = reviewed_turn_map(context, clarification, root)
    sources = {episode: [turn["source_message_id"] for turn in turns]
               for episode, turns in provenance.items()}
    honcho, telegent = snapshots([context, clarification, root])
    telegent["meal"]["capture_time"] = root["created_at"]
    return reviewed, context, clarification, root, honcho, telegent, sources, provenance


def test_valid_unannotated_context_and_clarification_turns_bind_to_their_own_exported_sources():
    reviewed, _, _, _, honcho, telegent, sources, provenance = reviewed_multiepisode_fixture()
    result = grade_manifest(Manifest(schema_version=1, goals=[reviewed]), honcho, telegent, now=NOW,
                            reviewed_turn_sources=sources, reviewed_turn_provenance=provenance)[0]
    assert result["a1"] == "PASS", (result.get("stage"), result.get("reason"))


def test_valid_context_turns_do_not_mask_missing_root_consumed_event():
    reviewed, context, clarification, _, _, _, sources, provenance = reviewed_multiepisode_fixture()
    honcho, telegent = snapshots([], canonical=False)
    honcho["messages"] = [context, clarification]
    result = grade_manifest(Manifest(schema_version=1, goals=[reviewed]), honcho, telegent, now=NOW,
                            reviewed_turn_sources=sources, reviewed_turn_provenance=provenance)[0]
    assert result["a1"] == "FAIL" and result["stage"] == "HONCHO_GOAL_MISMATCH", result


@pytest.mark.parametrize("mutation", ["source", "episode", "logical_turn", "operation", "principal"])
def test_mismatched_context_turn_export_binding_cannot_silently_pass(mutation):
    reviewed, context, clarification, root, honcho, telegent, sources, provenance = reviewed_multiepisode_fixture()
    row = json.loads(json.dumps(context))
    metadata = row["metadata"]
    if mutation == "source":
        metadata["source_message_id"] = "src-unreviewed-context"
    elif mutation == "episode":
        metadata["decision_trace_episode_id"] = "ep-unreviewed-context"
        metadata["decision_trace"]["episode_id"] = "ep-unreviewed-context"
    elif mutation == "logical_turn":
        metadata["logical_turn_id"] = "turn-forged-context"
    elif mutation == "operation":
        metadata["client_op_id"] = "turn-forged-context:assistant"
    else:
        metadata["source_principal"] = "telegram:foreign-principal"
    honcho["messages"] = [row, clarification, root]
    result = grade_manifest(Manifest(schema_version=1, goals=[reviewed]), honcho, telegent, now=NOW,
                            reviewed_turn_sources=sources, reviewed_turn_provenance=provenance)[0]
    assert result["a1"] == "INCONCLUSIVE", (mutation, result)


def test_missing_reviewed_episode_writes_inconclusive_snapshot_cli_report(tmp_path, monkeypatch):
    import sys

    from ohmo.evals.nutrition_persistence import main

    manifest_path = tmp_path / "manifest.json"
    export_path = tmp_path / "dialogue.json"
    honcho_path = tmp_path / "honcho.json"
    telegent_path = tmp_path / "telegent.json"
    report_path = tmp_path / "report.json"
    manifest_path.write_text(Manifest(schema_version=1, goals=[goal()]).model_dump_json())
    export_path.write_text(json.dumps({"privacy": "private", "episodes": []}))
    honcho, telegent = snapshots()
    honcho_path.write_text(json.dumps(honcho))
    telegent_path.write_text(json.dumps(telegent))

    monkeypatch.setattr(sys, "argv", [
        "nutrition-persistence", str(manifest_path), str(report_path),
        "--dialogue-export", str(export_path), "--honcho-snapshot", str(honcho_path),
        "--telegent-snapshot", str(telegent_path),
    ])
    main()

    report = json.loads(report_path.read_text())
    assert report["cases"][0]["a1"] == "INCONCLUSIVE"
    assert report["cases"][0]["stage"] == "DIALOGUE_BINDING_FAILED"
    assert report["dialogue_binding"]["tea"]["complete"] is False


@pytest.mark.parametrize("actual,verdict", [(25, "PASS"), (22.5, "PASS"), (27.5, "PASS"), (22.49, "FAIL"), (27.51, "FAIL")])
def test_same_event_projection_uses_frozen_numeric_boundary(actual, verdict):
    honcho, telegent = snapshots()
    nutrition = honcho["messages"][0]["metadata"]["decision_trace"]["annotations"]["nutrition"]
    nutrition.update(energy_kcal_min=0, energy_kcal_max=100, energy_kcal_best=actual)
    meal = telegent["meal"]
    meal["energy_kcal_best"] = actual
    assert grade(honcho, telegent)["a1"] == verdict


def test_deployed_canonical_shape_reports_missing_contributor_id_list_without_inventing_ids():
    honcho, telegent = snapshots()
    result = grade(honcho, telegent)
    assert result["a1"] == "PASS"
    assert result["canonical_contributing_event_ids"] is None
    assert "event_ids" in result["evidence_limitations"][0]


@pytest.mark.parametrize("field,value", [
    ("user_id", "other-owner"), ("source_message_id", "other-source"),
    ("latest_event_id", "forged"), ("day", "2026-09-30"), ("energy_kcal_best", 250),
])
def test_canonical_wrong_identity_date_id_or_kcal_never_pass(field, value):
    honcho, telegent = snapshots()
    telegent["meal"][field] = value
    if field == "day":
        telegent["meal"]["meal_date"] = value
    assert grade(honcho, telegent)["a1"] == "FAIL"


def test_false_saved_text_and_unrelated_equal_kcal_meal_cannot_pass():
    honcho, telegent = snapshots(events=[])
    honcho["assistant_says_saved"] = True
    assert grade(honcho, telegent)["a1"] == "FAIL"
    honcho, telegent = snapshots()
    telegent["meal"]["source_message_id"] = "unrelated"
    assert grade(honcho, telegent)["a1"] == "FAIL"


@pytest.mark.parametrize("side", ["honcho", "telegent"])
def test_unavailable_or_incomplete_reads_are_inconclusive(side):
    honcho, telegent = snapshots()
    (honcho if side == "honcho" else telegent)["complete"] = False
    assert grade(honcho, telegent)["a1"] == "INCONCLUSIVE"


def test_saved_projection_pending_then_overdue_failure_then_success():
    honcho, telegent = snapshots(canonical=False)
    result = grade(honcho, telegent)
    assert result["a1"] == "PENDING" and result["stage"] == "HONCHO_SAVED_TELEGENT_PENDING"
    old = datetime(2026, 10, 1, 11, 50, tzinfo=timezone.utc)
    honcho["messages"][0]["created_at"] = old.isoformat()
    assert grade(honcho, telegent)["a1"] == "FAIL"
    honcho["messages"][0]["created_at"] = NOW.isoformat()
    honcho, telegent = snapshots()
    assert grade(honcho, telegent)["a1"] == "PASS"
    wrong_saved, absent = snapshots([raw_event(kcal=250)], canonical=False)
    wrong = grade(wrong_saved, absent)
    assert wrong["a1"] == "FAIL" and wrong["stage"] == "HONCHO_GOAL_MISMATCH"


def test_prior_revision_pending_requires_prior_values_to_match_effective_history():
    root = per_turn_event("evt-root", source="src-tea", episode="ep-tea", turn="turn-root", kcal=100,
                          created="2026-10-01T11:58:00+00:00")
    correction = per_turn_event("evt-correction", source="src-correction", episode="ep-correction",
                                turn="turn-correction", kcal=25, kind="meal_correction", reply="src-tea",
                                created="2026-10-01T11:59:00+00:00")
    honcho, telegent = snapshots([root, correction])
    reviewed = Goal.model_validate({**goal(kcal=25).model_dump(mode="json"),
        "episode_ids": ["ep-tea", "ep-correction"]})
    telegent["meal"].update(latest_event_id="evt-root", energy_kcal_best=100,
                            capture_time="2026-10-01T11:58:00+00:00")
    sources = {"ep-tea": ["src-tea"], "ep-correction": ["src-correction"]}
    provenance = reviewed_turn_map(root, correction)
    assert grade_manifest(Manifest(schema_version=1, goals=[reviewed]), honcho, telegent,
        now=NOW, reviewed_turn_sources=sources, reviewed_turn_provenance=provenance)[0]["a1"] == "PENDING"
    telegent["meal"]["energy_kcal_best"] = 999
    assert grade_manifest(Manifest(schema_version=1, goals=[reviewed]), honcho, telegent,
        now=NOW, reviewed_turn_sources=sources, reviewed_turn_provenance=provenance)[0]["a1"] == "FAIL"


def test_real_distinct_turn_correction_binds_per_turn_and_nested_episode():
    root = per_turn_event("evt-root", source="src-tea", episode="ep-tea", turn="turn-root", kcal=100,
                          created="2026-10-01T11:58:00+00:00")
    correction = per_turn_event("evt-correction", source="src-correction", episode="ep-correction",
                                turn="turn-correction", kcal=25, kind="meal_correction", reply="src-tea",
                                created="2026-10-01T11:59:00+00:00")
    honcho, telegent = snapshots([root, correction])
    revised_goal = Goal.model_validate({**goal().model_dump(mode="json"),
        "episode_ids": ["ep-tea", "ep-correction"],
        "logical_turn_id": root["metadata"]["logical_turn_id"],
        "trace_episode_id": "ep-tea", "operation_id": root["metadata"]["client_op_id"]})
    telegent["meal"].update(latest_event_id="evt-correction", energy_kcal_best=25)
    telegent["meal"]["capture_time"] = "2026-10-01T11:59:00+00:00"
    sources = {"ep-tea": ["src-tea"], "ep-correction": ["src-correction"]}
    provenance = reviewed_turn_map(root, correction)
    result = grade_manifest(Manifest(schema_version=1, goals=[revised_goal]), honcho, telegent, now=NOW,
                            reviewed_turn_sources=sources, reviewed_turn_provenance=provenance)[0]
    assert result["a1"] == "PASS", (result.get("stage"), result.get("reason"))
    assert grade_manifest(Manifest(schema_version=1, goals=[revised_goal]), honcho, telegent, now=NOW,
                          reviewed_turn_sources=sources)[0]["a1"] == "INCONCLUSIVE"
    wrong_sources = {**sources, "ep-correction": ["unreviewed-source"]}
    assert grade_manifest(Manifest(schema_version=1, goals=[revised_goal]), honcho, telegent, now=NOW,
        reviewed_turn_sources=wrong_sources, reviewed_turn_provenance=provenance)[0]["a1"] == "INCONCLUSIVE"
    forged = json.loads(json.dumps(honcho))
    forged["messages"][1]["metadata"]["logical_turn_id"] = "forged-turn"
    forged["messages"][1]["metadata"]["client_op_id"] = "forged-turn:assistant"
    assert grade_manifest(Manifest(schema_version=1, goals=[revised_goal]), forged, telegent, now=NOW,
        reviewed_turn_sources=sources, reviewed_turn_provenance=provenance)[0]["a1"] == "INCONCLUSIVE"
    correction["metadata"]["decision_trace"].pop("episode_id")
    assert grade_manifest(Manifest(schema_version=1, goals=[revised_goal]), honcho, telegent,
                          now=NOW, reviewed_turn_sources=sources)[0]["a1"] == "INCONCLUSIVE"


def test_unannotated_raw_rows_must_stay_inside_scope_before_absence_can_pass():
    negative = Manifest(schema_version=1, goals=[goal(expected=False, kcal=None)])
    honcho, telegent = snapshots(events=[], canonical=False)
    row = {"id": "unannotated", "peer_id": "wrong-peer", "session_id": "session-1",
           "workspace_id": "wrong-workspace", "created_at": "2025-01-01T00:00:00+00:00",
           "metadata": {"tenant_id": "owner-1", "role": "assistant", "gateway_session_id": "session-1",
                        "source_principal": "telegram:owner-1", "logical_turn_id": "turn-x",
                        "decision_trace_episode_id": "ep-tea"}}
    honcho["messages"] = [row]
    assert grade_manifest(negative, honcho, telegent, now=NOW)[0]["a1"] == "INCONCLUSIVE"


@pytest.mark.parametrize("field", ["source_message_id", "day", "meal_at", "meal_date", "ingest_source",
                                    "confirmation_required", "reply_to_source_message_id", "received_at",
                                    "source_message_at"])
def test_required_nullable_canonical_keys_must_be_present(field):
    honcho, telegent = snapshots()
    telegent["meal"].pop(field)
    assert grade(honcho, telegent)["a1"] == "INCONCLUSIVE"


def test_canonical_bounds_and_capture_provenance_are_consistent():
    honcho, telegent = snapshots()
    telegent["start"], telegent["end"] = "2026-10-02T00:00:00+00:00", "2026-10-01T12:00:00+00:00"
    assert grade(honcho, telegent)["a1"] == "INCONCLUSIVE"
    honcho, telegent = snapshots()
    telegent["meal"]["capture_time"] = "2027-01-01T00:00:00+00:00"
    assert grade(honcho, telegent)["a1"] == "INCONCLUSIVE"
    honcho, telegent = snapshots()
    telegent["meal"]["day"] = "2026-10-01"
    telegent["meal"]["meal_date"] = "2026-09-30"
    assert grade(honcho, telegent)["a1"] == "INCONCLUSIVE"


def test_unresolved_correction_target_and_post_deletion_correction_are_inconclusive_or_absent():
    target = per_turn_event("evt-target", source="src-tea", episode="ep-tea", turn="turn-1", kcal=25,
                            created="2026-10-01T11:58:00+00:00")
    deletion = per_turn_event("evt-delete", source="src-delete", episode="ep-delete", turn="turn-delete",
                              kcal=0, kind="meal_deletion", reply="src-tea", status="not_consumed",
                              created="2026-10-01T11:59:00+00:00")
    late = per_turn_event("evt-late", source="src-late", episode="ep-late", turn="turn-late",
                          kcal=25, kind="meal_correction", reply="src-tea",
                          created="2026-10-01T12:00:00+00:00")
    negative_goal = Goal.model_validate({**goal(expected=False, kcal=None).model_dump(mode="json"),
        "episode_ids": ["ep-tea", "ep-delete", "ep-late"]})
    negative = Manifest(schema_version=1, goals=[negative_goal])
    honcho, telegent = snapshots([target, deletion, late], canonical=False)
    sources = {"ep-tea": ["src-tea"], "ep-delete": ["src-delete"], "ep-late": ["src-late"]}
    result = grade_manifest(negative, honcho, telegent, now=NOW,
        reviewed_turn_sources=sources, reviewed_turn_provenance=reviewed_turn_map(target, deletion, late))[0]
    assert result["a1"] == "PASS", (result.get("stage"), result.get("reason"))
    assert target["metadata"]["decision_trace"]["annotations"]["nutrition"]["energy_kcal_best"] == 25
    orphan = per_turn_event("evt-orphan", source="src-orphan", episode="ep-tea", turn="turn-orphan",
                            kcal=25, kind="meal_correction", reply=None)
    honcho, telegent = snapshots([orphan], canonical=False)
    assert grade_manifest(negative, honcho, telegent, now=NOW)[0]["a1"] == "INCONCLUSIVE"


def test_wrong_saved_date_and_unrelated_canonical_row_do_not_extend_grace():
    wrong_date = per_turn_event("evt-wrong-day", source="src-tea", episode="ep-tea", turn="turn-1",
                                kcal=25, created=NOW.isoformat())
    wrong_date["metadata"]["decision_trace"]["annotations"]["nutrition"]["meal_date"] = "2026-09-30"
    honcho, telegent = snapshots([wrong_date], canonical=False)
    assert grade(honcho, telegent)["a1"] == "FAIL"
    honcho, telegent = snapshots(canonical=False)
    honcho["messages"][0]["created_at"] = "2026-10-01T11:50:00+00:00"
    telegent["other_meals"] = [canonical_record(source="some-other-source")]
    assert grade(honcho, telegent)["a1"] == "FAIL"


def test_correction_uses_latest_same_source_event_and_retraction_counts_zero():
    first = per_turn_event("evt-1", source="src-tea", episode="ep-tea", turn="root", kcal=25,
                           created="2026-10-01T11:58:00+00:00")
    correction = per_turn_event("evt-2", source="src-correction", episode="ep-correction",
                                turn="correction", kcal=25, kind="meal_correction",
                                status="not_consumed", reply="src-tea", created="2026-10-01T11:59:00+00:00")
    honcho, telegent = snapshots([first, correction])
    telegent["meal"]["latest_event_id"] = "evt-2"
    telegent["meal"].pop("event_ids", None)
    telegent["meals"] = []  # Telegent omits retracted/not_consumed rows.
    telegent["meal"] = None
    telegent.pop("other_meals", None)
    negative_goal = Goal.model_validate({**goal(expected=False, kcal=None).model_dump(mode="json"),
        "episode_ids": ["ep-tea", "ep-correction"]})
    negative = Manifest(schema_version=1, goals=[negative_goal])
    sources = {"ep-tea": ["src-tea"], "ep-correction": ["src-correction"]}
    operations = reviewed_turn_map(first, correction)
    assert grade_manifest(negative, honcho, telegent, now=NOW, reviewed_turn_sources=sources,
                          reviewed_turn_provenance=operations)[0]["a1"] == "PASS"
    telegent["meal"] = canonical_record(latest="evt-1")
    telegent["meal"].update(user_id="owner-1", energy_kcal_best=25)
    telegent["meal"]["capture_time"] = "2026-10-01T11:58:00+00:00"
    assert grade_manifest(negative, honcho, telegent, now=NOW, reviewed_turn_sources=sources,
                          reviewed_turn_provenance=operations)[0]["a1"] == "PENDING"


def test_negative_goal_complete_absence_and_unexpected_consumed_row():
    negative = Manifest(schema_version=1, goals=[goal(expected=False, kcal=None)])
    honcho, telegent = snapshots(events=[], canonical=False)
    assert grade_manifest(negative, honcho, telegent, now=NOW)[0]["a1"] == "PASS"
    honcho, telegent = snapshots()
    assert grade_manifest(negative, honcho, telegent, now=NOW)[0]["a1"] == "FAIL"
    telegent["complete"] = False
    assert grade_manifest(negative, honcho, telegent, now=NOW)[0]["a1"] == "INCONCLUSIVE"


def test_manifest_expectation_independent_of_missing_final_annotation():
    honcho, telegent = snapshots(events=[])
    assert grade(honcho, telegent)["a1"] == "FAIL"


def test_eval_export_reads_read_only_sqlite_and_exports_gateway_final(tmp_path: Path):
    root = tmp_path / "evals"
    store = EvalStore(root)
    store.append_episode(EvalEpisode(episode_id="ep-clarify", session_id="session-1", source="gateway",
                                     created_at=NOW.replace(hour=10), metadata={"workspace": "synthetic-evals",
                                     "inbound": {"channel": "telegram", "sender_id": "owner-1"}}))
    store.append_event(EvalEvent(episode_id="ep-clarify", kind="inbound_message", payload={
        "user_text": "I drank tea.", "channel": "telegram", "sender_id": "owner-1",
        "chat_id": "chat-1", "timestamp": NOW.replace(hour=10).isoformat(),
        "metadata": {"message_id": "tea-clarification"},
    }))
    store.append_event(EvalEvent(episode_id="ep-clarify", kind="gateway_final", payload={"text": "Milk or sugar?"}))
    store.append_event(EvalEvent(episode_id="ep-clarify", kind="episode_finished", payload={"status": "completed"}))
    store.append_episode(EvalEpisode(episode_id="ep-tea", session_id="session-1", source="gateway",
                                     created_at=NOW, metadata={"workspace": "synthetic-evals",
                                     "inbound": {"channel": "telegram", "sender_id": "owner-1|mutable_name"}}))
    store.append_event(EvalEvent(episode_id="ep-tea", kind="inbound_message", payload={
        "user_text": "I drank tea.", "channel": "telegram", "sender_id": "owner-1|mutable_name",
        "chat_id": "chat-1", "timestamp": NOW.isoformat(), "metadata": {"message_id": "src-tea"},
    }))
    store.append_event(EvalEvent(episode_id="ep-tea", kind="assistant_update", payload={"text": "Was there sugar?"}))
    store.append_event(EvalEvent(episode_id="ep-tea", kind="gateway_final", payload={"text": "I could not confirm the save."}))
    store.append_event(EvalEvent(episode_id="ep-tea", kind="episode_finished", payload={"status": "completed"}))
    before = (root / "evals.sqlite").read_bytes()
    exported = export_eval_dialogue(root, episode_ids=["ep-tea", "ep-clarify"])
    assert [item["episode"]["episode_id"] for item in exported["episodes"]] == ["ep-clarify", "ep-tea"]
    assert exported["episodes"][1]["dialogue"] == [
        {"role": "user", "text": "I drank tea."},
        {"role": "assistant", "text": "Was there sugar?"},
        {"role": "assistant", "text": "I could not confirm the save."},
    ]
    assert exported["episodes"][1]["principal_id"] == "telegram:owner-1"
    assert exported["episodes"][1]["turn_provenance"][0]["logical_turn_id"] == goal().logical_turn_id
    assert exported["episodes"][1]["turn_provenance"][0]["operation_id"] == goal().operation_id
    assert exported["episodes"][0]["dialogue_complete"] is True
    assert (root / "evals.sqlite").read_bytes() == before
    multi_goal = Goal.model_validate({**goal().model_dump(mode="json"),
                                     "episode_ids": ["ep-clarify", "ep-tea"]})
    assert validate_dialogue_binding(Manifest(schema_version=1, goals=[multi_goal]), exported)["tea"]["complete"]
    forged_goal = Goal.model_validate({**multi_goal.model_dump(mode="json"),
        "logical_turn_id": "forged-logical", "operation_id": "forged-logical:assistant"})
    assert not validate_dialogue_binding(Manifest(schema_version=1, goals=[forged_goal]), exported)["tea"]["complete"]
    bounded = export_eval_dialogue(root, session_id="session-1", principal_id="telegram:owner-1",
                                   since="2026-10-01T00:00:00+00:00", until="2026-10-01T23:00:00+00:00")
    assert len(bounded["episodes"]) == 2
    with pytest.raises(ValueError):
        export_eval_dialogue(root, session_id="session-1", principal_id="telegram:wrong-owner",
                             since="2026-10-01T00:00:00+00:00", until="2026-10-01T23:00:00+00:00")
    partial = json.loads(json.dumps(exported))
    partial["episodes"][1]["dialogue_complete"] = False
    assert not validate_dialogue_binding(Manifest(schema_version=1, goals=[multi_goal]), partial)["tea"]["complete"]
    partial["episodes"][1]["dialogue_complete"] = True
    partial["episodes"][1]["source_message_ids"] = ["wrong-source"]
    assert not validate_dialogue_binding(Manifest(schema_version=1, goals=[multi_goal]), partial)["tea"]["complete"]
    uri = (root / "evals.sqlite").as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 2


def test_eval_export_requires_terminal_finish_event(tmp_path: Path):
    root = tmp_path / "evals"
    store = EvalStore(root)
    store.append_episode(EvalEpisode(episode_id="ep-unfinished", session_id="session-1", source="gateway",
                                     created_at=NOW, metadata={"workspace": "synthetic-evals",
                                     "inbound": {"channel": "telegram", "sender_id": "424242|mutable_name"}}))
    store.append_event(EvalEvent(episode_id="ep-unfinished", kind="inbound_message", payload={
        "user_text": "I drank tea.", "metadata": {"message_id": 401},
    }))
    store.append_event(EvalEvent(episode_id="ep-unfinished", kind="gateway_final", payload={"text": "Save failed."}))
    unfinished = export_eval_dialogue(root, episode_ids=["ep-unfinished"])
    assert unfinished["episodes"][0]["principal_id"] == "telegram:424242"
    assert unfinished["episodes"][0]["dialogue_complete"] is False
    store.append_event(EvalEvent(episode_id="ep-unfinished", kind="episode_finished", payload={"status": "completed"}))
    finished = export_eval_dialogue(root, episode_ids=["ep-unfinished"])
    assert finished["episodes"][0]["dialogue_complete"] is True
    store.append_event(EvalEvent(episode_id="ep-unfinished", kind="assistant_update", payload={"text": "late update"}))
    malformed = export_eval_dialogue(root, episode_ids=["ep-unfinished"])
    assert malformed["episodes"][0]["dialogue_complete"] is False


def test_eval_export_rejects_partial_episode_selection_and_bad_source(tmp_path: Path):
    with pytest.raises(ValueError):
        export_eval_dialogue(tmp_path, episode_ids=["missing"])
    snapshot = {"complete": True, "user_id": "wrong", "meals": [], "unassigned": []}
    assert bind_wellness_snapshot(snapshot, goal=goal())["complete"] is False


def test_bounded_episode_export_treats_equivalent_offset_windows_identically(tmp_path: Path):
    root = tmp_path / "evals"
    store = EvalStore(root)
    stamps = ["2026-10-01T16:57:59.999999Z", "2026-10-01T16:58:00Z",
              "2026-10-01T17:02:30.123456Z", "2026-10-01T17:06:00Z",
              "2026-10-01T17:06:00.000001Z"]
    for index, stamp in enumerate(stamps):
        store.append_episode(EvalEpisode(episode_id=f"ep-{index}", session_id="session-1", source="gateway",
            created_at=datetime.fromisoformat(stamp.replace("Z", "+00:00")), metadata={"workspace": "synthetic-evals",
            "inbound": {"channel": "telegram", "sender_id": "424242|mutable", "chat_id": "chat-1"}}))
    utc = export_eval_dialogue(root, session_id="session-1", principal_id="telegram:424242",
        since="2026-10-01T16:58:00.000000+00:00", until="2026-10-01T17:06:00.000000Z")
    moscow = export_eval_dialogue(root, session_id="session-1", principal_id="telegram:424242",
        since="2026-10-01T19:58:00+03:00", until="2026-10-01T20:06:00+03:00")
    assert [item["episode"]["episode_id"] for item in utc["episodes"]] == ["ep-1", "ep-2", "ep-3"]
    assert [item["episode"]["episode_id"] for item in moscow["episodes"]] == ["ep-1", "ep-2", "ep-3"]


def test_bounded_episode_export_includes_first_second_fractional_timestamps(tmp_path: Path):
    root = tmp_path / "evals"
    store = EvalStore(root)
    stamps = ["2026-10-01T16:57:59.999999Z", "2026-10-01T16:58:00Z",
              "2026-10-01T16:58:00.000001Z", "2026-10-01T16:58:00.500000Z",
              "2026-10-01T16:58:01Z", "2026-10-01T16:58:01.000001Z"]
    for index, stamp in enumerate(stamps):
        store.append_episode(EvalEpisode(episode_id=f"fraction-{index}", session_id="session-1",
            source="gateway", created_at=datetime.fromisoformat(stamp.replace("Z", "+00:00")),
            metadata={"workspace": "synthetic-evals", "inbound": {"channel": "telegram",
            "sender_id": "424242|mutable", "chat_id": "chat-1"}}))

    utc = export_eval_dialogue(root, session_id="session-1", principal_id="telegram:424242",
        since="2026-10-01T16:58:00.000001Z", until="2026-10-01T16:58:01Z")
    equivalent_offset = export_eval_dialogue(root, session_id="session-1", principal_id="telegram:424242",
        since="2026-10-01T19:58:00.000001+03:00", until="2026-10-01T19:58:01+03:00")
    expected = ["fraction-2", "fraction-3", "fraction-4"]
    assert [item["episode"]["episode_id"] for item in utc["episodes"]] == expected
    assert [item["episode"]["episode_id"] for item in equivalent_offset["episodes"]] == expected


def test_export_orders_fractional_and_whole_second_episodes_and_events_by_instant(tmp_path: Path):
    root = tmp_path / "evals"
    store = EvalStore(root)
    # Append correction first so explicit selection and SQL text ordering both exercise chronology.
    for episode_id, stamp, inbound_id in (
        ("ep-correction", "2026-10-01T11:50:00.000001Z", "402"),
        ("ep-root", "2026-10-01T11:50:00Z", "401"),
    ):
        store.append_episode(EvalEpisode(episode_id=episode_id, session_id="session-1", source="gateway",
            created_at=datetime.fromisoformat(stamp.replace("Z", "+00:00")), metadata={"workspace": "synthetic-evals",
            "inbound": {"channel": "telegram", "sender_id": "424242|mutable"}}))
        if episode_id == "ep-root":
            entries = [
                ("inbound_message", "2026-10-01T11:50:00Z", {"user_text": "I drank tea", "channel": "telegram",
                    "sender_id": "424242|mutable", "chat_id": "chat-1", "timestamp": stamp,
                    "metadata": {"message_id": inbound_id}}),
                ("assistant_update", "2026-10-01T11:50:00.000001Z", {"text": "Checking save"}),
                ("gateway_final", "2026-10-01T11:50:00.000002Z", {"text": "Save could not be confirmed"}),
                ("episode_finished", "2026-10-01T11:50:00.000003Z", {"status": "completed"}),
            ]
        else:
            entries = [
                ("inbound_message", stamp, {"user_text": "Correction", "channel": "telegram",
                    "sender_id": "424242|mutable", "chat_id": "chat-1", "timestamp": stamp,
                    "metadata": {"message_id": inbound_id}}),
                ("gateway_final", "2026-10-01T11:50:00.000002Z", {"text": "Corrected"}),
                ("episode_finished", "2026-10-01T11:50:00.000003Z", {"status": "completed"}),
            ]
        for kind, event_stamp, payload in entries:
            store.append_event(EvalEvent(episode_id=episode_id, kind=kind,
                timestamp=datetime.fromisoformat(event_stamp.replace("Z", "+00:00")), payload=payload))

    explicit = export_eval_dialogue(root, episode_ids=["ep-correction", "ep-root"])
    ranged = export_eval_dialogue(root, session_id="session-1", principal_id="telegram:424242",
        since="2026-10-01T14:50:00+03:00", until="2026-10-01T14:50:01+03:00")
    for exported in (explicit, ranged):
        episodes = exported["episodes"]
        assert [item["episode"]["episode_id"] for item in episodes] == ["ep-root", "ep-correction"]
        assert episodes[0]["dialogue"] == [
            {"role": "user", "text": "I drank tea"},
            {"role": "assistant", "text": "Checking save"},
            {"role": "assistant", "text": "Save could not be confirmed"},
        ]
        assert episodes[0]["dialogue_complete"] is True


@pytest.mark.parametrize("kind,payload", [
    ("inbound_message", {"user_text": None, "metadata": {"message_id": "401"}}),
    ("gateway_final", {"text": ""}),
    ("gateway_error", {"text": None}),
    ("assistant_update", {"text": None}),
])
def test_export_marks_malformed_public_turn_payload_incomplete(tmp_path: Path, kind, payload):
    root = tmp_path / "evals"
    store = EvalStore(root)
    store.append_episode(EvalEpisode(episode_id="ep-malformed", session_id="session-1", source="gateway",
        created_at=NOW, metadata={"workspace": "synthetic-evals", "inbound": {"channel": "telegram",
        "sender_id": "424242", "chat_id": "chat-1"}}))
    store.append_event(EvalEvent(episode_id="ep-malformed", kind="inbound_message", timestamp=NOW,
        payload={"user_text": "I drank tea", "channel": "telegram", "sender_id": "424242",
                 "chat_id": "chat-1", "timestamp": NOW.isoformat(), "metadata": {"message_id": "401"}}))
    store.append_event(EvalEvent(episode_id="ep-malformed", kind=kind,
        timestamp=NOW.replace(microsecond=1), payload=payload))
    if kind not in {"gateway_final", "gateway_error"}:
        store.append_event(EvalEvent(episode_id="ep-malformed", kind="gateway_final",
            timestamp=NOW.replace(microsecond=2), payload={"text": "Save could not be confirmed"}))
    store.append_event(EvalEvent(episode_id="ep-malformed", kind="episode_finished",
        timestamp=NOW.replace(microsecond=3), payload={"status": "completed"}))
    exported = export_eval_dialogue(root, episode_ids=["ep-malformed"])["episodes"][0]
    assert exported["dialogue_complete"] is False
    local_goal = Goal.model_validate({**goal().model_dump(mode="json"), "episode_ids": ["ep-malformed"],
                                     "trace_episode_id": "ep-malformed"})
    assert not validate_dialogue_binding(Manifest(schema_version=1, goals=[local_goal]),
        {"privacy": "private", "episodes": [exported]})["tea"]["complete"]


def test_export_retains_gateway_camera_scalar_provenance_from_actual_builder(tmp_path: Path):
    from types import SimpleNamespace
    from openharness.channels.bus.events import InboundMessage
    from ohmo.evals.recorder import GatewayEvalRecorder
    from ohmo.gateway.camera import CAMERA_AUTHORITY
    from ohmo.gateway.memory_gate import MemoryScope
    from ohmo.gateway.runtime import _build_conversation_turn_metadata
    from ohmo.gateway.turn_context import build_turn_context

    root = tmp_path / "evals"
    message = InboundMessage(channel="telegram", sender_id="424242", chat_id="chat-1", content="I drank tea",
        timestamp=NOW, metadata={"message_id": 401, "_camera_authority": CAMERA_AUTHORITY,
                                 "_camera_turn_id": "camera-retained-401"})
    turn_ctx = build_turn_context(message, session_id="session-1")
    logical, _, assistant = _build_conversation_turn_metadata(turn_ctx=turn_ctx, message=message,
        scope=MemoryScope("owner-1", ()))
    trusted = {"source_message_id": assistant["source_message_id"], "principal_id": assistant["source_principal"],
        "logical_turn_id": logical, "operation_id": assistant["client_op_id"]}
    recorder = GatewayEvalRecorder.start(workspace=tmp_path, bundle=SimpleNamespace(session_id="session-1", cwd=str(tmp_path)),
        message=message, session_key="telegram:424242", user_text=message.content or "", trusted_turn_provenance=trusted)
    recorder.record_gateway_final(text="I could not confirm the save")
    recorder.finish(status="completed")
    exported = export_eval_dialogue(root, episode_ids=[recorder.episode_id])["episodes"][0]
    provenance = exported["turn_provenance"][0]
    assert provenance["logical_turn_id"] == "camera-retained-401"
    assert provenance["operation_id"] == "camera-retained-401:assistant"
    assert provenance["episode_id"] == recorder.episode_id
    camera_goal = Goal.model_validate({**goal().model_dump(mode="json"), "episode_ids": [recorder.episode_id],
        "trace_episode_id": recorder.episode_id, "source_message_id": "401", "logical_turn_id": logical,
        "operation_id": assistant["client_op_id"], "principal_id": assistant["source_principal"],
        "eval_workspace": str(tmp_path.resolve()),
        "canonical_meal_id": derive_meal_id(tenant_id="owner-1", source_principal="telegram:424242",
            gateway_session_id="session-1", source_message_id="401")})
    binding = validate_dialogue_binding(Manifest(schema_version=1, goals=[camera_goal]),
        {"privacy": "private", "episodes": [exported]})["tea"]
    assert binding["complete"] is False
    assert "initial Camera context" in binding["reason"]


def test_export_does_not_trust_camera_turn_fields_from_inbound_json(tmp_path: Path):
    root = tmp_path / "evals"
    store = EvalStore(root)
    store.append_episode(EvalEpisode(episode_id="ep-forged-camera", session_id="session-1", source="gateway",
        created_at=NOW, metadata={"workspace": "synthetic-evals", "inbound": {"channel": "telegram",
        "sender_id": "424242", "chat_id": "chat-1", "_camera_turn_id": "camera-forged",
        "_camera_authority": "object at 0x123"}}))
    store.append_event(EvalEvent(episode_id="ep-forged-camera", kind="inbound_message", timestamp=NOW,
        payload={"user_text": "I drank tea", "channel": "telegram", "sender_id": "424242", "chat_id": "chat-1",
        "timestamp": NOW.isoformat(), "metadata": {"message_id": 401, "_camera_turn_id": "camera-forged",
        "_camera_authority": "object at 0x123"}}))
    store.append_event(EvalEvent(episode_id="ep-forged-camera", kind="gateway_final",
        timestamp=NOW.replace(microsecond=1), payload={"text": "Saved"}))
    store.append_event(EvalEvent(episode_id="ep-forged-camera", kind="episode_finished",
        timestamp=NOW.replace(microsecond=2), payload={"status": "completed"}))
    exported = export_eval_dialogue(root, episode_ids=["ep-forged-camera"])["episodes"][0]
    local_goal = Goal.model_validate({**goal().model_dump(mode="json"), "episode_ids": ["ep-forged-camera"],
                                     "trace_episode_id": "ep-forged-camera"})
    assert not validate_dialogue_binding(Manifest(schema_version=1, goals=[local_goal]),
        {"privacy": "private", "episodes": [exported]})["tea"]["complete"]


@pytest.mark.asyncio
async def test_honcho_raw_read_traverses_pages_and_rejects_401():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("messages/list"):
            page = int(request.url.params["page"])
            item = raw_event(f"evt-{page}")
            item["workspace_id"] = "workspace-1"
            return httpx.Response(200, json={"items": [item], "page": page, "size": 1,
                                             "pages": 2, "total": 2})
        return httpx.Response(404)

    client = HonchoClient("https://honcho.test", "synthetic-secret", "workspace-1",
                          transport=httpx.MockTransport(handler))
    messages = await client.list_messages_in_window(
        "session-1", expected_peer_id=None, since=NOW.replace(hour=0),
        until=NOW.replace(hour=23), page_size=1, max_pages=2,
    )
    await client.aclose()
    assert [item["id"] for item in messages] == ["evt-1", "evt-2"]
    assert [request.url.params["page"] for request in requests] == ["1", "2"]

    denied = HonchoClient("https://honcho.test", "synthetic-secret", "workspace-1",
                          transport=httpx.MockTransport(lambda request: httpx.Response(401)))
    with pytest.raises(HonchoError):
        await denied.list_messages_in_window(
            "session-1", expected_peer_id=None, since=NOW.replace(hour=0),
            until=NOW.replace(hour=23), page_size=1, max_pages=1,
        )
    await denied.aclose()


@pytest.mark.asyncio
async def test_telegent_mcp_reader_parses_fixture_and_hides_transport_errors(monkeypatch):
    import sys
    import types

    captured = {}
    response = {"user_id": "owner-1", "login": "synthetic", "nutrition_status": "complete",
                "interval": {"start": NOW.replace(hour=0).isoformat(),
                             "end": NOW.replace(hour=23).isoformat()},
                "nutrition_records": [{"latest_event_id": "evt-1"}],
                "nutrition_unassigned_records": []}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def initialize(self):
            return None

        async def call_tool(self, name, arguments):
            if captured.get("fail"):
                raise RuntimeError("fixture-token transport detail")
            captured["call"] = (name, arguments)
            return types.SimpleNamespace(isError=False, content=[
                types.SimpleNamespace(text=json.dumps(response))
            ], structuredContent=None)

    class FakeTransport:
        async def __aenter__(self):
            return None, None, None

        async def __aexit__(self, *args):
            return None

    def fake_transport(url, *, headers):
        captured["url"] = url
        captured["headers"] = headers
        return FakeTransport()

    mcp = types.ModuleType("mcp")
    mcp.ClientSession = lambda read, write: FakeSession()
    client = types.ModuleType("mcp.client")
    streamable = types.ModuleType("mcp.client.streamable_http")
    streamable.streamablehttp_client = fake_transport
    monkeypatch.setitem(sys.modules, "mcp", mcp)
    monkeypatch.setitem(sys.modules, "mcp.client", client)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", streamable)
    result = await read_telegent_wellness(mcp_url="https://fixture.invalid/mcp", token="fixture-token",
                                          owner_login="synthetic", start=NOW.replace(hour=0),
                                          end=NOW.replace(hour=23))
    assert result["complete"] is True
    assert result["meals"][0]["latest_event_id"] == "evt-1"
    assert captured["call"][0] == "get_wellness_data"
    assert captured["call"][1]["params"]["login"] == "synthetic"
    assert captured["headers"] == {"Authorization": "Bearer fixture-token"}
    captured["fail"] = True
    denied = await read_telegent_wellness(mcp_url="https://fixture.invalid/mcp", token="fixture-token",
                                          owner_login="synthetic", start=NOW.replace(hour=0),
                                          end=NOW.replace(hour=23))
    assert denied["complete"] is False and "fixture-token" not in denied["error"]


@pytest.mark.asyncio
async def test_telegent_live_reader_uses_selected_configured_oauth_manager(monkeypatch, tmp_path):
    import json
    import openharness.mcp.client as mcp_client
    from openharness.mcp.types import McpHttpServerConfig, McpOAuthConfig

    requested = {"login": "synthetic", "start": NOW.replace(hour=0).isoformat(),
                 "end": NOW.isoformat()}
    response = {"login": "synthetic", "user_id": "owner-1", "interval": requested.copy(),
                "nutrition_status": "complete", "nutrition_records": [],
                "nutrition_unassigned_records": []}

    class FakeManager:
        def __init__(self, configs):
            assert list(configs) == ["wellness-selected"]
            config = configs["wellness-selected"]
            assert config.headers == {"X-Tenant": "synthetic"}
            assert config.oauth.client_id == "synthetic-client"
            self.closed = False

        async def connect_all(self):
            return None

        def list_statuses(self):
            return [type("Status", (), {"state": "connected"})()]

        async def call_tool_result(self, server, name, args):
            assert server == "wellness-selected" and name == "get_wellness_data"
            assert args["params"] == requested
            return type("Result", (), {"output": json.dumps(response), "is_error": False})()

        async def close(self):
            self.closed = True

    monkeypatch.setattr(mcp_client, "McpClientManager", FakeManager)
    config = McpHttpServerConfig(url="https://synthetic.invalid/mcp", headers={"X-Tenant": "synthetic"},
        oauth=McpOAuthConfig(token_url="https://synthetic.invalid/token", client_id="synthetic-client",
                             token_file=str(tmp_path / "token.json")))
    result = await read_telegent_wellness(server_config=config, server_name="wellness-selected",
        owner_login="synthetic", start=NOW.replace(hour=0), end=NOW)
    assert result["complete"] is True
    assert result["start"] == requested["start"] and result["end"] == requested["end"]
    assert result["meals"] == [] and result["unassigned"] == []


def test_exact_replay_deduplicates_and_cross_workspace_event_is_inconclusive():
    event = raw_event()
    honcho, telegent = snapshots([event, dict(event)])
    assert grade(honcho, telegent)["a1"] == "PASS"
    honcho["messages"][0]["workspace_id"] = "foreign-workspace"
    assert grade(honcho, telegent)["a1"] == "INCONCLUSIVE"


def test_goal_rejects_tolerance_above_ten_percent_and_boolean_numbers():
    with pytest.raises(ValidationError):
        Goal.model_validate({**goal().model_dump(), "tolerance_fraction": 1})
    with pytest.raises(ValidationError):
        Goal.model_validate({**goal().model_dump(), "expected_kcal": True})


@pytest.mark.parametrize("field,value", [
    ("complete", "false"), ("since", None), ("queried_at", "2025-01-01T00:00:00Z"),
])
def test_malformed_honcho_scope_never_proves_absence(field, value):
    honcho, telegent = snapshots(events=[], canonical=False)
    honcho[field] = value
    assert grade(honcho, telegent)["a1"] == "INCONCLUSIVE"


@pytest.mark.parametrize("mutate", [
    lambda snapshot: snapshot.update(unassigned=None),
    lambda snapshot: snapshot.update(meals=["malformed-row"]),
])
def test_malformed_canonical_collections_never_prove_negative_absence(mutate):
    honcho, raw = snapshots(events=[], canonical=True)
    raw = {"complete": True, "user_id": "owner-1", "login": "owner",
           "start": "2026-10-01T00:00:00+00:00", "end": NOW.isoformat(),
           "queried_at": NOW.isoformat(), "meals": [], "unassigned": []}
    mutate(raw)
    canonical = bind_wellness_snapshot(raw, goal=goal(expected=False, kcal=None))
    negative = Manifest(schema_version=1, goals=[goal(expected=False, kcal=None)])
    assert grade_manifest(negative, honcho, canonical, now=NOW)[0]["a1"] == "INCONCLUSIVE"


def test_online_asof_and_moscow_local_day_are_scoped_without_future_completeness():
    honcho, telegent = snapshots(events=[], canonical=False)
    assert grade(honcho, telegent)["a1"] == "FAIL"
    moscow_goal = goal(expected=False, kcal=None).model_copy(update={"meal_timezone": "Europe/Moscow"})
    moscow_goal = Goal.model_validate({**moscow_goal.model_dump(mode="json"),
                                       "meal_timezone": "Europe/Moscow"})
    manifest = Manifest(schema_version=1, goals=[moscow_goal])
    honcho["since"] = "2026-09-30T21:00:00+00:00"
    honcho["until"] = NOW.isoformat()
    raw = {"complete": True, "user_id": "owner-1", "login": "owner",
           "start": "2026-09-30T21:00:00+00:00", "end": NOW.isoformat(),
           "queried_at": NOW.isoformat(), "meals": [], "unassigned": []}
    canonical = bind_wellness_snapshot(raw, goal=moscow_goal)
    assert grade_manifest(manifest, honcho, canonical, now=NOW)[0]["a1"] == "PASS"


def test_retry_and_noop_correction_keep_effective_latest_revision():
    first = raw_event("evt-1")
    retry = raw_event("evt-2")
    honcho, telegent = snapshots([first, retry])
    telegent["meal"]["latest_event_id"] = "evt-1"
    assert grade(honcho, telegent)["a1"] == "PASS"
    noop = per_turn_event("evt-noop", source="src-noop", episode="ep-noop", turn="noop", kcal=25,
                          kind="meal_correction", reply="src-tea")
    honcho, telegent = snapshots([first, noop])
    telegent["meal"]["latest_event_id"] = "evt-1"
    noop_goal = Goal.model_validate({**goal().model_dump(mode="json"), "episode_ids": ["ep-tea", "ep-noop"]})
    noop_result = grade_manifest(Manifest(schema_version=1, goals=[noop_goal]), honcho, telegent, now=NOW,
        reviewed_turn_sources={"ep-tea": ["src-tea"], "ep-noop": ["src-noop"]},
        reviewed_turn_provenance=reviewed_turn_map(first, noop))[0]
    assert noop_result["a1"] == "PASS"
    assert noop_result["actual_latest_event_id"] == "evt-1"


def test_prior_canonical_revision_is_pending_only_inside_grace():
    first = per_turn_event("evt-1", source="src-tea", episode="ep-tea", turn="root", kcal=100,
                           created="2026-10-01T11:58:00+00:00")
    correction = per_turn_event("evt-2", source="src-correction", episode="ep-correction",
        turn="correction", kcal=25, kind="meal_correction", reply="src-tea", created="2026-10-01T11:59:00+00:00")
    honcho, telegent = snapshots([first, correction])
    telegent["meal"]["latest_event_id"] = "evt-1"
    telegent["meal"]["energy_kcal_best"] = 100
    telegent["meal"]["capture_time"] = "2026-10-01T11:58:00+00:00"
    revised_goal = Goal.model_validate({**goal(kcal=25).model_dump(mode="json"),
        "episode_ids": ["ep-tea", "ep-correction"]})
    sources = {"ep-tea": ["src-tea"], "ep-correction": ["src-correction"]}
    operations = reviewed_turn_map(first, correction)
    assert grade_manifest(Manifest(schema_version=1, goals=[revised_goal]), honcho, telegent, now=NOW,
        reviewed_turn_sources=sources, reviewed_turn_provenance=operations)[0]["a1"] == "PENDING"
    assert grade_manifest(Manifest(schema_version=1, goals=[revised_goal]), honcho, telegent,
        now=NOW.replace(minute=10), reviewed_turn_sources=sources,
        reviewed_turn_provenance=operations)[0]["a1"] == "FAIL"
    telegent["meal"]["capture_time"] = "2027-01-01T00:00:00+00:00"
    future = grade_manifest(Manifest(schema_version=1, goals=[revised_goal]), honcho, telegent, now=NOW,
        reviewed_turn_sources=sources, reviewed_turn_provenance=operations)[0]
    assert future["a1"] == "INCONCLUSIVE"
    assert future["stage"] == "CANONICAL_PROVENANCE_MISMATCH"


def test_precise_meal_at_can_bind_local_day_without_meal_date():
    honcho, telegent = snapshots()
    annotation = honcho["messages"][0]["metadata"]["decision_trace"]["annotations"]["nutrition"]
    annotation["meal_date"] = None
    annotation["meal_at"] = "2026-10-01T10:00:00+00:00"
    telegent["meal"]["day"] = None
    telegent["meal"]["meal_date"] = None
    telegent["meal"]["meal_at"] = "2026-10-01T10:00:00+00:00"
    assert grade(honcho, telegent)["a1"] == "PASS"


def test_complete_unannotated_gateway_turn_without_trace_fails_known_consumed_goal(tmp_path: Path):
    from types import SimpleNamespace
    from openharness.channels.bus.events import InboundMessage
    from ohmo.evals.recorder import GatewayEvalRecorder
    from ohmo.gateway.memory_gate import MemoryScope
    from ohmo.gateway.runtime import _build_conversation_turn_metadata
    from ohmo.gateway.turn_context import build_turn_context

    message = InboundMessage(channel="telegram", sender_id="owner-1|mutable_name", chat_id="chat-1",
        content="I drank tea", timestamp=NOW, metadata={"message_id": "src-tea"})
    recorder = GatewayEvalRecorder.start(workspace=tmp_path,
        bundle=SimpleNamespace(session_id="session-1", cwd=str(tmp_path)), message=message,
        session_key="telegram:owner-1", user_text=message.content, user_goal="track tea")
    logical, _, assistant = _build_conversation_turn_metadata(
        turn_ctx=build_turn_context(message, session_id="session-1"), message=message,
        scope=MemoryScope("owner-1", ()), recorder=recorder)
    assert assistant["decision_trace_episode_id"] == recorder.episode_id
    assert "decision_trace" not in assistant
    recorder.record_gateway_final(text="I could not confirm the save")
    recorder.finish(status="completed")
    export = export_eval_dialogue(tmp_path / "evals", episode_ids=[recorder.episode_id])
    binding = validate_dialogue_binding(Manifest(schema_version=1, goals=[Goal.model_validate({
        **goal().model_dump(mode="json"), "episode_ids": [recorder.episode_id],
        "trace_episode_id": recorder.episode_id, "logical_turn_id": logical,
        "operation_id": assistant["client_op_id"], "eval_workspace": str(tmp_path.resolve()),
    })]), export)["tea"]
    assert binding["complete"] is True
    row = {"id": "unannotated-turn", "peer_id": "ohmo", "session_id": "session-1",
        "workspace_id": "workspace-1", "created_at": NOW.isoformat(), "content": "persisted assistant turn",
        "metadata": {**assistant, "role": "assistant"}}
    honcho, _ = snapshots(events=[])
    honcho["messages"] = [row]
    canonical_raw = {"complete": True, "user_id": "owner-1", "login": "owner",
        "start": "2026-10-01T00:00:00+00:00", "end": NOW.isoformat(), "queried_at": NOW.isoformat(),
        "meals": [], "unassigned": []}
    canonical = bind_wellness_snapshot(canonical_raw, goal=goal())
    result = grade_manifest(Manifest(schema_version=1, goals=[Goal.model_validate({
        **goal().model_dump(mode="json"), "episode_ids": [recorder.episode_id],
        "trace_episode_id": recorder.episode_id, "logical_turn_id": logical,
        "operation_id": assistant["client_op_id"], "eval_workspace": str(tmp_path.resolve()),
    })]), honcho, canonical, now=NOW, reviewed_turn_sources=binding["reviewed_turn_sources"],
        reviewed_turn_provenance=binding["reviewed_turn_provenance"])[0]
    assert result["a1"] == "FAIL"
    assert result["stage"] == "HONCHO_GOAL_MISMATCH"


def test_unannotated_authorized_camera_context_does_not_break_owner_meal_goal():
    from dataclasses import replace
    from openharness.channels.bus.events import InboundMessage
    from ohmo.gateway.camera import CAMERA_AUTHORITY
    from ohmo.gateway.memory_gate import MemoryScope
    from ohmo.gateway.runtime import _build_conversation_turn_metadata
    from ohmo.gateway.turn_context import build_turn_context

    initial = InboundMessage(channel="telegram", sender_id="__camera__", chat_id="123",
        content="Analyze this Camera image and ask whether the user ate it.", timestamp=NOW,
        metadata={"_synthetic": True, "_camera_authority": CAMERA_AUTHORITY,
            "_camera_candidate_id": "candidate-synthetic", "_camera_photo_id": 76})
    context = replace(build_turn_context(initial, session_id="session-1"), camera_authorized=True)
    _, _, camera_metadata = _build_conversation_turn_metadata(
        turn_ctx=context, message=initial, scope=MemoryScope("owner-1", ()))
    camera_metadata.update(role="assistant", decision_trace_episode_id="ep-camera-context")
    camera_context = {"id": "camera-context", "peer_id": "ohmo", "session_id": "session-1",
        "workspace_id": "workspace-1", "created_at": NOW.replace(hour=11, minute=59).isoformat(),
        "metadata": camera_metadata}
    honcho, telegent = snapshots()
    honcho["messages"].append(camera_context)
    assert grade(honcho, telegent)["a1"] == "PASS"
    annotated_camera = raw_event("annotated-camera-context")
    annotated_camera["metadata"]["source_principal"] = "telegram:__camera__"
    scoped, canonical = snapshots([annotated_camera])
    mismatch = grade(scoped, canonical)
    assert mismatch["a1"] == "INCONCLUSIVE"
    assert mismatch["stage"] == "HONCHO_SCOPE_MISMATCH"
    missing_tenant = json.loads(json.dumps(honcho))
    missing_tenant["messages"][1]["metadata"].pop("tenant_id")
    unscoped = grade(missing_tenant, telegent)
    assert unscoped["a1"] == "INCONCLUSIVE"
    assert unscoped["stage"] == "HONCHO_SCOPE_MISMATCH"
