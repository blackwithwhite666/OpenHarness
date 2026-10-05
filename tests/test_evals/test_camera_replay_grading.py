from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone

import pytest

from ohmo.evals.nutrition_persistence import (
    Goal,
    Manifest,
    bind_wellness_snapshot,
    derive_meal_id,
    grade_manifest,
    _exported_camera_context,
    _indexed_camera_callback_matches,
    _derive_exported_turn_provenance,
    _validated_typed_camera_replay_selection,
    _camera_typed_selection_matches_owner_input,
    validate_dialogue_binding,
)

NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
INITIAL_ID = "camera-initial"
OWNER_ID = "camera-owner"
REPLAY_ID = "camera-replay"
OWNER_SOURCE = "77"
REPLAY_SOURCE = "synthetic-replay-source"
EVENT_ID = "synthetic-original-event"
CANDIDATE_ID = "synthetic-camera-candidate"
PHOTO_ID = 77
OWNER = "synthetic-owner"
PRINCIPAL = "telegram:123"
SESSION = "synthetic-camera-session"
WORKSPACE = "/synthetic/evals"
OWNER_LOGICAL = "synthetic-owner-logical-turn"
OWNER_OPERATION = f"{OWNER_LOGICAL}:assistant"


def _annotation(kcal: float = 25.0) -> dict:
    return {
        "schema_version": 2,
        "record_type": "meal_observation",
        "consumption_status": "consumed",
        "meal_date": "2026-10-01",
        "energy_kcal_min": kcal,
        "energy_kcal_max": kcal,
        "energy_kcal_best": kcal,
    }


def _camera_context(kind: str, episode_id: str, *, source: str | None = None,
                    principal: str | None = None) -> dict:
    context = {
        "kind": kind,
        "episode_id": episode_id,
        "candidate_id": CANDIDATE_ID,
        "native_photo_id": PHOTO_ID,
        "tenant_id": OWNER,
        "gateway_session_id": SESSION,
        "recipient_principal": PRINCIPAL,
    }
    if kind == "initial_context":
        context.update({
            "source_principal": "telegram:__camera__",
            "logical_turn_id": "synthetic-camera-analysis-turn",
            "operation_id": "synthetic-camera-analysis-turn:assistant",
        })
    else:
        context.update({
            "source_message_id": source,
            "principal_id": principal or PRINCIPAL,
            "logical_turn_id": OWNER_LOGICAL,
            "operation_id": OWNER_OPERATION,
        })
    return context


def _export_turn(episode_id: str, source: str, *, final: dict | None = None) -> dict:
    result = {
        "episode_id": episode_id,
        "source_message_id": source,
        "logical_turn_id": OWNER_LOGICAL,
        "operation_id": OWNER_OPERATION,
        "principal_id": PRINCIPAL,
    }
    if final is not None:
        result["gateway_final_metadata"] = final
    return result


def _exported_episode(episode_id: str, created: str, source: str | None,
                      context: dict, turn: dict | None) -> dict:
    inbound = {
        "channel": "telegram",
        "sender_id": "__camera__" if source is None else "123",
        "chat_id": "123",
        "timestamp": created,
        "metadata": ({
            "_synthetic": True,
            "_camera_candidate_id": CANDIDATE_ID,
            "_camera_photo_id": PHOTO_ID,
        } if source is None else {
            "message_id": source,
            "_camera_candidate_id": CANDIDATE_ID,
        }),
    }
    episode_metadata = {"workspace": WORKSPACE, "inbound": inbound,
                        "trusted_camera_context": context}
    turns = []
    sources = []
    if turn is not None:
        episode_metadata["trusted_camera_turn_provenance"] = {
            key: turn[key] for key in (
                "episode_id", "source_message_id", "logical_turn_id", "operation_id", "principal_id")
        }
        turns = [turn]
        sources = [source]
    return {
        "episode": {"episode_id": episode_id, "created_at": created,
                    "session_id": SESSION, "metadata": episode_metadata},
        "dialogue": ([{"role": "assistant", "text": "Synthetic Camera analysis finished."}]
                     if source is None else [
                         {"role": "user", "text": "Synthetic typed meal confirmation."},
                         {"role": "assistant", "text": "Synthetic save confirmed."},
                     ]),
        "dialogue_complete": True,
        "source_message_ids": sources,
        "turn_provenance": turns,
        "principal_id": "telegram:__camera__" if source is None else PRINCIPAL,
        "trusted_camera_context": context,
    }


def _fixture(kcal: float = 25.0):
    initial_created = "2026-10-01T10:00:00+00:00"
    owner_created = "2026-10-01T11:00:00+00:00"
    replay_created = "2026-10-01T11:15:00+00:00"
    event_created = "2026-10-01T11:01:00+00:00"
    initial_context = _camera_context("initial_context", INITIAL_ID)
    owner_context = _camera_context("owner_turn", OWNER_ID, source=OWNER_SOURCE)
    replay_context = _camera_context("owner_turn", REPLAY_ID, source=REPLAY_SOURCE)
    finalizer = {"schema_version": 1, "annotation": _annotation(kcal)}
    owner_final = {"nutrition_append_event_id": EVENT_ID, "nutrition_finalization": finalizer}
    replay_final = {"nutrition_append_event_id": EVENT_ID}
    initial = _exported_episode(INITIAL_ID, initial_created, None, initial_context, None)
    derived_initial = _derive_exported_turn_provenance(
        initial["episode"], initial["episode"]["metadata"]["inbound"]
    )
    assert derived_initial is not None
    initial_context.update({
        "source_principal": derived_initial["principal_id"],
        "logical_turn_id": derived_initial["logical_turn_id"],
        "operation_id": derived_initial["operation_id"],
    })
    initial["trusted_camera_context"] = initial_context
    initial["episode"]["metadata"]["trusted_camera_context"] = initial_context
    owner_turn = _export_turn(OWNER_ID, OWNER_SOURCE, final=owner_final)
    owner = _exported_episode(OWNER_ID, owner_created, OWNER_SOURCE, owner_context, owner_turn)
    replay_turn = _export_turn(REPLAY_ID, REPLAY_SOURCE, final=replay_final)
    replay = _exported_episode(REPLAY_ID, replay_created, REPLAY_SOURCE, replay_context, replay_turn)
    export = {"privacy": "private", "episodes": [initial, owner, replay]}
    canonical_meal_id = derive_meal_id(
        tenant_id=OWNER, source_principal=PRINCIPAL,
        gateway_session_id=SESSION, source_message_id=OWNER_SOURCE,
    )
    reviewed_goal = Goal(
        case_id="synthetic-camera-meal",
        episode_ids=[INITIAL_ID, OWNER_ID, REPLAY_ID],
        owner_id=OWNER,
        canonical_owner_id=OWNER,
        principal_id=PRINCIPAL,
        workspace_id="synthetic-workspace",
        eval_workspace=WORKSPACE,
        peer_id="ohmo",
        canonical_login="synthetic-owner",
        session_id="synthetic-honcho-session",
        gateway_session_id=SESSION,
        source_message_id=OWNER_SOURCE,
        meal_date=date(2026, 10, 1),
        meal_timezone="UTC",
        trajectory_started_at=datetime.fromisoformat(initial_created),
        trajectory_as_of=datetime.fromisoformat(replay_created),
        logical_turn_id=OWNER_LOGICAL,
        trace_episode_id=OWNER_ID,
        operation_id=OWNER_OPERATION,
        canonical_meal_id=canonical_meal_id,
        expected_consumed=True,
        expected_kcal=kcal,
        tolerance_fraction=0.0,
        expectation_origin="explicit_fixture",
        expectation_source="synthetic-camera-replay-fixture",
        review_notes="Synthetic source-derived fixture.",
    )
    annotation = _annotation(kcal)
    raw_event = {
        "id": EVENT_ID,
        "peer_id": "ohmo",
        "session_id": "synthetic-honcho-session",
        "workspace_id": "synthetic-workspace",
        "created_at": event_created,
        "content": "Synthetic persisted nutrition event.",
        "metadata": {
            "role": "assistant",
            "tenant_id": OWNER,
            "gateway_session_id": SESSION,
            "client_op_id": OWNER_OPERATION,
            "source_principal": PRINCIPAL,
            "logical_turn_id": OWNER_LOGICAL,
            "source_message_id": OWNER_SOURCE,
            "decision_trace_episode_id": OWNER_ID,
            "decision_trace": {"episode_id": OWNER_ID, "annotations": {"nutrition": annotation}},
        },
    }
    honcho = {
        "complete": True,
        "workspace_id": "synthetic-workspace",
        "session_id": "synthetic-honcho-session",
        "owner_id": OWNER,
        "since": "2026-10-01T00:00:00+00:00",
        "until": NOW.isoformat(),
        "queried_at": NOW.isoformat(),
        "messages": [raw_event],
    }
    record = {
        "meal_id": canonical_meal_id,
        "revision": 1,
        "status": "active",
        "latest_event_id": EVENT_ID,
        "day": "2026-10-01",
        "provisional": True,
        "capture_time": event_created,
        "meal_at": None,
        "meal_date": "2026-10-01",
        "source_message_id": OWNER_SOURCE,
        "ingest_source": "telegram",
        "confirmation_required": None,
        "reply_to_source_message_id": None,
        "received_at": None,
        "is_forwarded": False,
        "source_message_at": None,
        "is_estimate": True,
        "basis": [],
        "consumption_status": "consumed",
        "energy_kcal_min": kcal,
        "energy_kcal_max": kcal,
        "energy_kcal_best": kcal,
        "protein_g": None,
        "fat_g": None,
        "carbohydrate_g": None,
        "items": [],
        "confidence": "medium",
        "assumptions": [],
        "warnings": [],
    }
    snapshot = {
        "complete": True,
        "user_id": OWNER,
        "login": "synthetic-owner",
        "start": "2026-10-01T00:00:00+00:00",
        "end": NOW.isoformat(),
        "queried_at": NOW.isoformat(),
        "meals": [record],
        "unassigned": [],
    }
    return reviewed_goal, export, honcho, snapshot


def _grade_fixture(fixture):
    reviewed_goal, export, honcho, snapshot = fixture
    binding = validate_dialogue_binding(
        Manifest(schema_version=1, goals=[reviewed_goal]), export
    )[reviewed_goal.case_id]
    canonical = bind_wellness_snapshot(snapshot, goal=reviewed_goal)
    result = grade_manifest(
        Manifest(schema_version=1, goals=[reviewed_goal]), honcho, canonical, now=NOW,
        reviewed_turn_sources=binding["reviewed_turn_sources"],
        reviewed_turn_provenance=binding["reviewed_turn_provenance"],
    )[0]
    return result, binding, canonical


def _native_repeat_fixture(kcal: float = 25.0, *, target: int | str = 77,
                           same_query: bool = False):
    fixture = _fixture(kcal)
    goal, export, honcho, snapshot = fixture
    owner = export["episodes"][1]
    replay = export["episodes"][2]
    owner["episode"]["metadata"]["inbound"]["user_text"] = "2 eggs and rice"
    owner["episode"]["metadata"]["inbound"]["metadata"].update({
        "message_id": target,
        "native_message_id": target,
        "_camera_candidate_id": CANDIDATE_ID,
        "_camera_native_binding": str(target),
        "callback_query": True,
        "_camera_route": "callback",
        "callback_query_id": "native-callback-original-1",
        "callback_data": "ask:0",
        "native_keyboard_options": ["2 eggs and rice", "Not eaten"],
        "native_keyboard_selected_index": 0,
        "native_keyboard_selected_label": "2 eggs and rice",
        "native_keyboard_prompt": "Question",
        "native_keyboard_question": "Question",
        "native_keyboard_reflection_confirmed": True,
        "native_keyboard_reflection": "Question\n\n✅ 2 eggs and rice",
    })
    owner["dialogue"][0]["text"] = "2 eggs and rice"
    context = replay["trusted_camera_context"]
    turn = replay["turn_provenance"][0]
    context["source_message_id"] = OWNER_SOURCE
    turn["source_message_id"] = OWNER_SOURCE
    replay["source_message_ids"] = [OWNER_SOURCE]
    replay["episode"]["metadata"]["trusted_camera_context"]["source_message_id"] = OWNER_SOURCE
    replay["episode"]["metadata"]["trusted_camera_turn_provenance"]["source_message_id"] = OWNER_SOURCE
    replay["episode"]["metadata"]["inbound"].update({
        "channel": "telegram", "sender_id": "123", "chat_id": "123",
        "user_text": "2 eggs and rice",
    })
    replay["episode"]["metadata"]["inbound"]["metadata"].update({
        "_camera_candidate_id": CANDIDATE_ID,
        "message_id": target,
        "native_message_id": target,
        "callback_query": True,
        "_camera_route": "callback",
        "_camera_existing_meal_replay": True,
        "_camera_ingress_callback_eligible": True,
        "callback_query_id": ("native-callback-original-1" if same_query
                               else "native-callback-replay-2"),
        "callback_data": "ask:0",
        "native_keyboard_options": ["2 eggs and rice", "Not eaten"],
        "native_keyboard_selected_index": 0,
        "native_keyboard_selected_label": "2 eggs and rice",
        "native_keyboard_prompt": "Question",
        "native_keyboard_question": "Question",
        "native_keyboard_reflection_confirmed": True,
        "native_keyboard_reflection": "Question\n\n✅ 2 eggs and rice",
        "_camera_native_binding": str(target),
    })
    replay["dialogue"][0]["text"] = "2 eggs and rice"
    return fixture


def test_complete_camera_owner_replay_passes_with_distinct_source_and_same_receipt():
    result, binding, canonical = _grade_fixture(_fixture())
    assert binding["complete"] is True, binding
    assert canonical["complete"] is True
    assert result["a1"] == "PASS", result
    assert result["a2"] == "NOT_RUN"


@pytest.mark.parametrize("target", [77, "77"])
def test_complete_camera_owner_replay_passes_for_repeated_native_tap_on_same_message(target):
    result, binding, canonical = _grade_fixture(_native_repeat_fixture(target=target))
    assert binding["complete"] is True, binding
    assert canonical["complete"] is True
    assert result["a1"] == "PASS", result
    assert result["a2"] == "NOT_RUN"


def test_same_query_id_later_native_retry_remains_idempotent():
    result, binding, canonical = _grade_fixture(_native_repeat_fixture(same_query=True))
    assert binding["complete"] is True, binding
    assert canonical["complete"] is True
    assert result["a1"] == "PASS", result
    assert result["a2"] == "NOT_RUN"


@pytest.mark.parametrize("mutation", [
    "same_source_without_click", "wrong_target", "missing_query_id", "bad_option_index",
    "callback_data", "reflection", "native_binding", "route", "not_callback", "not_eligible",
    "not_existing_replay", "bool_target", "negative_target", "missing_trusted_context",
    "forged_selection", "null_target", "garbage_target", "foreign_target",
])
def test_same_source_camera_replay_requires_matching_native_click(mutation):
    goal, export, honcho, snapshot = _native_repeat_fixture()
    replay = export["episodes"][2]
    click = replay["episode"]["metadata"]["inbound"]["metadata"]
    if mutation == "same_source_without_click":
        replay["episode"]["metadata"]["inbound"]["metadata"] = {
            "message_id": OWNER_SOURCE, "_camera_candidate_id": CANDIDATE_ID,
        }
    elif mutation == "wrong_target":
        click["native_message_id"] = "other-message"
    elif mutation == "missing_query_id":
        click["callback_query_id"] = None
    elif mutation == "bad_option_index":
        click["native_keyboard_selected_index"] = 1
    elif mutation == "callback_data":
        click["callback_data"] = "ask:1"
    elif mutation == "reflection":
        click["native_keyboard_reflection_confirmed"] = False
    elif mutation == "native_binding":
        click["_camera_native_binding"] = "other-photo"
    elif mutation == "route":
        click["_camera_route"] = "reply"
    elif mutation == "not_callback":
        click["callback_query"] = False
    elif mutation == "not_eligible":
        click["_camera_ingress_callback_eligible"] = False
    elif mutation == "not_existing_replay":
        click.pop("_camera_existing_meal_replay")
    elif mutation == "bool_target":
        click["message_id"] = True
    elif mutation == "negative_target":
        click["native_message_id"] = -77
    elif mutation == "null_target":
        click["native_message_id"] = None
    elif mutation == "garbage_target":
        click["_camera_native_binding"] = "77x"
    elif mutation == "foreign_target":
        click["message_id"] = 78
    elif mutation == "missing_trusted_context":
        replay.pop("trusted_camera_context")
        replay["episode"]["metadata"].pop("trusted_camera_context")
    elif mutation == "forged_selection":
        click["native_keyboard_selected_label"] = "different offered option"
    binding = validate_dialogue_binding(Manifest(schema_version=1, goals=[goal]), export)[goal.case_id]
    canonical = bind_wellness_snapshot(snapshot, goal=goal)
    result = grade_manifest(
        Manifest(schema_version=1, goals=[goal]), honcho, canonical, now=NOW,
        reviewed_turn_sources=binding["reviewed_turn_sources"],
        reviewed_turn_provenance=binding["reviewed_turn_provenance"],
    )[0]
    assert binding["complete"] is False, (mutation, binding)
    assert result["a1"] == "INCONCLUSIVE", (mutation, result)


def test_indexed_inbound_and_episode_inbound_must_preserve_native_click_fields():
    _, export, _, _ = _native_repeat_fixture(target=77)
    replay = export["episodes"][2]
    recorded = replay["episode"]["metadata"]["inbound"]
    indexed_payload = {
        "channel": recorded["channel"], "sender_id": recorded["sender_id"],
        "chat_id": recorded["chat_id"], "metadata": deepcopy(recorded["metadata"]),
    }
    assert _indexed_camera_callback_matches(replay["episode"], indexed_payload) is True
    indexed_payload["metadata"]["native_message_id"] = 78
    assert _indexed_camera_callback_matches(replay["episode"], indexed_payload) is False
    indexed_payload = {
        "channel": recorded["channel"], "sender_id": recorded["sender_id"],
        "chat_id": recorded["chat_id"], "metadata": deepcopy(recorded["metadata"]),
    }
    indexed_payload["metadata"]["callback_query_id"] = "unrecorded-query"
    assert _indexed_camera_callback_matches(replay["episode"], indexed_payload) is False
    for field, value in (
        ("_camera_route", "reply"),
        ("_camera_ingress_callback_eligible", False),
        ("native_keyboard_selected_label", "Not eaten"),
        ("_camera_native_binding", "78"),
    ):
        indexed_payload = {
            "channel": recorded["channel"], "sender_id": recorded["sender_id"],
            "chat_id": recorded["chat_id"], "metadata": deepcopy(recorded["metadata"]),
        }
        indexed_payload["metadata"][field] = value
        assert _indexed_camera_callback_matches(replay["episode"], indexed_payload) is False


def test_exporter_rejects_native_callback_indicator_disagreement_and_grader_stays_inconclusive():
    fixture = _native_repeat_fixture()
    goal, export, honcho, snapshot = fixture
    replay = export["episodes"][2]
    episode = replay["episode"]
    recorded = episode["metadata"]["inbound"]
    indexed_payload = {
        "channel": recorded["channel"], "sender_id": recorded["sender_id"],
        "chat_id": recorded["chat_id"], "metadata": deepcopy(recorded["metadata"]),
    }
    event = {"kind": "inbound_message", "payload": indexed_payload}
    assert _exported_camera_context(episode, [event]) is not None

    indicators = (
        "callback_query", "_camera_route", "_camera_existing_meal_replay",
        "_camera_ingress_callback_eligible",
    )
    for field in indicators:
        indexed_payload["metadata"].pop(field)
    assert _exported_camera_context(episode, [event]) is None

    # Match export_eval_dialogue's invalid-context result before ordinary grading.
    replay.pop("trusted_camera_context")
    replay["dialogue_complete"] = False
    result, binding, _ = _grade_fixture((goal, export, honcho, snapshot))
    assert binding["complete"] is False, binding
    assert result["a1"] == "INCONCLUSIVE", result

    # The reverse direction is also reachable: indexed data retains native fields
    # while the episode's recorded inbound has lost them.
    _, reverse_export, _, _ = _native_repeat_fixture()
    reverse_episode = reverse_export["episodes"][2]["episode"]
    reverse_recorded = reverse_episode["metadata"]["inbound"]
    reverse_payload = {
        "channel": reverse_recorded["channel"], "sender_id": reverse_recorded["sender_id"],
        "chat_id": reverse_recorded["chat_id"], "metadata": deepcopy(reverse_recorded["metadata"]),
    }
    reverse_event = {"kind": "inbound_message", "payload": reverse_payload}
    for field in indicators:
        reverse_recorded["metadata"].pop(field)
    assert _exported_camera_context(reverse_episode, [reverse_event]) is None


def _multi_operation_camera_fixture():
    goal, export, _, _ = _native_repeat_fixture()
    initial, root, root_repeat = export["episodes"]
    clarification_id = "camera-owner-clarification"
    clarification_source = "88"
    clarification_logical = "camera-clarification-turn"
    clarification_context = _camera_context(
        "owner_turn", clarification_id, source=clarification_source
    )
    clarification_context.update({
        "logical_turn_id": clarification_logical,
        "operation_id": f"{clarification_logical}:assistant",
    })
    clarification_turn = _export_turn(clarification_id, clarification_source)
    clarification_turn.update({
        "logical_turn_id": clarification_logical,
        "operation_id": f"{clarification_logical}:assistant",
    })
    clarification = _exported_episode(
        clarification_id, "2026-10-01T10:30:00+00:00", clarification_source,
        clarification_context, clarification_turn,
    )

    correction_id = "camera-quantity-correction"
    correction_logical = "camera-correction-turn"
    correction = deepcopy(root_repeat)
    correction["episode"]["episode_id"] = correction_id
    correction["episode"]["created_at"] = "2026-10-01T11:10:00+00:00"
    correction_context = correction["trusted_camera_context"]
    correction_context.update({
        "episode_id": correction_id,
        "logical_turn_id": correction_logical,
        "operation_id": f"{correction_logical}:assistant",
    })
    correction["episode"]["metadata"]["trusted_camera_context"] = deepcopy(correction_context)
    correction_turn = correction["turn_provenance"][0]
    correction_turn.update({
        "episode_id": correction_id,
        "logical_turn_id": correction_logical,
        "operation_id": f"{correction_logical}:assistant",
        "gateway_final_metadata": {
            "nutrition_append_event_id": "synthetic-correction-event",
            "nutrition_finalization": {"schema_version": 1,
                                       "annotation": _annotation(20.0)},
        },
    })
    correction["episode"]["metadata"]["trusted_camera_turn_provenance"] = {
        key: correction_turn[key] for key in (
            "episode_id", "source_message_id", "logical_turn_id", "operation_id", "principal_id")
    }
    correction["episode"]["metadata"]["inbound"]["metadata"]["callback_query_id"] = "correction-query"

    repeated_id = "camera-correction-repeat"
    repeated = deepcopy(correction)
    repeated["episode"]["episode_id"] = repeated_id
    repeated["episode"]["created_at"] = "2026-10-01T11:15:00+00:00"
    repeated_context = repeated["trusted_camera_context"]
    repeated_context["episode_id"] = repeated_id
    repeated["episode"]["metadata"]["trusted_camera_context"] = deepcopy(repeated_context)
    repeated_turn = repeated["turn_provenance"][0]
    repeated_turn["episode_id"] = repeated_id
    repeated_turn["gateway_final_metadata"] = {
        "nutrition_append_event_id": "synthetic-correction-event",
    }
    repeated["episode"]["metadata"]["trusted_camera_turn_provenance"] = {
        key: repeated_turn[key] for key in (
            "episode_id", "source_message_id", "logical_turn_id", "operation_id", "principal_id")
    }
    repeated["episode"]["metadata"]["inbound"]["metadata"]["callback_query_id"] = "correction-repeat-query"

    goal = goal.model_copy(update={
        "episode_ids": [INITIAL_ID, clarification_id, OWNER_ID, correction_id, repeated_id],
        "trajectory_as_of": datetime.fromisoformat("2026-10-01T11:15:00+00:00"),
    })
    return goal, {"privacy": "private", "episodes": [
        initial, clarification, root, correction, repeated,
    ]}


def test_camera_history_binds_clarification_and_correction_to_their_own_operations():
    goal, export = _multi_operation_camera_fixture()
    manifest = Manifest(schema_version=1, goals=[goal])
    binding = validate_dialogue_binding(manifest, export)[goal.case_id]
    assert binding["complete"] is True, binding

    # A repeat must retain the correction operation and its own exact source turn.
    malformed = deepcopy(export)
    malformed["episodes"][-1]["trusted_camera_context"]["operation_id"] = OWNER_OPERATION
    malformed["episodes"][-1]["episode"]["metadata"]["trusted_camera_context"]["operation_id"] = OWNER_OPERATION
    rejected = validate_dialogue_binding(manifest, malformed)[goal.case_id]
    assert rejected["complete"] is False, rejected


def _as_typed_current_correction_repeat(goal, export, text):
    repeated = export["episodes"][-1]
    source = "typed-current-correction-repeat"
    context = repeated["trusted_camera_context"]
    context.update(source_message_id=source)
    repeated["episode"]["metadata"]["trusted_camera_context"] = deepcopy(context)
    turn = repeated["turn_provenance"][0]
    turn.update(source_message_id=source)
    repeated["episode"]["metadata"]["trusted_camera_turn_provenance"]["source_message_id"] = source
    repeated["source_message_ids"] = [source]
    inbound = repeated["episode"]["metadata"]["inbound"]
    inbound["channel"] = "telegram"
    inbound["user_text"] = text
    inbound["metadata"] = {
        "message_id": source,
        "reply_to_message_id": goal.source_message_id,
        "_telegram_raw_text": text,
        "_camera_correction_replay_typed": True,
        "_camera_candidate_id": CANDIDATE_ID,
        "_camera_turn_id": context["logical_turn_id"],
        "is_group": False,
    }
    repeated["dialogue"][0]["text"] = text
    return repeated


def test_typed_correction_repeat_uses_accepted_owner_selection_not_model_quantity_text():
    goal, export = _multi_operation_camera_fixture()
    correction = export["episodes"][-2]
    annotation = correction["turn_provenance"][0]["gateway_final_metadata"][
        "nutrition_finalization"]["annotation"]
    annotation["items"] = [{"name": "synthetic food", "quantity_text": "2 eggs and rice ",
                            "energy_kcal_best": 20.0}]
    _as_typed_current_correction_repeat(goal, export, "2 eggs and rice")
    manifest = Manifest(schema_version=1, goals=[goal])
    binding = validate_dialogue_binding(manifest, export)[goal.case_id]
    assert binding["complete"] is True, binding

    wrong = deepcopy(export)
    _as_typed_current_correction_repeat(goal, wrong, "2 pieces")
    rejected = validate_dialogue_binding(manifest, wrong)[goal.case_id]
    assert rejected["complete"] is False, rejected


def test_native_callback_shape_cannot_pass_as_a_typed_distinct_source_replay():
    fixture = _fixture()
    goal, export, honcho, snapshot = fixture
    replay = export["episodes"][2]
    replay["episode"]["metadata"]["inbound"]["metadata"]["callback_data"] = "ask:0"
    result, binding, _ = _grade_fixture(fixture)
    assert binding["complete"] is False, binding
    assert result["a1"] == "INCONCLUSIVE", result


def test_typed_distinct_source_reply_route_remains_a_valid_replay():
    fixture = _fixture()
    replay = fixture[1]["episodes"][2]
    replay["episode"]["metadata"]["inbound"]["metadata"].update({
        "callback_query": False,
        "_camera_route": "reply",
    })
    result, binding, canonical = _grade_fixture(fixture)
    assert binding["complete"] is True, binding
    assert canonical["complete"] is True
    assert result["a1"] == "PASS", result


def _typed_current_correction_item(goal, *, text="1 piece", message_id=None, chat_type="private"):
    _, export, _, _ = _native_repeat_fixture()
    item = deepcopy(export["episodes"][2])
    episode = item["episode"]
    context = item["trusted_camera_context"]
    source = (str(message_id) if type(message_id) is int and message_id > 0
              else "typed-current-correction-source")
    logical = "typed-current-correction-turn"
    operation = f"{logical}:assistant"
    context.update(source_message_id=source, logical_turn_id=logical, operation_id=operation)
    episode["metadata"]["trusted_camera_context"] = deepcopy(context)
    turn = item["turn_provenance"][0]
    turn.update(source_message_id=source, logical_turn_id=logical, operation_id=operation)
    episode["metadata"]["trusted_camera_turn_provenance"] = {
        key: turn[key] for key in (
            "episode_id", "source_message_id", "logical_turn_id", "operation_id", "principal_id")
    }
    item["source_message_ids"] = [source]
    inbound = episode["metadata"]["inbound"]
    inbound.update(channel="telegram", sender_id="123", chat_id="123", user_text=text)
    inbound["metadata"] = {
        "message_id": source if message_id is None else message_id,
        "reply_to_message_id": goal.source_message_id,
        "_telegram_raw_text": text,
        "_camera_correction_replay_typed": True,
        "_camera_candidate_id": CANDIDATE_ID,
        "_camera_turn_id": logical,
        "is_group": False,
    }
    if chat_type is not None:
        inbound["metadata"]["chat_type"] = chat_type
    item["dialogue"] = [
        {"role": "user", "text": text},
        {"role": "assistant", "text": "Already saved."},
    ]
    return item, context


def test_typed_current_correction_requires_bound_source_and_plain_message_shape():
    goal, _, _, _ = _fixture()
    item, context = _typed_current_correction_item(goal)
    assert _validated_typed_camera_replay_selection(item, goal, context) == "1 piece"

    for path, value in (
        (("episode", "metadata", "inbound", "metadata", "message_id"), "other-source"),
        (("episode", "metadata", "inbound", "metadata", "reply_to_message_id"), "78"),
        (("episode", "metadata", "inbound", "metadata", "_camera_turn_id"), "other-turn"),
        (("episode", "metadata", "inbound", "metadata", "callback_query_id"), "fake-click"),
        (("episode", "metadata", "inbound", "metadata", "_camera_candidate_id"), "other-candidate"),
    ):
        changed = deepcopy(item)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert _validated_typed_camera_replay_selection(
            changed, goal, changed["trusted_camera_context"]
        ) is None, path

    changed = deepcopy(item)
    changed["dialogue"][0]["text"] = "2 pieces"
    assert _validated_typed_camera_replay_selection(
        changed, goal, changed["trusted_camera_context"]
    ) is None


def test_typed_correction_replay_requires_indexed_and_recorded_envelope_agreement():
    goal, _, _, _ = _fixture()
    item, _ = _typed_current_correction_item(goal)
    episode = item["episode"]
    recorded = episode["metadata"]["inbound"]
    payload = {
        "channel": recorded["channel"], "sender_id": recorded["sender_id"],
        "chat_id": recorded["chat_id"], "metadata": deepcopy(recorded["metadata"]),
    }
    event = {"kind": "inbound_message", "payload": payload}
    assert _exported_camera_context(episode, [event]) is not None
    payload["metadata"]["_telegram_raw_text"] = "different text"
    assert _exported_camera_context(episode, [event]) is None


def test_typed_repeat_matches_accepted_owner_input_not_generated_quantity_wording():
    accepted_owner_input = "1 piece"
    generated_quantity_text = "1 piece "
    # Generated wording is deliberately not part of the owner-input comparison.
    assert generated_quantity_text != accepted_owner_input
    assert _camera_typed_selection_matches_owner_input("1 piece", accepted_owner_input)
    assert not _camera_typed_selection_matches_owner_input("2 pieces", accepted_owner_input)


def test_typed_telegram_wire_id_and_private_scope_accept_actual_runtime_shape():
    goal, _, _, _ = _fixture()
    item, context = _typed_current_correction_item(goal, message_id=777, chat_type=None)
    assert item["episode"]["metadata"]["inbound"]["metadata"]["message_id"] == 777
    assert "chat_type" not in item["episode"]["metadata"]["inbound"]["metadata"]
    assert _validated_typed_camera_replay_selection(item, goal, context) == "1 piece"

    episode = item["episode"]
    inbound = episode["metadata"]["inbound"]
    payload = {key: inbound[key] for key in ("channel", "sender_id", "chat_id")}
    payload["metadata"] = deepcopy(inbound["metadata"])
    assert _exported_camera_context(episode, [{"kind": "inbound_message", "payload": payload}]) is not None
    payload["metadata"]["message_id"] = 778
    assert _exported_camera_context(episode, [{"kind": "inbound_message", "payload": payload}]) is None

    for invalid_id in (True, 0, -1):
        changed, changed_context = _typed_current_correction_item(
            goal, message_id=invalid_id, chat_type=None
        )
        assert _validated_typed_camera_replay_selection(
            changed, goal, changed_context
        ) is None
    shared, shared_context = _typed_current_correction_item(goal, chat_type="group")
    assert _validated_typed_camera_replay_selection(shared, goal, shared_context) is None
    group, group_context = _typed_current_correction_item(goal, chat_type=None)
    group["episode"]["metadata"]["inbound"]["metadata"]["is_group"] = True
    assert _validated_typed_camera_replay_selection(group, goal, group_context) is None

    for field in ("_camera_candidate_id", "_camera_turn_id", "_telegram_raw_text"):
        changed, changed_context = _typed_current_correction_item(goal, chat_type=None)
        changed["episode"]["metadata"]["inbound"]["metadata"].pop(field)
        assert _validated_typed_camera_replay_selection(
            changed, goal, changed_context
        ) is None, field
    mixed, mixed_context = _typed_current_correction_item(goal, chat_type=None)
    mixed["episode"]["metadata"]["inbound"]["metadata"]["native_keyboard_selected_index"] = 0
    assert _validated_typed_camera_replay_selection(mixed, goal, mixed_context) is None


@pytest.mark.parametrize("mutation", [
    "missing_replay_receipt", "wrong_replay_receipt", "candidate", "photo",
    "tenant", "session", "principal", "logical_turn", "operation",
    "owner_finalizer", "owner_finalizer_annotation", "owner_receipt", "owner_event_identity", "owner_source_list",
    "replay_episode_time", "native_source_envelope", "native_candidate", "missing_initial",
    "different_valid_selection",
])
def test_unproved_or_conflicting_camera_replay_stays_inconclusive(mutation):
    fixture = _native_repeat_fixture()
    goal_value, export, honcho, snapshot = fixture
    export = deepcopy(export)
    replay = export["episodes"][2]
    owner = export["episodes"][1]
    replay_turn = replay["turn_provenance"][0]
    replay_final = replay_turn["gateway_final_metadata"]
    if mutation == "missing_replay_receipt":
        replay_final.pop("nutrition_append_event_id")
    elif mutation == "wrong_replay_receipt":
        replay_final["nutrition_append_event_id"] = "synthetic-unmatched-event"
    elif mutation in {"candidate", "photo", "tenant", "session", "principal", "logical_turn", "operation"}:
        key, value = {
            "candidate": ("candidate_id", "synthetic-other-candidate"),
            "photo": ("native_photo_id", 78),
            "tenant": ("tenant_id", "synthetic-other-owner"),
            "session": ("gateway_session_id", "synthetic-other-session"),
            "principal": ("principal_id", "telegram:456"),
            "logical_turn": ("logical_turn_id", "synthetic-other-logical"),
            "operation": ("operation_id", "synthetic-other-operation:assistant"),
        }[mutation]
        replay["trusted_camera_context"][key] = value
    elif mutation == "owner_finalizer":
        owner["turn_provenance"][0]["gateway_final_metadata"].pop("nutrition_finalization")
    elif mutation == "owner_finalizer_annotation":
        owner["turn_provenance"][0]["gateway_final_metadata"]["nutrition_finalization"]["annotation"]["energy_kcal_best"] = 26.0
    elif mutation == "owner_receipt":
        owner["turn_provenance"][0]["gateway_final_metadata"]["nutrition_append_event_id"] = "wrong-original"
    elif mutation == "owner_event_identity":
        honcho["messages"][0]["id"] = "synthetic-other-event"
    elif mutation == "owner_source_list":
        owner["source_message_ids"] = []
    elif mutation == "replay_episode_time":
        replay["episode"]["created_at"] = "2026-10-01T10:30:00+00:00"
    elif mutation == "native_source_envelope":
        replay["episode"]["metadata"]["inbound"]["metadata"]["message_id"] = "other-message"
    elif mutation == "native_candidate":
        replay["episode"]["metadata"]["inbound"]["metadata"]["_camera_candidate_id"] = "other-candidate"
    elif mutation == "missing_initial":
        export["episodes"] = export["episodes"][1:]
    elif mutation == "different_valid_selection":
        click = replay["episode"]["metadata"]["inbound"]["metadata"]
        click.update({
            "native_keyboard_options": ["Not eaten", "2 eggs and rice"],
            "native_keyboard_selected_index": 0,
            "native_keyboard_selected_label": "Not eaten",
            "callback_data": "ask:0",
            "native_keyboard_reflection": "Question\n\n✅ Not eaten",
        })
        replay["dialogue"][0]["text"] = "Not eaten"

    binding = validate_dialogue_binding(
        Manifest(schema_version=1, goals=[goal_value]), export
    )[goal_value.case_id]
    canonical = bind_wellness_snapshot(snapshot, goal=goal_value)
    result = grade_manifest(
        Manifest(schema_version=1, goals=[goal_value]), honcho, canonical, now=NOW,
        reviewed_turn_sources=binding["reviewed_turn_sources"],
        reviewed_turn_provenance=binding["reviewed_turn_provenance"],
    )[0]
    assert result["a1"] == "INCONCLUSIVE", (mutation, result)


@pytest.mark.parametrize("mutation", ["new_event", "duplicate_canonical_row"])
def test_extra_event_or_duplicate_canonical_row_never_passes(mutation):
    fixture = _fixture()
    goal_value, export, honcho, snapshot = fixture
    if mutation == "new_event":
        extra = deepcopy(honcho["messages"][0])
        extra["id"] = "synthetic-extra-event"
        extra["created_at"] = "2026-10-01T11:02:00+00:00"
        extra["metadata"]["source_message_id"] = REPLAY_SOURCE
        extra["metadata"]["decision_trace_episode_id"] = REPLAY_ID
        extra["metadata"]["decision_trace"]["episode_id"] = REPLAY_ID
        honcho["messages"].append(extra)
    else:
        snapshot["meals"].append(deepcopy(snapshot["meals"][0]))
    result, binding, canonical = _grade_fixture((goal_value, export, honcho, snapshot))
    assert binding["complete"] is True
    assert result["a1"] != "PASS", result
    if mutation == "duplicate_canonical_row":
        assert result["a1"] == "FAIL" and result["stage"] == "CANONICAL_MISMATCH", result
