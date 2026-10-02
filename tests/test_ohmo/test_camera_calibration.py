"""Synthetic-only tests for the private Camera grader prototype."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import sys
from pathlib import Path
import pytest
from PIL import Image
from pydantic import ValidationError

from ohmo.evals.camera_calibration import (
    A2Vote,
    CallBudget,
    Case,
    JudgeCase,
    Reference,
    SOL_PROMPT_VERSION,
    a2_prompt,
    calibrate_case,
    calibrate_judge_case,
    checked_image,
    effective_meal,
    main,
    score_a1,
    sol_prompt,
    write_report,
)
from ohmo.evals.camera_subscription_results import SubscriptionResults, read_private_json
from ohmo.evals.nutrition_persistence import Goal, derive_meal_id
from datetime import date, datetime, timezone


def annotation(record_type="meal_observation", **values):
    return {"schema_version": 2, "record_type": record_type, **values}


def make_case(tmp_path, *, state="consumed", events=None):
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 3), (20, 80, 140)).save(image)
    if events is None:
        events = [
            event(
                annotation("meal_observation", consumption_status="consumed", energy_kcal_best=440)
            )
        ]
    case = Case.model_validate(
        {
            "case_id": "private-1",
            "image_path": image,
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "prefix": [
                {"role": "user", "text": "I ate this."},
                {"role": "assistant", "text": "What was the amount?"},
            ],
            "dialogue": [
                {"role": "user", "text": "I ate this."},
                {"role": "assistant", "text": "What was the amount?"},
                {"role": "assistant", "text": "I recorded the meal."},
            ],
            "reviewed_state": state,
            "origin": "camera",
            "source_message_id": "source-1",
            "owner_id": "owner-1",
            "operation_id": "operation-1",
            "meal_id": "meal-1",
            "native_receipt_id": "receipt-1",
            "native_receipt_validated": True,
            "cutoff_position": 5,
            "ledger_snapshot_id": "snapshot-1",
            "ledger_verified_complete": True,
            "events": events,
        }
    )
    case.persistence_evidence = persistence_evidence(case, consumed=state == "consumed")
    return case


PERSIST_NOW = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


def persistence_evidence(case, *, consumed=True, kcal=400, trace_id="ep-product"):
    from openharness.channels.bus.events import InboundMessage
    from ohmo.gateway.memory_gate import MemoryScope
    from ohmo.gateway.runtime import _build_conversation_turn_metadata
    from ohmo.gateway.turn_context import build_turn_context

    inbound = InboundMessage(channel="telegram", sender_id="owner-1|mutable_name", chat_id="chat-product",
        content="I ate this.", timestamp=PERSIST_NOW, metadata={"message_id": case.source_message_id})
    turn_context = build_turn_context(inbound, session_id="gateway-session")
    logical_turn_id, _, assistant_metadata = _build_conversation_turn_metadata(
        turn_ctx=turn_context, message=inbound, scope=MemoryScope(case.owner_id, ()))
    goal = Goal(
        case_id=case.case_id, episode_ids=[trace_id], owner_id=case.owner_id,
        principal_id="telegram:owner-1", workspace_id="workspace-1", eval_workspace="synthetic-evals",
        peer_id="ohmo", canonical_owner_id=case.owner_id, canonical_login="owner",
        session_id="honcho-session",
        gateway_session_id="gateway-session", source_message_id=case.source_message_id,
        meal_date=date(2026, 10, 1), meal_timezone="UTC",
        trajectory_started_at=PERSIST_NOW.replace(hour=11), trajectory_as_of=PERSIST_NOW,
        logical_turn_id=logical_turn_id, trace_episode_id=trace_id,
        operation_id=assistant_metadata["client_op_id"],
        canonical_meal_id=derive_meal_id(tenant_id=case.owner_id, source_principal="telegram:owner-1",
                                         gateway_session_id="gateway-session", source_message_id=case.source_message_id),
        expected_consumed=consumed,
        expected_kcal=kcal if consumed else None, expectation_origin="reviewed_user_dialogue",
        expectation_source="review:camera-product-1",
    )
    nutrition = {
        "schema_version": 2, "record_type": "meal_observation", "basis": ["user_report"],
        "consumption_status": "consumed" if consumed else "not_consumed", "meal_date": "2026-10-01",
        "energy_kcal_min": kcal if consumed else None, "energy_kcal_max": kcal if consumed else None,
        "energy_kcal_best": kcal if consumed else None,
    }
    persisted = {
        "id": "persisted-product-1", "peer_id": "ohmo", "session_id": "honcho-session",
        "workspace_id": "workspace-1", "created_at": PERSIST_NOW.isoformat(),
        "metadata": {**assistant_metadata, "tenant_id": case.owner_id, "role": "assistant",
                     "decision_trace_episode_id": trace_id,
                     "decision_trace": {"episode_id": trace_id, "annotations": {"nutrition": nutrition}}},
    }
    honcho = {"complete": True, "workspace_id": "workspace-1", "session_id": "honcho-session",
              "owner_id": case.owner_id, "since": "2026-10-01T00:00:00+00:00",
              "until": PERSIST_NOW.isoformat(), "queried_at": PERSIST_NOW.isoformat(),
              "messages": [persisted] if consumed else []}
    meals = [{"meal_id": goal.canonical_meal_id, "revision": 1, "status": "active",
              "latest_event_id": "persisted-product-1", "day": "2026-10-01", "provisional": True,
              "capture_time": PERSIST_NOW.isoformat(), "meal_at": None, "meal_date": "2026-10-01",
              "source_message_id": case.source_message_id, "ingest_source": "telegram",
              "confirmation_required": None, "reply_to_source_message_id": None, "received_at": None,
              "is_forwarded": False, "source_message_at": None, "is_estimate": True,
              "basis": ["user_report"], "consumption_status": "consumed",
              "energy_kcal_min": kcal, "energy_kcal_max": kcal, "energy_kcal_best": kcal,
              "protein_g": None, "fat_g": None, "carbohydrate_g": None, "items": [],
              "confidence": "medium", "assumptions": [], "warnings": []}] if consumed else []
    dialogue = [turn.model_dump() for turn in case.dialogue]
    export = {"privacy": "private", "episodes": [{"episode": {"episode_id": trace_id,
              "session_id": "gateway-session", "metadata": {"workspace": "synthetic-evals"}},
              "principal_id": "telegram:owner-1", "dialogue": dialogue,
              "dialogue_complete": True, "source_message_ids": [case.source_message_id],
              "turn_provenance": [{"source_message_id": case.source_message_id,
                  "logical_turn_id": logical_turn_id, "operation_id": assistant_metadata["client_op_id"],
                  "principal_id": assistant_metadata["source_principal"],
                  "episode_id": trace_id}]}]}
    telegent = {"complete": True, "user_id": case.owner_id, "login": "owner",
                "start": "2026-10-01T00:00:00+00:00", "end": PERSIST_NOW.isoformat(),
                "queried_at": PERSIST_NOW.isoformat(), "meals": meals, "unassigned": []}
    return {"goal": goal.model_dump(mode="json"), "honcho_snapshot": honcho,
            "telegent_snapshot": telegent, "dialogue_export": export}


def set_persisted_kcal(case, kcal):
    evidence = case.persistence_evidence
    evidence["honcho_snapshot"]["messages"][0]["metadata"]["decision_trace"]["annotations"]["nutrition"].update(
        energy_kcal_min=kcal, energy_kcal_max=kcal, energy_kcal_best=kcal,
    )
    evidence["telegent_snapshot"]["meals"][0]["energy_kcal_best"] = kcal


def event(payload, *, event_id="event-1", position=1, target=None):
    return {
        "event_id": event_id,
        "position": position,
        "source_message_id": "source-1",
        "owner_id": "owner-1",
        "operation_id": "operation-1",
        "meal_id": "meal-1",
        "committed": True,
        "finalizer_validated": True,
        "annotation": payload,
        "target_event_id": target,
    }


def reference(kcal=400):
    return Reference(consumption_state="consumed", kcal=kcal, uncertainty="estimated portion")


def make_judge_case(tmp_path, *, state="consumed", labels=None):
    product = make_case(tmp_path)
    return JudgeCase.model_validate(
        {
            "case_id": product.case_id,
            "image_path": product.image_path,
            "image_sha256": product.image_sha256,
            "reference_prefix": [turn.model_dump() for turn in product.prefix],
            "dialogue": [turn.model_dump() for turn in product.prefix],
            "labels": labels
            or {
                "consumption_state": state,
                "kcal_min": 350 if state == "consumed" else None,
                "kcal_max": 450 if state == "consumed" else None,
                "avoidable_turns": 1,
                "repeated_questions": 0,
            },
        }
    )


def assert_original_payload(case, payload):
    assert payload == case.image_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == case.image_sha256


def test_exif_orientation_is_retained_in_original_bytes(tmp_path):
    case = make_case(tmp_path)
    image_path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (4, 3), (20, 80, 140)).save(image_path, exif=exif)
    case.image_path = image_path
    case.image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    payload = checked_image(case)
    assert_original_payload(case, payload)
    with Image.open(io.BytesIO(payload)) as original:
        assert original.size == (4, 3)
        assert original.getexif()[274] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["product_a1", "judge_calibration"])
async def test_gps_exif_original_bytes_reach_sol_and_hashes_match(tmp_path, lane):
    case = make_case(tmp_path) if lane == "product_a1" else make_judge_case(tmp_path)
    image_path = tmp_path / "private.jpg"
    exif = Image.Exif()
    exif[34853] = {1: "N", 2: (1, 1, 1)}
    Image.new("RGB", (4, 3), (20, 80, 140)).save(image_path, exif=exif)
    original = image_path.read_bytes()
    with Image.open(io.BytesIO(original)) as source:
        assert source.getexif().get_ifd(34853)
    case.image_path = image_path
    case.image_sha256 = hashlib.sha256(original).hexdigest()
    calls = []

    async def fake(model, effort, prompt, image):
        calls.append((model, image))
        if "sol" in model:
            return reference().model_dump_json()
        assert image is None
        return A2Vote(
            score=4, avoidable_turns=1, repeated_questions=0, reason_codes=["concise"]
        ).model_dump_json()

    budget = CallBudget(max_calls=4)
    result = (
        await calibrate_case(case, fake, budget)
        if lane == "product_a1"
        else await calibrate_judge_case(case, fake, budget)
    )
    sent = calls[0][1]
    assert sent == original
    assert_original_payload(case, sent)
    with Image.open(io.BytesIO(sent)) as transmitted:
        assert transmitted.getexif().get_ifd(34853)
    assert result["source_image_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["validated_image_sha256"] == hashlib.sha256(sent).hexdigest()
    assert result["source_image_sha256"] == result["validated_image_sha256"]
    assert len(calls) == 4 and all(image is None for _, image in calls[1:])
    assert image_path.read_bytes() == original
    assert original not in json.dumps(result).encode()
    assert "GPS" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["product_a1", "judge_calibration"])
async def test_malformed_or_hash_mismatched_image_never_calls_model(tmp_path, lane):
    case = make_case(tmp_path) if lane == "product_a1" else make_judge_case(tmp_path)
    valid_image = case.image_path.read_bytes()
    calls = []

    async def fake(*args):
        calls.append(args)
        raise AssertionError("model must not be called")

    for image_bytes, expected_sha in (
        (b"not an image", hashlib.sha256(b"not an image").hexdigest()),
        (valid_image, "0" * 64),
    ):
        case.image_path.write_bytes(image_bytes)
        case.image_sha256 = expected_sha
        budget = CallBudget(max_calls=4)
        result = (
            await calibrate_case(case, fake, budget)
            if lane == "product_a1"
            else await calibrate_judge_case(case, fake, budget)
        )
        assert result["a1" if lane == "product_a1" else "reference_quality"] == "INCONCLUSIVE"
        assert calls == [] and budget.calls == 0


@pytest.mark.asyncio
async def test_private_report_path_has_restricted_mode_and_no_source_material(tmp_path):
    case = make_judge_case(tmp_path)
    case.reference_prefix[0].text = "private reference dialogue"
    case.dialogue[0].text = "private candidate dialogue"

    async def fake(model, effort, prompt, image):
        if "sol" in model:
            return reference().model_dump_json()
        return A2Vote(
            score=4, avoidable_turns=1, repeated_questions=0, reason_codes=["concise"]
        ).model_dump_json()

    result = await calibrate_judge_case(case, fake, CallBudget(max_calls=4))
    report = {
        "lane": "JUDGE_CALIBRATION",
        "cases": [result],
    }
    path = tmp_path / "evals" / "reports" / "camera_calibration.json"
    assert not path.parent.exists()

    write_report(path, report)

    assert path.parent.is_dir()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text()) == report
    assert case.image_path.read_bytes() not in path.read_bytes()
    assert case.reference_prefix[0].text not in path.read_text()
    assert case.dialogue[0].text not in path.read_text()


@pytest.mark.asyncio
async def test_product_a1_does_not_accept_legacy_commit_booleans_without_persisted_evidence(tmp_path):
    case = make_case(tmp_path)
    case.persistence_evidence = None
    calls = []

    async def fake(model, effort, prompt, image):
        calls.append(model)
        return reference().model_dump_json()

    result = await calibrate_case(case, fake, CallBudget(max_calls=4))
    assert result["a1"] == "INCONCLUSIVE"
    assert result["a2"] == "NOT_RUN"
    assert calls == ["openai/gpt-6.1-sol"]


def test_product_a2_refuses_to_fall_back_to_reference_prefix(tmp_path):
    case = make_case(tmp_path)
    case.dialogue = None
    with pytest.raises(ValueError, match="full dialogue"):
        a2_prompt(case)


@pytest.mark.parametrize("missing", ["reference_prefix", "dialogue"])
def test_judge_requires_both_conversation_fields(tmp_path, missing):
    data = make_judge_case(tmp_path).model_dump()
    del data[missing]
    with pytest.raises(ValidationError, match=missing):
        JudgeCase.model_validate(data)


@pytest.mark.parametrize("lane", ["product_a1", "judge_calibration"])
def test_sol_prompt_scopes_nonfood_to_attached_image_without_erasing_food_uncertainty(
    tmp_path, lane
):
    case = make_case(tmp_path) if lane == "product_a1" else make_judge_case(tmp_path)
    prompt = sol_prompt(case)

    assert "food shown in the attached image and this selected trajectory" in prompt
    assert "not whether the person ate unrelated food elsewhere" in prompt
    assert "image clearly contains no food, return not_consumed" in prompt
    assert "kcal null and estimated_kcal null" in prompt
    assert "without claiming the person ate nothing elsewhere" in prompt
    assert "image contains food but whether it was eaten is unclear" in prompt
    assert "return uncertain with kcal null" in prompt
    assert "supportable photo estimate may still be provided" in prompt
    if lane == "judge_calibration":
        assert "A request to estimate calories does not establish" in prompt
        assert "For food shown in the image, keep consumption_state uncertain" in prompt
    else:
        assert "If the image contains food and consumption or amount is unclear" in prompt


@pytest.mark.asyncio
async def test_judge_reference_is_blind_to_candidate_answer_and_luna_sees_dialogue(tmp_path):
    case = make_judge_case(tmp_path)
    candidate_answer = "Candidate answer: 913 kcal."
    case.dialogue.append(type(case.dialogue[0])(role="assistant", text=candidate_answer))
    calls = []

    async def fake(model, effort, prompt, image):
        calls.append((model, prompt, image))
        if "sol" in model:
            return reference().model_dump_json()
        return A2Vote(
            score=4, avoidable_turns=1, repeated_questions=0, reason_codes=["concise"]
        ).model_dump_json()

    await calibrate_judge_case(case, fake, CallBudget(max_calls=4))
    assert len(calls) == 4
    assert candidate_answer not in calls[0][1]
    assert calls[0][2] is not None
    assert_original_payload(case, calls[0][2])
    assert all(candidate_answer in prompt and image is None for _, prompt, image in calls[1:])


def test_positive_inclusive_boundary_and_over(tmp_path):
    case = make_case(tmp_path)
    assert score_a1(case, reference())[0] == "PASS"
    set_persisted_kcal(case, 440.1)
    assert score_a1(case, reference())[0] == "FAIL"
    set_persisted_kcal(case, 360)
    assert score_a1(case, reference())[0] == "PASS"
    set_persisted_kcal(case, 359.9)
    assert score_a1(case, reference())[0] == "FAIL"


def test_negative_exact_absence_and_retraction(tmp_path):
    negative = Reference(consumption_state="not_consumed", kcal=None, uncertainty="nonfood")
    assert score_a1(make_case(tmp_path, state="never_recorded", events=[]), negative)[0] == "PASS"
    observed = event(
        annotation("meal_observation", consumption_status="consumed", energy_kcal_best=400)
    )
    correction = event(
        annotation(
            "meal_correction",
            changed_fields=["consumption_status"],
            consumption_status="not_consumed",
        ),
        event_id="event-2",
        position=2,
        target="event-1",
    )
    case = make_case(tmp_path, state="validly_retracted", events=[observed, correction])
    assert effective_meal(case) == ("validly_retracted", None)
    case.persistence_evidence = persistence_evidence(case, consumed=False)
    assert score_a1(case, negative)[0] == "PASS"
    unexpected = make_case(tmp_path, state="consumed")
    assert score_a1(unexpected, negative)[0] == "FAIL"


def test_replay_dedup_and_authoritative_order(tmp_path):
    observed = event(
        annotation("meal_observation", consumption_status="consumed", energy_kcal_best=400),
        event_id="z-last",
        position=1,
    )
    correction = event(
        annotation("meal_correction", changed_fields=["energy_kcal_best"], energy_kcal_best=440),
        event_id="a-first",
        position=2,
        target="z-last",
    )
    case = make_case(tmp_path, events=[correction, observed, observed])
    assert effective_meal(case) == ("consumed", 440)
    set_persisted_kcal(case, 440)
    assert score_a1(case, reference())[0] == "PASS"
    case.persistence_evidence["honcho_snapshot"]["complete"] = False
    assert score_a1(case, reference())[0] == "INCONCLUSIVE"


@pytest.mark.parametrize(
    "break_case",
    [
        lambda c: c.persistence_evidence["honcho_snapshot"].update(complete=False),
        lambda c: c.persistence_evidence["goal"].update(operation_id="foreign"),
        lambda c: c.persistence_evidence["goal"].update(source_message_id="foreign"),
        lambda c: c.persistence_evidence["goal"].update(owner_id="foreign"),
    ],
)
def test_persistence_evidence_incomplete_or_unbound_fails_closed(tmp_path, break_case):
    case = make_case(tmp_path)
    break_case(case)
    assert score_a1(case, reference())[0] == "INCONCLUSIVE"


@pytest.mark.asyncio
async def test_a1_gates_a2_and_a2_prompt_is_blind(tmp_path):
    calls = []

    async def fake(model, effort, prompt, image):
        calls.append((model, effort, prompt, image))
        if "sol" in model:
            return reference().model_dump_json()
        return A2Vote(
            score=4,
            avoidable_turns=1,
            repeated_questions=0,
            reason_codes=["necessary_clarification"],
        ).model_dump_json()

    failed = make_case(tmp_path)
    set_persisted_kcal(failed, 500)
    result = await calibrate_case(failed, fake, CallBudget(max_calls=4))
    assert result["a1"] == "FAIL" and result["a2"] == "NOT_RUN"
    assert len(calls) == 1
    calls.clear()
    passed = make_case(tmp_path)
    result = await calibrate_case(passed, fake, CallBudget(max_calls=4))
    assert result["a1"] == "PASS" and result["a2"] == "SCORED"
    assert len(calls) == 4
    for model, effort, prompt, image in calls[1:]:
        assert model == "openai/gpt-6-luna" and effort == "medium" and image is None
        assert "400" not in prompt and "private-1" not in prompt
        assert "PASS" not in prompt and "gpt-6-sol" not in prompt.lower()
    assert calls[0][0] == "openai/gpt-6.1-sol" and calls[0][1] == "high"
    assert calls[0][3] is not None
    assert_original_payload(passed, calls[0][3])
    assert "private-1" not in a2_prompt(passed)


def test_full_product_dialogue_is_for_a2_while_sol_uses_reference_prefix(tmp_path):
    case = make_case(tmp_path)
    final = {"role": "assistant", "text": "I could not confirm the journal save."}
    duplicate = {"role": "assistant", "text": "Was there sugar?"}
    full = Case.model_validate({
        **case.model_dump(), "dialogue": [*case.prefix, duplicate, final],
    })
    assert final["text"] in a2_prompt(full)
    assert duplicate["text"] in a2_prompt(full)
    assert final["text"] not in sol_prompt(full)


@pytest.mark.asyncio
async def test_missing_model_output_stops_without_luna(tmp_path):
    calls = []

    async def blank(*args):
        calls.append(args)
        return ""

    result = await calibrate_case(make_case(tmp_path), blank, CallBudget(max_calls=4))
    assert result["a1"] == "INCONCLUSIVE" and result["a2"] == "NOT_RUN"
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,kcal",
    [
        ("consumed", 400),
        ("not_consumed", None),
        ("uncertain", None),
    ],
)
async def test_judge_lane_without_events_and_human_efficiency(tmp_path, state, kcal):
    from ohmo.evals.camera_calibration import calibrate_judge_case

    calls = []

    async def fake(model, effort, prompt, image):
        calls.append((model, effort, prompt, image))
        if "sol" in model:
            return Reference(
                consumption_state=state, kcal=kcal, uncertainty="I ate this. What was the amount?"
            ).model_dump_json()
        return A2Vote(
            score=4, avoidable_turns=1, repeated_questions=0, reason_codes=["concise"]
        ).model_dump_json()

    case = make_judge_case(tmp_path, state=state)
    result = await calibrate_judge_case(case, fake, CallBudget(max_calls=4))
    assert result["lane"] == "JUDGE_CALIBRATION"
    assert result["reference_quality"] == "PASS"
    assert result["efficiency_quality"] == "PASS"
    assert "a1" not in result
    assert len(calls) == 4
    assert calls[0][0:2] == ("openai/gpt-6.1-sol", "high")
    assert calls[0][3] is not None
    assert_original_payload(case, calls[0][3])
    assert all(
        model == "openai/gpt-6-luna" and effort == "medium" and image is None
        for model, effort, _, image in calls[1:]
    )
    assert "I ate this" not in str(result)


@pytest.mark.asyncio
async def test_judge_sol_mismatch_blocks_luna(tmp_path):
    from ohmo.evals.camera_calibration import calibrate_judge_case

    calls = []

    async def fake(*args):
        calls.append(args)
        return Reference(
            consumption_state="not_consumed", kcal=None, uncertainty="no meal"
        ).model_dump_json()

    result = await calibrate_judge_case(make_judge_case(tmp_path), fake, CallBudget(max_calls=4))
    assert result["reference_quality"] == "FAIL"
    assert result["efficiency_quality"] == "NOT_RUN"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_judge_invalid_reference_is_inconclusive(tmp_path):
    from ohmo.evals.camera_calibration import calibrate_judge_case

    calls = []

    async def invalid(*args):
        calls.append(args)
        return '{"consumption_state":"consumed","kcal":null,"uncertainty":"unknown"}'

    result = await calibrate_judge_case(
        make_judge_case(tmp_path),
        invalid,
        CallBudget(max_calls=4),
    )
    assert result["reference_quality"] == "INCONCLUSIVE"
    assert result["efficiency_quality"] == "NOT_RUN"
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,estimated_kcal,expected_reference,expected_estimate,expected_luna",
    [
        ("uncertain", 400, "PASS", "PASS", 3),
        ("consumed", 400, "FAIL", "PASS", 0),
        ("uncertain", 600, "FAIL", "FAIL", 0),
        ("uncertain", None, "FAIL", "FAIL", 0),
    ],
)
async def test_estimate_only_does_not_imply_consumption(
    tmp_path, state, estimated_kcal, expected_reference, expected_estimate, expected_luna
):
    from ohmo.evals.camera_calibration import calibrate_judge_case

    case = make_judge_case(
        tmp_path,
        state="uncertain",
        labels={
            "consumption_state": "uncertain",
            "estimate_kcal_min": 350,
            "estimate_kcal_max": 450,
            "avoidable_turns": 1,
            "repeated_questions": 0,
        },
    )
    case.reference_prefix[0].text = "Estimate the calories in this photo."
    case.dialogue[0].text = "Estimate the calories in this photo."
    calls = []

    async def fake(model, effort, prompt, image):
        calls.append((model, prompt, image))
        if "sol" in model:
            return Reference(
                consumption_state=state,
                kcal=400 if state == "consumed" else None,
                estimated_kcal=estimated_kcal,
                uncertainty="not stated whether eaten",
            ).model_dump_json()
        return A2Vote(
            score=4, avoidable_turns=1, repeated_questions=0, reason_codes=["concise"]
        ).model_dump_json()

    result = await calibrate_judge_case(case, fake, CallBudget(max_calls=4))
    assert result["reference_quality"] == expected_reference
    assert result["estimate_quality"] == expected_estimate
    assert len(calls) == 1 + expected_luna
    assert "does not establish" in calls[0][1]
    assert result["reference"]["kcal"] == (400 if state == "consumed" else None)
    assert "a1" not in result


@pytest.mark.asyncio
async def test_judge_luna_aggregate_fails_human_tolerance(tmp_path):
    from ohmo.evals.camera_calibration import calibrate_judge_case

    votes = iter([0, 2, 2])

    async def fake(model, effort, prompt, image):
        if "sol" in model:
            return reference().model_dump_json()
        return A2Vote(
            score=3, avoidable_turns=next(votes), repeated_questions=1, reason_codes=["avoidable"]
        ).model_dump_json()

    result = await calibrate_judge_case(make_judge_case(tmp_path), fake, CallBudget(max_calls=4))
    assert result["reference_quality"] == "PASS"
    assert result["efficiency_quality"] == "FAIL"
    assert result["a2_avoidable_turns"] == 2


@pytest.mark.asyncio
async def test_judge_repeated_questions_uses_conservative_max(tmp_path):
    votes = iter([0, 0, 1])

    async def fake(model, effort, prompt, image):
        if "sol" in model:
            return reference().model_dump_json()
        return A2Vote(
            score=4,
            avoidable_turns=1,
            repeated_questions=next(votes),
            reason_codes=["repeat"],
        ).model_dump_json()

    result = await calibrate_judge_case(make_judge_case(tmp_path), fake, CallBudget(max_calls=4))
    assert result["a2_repeated_questions"] == 1
    assert result["efficiency_quality"] == "FAIL"


@pytest.mark.asyncio
async def test_budget_exhaustion_propagates(tmp_path):
    from ohmo.evals.camera_calibration import calibrate_judge_case

    async def fake(model, effort, prompt, image):
        if "sol" in model:
            return reference().model_dump_json()
        return A2Vote(
            score=3, avoidable_turns=1, repeated_questions=0, reason_codes=["concise"]
        ).model_dump_json()

    with pytest.raises(ValueError, match="model call cap exhausted"):
        await calibrate_judge_case(
            make_judge_case(tmp_path),
            fake,
            CallBudget(max_calls=1),
        )


@pytest.mark.asyncio
async def test_programming_error_propagates(tmp_path):
    from ohmo.evals.camera_calibration import calibrate_judge_case

    async def broken(*args):
        raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        await calibrate_judge_case(
            make_judge_case(tmp_path),
            broken,
            CallBudget(max_calls=4),
        )


def subscription_entries(case, *, luna=True):
    entries = [
        {
            "route": "native_subscription_padavan",
            "case_id": case.case_id,
            "model": "openai/gpt-6.1-sol",
            "reasoning_effort": "high",
            "prompt": sol_prompt(case),
            "source_image_sha256": case.image_sha256,
            "padavan_session_id": "sol-session",
            "padavan_turn_id": "turn-1",
            "response_json": reference().model_dump_json(),
        }
    ]
    if luna:
        for index in range(3):
            entries.append(
                {
                    "route": "native_subscription_padavan",
                    "case_id": case.case_id,
                    "model": "openai/gpt-6-luna",
                    "reasoning_effort": "medium",
                    "prompt": a2_prompt(case),
                    "source_image_sha256": None,
                    "padavan_session_id": f"luna-session-{index}",
                    "padavan_turn_id": "turn-1",
                    "response_json": A2Vote(
                        score=4,
                        avoidable_turns=1,
                        repeated_questions=0,
                        reason_codes=["concise"],
                    ).model_dump_json(),
                }
            )
    return entries


@pytest.mark.asyncio
async def test_subscription_intake_exact_route_and_exif_bytes(tmp_path):
    case = make_case(tmp_path)
    image_path = tmp_path / "private.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (4, 3), (20, 80, 140)).save(image_path, exif=exif)
    original = image_path.read_bytes()
    case.image_path = image_path
    case.image_sha256 = hashlib.sha256(original).hexdigest()
    entries = subscription_entries(case)
    intake = SubscriptionResults(entries, [case])
    result = await calibrate_case(case, intake.call, CallBudget(max_calls=4))
    intake.assert_exhausted()
    assert result["a1"] == "PASS" and result["a2"] == "SCORED"
    assert result["validated_image_sha256"] == case.image_sha256
    assert image_path.read_bytes() == original
    with Image.open(io.BytesIO(original)) as image:
        assert image.getexif()[274] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("route", "direct_api"),
        ("model", "openai/gpt-6-luna"),
        ("model", "openai/gpt-6-sol"),
        ("reasoning_effort", "medium"),
        ("prompt", "different prompt"),
        ("source_image_sha256", "0" * 64),
        ("case_id", "other-case"),
    ],
)
async def test_subscription_rejects_wrong_provenance(tmp_path, field, value):
    case = make_case(tmp_path)
    entries = subscription_entries(case)
    entries[0][field] = value
    if field in {"route", "model"}:
        with pytest.raises(ValueError, match="schema invalid"):
            SubscriptionResults(entries, [case])
    else:
        intake = SubscriptionResults(entries, [case])
        with pytest.raises(ValueError, match="exact requested call"):
            await calibrate_case(case, intake.call, CallBudget(max_calls=4))


@pytest.mark.asyncio
async def test_subscription_rejects_missing_extra_and_reused_evidence(tmp_path):
    case = make_case(tmp_path)
    entries = subscription_entries(case)
    with pytest.raises(ValueError, match="reused Padavan"):
        reused = [dict(item) for item in entries]
        reused[2]["padavan_session_id"] = reused[1]["padavan_session_id"]
        SubscriptionResults(reused, [case])
    same_session = [dict(item) for item in entries]
    same_session[2]["padavan_session_id"] = same_session[1]["padavan_session_id"]
    same_session[2]["padavan_turn_id"] = "turn-2"
    intake = SubscriptionResults(same_session, [case])
    with pytest.raises(ValueError, match="distinct Padavan sessions"):
        await calibrate_case(case, intake.call, CallBudget(max_calls=4))
    intake = SubscriptionResults(entries[:3], [case])
    with pytest.raises(ValueError, match="missing subscription result"):
        await calibrate_case(case, intake.call, CallBudget(max_calls=4))
    failed = make_case(tmp_path)
    set_persisted_kcal(failed, 500)
    intake = SubscriptionResults(entries, [failed])
    result = await calibrate_case(failed, intake.call, CallBudget(max_calls=4))
    assert result["a1"] == "FAIL" and result["a2"] == "NOT_RUN"
    with pytest.raises(ValueError, match="extra unused"):
        intake.assert_exhausted()


def test_private_subscription_file_mode_and_location(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="mode 0600"):
        read_private_json(path)
    path.chmod(0o600)
    assert read_private_json(path) == []
    with pytest.raises(ValueError, match="outside the repository"):
        read_private_json(Path(__file__))


@pytest.mark.parametrize(
    "change",
    [
        lambda item: item.pop("padavan_turn_id"),
        lambda item: item.update(unexpected="extra"),
        lambda item: item.update(response_json="{}"),
    ],
)
def test_subscription_rejects_incomplete_or_invalid_entry(tmp_path, change):
    case = make_case(tmp_path)
    entries = subscription_entries(case)
    change(entries[0])
    with pytest.raises(ValueError, match="schema invalid"):
        SubscriptionResults(entries, [case])


def test_subscription_cli_writes_only_aggregate_report(tmp_path, monkeypatch):
    from ohmo.evals.adapter import get_eval_store

    cases = []
    entries = []
    for index in range(3):
        case = make_case(tmp_path)
        case.case_id = f"case-{index}"
        cases.append(case)
        for item in subscription_entries(case):
            item["padavan_session_id"] += f"-{index}"
            entries.append(item)
    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.json"
    manifest_path.write_text(json.dumps([case.model_dump(mode="json") for case in cases]))
    results_path.write_text(json.dumps(entries))
    manifest_path.chmod(0o600)
    results_path.chmod(0o600)
    workspace = tmp_path / "private-workspace"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "camera_calibration",
            str(manifest_path),
            "--subscription-results",
            str(results_path),
            "--workspace",
            str(workspace),
        ],
    )
    main()
    path = get_eval_store(workspace).root / "reports" / "camera_calibration.json"
    report = json.loads(path.read_text())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert report["subscription_turns"] == 12
    assert report["reference_model"] == "openai/gpt-6.1-sol"
    assert report["reference_reasoning_effort"] == "high"
    assert report["reference_selection"] == "human_authorized_temporary"
    assert report["sol_prompt_version"] == SOL_PROMPT_VERSION == "image_target_v2"
    assert report["actual_usd"] == "unknown"
    assert all(item["a1"] == "PASS" for item in report["cases"])
    assert '"prompt":' not in path.read_text()
    assert "padavan_session_id" not in path.read_text()
    assert "estimated portion" not in path.read_text()
