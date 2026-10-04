"""Small, private Camera grader calibration; never a release gate.

The caller supplies manually reviewed manifests and authoritative ledger evidence.
No fixture images, dialogue, or model replies are written to the report.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
import re
import warnings
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ohmo.evals.adapter import get_eval_store
from ohmo.evals.nutrition_trace import NutritionAnnotationV2
from openharness.utils.fs import atomic_write_text

SOL_MODEL = "openai/gpt-6.1-sol"
LUNA_MODEL = "openai/gpt-6-luna"
SOL_PROMPT_VERSION = "image_target_v2"
MAX_CASES = 5
MAX_PREFIX = 12
MAX_DIALOGUE = 100
MAX_EVENTS = 24
MAX_IMAGE_BYTES = 10_000_000
MAX_IMAGE_PIXELS = 20_000_000


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Turn(StrictModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=4000)


class CommitEvent(StrictModel):
    event_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    source_message_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    meal_id: str = Field(min_length=1)
    committed: bool
    finalizer_validated: bool
    annotation: dict[str, Any]
    target_event_id: str | None = None


class Case(StrictModel):
    case_id: str = Field(min_length=1)
    image_path: Path
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prefix: list[Turn] = Field(min_length=1, max_length=MAX_PREFIX)
    # Complete product dialogue for A2; reference judging remains prefix-only.
    dialogue: list[Turn] | None = Field(default=None, max_length=MAX_DIALOGUE)
    # Bound raw snapshots and the read-only eval export used by product A1.
    persistence_evidence: dict[str, Any] | None = None
    reviewed_state: Literal["consumed", "never_recorded", "validly_retracted"]
    origin: Literal["camera", "person"]
    source_message_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    meal_id: str = Field(min_length=1)
    native_receipt_id: str | None = None
    native_receipt_validated: bool = False
    cutoff_position: int = Field(ge=0)
    ledger_snapshot_id: str = Field(min_length=1)
    ledger_verified_complete: bool
    events: list[CommitEvent] = Field(max_length=MAX_EVENTS)

    @model_validator(mode="after")
    def receipt_required(self) -> Case:
        if self.origin == "camera" and (
            not self.native_receipt_id or not self.native_receipt_validated
        ):
            raise ValueError("Camera source requires a validated native receipt")
        return self


class JudgeLabels(StrictModel):
    consumption_state: Literal["consumed", "not_consumed", "uncertain"]
    kcal_min: float | None = None
    kcal_max: float | None = None
    estimate_kcal_min: float | None = None
    estimate_kcal_max: float | None = None
    avoidable_turns: int = Field(ge=0, le=MAX_DIALOGUE)
    repeated_questions: int = Field(ge=0, le=MAX_DIALOGUE)
    avoidable_tolerance: int = Field(default=0, ge=0, le=MAX_PREFIX)
    repeated_tolerance: int = Field(default=0, ge=0, le=MAX_PREFIX)

    @model_validator(mode="after")
    def valid_interval(self) -> JudgeLabels:
        if self.consumption_state == "consumed":
            if (
                self.kcal_min is None
                or self.kcal_max is None
                or not math.isfinite(self.kcal_min)
                or not math.isfinite(self.kcal_max)
                or self.kcal_min <= 0
                or self.kcal_max < self.kcal_min
            ):
                raise ValueError("consumed label requires a positive finite kcal interval")
        elif self.kcal_min is not None or self.kcal_max is not None:
            raise ValueError("non-consumed or uncertain label must omit kcal interval")
        if (self.estimate_kcal_min is None) != (self.estimate_kcal_max is None):
            raise ValueError("estimate interval requires both endpoints")
        if self.estimate_kcal_min is not None:
            assert self.estimate_kcal_max is not None
            if (
                not math.isfinite(self.estimate_kcal_min)
                or not math.isfinite(self.estimate_kcal_max)
                or self.estimate_kcal_min <= 0
                or self.estimate_kcal_max < self.estimate_kcal_min
            ):
                raise ValueError("estimate interval must be positive and finite")
        return self


class JudgeCase(StrictModel):
    case_id: str = Field(min_length=1)
    image_path: Path
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_prefix: list[Turn] = Field(min_length=1, max_length=MAX_PREFIX)
    dialogue: list[Turn] = Field(min_length=1, max_length=MAX_DIALOGUE)
    labels: JudgeLabels


class Reference(StrictModel):
    consumption_state: Literal["consumed", "not_consumed", "uncertain"]
    kcal: float | None
    estimated_kcal: float | None = None
    uncertainty: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def valid_kcal(self) -> Reference:
        if self.consumption_state == "consumed":
            if self.kcal is None or not math.isfinite(self.kcal) or self.kcal <= 0:
                raise ValueError("consumed reference requires positive finite kcal")
        elif self.kcal is not None:
            raise ValueError("negative or uncertain reference must have null kcal")
        if self.estimated_kcal is not None and (
            not math.isfinite(self.estimated_kcal) or self.estimated_kcal <= 0
        ):
            raise ValueError("photo estimate must be positive finite kcal")
        return self


class A2Vote(StrictModel):
    score: int = Field(ge=0, le=5)
    avoidable_turns: int = Field(ge=0, le=MAX_DIALOGUE)
    repeated_questions: int = Field(ge=0, le=MAX_DIALOGUE)
    reason_codes: list[Literal["repeat", "avoidable", "necessary_clarification", "concise"]]
    useful_button_click: bool = False


class CallBudget:
    def __init__(self, *, max_calls: int):
        if not 1 <= max_calls <= MAX_CASES * 4:
            raise ValueError("max_calls must be 1..20")
        self.max_calls = max_calls
        self.calls = 0

    def reserve(self) -> None:
        if self.calls >= self.max_calls:
            raise ValueError("model call cap exhausted")
        self.calls += 1


# A request is deliberately plain data so fake clients can inspect exact prompt contents.
ModelCall = Callable[[str, str, str, bytes | None], Awaitable[str]]


def checked_image(case: Case | JudgeCase) -> bytes:
    with case.image_path.open("rb") as image_file:
        data = image_file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("original image exceeds 10 MB cap")
    if not data or hashlib.sha256(data).hexdigest() != case.image_sha256:
        raise ValueError("original image missing or SHA-256 mismatch")
    if case.image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("unsupported original image type")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("unsupported decoded image type")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("multi-frame image is unsupported")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("image exceeds pixel cap")
                source.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning) as exc:
        raise ValueError("image decoding or validation failed") from exc
    return data


def effective_meal(case: Case) -> tuple[str, float | None]:
    """Fold validated operation-bound events in authoritative commit position order."""
    if not case.ledger_verified_complete:
        raise ValueError("authoritative ledger completeness is unverified")
    unique: dict[str, CommitEvent] = {}
    for event in case.events:
        if (
            not event.committed
            or not event.finalizer_validated
            or event.position > case.cutoff_position
            or event.source_message_id != case.source_message_id
            or event.owner_id != case.owner_id
            or event.operation_id != case.operation_id
            or event.meal_id != case.meal_id
        ):
            raise ValueError("uncertain commit or source/operation/meal binding")
        old = unique.setdefault(event.event_id, event)
        if old != event:
            raise ValueError("conflicting replay of event ID")
    events = sorted(unique.values(), key=lambda event: event.position)
    if len({event.position for event in events}) != len(events):
        raise ValueError("unresolved authoritative position tie")
    observation_id: str | None = None
    status: str | None = None
    kcal: float | None = None
    retracted = False
    for event in events:
        annotation = NutritionAnnotationV2.model_validate(event.annotation)
        if annotation.record_type == "meal_observation":
            if observation_id is not None:
                raise ValueError("multiple observations for one meal")
            if event.target_event_id is not None:
                raise ValueError("observation must not target an event")
            observation_id = event.event_id
            status = annotation.consumption_status
            kcal = annotation.energy_kcal_best
            if status == "consumed" and (kcal is None or kcal <= 0):
                raise ValueError("consumed observation needs best kcal")
        elif annotation.record_type in {"meal_correction", "meal_deletion"}:
            if observation_id is None or event.target_event_id != observation_id:
                raise ValueError("correction/deletion target is unbound")
            if annotation.record_type == "meal_deletion":
                status, kcal, retracted = "not_consumed", None, True
            else:
                if status == "not_consumed":
                    raise ValueError("correction after retraction is unresolved")
                if "consumption_status" in annotation.changed_fields:
                    status = annotation.consumption_status
                    if status == "not_consumed":
                        kcal, retracted = None, True
                if "energy_kcal_best" in annotation.changed_fields:
                    kcal = annotation.energy_kcal_best
        else:
            raise ValueError("day summary is not a meal commit")
    if observation_id is None:
        return "never_recorded", None
    if retracted:
        return "validly_retracted", None
    if status == "consumed" and kcal is not None and kcal > 0:
        return "consumed", kcal
    raise ValueError("effective meal state or kcal is unresolved")


def _score_a1_details(case: Case, reference: Reference) -> dict[str, Any]:
    """Require raw persisted evidence; legacy Case commit booleans are historical only."""
    evidence = case.persistence_evidence
    if not isinstance(evidence, dict):
        return {"a1": "INCONCLUSIVE", "reason": "scoped Honcho and Telegent persistence evidence is missing",
                "persistence_stage": "EVIDENCE_MISSING"}
    if reference.consumption_state == "uncertain":
        return {"a1": "INCONCLUSIVE", "reason": "frozen reference is uncertain",
                "persistence_stage": "REFERENCE_UNCERTAIN"}
    try:
        from ohmo.evals.nutrition_persistence import (
            Goal, Manifest, grade_manifest, validate_dialogue_binding,
        )

        goal = Goal.model_validate(evidence["goal"])
        manifest = Manifest(schema_version=1, goals=[goal])
        bound = validate_dialogue_binding(manifest, evidence["dialogue_export"])
        if bound[goal.case_id]["complete"] is not True:
            return {"a1": "INCONCLUSIVE", "reason": "reviewed goal is not bound to complete owner dialogue",
                    "persistence_stage": "DIALOGUE_BINDING_FAILED"}
        expected_turns = []
        for exported in evidence["dialogue_export"]["episodes"]:
            if exported.get("episode", {}).get("episode_id") in goal.episode_ids:
                expected_turns.extend(
                    {"role": turn["role"], "text": turn["text"]}
                    for turn in exported.get("dialogue", []) if turn.get("role") in {"user", "assistant"}
                )
        if case.dialogue is None or [turn.model_dump() for turn in case.dialogue] != expected_turns:
            return {"a1": "INCONCLUSIVE", "reason": "full Case dialogue is missing or differs from bound export",
                    "persistence_stage": "DIALOGUE_BINDING_FAILED"}
        if goal.source_message_id != case.source_message_id or goal.owner_id != case.owner_id:
            return {"a1": "INCONCLUSIVE", "reason": "reviewed nutrition goal does not bind to product Case identity",
                    "persistence_stage": "CASE_BINDING_FAILED"}
        telegent = evidence["telegent_snapshot"]
        from ohmo.evals.nutrition_persistence import bind_wellness_snapshot

        canonical = bind_wellness_snapshot(telegent, goal=goal)
        result = grade_manifest(manifest, evidence["honcho_snapshot"], canonical,
                                reviewed_turn_sources=bound[goal.case_id]["reviewed_turn_sources"],
                                reviewed_turn_provenance=bound[goal.case_id]["reviewed_turn_provenance"])[0]
    except (KeyError, TypeError, ValueError, AttributeError, ValidationError):
        return {"a1": "INCONCLUSIVE", "reason": "persistence evidence is malformed or ambiguously bound",
                "persistence_stage": "EVIDENCE_INVALID"}
    details = {"persistence_stage": result.get("stage", "UNKNOWN")}
    details.update({key: result[key] for key in ("actual_event_ids", "actual_latest_event_id",
                                                  "actual_kcal", "evidence_limitations") if key in result})
    details["persistence_reason"] = result.get("reason", "")
    if result["a1"] != "PASS":
        return {"a1": result["a1"], "reason": result["reason"], **details}
    expected_state = "consumed" if goal.expected_consumed else "not_consumed"
    if reference.consumption_state != expected_state:
        return {"a1": "FAIL", "reason": "frozen Sol reference disagrees with reviewed persisted goal", **details}
    if goal.expected_consumed:
        assert reference.kcal is not None
        actual_kcal = result.get("actual_kcal")
        if not isinstance(actual_kcal, (int, float)) or isinstance(actual_kcal, bool):
            return {"a1": "INCONCLUSIVE", "reason": "persisted numeric meal value is unavailable", **details}
        if abs(actual_kcal - reference.kcal) > reference.kcal * goal.tolerance_fraction + 1e-9:
            return {"a1": "FAIL", "reason": "persisted kcal exceeds frozen Sol reference tolerance", **details}
    return {"a1": "PASS", "reason": "persisted same-event meal matches reviewed goal and frozen reference", **details}


def score_a1(case: Case, reference: Reference) -> tuple[str, str]:
    result = _score_a1_details(case, reference)
    return result["a1"], result["reason"]


def sol_prompt(case: Case | JudgeCase) -> str:
    turns = case.reference_prefix if isinstance(case, JudgeCase) else case.prefix
    prefix = [{"role": turn.role, "text": turn.text} for turn in turns]
    if isinstance(case, JudgeCase):
        target_scope = (
            "Judge consumption_state and kcal for the food shown in the attached image and "
            "this selected trajectory, not whether the person ate unrelated food elsewhere. "
            "If the image clearly contains no food, return not_consumed with kcal null and "
            "estimated_kcal null: no meal from this image, without claiming the person ate "
            "nothing elsewhere. If the image contains food but whether it was eaten is unclear, "
            "return uncertain with kcal null; a supportable photo estimate may still be provided "
            "as estimated_kcal. "
        )
        return (
            "Use only this image and dialogue prefix. "
            + target_scope
            + "Return one JSON object with "
            "consumption_state (consumed|not_consumed|uncertain), kcal (positive number "
            "only for a confirmed consumed meal, else null), estimated_kcal (positive "
            "photo calorie estimate when supportable, else null), and uncertainty "
            "(brief explanation). A request to estimate calories does not establish that "
            "the person ate the food. For food shown in the image, keep consumption_state "
            "uncertain if the dialogue does not resolve whether it was eaten; an estimated_kcal "
            "may still be provided. Do not infer "
            "a meal record from an estimate. Prefix: " + json.dumps(prefix, ensure_ascii=False)
        )
    target_scope = (
        "Judge the expected policy meal state and kcal for the food shown in this image and "
        "selected trajectory. Here consumed means the meal state expected by the selected "
        "source policy; it does not assert physical ingestion or that a database record was saved. "
        "If the image clearly contains no food, return not_consumed with kcal null: there is no "
        "meal from this image, without claiming the person ate nothing elsewhere. Explicitly "
        "stated denial, analysis-only or informational context, and recipe requests override a "
        "person-photo meal default and mean not_consumed. For origin=person, clear food with a "
        "supportable visible portion or known unit defaults to consumed without separate eating "
        "confirmation; estimate kcal from the visible amount and state calorie uncertainty "
        "honestly. Do not ask for exact grams or a nutrition label when a useful estimate exists. "
        "An explicit partial amount overrides a whole-unit default. Unknown exact grams alone do "
        "not force uncertainty when a reasonable visible-portion or known-unit estimate exists. "
        "For origin=camera, food pixels and a native delivery receipt alone never confirm "
        "consumption: require a meaningful owner answer in this curated prefix, bound to this "
        "image, including any stated unit or partial amount. Without that answer, return uncertain, "
        "not not_consumed. For either origin, unclear food or a genuinely unsupported meaningful "
        "amount remains uncertain pending useful clarification. "
    )
    return (
        "Use only this original image, trusted Case.origin, and curated dialogue prefix. "
        f"Selected trusted source: origin={case.origin}. "
        + target_scope + "Return one JSON object with consumption_state "
        "(consumed|not_consumed|uncertain), kcal (positive number only if consumed, else null), "
        "and uncertainty (brief explanation). Do not use reviewed state, persistence evidence, "
        "expected goal/kcal, candidate dialogue, answer, receipt, or result to select the reference. "
        "Prefix: " + json.dumps(prefix, ensure_ascii=False)
    )


def a2_prompt(case: Case | JudgeCase) -> str:
    turns = case.dialogue
    if turns is None:
        raise ValueError("full dialogue is required before A2 scoring")
    dialogue = [{"role": turn.role, "text": turn.text} for turn in turns]
    click_evidence = _button_click_evidence(case)
    prompt = (
        "Judge only the efficiency and user friction of the complete public dialogue. "
        "Return JSON with score (integer 0..5), avoidable_turns (integer), "
        "repeated_questions (integer), reason_codes (repeat|avoidable|necessary_clarification|concise), "
        "and useful_button_click (boolean). Count removable public assistant messages or questions "
        "as avoidable_turns, including technical progress narration that adds no useful information, "
        "a separate duplicate saved acknowledgement, duplicate questions, and asking for precise grams "
        "again when a useful known-unit or visible-portion estimate is available. repeated_questions "
        "counts the repeated or unnecessary questions already included in avoidable_turns; do not add "
        "them a second time when choosing the score. Do not count user turns. A necessary Camera owner "
        "confirmation or a question that resolves a meaningful ambiguity is not avoidable. Unknown exact "
        "grams alone do not make such a question necessary for a known package. Do not penalize a useful "
        "substantive estimate, receipt, or notice that a balance remains pending; do not treat honest "
        "uncertainty by itself as friction. Set the base score from avoidable_turns: 5 for zero, 4 for one, "
        "3 for two, 2 for three or four, 1 for five or six, and 0 for seven or more or dialogue so "
        "obstructive that the user cannot complete the task. Count repeated questions accurately and "
        "choose reason codes that fit the dialogue. This score says nothing about whether data was saved. "
        "Set useful_button_click true only when the supplied native evidence proves the user clicked an "
        "actually offered, relevant option. A typed answer, merely offered button, or untrusted or "
        "irrelevant callback is not a click and earns no bonus; a button attached only to a useless "
        "or unnecessary repeated question is not relevant. The click bonus is separate from the "
        "base score, is applied only by the evaluator, and is capped at 5."
    )
    if click_evidence:
        prompt += " Native callback evidence: " + json.dumps(click_evidence, ensure_ascii=False) + "."
    else:
        prompt += " No valid native callback evidence is supplied, so useful_button_click must be false."
    return prompt + " Dialogue: " + json.dumps(dialogue, ensure_ascii=False)


def _button_click_evidence(case: Case | JudgeCase) -> list[dict[str, Any]]:
    """Return native clicks bound to the selected owner trajectory."""
    if not isinstance(case, Case) or not case.persistence_evidence:
        return []
    try:
        goal = case.persistence_evidence["goal"]
        exported = case.persistence_evidence["dialogue_export"]["episodes"]
        episode_ids = set(goal["episode_ids"])
    except (KeyError, TypeError):
        return []
    evidence = []
    for episode in exported:
        try:
            if episode.get("episode", {}).get("episode_id") not in episode_ids:
                continue
            ctx = episode.get("trusted_camera_context")
            turn_provenance = episode.get("turn_provenance")
            inbound = episode.get("episode", {}).get("metadata", {}).get("inbound", {})
            md = inbound.get("metadata", {})
            camera_callback_candidate = md.get("_camera_ingress_callback_candidate")
            camera_callback_eligible = md.get("_camera_ingress_callback_eligible")
            feedback_receipt = md.get("_camera_feedback_receipt")
            options = md.get("native_keyboard_options")
            idx = md.get("native_keyboard_selected_index")
            label = md.get("native_keyboard_selected_label")
            data = md.get("callback_data")
            episode_id = episode.get("episode", {}).get("episode_id")
            common_binding = (
                episode.get("principal_id") == goal.get("principal_id")
                and episode.get("episode", {}).get("session_id") == goal.get("gateway_session_id")
            )
            camera_binding = (
                isinstance(ctx, dict) and ctx.get("kind") == "owner_turn"
                and ctx.get("tenant_id") == case.owner_id
                and ctx.get("principal_id") == goal.get("principal_id")
                and ctx.get("recipient_principal") == goal.get("principal_id")
                and ctx.get("gateway_session_id") == goal.get("gateway_session_id")
                and ctx.get("episode_id") == episode_id
                and isinstance(ctx.get("source_message_id"), str)
                and isinstance(ctx.get("candidate_id"), str) and bool(ctx.get("candidate_id"))
                and type(ctx.get("native_photo_id")) is int
                and md.get("_camera_candidate_id") == ctx.get("candidate_id")
                and md.get("_camera_native_binding") == str(md.get("native_message_id"))
                and any(
                    isinstance(item, dict)
                    and item.get("episode_id") == episode_id
                    and item.get("source_message_id") == ctx.get("source_message_id")
                    and item.get("principal_id") == ctx.get("principal_id")
                    and item.get("operation_id") == ctx.get("operation_id")
                    for item in (turn_provenance if isinstance(turn_provenance, list) else [])
                )
            )
            provenance_binding = any(
                isinstance(item, dict)
                and item.get("source_message_id") == str(md.get("message_id"))
                and item.get("principal_id") == goal.get("principal_id")
                and item.get("episode_id") == episode_id
                and isinstance(item.get("operation_id"), str)
                and item.get("operation_id") == f"{item.get('logical_turn_id')}:assistant"
                for item in (turn_provenance if isinstance(turn_provenance, list) else [])
            )
            callback_id = md.get("callback_query_id")
            callback_message_id = md.get("message_id")
            feedback_binding = (
                ctx is None
                and isinstance(feedback_receipt, dict)
                and feedback_receipt.get("kind") == "not_food"
                and feedback_receipt.get("tenant_id") == case.owner_id
                and feedback_receipt.get("owner_principal") == goal.get("principal_id")
                and feedback_receipt.get("gateway_session_id") == goal.get("gateway_session_id")
                and feedback_receipt.get("candidate_id") == camera_callback_candidate
                and isinstance(camera_callback_candidate, str)
                and bool(camera_callback_candidate)
                and camera_callback_eligible is True
                and type(feedback_receipt.get("native_photo_id")) is int
                and feedback_receipt.get("native_photo_id") > 0
                and feedback_receipt.get("native_message_id") == str(md.get("native_message_id"))
                and str(feedback_receipt.get("native_message_id")) == str(callback_message_id)
                and feedback_receipt.get("source_message_id") == str(callback_message_id)
                and isinstance(callback_id, str) and bool(callback_id)
                and feedback_receipt.get("callback_query_id") == callback_id
                and feedback_receipt.get("operation_id")
                == f"{camera_callback_candidate}:classifier_feedback:{callback_id}"
                and md.get("native_keyboard_reflection_confirmed") is True
                and "✅ Это не еда" in str(md.get("native_keyboard_reflection") or "")
                and isinstance(turn_provenance, list)
            )
            bound_native_episode = (
                camera_binding if ctx is not None
                else feedback_binding if isinstance(feedback_receipt, dict)
                else provenance_binding
            )
            camera_callback_facts = (
                camera_callback_candidate is None
                or (
                    isinstance(ctx, dict)
                    and camera_callback_candidate == ctx.get("candidate_id")
                    and camera_callback_eligible is True
                )
                or feedback_binding
            )
            # Camera captions use either a photo-made date or an explicit
            # unknown-date sentence. Keep both forms behind the same question
            # prefix so an unbound Camera click cannot fall back to user provenance.
            prompt_text = md.get("native_keyboard_prompt", "")
            camera_caption = isinstance(prompt_text, str) and re.match(
                r"^\s*Съели ли вы это\?\s*(?:Фото сделано\b|Дата съёмки неизвестна\b)", prompt_text,
            ) is not None
            if camera_caption and ctx is None and not feedback_binding:
                camera_callback_facts = False
            if (
                not common_binding or not bound_native_episode
                or not camera_callback_facts
                or inbound.get("channel") != "telegram"
                or md.get("callback_query") is not True
                or type(md.get("native_message_id")) is not int
                or md.get("native_message_id") <= 0
                or not isinstance(data, str) or not data.startswith("ask:")
                or not isinstance(options, list) or not 2 <= len(options) <= 8
                or any(not isinstance(option, str) or not option or len(option) > 60 for option in options)
                or type(idx) is not int or not 0 <= idx < len(options)
                or options[idx] != label or not isinstance(label, str) or not label
                or not isinstance(md.get("native_keyboard_prompt"), str)
                or not md["native_keyboard_prompt"].strip()
                or len(md["native_keyboard_prompt"]) > 2000
                or data != f"ask:{idx}"
            ):
                continue
            evidence.append({"episode_id": episode["episode"]["episode_id"],
                             "source_message_id": str(md["message_id"]),
                             "native_message_id": md["native_message_id"],
                             "operation_id": (ctx or {}).get("operation_id") if isinstance(ctx, dict)
                             else next((item.get("operation_id") for item in turn_provenance
                                        if isinstance(item, dict) and item.get("source_message_id") == str(md.get("message_id"))), None),
                             "prompt": md["native_keyboard_prompt"],
                             "options": options, "selected_index": idx, "selected_label": label})
        except (KeyError, TypeError, AttributeError):
            continue
    return evidence[:MAX_EVENTS]


class ModelResponseError(ValueError):
    """The injected model call returned no usable response."""


async def model_result(
    case: Case | JudgeCase,
    call: ModelCall,
    budget: CallBudget,
    *,
    sol: bool,
    image: bytes | None = None,
) -> Reference | A2Vote:
    budget.reserve()  # Exhaustion is a run error, never an inconclusive model result.
    try:
        if sol:
            raw = await call(SOL_MODEL, "high", sol_prompt(case), image)
            return Reference.model_validate_json(raw)
        raw = await call(LUNA_MODEL, "medium", a2_prompt(case), None)
        return A2Vote.model_validate_json(raw)
    except (OSError, ValidationError, ModelResponseError):
        raise ModelResponseError("model result unavailable or invalid") from None


def aggregate_votes(votes: list[A2Vote], *, click_evidence: bool = False) -> dict[str, Any]:
    base_score = sorted(v.score for v in votes)[1]
    useful_click_votes = sum(v.useful_button_click for v in votes) if click_evidence else 0
    bonus = int(useful_click_votes >= 2)
    return {
        "a2_base_score": base_score,
        "a2_button_bonus": bonus,
        "a2_useful_button_click_votes": useful_click_votes,
        "a2_scores": min(5, base_score + bonus),
        "a2_avoidable_turns": sorted(v.avoidable_turns for v in votes)[1],
        "a2_repeated_questions": max(v.repeated_questions for v in votes),
        "a2_reason_codes": sorted(
            code
            for code in {item for vote in votes for item in vote.reason_codes}
            if sum(code in vote.reason_codes for vote in votes) >= 2
        ),
    }


def score_reference(labels: JudgeLabels, reference: Reference) -> tuple[str, str]:
    if reference.consumption_state != labels.consumption_state:
        return "FAIL", "reference consumption state differs from human label"
    if labels.consumption_state == "consumed":
        assert reference.kcal is not None and labels.kcal_min is not None
        assert labels.kcal_max is not None
        if not labels.kcal_min <= reference.kcal <= labels.kcal_max:
            return "FAIL", "reference kcal outside human interval"
    if labels.consumption_state == "uncertain":
        return "PASS", "reference preserves human-labeled uncertainty"
    return "PASS", "reference matches human consumption label"


def score_estimate(labels: JudgeLabels, reference: Reference) -> tuple[str, str]:
    if labels.estimate_kcal_min is None:
        return "NOT_ASSESSED", "no human photo-estimate interval supplied"
    assert labels.estimate_kcal_max is not None
    if reference.estimated_kcal is None:
        return "FAIL", "reference omitted expected photo estimate"
    if not labels.estimate_kcal_min <= reference.estimated_kcal <= labels.estimate_kcal_max:
        return "FAIL", "reference photo estimate outside human interval"
    return "PASS", "reference photo estimate matches human interval"


async def calibrate_judge_case(
    case: JudgeCase,
    call: ModelCall,
    budget: CallBudget,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "lane": "JUDGE_CALIBRATION",
        "reference_quality": "INCONCLUSIVE",
        "consumption_quality": "INCONCLUSIVE",
        "estimate_quality": (
            "INCONCLUSIVE" if case.labels.estimate_kcal_min is not None else "NOT_ASSESSED"
        ),
        "efficiency_quality": "NOT_RUN",
    }
    try:
        image = checked_image(case)
    except (OSError, ValueError):
        result["reference_reason"] = "original image unavailable or invalid"
        return result
    result["source_image_sha256"] = case.image_sha256
    result["validated_image_sha256"] = hashlib.sha256(image).hexdigest()
    try:
        reference = await model_result(case, call, budget, sol=True, image=image)
    except ModelResponseError:
        result["reference_reason"] = "reference result unavailable or invalid"
        return result
    assert isinstance(reference, Reference)
    consumption_verdict, consumption_reason = score_reference(case.labels, reference)
    estimate_verdict, estimate_reason = score_estimate(case.labels, reference)
    verdict = "PASS" if consumption_verdict == "PASS" and estimate_verdict != "FAIL" else "FAIL"
    reason = consumption_reason if consumption_verdict != "PASS" else estimate_reason
    result.update(
        reference_quality=verdict,
        reference_reason=reason,
        consumption_quality=consumption_verdict,
        estimate_quality=estimate_verdict,
        reference={
            "consumption_state": reference.consumption_state,
            "kcal": reference.kcal,
            "estimated_kcal": reference.estimated_kcal,
        },
    )
    if verdict != "PASS":
        return result
    votes: list[A2Vote] = []
    for _ in range(3):
        try:
            vote = await model_result(case, call, budget, sol=False)
        except ModelResponseError:
            result["efficiency_quality"] = "INCONCLUSIVE"
            result["efficiency_reason"] = "provider unavailable or invalid Luna vote"
            return result
        assert isinstance(vote, A2Vote)
        votes.append(vote)
    aggregate = aggregate_votes(votes)
    result.update(aggregate)
    labels = case.labels
    avoidable_ok = (
        abs(aggregate["a2_avoidable_turns"] - labels.avoidable_turns) <= labels.avoidable_tolerance
    )
    repeated_ok = (
        abs(aggregate["a2_repeated_questions"] - labels.repeated_questions)
        <= labels.repeated_tolerance
    )
    result["efficiency_quality"] = "PASS" if avoidable_ok and repeated_ok else "FAIL"
    result["efficiency_reason"] = (
        "Luna aggregate matches human efficiency labels"
        if avoidable_ok and repeated_ok
        else "Luna aggregate differs from human efficiency labels"
    )
    return result


async def calibrate_case(case: Case, call: ModelCall, budget: CallBudget) -> dict[str, Any]:
    """Sol runs first; three independent Luna votes only follow A1 PASS."""
    try:
        image = checked_image(case)
    except ValidationError:
        return {
            "case_id": case.case_id,
            "lane": "PRODUCT_A1",
            "a1": "INCONCLUSIVE",
            "reason": "invalid nutrition annotation",
            "a2": "NOT_RUN",
        }
    except (OSError, ValueError) as exc:
        return {
            "case_id": case.case_id,
            "lane": "PRODUCT_A1",
            "a1": "INCONCLUSIVE",
            "reason": str(exc),
            "a2": "NOT_RUN",
        }
    image_hashes = {
        "source_image_sha256": case.image_sha256,
        "validated_image_sha256": hashlib.sha256(image).hexdigest(),
    }
    try:
        reference = await model_result(case, call, budget, sol=True, image=image)
    except ModelResponseError:
        return {
            "case_id": case.case_id,
            "lane": "PRODUCT_A1",
            "a1": "INCONCLUSIVE",
            "reason": "reference unavailable or invalid",
            "a2": "NOT_RUN",
            **image_hashes,
        }
    assert isinstance(reference, Reference)
    a1_result = _score_a1_details(case, reference)
    verdict, reason = a1_result["a1"], a1_result["reason"]
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "lane": "PRODUCT_A1",
        "a1": verdict,
        "reason": reason,
        "a2": "NOT_RUN",
        **image_hashes,
    }
    result.update({key: value for key, value in a1_result.items()
                   if key not in {"a1", "reason"}})
    result["reference"] = {
        "consumption_state": reference.consumption_state,
        "kcal": reference.kcal,
    }
    if verdict != "PASS":
        return result
    if case.dialogue is None:
        result["a2"] = "INCONCLUSIVE"
        result["a2_reason"] = "full dialogue is missing"
        return result
    votes: list[A2Vote] = []
    for _ in range(3):
        try:
            vote = await model_result(case, call, budget, sol=False)
        except ModelResponseError:
            result["a2"] = "INCONCLUSIVE"
            result["a2_reason"] = "Luna vote unavailable or invalid"
            return result
        assert isinstance(vote, A2Vote)
        votes.append(vote)
    result["a2"] = "SCORED"
    result.update(aggregate_votes(votes, click_evidence=bool(_button_click_evidence(case))))
    return result


async def calibrate_cases(
    cases: Sequence[Case | JudgeCase], call: ModelCall, budget: CallBudget
) -> list[dict[str, Any]]:
    if not 3 <= len(cases) <= MAX_CASES:
        raise ValueError("calibration requires 3..5 manually reviewed cases")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("duplicate case ID")
    return [
        await calibrate_judge_case(case, call, budget)
        if isinstance(case, JudgeCase)
        else await calibrate_case(case, call, budget)
        for case in cases
    ]


def write_report(path: Path, report: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(report, indent=2) + "\n", mode=0o600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="private JSON array of 3..5 reviewed cases")
    parser.add_argument(
        "--lane",
        choices=("product_a1", "judge_calibration"),
        default="product_a1",
        help="explicit calibration lane",
    )
    parser.add_argument("--subscription-results", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True, help="private Ohmo workspace")
    parser.add_argument("--max-calls", type=int, default=20)
    args = parser.parse_args()
    from ohmo.evals.camera_subscription_results import (
        REPOSITORY,
        SubscriptionResults,
        read_private_json,
    )

    if args.workspace.resolve().is_relative_to(REPOSITORY):
        parser.error("private workspace must be outside the repository")

    case_type = JudgeCase if args.lane == "judge_calibration" else Case
    manifest = read_private_json(args.manifest)
    if not isinstance(manifest, list) or not 3 <= len(manifest) <= MAX_CASES:
        parser.error("manifest must contain 3..5 cases")
    cases = [case_type.model_validate(item) for item in manifest]
    if not 3 <= len(cases) <= MAX_CASES:
        parser.error("manifest must contain 3..5 cases")
    budget = CallBudget(max_calls=args.max_calls)
    intake = SubscriptionResults(read_private_json(args.subscription_results), cases)
    result = asyncio.run(calibrate_cases(cases, intake.call, budget))
    intake.assert_exhausted()
    report = {
        "prototype": "camera_grader_calibration",
        "lane": args.lane.upper(),
        "reference_model": SOL_MODEL,
        "reference_reasoning_effort": "high",
        "reference_selection": "human_authorized_temporary",
        "sol_prompt_version": SOL_PROMPT_VERSION,
        "subscription_turns": budget.calls,
        "actual_usd": "unknown",
        "cases": result,
    }
    path = get_eval_store(args.workspace).root / "reports" / "camera_calibration.json"
    write_report(path, report)
    print(path)


if __name__ == "__main__":
    main()
