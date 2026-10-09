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


def _diagnostic_reviewed_turn_binding(
    goal: Any, export: object,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]] | None:
    """Recompute ordinary source turns for a FAIL-only state diagnostic.

    This intentionally does not grant Camera authority or change the primary
    dialogue-binding result. It lets the persistence grader ignore no source
    claims: every ordinary turn must be independently reproducible from the
    exported inbound envelope, and the root must match the reviewed Goal.
    """
    from ohmo.evals.nutrition_persistence import _derive_exported_turn_provenance

    if not isinstance(export, dict) or export.get("privacy") != "private":
        return None
    raw_episodes = export.get("episodes")
    if not isinstance(raw_episodes, list):
        return None
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_episodes:
        if not isinstance(item, dict) or not isinstance(item.get("episode"), dict):
            return None
        episode_id = item["episode"].get("episode_id")
        if not isinstance(episode_id, str) or episode_id in by_id:
            return None
        by_id[episode_id] = item

    sources: dict[str, list[str]] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    root_matches = 0
    identity_fields = (
        "episode_id", "source_message_id", "logical_turn_id", "operation_id", "principal_id",
    )
    for episode_id in goal.episode_ids:
        item = by_id.get(episode_id)
        if item is None or item.get("dialogue_complete") is not True:
            return None
        episode = item["episode"]
        episode_metadata = episode.get("metadata")
        if (
            episode.get("session_id") != goal.gateway_session_id
            or not isinstance(episode_metadata, dict)
            or episode_metadata.get("workspace") != goal.eval_workspace
        ):
            return None
        inbound = episode_metadata.get("inbound")
        derived = _derive_exported_turn_provenance(episode, inbound)
        exported_turns = item.get("turn_provenance")
        if (
            derived is None
            or not isinstance(exported_turns, list)
            or len(exported_turns) != 1
            or not isinstance(exported_turns[0], dict)
            or any(exported_turns[0].get(key) != derived.get(key) for key in identity_fields)
        ):
            return None
        if (item.get("trusted_camera_context") or {}).get("kind") == "initial_context":
            # Initial Camera receipts are deliberately not promoted to source
            # authority here; the ordinary owner turn below must stand alone.
            continue
        source_id = derived.get("source_message_id")
        source_ids = item.get("source_message_ids")
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(source_ids, list)
            or len(source_ids) != 1
            or source_ids[0] != source_id
            or derived.get("principal_id") != goal.principal_id
            or item.get("principal_id") != goal.principal_id
        ):
            return None
        sources[episode_id] = [source_id]
        provenance[episode_id] = [{key: derived[key] for key in identity_fields}]
        if episode_id == goal.trace_episode_id:
            root_matches += int(
                source_id == goal.source_message_id
                and derived.get("logical_turn_id") == goal.logical_turn_id
                and derived.get("operation_id") == goal.operation_id
                and derived.get("principal_id") == goal.principal_id
            )
    if root_matches != 1:
        return None
    return sources, provenance


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
            Goal, Manifest, _event_from_raw, _fold_events,
            bind_wellness_snapshot, grade_manifest,
            validate_dialogue_binding,
        )

        goal = Goal.model_validate(evidence["goal"])
        manifest = Manifest(schema_version=1, goals=[goal])
        bound = validate_dialogue_binding(manifest, evidence["dialogue_export"])
        if bound[goal.case_id]["complete"] is not True:
            expected_turns = []
            for exported in evidence["dialogue_export"]["episodes"]:
                if exported.get("episode", {}).get("episode_id") in goal.episode_ids:
                    expected_turns.extend(
                        {"role": turn["role"], "text": turn["text"]}
                        for turn in exported.get("dialogue", [])
                        if turn.get("role") in {"user", "assistant"}
                    )
            same_case = (
                goal.case_id == case.case_id
                and goal.source_message_id == case.source_message_id
                and goal.owner_id == case.owner_id
                and goal.operation_id == case.operation_id
                and goal.canonical_meal_id == case.meal_id
            )
            same_dialogue = case.dialogue is not None and [
                turn.model_dump() for turn in case.dialogue
            ] == expected_turns
            if same_case and same_dialogue:
                diagnostic_binding = _diagnostic_reviewed_turn_binding(
                    goal, evidence["dialogue_export"],
                )
                if diagnostic_binding is not None:
                    diagnostic = grade_manifest(
                        manifest, evidence["honcho_snapshot"],
                        bind_wellness_snapshot(evidence["telegent_snapshot"], goal=goal),
                        reviewed_turn_sources=diagnostic_binding[0],
                        reviewed_turn_provenance=diagnostic_binding[1],
                    )[0]
                    if diagnostic.get("a1") == "FAIL" and diagnostic.get("state_failures"):
                        return {
                            "a1": "FAIL",
                            "reason": "scoped persisted nutrition state proves a reviewed goal mismatch "
                            "although Camera dialogue source binding is incomplete",
                            "persistence_stage": diagnostic.get("stage"),
                            "state_failures": diagnostic["state_failures"],
                        }
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
        if (
            goal.case_id != case.case_id
            or goal.source_message_id != case.source_message_id
            or goal.owner_id != case.owner_id
            or goal.operation_id != case.operation_id
            or goal.canonical_meal_id != case.meal_id
        ):
            return {"a1": "INCONCLUSIVE", "reason": "reviewed nutrition goal does not bind to product Case identity",
                    "persistence_stage": "CASE_BINDING_FAILED"}
        telegent = evidence["telegent_snapshot"]
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
    if (case.reviewed_state == "consumed") is not goal.expected_consumed:
        return {"a1": "FAIL", "reason": "reviewed Case consumption state disagrees with reviewed nutrition goal",
                **details}
    actual_event_ids = result.get("actual_event_ids")
    if case.reviewed_state == "never_recorded":
        if result.get("stage") != "COMPLETE_ABSENCE" or actual_event_ids != []:
            return {"a1": "INCONCLUSIVE", "reason": "reviewed never-recorded state is not proved by complete absence",
                    **details}
    elif case.reviewed_state == "validly_retracted":
        try:
            raw_messages = evidence["honcho_snapshot"]["messages"]
            by_id = {
                parsed["event_id"]: parsed
                for raw in raw_messages
                if isinstance(raw, dict)
                if (parsed := _event_from_raw(raw)) is not None
                and not parsed.get("invalid") and not parsed.get("unannotated")
            }
            if not isinstance(actual_event_ids, list) or not actual_event_ids:
                raise ValueError("no validated persisted source history")
            history_events = [by_id[event_id] for event_id in actual_event_ids]
            for event in history_events:
                event["_created_at"] = event["created_at"]
            history_events.sort(key=lambda event: (event["_created_at"], event["event_id"]))
            history = _fold_events(history_events, goal.meal_timezone)["history"]
            retracted = (
                len(history) >= 2
                and history[0]["consumed"] is True
                and history[-1]["consumed"] is False
                and any(item["consumed"] is True for item in history[:-1])
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            retracted = False
        if not retracted:
            return {"a1": "INCONCLUSIVE", "reason": "reviewed retraction lacks validated consumed-to-not-consumed event history",
                    **details}
    elif case.reviewed_state != "consumed":
        return {"a1": "INCONCLUSIVE", "reason": "reviewed meal state is unsupported by persistence evidence",
                **details}
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
        "them a second time when choosing the score. After the user selects a portion, asking them to "
        "confirm that same selection again is avoidable unless new material ambiguity appeared. Do not "
        "treat a reply instruction as useful when it only works around the bot failing to honor an option "
        "the user already selected. Do not count user turns. A necessary Camera owner "
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
            native_click_receipt = md.get("_camera_native_click_receipt")
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
            native_click_binding = False
            if isinstance(native_click_receipt, dict) and provenance_binding:
                initial_matches = any(
                    isinstance(initial, dict)
                    and initial.get("kind") == "initial_context"
                    and initial.get("candidate_id") == native_click_receipt.get("candidate_id")
                    and initial.get("native_photo_id") == native_click_receipt.get("native_photo_id")
                    and initial.get("tenant_id") == native_click_receipt.get("tenant_id") == case.owner_id
                    and initial.get("recipient_principal") == native_click_receipt.get("owner_principal")
                        == goal.get("principal_id")
                    and initial.get("gateway_session_id") == goal.get("gateway_session_id")
                    for selected in exported if selected.get("episode", {}).get("episode_id") in episode_ids
                    for initial in [selected.get("trusted_camera_context") or {}]
                )
                issued_ids = native_click_receipt.get("issued_keyboard_message_ids")
                prompt = md.get("native_keyboard_prompt")
                expected_options_hash = hashlib.sha256(json.dumps(
                    options, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")).hexdigest() if isinstance(options, list) else ""
                expected_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest() \
                    if isinstance(prompt, str) else ""
                native_click_binding = bool(
                    initial_matches
                    and native_click_receipt.get("kind") == "issued_model_button"
                    and native_click_receipt.get("native_message_id") == str(md.get("native_message_id"))
                    and isinstance(issued_ids, list) and str(md.get("native_message_id")) in issued_ids
                    and native_click_receipt.get("callback_message_id") == str(md.get("message_id"))
                    and native_click_receipt.get("callback_query_id") == callback_id
                    and native_click_receipt.get("owner_principal") == goal.get("principal_id")
                    and native_click_receipt.get("issued_options_sha256") == expected_options_hash
                    and native_click_receipt.get("issued_prompt_sha256") == expected_prompt_hash
                    and native_click_receipt.get("callback_data") == data == f"ask:{idx}"
                    and native_click_receipt.get("selected_index") == idx
                    and native_click_receipt.get("selected_label") == label
                    and native_click_receipt.get("operation_id")
                        == f"{native_click_receipt.get('candidate_id')}:issued_button:{callback_id}"
                )
            camera_callback_without_receipt = (
                ctx is None
                and ("_camera_candidate_id" in md or "_camera_native_binding" in md
                     or camera_callback_candidate is not None)
                and not native_click_binding
            )
            bound_native_episode = (
                camera_binding if ctx is not None
                else feedback_binding if isinstance(feedback_receipt, dict)
                else native_click_binding or (provenance_binding and not camera_callback_without_receipt)
            )
            camera_callback_facts = (
                camera_callback_candidate is None
                or (
                    isinstance(ctx, dict)
                    and camera_callback_candidate == ctx.get("candidate_id")
                    and camera_callback_eligible is True
                )
                or feedback_binding
                or native_click_binding
            )
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
