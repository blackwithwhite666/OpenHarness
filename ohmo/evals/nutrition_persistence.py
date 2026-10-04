"""Read-only persistence grading for reviewed nutrition goals.

The manifest, not a trace annotation or assistant claim, defines expected goals.
Live acquisition is deliberately separate from the pure decision function.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ohmo.evals.nutrition_trace import NutritionAnnotationV2
from ohmo.gateway.turn_context import canonical_principal
from ohmo.memory_service.honcho_client import HonchoClient, HonchoError
from openharness.utils.fs import atomic_write_text

MAX_GOALS = 100
MAX_MESSAGES = 20_000
MAX_EPISODES = 500


class Goal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=120)
    episode_ids: list[str] = Field(min_length=1, max_length=MAX_EPISODES)
    owner_id: str = Field(min_length=1, max_length=512)
    principal_id: str = Field(min_length=1, max_length=512)
    workspace_id: str = Field(min_length=1, max_length=512)
    eval_workspace: str = Field(min_length=1, max_length=2048)
    peer_id: str = Field(min_length=1, max_length=128)
    canonical_owner_id: str = Field(min_length=1, max_length=512)
    canonical_login: str = Field(min_length=1, max_length=256)
    session_id: str = Field(min_length=1, max_length=512)
    gateway_session_id: str = Field(min_length=1, max_length=512)
    source_message_id: str = Field(min_length=1, max_length=512)
    meal_date: date
    meal_timezone: str = "UTC"
    trajectory_started_at: datetime
    trajectory_as_of: datetime
    logical_turn_id: str = Field(min_length=1, max_length=512)
    trace_episode_id: str = Field(min_length=1, max_length=512)
    operation_id: str = Field(min_length=1, max_length=512)
    canonical_meal_id: str = Field(min_length=1, max_length=512)
    expected_consumed: bool = Field(strict=True)
    expected_kcal: float | None = Field(default=None, strict=True)
    tolerance_fraction: float = Field(default=0.30, ge=0, le=0.30, strict=True)
    expectation_origin: Literal["explicit_fixture", "reviewed_user_dialogue", "frozen_photo_reference"]
    expectation_source: str = Field(min_length=1, max_length=512)
    review_notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _goal_shape(self) -> "Goal":
        if not math.isfinite(self.tolerance_fraction):
            raise ValueError("tolerance_fraction must be finite")
        try:
            ZoneInfo(self.meal_timezone)
        except (ZoneInfoNotFoundError, TypeError):
            raise ValueError("meal_timezone must be a valid IANA timezone") from None
        for stamp in (self.trajectory_started_at, self.trajectory_as_of):
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ValueError("reviewed trajectory bounds must be timezone-aware")
        if self.trajectory_started_at > self.trajectory_as_of:
            raise ValueError("trajectory_started_at must not exceed trajectory_as_of")
        if self.canonical_meal_id != derive_meal_id(
            tenant_id=self.owner_id, source_principal=self.principal_id,
            gateway_session_id=self.gateway_session_id, source_message_id=self.source_message_id,
        ):
            raise ValueError("canonical_meal_id does not match trusted source identity")
        if self.operation_id != f"{self.logical_turn_id}:assistant":
            raise ValueError("operation_id must be the gateway assistant operation for logical_turn_id")
        if self.expected_consumed:
            if self.expected_kcal is None or not math.isfinite(self.expected_kcal) or self.expected_kcal < 0:
                raise ValueError("consumed goal requires finite non-negative expected_kcal")
        elif self.expected_kcal is not None:
            raise ValueError("absence goal must omit expected_kcal")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("episode_ids must be unique")
        if self.trace_episode_id not in self.episode_ids:
            raise ValueError("trace_episode_id must be one of the reviewed episode IDs")
        return self


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    goals: list[Goal] = Field(min_length=1, max_length=MAX_GOALS)

    @model_validator(mode="after")
    def _unique_cases(self) -> "Manifest":
        if len({goal.case_id for goal in self.goals}) != len(self.goals):
            raise ValueError("case_id values must be unique")
        return self


def _dict(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _normalize_login(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("@"):
        normalized = normalized[1:]
    return normalized.lower()


def _validated_nutrition_annotation(value: object) -> dict[str, Any] | None:
    """Normalize one recorded schema-v2 annotation or fail closed."""
    try:
        return NutritionAnnotationV2.model_validate(value).model_dump(mode="json")
    except (TypeError, ValueError):
        return None


def derive_meal_id(*, tenant_id: str, source_principal: str, gateway_session_id: str,
                   source_message_id: str) -> str:
    """Mirror Telegent's published stable identity derivation without importing it."""
    seed = "\0".join(("telegent-nutrition-meal-v1", tenant_id, source_principal,
                      gateway_session_id, source_message_id))
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _event_from_raw(message: dict[str, Any]) -> dict[str, Any] | None:
    """Extract persisted identity and validated annotation from one raw message."""
    metadata = _dict(message.get("metadata"))
    if metadata is None:
        return {"invalid": True}
    raw_identity = (message.get("id"), message.get("session_id"), message.get("workspace_id"),
                    message.get("peer_id"), message.get("created_at"))
    if not all(isinstance(value, str) and value for value in raw_identity):
        return {"invalid": True}
    try:
        created_at = _parse_time(message["created_at"])
    except (TypeError, ValueError):
        return {"invalid": True}
    for key in ("tenant_id", "gateway_session_id", "source_principal"):
        value = metadata.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            return {"invalid": True}
    trace = _dict(metadata.get("decision_trace")) if metadata else None
    raw_annotations = trace.get("annotations") if trace else None
    annotations = _dict(raw_annotations)
    has_annotations = trace is not None and "annotations" in trace
    has_nutrition = annotations is not None and "nutrition" in annotations
    raw_nutrition = annotations.get("nutrition") if annotations else None
    nutrition = _dict(raw_nutrition)
    malformed_annotation = (
        (has_annotations and annotations is None)
        or (has_nutrition and nutrition is None)
    )
    if nutrition is None:
        return {"unannotated": True, "metadata": metadata, "created_at": created_at,
                "session_id": message["session_id"], "workspace_id": message["workspace_id"],
                "peer_id": message["peer_id"], "malformed_annotation": malformed_annotation}
    try:
        annotation = NutritionAnnotationV2.model_validate(nutrition)
    except ValidationError:
        return {"invalid": True}
    assert metadata is not None
    required = (message.get("id"), metadata.get("tenant_id"), metadata.get("gateway_session_id"),
                message.get("session_id"), message.get("workspace_id"), metadata.get("client_op_id"),
                metadata.get("source_principal"), metadata.get("logical_turn_id"),
                metadata.get("decision_trace_episode_id"), trace.get("episode_id") if trace else None,
                message.get("peer_id"), metadata.get("source_message_id"))
    if not all(isinstance(value, str) and value for value in required):
        return {"invalid": True}
    if metadata.get("role") != "assistant":
        return {"invalid": True}
    return {
        "event_id": message["id"],
        "metadata": metadata,
        "owner_id": metadata["tenant_id"],
        "session_id": message["session_id"],
        "workspace_id": message["workspace_id"],
        "gateway_session_id": metadata["gateway_session_id"],
        "source_message_id": metadata.get("source_message_id"),
        "reply_to_source_message_id": metadata.get("reply_to_source_message_id"),
        "root_source_message_id": (
            metadata.get("reply_to_source_message_id")
            if annotation.record_type in {"meal_correction", "meal_deletion"}
            else metadata.get("source_message_id")
        ),
        "operation_id": metadata.get("client_op_id"),
        "principal_id": metadata["source_principal"],
        "logical_turn_id": metadata["logical_turn_id"],
        "trace_episode_id": metadata["decision_trace_episode_id"],
        "nested_trace_episode_id": trace["episode_id"],
        "attachment_fingerprints": metadata.get("attachment_fingerprints", []),
        "peer_id": message["peer_id"],
        "created_at": created_at,
        "annotation": annotation.model_dump(mode="json"),
    }


def _trusted_context_target(goal: Goal, event: dict[str, Any], turns: list[dict[str, Any]],
                            events: list[dict[str, Any]]) -> str | None:
    """Resolve a no-reply edit only through both recorded accepted receipts."""
    matches = [turn for turn in turns
               if turn.get("episode_id") == event.get("trace_episode_id")
               and turn.get("source_message_id") == event.get("source_message_id")
               and turn.get("logical_turn_id") == event.get("logical_turn_id")
               and turn.get("operation_id") == event.get("operation_id")
               and turn.get("principal_id") == event.get("principal_id")]
    if len(matches) != 1:
        return None
    final = matches[0].get("gateway_final_metadata")
    evidence = _dict(final.get("nutrition_context_evidence")) if isinstance(final, dict) else None
    committed = _dict(final.get("nutrition_committed_annotation")) if isinstance(final, dict) else None
    proposal = _dict(final.get("nutrition_model_proposal_annotation")) if isinstance(final, dict) else None
    execution = _dict(final.get("nutrition_finalization")) if isinstance(final, dict) else None
    committed_annotation = _validated_nutrition_annotation(committed)
    proposed_annotation = _validated_nutrition_annotation(proposal)
    executed_annotation = _validated_nutrition_annotation(
        execution.get("annotation") if execution is not None else None
    )
    proposal_matches = final.get("nutrition_proposal_matches_committed") if isinstance(final, dict) else None
    fields = ("tenant_id", "source_principal", "gateway_session_id", "photo_source_message_id",
              "photo_received_at",
              "consumed_source_message_id", "target_meal_id", "original_receipt_event_id",
              "original_operation_id", "current_receipt_event_id", "current_operation_id",
              "current_logical_turn_id", "current_trace_episode_id")
    if (evidence is None or type(evidence.get("schema_version")) is not int
            or evidence.get("schema_version") != 1
            or any(not isinstance(evidence.get(key), str) or not evidence[key] for key in fields)
            or evidence.get("tenant_id") != goal.owner_id
            or evidence.get("source_principal") != goal.principal_id
            or evidence.get("gateway_session_id") != goal.gateway_session_id
            or evidence.get("current_receipt_event_id") != event.get("event_id")
            or evidence.get("current_operation_id") != event.get("operation_id")
            or evidence.get("current_logical_turn_id") != event.get("logical_turn_id")
            or evidence.get("current_trace_episode_id") != event.get("trace_episode_id")
            or final.get("nutrition_append_event_id") != event.get("event_id")
            or execution is None or type(execution.get("schema_version")) is not int
            or execution.get("schema_version") != 1
            or committed_annotation is None or proposed_annotation is None
            or executed_annotation is None
            or executed_annotation.get("record_type") not in {"meal_correction", "meal_deletion"}
            or committed_annotation != event.get("annotation")
            or executed_annotation != proposed_annotation
            or type(proposal_matches) is not bool
            or proposal_matches is not (proposed_annotation == committed_annotation)):
        return None
    original_rows = [item for item in events if item.get("event_id") == evidence["original_receipt_event_id"]]
    if len(original_rows) != 1:
        return None
    original = original_rows[0]
    try:
        _parse_time(evidence["photo_received_at"])
    except (TypeError, ValueError):
        return None
    original_turns = [turn for turn in turns
                      if turn.get("operation_id") == evidence["original_operation_id"]
                      and turn.get("source_message_id") == evidence["consumed_source_message_id"]
                      and turn.get("principal_id") == goal.principal_id
                      and turn.get("episode_id") == original.get("trace_episode_id")]
    if not original_turns:
        return None
    original_finals = [turn.get("gateway_final_metadata") for turn in original_turns]
    if any(not isinstance(item, dict) for item in original_finals):
        return None
    original_final = original_finals[0]
    if any(item != original_final for item in original_finals[1:]):
        return None
    occurrence = _dict(original_final.get("nutrition_consumed_occurrence")) if isinstance(original_final, dict) else None
    original_committed = _dict(original_final.get("nutrition_committed_annotation")) if isinstance(original_final, dict) else None
    original_execution = _dict(original_final.get("nutrition_finalization")) if isinstance(original_final, dict) else None
    original_committed_annotation = _validated_nutrition_annotation(original_committed)
    original_executed_annotation = _validated_nutrition_annotation(
        original_execution.get("annotation") if original_execution is not None else None
    )
    if (not isinstance(original_final, dict)
            or original_execution is None or type(original_execution.get("schema_version")) is not int
            or original_execution.get("schema_version") != 1
            or original_committed_annotation is None or original_executed_annotation is None
            or original_executed_annotation.get("record_type") != "meal_observation"
            or original_executed_annotation.get("consumption_status") != "consumed"
            or original_executed_annotation != original.get("annotation")
            or original.get("annotation", {}).get("record_type") != "meal_observation"
            or original.get("annotation", {}).get("consumption_status") != "consumed"
            or original.get("operation_id") != evidence["original_operation_id"]
            or original.get("source_message_id") != evidence["consumed_source_message_id"]
            or original.get("created_at") > event.get("created_at")
            or original_final.get("nutrition_append_event_id") != original.get("event_id")
            or original_committed_annotation != original.get("annotation")
            or original_executed_annotation != original_committed_annotation
            or occurrence is None or occurrence.get("schema_version") != 1
            or occurrence.get("receipt_event_id") != original.get("event_id")
            or occurrence.get("client_op_id") != original.get("operation_id")
            or occurrence.get("tenant_id") != goal.owner_id
            or occurrence.get("source_principal") != goal.principal_id
            or occurrence.get("gateway_session_id") != goal.gateway_session_id
            or type(occurrence.get("schema_version")) is not int
            or occurrence.get("schema_version") != 1
            or occurrence.get("append_source_message_id") != evidence["consumed_source_message_id"]
            or occurrence.get("photo_source_message_id") != evidence.get("photo_source_message_id")
            or occurrence.get("photo_received_at") != evidence.get("photo_received_at")
            ):
        return None
    return evidence["consumed_source_message_id"]


def _is_receipt_reconciled_retry(goal: Goal, event: dict[str, Any], turn: dict[str, Any],
                                accepted_turn: dict[str, Any], principal: Any) -> bool:
    """Prove an extra exported turn reconciled to this exact accepted receipt."""
    final = _dict(turn.get("gateway_final_metadata"))
    evidence = _dict(final.get("nutrition_context_evidence")) if final is not None else None
    committed = _dict(final.get("nutrition_committed_annotation")) if final is not None else None
    proposal = _dict(final.get("nutrition_model_proposal_annotation")) if final is not None else None
    execution = _dict(final.get("nutrition_finalization")) if final is not None else None
    committed_annotation = _validated_nutrition_annotation(committed)
    proposed_annotation = _validated_nutrition_annotation(proposal)
    executed_annotation = _validated_nutrition_annotation(
        execution.get("annotation") if execution is not None else None
    )
    accepted_final = _dict(accepted_turn.get("gateway_final_metadata"))
    accepted_evidence = (_dict(accepted_final.get("nutrition_context_evidence"))
                         if accepted_final is not None else None)
    if (final is None or evidence is None or committed is None or proposal is None
            or execution is None or type(execution.get("schema_version")) is not int
            or execution.get("schema_version") != 1
            or committed_annotation is None or proposed_annotation is None
            or executed_annotation is None
            or accepted_evidence is None):
        return False
    proposal_matches = final.get("nutrition_proposal_matches_committed")
    return (
        turn.get("episode_id") != event.get("trace_episode_id")
        and turn.get("source_message_id") == event.get("source_message_id")
        and turn.get("logical_turn_id") == event.get("logical_turn_id")
        and turn.get("operation_id") == event.get("operation_id")
        and turn.get("principal_id") == principal == goal.principal_id
        and type(proposal_matches) is bool
        and proposal_matches == (proposed_annotation == committed_annotation)
        and executed_annotation == proposed_annotation
        and type(evidence.get("schema_version")) is int and evidence.get("schema_version") == 1
        and evidence.get("tenant_id") == goal.owner_id
        and evidence.get("source_principal") == principal
        and evidence.get("gateway_session_id") == goal.gateway_session_id
        and evidence.get("current_receipt_event_id") == event.get("event_id")
        and evidence.get("current_operation_id") == event.get("operation_id")
        and evidence.get("current_logical_turn_id") == event.get("logical_turn_id")
        and evidence.get("current_trace_episode_id") == event.get("trace_episode_id")
        and evidence == accepted_evidence
        and final.get("nutrition_append_event_id") == event.get("event_id")
        and committed_annotation == event.get("annotation")
    )


def _grade_one(goal: Goal, honcho: dict[str, Any], telegent: dict[str, Any], *, now: datetime,
               grace_seconds: int, reviewed_turn_sources: dict[str, list[str]] | None = None,
               reviewed_turn_provenance: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    base = {"case_id": goal.case_id, "expectation_origin": goal.expectation_origin,
            "expectation_source": goal.expectation_source, "a2": "NOT_RUN"}
    if honcho.get("complete") is not True or telegent.get("complete") is not True:
        return {**base, "a1": "INCONCLUSIVE", "stage": "READ_INCOMPLETE", "reason": "scoped authoritative read incomplete"}
    if (telegent.get("user_id") != goal.canonical_owner_id
            or telegent.get("login") != _normalize_login(goal.canonical_login)):
        return {**base, "a1": "INCONCLUSIVE", "stage": "TELEGENT_SCOPE_MISMATCH",
                "reason": "canonical snapshot is not bound to reviewed owner"}
    canonical_rows = telegent.get("canonical_meals")
    if not isinstance(canonical_rows, list) or any(
        not isinstance(row, dict)
        or not _valid_canonical_meal(row, allow_unassigned=row.get("status") == "unassigned")
        or row.get("user_id") != goal.canonical_owner_id
        for row in canonical_rows
    ):
        return {**base, "a1": "INCONCLUSIVE", "stage": "CANONICAL_INVALID",
                "reason": "complete canonical read does not preserve its validated scoped row set"}
    source_count_present = "source_occurrence_count" in telegent
    source_occurrence_count = telegent.get("source_occurrence_count")
    if source_count_present and (isinstance(source_occurrence_count, bool)
            or not isinstance(source_occurrence_count, int) or source_occurrence_count < 0):
        return {**base, "a1": "INCONCLUSIVE", "stage": "CANONICAL_INVALID",
                "reason": "canonical source occurrence count is malformed"}
    try:
        t_start = _parse_time(telegent.get("start"))
        t_end = _parse_time(telegent.get("end"))
        local_day_start = datetime.combine(goal.meal_date, datetime.min.time(), ZoneInfo(goal.meal_timezone))
        if (t_start > local_day_start or t_end < goal.trajectory_as_of.astimezone(timezone.utc)
                or t_start > t_end):
            raise ValueError
        queried_at = _parse_time(telegent.get("queried_at"))
        if queried_at < t_end or queried_at > now or t_end > now or t_start > queried_at:
            raise ValueError
    except (ValueError, TypeError):
        return {**base, "a1": "INCONCLUSIVE", "stage": "TELEGENT_BOUNDS_MISMATCH", "reason": "canonical read bounds do not cover reviewed goal day"}
    try:
        messages = honcho["messages"]
        if not isinstance(messages, list):
            raise ValueError
        events = [_event_from_raw(item) for item in messages if isinstance(item, dict)]
        if len(events) != len(messages) or any(event and event.get("invalid") for event in events):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_INVALID", "reason": "persisted message identity or nutrition schema is ambiguous"}
    if (honcho.get("session_id") != goal.session_id or honcho.get("owner_id") != goal.owner_id
            or honcho.get("workspace_id") != goal.workspace_id):
        return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SCOPE_MISMATCH", "reason": "Honcho snapshot is not bound to manifest owner/session/workspace"}
    try:
        lower = _parse_time(honcho.get("since"))
        upper = _parse_time(honcho.get("until"))
        queried_at = _parse_time(honcho.get("queried_at"))
        if (lower > goal.trajectory_started_at or upper < goal.trajectory_as_of
                or upper < lower or upper > queried_at or queried_at > now):
            raise ValueError
    except (ValueError, TypeError):
        return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_BOUNDS_MISMATCH", "reason": "Honcho query bounds do not cover the reviewed calendar day"}

    reviewed_sources = {source for episode in goal.episode_ids
                        for source in (reviewed_turn_sources or {}).get(episode, [])}
    reviewed_turn_bindings = [(episode, turn) for episode in goal.episode_ids
                              for turn in (reviewed_turn_provenance or {}).get(episode, [])]
    reviewed_turns = [turn for _, turn in reviewed_turn_bindings]
    reviewed_logical_turns = {turn.get("logical_turn_id") for turn in reviewed_turns
                              if isinstance(turn.get("logical_turn_id"), str)}
    reviewed_operations = {turn.get("operation_id") for turn in reviewed_turns
                           if isinstance(turn.get("operation_id"), str)}
    reviewed_camera_contexts = [
        context for turn in reviewed_turns
        if isinstance((context := turn.get("trusted_camera_initial_context")), dict)
    ]
    reviewed_logical_turns.update(context.get("logical_turn_id") for context in reviewed_camera_contexts
                                  if isinstance(context.get("logical_turn_id"), str))
    reviewed_operations.update(context.get("operation_id") for context in reviewed_camera_contexts
                               if isinstance(context.get("operation_id"), str))
    contextual_targets: dict[str, str | None] = {}
    for event in events:
        metadata = event.get("metadata", {})
        event_tenant = metadata.get("tenant_id") if event.get("unannotated") else event.get("owner_id")
        if (event["session_id"] != goal.session_id or event["workspace_id"] != goal.workspace_id
                or event["peer_id"] != goal.peer_id or not lower <= event["created_at"] <= upper
                or event_tenant != goal.owner_id):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SCOPE_MISMATCH",
                    "reason": "persisted message escaped queried owner, session, peer, workspace, or time scope"}

        # The Honcho read is scoped to the shared owner/workspace/session/peer and
        # can contain legitimate traffic from rotated gateway sessions. Only rows
        # with a possible identity edge to this goal need goal-level binding.
        source_id = metadata.get("source_message_id")
        reply_id = metadata.get("reply_to_source_message_id")
        trace_episode = (metadata.get("decision_trace_episode_id")
                         if event.get("unannotated") else event.get("trace_episode_id"))
        raw_trace = _dict(metadata.get("decision_trace"))
        nested_trace_episode = raw_trace.get("episode_id") if raw_trace else None
        if not event.get("unannotated"):
            nested_trace_episode = event.get("nested_trace_episode_id")
        logical_turn = metadata.get("logical_turn_id")
        operation = metadata.get("client_op_id")
        stable_meal_ids = (metadata.get("canonical_meal_id"), metadata.get("meal_id"))
        camera_contexts = [context for episode, turn in reviewed_turn_bindings
                           if episode == trace_episode
                           and isinstance((context := turn.get("trusted_camera_initial_context")), dict)]
        initial_identity_edges = [context for context in reviewed_camera_contexts
                                  if (isinstance(context.get("logical_turn_id"), str)
                                      and logical_turn == context.get("logical_turn_id"))
                                  or (isinstance(context.get("operation_id"), str)
                                      and operation == context.get("operation_id"))]
        raw_annotations = raw_trace.get("annotations") if raw_trace else None
        contextual_target = None
        if (not event.get("unannotated")
                and event["annotation"]["record_type"] in {"meal_correction", "meal_deletion"}):
            contextual_target = _trusted_context_target(goal, event, reviewed_turns, events)
            contextual_targets[event["event_id"]] = contextual_target
        relevant = (bool(initial_identity_edges)
                    or source_id == goal.source_message_id or reply_id == goal.source_message_id
                    or trace_episode in goal.episode_ids
                    or nested_trace_episode in goal.episode_ids
                    or logical_turn == goal.logical_turn_id or operation == goal.operation_id
                    or (isinstance(source_id, str) and source_id in reviewed_sources)
                    or (isinstance(logical_turn, str) and logical_turn in reviewed_logical_turns)
                    or (isinstance(operation, str) and operation in reviewed_operations)
                    or goal.canonical_meal_id in stable_meal_ids
                    or contextual_target == goal.source_message_id)
        is_contextual_edit = (not event.get("unannotated")
                              and event["annotation"]["record_type"] in {"meal_correction", "meal_deletion"})
        is_unresolved_edit = is_contextual_edit and not reply_id
        contextual_binding_claimed = (
            "selected_source" in metadata
            or any(
                turn.get("episode_id") == trace_episode
                and turn.get("source_message_id") == source_id
                and turn.get("logical_turn_id") == logical_turn
                and turn.get("operation_id") == operation
                and "nutrition_context_evidence" in (_dict(turn.get("gateway_final_metadata")) or {})
                for turn in reviewed_turns
            )
        )
        raw_target_relevant = (
            metadata.get("target_meal_id") == goal.canonical_meal_id
            or (_dict(metadata.get("selected_source")) or {}).get("append_source_message_id") == goal.source_message_id
        )
        if (is_contextual_edit and (is_unresolved_edit or contextual_binding_claimed)
                and contextual_target is None and (relevant or raw_target_relevant)):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TARGET_UNAVAILABLE",
                    "reason": "correction or deletion target is unresolved in bounded history"}
        if not relevant:
            continue
        if contextual_target is not None:
            if reply_id and reply_id != contextual_target:
                return {**base, "a1": "FAIL", "stage": "HONCHO_TARGET_MISMATCH",
                        "reason": "native reply target contradicts independently receipt-proven consumed occurrence"}
            contextual_turn = next(turn for turn in reviewed_turns
                if turn.get("episode_id") == event.get("trace_episode_id")
                and turn.get("source_message_id") == event.get("source_message_id")
                and turn.get("operation_id") == event.get("operation_id"))
            context_metadata = contextual_turn["gateway_final_metadata"]
            context_evidence = context_metadata["nutrition_context_evidence"]
            selected_source = _dict(metadata.get("selected_source"))
            persisted_target = metadata.get("target_meal_id")
            if (selected_source is None or not isinstance(persisted_target, str) or not persisted_target
                    or type(selected_source.get("schema_version")) is not int
                    or selected_source.get("schema_version") != 1
                    or selected_source.get("tenant_id") != goal.owner_id
                    or selected_source.get("source_principal") != goal.principal_id
                    or selected_source.get("gateway_session_id") != goal.gateway_session_id
                    or selected_source.get("is_private") is not True
                    or selected_source.get("is_forwarded") is not False
                    or selected_source.get("is_group") is not False):
                return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TARGET_UNAVAILABLE",
                        "reason": "persisted target receipt fields are missing, malformed, or outside the reviewed owner scope"}
            if selected_source.get("source_message_id") != context_evidence.get("photo_source_message_id"):
                return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SOURCE_MISMATCH",
                        "reason": "persisted photo identity conflicts with the independently receipt-proven occurrence"}
            expected_target_meal_id = derive_meal_id(
                tenant_id=goal.owner_id, source_principal=goal.principal_id,
                gateway_session_id=goal.gateway_session_id, source_message_id=contextual_target)
            if (contextual_target != goal.source_message_id
                    or context_evidence.get("target_meal_id") != expected_target_meal_id
                    or persisted_target != expected_target_meal_id
                    or selected_source.get("append_source_message_id") != contextual_target
                    or expected_target_meal_id != goal.canonical_meal_id):
                return {**base, "a1": "FAIL", "stage": "HONCHO_TARGET_MISMATCH",
                        "reason": "receipt-proven intended occurrence differs from the persisted target"}
        if event.get("malformed_annotation"):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_INVALID",
                    "reason": "goal-relevant persisted nutrition annotation is malformed"}
        if "decision_trace" in metadata and not isinstance(metadata["decision_trace"], dict):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_INVALID",
                    "reason": "goal-relevant persisted decision trace is malformed"}
        if initial_identity_edges:
            exact_initial_context = (
                len(initial_identity_edges) == 1 and len(camera_contexts) == 1
                and camera_contexts[0] == initial_identity_edges[0]
                and event.get("unannotated") is True
                and metadata.get("role") == "assistant"
                and source_id is None and reply_id is None
                and all(value is None for value in stable_meal_ids)
                and ("decision_trace" not in metadata or (
                    raw_trace is not None
                    and raw_trace.get("episode_id") == trace_episode
                    and isinstance(raw_annotations, dict) and not raw_annotations
                ))
                and trace_episode == initial_identity_edges[0].get("episode_id")
                and nested_trace_episode in (None, trace_episode)
                and initial_identity_edges[0].get("kind") == "initial_context"
                and initial_identity_edges[0].get("tenant_id") == goal.owner_id
                and initial_identity_edges[0].get("gateway_session_id") == goal.gateway_session_id
                and initial_identity_edges[0].get("recipient_principal") == goal.principal_id
                and logical_turn == initial_identity_edges[0].get("logical_turn_id")
                and operation == initial_identity_edges[0].get("operation_id")
                and operation == f"{logical_turn}:assistant"
                and metadata.get("source_principal") == initial_identity_edges[0].get("source_principal")
                and metadata.get("gateway_session_id") == goal.gateway_session_id
                and metadata.get("tenant_id") == goal.owner_id
            )
            if not exact_initial_context:
                return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TURN_BINDING_MISMATCH",
                        "reason": "goal-linked initial Camera context has contradictory identity"}
            continue
        if (metadata.get("gateway_session_id") not in (None, goal.gateway_session_id)):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SCOPE_MISMATCH",
                    "reason": "goal-relevant persisted message has a different gateway session"}
        relevant_principal = (metadata.get("source_principal") if event.get("unannotated")
                              else event.get("principal_id"))
        if relevant_principal != goal.principal_id:
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SCOPE_MISMATCH",
                    "reason": "goal-relevant persisted message has a different source principal"}
        if event.get("unannotated") and metadata.get("role") == "user":
            linked_user_turns = [turn for episode, turn in reviewed_turn_bindings
                if episode == trace_episode and turn.get("source_message_id") == source_id
                and turn.get("logical_turn_id") == logical_turn
                and turn.get("principal_id") == relevant_principal]
            if (len(linked_user_turns) != 1 or not isinstance(logical_turn, str)
                    or operation != f"{logical_turn}:user"
                    or metadata.get("tenant_id") != goal.owner_id
                    or metadata.get("gateway_session_id") != goal.gateway_session_id
                    or nested_trace_episode not in (None, trace_episode)):
                return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TURN_BINDING_MISMATCH",
                        "reason": "goal-linked user event lacks its exact recorded inbound turn binding"}
            continue
        if (goal.canonical_meal_id in stable_meal_ids
                and source_id != goal.source_message_id and reply_id != goal.source_message_id):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SOURCE_MISMATCH",
                    "reason": "reviewed stable meal identity is attached to a different source"}
        reviewed_turn_edge = (
            (isinstance(source_id, str) and source_id in reviewed_sources)
            or (isinstance(logical_turn, str) and logical_turn in reviewed_logical_turns)
            or (isinstance(operation, str) and operation in reviewed_operations))
        export_identity_matches = False
        if reviewed_turn_edge:
            linked_turns = [(episode, turn) for episode, turn in reviewed_turn_bindings
                            if (turn.get("source_message_id") == source_id
                                or turn.get("logical_turn_id") == logical_turn
                                or turn.get("operation_id") == operation)]
            exact_turns = [(episode, turn) for episode, turn in linked_turns
                           if (episode == trace_episode
                               and turn.get("source_message_id") == source_id
                               and turn.get("logical_turn_id") == logical_turn
                               and turn.get("operation_id") == operation
                               and turn.get("principal_id") == relevant_principal)]
            retry_turns = [pair for pair in linked_turns if pair not in exact_turns]
            export_identity_matches = (
                len(exact_turns) == 1
                and len(exact_turns) + len(retry_turns) == len(linked_turns)
                and all(_is_receipt_reconciled_retry(
                            goal, event, turn, exact_turns[0][1], relevant_principal)
                        for _, turn in retry_turns)
            )
            if not export_identity_matches:
                return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TURN_BINDING_MISMATCH",
                        "reason": "persisted source, episode, logical turn, or operation conflicts with reviewed export"}
        if nested_trace_episode is not None and nested_trace_episode != trace_episode:
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TURN_BINDING_MISMATCH",
                    "reason": "goal-relevant message has conflicting outer and nested trace episodes"}
        if (logical_turn == goal.logical_turn_id and source_id != goal.source_message_id):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SOURCE_MISMATCH",
                    "reason": "reviewed logical turn is attached to a different source message"}
        if (operation == goal.operation_id
                and (logical_turn != goal.logical_turn_id or source_id != goal.source_message_id)):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SOURCE_MISMATCH",
                    "reason": "reviewed operation is attached to a different source turn"}
        if ((trace_episode in goal.episode_ids or nested_trace_episode in goal.episode_ids)
                and source_id != goal.source_message_id and reply_id != goal.source_message_id
                and contextual_target != goal.source_message_id
                and not (event.get("unannotated") and export_identity_matches)):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TURN_BINDING_MISMATCH",
                    "reason": "reviewed trace episode is attached to a different source meal"}
        if event.get("unannotated"):
            principal = metadata.get("source_principal")
            exported_turn = next((turn for turn in
                (reviewed_turn_provenance or {}).get(trace_episode, [])
                if turn.get("source_message_id") == source_id), None)
            if (not isinstance(source_id, str) or not source_id or trace_episode not in goal.episode_ids
                    or source_id not in (reviewed_turn_sources or {}).get(trace_episode, [])
                    or not isinstance(logical_turn, str) or operation != f"{logical_turn}:assistant"
                    or principal != goal.principal_id or exported_turn is None
                    or exported_turn.get("logical_turn_id") != logical_turn
                    or exported_turn.get("operation_id") != operation
                    or exported_turn.get("principal_id") != principal
                    or (source_id == goal.source_message_id and
                        (trace_episode != goal.trace_episode_id or logical_turn != goal.logical_turn_id
                         or operation != goal.operation_id))):
                return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TURN_BINDING_MISSING",
                        "reason": "goal-linked unannotated turn lacks exact reviewed source and operation binding"}

    selected = []
    for event in events:
        if event.get("unannotated"):
            continue
        contextual_target = contextual_targets.get(event["event_id"])
        record_type = event["annotation"]["record_type"]
        if record_type in {"meal_correction", "meal_deletion"} and not event["reply_to_source_message_id"]:
            if contextual_target is None:
                continue
        if (event["root_source_message_id"] != goal.source_message_id
                and contextual_target != goal.source_message_id):
            continue
        if (event["owner_id"] != goal.owner_id or event["session_id"] != goal.session_id
                or event["gateway_session_id"] != goal.gateway_session_id
                or event["workspace_id"] != goal.workspace_id or event["principal_id"] != goal.principal_id
                or event["peer_id"] != goal.peer_id or event["trace_episode_id"] not in goal.episode_ids
                or event["nested_trace_episode_id"] != event["trace_episode_id"]
                or event["operation_id"] != f"{event['logical_turn_id']}:assistant"
                or not event["source_message_id"]):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SCOPE_MISMATCH",
                    "reason": "persisted source has wrong principal, workspace, peer, trace, or operation identity"}
        if event["trace_episode_id"] == goal.trace_episode_id:
            if (event["source_message_id"] != goal.source_message_id
                    or event["logical_turn_id"] != goal.logical_turn_id
                    or event["operation_id"] != goal.operation_id):
                return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_SOURCE_MISMATCH",
                        "reason": "root source turn differs from reviewed source, logical turn, or operation"}
        elif (reviewed_turn_sources is None
              or event["source_message_id"] not in reviewed_turn_sources.get(event["trace_episode_id"], [])):
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TURN_BINDING_MISSING",
                    "reason": "contributing persisted turn is not bound to its exported source message"}
        if reviewed_turn_provenance is not None:
            expected_turn = next((turn for turn in reviewed_turn_provenance.get(event["trace_episode_id"], [])
                                  if turn.get("source_message_id") == event["source_message_id"]), None)
            if (expected_turn is None or expected_turn.get("logical_turn_id") != event["logical_turn_id"]
                    or expected_turn.get("operation_id") != event["operation_id"]
                    or expected_turn.get("principal_id") != event["principal_id"]):
                return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TURN_BINDING_MISMATCH",
                        "reason": "persisted turn operation differs from gateway-derived exported identity"}
        elif event["trace_episode_id"] != goal.trace_episode_id:
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_TURN_BINDING_MISSING",
                    "reason": "correction turn has no exported gateway-derived operation identity"}
        selected.append(event)
    try:
        for event in selected:
            event["_created_at"] = event["created_at"]
        selected.sort(key=lambda item: (item["_created_at"], item["event_id"]))
    except (TypeError, ValueError):
        return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_BOUNDS_MISMATCH",
                "reason": "persisted event timestamp is invalid or outside query scope"}
    # Replays are idempotent; an ID collision with different raw event content is ambiguous.
    ids: dict[str, dict[str, Any]] = {}
    for event in selected:
        old = ids.setdefault(event["event_id"], event)
        if old != event:
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_AMBIGUOUS", "reason": "conflicting replay of persisted event id"}
    selected = list(ids.values())
    operation_events: dict[str, str] = {}
    for event in selected:
        previous = operation_events.setdefault(event["operation_id"], event["event_id"])
        if previous != event["event_id"]:
            return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_AMBIGUOUS",
                    "reason": "one gateway operation has multiple distinct committed nutrition events"}
    try:
        effective = _fold_events(selected, goal.meal_timezone)
    except ValueError as exc:
        return {**base, "a1": "INCONCLUSIVE", "stage": "HONCHO_AMBIGUOUS", "reason": str(exc)}
    if selected and effective.get("meal_date") == goal.meal_date.isoformat():
        original_annotation = selected[0]["annotation"]
        original_date = original_annotation.get("meal_date")
        if original_date is None and original_annotation.get("meal_at") is not None:
            original_date = _parse_time(original_annotation["meal_at"]).astimezone(
                ZoneInfo(goal.meal_timezone)).date().isoformat()
        if original_date is not None and original_date != effective.get("meal_date"):
            original_day = date.fromisoformat(original_date)
            original_start = datetime.combine(original_day, datetime.min.time(), ZoneInfo(goal.meal_timezone))
            corrected_day = date.fromisoformat(effective["meal_date"])
            corrected_start = datetime.combine(corrected_day, datetime.min.time(), ZoneInfo(goal.meal_timezone))
            day_ends = [
                datetime.combine(day + timedelta(days=1), datetime.min.time(), ZoneInfo(goal.meal_timezone))
                - timedelta(microseconds=1)
                for day in (original_day, corrected_day)
            ]
            required_end = min(max(day_ends), now.astimezone(ZoneInfo(goal.meal_timezone)))
            if t_start > min(original_start, corrected_start) or t_end < required_end:
                return {**base, "a1": "INCONCLUSIVE", "stage": "TELEGENT_BOUNDS_MISMATCH",
                        "reason": "canonical interval does not cover both verified local days through observation time"}
    if any(target is not None for target in contextual_targets.values()) and not source_count_present:
        return {**base, "a1": "INCONCLUSIVE", "stage": "CANONICAL_INVALID",
                "reason": "canonical read does not report complete source occurrence coverage"}
    if source_occurrence_count is not None and source_occurrence_count > 1:
        return {**base, "a1": "FAIL", "stage": "CANONICAL_MISMATCH",
                "reason": "complete canonical read contains duplicate rows for the reviewed meal source"}
    selected_ids = {event["event_id"] for event in selected}
    edge_rows = [row for row in canonical_rows
                 if row.get("latest_event_id") in selected_ids
                 or row.get("meal_id") == goal.canonical_meal_id
                 or row.get("source_message_id") == goal.source_message_id]
    selected_event_rows: dict[str, list[dict[str, Any]]] = {}
    for row in canonical_rows:
        if row.get("latest_event_id") in selected_ids:
            selected_event_rows.setdefault(row["latest_event_id"], []).append(row)
    if any(len(rows) > 1 for rows in selected_event_rows.values()):
        return {**base, "a1": "FAIL", "stage": "CANONICAL_MISMATCH",
                "reason": "one persisted source event appears under multiple canonical meal rows",
                "actual_event_ids": [event["event_id"] for event in selected]}
    for row in edge_rows:
        if (row.get("meal_id") == goal.canonical_meal_id
                and row.get("source_message_id") != goal.source_message_id):
            return {**base, "a1": "FAIL", "stage": "CANONICAL_MISMATCH",
                    "reason": "reviewed canonical meal id is currently bound to a different source",
                    "actual_event_ids": [event["event_id"] for event in selected],
                    "actual_latest_event_id": row.get("latest_event_id")}
        if (row.get("source_message_id") == goal.source_message_id
                and row.get("meal_id") != goal.canonical_meal_id):
            return {**base, "a1": "FAIL", "stage": "CANONICAL_MISMATCH",
                    "reason": "reviewed source is currently projected under a different canonical meal id",
                    "actual_event_ids": [event["event_id"] for event in selected],
                    "actual_latest_event_id": row.get("latest_event_id")}
        latest_event = next((event for event in selected
                             if event["event_id"] == row.get("latest_event_id")), None)
        if latest_event is not None and (
            row.get("meal_id") != goal.canonical_meal_id
            or row.get("source_message_id") != goal.source_message_id
            or latest_event.get("metadata", {}).get("target_meal_id", goal.canonical_meal_id)
            != goal.canonical_meal_id
        ):
            return {**base, "a1": "FAIL", "stage": "CANONICAL_MISMATCH",
                    "reason": "persisted goal event is projected under a different source or meal edge",
                    "actual_event_ids": [event["event_id"] for event in selected],
                    "actual_latest_event_id": row.get("latest_event_id")}
    if selected and effective["meal_date"] != goal.meal_date.isoformat():
        return {**base, "a1": "FAIL", "stage": "HONCHO_GOAL_MISMATCH",
                "reason": "effective persisted meal date differs from reviewed goal",
                "actual_event_ids": [event["event_id"] for event in selected],
                "actual_kcal": effective["energy_kcal_best"]}
    if effective["consumed"] != goal.expected_consumed:
        return {**base, "a1": "FAIL", "stage": "HONCHO_GOAL_MISMATCH",
                "reason": "effective persisted consumption state differs from reviewed goal",
                "actual_event_ids": [event["event_id"] for event in selected],
                "actual_kcal": effective["energy_kcal_best"]}
    if goal.expected_consumed:
        expected_kcal = goal.expected_kcal
        actual_kcal = effective["energy_kcal_best"]
        if (expected_kcal is None or not isinstance(actual_kcal, (int, float))
                or isinstance(actual_kcal, bool)
                or abs(actual_kcal - expected_kcal) > abs(expected_kcal) * goal.tolerance_fraction):
            return {**base, "a1": "FAIL", "stage": "HONCHO_GOAL_MISMATCH",
                    "reason": "effective persisted kcal differs from reviewed goal tolerance",
                    "actual_event_ids": [event["event_id"] for event in selected],
                    "actual_kcal": actual_kcal}

    # The canonical API identifies only its latest immutable event.
    canonical = telegent.get("meal")
    if canonical is None:
        if not goal.expected_consumed and not effective["consumed"]:
            return {**base, "a1": "PASS", "stage": "COMPLETE_ABSENCE",
                    "reason": "complete reads confirm current reviewed source is not consumed",
                    "actual_event_ids": [item["event_id"] for item in selected], "actual_kcal": 0,
                    "evidence_limitations": [
                        "wellness read does not prove an exact deletion tombstone, internal index keys, or aggregate balance",
                        "one bounded snapshot cannot prove unrelated meal records stayed unchanged"]}
        if not selected:
            return {**base, "a1": "PASS" if not goal.expected_consumed else "FAIL",
                    "stage": "COMPLETE_ABSENCE", "reason": "complete scoped reads show no persisted or canonical meal",
                    "actual_event_ids": [], "actual_kcal": None,
                    "evidence_limitations": [
                        "wellness read does not prove an exact deletion tombstone, internal index keys, or aggregate balance",
                        "one bounded snapshot cannot prove unrelated meal records stayed unchanged"]}
        age = (now - effective["latest_created_at"]).total_seconds()
        if age <= grace_seconds:
            return {**base, "a1": "PENDING", "stage": "HONCHO_SAVED_TELEGENT_PENDING", "reason": "persisted event is inside sync grace period",
                    "actual_event_ids": [item["event_id"] for item in selected]}
        return {**base, "a1": "FAIL", "stage": "TELEGENT_MISSING", "reason": "persisted meal is absent from canonical read after grace period",
                "actual_event_ids": [item["event_id"] for item in selected]}

    try:
        if not isinstance(canonical, dict) or not _valid_canonical_meal(canonical):
            return {**base, "a1": "INCONCLUSIVE", "stage": "CANONICAL_INVALID",
                    "reason": "canonical meal is missing required API fields or contains malformed values"}
        if canonical.get("user_id") != goal.canonical_owner_id:
            raise ValueError("canonical owner mismatch")
        if canonical.get("source_message_id") != goal.source_message_id:
            raise ValueError("canonical source mismatch")
        canonical_day = _canonical_local_day(canonical, goal.meal_timezone)
        if canonical.get("status") != "active":
            raise ValueError("canonical meal status mismatch")
        latest = canonical.get("latest_event_id")
        if not isinstance(latest, str) or latest not in {event["event_id"] for event in selected}:
            raise ValueError("canonical latest event is not the exact persisted source event")
        if canonical.get("meal_id") != goal.canonical_meal_id:
            raise ValueError("canonical meal id differs from reviewed source identity")
        latest_event = next(event for event in selected if event["event_id"] == latest)
        try:
            capture_time = _parse_time(canonical["capture_time"])
            canonical_queried_at = _parse_time(telegent.get("queried_at"))
            if capture_time != latest_event["_created_at"] or capture_time > canonical_queried_at:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            return {**base, "a1": "INCONCLUSIVE", "stage": "CANONICAL_PROVENANCE_MISMATCH",
                    "reason": "canonical capture time does not match latest persisted event or query time",
                    "actual_event_ids": [item["event_id"] for item in selected],
                    "actual_latest_event_id": latest}
        if latest != effective["latest_event_id"]:
            prior = next((item for item in effective["history"] if item["event_id"] == latest), None)
            prior_matches = prior is not None and (
                canonical.get("consumption_status") == ("consumed" if prior["consumed"] else "not_consumed")
                and (not prior["consumed"] or canonical.get("energy_kcal_best") == prior["energy_kcal_best"])
                and _canonical_local_day(canonical, goal.meal_timezone) == prior["meal_date"]
                and canonical.get("meal_date") == prior.get("meal_date_field")
                and _same_instant(canonical.get("meal_at"), prior.get("meal_at"))
                and canonical.get("revision") == prior.get("revision"))
            if prior_matches and (now - effective["latest_created_at"]).total_seconds() <= grace_seconds:
                return {**base, "a1": "PENDING", "stage": "HONCHO_SAVED_TELEGENT_PENDING",
                        "reason": "canonical still shows a verified prior revision inside sync grace",
                        "actual_event_ids": [event["event_id"] for event in selected],
                        "actual_latest_event_id": latest,
                        "actual_kcal": canonical.get("energy_kcal_best")}
            raise ValueError("canonical latest revision is stale relative to effective persisted source history")
        consumed = effective["consumed"]
        kcal = effective["energy_kcal_best"]
        if canonical_day != goal.meal_date.isoformat():
            raise ValueError("canonical meal date differs from the reviewed effective date")
        if canonical.get("meal_date") != effective.get("meal_date_field"):
            raise ValueError("canonical explicit meal_date differs from effective persisted source history")
        if canonical.get("revision") != effective["revision"]:
            raise ValueError("canonical revision differs from effective persisted source history")
        if not _same_instant(canonical.get("meal_at"), effective.get("meal_at")):
            raise ValueError("canonical meal time differs from effective persisted source history")
        expected_status = "consumed" if consumed else "not_consumed"
        if canonical.get("consumption_status") != expected_status:
            raise ValueError("canonical consumption state disagrees with persisted current event")
        if consumed and (kcal is None or canonical.get("energy_kcal_best") != kcal):
            raise ValueError("canonical kcal disagrees with persisted current event")
        if consumed != goal.expected_consumed:
            raise ValueError("effective current consumption state differs from expected goal")
        if consumed:
            expected = goal.expected_kcal
            assert expected is not None and isinstance(kcal, (int, float)) and not isinstance(kcal, bool)
            if abs(kcal - expected) > abs(expected) * goal.tolerance_fraction:
                raise ValueError("effective current kcal is outside frozen goal tolerance")
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError) as exc:
        return {**base, "a1": "FAIL", "stage": "CANONICAL_MISMATCH", "reason": str(exc),
                "actual_event_ids": [item["event_id"] for item in selected],
                "actual_latest_event_id": canonical.get("latest_event_id") if isinstance(canonical, dict) else None,
                "actual_kcal": canonical.get("energy_kcal_best") if isinstance(canonical, dict) else None}
    result = {**base, "a1": "PASS", "stage": "SAME_EVENT_PROJECTED",
              "reason": "same validated persisted latest event is current in canonical meal read",
              "actual_event_ids": [item["event_id"] for item in selected],
              "actual_latest_event_id": latest, "actual_kcal": kcal if consumed else 0,
              "canonical_contributing_event_ids": None,
              "evidence_limitations": [
                  "canonical meal record has latest_event_id but no contributing event_ids list",
                  "wellness read does not expose internal index keys or aggregate balance",
                  "one bounded snapshot cannot prove unrelated meal records stayed unchanged"]}
    return result


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("persisted created_at must be timezone aware")
    return result


def _same_instant(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return _parse_time(left).astimezone(timezone.utc) == _parse_time(right).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return False


def _utc_index_timestamp(value: datetime) -> str:
    """Match EvalEpisode's UTC JSON serialization used by the indexed text column."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _indexed_terminal_exception(episode_id: str, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Describe only a well-formed indexed exception followed by its terminal finish."""
    exceptions = [event for event in events if event.get("kind") == "exception"]
    finishes = [event for event in events if event.get("kind") == "episode_finished"]
    if len(exceptions) != 1 or len(finishes) != 1 or not events or finishes[0] is not events[-1]:
        return None
    exception, finish = exceptions[0], finishes[0]
    exception_payload = _dict(exception.get("payload")) or {}
    finish_payload = _dict(finish.get("payload")) or {}
    exception_id, finish_id = exception.get("_index_id"), finish.get("_index_id")
    if (type(exception_id) is not int or type(finish_id) is not int
            or exception_id <= 0 or finish_id <= exception_id
            or exception.get("episode_id") != episode_id or finish.get("episode_id") != episode_id
            or exception.get("is_error") is not True or finish.get("is_error") is not True
            or finish_payload.get("status") != "exception"
            or not isinstance(exception_payload.get("type"), str)
            or not exception_payload["type"].strip()
            or not isinstance(exception_payload.get("message"), str)):
        return None
    exception_position = events.index(exception)
    if any(
        event.get("kind") != "resource_snapshot"
        or (_dict(event.get("payload")) or {}).get("phase") != "world_after"
        for event in events[exception_position + 1:-1]
    ):
        return None
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "exception_event_id": exception_id,
        "finish_event_id": finish_id,
        "status": "exception",
    }


def _valid_terminal_failure_marker(value: object, episode_id: str) -> bool:
    """Check the narrow export marker shape before it can relax assistant-turn binding."""
    marker = _dict(value)
    return bool(
        marker is not None
        and set(marker) == {"schema_version", "episode_id", "exception_event_id", "finish_event_id", "status"}
        and type(marker.get("schema_version")) is int
        and marker.get("schema_version") == 1
        and marker.get("episode_id") == episode_id
        and type(marker.get("exception_event_id")) is int
        and marker["exception_event_id"] > 0
        and type(marker.get("finish_event_id")) is int
        and marker["finish_event_id"] > marker["exception_event_id"]
        and marker.get("status") == "exception"
    )


def _is_recorded_camera_initial_prompt(episode: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Recognize the recorded synthetic Camera envelope for display omission only."""
    episode_metadata = _dict(episode.get("metadata")) or {}
    context = _dict(episode_metadata.get("trusted_camera_context"))
    metadata = _dict(payload.get("metadata"))
    if (context is None or context.get("kind") != "initial_context"
            or context.get("episode_id") != episode.get("episode_id")
            or payload.get("sender_id") != "__camera__" or metadata is None
            or metadata.get("_synthetic") is not True
            or not isinstance(metadata.get("_camera_candidate_id"), str)
            or not metadata.get("_camera_candidate_id")
            or type(metadata.get("_camera_photo_id")) is not int
            or metadata.get("_camera_photo_id") <= 0):
        return False
    nested = _dict(metadata.get("metadata")) or {}
    return not any(key in metadata or key in nested for key in (
        "message_id", "source_message_id", "reply_to_message_id"))


def _exported_camera_context(episode: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    episode_metadata = _dict(episode.get("metadata")) or {}
    context = _dict(episode_metadata.get("trusted_camera_context"))
    if context is None:
        return None
    required = (context.get("episode_id"), context.get("candidate_id"), context.get("tenant_id"),
                context.get("gateway_session_id"), context.get("recipient_principal"))
    photo_id = context.get("native_photo_id")
    if (not all(isinstance(value, str) and value for value in required)
            or context.get("episode_id") != episode.get("episode_id")
            or context.get("gateway_session_id") != episode.get("session_id")
            or type(photo_id) is not int or photo_id <= 0):
        return None
    inbound = next((event for event in events if event.get("kind") == "inbound_message"), None)
    payload = _dict(inbound.get("payload")) if inbound else None
    channel_metadata = _dict(payload.get("metadata")) if payload else None
    if (payload is None or channel_metadata is None
            or channel_metadata.get("_camera_candidate_id") != context.get("candidate_id")):
        return None
    if context.get("kind") == "initial_context":
        logical_turn = context.get("logical_turn_id")
        operation = context.get("operation_id")
        source_principal = context.get("source_principal")
        if (payload.get("sender_id") != "__camera__" or channel_metadata.get("_synthetic") is not True
                or channel_metadata.get("message_id") is not None
                or channel_metadata.get("_camera_photo_id") != photo_id):
            return None
        derived = _derive_exported_turn_provenance(episode, payload)
        if (derived is None or not isinstance(logical_turn, str) or not logical_turn
                or operation != f"{logical_turn}:assistant"
                or derived.get("logical_turn_id") != logical_turn
                or derived.get("operation_id") != operation
                or source_principal != derived.get("principal_id")
                or source_principal != _inbound_principal(payload)):
            return None
        return {key: context[key] for key in (
            "kind", "episode_id", "candidate_id", "native_photo_id", "tenant_id",
            "gateway_session_id", "recipient_principal", "source_principal",
            "logical_turn_id", "operation_id")}
    if context.get("kind") != "owner_turn":
        return None
    turn = _dict(episode_metadata.get("trusted_camera_turn_provenance")) or {}
    source_id = channel_metadata.get("message_id")
    if isinstance(source_id, int) and not isinstance(source_id, bool):
        source_id = str(source_id)
    if (source_id != context.get("source_message_id")
            or context.get("source_message_id") != turn.get("source_message_id")
            or context.get("principal_id") != context.get("recipient_principal")
            or context.get("principal_id") != turn.get("principal_id")
            or context.get("logical_turn_id") != turn.get("logical_turn_id")
            or context.get("operation_id") != turn.get("operation_id")
            or context.get("operation_id") != f"{context.get('logical_turn_id')}:assistant"
            or _inbound_principal(payload) != context.get("principal_id")):
        return None
    return {key: context[key] for key in (
        "kind", "episode_id", "candidate_id", "native_photo_id", "tenant_id",
        "gateway_session_id", "recipient_principal", "source_message_id", "principal_id",
        "logical_turn_id", "operation_id")}


def _canonical_local_day(meal: dict[str, Any], timezone_name: str) -> str | None:
    value = meal.get("day") or meal.get("meal_date")
    if value is not None:
        if not isinstance(value, str):
            raise ValueError("canonical meal date must be an ISO date")
        return date.fromisoformat(value).isoformat()
    meal_at = meal.get("meal_at")
    if meal_at is None:
        return None
    return _parse_time(meal_at).astimezone(ZoneInfo(timezone_name)).date().isoformat()


def _fold_events(events: list[dict[str, Any]], meal_timezone: str = "UTC") -> dict[str, Any]:
    state: dict[str, Any] | None = None
    deleted = False
    history: list[dict[str, Any]] = []
    revision = 0
    for event in events:
        annotation = event["annotation"]
        kind = annotation["record_type"]
        if kind == "meal_observation":
            if state is not None:
                # Telegent deduplicates a second observation for the same root source.
                if event.get("root_source_message_id") == events[0].get("root_source_message_id"):
                    event["effective_revision"] = False
                    continue
                raise ValueError("multiple observations bind to one reviewed source")
            state = dict(annotation)
            event["effective_revision"] = True
            deleted = False
        elif kind == "meal_correction":
            if state is None:
                raise ValueError("correction has no persisted observation in scoped source history")
            if deleted:
                event["effective_revision"] = False
                continue
            changes = annotation.get("changed_fields")
            if not isinstance(changes, list) or not changes:
                raise ValueError("correction has no validated change mask")
            changed = False
            for field in changes:
                value = annotation.get(field)
                if field in state and state[field] != value:
                    changed = True
                state[field] = value
            if "meal_at" in changes and "meal_date" not in changes:
                changed = changed or state.get("meal_date") is not None
                state["meal_date"] = None
            if "meal_date" in changes and "meal_at" not in changes:
                # Telegent's date-only correction clears the old precise instant.
                changed = changed or state.get("meal_at") is not None
                state["meal_at"] = None
            if not changed:
                event["effective_revision"] = False
            else:
                event["effective_revision"] = True
        elif kind == "meal_deletion":
            if state is None:
                raise ValueError("deletion has no persisted observation in scoped source history")
            state["consumption_status"] = "not_consumed"
            state["energy_kcal_best"] = 0
            event["effective_revision"] = True
            deleted = True
        else:
            raise ValueError("non-meal nutrition event is ambiguous for this source goal")
        if event.get("effective_revision"):
            revision += 1
            date_value = state.get("meal_date")
            if date_value is None and state.get("meal_at") is not None:
                date_value = _parse_time(state["meal_at"]).astimezone(ZoneInfo(meal_timezone)).date().isoformat()
            history.append({"event_id": event["event_id"],
                            "consumed": state.get("consumption_status") == "consumed",
                            "energy_kcal_best": state.get("energy_kcal_best"), "meal_date": date_value,
                            "meal_date_field": state.get("meal_date"),
                            "meal_at": state.get("meal_at"), "revision": revision})
    if state is None:
        return {"consumed": False, "meal_date": None, "energy_kcal_best": 0,
                "latest_event_id": None, "history": history}
    effective_events = [event for event in events if event.get("effective_revision", True)]
    latest = effective_events[-1] if effective_events else events[0]
    meal_date = state.get("meal_date")
    if meal_date is None and state.get("meal_at") is not None:
        meal_date = _parse_time(state["meal_at"]).astimezone(ZoneInfo(meal_timezone)).date().isoformat()
    return {"consumed": state.get("consumption_status") == "consumed",
            "meal_date": meal_date,
            "meal_date_field": state.get("meal_date"),
            "energy_kcal_best": state.get("energy_kcal_best"),
            "meal_at": state.get("meal_at"), "revision": revision,
            "effective_event_ids": [event["event_id"] for event in effective_events],
            "latest_event_id": latest["event_id"],
            "latest_created_at": latest["_created_at"], "history": history}


def grade_manifest(manifest: Manifest, honcho: dict[str, Any], telegent: dict[str, Any], *,
                   now: datetime | None = None, grace_seconds: int = 300,
                   reviewed_turn_sources: dict[str, list[str]] | None = None,
                   reviewed_turn_provenance: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
    """Grade explicitly reviewed goals; output contains IDs and numeric evidence only."""
    if isinstance(grace_seconds, bool) or not isinstance(grace_seconds, int) or grace_seconds < 0:
        raise ValueError("grace_seconds cannot be negative")
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None or clock.utcoffset() is None:
        raise ValueError("grading time must be timezone-aware")
    if not isinstance(honcho, dict) or not isinstance(telegent, dict):
        return [{"case_id": goal.case_id, "expectation_origin": goal.expectation_origin,
                 "expectation_source": goal.expectation_source, "a1": "INCONCLUSIVE",
                 "stage": "SNAPSHOT_INVALID", "persistence_stage": "SNAPSHOT_INVALID",
                 "reason": "authoritative snapshots must be JSON objects", "a2": "NOT_RUN"}
                for goal in manifest.goals]
    results = []
    for goal in manifest.goals:
        result = _grade_one(goal, honcho, telegent, now=clock, grace_seconds=grace_seconds,
                            reviewed_turn_sources=reviewed_turn_sources,
                            reviewed_turn_provenance=reviewed_turn_provenance)
        result.setdefault("persistence_stage", result.get("stage", "UNKNOWN"))
        results.append(result)
    return results


def export_eval_dialogue(eval_root: str | Path, *, episode_ids: list[str] | None = None,
                         session_id: str | None = None, principal_id: str | None = None, since: str | None = None,
                         until: str | None = None, limit: int = 100) -> dict[str, Any]:
    """Read indexed episode/event offsets without initializing or writing EvalStore."""
    if (episode_ids is None) == (session_id is None):
        raise ValueError("supply episode_ids or a session_id range")
    if session_id:
        if not principal_id or not since or not until:
            raise ValueError("session range requires principal_id, since, and until")
        lower_bound = _parse_time(since)
        upper_bound = _parse_time(until)
        if lower_bound > upper_bound:
            raise ValueError("session range since must not be after until")
        if upper_bound - lower_bound > timedelta(days=31):
            raise ValueError("session range may not exceed 31 days")
        # Fractional UTC timestamps sort before the same whole-second key ending in Z.
        # The dot prefix includes all forms in this second; parsed bounds below stay exact.
        index_since = _utc_index_timestamp(lower_bound.replace(microsecond=0))[:-1] + "."
        index_until = _utc_index_timestamp(upper_bound.replace(microsecond=0) + timedelta(seconds=1))
    if not 1 <= limit <= MAX_EPISODES:
        raise ValueError("episode limit outside safe bound")
    root = Path(eval_root).expanduser().resolve()
    db_path = root / "evals.sqlite"
    if not db_path.is_file():
        raise ValueError("evals.sqlite does not exist")
    uri = db_path.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        if episode_ids is not None:
            ids = list(dict.fromkeys(episode_ids))
            if not ids or len(ids) > limit:
                raise ValueError("episode_ids must contain 1..limit entries")
            marks = ",".join("?" for _ in ids)
            rows = connection.execute(
                f"SELECT * FROM episodes WHERE episode_id IN ({marks}) ORDER BY created_at, episode_id", ids
            ).fetchall()
        else:
            query = "SELECT * FROM episodes WHERE session_id = ?"
            args: list[Any] = [session_id]
            if since:
                query += " AND created_at >= ?"
                args.append(index_since)
            if until:
                query += " AND created_at <= ?"
                args.append(index_until)
            query += " ORDER BY created_at, episode_id LIMIT ?"
            args.append(limit + 1)
            rows = connection.execute(query, args).fetchall()
            if len(rows) > limit:
                raise ValueError("session/time query exceeds episode limit")
            rows = [row for row in rows if lower_bound <= _parse_time(row["created_at"]) <= upper_bound]
    rows.sort(key=lambda row: (_parse_time(row["created_at"]), row["episode_id"]))
    if episode_ids is not None and len(rows) != len(set(episode_ids)):
        raise ValueError("one or more selected episode IDs are missing")
    episodes = []
    for row in rows:
        episode = _read_offset(root, row["jsonl_path"], row["jsonl_offset"])
        if (episode.get("episode_id") != row["episode_id"]
                or episode.get("session_id") != row["session_id"]
                or _parse_time(episode.get("created_at")) != _parse_time(row["created_at"])):
            raise ValueError("episode index does not match indexed JSONL record")
        episode_metadata = _dict(episode.get("metadata")) or {}
        inbound = _dict(episode_metadata.get("inbound")) or {}
        channel = inbound.get("channel")
        sender_id = inbound.get("sender_id")
        actual_principal = (f"{str(channel).strip().lower()}:{canonical_principal(str(channel), str(sender_id))}"
                            if isinstance(channel, str) and isinstance(sender_id, (str, int))
                            and not isinstance(sender_id, bool) else "")
        if session_id:
            if actual_principal != principal_id:
                continue
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            event_rows = connection.execute(
                "SELECT id,episode_id,kind,timestamp,tool_name,tool_call_id,is_error,jsonl_path,jsonl_offset "
                "FROM events WHERE episode_id=? ORDER BY timestamp,id",
                (row["episode_id"],)
            ).fetchall()
        events = []
        for item in event_rows:
            event = _read_offset(root, item["jsonl_path"], item["jsonl_offset"])
            if (item["episode_id"] != row["episode_id"] or event.get("episode_id") != item["episode_id"]
                    or event.get("kind") != item["kind"] or event.get("tool_name") != item["tool_name"]
                    or event.get("tool_call_id") != item["tool_call_id"]
                    or bool(event.get("is_error")) != bool(item["is_error"])
                    or _parse_time(event.get("timestamp")) != _parse_time(item["timestamp"])):
                raise ValueError("event index does not match indexed JSONL record")
            event["_index_id"] = item["id"]
            events.append(event)
        events.sort(key=lambda event: (_parse_time(event["timestamp"]), event["_index_id"]))
        camera_context_present = "trusted_camera_context" in episode_metadata
        camera_context = _exported_camera_context(episode, events)
        camera_context_invalid = camera_context_present and camera_context is None
        is_initial_camera_context = camera_context is not None and camera_context.get("kind") == "initial_context"
        initial_camera_prompt_id = next((event.get("_index_id") for event in events
            if event.get("kind") == "inbound_message"
            and _is_recorded_camera_initial_prompt(episode, _dict(event.get("payload")) or {})), None)
        turns = _dialogue(events, episode=episode, suppress_camera_initial_id=initial_camera_prompt_id)
        terminal_failure = _indexed_terminal_exception(episode["episode_id"], events)
        public_turns_valid = all(
            _public_turn_input_valid(event, episode)
            for event in events
            if event.get("kind") in {"inbound_message", "assistant_update", "gateway_final", "gateway_error"}
        )
        inbound_sources = []
        turn_provenance = []
        for event in events:
            if event.get("kind") != "inbound_message":
                continue
            payload = _dict(event.get("payload")) or {}
            metadata = _dict(payload.get("metadata")) or {}
            nested_metadata = _dict(metadata.get("metadata")) or {}
            source_id = metadata.get("message_id") or metadata.get("source_message_id") or nested_metadata.get("message_id")
            if isinstance(source_id, int) and not isinstance(source_id, bool):
                source_id = str(source_id)
            if isinstance(source_id, str) and source_id:
                inbound_sources.append(source_id)
            trusted_turn = _derive_exported_turn_provenance(episode, payload)
            if trusted_turn is not None:
                final_provenance = _export_gateway_final_provenance(events)
                if final_provenance is not None:
                    trusted_turn["gateway_final_metadata"] = final_provenance
                turn_provenance.append(trusted_turn)
        episodes.append({
            "episode": episode,
            "dialogue": turns,
            "dialogue_complete": bool(events) and events[-1].get("kind") == "episode_finished"
            and ((_dict(events[-1].get("payload")) or {}).get("status") in {"completed", "ok", "failed", "error"}
                 or terminal_failure is not None and not is_initial_camera_context)
            and public_turns_valid
            and not camera_context_invalid
            and (terminal_failure is not None and not is_initial_camera_context or any(
                event.get("kind") in {"gateway_final", "gateway_error"}
                and isinstance((_dict(event.get("payload")) or {}).get("text"), str)
                and bool((_dict(event.get("payload")) or {}).get("text"))
                for event in events
            ))
            and any(event.get("kind") == "inbound_message" for event in events)
            and (any(turn.get("role") == "user" for turn in turns)
                 or is_initial_camera_context and not any(turn.get("role") == "user" for turn in turns)),
            "source_message_ids": sorted(set(inbound_sources)),
            "turn_provenance": turn_provenance,
            "principal_id": actual_principal,
            **({"terminal_failure": terminal_failure} if terminal_failure is not None else {}),
            **({"trusted_camera_context": camera_context} if camera_context is not None else {}),
        })
    if not episodes:
        raise ValueError("selected eval range contains no episodes for the bound owner")
    return {"privacy": "private", "exported_at": datetime.now(timezone.utc).isoformat(), "episodes": episodes}


def validate_dialogue_binding(manifest: Manifest, export: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Require every reviewed target to bind to a complete exported user dialogue."""
    raw_episodes = export.get("episodes")
    if export.get("privacy") != "private" or not isinstance(raw_episodes, list):
        return {goal.case_id: {"complete": False, "reason": "dialogue export is malformed"}
                for goal in manifest.goals}
    by_id = {}
    for item in raw_episodes:
        episode = _dict(item.get("episode")) if isinstance(item, dict) else None
        if episode and isinstance(episode.get("episode_id"), str):
            if episode["episode_id"] in by_id:
                return {goal.case_id: {"complete": False, "reason": "duplicate episode in dialogue export"}
                        for goal in manifest.goals}
            by_id[episode["episode_id"]] = item
    result = {}
    for goal in manifest.goals:
        missing = [episode_id for episode_id in goal.episode_ids if episode_id not in by_id]
        incomplete = []
        for episode_id in goal.episode_ids:
            if episode_id not in by_id:
                continue
            dialogue = by_id[episode_id].get("dialogue")
            roles = [turn.get("role") for turn in dialogue if isinstance(turn, dict)] if isinstance(dialogue, list) else []
            episode_data = _dict(by_id[episode_id].get("episode")) or {}
            terminal_failure = _valid_terminal_failure_marker(by_id[episode_id].get("terminal_failure"), episode_id)
            camera_context = _dict(by_id[episode_id].get("trusted_camera_context")) or {}
            initial_camera_context = camera_context.get("kind") == "initial_context"
            inbound = _dict((_dict(episode_data.get("metadata")) or {}).get("inbound")) or {}
            derived_initial = (_derive_exported_turn_provenance(episode_data, inbound)
                               if initial_camera_context else None)
            initial_identity_valid = (
                not initial_camera_context or (
                    derived_initial is not None
                    and camera_context.get("source_principal") == derived_initial.get("principal_id")
                    and camera_context.get("logical_turn_id") == derived_initial.get("logical_turn_id")
                    and camera_context.get("operation_id") == derived_initial.get("operation_id")
                    and camera_context.get("operation_id") == f"{camera_context.get('logical_turn_id')}:assistant"
                    and camera_context.get("source_principal") == _inbound_principal(inbound)
                )
            )
            public_roles_valid = ("assistant" in roles and "user" not in roles if initial_camera_context
                                  else "user" in roles and ("assistant" in roles or terminal_failure))
            principal_valid = (by_id[episode_id].get("principal_id") == goal.principal_id
                               if not initial_camera_context else
                               camera_context.get("recipient_principal") == goal.principal_id
                               and camera_context.get("tenant_id") == goal.owner_id
                               and initial_identity_valid)
            if (by_id[episode_id].get("dialogue_complete") is not True
                    or ("terminal_failure" in by_id[episode_id] and not terminal_failure)
                    or not public_roles_valid
                    or episode_data.get("session_id") != goal.gateway_session_id
                    or not principal_valid
                    or (_dict(episode_data.get("metadata")) or {}).get("workspace") != goal.eval_workspace):
                incomplete.append(episode_id)
        bound_sources: set[str] = set()
        for episode_id in goal.episode_ids:
            if episode_id not in by_id:
                continue
            source_values = by_id[episode_id].get("source_message_ids")
            if isinstance(source_values, list):
                bound_sources.update(source_id for source_id in source_values
                                     if isinstance(source_id, str))
        source_bound = goal.source_message_id in bound_sources
        turn_sources = {}
        turn_provenance = {}
        for episode_id in goal.episode_ids:
            if episode_id not in by_id:
                continue
            values = by_id[episode_id].get("source_message_ids", [])
            turn_sources[episode_id] = sorted({value for value in values if isinstance(value, str)})
            provenance = by_id[episode_id].get("turn_provenance", [])
            turn_provenance[episode_id] = [item for item in provenance if isinstance(item, dict)]
        source_bound = goal.source_message_id in turn_sources.get(goal.trace_episode_id, [])
        root_turn = next((item for item in turn_provenance.get(goal.trace_episode_id, [])
                          if item.get("source_message_id") == goal.source_message_id), None)
        root_identity_bound = (root_turn is not None
                               and root_turn.get("logical_turn_id") == goal.logical_turn_id
                               and root_turn.get("operation_id") == goal.operation_id
                               and root_turn.get("principal_id") == goal.principal_id)
        selected_camera_contexts = [(_dict(by_id[episode_id].get("trusted_camera_context")) or {})
                                    for episode_id in goal.episode_ids if episode_id in by_id
                                    and by_id[episode_id].get("trusted_camera_context") is not None]
        initial_contexts = [item for item in selected_camera_contexts if item.get("kind") == "initial_context"]
        root_export = by_id.get(goal.trace_episode_id, {})
        root_camera_context = _dict(root_export.get("trusted_camera_context")) or {}
        root_episode = _dict(root_export.get("episode")) or {}
        root_episode_metadata = _dict(root_episode.get("metadata")) or {}
        camera_owner_turn = (
            root_camera_context.get("kind") == "owner_turn"
            or "trusted_camera_turn_provenance" in root_episode_metadata
        )
        camera_context_reason = None
        matching_initials = []
        for episode_id in goal.episode_ids:
            if episode_id not in by_id:
                continue
            initial = _dict(by_id[episode_id].get("trusted_camera_context")) or {}
            if (initial.get("kind") == "initial_context"
                    and initial.get("candidate_id") == root_camera_context.get("candidate_id")
                    and initial.get("native_photo_id") == root_camera_context.get("native_photo_id")
                    and initial.get("tenant_id") == goal.owner_id == root_camera_context.get("tenant_id")
                    and initial.get("gateway_session_id") == goal.gateway_session_id
                        == root_camera_context.get("gateway_session_id")
                    and initial.get("recipient_principal") == goal.principal_id
                        == root_camera_context.get("recipient_principal")):
                matching_initials.append((episode_id, initial))
        if camera_owner_turn:
            initial_episode_id, initial = matching_initials[0] if len(matching_initials) == 1 else (None, {})
            initial_episode = _dict(by_id.get(initial_episode_id, {}).get("episode")) or {}
            chronology_valid = (
                initial_episode_id is not None
                and _parse_time(initial_episode.get("created_at"))
                < _parse_time(root_episode.get("created_at"))
            )
            camera_context_bound = (
                len(matching_initials) == 1 and root_camera_context.get("kind") == "owner_turn"
                and chronology_valid
                and root_camera_context.get("source_message_id") == goal.source_message_id
                and root_camera_context.get("logical_turn_id") == goal.logical_turn_id
                and root_camera_context.get("operation_id") == goal.operation_id
            )
            if not camera_context_bound:
                if not matching_initials:
                    camera_context_reason = (
                        "include the initial Camera context episode matching this photo receipt"
                    )
                elif len(matching_initials) > 1:
                    camera_context_reason = "multiple matching initial Camera contexts make the source ambiguous"
                else:
                    camera_context_reason = "matching initial Camera context is incomplete or out of order"
        elif initial_contexts:
            camera_context_bound = False
            camera_context_reason = "initial Camera context is not bound to a Camera owner turn"
        else:
            camera_context_bound = True
        good = not missing and not incomplete and source_bound and root_identity_bound and camera_context_bound
        if good and initial_contexts:
            for episode_id in goal.episode_ids:
                if episode_id not in by_id:
                    continue
                context = _dict(by_id[episode_id].get("trusted_camera_context")) or {}
                if context.get("kind") == "initial_context":
                    turn_provenance.setdefault(episode_id, []).append(
                        {"trusted_camera_initial_context": context}
                    )
        result[goal.case_id] = {"complete": good,
                                "reason": ("reviewed source is bound to complete exported dialogue" if good else
                                           camera_context_reason or
                                           "missing episodes, partial transcript, or source-message mismatch"),
                                "episode_ids": goal.episode_ids,
                                "reviewed_turn_sources": turn_sources,
                                "reviewed_turn_provenance": turn_provenance}
    return result


def _read_offset(root: Path, relative: str, offset: int) -> dict[str, Any]:
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("indexed JSONL path escapes eval root")
    with path.open("rb") as stream:
        stream.seek(int(offset))
        line = stream.readline()
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("indexed JSONL record is not an object")
    return value


def _attachment_only_count(episode: dict[str, Any], payload: dict[str, Any]) -> int | None:
    """Return the recorder-authenticated attachment count for an empty human input."""
    if payload.get("user_text") != "":
        return None
    event_count = payload.get("media_count")
    inbound = _dict((_dict(episode.get("metadata")) or {}).get("inbound")) or {}
    episode_count = inbound.get("media_count")
    if (type(event_count) is not int or event_count <= 0
            or type(episode_count) is not int or episode_count != event_count):
        return None
    return event_count


def _public_turn_input_valid(event: dict[str, Any], episode: dict[str, Any]) -> bool:
    payload = _dict(event.get("payload")) or {}
    text = payload.get("user_text" if event.get("kind") == "inbound_message" else "text")
    if isinstance(text, str) and text:
        return True
    return (event.get("kind") == "inbound_message" and text == ""
            and _attachment_only_count(episode, payload) is not None)


def _dialogue(events: list[dict[str, Any]], *, episode: dict[str, Any],
              suppress_camera_initial_id: Any = None) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for event in events:
        kind = event.get("kind")
        payload = _dict(event.get("payload")) or {}
        if kind == "inbound_message":
            if suppress_camera_initial_id is not None and event.get("_index_id") == suppress_camera_initial_id:
                continue
            text = payload.get("user_text")
            if isinstance(text, str) and text:
                turns.append({"role": "user", "text": text})
            elif count := _attachment_only_count(episode, payload):
                noun = "attachment" if count == 1 else "attachments"
                turns.append({"role": "user", "text": f"Sent {count} {noun}."})
        elif kind in {"assistant_update", "gateway_final", "gateway_error"}:
            text = payload.get("text")
            if isinstance(text, str) and text:
                turns.append({"role": "assistant", "text": text})
        elif kind == "tool_completed":
            name = event.get("tool_name")
            output = payload.get("output")
            if isinstance(output, str) and output and name in {"get_wellness_data", "record_nutrition"}:
                turns.append({"role": "tool", "text": output})
    return turns


def _export_gateway_final_provenance(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Export only recorder metadata that can bind nutrition persistence evidence."""
    finals = [event for event in events if event.get("kind") == "gateway_final"]
    if len(finals) != 1:
        return None
    payload = _dict(finals[0].get("payload")) or {}
    metadata = _dict(payload.get("metadata"))
    if metadata is None:
        return None
    exported: dict[str, Any] = {}
    for key in (
        "nutrition_append_event_id", "nutrition_sync_status", "nutrition_committed_annotation",
        "nutrition_model_proposal_annotation", "nutrition_proposal_matches_committed",
        "nutrition_consumed_occurrence", "nutrition_context_evidence",
    ):
        if key in metadata:
            exported[key] = metadata[key]
    executions = [event for event in events if event.get("kind") == "trace_finalization"
                  and not event.get("is_error")]
    if len(executions) == 1:
        payload = _dict(executions[0].get("payload")) or {}
        annotations = _dict(payload.get("annotations"))
        nutrition = _dict(annotations.get("nutrition")) if annotations is not None else None
        try:
            validated = NutritionAnnotationV2.model_validate(nutrition)
        except (TypeError, ValueError):
            validated = None
        if validated is not None:
            exported["nutrition_finalization"] = {
                "schema_version": 1,
                "annotation": validated.model_dump(
                    mode="json", exclude_unset=validated.record_type == "meal_correction"
                ),
            }
    return exported or None


def _derive_exported_turn_provenance(episode: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recompute the trusted ordinary-turn operation from its recorded inbound envelope."""
    inbound_metadata = _dict(payload.get("metadata"))
    episode_metadata = _dict(episode.get("metadata")) or {}
    trusted = _dict(episode_metadata.get("trusted_camera_turn_provenance"))
    camera_context = _dict(episode_metadata.get("trusted_camera_context")) or {}
    initial_camera_context = camera_context.get("kind") == "initial_context"
    if initial_camera_context and "trusted_camera_turn_provenance" in episode_metadata:
        return None
    if trusted is not None:
        source = inbound_metadata.get("message_id") if inbound_metadata else None
        if isinstance(source, int) and not isinstance(source, bool):
            source = str(source)
        if (trusted.get("episode_id") == episode.get("episode_id")
                and trusted.get("source_message_id") == source
                and trusted.get("logical_turn_id")
                and trusted.get("operation_id") == f"{trusted.get('logical_turn_id')}:assistant"
                and trusted.get("principal_id") == _inbound_principal(payload)):
            return {key: trusted[key] for key in
                    ("source_message_id", "logical_turn_id", "operation_id", "principal_id")} | {
                        "episode_id": trusted["episode_id"]}
        return None
    if initial_camera_context and inbound_metadata and "_camera_turn_id" in inbound_metadata:
        return None
    if inbound_metadata and not initial_camera_context and (
            "_camera_turn_id" in inbound_metadata or "_camera_authority" in inbound_metadata):
        return None
    channel, sender_id, chat_id = payload.get("channel"), payload.get("sender_id"), payload.get("chat_id")
    timestamp = payload.get("timestamp")
    if (inbound_metadata is None or not isinstance(channel, str) or not channel
            or not isinstance(sender_id, str) or not sender_id
            or not isinstance(chat_id, str) or not chat_id):
        return None
    try:
        from openharness.channels.bus.events import InboundMessage
        from ohmo.gateway.memory_gate import MemoryScope
        from ohmo.gateway.runtime import _build_conversation_turn_metadata
        from ohmo.gateway.turn_context import build_turn_context

        recompute_metadata = dict(inbound_metadata)
        if initial_camera_context:
            recompute_metadata.pop("_camera_authority", None)
            recompute_metadata.pop("_camera_turn_id", None)
        message = InboundMessage(channel=channel, sender_id=sender_id, chat_id=chat_id,
            content=str(payload.get("user_text") or ""), timestamp=_parse_time(timestamp),
            metadata=recompute_metadata)
        session = episode.get("session_id")
        if not isinstance(session, str) or not session:
            return None
        turn_context = build_turn_context(message, session_id=session)
        logical_turn_id, _, assistant_metadata = _build_conversation_turn_metadata(
            turn_ctx=turn_context, message=message, scope=MemoryScope("export", ()))
        source_id = assistant_metadata.get("source_message_id")
        operation_id = assistant_metadata.get("client_op_id")
        principal_id = assistant_metadata.get("source_principal")
        if (not isinstance(operation_id, str)
                or not isinstance(principal_id, str) or not principal_id):
            return None
        if initial_camera_context:
            if (source_id is not None or sender_id != "__camera__"
                    or inbound_metadata.get("_synthetic") is not True):
                return None
            return {"logical_turn_id": logical_turn_id, "operation_id": operation_id,
                    "principal_id": principal_id, "episode_id": episode["episode_id"]}
        if not isinstance(source_id, str) or not source_id:
            return None
        return {"source_message_id": source_id, "logical_turn_id": logical_turn_id,
                "operation_id": operation_id, "principal_id": principal_id,
                "episode_id": episode["episode_id"]}
    except (KeyError, TypeError, ValueError):
        return None


def _inbound_principal(payload: dict[str, Any]) -> str | None:
    channel, sender_id = payload.get("channel"), payload.get("sender_id")
    if not isinstance(channel, str) or not isinstance(sender_id, str):
        return None
    return f"{channel.strip().lower()}:{canonical_principal(channel, sender_id)}"


async def read_honcho_messages(*, base_url: str, api_key: str, workspace: str, session: str,
                               owner_id: str, since: datetime, until: datetime, peer_id: str,
                               page_size: int = 100, max_pages: int = 100) -> dict[str, Any]:
    """Fetch all actual persisted messages in one bounded owner/session/time window."""
    if since.tzinfo is None or until.tzinfo is None or since > until:
        raise ValueError("Honcho bounds must be ordered timezone-aware datetimes")
    if until - since > timedelta(days=31):
        raise ValueError("Honcho audit window may not exceed 31 days")
    if not 1 <= page_size <= 100 or not 1 <= max_pages <= 100:
        raise ValueError("Honcho page bounds outside safe limits")
    try:
        async with HonchoClient(base_url, api_key, workspace, timeout=20) as client:
            messages = await client.list_messages_in_window(
                session, expected_peer_id=peer_id, since=since, until=until,
                page_size=page_size, max_pages=max_pages,
            )
        if len(messages) > MAX_MESSAGES:
            return {"complete": False, "error": "Honcho result exceeds message bound", "messages": []}
        return {"complete": True, "workspace_id": workspace, "session_id": session,
                "owner_id": owner_id, "since": since.isoformat(), "until": until.isoformat(),
                "queried_at": datetime.now(timezone.utc).isoformat(), "messages": messages}
    except (HonchoError, httpx.HTTPError, ValueError):
        # Exception text can contain service details; never copy it into reports.
        return {"complete": False, "error": "Honcho read unavailable or incomplete", "messages": []}


async def read_telegent_wellness(*, owner_login: str, start: datetime, end: datetime,
                                 server_config: object | None = None, server_name: str = "telegent",
                                 mcp_url: str | None = None, token: str | None = None) -> dict[str, Any]:
    """Call get_wellness_data via configured OAuth MCP manager, or synthetic static transport tests."""
    if start.tzinfo is None or end.tzinfo is None or end < start:
        raise ValueError("Telegent interval must be ordered and timezone-aware")
    if end - start > timedelta(days=31):
        raise ValueError("Telegent interval may not exceed 31 days")
    try:
        arguments = {"params": {"login": owner_login, "start": start.isoformat(), "end": end.isoformat()}}
        if server_config is not None:
            from openharness.mcp.client import McpClientManager
            from openharness.mcp.types import McpHttpServerConfig

            if not isinstance(server_config, McpHttpServerConfig):
                return {"complete": False, "error": "selected Telegent MCP server is not HTTP"}
            manager = McpClientManager({server_name: server_config})
            try:
                await manager.connect_all()
                statuses = manager.list_statuses()
                if not statuses or statuses[0].state != "connected":
                    return {"complete": False, "error": "Telegent MCP server unavailable"}
                tool_result = await manager.call_tool_result(server_name, "get_wellness_data", arguments)
                raw_output = tool_result.output
                tool_error = tool_result.is_error
            finally:
                await manager.close()
        elif mcp_url and token:
            # Kept for isolated synthetic transport tests. The live CLI never selects this branch.
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            async with streamablehttp_client(mcp_url, headers={"Authorization": f"Bearer {token}"}) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("get_wellness_data", arguments)
            tool_error = getattr(result, "isError", False)
            raw_output = None
            if not tool_error:
                content = getattr(result, "content", None)
                if isinstance(content, list):
                    raw_output = next((block.text for block in content if isinstance(getattr(block, "text", None), str)), None)
                if raw_output is None:
                    structured = getattr(result, "structuredContent", None)
                    raw_output = json.dumps(structured) if isinstance(structured, dict) else None
        else:
            return {"complete": False, "error": "configured Telegent MCP server is required"}
        if tool_error or not isinstance(raw_output, str):
            return {"complete": False, "error": "Telegent canonical read failed"}
        parsed = json.loads(raw_output)
        interval = parsed.get("interval") if isinstance(parsed, dict) else None
        meals = parsed.get("nutrition_records") if isinstance(parsed, dict) else None
        unassigned = parsed.get("nutrition_unassigned_records") if isinstance(parsed, dict) else None
        if (not isinstance(parsed, dict) or not isinstance(interval, dict)
                or not isinstance(meals, list) or not isinstance(unassigned, list)):
            return {"complete": False, "error": "Telegent canonical response incomplete"}
        normalized_login = _normalize_login(owner_login)
        if (parsed.get("nutrition_status") != "complete" or not isinstance(parsed.get("user_id"), str)
                or not parsed["user_id"] or parsed.get("login") != normalized_login
                or _parse_time(interval.get("start")) != start
                or _parse_time(interval.get("end")) != end):
            return {"complete": False, "error": "Telegent nutrition read is stale or unavailable"}
        return {"complete": True, "user_id": parsed["user_id"], "login": parsed["login"],
                "start": interval["start"], "end": interval["end"],
                "queried_at": datetime.now(timezone.utc).isoformat(),
                "meals": meals, "unassigned": unassigned}
    except Exception:
        # Keep token-bearing transport/library exception strings out of outputs.
        return {"complete": False, "error": "Telegent MCP read unavailable or malformed"}


def bind_wellness_snapshot(snapshot: dict[str, Any], *, goal: Goal) -> dict[str, Any]:
    """Select the goal meal and retain every validated row from its scoped read."""
    if not isinstance(snapshot, dict):
        return {"complete": False, "error": "canonical snapshot is not an object"}
    scope = {key: snapshot.get(key) for key in ("user_id", "login", "start", "end", "queried_at")}
    if (snapshot.get("complete") is not True
            or snapshot.get("user_id") != goal.canonical_owner_id
            or snapshot.get("login") != _normalize_login(goal.canonical_login)):
        return {"complete": False}
    meals = snapshot.get("meals")
    unassigned = snapshot.get("unassigned")
    if not isinstance(meals, list) or not isinstance(unassigned, list):
        return {"complete": False}
    try:
        start = _parse_time(snapshot.get("start"))
        end = _parse_time(snapshot.get("end"))
        day_start = datetime.combine(goal.meal_date, datetime.min.time(), ZoneInfo(goal.meal_timezone))
        queried_at = _parse_time(snapshot.get("queried_at"))
        if (start > day_start or end < goal.trajectory_as_of.astimezone(timezone.utc)
                or start > end or end > queried_at or queried_at < end):
            return {"complete": False, "error": "canonical interval omits reviewed goal day"}
    except (ValueError, TypeError):
        return {"complete": False, "error": "canonical interval missing or invalid"}
    if any(not _valid_canonical_meal(meal) for meal in meals) or any(
        not _valid_canonical_meal(meal, allow_unassigned=True) for meal in unassigned
    ):
        return {"complete": False, **scope, "error": "canonical response contains malformed meal rows"}
    all_canonical_rows = [
        {**meal, "user_id": snapshot["user_id"]} for meal in [*meals, *unassigned]
    ]
    source_matches = [meal for meal in [*meals, *unassigned] if isinstance(meal, dict)
                      and meal.get("source_message_id") == goal.source_message_id]
    matches = [meal for meal in meals if isinstance(meal, dict)
               and meal.get("source_message_id") == goal.source_message_id]
    other_meals = [meal for meal in meals if isinstance(meal, dict)
                   and meal.get("day") == goal.meal_date.isoformat()
                   and meal.get("source_message_id") != goal.source_message_id]
    if len(source_matches) > 1:
        return {"complete": True, **scope, "meal": None,
                "source_occurrence_count": len(source_matches), "other_meals": other_meals,
                "canonical_meals": all_canonical_rows}
    if not source_matches:
        same_id_rows = [meal for meal in meals if meal.get("meal_id") == goal.canonical_meal_id]
        if any(isinstance(meal.get("source_message_id"), str)
               and meal.get("source_message_id") != goal.source_message_id
               for meal in same_id_rows):
            return {"complete": True, **scope, "meal": None,
                    "source_occurrence_count": 0, "other_meals": other_meals,
                    "canonical_meals": all_canonical_rows}
        if same_id_rows:
            return {"complete": False, **scope, "error": "canonical source identity is missing for the reviewed meal"}
        return {"complete": True, **scope, "meal": None,
                "source_occurrence_count": 0, "other_meals": other_meals,
                "canonical_meals": all_canonical_rows}
    if source_matches[0] in unassigned:
        return {"complete": True, **scope, "meal": {"user_id": goal.canonical_owner_id,
                "source_message_id": goal.source_message_id, "status": "unassigned"},
                "source_occurrence_count": 1, "other_meals": other_meals,
                "canonical_meals": all_canonical_rows}
    return {"complete": True, **scope, "meal": {**matches[0], "user_id": snapshot["user_id"]},
            "source_occurrence_count": 1, "other_meals": other_meals,
            "canonical_meals": all_canonical_rows}


def _valid_canonical_meal(meal: Any, *, allow_unassigned: bool = False) -> bool:
    if not isinstance(meal, dict):
        return False
    nullable_fields = ("source_message_id", "day", "meal_at", "meal_date", "ingest_source",
                       "confirmation_required", "reply_to_source_message_id", "received_at",
                       "source_message_at")
    if any(field not in meal for field in nullable_fields):
        return False
    for field in ("meal_id", "latest_event_id", "confidence", "consumption_status"):
        if not isinstance(meal.get(field), str) or not meal[field]:
            return False
    if (isinstance(meal.get("revision"), bool) or not isinstance(meal.get("revision"), int)
            or meal["revision"] < 1 or not isinstance(meal.get("provisional"), bool)
            or not isinstance(meal.get("is_estimate"), bool)
            or not isinstance(meal.get("is_forwarded"), bool)):
        return False
    if meal.get("source_message_id") is not None and not isinstance(meal.get("source_message_id"), str):
        return False
    if meal.get("status") != ("unassigned" if allow_unassigned else "active"):
        return False
    if meal.get("confidence") not in {"low", "medium", "high"}:
        return False
    if meal.get("consumption_status") not in {"unknown", "consumed", "planned", "not_consumed"}:
        return False
    try:
        _parse_time(meal.get("capture_time"))
    except (TypeError, ValueError):
        return False
    for field in ("meal_at", "received_at", "source_message_at"):
        if meal.get(field) is not None:
            try:
                _parse_time(meal[field])
            except (TypeError, ValueError):
                return False
    for field in ("energy_kcal_min", "energy_kcal_max", "energy_kcal_best",
                  "protein_g", "fat_g", "carbohydrate_g"):
        value = meal.get(field)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value < 0):
            return False
    for field in ("basis", "items", "assumptions", "warnings"):
        if not isinstance(meal.get(field), list):
            return False
    if any(not isinstance(value, str) or not value for field in ("basis", "assumptions", "warnings")
           for value in meal[field]):
        return False
    for item in meal["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            return False
        if not isinstance(item.get("quantity_text"), str) or not item["quantity_text"]:
            return False
        for field in ("energy_kcal_min", "energy_kcal_max", "energy_kcal_best"):
            value = item.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0):
                return False
    if (meal.get("day") is not None and meal.get("meal_date") is not None
            and meal["day"] != meal["meal_date"]):
        return False
    day = meal.get("day") if meal.get("day") is not None else meal.get("meal_date")
    if day is not None:
        try:
            date.fromisoformat(day)
        except (TypeError, ValueError):
            return False
    elif allow_unassigned and meal.get("status") == "unassigned":
        pass
    elif meal.get("meal_at") is not None:
        try:
            _parse_time(meal["meal_at"])
        except (TypeError, ValueError):
            return False
    else:
        return False
    return True


def main() -> None:
    """Run one bounded private audit from snapshots or the two read-only APIs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dialogue-export", type=Path, required=True)
    parser.add_argument("--honcho-snapshot", type=Path)
    parser.add_argument("--telegent-snapshot", type=Path)
    parser.add_argument("--honcho-base-url")
    parser.add_argument("--honcho-workspace")
    parser.add_argument("--honcho-session")
    parser.add_argument("--honcho-owner")
    parser.add_argument("--principal-id")
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--telegent-server", default="telegent")
    parser.add_argument("--telegent-login")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--grace-seconds", type=int, default=300)
    args = parser.parse_args()
    manifest = Manifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    dialogue_export = json.loads(args.dialogue_export.read_text(encoding="utf-8"))
    dialogue_binding = validate_dialogue_binding(manifest, dialogue_export)
    if bool(args.honcho_snapshot) != bool(args.telegent_snapshot):
        parser.error("both snapshot paths must be supplied together")
    if args.honcho_snapshot:
        honcho = json.loads(args.honcho_snapshot.read_text(encoding="utf-8"))
        telegent = json.loads(args.telegent_snapshot.read_text(encoding="utf-8"))
        if not isinstance(honcho, dict) or not isinstance(telegent, dict):
            parser.error("snapshots must be JSON objects")
    else:
        goals = manifest.goals
        if len({(item.owner_id, item.session_id, item.workspace_id, item.peer_id,
                 item.principal_id, item.canonical_owner_id, item.canonical_login)
                for item in goals}) != 1:
            parser.error("one live audit may cover one scoped principal, owner and Honcho session")
        owner, session_id = goals[0].owner_id, goals[0].session_id
        if args.honcho_workspace != goals[0].workspace_id:
            parser.error("Honcho workspace flag must match the reviewed goal workspace")
        if args.telegent_login != goals[0].canonical_login:
            parser.error("Telegent login flag must match the reviewed canonical login")
        if args.honcho_owner != owner or args.honcho_session != session_id:
            parser.error("live owner/session flags must exactly match manifest")
        required = (args.honcho_base_url, args.honcho_workspace, args.since, args.until,
                    args.telegent_login, args.start, args.end,
                    os.environ.get("OHMO_NUTRITION_AUDIT_HONCHO_TOKEN"))
        if not all(required):
            parser.error("live URLs, bounds, identities and the Honcho audit token are required")
        from openharness.config import load_settings

        server_configs = load_settings().mcp_servers
        telegent_config = server_configs.get(args.telegent_server)
        if telegent_config is None:
            parser.error("selected Telegent MCP server is absent from configured OpenHarness settings")
        honcho = asyncio.run(read_honcho_messages(
            base_url=args.honcho_base_url,
            api_key=os.environ["OHMO_NUTRITION_AUDIT_HONCHO_TOKEN"],
            workspace=args.honcho_workspace, session=session_id, owner_id=owner,
            since=datetime.fromisoformat(args.since), until=datetime.fromisoformat(args.until),
            peer_id=goals[0].peer_id,
        ))
        telegent = asyncio.run(read_telegent_wellness(
            server_config=telegent_config, server_name=args.telegent_server,
            owner_login=args.telegent_login,
            start=datetime.fromisoformat(args.start), end=datetime.fromisoformat(args.end),
        ))
    results = []
    for item in manifest.goals:
        if not dialogue_binding[item.case_id]["complete"]:
            results.append({"case_id": item.case_id, "expectation_origin": item.expectation_origin,
                            "expectation_source": item.expectation_source, "a1": "INCONCLUSIVE",
                            "stage": "DIALOGUE_BINDING_FAILED", "reason": dialogue_binding[item.case_id]["reason"],
                            "a2": "NOT_RUN"})
            continue
        wellness_for_goal = bind_wellness_snapshot(telegent, goal=item)
        results.append(grade_manifest(
            Manifest(schema_version=1, goals=[item]), honcho, wellness_for_goal,
            grace_seconds=args.grace_seconds,
            reviewed_turn_sources=dialogue_binding[item.case_id]["reviewed_turn_sources"],
            reviewed_turn_provenance=dialogue_binding[item.case_id]["reviewed_turn_provenance"],
        )[0])
    report = {
        "schema_version": 1,
        "privacy": "private",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "honcho_scope": {key: honcho.get(key) for key in
                         ("workspace_id", "session_id", "owner_id", "since", "until", "queried_at")},
        "telegent_scope": {key: telegent.get(key) for key in
                           ("user_id", "login", "start", "end", "queried_at")},
        "dialogue_binding": dialogue_binding,
        "cases": results,
    }
    atomic_write_text(args.output.expanduser(), json.dumps(report, indent=2, allow_nan=False) + "\n", mode=0o600)


if __name__ == "__main__":
    main()
