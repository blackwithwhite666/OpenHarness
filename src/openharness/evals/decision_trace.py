"""Validation and recording helpers for decision-trace eval events."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any

from openharness.evals.models import EvalEvent
from openharness.evals.store import EvalStore

DECISION_TRACE_ENV_VAR = "OPENHARNESS_DECISION_TRACE"
DECISION_TRACE_SCHEMA_VERSION = 1
DECISION_TRACE_MAX_PAYLOAD_BYTES = 16 * 1024

TRACE_INTENT = "trace_intent"
TRACE_DECISION = "trace_decision"
TRACE_OBSERVATION = "trace_observation"
TRACE_UNCERTAINTY = "trace_uncertainty"
TRACE_STOP_CONDITION = "trace_stop_condition"
TRACE_FINALIZATION = "trace_finalization"
TRACE_MISSING_REQUIRED = "trace_missing_required"

STRUCTURAL_TURN_STARTED = "turn_started"
STRUCTURAL_TURN_CONTINUED = "turn_continued"
STRUCTURAL_MODEL_CALL = "model_call"
STRUCTURAL_ASSISTANT_FINAL = "assistant_final"
STRUCTURAL_TOOL_STARTED = "tool_started"
STRUCTURAL_TOOL_PERMISSION = "tool_permission"
STRUCTURAL_TOOL_COMPLETED = "tool_completed"
STRUCTURAL_ENGINE_ERROR = "engine_error"
STRUCTURAL_INBOUND_MESSAGE = "inbound_message"
STRUCTURAL_RESOURCE_SNAPSHOT = "resource_snapshot"
STRUCTURAL_GATEWAY_FINAL = "gateway_final"
STRUCTURAL_GATEWAY_ERROR = "gateway_error"
STRUCTURAL_EXCEPTION = "exception"
STRUCTURAL_EPISODE_FINISHED = "episode_finished"

DECISION_TRACE_MODEL_EVENT_KINDS = frozenset(
    {
        TRACE_INTENT,
        TRACE_DECISION,
        TRACE_OBSERVATION,
        TRACE_UNCERTAINTY,
        TRACE_STOP_CONDITION,
        TRACE_FINALIZATION,
    }
)
DECISION_TRACE_DIAGNOSTIC_EVENT_KINDS = frozenset({TRACE_MISSING_REQUIRED})
DECISION_TRACE_EVENT_KINDS = (
    DECISION_TRACE_MODEL_EVENT_KINDS | DECISION_TRACE_DIAGNOSTIC_EVENT_KINDS
)
DECISION_TRACE_STRUCTURAL_EVENT_KINDS = frozenset(
    {
        STRUCTURAL_TURN_STARTED,
        STRUCTURAL_TURN_CONTINUED,
        STRUCTURAL_MODEL_CALL,
        STRUCTURAL_ASSISTANT_FINAL,
        STRUCTURAL_TOOL_STARTED,
        STRUCTURAL_TOOL_PERMISSION,
        STRUCTURAL_TOOL_COMPLETED,
        STRUCTURAL_ENGINE_ERROR,
        STRUCTURAL_INBOUND_MESSAGE,
        STRUCTURAL_RESOURCE_SNAPSHOT,
        STRUCTURAL_GATEWAY_FINAL,
        STRUCTURAL_GATEWAY_ERROR,
        STRUCTURAL_EXCEPTION,
        STRUCTURAL_EPISODE_FINISHED,
    }
)
DECISION_TRACE_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})
DECISION_TRACE_SENSITIVITY_LABELS = frozenset(
    {
        "public",
        "personal",
        "private",
        "secret",
    }
)
DECISION_TRACE_RETENTION_LABELS = frozenset(
    {
        "ephemeral",
        "session",
        "durable",
    }
)

_DEFAULT_SENSITIVITY = "private"
_DEFAULT_RETENTION = "durable"
_SECRET_ASSIGNMENT_RE = re.compile(
    r"""
    ["']?
    [A-Za-z0-9_.-]*(?:api[_-]?key|token|secret|password)[A-Za-z0-9_.-]*
    ["']?
    \s*[:=]\s*
    ["']?
    [A-Za-z0-9_./+=-]{16,}
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_OPENAI_STYLE_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_PEM_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")


class DecisionTraceValidationError(ValueError):
    """Raised when a decision-trace payload is invalid or unsafe to record."""


def decision_trace_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether decision-trace recording is enabled by environment."""
    environment = os.environ if env is None else env
    value = environment.get(DECISION_TRACE_ENV_VAR)
    if value is None:
        return True
    return value.strip().lower() not in DECISION_TRACE_DISABLED_VALUES


def validate_decision_trace_payload(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and default a decision-trace payload for storage."""
    if kind not in DECISION_TRACE_EVENT_KINDS:
        raise DecisionTraceValidationError(f"unknown decision trace event kind: {kind}")
    if not isinstance(payload, Mapping):
        raise DecisionTraceValidationError("decision trace payload must be a mapping")

    validated = dict(payload)
    schema_version = validated.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != DECISION_TRACE_SCHEMA_VERSION
    ):
        raise DecisionTraceValidationError(
            f"decision trace payload schema_version must be {DECISION_TRACE_SCHEMA_VERSION}"
        )

    _validate_required_id(validated, "trace_event_id")
    _validate_optional_id(validated, "parent_event_id")
    _validate_optional_id(validated, "related_tool_call_id")
    _default_and_validate_label(
        validated,
        "sensitivity",
        default=_DEFAULT_SENSITIVITY,
        allowed=DECISION_TRACE_SENSITIVITY_LABELS,
    )
    _default_and_validate_label(
        validated,
        "retention",
        default=_DEFAULT_RETENTION,
        allowed=DECISION_TRACE_RETENTION_LABELS,
    )

    json_bytes = _json_payload_bytes(validated)
    if len(json_bytes) > DECISION_TRACE_MAX_PAYLOAD_BYTES:
        raise DecisionTraceValidationError(
            "decision trace payload exceeds "
            f"{DECISION_TRACE_MAX_PAYLOAD_BYTES} byte limit"
        )
    _reject_obvious_secrets(json_bytes.decode("utf-8"))
    return validated


def validate_decision_trace_structural_payload(
    kind: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a compact engine-captured structural payload for storage."""
    if kind not in DECISION_TRACE_STRUCTURAL_EVENT_KINDS:
        raise DecisionTraceValidationError(
            f"unknown decision trace structural event kind: {kind}"
        )
    if not isinstance(payload, Mapping):
        raise DecisionTraceValidationError(
            "decision trace structural payload must be a mapping"
        )

    validated = dict(payload)
    json_bytes = _json_payload_bytes(validated)
    if len(json_bytes) > DECISION_TRACE_MAX_PAYLOAD_BYTES:
        raise DecisionTraceValidationError(
            "decision trace structural payload exceeds "
            f"{DECISION_TRACE_MAX_PAYLOAD_BYTES} byte limit"
        )
    _reject_obvious_secrets(json_bytes.decode("utf-8"))
    return validated


def validate_decision_trace_legacy_structural_payload(
    kind: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a legacy gateway structural payload for storage."""
    if kind not in DECISION_TRACE_STRUCTURAL_EVENT_KINDS:
        raise DecisionTraceValidationError(
            f"unknown decision trace structural event kind: {kind}"
        )
    if not isinstance(payload, Mapping):
        raise DecisionTraceValidationError(
            "decision trace structural payload must be a mapping"
        )

    validated = dict(payload)
    _json_payload_bytes(validated)
    return validated


class DecisionTraceRecorder:
    """Append validated decision-trace events to an existing eval episode."""

    def __init__(
        self,
        store: EvalStore,
        episode_id: str,
        enabled: bool | None = None,
    ) -> None:
        self.store = store
        self.episode_id = episode_id
        self.enabled = decision_trace_enabled() if enabled is None else enabled

    def record(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> EvalEvent | None:
        """Validate and append a decision-trace event, or no-op when disabled."""
        if not self.enabled:
            return None

        validated_payload = validate_decision_trace_payload(kind, payload)
        effective_tool_call_id = tool_call_id
        if effective_tool_call_id is None:
            effective_tool_call_id = validated_payload.get("related_tool_call_id")

        event = EvalEvent(
            episode_id=self.episode_id,
            kind=kind,
            payload=validated_payload,
            tool_name=tool_name,
            tool_call_id=effective_tool_call_id,
            is_error=is_error,
        )
        self.store.append_event(event)
        return event

    def record_structural(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> EvalEvent | None:
        """Validate and append a structural engine event, or no-op when disabled."""
        if not self.enabled:
            return None

        event = EvalEvent(
            episode_id=self.episode_id,
            kind=kind,
            payload=validate_decision_trace_structural_payload(kind, payload),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )
        self.store.append_event(event)
        return event

    def record_legacy_structural(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> EvalEvent | None:
        """Append a legacy gateway structural event, or no-op when disabled."""
        if not self.enabled:
            return None

        event = EvalEvent(
            episode_id=self.episode_id,
            kind=kind,
            payload=validate_decision_trace_legacy_structural_payload(kind, payload),
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )
        self.store.append_event(event)
        return event


def _validate_required_id(payload: Mapping[str, Any], field_name: str) -> None:
    value = payload.get(field_name)
    if not _is_non_empty_string(value):
        raise DecisionTraceValidationError(
            f"decision trace payload {field_name} must be a non-empty string"
        )


def _validate_optional_id(payload: Mapping[str, Any], field_name: str) -> None:
    if field_name not in payload:
        return
    value = payload[field_name]
    if not _is_non_empty_string(value):
        raise DecisionTraceValidationError(
            f"decision trace payload {field_name} must be a non-empty string"
        )


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _default_and_validate_label(
    payload: dict[str, Any],
    field_name: str,
    *,
    default: str,
    allowed: frozenset[str],
) -> None:
    if field_name not in payload:
        payload[field_name] = default
        return

    value = payload[field_name]
    if not isinstance(value, str) or value not in allowed:
        allowed_labels = ", ".join(sorted(allowed))
        raise DecisionTraceValidationError(
            f"decision trace payload {field_name} must be one of: {allowed_labels}"
        )


def _json_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DecisionTraceValidationError(
            "decision trace payload must be JSON-serializable with finite numbers"
        ) from exc


def _reject_obvious_secrets(payload_text: str) -> None:
    if (
        _SECRET_ASSIGNMENT_RE.search(payload_text)
        or _OPENAI_STYLE_KEY_RE.search(payload_text)
        or _PEM_PRIVATE_KEY_RE.search(payload_text)
    ):
        raise DecisionTraceValidationError(
            "decision trace payload appears to contain an obvious secret"
        )
