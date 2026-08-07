"""Ohmo gateway episode recorder built on the generic eval store."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ohmo.evals.adapter import get_eval_store
from ohmo.evals.nutrition_trace import validate_trace_finalization_annotations
from ohmo.evals.resources import ResourceSnapshotWrite, write_ohmo_resource_snapshot
from openharness.channels.bus.events import InboundMessage
from openharness.engine.stream_events import (
    AssistantTurnComplete,
    ErrorEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.evals import (
    TRACE_FINALIZATION,
    DecisionTraceRecorder,
    DecisionTraceValidationError,
    EvalEpisode,
    EvalEvent,
    EvalStore,
)
from openharness.evals.tool_labels import effective_tool_label, tool_call_binaries


@dataclass
class GatewayEvalRecorder:
    """Append Ohmo gateway episodes and events to the local eval store."""

    store: EvalStore
    episode_id: str
    user_goal: str = ""
    _finished: bool = False
    _structural_recorder: DecisionTraceRecorder = field(init=False, repr=False)
    _runtime_recorder: _GatewayDecisionTraceRecorderAdapter = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._structural_recorder = DecisionTraceRecorder(
            store=self.store,
            episode_id=self.episode_id,
            enabled=True,
        )
        self._runtime_recorder = _GatewayDecisionTraceRecorderAdapter(
            self._structural_recorder,
            user_goal=self.user_goal,
        )

    @property
    def decision_trace_recorder(self) -> _GatewayDecisionTraceRecorderAdapter:
        """Return the runtime recorder adapter for one gateway engine turn."""
        return self._runtime_recorder

    @classmethod
    def start(
        cls,
        *,
        workspace: str | Path,
        bundle: Any,
        message: InboundMessage,
        session_key: str,
        user_text: str,
        user_goal: str | None = None,
    ) -> GatewayEvalRecorder:
        normalized_user_goal = user_goal or ""
        recorder = cls(
            store=get_eval_store(workspace),
            episode_id=f"ohmo-gateway-{uuid4().hex}",
            user_goal=normalized_user_goal,
        )
        metadata = {
            "workspace": str(Path(workspace).expanduser().resolve()),
            "session_key": session_key,
            "cwd": _bundle_cwd(bundle),
            "model": _bundle_model(bundle),
            "media_count": len(message.media or []),
            "inbound": _inbound_metadata(message),
        }
        recorder.store.append_episode(
            EvalEpisode(
                episode_id=recorder.episode_id,
                source="gateway",
                app="ohmo",
                session_id=str(getattr(bundle, "session_id", "") or ""),
                user_goal=normalized_user_goal,
                user_text=user_text,
                tags=["gateway", str(message.channel)],
                privacy=str((message.metadata or {}).get("privacy") or "private"),
                status=str((message.metadata or {}).get("status") or "open"),
                metadata=_json_safe_mapping(metadata),
            )
        )
        recorder.record_inbound_message(
            message,
            user_text=user_text,
            user_goal=normalized_user_goal,
        )
        recorder.record_resource_snapshot(workspace=workspace, bundle=bundle)
        return recorder

    def record_inbound_message(
        self, message: InboundMessage, *, user_text: str, user_goal: str | None = None
    ) -> None:
        user_goal = user_goal or self.user_goal
        self.record_event(
            "inbound_message",
            payload={
                **_inbound_metadata(message),
                "user_text": user_text,
                "user_goal": user_goal,
            },
        )

    def record_resource_snapshot(
        self,
        *,
        workspace: str | Path | None,
        bundle: Any | None,
        phase: str = "world_before",
    ) -> ResourceSnapshotWrite:
        snapshot = write_ohmo_resource_snapshot(
            store=self.store,
            episode_id=self.episode_id,
            workspace=workspace,
            bundle=bundle,
            phase=phase,
        )
        self.record_event(
            "resource_snapshot",
            payload={
                "path": snapshot.relative_path,
                "phase": phase,
                "resource_count": snapshot.resource_count,
                "local_resource_count": snapshot.local_resource_count,
                "tool_count": snapshot.tool_count,
            },
        )
        return snapshot

    def record_tool_started(self, event: ToolExecutionStarted) -> None:
        payload: dict[str, Any] = {
            "input_summary": _summary(event.tool_input),
            "input": event.tool_input,
        }
        # Lift the real capability out of a shell tool's command so the
        # trajectory is not collapsed to "bash": record which binaries were
        # invoked and the effective label (e.g. "bash:maps-cli reviews").
        binaries = tool_call_binaries(event.tool_name, event.tool_input)
        if binaries:
            payload["binaries"] = binaries
            payload["capability"] = effective_tool_label(event.tool_name, event.tool_input)
        self.record_event(
            "tool_started",
            payload=payload,
            tool_name=event.tool_name,
            tool_call_id=event.tool_call_id,
        )

    def record_tool_completed(self, event: ToolExecutionCompleted) -> None:
        self.record_event(
            "tool_completed",
            payload={
                "output_summary": _summary(event.output),
                "output": event.output,
            },
            tool_name=event.tool_name,
            tool_call_id=event.tool_call_id,
            is_error=event.is_error,
        )

    def record_model_call(self, event: AssistantTurnComplete, *, model: str) -> None:
        self.record_event(
            "model_call",
            payload={
                "model": model,
                "input_tokens": event.usage.input_tokens,
                "output_tokens": event.usage.output_tokens,
                "cached_input_tokens": event.usage.cached_input_tokens,
                "cache_write_input_tokens": event.usage.cache_write_input_tokens,
            },
        )

    def record_engine_error(self, event: ErrorEvent) -> None:
        self.record_event(
            "engine_error",
            payload={"message": event.message, "recoverable": event.recoverable},
            is_error=True,
        )

    def record_gateway_final(self, *, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self.record_event(
            "gateway_final",
            payload={"text": text, "metadata": metadata or {}},
        )

    def record_gateway_error(
        self, *, text: str, metadata: Mapping[str, Any] | None = None
    ) -> None:
        self.record_event(
            "gateway_error",
            payload={"text": text, "metadata": metadata or {}},
            is_error=True,
        )

    def record_exception(self, exc: Exception) -> None:
        self.record_event(
            "exception",
            payload={"type": type(exc).__name__, "message": str(exc)},
            is_error=True,
        )

    def finish(self, *, status: str) -> None:
        if self._finished:
            return
        self.record_event(
            "episode_finished",
            payload={"status": status},
            is_error=status not in {"completed", "ok"},
        )
        self._finished = True

    def record_event(
        self,
        kind: str,
        *,
        payload: Mapping[str, Any] | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> None:
        self._structural_recorder.record_legacy_structural(
            kind,
            _json_safe_mapping(payload or {}),
            tool_name=tool_name or None,
            tool_call_id=tool_call_id or None,
            is_error=is_error,
        )

    @property
    def decision_trace_status(self) -> str:
        """Return the latest trace-finalization status for this turn."""
        return self._runtime_recorder.decision_trace_status

    @property
    def nutrition_annotation_status(self) -> str:
        """Return the latest nutrition annotation status for this turn."""
        return self._runtime_recorder.nutrition_annotation_status

    @property
    def decision_trace_envelope(self) -> Mapping[str, Any] | None:
        """Return a JSON-safe snapshot of the latest finalization event."""
        envelope = self._runtime_recorder.decision_trace_envelope
        if envelope is None:
            return None
        return envelope

    @property
    def validated_nutrition_envelope(self) -> Mapping[str, Any] | None:
        """Return the recorder-owned, schema-validated nutrition annotation."""
        envelope = self.decision_trace_envelope
        if envelope is None:
            return None
        annotations = envelope.get("annotations")
        if not isinstance(annotations, Mapping):
            return None
        nutrition = annotations.get("nutrition")
        return nutrition if isinstance(nutrition, Mapping) else None

    def set_authoritative_nutrition_meal_at(self, meal_at: datetime | None) -> None:
        """Stamp the trusted Dropbox capture time before trace validation."""
        self._runtime_recorder.set_authoritative_nutrition_meal_at(meal_at)


_RUNTIME_STRUCTURAL_SKIP_KINDS = frozenset(
    {
        "model_call",
        "tool_started",
    }
)

_NUTRITION_REQUIREMENT_SIGNAL = "ohmo_nutrition_request"
_NUTRITION_REQUIREMENT_MARKERS = (
    re.compile(r"\bcalorie(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bkcal\b", re.IGNORECASE),
    re.compile(r"\bnutrition(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bmacronutrient(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bmacro(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bprotein(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bcarb(?:ohydrate|o?hydrates)?(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bfat(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bкалори[йя]\b", re.IGNORECASE),
    re.compile(r"\bкалорийность\b", re.IGNORECASE),
    re.compile(r"\bккал\b", re.IGNORECASE),
    re.compile(r"\bбжу\b", re.IGNORECASE),
    re.compile(r"\bбелк[а-я]*\b", re.IGNORECASE),
    re.compile(r"\bжир[а-я]*\b", re.IGNORECASE),
    re.compile(r"\bуглевод[а-я]*\b", re.IGNORECASE),
)

_DECISION_TRACE_STATUS_DISABLED = "disabled"
_DECISION_TRACE_STATUS_INVALID = "invalid"
_DECISION_TRACE_STATUS_MISSING = "missing"
_DECISION_TRACE_STATUS_RECORDED = "recorded"

_NUTRITION_ANNOTATION_STATUS_DISABLED = "disabled"
_NUTRITION_ANNOTATION_STATUS_INVALID = "invalid"
_NUTRITION_ANNOTATION_STATUS_MISSING = "missing"
_NUTRITION_ANNOTATION_STATUS_NOT_APPLICABLE = "not_applicable"
_NUTRITION_ANNOTATION_STATUS_RECORDED = "recorded"


def _contains_nutrition_marker(*texts: str | None) -> bool:
    haystack = " ".join(text or "" for text in texts).lower()
    return any(marker.search(haystack) for marker in _NUTRITION_REQUIREMENT_MARKERS)


class _GatewayDecisionTraceRecorderAdapter:
    """Runtime recorder bridge for gateway-owned eval episodes."""

    def __init__(
        self,
        recorder: DecisionTraceRecorder,
        *,
        user_goal: str = "",
    ) -> None:
        self._recorder = recorder
        self._user_goal = user_goal
        self._latest_finalization: EvalEvent | None = None
        self._saw_invalid_finalization = False
        self._nutrition_applicable = False
        self._authoritative_nutrition_meal_at: datetime | None = None

    def set_authoritative_nutrition_meal_at(self, meal_at: datetime | None) -> None:
        if meal_at is not None and (meal_at.tzinfo is None or meal_at.utcoffset() is None):
            raise ValueError("authoritative nutrition meal_at must be timezone-aware")
        self._authoritative_nutrition_meal_at = meal_at

    def record(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> EvalEvent | None:
        if kind == TRACE_FINALIZATION:
            payload = self._stamp_authoritative_nutrition_meal_at(payload)
            try:
                payload = validate_trace_finalization_annotations(payload)
                payload = self._stamp_authoritative_nutrition_meal_at(payload)
                event = self._recorder.record(
                    kind,
                    payload,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    is_error=is_error,
                )
            except DecisionTraceValidationError:
                self._saw_invalid_finalization = True
                raise
            if event is not None:
                self._latest_finalization = event
            return event
        return self._recorder.record(
            kind,
            payload,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )

    def _stamp_authoritative_nutrition_meal_at(
        self, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        meal_at = self._authoritative_nutrition_meal_at
        if meal_at is None:
            return payload
        annotations = payload.get("annotations")
        nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
        if not isinstance(nutrition, Mapping) or nutrition.get("schema_version") != 2:
            return payload
        stamped = copy.deepcopy(dict(payload))
        stamped_annotations = dict(stamped.get("annotations") or {})
        stamped_nutrition = dict(stamped_annotations.get("nutrition") or {})
        stamped_nutrition["meal_at"] = meal_at.isoformat()
        for field_name in ("assumptions", "warnings"):
            values = stamped_nutrition.get(field_name)
            if isinstance(values, list):
                stamped_nutrition[field_name] = [
                    value
                    for value in values
                    if not (isinstance(value, str) and "exif" in value.casefold())
                ]
        stamped_annotations["nutrition"] = stamped_nutrition
        stamped["annotations"] = stamped_annotations
        return stamped

    def trace_requirement_signals(self, final_text: str) -> tuple[str, ...]:
        if _contains_nutrition_marker(final_text, self._user_goal):
            self._nutrition_applicable = True
            return (_NUTRITION_REQUIREMENT_SIGNAL,)
        return ()

    @property
    def decision_trace_status(self) -> str:
        if not self._recorder.enabled:
            return _DECISION_TRACE_STATUS_DISABLED
        if self._latest_finalization is not None:
            return _DECISION_TRACE_STATUS_RECORDED
        if self._saw_invalid_finalization:
            return _DECISION_TRACE_STATUS_INVALID
        return _DECISION_TRACE_STATUS_MISSING

    @property
    def nutrition_annotation_status(self) -> str:
        if not self._recorder.enabled:
            return _NUTRITION_ANNOTATION_STATUS_DISABLED
        finalization = self._latest_finalization
        if finalization is None:
            if self._saw_invalid_finalization:
                return _NUTRITION_ANNOTATION_STATUS_INVALID
            if self._nutrition_applicable:
                return _NUTRITION_ANNOTATION_STATUS_MISSING
            return _NUTRITION_ANNOTATION_STATUS_NOT_APPLICABLE

        annotations = finalization.payload.get("annotations")
        if isinstance(annotations, Mapping) and "nutrition" in annotations:
            return _NUTRITION_ANNOTATION_STATUS_RECORDED
        return _NUTRITION_ANNOTATION_STATUS_MISSING

    @property
    def decision_trace_envelope(self) -> Mapping[str, Any] | None:
        if self._latest_finalization is None:
            return None
        envelope = {
            "kind": self._latest_finalization.kind,
            "episode_id": self._latest_finalization.episode_id,
            "timestamp": self._latest_finalization.timestamp,
            **self._latest_finalization.payload,
        }
        return _freeze_json_mapping(_json_safe(envelope))

    def record_structural(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> EvalEvent | None:
        if kind in _RUNTIME_STRUCTURAL_SKIP_KINDS:
            return None
        return self._recorder.record_structural(
            kind,
            payload,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )


class _FrozenMapping(dict[str, Any]):
    """Simple immutable mapping used for returning read-only tracing envelopes."""

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError("Frozen mapping is read-only")

    def __delitem__(self, key: str) -> None:
        raise TypeError("Frozen mapping is read-only")


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _FrozenMapping({str(key): _freeze_json_value(item) for key, item in value.items()})


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value)
    if isinstance(value, list):
        return [_freeze_json_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _inbound_metadata(message: InboundMessage) -> dict[str, Any]:
    return {
        "channel": message.channel,
        "chat_id": str(message.chat_id),
        "sender_id": str(message.sender_id),
        "timestamp": message.timestamp,
        "media_count": len(message.media or []),
        "metadata": message.metadata or {},
    }


def _bundle_cwd(bundle: Any) -> str:
    cwd = getattr(bundle, "cwd", "")
    return str(cwd) if cwd else ""


def _bundle_model(bundle: Any) -> str:
    settings = getattr(bundle, "current_settings", None)
    if callable(settings):
        current = settings()
        model = getattr(current, "model", None)
        if model:
            return str(model)
    engine = getattr(bundle, "engine", None)
    model = getattr(engine, "model", None)
    return str(model or "")


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    safe = _json_safe(value)
    if not isinstance(safe, dict):
        return {"value": safe}
    return safe


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _summary(value: Any, *, limit: int = 160) -> str:
    safe = _json_safe(value)
    if isinstance(safe, str):
        text = safe
    else:
        text = repr(safe)
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."
