"""Core tool-aware query loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol
from uuid import uuid4

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiTextDeltaEvent,
    SupportsStreamingMessages,
)
from openharness.api.provider import is_model_multimodal
from openharness.api.usage import UsageSnapshot
from openharness.config.paths import get_data_dir
from openharness.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
)
from openharness.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    CompactProgressEvent,
    ErrorEvent,
    StatusEvent,
    StreamEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.hooks import HookEvent, HookExecutor
from openharness.permissions.checker import PermissionChecker
from openharness.services.tool_outputs import tool_output_inline_chars, tool_output_preview_chars
from openharness.tools.base import ToolExecutionContext
from openharness.tools.base import ToolRegistry

AUTO_COMPACT_STATUS_MESSAGE = "Auto-compacting conversation memory to keep things fast and focused."
REACTIVE_COMPACT_STATUS_MESSAGE = "Prompt too long; compacting conversation memory and retrying."
MAX_SAFE_COMPLETION_TOKENS = 128_000

log = logging.getLogger(__name__)


PermissionPrompt = Callable[[str, str], Awaitable[bool]]
AskUserPrompt = Callable[[str], Awaitable[str]]

DECISION_TRACE_RECORDER_METADATA_KEY = "decision_trace_recorder"

_TRACE_KIND_TURN_STARTED = "turn_started"
_TRACE_KIND_TURN_CONTINUED = "turn_continued"
_TRACE_KIND_MODEL_CALL = "model_call"
_TRACE_KIND_ASSISTANT_FINAL = "assistant_final"
_TRACE_KIND_TOOL_STARTED = "tool_started"
_TRACE_KIND_TOOL_PERMISSION = "tool_permission"
_TRACE_KIND_TOOL_COMPLETED = "tool_completed"
_TRACE_KIND_ENGINE_ERROR = "engine_error"
_TRACE_TOOL_NAME = "trace"
_TRACE_MODEL_KIND_FINALIZATION = "trace_finalization"
_TRACE_MISSING_REQUIRED = "trace_missing_required"
_TRACE_REQUIRED_MIN_TEXT_CHARS = 80
_STRUCTURAL_TEXT_SUMMARY_CHARS = 240
_STRUCTURAL_VALUE_SUMMARY_CHARS = 320
_TRACE_TRIVIAL_FINAL_TEXTS = frozenset(
    {
        "ok",
        "okay",
        "done",
        "thanks",
        "thank you",
        "sure",
        "yes",
        "no",
    }
)
_TRACE_UNCERTAINTY_MARKERS = (
    "i'm not sure",
    "i am not sure",
    "unclear",
    "uncertain",
    "maybe",
    "might",
    "probably",
    "appears",
    "seems",
)
_TRACE_EVIDENCE_FINALIZATION_MARKERS = (
    "because",
    "based on",
    "evidence",
    "verified",
    "confirmed",
    "therefore",
    "so the",
    "final",
    "conclusion",
    "result",
)

MAX_TRACKED_READ_FILES = 6
MAX_TRACKED_SKILLS = 8
MAX_TRACKED_ASYNC_AGENT_EVENTS = 8
MAX_TRACKED_ASYNC_AGENT_TASKS = 12
MAX_TRACKED_WORK_LOG = 10
MAX_TRACKED_USER_GOALS = 5
MAX_TRACKED_ACTIVE_ARTIFACTS = 8
MAX_TRACKED_VERIFIED_WORK = 10


class DecisionTraceRecorderLike(Protocol):
    """Recorder surface used by the generic engine without importing evals."""

    def record(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> object | None:
        """Append a model-authored or diagnostic decision-trace event."""

    def record_structural(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> object | None:
        """Append a compact structural decision-trace event."""


@dataclass(frozen=True)
class _TraceRequirement:
    required: bool
    reason: str
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class _TraceObservation:
    tool_call_id: str
    label: str
    summary: str
    is_error: bool


# Cap how many observations are offered back in the repair prompt so the nudge
# stays compact (the repair model call is bounded to 1024 tokens).
_TRACE_REPAIR_MAX_OBSERVATIONS = 12
_TRACE_OBSERVATION_SUMMARY_CHARS = 160


def _trace_tool_kind(tool_input: Any) -> str | None:
    """Extract the model-authored trace kind from a `trace` tool call input."""
    if isinstance(tool_input, Mapping):
        kind = tool_input.get("kind")
        if isinstance(kind, str):
            return kind
    return None


@dataclass
class _DecisionTraceRunState:
    successful_model_trace: bool = False
    # A trace_required turn is only satisfied by a trace_finalization, not by any
    # model-authored trace event: a proactive trace_observation alone must still
    # leave the finalization requirement (and its repair) in force.
    successful_finalization: bool = False
    tool_call_count: int = 0
    tool_result_count: int = 0
    failed_tool_result_count: int = 0
    # Per-turn observation index: (tool_call_id, capability label, summary, is_error).
    # Offered back to the model in the repair prompt so finalization claims can be
    # linked to concrete evidence ids without re-embedding raw tool output (D5).
    observations: list["_TraceObservation"] = field(default_factory=list)


def _is_prompt_too_long_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        needle in text
        for needle in (
            "prompt too long",
            "context_length_exceeded",
            "context length",
            "maximum context",
            "context window",
            "input tokens exceed",
            "messages resulted in",
            "reduce the length of the messages",
            "configured limit",
            "too many tokens",
            "too large for the model",
            "maximum context length",
            "exceed_context",
            "exceeds the available context size",
            "available context size",
        )
    )


def _bounded_completion_tokens(max_tokens: int, context_window_tokens: int | None = None) -> int:
    """Return a conservative per-request output token cap.

    Some OpenAI-compatible providers reject very large ``max_tokens`` before
    the request reaches model-side context management.  Keep oversized user
    config from making every turn fail while preserving normal defaults.
    """
    limit = MAX_SAFE_COMPLETION_TOKENS
    if context_window_tokens is not None and context_window_tokens > 0:
        limit = min(limit, int(context_window_tokens))
    return max(1, min(int(max_tokens), limit))


def _extract_completion_token_limit(exc: Exception) -> int | None:
    """Parse provider errors such as "supports at most 128000 completion tokens"."""
    text = str(exc).lower().replace(",", "")
    patterns = (
        r"supports at most\s+(\d+)\s+completion tokens",
        r"at most\s+(\d+)\s+completion tokens",
        r"max(?:imum)?(?:_completion)?[_\s-]tokens.*?(?:<=|less than or equal to|at most)\s+(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return max(1, int(match.group(1)))
            except ValueError:
                return None
    return None


def _is_completion_token_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        ("max_tokens" in text or "max_completion_tokens" in text)
        and ("too large" in text or "at most" in text or "completion tokens" in text)
    )


class MaxTurnsExceeded(RuntimeError):
    """Raised when the agent exceeds the configured max_turns for one user prompt."""

    def __init__(self, max_turns: int) -> None:
        super().__init__(f"Exceeded maximum turn limit ({max_turns})")
        self.max_turns = max_turns


@dataclass
class QueryContext:
    """Context shared across a query run."""

    api_client: SupportsStreamingMessages
    tool_registry: ToolRegistry
    permission_checker: PermissionChecker
    cwd: Path
    model: str
    system_prompt: str
    max_tokens: int
    effort: str | None = None
    context_window_tokens: int | None = None
    auto_compact_threshold_tokens: int | None = None
    permission_prompt: PermissionPrompt | None = None
    ask_user_prompt: AskUserPrompt | None = None
    max_turns: int | None = 200
    hook_executor: HookExecutor | None = None
    tool_metadata: dict[str, object] | None = None
    decision_trace_recorder: DecisionTraceRecorderLike | None = None


def _record_decision_trace_structural(
    recorder: DecisionTraceRecorderLike | None,
    kind: str,
    payload: Mapping[str, Any],
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
    is_error: bool = False,
) -> None:
    if recorder is None:
        return
    try:
        recorder.record_structural(
            kind,
            payload,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )
    except Exception:
        log.exception("decision trace recorder failed for structural event %s", kind)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _compact_text_summary(text: str, *, limit: int = _STRUCTURAL_TEXT_SUMMARY_CHARS) -> str:
    normalized = " ".join(text.split())
    return normalized[:limit]


def _structural_text_fields(
    prefix: str,
    text: str,
    *,
    limit: int = _STRUCTURAL_TEXT_SUMMARY_CHARS,
) -> dict[str, object]:
    return {
        f"{prefix}_summary": _compact_text_summary(text, limit=limit),
        f"{prefix}_length": len(text),
        f"{prefix}_sha256": _sha256_text(text),
    }


def _structural_value_fields(
    prefix: str,
    value: Any,
    *,
    limit: int = _STRUCTURAL_VALUE_SUMMARY_CHARS,
) -> dict[str, object]:
    encoding = "json"
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        encoding = "repr"
        text = repr(value)
    return {
        f"{prefix}_summary": _compact_text_summary(text, limit=limit),
        f"{prefix}_length": len(text),
        f"{prefix}_sha256": _sha256_text(text),
        f"{prefix}_encoding": encoding,
    }


def _message_block_counts(message: ConversationMessage) -> dict[str, int]:
    counts = {"text": 0, "image": 0, "tool_use": 0, "tool_result": 0}
    for block in message.content:
        if isinstance(block, TextBlock):
            counts["text"] += 1
        elif isinstance(block, ImageBlock):
            counts["image"] += 1
        elif isinstance(block, ToolResultBlock):
            counts["tool_result"] += 1
        else:
            counts["tool_use"] += 1
    return counts


def _turn_started_trace_payload(
    user_message: ConversationMessage,
    *,
    model: str,
    cwd: Path,
) -> dict[str, object]:
    return {
        "model": model,
        "cwd": str(cwd),
        "role": user_message.role,
        "content_blocks": len(user_message.content),
        "block_counts": _message_block_counts(user_message),
        **_structural_text_fields("user_text", user_message.text),
    }


def _turn_continued_trace_payload(
    messages: list[ConversationMessage],
    *,
    model: str,
    cwd: Path,
    max_turns: int | None,
) -> dict[str, object]:
    pending_tool_result_ids: list[str] = []
    if messages:
        last_message = messages[-1]
        pending_tool_result_ids = [
            block.tool_use_id
            for block in last_message.content
            if isinstance(block, ToolResultBlock)
        ][-8:]
    return {
        "model": model,
        "cwd": str(cwd),
        "message_count": len(messages),
        "max_turns": max_turns,
        "pending_tool_result_count": len(pending_tool_result_ids),
        "pending_tool_result_ids": pending_tool_result_ids,
    }


def _model_call_trace_payload(
    message: ConversationMessage,
    *,
    model: str,
    usage: UsageSnapshot,
) -> dict[str, object]:
    return {
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "assistant_role": message.role,
        "content_blocks": len(message.content),
        "block_counts": _message_block_counts(message),
        "tool_use_count": len(message.tool_uses),
        "tool_names": [tool_use.name for tool_use in message.tool_uses][:12],
        **_structural_text_fields("assistant_text", message.text),
    }


def _assistant_final_trace_payload(
    message: ConversationMessage,
    *,
    model: str,
    trace_requirement: _TraceRequirement | None = None,
    model_trace_recorded: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "stop_reason": "tool_uses_empty",
        "content_blocks": len(message.content),
        "block_counts": _message_block_counts(message),
        **_structural_text_fields("assistant_text", message.text),
    }
    if trace_requirement is not None:
        payload.update(
            {
                "trace_required": trace_requirement.required,
                "trace_required_reason": trace_requirement.reason,
                "trace_required_signals": list(trace_requirement.signals),
                "model_trace_recorded": model_trace_recorded,
            }
        )
    return payload


def _tool_started_trace_payload(tool_input: dict[str, object]) -> dict[str, object]:
    return {
        "input_keys": sorted(str(key) for key in tool_input.keys())[:40],
        **_structural_value_fields("input", tool_input),
    }


def _tool_completed_trace_payload(
    output: str,
    *,
    is_error: bool,
    duration_ms: float,
) -> dict[str, object]:
    return {
        "is_error": is_error,
        "duration_ms": duration_ms,
        **_structural_text_fields(
            "output",
            output,
            limit=_STRUCTURAL_VALUE_SUMMARY_CHARS,
        ),
    }


def _tool_permission_trace_payload(
    *,
    allowed: bool,
    requires_confirmation: bool,
    reason: str,
    read_only: bool,
    file_path: str | None,
    command: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "allowed": allowed,
        "requires_confirmation": requires_confirmation,
        "reason": _compact_text_summary(reason),
        "reason_length": len(reason),
        "read_only": read_only,
    }
    if file_path is not None:
        payload.update(_structural_text_fields("path", file_path))
    if command is not None:
        payload.update(_structural_text_fields("command", command))
    return payload


def _engine_error_trace_payload(
    message: str,
    *,
    recoverable: bool,
    error_type: str,
) -> dict[str, object]:
    return {
        "recoverable": recoverable,
        "error_type": error_type,
        **_structural_text_fields("message", message),
    }


def _tool_execution_metadata(
    context: QueryContext,
    base: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        **base,
        **(context.tool_metadata or {}),
    }
    if context.decision_trace_recorder is not None:
        metadata[DECISION_TRACE_RECORDER_METADATA_KEY] = context.decision_trace_recorder
    return metadata


def _messages_include_tool_results(messages: list[ConversationMessage]) -> bool:
    return any(
        isinstance(block, ToolResultBlock)
        for message in messages
        for block in message.content
    )


def _normalized_trace_final_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _is_trivial_trace_final_text(text: str) -> bool:
    normalized = _normalized_trace_final_text(text).strip(" .!?:;").lower()
    if not normalized:
        return True
    return normalized in _TRACE_TRIVIAL_FINAL_TEXTS


def _contains_any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _classify_trace_requirement(
    final_message: ConversationMessage,
    *,
    run_state: _DecisionTraceRunState,
    prior_tool_results_seen: bool,
) -> _TraceRequirement:
    text = _normalized_trace_final_text(final_message.text)
    if _is_trivial_trace_final_text(text):
        return _TraceRequirement(False, "trivial_final_answer")

    signals: list[str] = []
    if len(text) >= _TRACE_REQUIRED_MIN_TEXT_CHARS:
        signals.append("substantive_final_answer")
    if run_state.tool_call_count > 0 or run_state.tool_result_count > 0:
        signals.append("current_run_tool_use")
    if run_state.failed_tool_result_count > 0:
        signals.append("failed_tool_result")
    if _contains_any_marker(text, _TRACE_UNCERTAINTY_MARKERS):
        signals.append("uncertainty_language")
    if len(text) >= 24 and _contains_any_marker(text, _TRACE_EVIDENCE_FINALIZATION_MARKERS):
        signals.append("evidence_or_finalization_language")
    if prior_tool_results_seen and len(text) >= 24:
        signals.append("prior_tool_results")

    if not signals:
        return _TraceRequirement(False, "no_trace_required_signals")
    return _TraceRequirement(True, signals[0], tuple(signals))


def _record_trace_observation(
    run_state: _DecisionTraceRunState,
    tool_call: Any,
    result: Any,
) -> None:
    """Index a completed non-trace tool call as a linkable evidence observation."""
    name = getattr(tool_call, "name", "") or ""
    if name == _TRACE_TOOL_NAME:
        return
    call_id = getattr(tool_call, "id", None)
    if not call_id:
        return
    # Local import: openharness.evals.__init__ imports back into this module
    # (executor -> engine.query), so a top-level import would be circular.
    from openharness.evals.tool_labels import effective_tool_label

    label = effective_tool_label(name, getattr(tool_call, "input", None))
    content = getattr(result, "content", "")
    summary = _compact_text_summary(
        content if isinstance(content, str) else str(content),
        limit=_TRACE_OBSERVATION_SUMMARY_CHARS,
    )
    run_state.observations.append(
        _TraceObservation(
            tool_call_id=str(call_id),
            label=label,
            summary=summary,
            is_error=bool(getattr(result, "is_error", False)),
        )
    )


def _format_repair_observations(observations: list[_TraceObservation]) -> str:
    """Render the turn's observations as an id-keyed evidence list for the prompt."""
    shown = observations[-_TRACE_REPAIR_MAX_OBSERVATIONS:]
    lines = []
    for obs in shown:
        flag = " [ERROR]" if obs.is_error else ""
        lines.append(f"- {obs.tool_call_id} | {obs.label}{flag} | {obs.summary}")
    return "\n".join(lines)


def _decision_trace_repair_instruction(
    final_message: ConversationMessage,
    requirement: _TraceRequirement,
    observations: list[_TraceObservation],
) -> str:
    trace_event_id = f"trace_repair_{uuid4().hex}"
    has_failure = "failed_tool_result" in requirement.signals
    has_obs = bool(observations)

    parts = [
        "The previous assistant response needs a structured decision-trace "
        "breadcrumb before finalization. Use the `trace` tool. Do not write prose, "
        "chain-of-thought, secrets, or private raw content. Keep payloads compact.",
        "",
    ]

    if has_failure:
        parts.append(
            "1. First call `trace` with kind `trace_observation` for the tool that "
            "failed this turn: payload `schema_version: 1`, a unique `trace_event_id`, "
            "`related_tool_call_id` (the failing tool_call_id below), a short "
            "`summary` of what failed, and `confidence`."
        )

    finalization_step = "2." if has_failure else "1."
    parts.append(
        f"{finalization_step} Call `trace` with kind `trace_finalization`: payload "
        f"`schema_version: 1`, `trace_event_id: \"{trace_event_id}\"`, `reason`, a short "
        "`final_answer_summary`, and `answer_claims`: a list of "
        "`{\"claim\": <short claim>, \"supported_by\": [<tool_call_id>, ...]}`. "
        "Each user-visible claim in the final answer MUST map to the tool_call_id(s) "
        "of the observation(s) below that support it. Add an `uncertainties` list "
        "(empty if none) for claims you could not ground in an observation."
    )

    if has_obs:
        parts.append("")
        parts.append(
            "Observations from this turn (use these tool_call_ids as evidence — do "
            "not invent ids, do not copy raw output):"
        )
        parts.append(_format_repair_observations(observations))
    else:
        parts.append("")
        parts.append(
            "No tool observations were recorded this turn; ground `answer_claims` in "
            "the user request and leave `supported_by` empty where there is no "
            "tool evidence."
        )

    parts.append("")
    parts.append(f"Trace requirement reason: {requirement.reason}")
    parts.append(f"Signals: {', '.join(requirement.signals)}")
    parts.append(f"Final answer summary: {_compact_text_summary(final_message.text)}")
    return "\n".join(parts)


def _record_trace_missing_required(
    recorder: DecisionTraceRecorderLike | None,
    *,
    final_message: ConversationMessage,
    requirement: _TraceRequirement,
) -> None:
    if recorder is None:
        return
    record = getattr(recorder, "record", None)
    if not callable(record):
        return

    payload = {
        "schema_version": 1,
        "trace_event_id": f"trace_missing_required_{uuid4().hex}",
        "missing": ["model_authored_trace"],
        "reason": requirement.reason,
        "signals": list(requirement.signals),
        "final_answer_length": len(final_message.text),
        "final_answer_sha256": _sha256_text(final_message.text),
        "final_answer_summary": _compact_text_summary(final_message.text),
    }
    try:
        record(_TRACE_MISSING_REQUIRED, payload, is_error=True)
    except Exception:
        log.exception("decision trace recorder failed for missing-required diagnostic")


async def _attempt_decision_trace_repair(
    context: QueryContext,
    messages: list[ConversationMessage],
    *,
    final_message: ConversationMessage,
    requirement: _TraceRequirement,
    observations: list[_TraceObservation],
    effective_max_tokens: int,
) -> bool:
    if context.decision_trace_recorder is None:
        return False

    trace_tool = context.tool_registry.get(_TRACE_TOOL_NAME)
    if trace_tool is None:
        return False

    repair_messages = [
        *messages,
        ConversationMessage.from_user_text(
            _decision_trace_repair_instruction(final_message, requirement, observations)
        ),
    ]
    repair_message: ConversationMessage | None = None
    repair_usage = UsageSnapshot()

    try:
        async for event in context.api_client.stream_message(
            ApiMessageRequest(
                model=context.model,
                messages=repair_messages,
                system_prompt=context.system_prompt,
                max_tokens=min(effective_max_tokens, 1024),
                tools=[trace_tool.to_api_schema()],
            )
        ):
            if isinstance(event, ApiMessageCompleteEvent):
                repair_message = event.message
                repair_usage = event.usage
    except Exception:
        log.exception("decision trace repair model call failed")
        return False

    if repair_message is None:
        return False

    model_payload = _model_call_trace_payload(
        repair_message,
        model=context.model,
        usage=repair_usage,
    )
    model_payload.update(
        {
            "repair": True,
            "trace_required_reason": requirement.reason,
            "trace_required_signals": list(requirement.signals),
        }
    )
    _record_decision_trace_structural(
        context.decision_trace_recorder,
        _TRACE_KIND_MODEL_CALL,
        model_payload,
    )

    recorded_finalization = False
    for tool_call in repair_message.tool_uses:
        if tool_call.name != _TRACE_TOOL_NAME:
            continue
        start_payload = _tool_started_trace_payload(tool_call.input)
        start_payload["repair"] = True
        _record_decision_trace_structural(
            context.decision_trace_recorder,
            _TRACE_KIND_TOOL_STARTED,
            start_payload,
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
        )
        tool_started_at = time.monotonic()
        result = await _execute_tool_call(
            context,
            tool_call.name,
            tool_call.id,
            tool_call.input,
        )
        duration_ms = (time.monotonic() - tool_started_at) * 1000
        completed_payload = _tool_completed_trace_payload(
            result.content,
            is_error=result.is_error,
            duration_ms=duration_ms,
        )
        completed_payload["repair"] = True
        _record_decision_trace_structural(
            context.decision_trace_recorder,
            _TRACE_KIND_TOOL_COMPLETED,
            completed_payload,
            tool_name=tool_call.name,
            tool_call_id=tool_call.id,
            is_error=result.is_error,
        )
        if (
            not result.is_error
            and _trace_tool_kind(tool_call.input) == _TRACE_MODEL_KIND_FINALIZATION
        ):
            recorded_finalization = True

    # The requirement is only satisfied by a finalization: a repair that emitted
    # only a trace_observation (e.g. a failed-tool turn) has not closed coverage.
    return recorded_finalization


def _append_capped_unique(bucket: list[Any], value: Any, *, limit: int) -> None:
    if value in bucket:
        bucket.remove(value)
    bucket.append(value)
    if len(bucket) > limit:
        del bucket[:-limit]


def _task_focus_state(tool_metadata: dict[str, object] | None) -> dict[str, object]:
    if tool_metadata is None:
        return {}
    value = tool_metadata.setdefault(
        "task_focus_state",
        {
            "goal": "",
            "recent_goals": [],
            "active_artifacts": [],
            "verified_state": [],
            "next_step": "",
        },
    )
    if isinstance(value, dict):
        value.setdefault("goal", "")
        value.setdefault("recent_goals", [])
        value.setdefault("active_artifacts", [])
        value.setdefault("verified_state", [])
        value.setdefault("next_step", "")
        return value
    replacement = {
        "goal": "",
        "recent_goals": [],
        "active_artifacts": [],
        "verified_state": [],
        "next_step": "",
    }
    tool_metadata["task_focus_state"] = replacement
    return replacement


def _summarize_focus_text(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    return normalized[:240]


def remember_user_goal(
    tool_metadata: dict[str, object] | None,
    prompt: str,
) -> None:
    state = _task_focus_state(tool_metadata)
    summary = _summarize_focus_text(prompt)
    if not summary:
        return
    recent_goals = state.setdefault("recent_goals", [])
    if isinstance(recent_goals, list):
        _append_capped_unique(recent_goals, summary, limit=MAX_TRACKED_USER_GOALS)
    state["goal"] = summary


def _remember_active_artifact(
    tool_metadata: dict[str, object] | None,
    artifact: str,
) -> None:
    normalized = artifact.strip()
    if not normalized:
        return
    state = _task_focus_state(tool_metadata)
    artifacts = state.setdefault("active_artifacts", [])
    if isinstance(artifacts, list):
        _append_capped_unique(artifacts, normalized[:240], limit=MAX_TRACKED_ACTIVE_ARTIFACTS)


def _remember_verified_work(
    tool_metadata: dict[str, object] | None,
    entry: str,
) -> None:
    normalized = entry.strip()
    if not normalized:
        return
    bucket = _tool_metadata_bucket(tool_metadata, "recent_verified_work")
    _append_capped_unique(bucket, normalized[:320], limit=MAX_TRACKED_VERIFIED_WORK)
    state = _task_focus_state(tool_metadata)
    verified_state = state.setdefault("verified_state", [])
    if isinstance(verified_state, list):
        _append_capped_unique(verified_state, normalized[:320], limit=MAX_TRACKED_VERIFIED_WORK)


def _tool_metadata_bucket(
    tool_metadata: dict[str, object] | None,
    key: str,
) -> list[Any]:
    if tool_metadata is None:
        return []
    value = tool_metadata.setdefault(key, [])
    if isinstance(value, list):
        return value
    replacement: list[Any] = []
    tool_metadata[key] = replacement
    return replacement


def _remember_read_file(
    tool_metadata: dict[str, object] | None,
    *,
    path: str,
    offset: int,
    limit: int,
    output: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "read_file_state")
    preview_lines = [line.strip() for line in output.splitlines()[:6] if line.strip()]
    entry = {
        "path": path,
        "span": f"lines {offset + 1}-{offset + limit}",
        "preview": " | ".join(preview_lines)[:320],
        "timestamp": time.time(),
    }
    if isinstance(bucket, list):
        bucket[:] = [
            existing
            for existing in bucket
            if not isinstance(existing, dict) or str(existing.get("path") or "") != path
        ]
        bucket.append(entry)
        if len(bucket) > MAX_TRACKED_READ_FILES:
            del bucket[:-MAX_TRACKED_READ_FILES]


def _remember_skill_invocation(
    tool_metadata: dict[str, object] | None,
    *,
    skill_name: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "invoked_skills")
    normalized = skill_name.strip()
    if not normalized:
        return
    if normalized in bucket:
        bucket.remove(normalized)
    bucket.append(normalized)
    if len(bucket) > MAX_TRACKED_SKILLS:
        del bucket[:-MAX_TRACKED_SKILLS]


def _remember_async_agent_activity(
    tool_metadata: dict[str, object] | None,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    output: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "async_agent_state")
    if tool_name == "agent":
        description = str(tool_input.get("description") or tool_input.get("prompt") or "").strip()
        summary = f"Spawned async agent. {description}".strip()
        if output.strip():
            summary = f"{summary} [{output.strip()[:180]}]".strip()
    elif tool_name == "send_message":
        target = str(tool_input.get("task_id") or "").strip()
        summary = f"Sent follow-up message to async agent {target}".strip()
    else:
        summary = output.strip()[:220] or f"Async agent activity via {tool_name}"
    bucket.append(summary)
    if len(bucket) > MAX_TRACKED_ASYNC_AGENT_EVENTS:
        del bucket[:-MAX_TRACKED_ASYNC_AGENT_EVENTS]


def _parse_spawned_agent_identity(
    output: str,
    metadata: dict[str, object] | None = None,
) -> tuple[str, str] | None:
    if isinstance(metadata, dict):
        agent_id = str(metadata.get("agent_id") or "").strip()
        task_id = str(metadata.get("task_id") or "").strip()
        if agent_id and task_id:
            return agent_id, task_id
    match = re.search(r"Spawned agent (.+?) \(task_id=(\S+?)(?:[,)]|$)", output.strip())
    if match is None:
        return None
    return match.group(1).strip(), match.group(2).strip()


def _remember_async_agent_task(
    tool_metadata: dict[str, object] | None,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    output: str,
    result_metadata: dict[str, object] | None = None,
) -> None:
    if tool_name != "agent":
        return
    identity = _parse_spawned_agent_identity(output, result_metadata)
    if identity is None:
        return
    agent_id, task_id = identity
    bucket = _tool_metadata_bucket(tool_metadata, "async_agent_tasks")
    description = str(tool_input.get("description") or tool_input.get("prompt") or "").strip()
    entry = {
        "agent_id": agent_id,
        "task_id": task_id,
        "description": description[:240],
        "status": "spawned",
        "notification_sent": False,
        "spawned_at": time.time(),
    }
    bucket[:] = [
        existing
        for existing in bucket
        if not isinstance(existing, dict) or str(existing.get("task_id") or "") != task_id
    ]
    bucket.append(entry)
    if len(bucket) > MAX_TRACKED_ASYNC_AGENT_TASKS:
        del bucket[:-MAX_TRACKED_ASYNC_AGENT_TASKS]


def _remember_work_log(
    tool_metadata: dict[str, object] | None,
    *,
    entry: str,
) -> None:
    bucket = _tool_metadata_bucket(tool_metadata, "recent_work_log")
    normalized = entry.strip()
    if not normalized:
        return
    bucket.append(normalized[:320])
    if len(bucket) > MAX_TRACKED_WORK_LOG:
        del bucket[:-MAX_TRACKED_WORK_LOG]


def _update_plan_mode(tool_metadata: dict[str, object] | None, mode: str) -> None:
    if tool_metadata is None:
        return
    tool_metadata["permission_mode"] = mode


def _record_tool_carryover(
    context: QueryContext,
    *,
    tool_name: str,
    tool_input: dict[str, object],
    tool_output: str,
    tool_result_metadata: dict[str, object] | None,
    is_error: bool,
    resolved_file_path: str | None,
) -> None:
    if is_error:
        return
    if resolved_file_path is not None:
        _remember_active_artifact(context.tool_metadata, resolved_file_path)
    if tool_name == "read_file" and resolved_file_path is not None:
        offset = int(tool_input.get("offset") or 0)
        limit = int(tool_input.get("limit") or 200)
        _remember_read_file(
            context.tool_metadata,
            path=resolved_file_path,
            offset=offset,
            limit=limit,
            output=tool_output,
        )
        _remember_verified_work(
            context.tool_metadata,
            f"Inspected file {resolved_file_path} (lines {offset + 1}-{offset + limit})",
        )
    elif tool_name == "skill":
        _remember_skill_invocation(
            context.tool_metadata,
            skill_name=str(tool_input.get("name") or ""),
        )
        skill_name = str(tool_input.get("name") or "").strip()
        if skill_name:
            _remember_active_artifact(context.tool_metadata, f"skill:{skill_name}")
            _remember_verified_work(context.tool_metadata, f"Loaded skill {skill_name}")
    elif tool_name in {"agent", "send_message"}:
        _remember_async_agent_activity(
            context.tool_metadata,
            tool_name=tool_name,
            tool_input=tool_input,
            output=tool_output,
        )
        _remember_async_agent_task(
            context.tool_metadata,
            tool_name=tool_name,
            tool_input=tool_input,
            output=tool_output,
            result_metadata=tool_result_metadata,
        )
        description = str(tool_input.get("description") or tool_input.get("prompt") or tool_name).strip()
        _remember_verified_work(
            context.tool_metadata,
            f"Confirmed async-agent activity via {tool_name}: {description[:180]}",
        )
    elif tool_name == "enter_plan_mode":
        _update_plan_mode(context.tool_metadata, "plan")
    elif tool_name == "exit_plan_mode":
        _update_plan_mode(context.tool_metadata, "default")
    elif tool_name == "web_fetch":
        url = str(tool_input.get("url") or "").strip()
        if url:
            _remember_active_artifact(context.tool_metadata, url)
            _remember_verified_work(context.tool_metadata, f"Fetched remote content from {url}")
    elif tool_name == "web_search":
        query = str(tool_input.get("query") or "").strip()
        if query:
            _remember_verified_work(context.tool_metadata, f"Ran web search for {query[:180]}")
    elif tool_name == "glob":
        pattern = str(tool_input.get("pattern") or "").strip()
        if pattern:
            _remember_verified_work(context.tool_metadata, f"Expanded glob pattern {pattern[:180]}")
    elif tool_name == "grep":
        pattern = str(tool_input.get("pattern") or "").strip()
        if pattern:
            _remember_verified_work(context.tool_metadata, f"Checked repository matches for grep pattern {pattern[:180]}")
    elif tool_name == "bash":
        command = str(tool_input.get("command") or "").strip()
        summary = tool_output.splitlines()[0].strip() if tool_output.strip() else "no output"
        _remember_verified_work(
            context.tool_metadata,
            f"Ran bash command {command[:160]} [{summary[:120]}]",
        )
    if tool_name == "read_file" and resolved_file_path is not None:
        _remember_work_log(
            context.tool_metadata,
            entry=f"Read file {resolved_file_path}",
        )
    elif tool_name == "bash":
        command = str(tool_input.get("command") or "").strip()
        summary = tool_output.splitlines()[0].strip() if tool_output.strip() else "no output"
        _remember_work_log(
            context.tool_metadata,
            entry=f"Ran bash: {command[:160]} [{summary[:120]}]",
        )
    elif tool_name == "grep":
        pattern = str(tool_input.get("pattern") or "").strip()
        _remember_work_log(
            context.tool_metadata,
            entry=f"Searched with grep pattern={pattern[:160]}",
        )
    elif tool_name == "skill":
        _remember_work_log(
            context.tool_metadata,
            entry=f"Loaded skill {str(tool_input.get('name') or '').strip()}",
        )
    elif tool_name in {"agent", "send_message"}:
        _remember_work_log(
            context.tool_metadata,
            entry=f"Async agent action via {tool_name}",
        )
    elif tool_name == "enter_plan_mode":
        _remember_work_log(context.tool_metadata, entry="Entered plan mode")
    elif tool_name == "exit_plan_mode":
        _remember_work_log(context.tool_metadata, entry="Exited plan mode")


def _tool_artifact_dir() -> Path:
    artifact_dir = get_data_dir() / "tool_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _safe_tool_artifact_name(tool_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name.strip())
    return (normalized or "tool")[:80]


def _offload_tool_output_if_needed(
    *,
    tool_name: str,
    tool_use_id: str,
    output: str,
) -> tuple[str, Path | None]:
    inline_limit = tool_output_inline_chars()
    if len(output) <= inline_limit:
        return output, None

    artifact_path = (
        _tool_artifact_dir()
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{_safe_tool_artifact_name(tool_name)}-{uuid4().hex[:12]}.txt"
    )
    artifact_path.write_text(output, encoding="utf-8", errors="replace")
    preview = output[:tool_output_preview_chars()]
    omitted = max(0, len(output) - len(preview))
    inline = (
        "[Tool output truncated]\n"
        f"Tool: {tool_name}\n"
        f"Tool use id: {tool_use_id}\n"
        f"Original size: {len(output)} chars\n"
        f"Full output saved to: {artifact_path}\n"
        f"Inline preview: first {len(preview)} chars"
    )
    if omitted:
        inline += f" ({omitted} chars omitted)"
    if preview:
        inline += f"\n\nPreview:\n{preview}"
    return inline, artifact_path


# ---------------------------------------------------------------------------
# Image preprocessing — convert ImageBlocks to text for non-multimodal models
# ---------------------------------------------------------------------------

_IMAGE_PREPROCESS_STATUS = "Converting image to text description via vision model…"


async def _preprocess_images_in_messages(
    messages: list[ConversationMessage],
    context: QueryContext,
) -> AsyncIterator[StreamEvent]:
    """Scan messages for ImageBlocks and convert them to text if the active
    model does not support multimodal input.

    Yields status events during conversion so the UI stays responsive.
    """
    if is_model_multimodal(context.model):
        return

    vision_config = context.tool_metadata.get("vision_model_config")
    if not vision_config:
        # No vision model configured — skip preprocessing.
        return

    # Collect all ImageBlocks with their parent message index and block index
    pending: list[tuple[int, int, ImageBlock]] = []
    for msg_idx, msg in enumerate(messages):
        if msg.role != "user":
            continue
        for blk_idx, block in enumerate(msg.content):
            if isinstance(block, ImageBlock):
                pending.append((msg_idx, blk_idx, block))

    if not pending:
        return

    yield StatusEvent(message=_IMAGE_PREPROCESS_STATUS)

    # Process images in parallel
    async def _describe(msg_idx: int, blk_idx: int, block: ImageBlock) -> tuple[int, int, str]:
        tool = context.tool_registry.get("image_to_text")
        if tool is None:
            return msg_idx, blk_idx, "[Image: could not describe — image_to_text tool not available]"

        # Build tool input
        tool_input_data: dict[str, object] = {
            "image_data": block.data,
            "media_type": block.media_type,
            "prompt": "Describe this image in detail, including any text, "
                      "UI elements, code, diagrams, or visual information present.",
        }

        try:
            parsed = tool.input_model.model_validate(tool_input_data)
        except Exception:
            return msg_idx, blk_idx, "[Image: could not parse image data]"

        exec_context = ToolExecutionContext(
            cwd=context.cwd,
            metadata=_tool_execution_metadata(
                context,
                {"vision_model_config": vision_config},
            ),
        )
        result = await tool.execute(parsed, exec_context)
        if result.is_error:
            return msg_idx, blk_idx, f"[Image description failed: {result.output}]"
        return msg_idx, blk_idx, result.output

    results = await asyncio.gather(*[_describe(mi, bi, blk) for mi, bi, blk in pending])

    # Replace ImageBlocks with TextBlocks in-place
    for msg_idx, blk_idx, description in results:
        msg = messages[msg_idx]
        msg.content[blk_idx] = TextBlock(text=description)


async def run_query(
    context: QueryContext,
    messages: list[ConversationMessage],
) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
    """Run the conversation loop until the model stops requesting tools.

    Auto-compaction is checked at the start of each turn.  When the
    estimated token count exceeds the model's auto-compact threshold,
    the engine first tries a cheap microcompact (clearing old tool result
    content) and, if that is not enough, performs a full LLM-based
    summarization of older messages.
    """
    from openharness.services.compact import (
        AutoCompactState,
        auto_compact_if_needed,
    )

    compact_state = AutoCompactState()
    reactive_compact_attempted = False
    last_compaction_result: tuple[list[ConversationMessage], bool] = (messages, False)
    effective_max_tokens = _bounded_completion_tokens(
        context.max_tokens,
        context.context_window_tokens,
    )
    reported_token_clamp = False
    trace_run_state = _DecisionTraceRunState()
    prior_tool_results_seen = _messages_include_tool_results(messages)

    async def _stream_compaction(
        *,
        trigger: str,
        force: bool = False,
    ) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
        nonlocal last_compaction_result
        progress_queue: asyncio.Queue[CompactProgressEvent] = asyncio.Queue()

        async def _progress(event: CompactProgressEvent) -> None:
            await progress_queue.put(event)

        task = asyncio.create_task(
            auto_compact_if_needed(
                messages,
                api_client=context.api_client,
                model=context.model,
                system_prompt=context.system_prompt,
                state=compact_state,
                progress_callback=_progress,
                force=force,
                trigger=trigger,
                hook_executor=context.hook_executor,
                carryover_metadata=context.tool_metadata,
                context_window_tokens=context.context_window_tokens,
                auto_compact_threshold_tokens=context.auto_compact_threshold_tokens,
            )
        )
        while True:
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=0.05)
                yield event, None
            except asyncio.TimeoutError:
                if task.done():
                    break
                continue
        while not progress_queue.empty():
            yield progress_queue.get_nowait(), None
        last_compaction_result = await task
        return

    turn_count = 0
    while context.max_turns is None or turn_count < context.max_turns:
        turn_count += 1
        if effective_max_tokens != context.max_tokens and not reported_token_clamp:
            reported_token_clamp = True
            yield StatusEvent(
                message=(
                    "Requested max_tokens="
                    f"{context.max_tokens} exceeds the safe per-request output cap; "
                    f"using {effective_max_tokens}."
                )
            ), None
        # --- auto-compact check before calling the model ---------------
        async for event, usage in _stream_compaction(trigger="auto"):
            yield event, usage
        compacted_messages, was_compacted = last_compaction_result
        if compacted_messages is not messages:
            messages[:] = compacted_messages
        # ---------------------------------------------------------------

        # --- image preprocessing: convert ImageBlocks to text for non-vision models ---
        async for event in _preprocess_images_in_messages(messages, context):
            yield event, None
        # -----------------------------------------------------------------------------

        final_message: ConversationMessage | None = None
        usage = UsageSnapshot()

        try:
            async for event in context.api_client.stream_message(
                ApiMessageRequest(
                    model=context.model,
                    messages=messages,
                    system_prompt=context.system_prompt,
                    max_tokens=effective_max_tokens,
                    tools=context.tool_registry.to_api_schema(),
                    effort=context.effort,
                )
            ):
                if isinstance(event, ApiTextDeltaEvent):
                    yield AssistantTextDelta(text=event.text), None
                    continue
                if isinstance(event, ApiRetryEvent):
                    yield StatusEvent(
                        message=(
                            f"Request failed; retrying in {event.delay_seconds:.1f}s "
                            f"(attempt {event.attempt + 1} of {event.max_attempts}): {event.message}"
                        )
                    ), None
                    continue

                if isinstance(event, ApiMessageCompleteEvent):
                    final_message = event.message
                    usage = event.usage
        except Exception as exc:
            error_msg = str(exc)
            if _is_completion_token_limit_error(exc):
                supported_limit = _extract_completion_token_limit(exc)
                if supported_limit is not None and effective_max_tokens > supported_limit:
                    previous_max_tokens = effective_max_tokens
                    effective_max_tokens = supported_limit
                    yield StatusEvent(
                        message=(
                            f"Model rejected max_tokens={previous_max_tokens}; "
                            f"retrying with provider limit {effective_max_tokens}."
                        )
                    ), None
                    turn_count = max(0, turn_count - 1)
                    continue
            if not reactive_compact_attempted and _is_prompt_too_long_error(exc):
                reactive_compact_attempted = True
                yield StatusEvent(message=REACTIVE_COMPACT_STATUS_MESSAGE), None
                async for event, usage in _stream_compaction(trigger="reactive", force=True):
                    yield event, usage
                compacted_messages, was_compacted = last_compaction_result
                if compacted_messages is not messages:
                    messages[:] = compacted_messages
                if was_compacted:
                    continue
            if "connect" in error_msg.lower() or "timeout" in error_msg.lower() or "network" in error_msg.lower():
                _record_decision_trace_structural(
                    context.decision_trace_recorder,
                    _TRACE_KIND_ENGINE_ERROR,
                    _engine_error_trace_payload(
                        error_msg,
                        recoverable=True,
                        error_type=type(exc).__name__,
                    ),
                    is_error=True,
                )
                yield ErrorEvent(message=f"Network error: {error_msg}. Check your internet connection and try again."), None
            else:
                _record_decision_trace_structural(
                    context.decision_trace_recorder,
                    _TRACE_KIND_ENGINE_ERROR,
                    _engine_error_trace_payload(
                        error_msg,
                        recoverable=False,
                        error_type=type(exc).__name__,
                    ),
                    is_error=True,
                )
                yield ErrorEvent(message=f"API error: {error_msg}"), None
            return

        if final_message is None:
            _record_decision_trace_structural(
                context.decision_trace_recorder,
                _TRACE_KIND_ENGINE_ERROR,
                _engine_error_trace_payload(
                    "Model stream finished without a final message",
                    recoverable=False,
                    error_type="RuntimeError",
                ),
                is_error=True,
            )
            raise RuntimeError("Model stream finished without a final message")

        coordinator_context_message: ConversationMessage | None = None
        if context.system_prompt.startswith("You are a **coordinator**."):
            if messages and messages[-1].role == "user" and messages[-1].text.startswith("# Coordinator User Context"):
                coordinator_context_message = messages.pop()

        if final_message.role == "assistant" and final_message.is_effectively_empty():
            log.warning("dropping empty assistant message from provider response")
            _record_decision_trace_structural(
                context.decision_trace_recorder,
                _TRACE_KIND_ENGINE_ERROR,
                _engine_error_trace_payload(
                    "Model returned an empty assistant message.",
                    recoverable=False,
                    error_type="EmptyAssistantMessage",
                ),
                is_error=True,
            )
            yield ErrorEvent(
                message=(
                    "Model returned an empty assistant message. "
                    "The turn was ignored to keep the session healthy."
                )
            ), usage
            return

        messages.append(final_message)
        _record_decision_trace_structural(
            context.decision_trace_recorder,
            _TRACE_KIND_MODEL_CALL,
            _model_call_trace_payload(
                final_message,
                model=context.model,
                usage=usage,
            ),
        )

        if coordinator_context_message is not None:
            messages.append(coordinator_context_message)

        if not final_message.tool_uses:
            trace_requirement = _classify_trace_requirement(
                final_message,
                run_state=trace_run_state,
                prior_tool_results_seen=prior_tool_results_seen,
            )
            if trace_requirement.required and not trace_run_state.successful_finalization:
                repaired_finalization = await _attempt_decision_trace_repair(
                    context,
                    messages,
                    final_message=final_message,
                    requirement=trace_requirement,
                    observations=trace_run_state.observations,
                    effective_max_tokens=effective_max_tokens,
                )
                if repaired_finalization:
                    trace_run_state.successful_finalization = True
                    trace_run_state.successful_model_trace = True
                else:
                    _record_trace_missing_required(
                        context.decision_trace_recorder,
                        final_message=final_message,
                        requirement=trace_requirement,
                    )

            yield AssistantTurnComplete(message=final_message, usage=usage), usage
            _record_decision_trace_structural(
                context.decision_trace_recorder,
                _TRACE_KIND_ASSISTANT_FINAL,
                _assistant_final_trace_payload(
                    final_message,
                    model=context.model,
                    trace_requirement=trace_requirement,
                    model_trace_recorded=trace_run_state.successful_finalization,
                ),
            )
            if context.hook_executor is not None:
                await context.hook_executor.execute(
                    HookEvent.STOP,
                    {
                        "event": HookEvent.STOP.value,
                        "stop_reason": "tool_uses_empty",
                    },
                )
            return

        yield AssistantTurnComplete(message=final_message, usage=usage), usage
        tool_calls = final_message.tool_uses
        trace_run_state.tool_call_count += len(tool_calls)

        if len(tool_calls) == 1:
            # Single tool: sequential (stream events immediately)
            tc = tool_calls[0]
            _record_decision_trace_structural(
                context.decision_trace_recorder,
                _TRACE_KIND_TOOL_STARTED,
                _tool_started_trace_payload(tc.input),
                tool_name=tc.name,
                tool_call_id=tc.id,
            )
            yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input, tool_call_id=tc.id), None
            tool_started_at = time.monotonic()
            try:
                result = await _execute_tool_call(context, tc.name, tc.id, tc.input)
            except Exception as exc:
                log.exception("tool execution raised: name=%s id=%s", tc.name, tc.id)
                result = ToolResultBlock(
                    tool_use_id=tc.id,
                    content=f"Tool {tc.name} failed: {type(exc).__name__}: {exc}",
                    is_error=True,
                )
            duration_ms = (time.monotonic() - tool_started_at) * 1000
            trace_run_state.tool_result_count += 1
            if result.is_error:
                trace_run_state.failed_tool_result_count += 1
            if tc.name == _TRACE_TOOL_NAME and not result.is_error:
                trace_run_state.successful_model_trace = True
                if _trace_tool_kind(tc.input) == _TRACE_MODEL_KIND_FINALIZATION:
                    trace_run_state.successful_finalization = True
            _record_trace_observation(trace_run_state, tc, result)
            _record_decision_trace_structural(
                context.decision_trace_recorder,
                _TRACE_KIND_TOOL_COMPLETED,
                _tool_completed_trace_payload(
                    result.content,
                    is_error=result.is_error,
                    duration_ms=duration_ms,
                ),
                tool_name=tc.name,
                tool_call_id=tc.id,
                is_error=result.is_error,
            )
            yield ToolExecutionCompleted(
                tool_name=tc.name,
                output=result.content,
                is_error=result.is_error,
                tool_call_id=tc.id,
                metadata=result.result_metadata,
            ), None
            tool_results = [result]
        else:
            # Multiple tools: execute concurrently, emit events after
            for tc in tool_calls:
                _record_decision_trace_structural(
                    context.decision_trace_recorder,
                    _TRACE_KIND_TOOL_STARTED,
                    _tool_started_trace_payload(tc.input),
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                )
                yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input, tool_call_id=tc.id), None

            async def _run(tc):
                tool_started_at = time.monotonic()
                try:
                    result = await _execute_tool_call(context, tc.name, tc.id, tc.input)
                except Exception as exc:
                    return exc, (time.monotonic() - tool_started_at) * 1000
                return result, (time.monotonic() - tool_started_at) * 1000

            # Use return_exceptions=True so a single failing tool does not abandon
            # its siblings as cancelled coroutines and leave the conversation with
            # un-replied tool_use blocks (Anthropic's API rejects the next request
            # on the session if any tool_use is missing a matching tool_result).
            raw_results = await asyncio.gather(
                *[_run(tc) for tc in tool_calls], return_exceptions=True
            )
            tool_results = []
            durations_ms = []
            for tc, result in zip(tool_calls, raw_results):
                duration_ms = 0.0
                if isinstance(result, tuple):
                    result, duration_ms = result
                if isinstance(result, BaseException):
                    log.exception(
                        "tool execution raised: name=%s id=%s",
                        tc.name,
                        tc.id,
                        exc_info=result,
                    )
                    result = ToolResultBlock(
                        tool_use_id=tc.id,
                        content=f"Tool {tc.name} failed: {type(result).__name__}: {result}",
                        is_error=True,
                    )
                tool_results.append(result)
                durations_ms.append(duration_ms)
                trace_run_state.tool_result_count += 1
                if result.is_error:
                    trace_run_state.failed_tool_result_count += 1
                if tc.name == _TRACE_TOOL_NAME and not result.is_error:
                    trace_run_state.successful_model_trace = True
                    if _trace_tool_kind(tc.input) == _TRACE_MODEL_KIND_FINALIZATION:
                        trace_run_state.successful_finalization = True
                _record_trace_observation(trace_run_state, tc, result)

            for tc, result, duration_ms in zip(tool_calls, tool_results, durations_ms):
                _record_decision_trace_structural(
                    context.decision_trace_recorder,
                    _TRACE_KIND_TOOL_COMPLETED,
                    _tool_completed_trace_payload(
                        result.content,
                        is_error=result.is_error,
                        duration_ms=duration_ms,
                    ),
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    is_error=result.is_error,
                )
                yield ToolExecutionCompleted(
                    tool_name=tc.name,
                    output=result.content,
                    is_error=result.is_error,
                    tool_call_id=tc.id,
                    metadata=result.result_metadata,
                ), None

        messages.append(ConversationMessage(role="user", content=tool_results))

    if context.max_turns is not None:
        _record_decision_trace_structural(
            context.decision_trace_recorder,
            _TRACE_KIND_ENGINE_ERROR,
            _engine_error_trace_payload(
                f"Exceeded maximum turn limit ({context.max_turns})",
                recoverable=False,
                error_type="MaxTurnsExceeded",
            ),
            is_error=True,
        )
        raise MaxTurnsExceeded(context.max_turns)
    raise RuntimeError("Query loop exited without a max_turns limit or final response")


async def _execute_tool_call(
    context: QueryContext,
    tool_name: str,
    tool_use_id: str,
    tool_input: dict[str, object],
) -> ToolResultBlock:
    if context.hook_executor is not None:
        pre_hooks = await context.hook_executor.execute(
            HookEvent.PRE_TOOL_USE,
            {"tool_name": tool_name, "tool_input": tool_input, "event": HookEvent.PRE_TOOL_USE.value},
        )
        if pre_hooks.blocked:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=pre_hooks.reason or f"pre_tool_use hook blocked {tool_name}",
                is_error=True,
            )

    log.debug("tool_call start: %s id=%s", tool_name, tool_use_id)

    tool = context.tool_registry.get(tool_name)
    if tool is None:
        log.warning("unknown tool: %s", tool_name)
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Unknown tool: {tool_name}",
            is_error=True,
        )

    try:
        parsed_input = tool.input_model.model_validate(tool_input)
    except Exception as exc:
        log.warning("invalid input for %s: %s", tool_name, exc)
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Invalid input for {tool_name}: {exc}",
            is_error=True,
        )

    # Normalize common tool inputs before permission checks so path rules apply
    # consistently across built-in tools that use `file_path`, `path`, or
    # directory-scoped roots such as `glob`/`grep`.
    _file_path = _resolve_permission_file_path(context.cwd, tool_input, parsed_input)
    _command = _extract_permission_command(tool_input, parsed_input)
    read_only = tool.is_read_only(parsed_input)
    log.debug("permission check: %s read_only=%s path=%s cmd=%s",
              tool_name, read_only, _file_path, _command and _command[:80])
    decision = context.permission_checker.evaluate(
        tool_name,
        is_read_only=read_only,
        file_path=_file_path,
        command=_command,
    )
    _record_decision_trace_structural(
        context.decision_trace_recorder,
        _TRACE_KIND_TOOL_PERMISSION,
        _tool_permission_trace_payload(
            allowed=decision.allowed,
            requires_confirmation=decision.requires_confirmation,
            reason=decision.reason,
            read_only=read_only,
            file_path=_file_path,
            command=_command,
        ),
        tool_name=tool_name,
        tool_call_id=tool_use_id,
        is_error=not decision.allowed and not decision.requires_confirmation,
    )
    if not decision.allowed:
        if decision.requires_confirmation and context.permission_prompt is not None:
            log.debug("permission prompt for %s: %s", tool_name, decision.reason)
            if context.hook_executor is not None:
                await context.hook_executor.execute(
                    HookEvent.NOTIFICATION,
                    {
                        "event": HookEvent.NOTIFICATION.value,
                        "notification_type": "permission_prompt",
                        "tool_name": tool_name,
                        "reason": decision.reason,
                    },
                )
            confirmed = await context.permission_prompt(tool_name, decision.reason)
            if not confirmed:
                log.debug("permission denied by user for %s", tool_name)
                return ToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=decision.reason or f"Permission denied for {tool_name}",
                    is_error=True,
                )
        else:
            log.debug("permission blocked for %s: %s", tool_name, decision.reason)
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=decision.reason or f"Permission denied for {tool_name}",
                is_error=True,
            )

    log.debug("executing %s ...", tool_name)
    t0 = time.monotonic()
    try:
        result = await tool.execute(
            parsed_input,
            ToolExecutionContext(
                cwd=context.cwd,
                metadata=_tool_execution_metadata(
                    context,
                    {
                        "tool_registry": context.tool_registry,
                        "ask_user_prompt": context.ask_user_prompt,
                    },
                ),
                hook_executor=context.hook_executor,
            ),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Never let a failing/timed-out tool abort the turn. The single-tool path
        # (see caller) appends a tool_result only if this returns; a propagated
        # exception there would leave a dangling tool_use and poison the session
        # (model API: "No tool output found for function call ...").
        log.exception("tool execution failed: %s id=%s", tool_name, tool_use_id)
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Tool {tool_name} failed: {type(exc).__name__}: {exc}",
            is_error=True,
        )
    elapsed = time.monotonic() - t0
    log.debug("executed %s in %.2fs err=%s output_len=%d",
              tool_name, elapsed, result.is_error, len(result.output or ""))
    inline_output, artifact_path = _offload_tool_output_if_needed(
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        output=result.output,
    )
    if artifact_path is not None:
        _remember_active_artifact(context.tool_metadata, str(artifact_path))
    tool_result = ToolResultBlock(
        tool_use_id=tool_use_id,
        content=inline_output,
        is_error=result.is_error,
        result_metadata=dict(result.metadata or {}),
    )
    _record_tool_carryover(
        context,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_result.content,
        tool_result_metadata=result.metadata,
        is_error=tool_result.is_error,
        resolved_file_path=_file_path,
    )
    if context.hook_executor is not None:
        await context.hook_executor.execute(
            HookEvent.POST_TOOL_USE,
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_result.content,
                "tool_is_error": tool_result.is_error,
                "event": HookEvent.POST_TOOL_USE.value,
            },
        )
    return tool_result


def _resolve_permission_file_path(
    cwd: Path,
    raw_input: dict[str, object],
    parsed_input: object,
) -> str | None:
    for key in ("file_path", "path", "root"):
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = cwd / path
            return str(path.resolve())

    for attr in ("file_path", "path", "root"):
        value = getattr(parsed_input, attr, None)
        if isinstance(value, str) and value.strip():
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = cwd / path
            return str(path.resolve())

    return None


def _extract_permission_command(
    raw_input: dict[str, object],
    parsed_input: object,
) -> str | None:
    value = raw_input.get("command")
    if isinstance(value, str) and value.strip():
        return value

    value = getattr(parsed_input, "command", None)
    if isinstance(value, str) and value.strip():
        return value

    return None
