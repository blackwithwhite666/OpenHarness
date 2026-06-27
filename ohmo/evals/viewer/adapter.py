"""Adapters from ohmo eval episodes to trace-viewer DTOs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from openharness.evals import EvalEpisode, EvalEvent
from openharness.evals.decision_trace import (
    DECISION_TRACE_EVENT_KINDS,
    STRUCTURAL_ASSISTANT_FINAL,
    TRACE_MISSING_REQUIRED,
    TRACE_UNCERTAINTY,
)
from openharness.evals.decision_trace_summary import (
    summarize_decision_trace,
    unsupported_claim_count_from_payload,
)
from openharness.evals.tool_labels import effective_tool_label

TOOL_COMPLETED_KINDS = {"tool_completed", "tool_completed_error"}
_TRACE_TOOL_NAME = "trace"
_ASSISTANT_FINAL_TRACE_METADATA_KEYS = (
    "model",
    "stop_reason",
    "content_blocks",
    "block_counts",
    "trace_required",
    "trace_required_reason",
    "trace_required_signals",
    "model_trace_recorded",
)
_DECISION_TRACE_ATTRIBUTE_KEYS = (
    "trace_event_id",
    "related_tool_call_id",
    "parent_event_id",
    "sensitivity",
    "retention",
    "reason",
    "signals",
    "missing",
)


def episode_to_trace_viewer_data(store: Any, episode_id: str) -> dict[str, Any]:
    """Return one episode as the JSON-friendly trace viewer DTO."""
    episode = store.get_episode(episode_id)
    if episode is None:
        raise KeyError(episode_id)

    events = list(store.iter_events(episode_id))
    child_spans = _child_spans(episode_id, events)
    decision_trace_summary = summarize_decision_trace(events)
    start_ms, end_ms = _trace_bounds_ms(episode, events)
    final_output = _gateway_final_text(events)
    total_tokens = _model_call_total_tokens(events)
    root_status = "error" if any(span["status"] == "error" for span in child_spans) else "success"
    duration_ms = max(0, end_ms - start_ms)

    root_span = {
        "id": episode.episode_id,
        "title": episode.user_text or "(turn)",
        "startTimeMs": start_ms,
        "endTimeMs": end_ms,
        "durationMs": duration_ms,
        "type": "agent_invocation",
        "status": root_status,
        "input": episode.user_text or None,
        "output": final_output,
        "raw": _json_dumps(
            {
                "episode": _model_dump(episode),
                "events": [_safe_event_dump(event) for event in events],
            }
        ),
        "attributes": _root_attributes(episode, decision_trace_summary),
        "children": child_spans,
    }

    return {
        "traceRecord": {
            "id": episode.episode_id,
            "name": episode.user_text or "(turn)",
            "spansCount": 1 + len(child_spans),
            "durationMs": duration_ms,
            "agentDescription": _string_value(episode.metadata.get("model")) or "",
            "startTimeMs": start_ms,
            "totalTokens": total_tokens,
        },
        "spans": [root_span],
    }


def list_prod_traces(
    store: Any,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Return a paged summary of production traces in newest-first order."""
    query = (q or "").casefold()
    episodes = [
        episode
        for episode_id in store.list_episode_ids()
        if (episode := store.get_episode(episode_id)) is not None
    ]
    if query:
        episodes = [episode for episode in episodes if query in episode.user_text.casefold()]

    episodes.sort(key=lambda episode: episode.created_at, reverse=True)
    total = len(episodes)
    bounded_offset = max(0, offset)
    bounded_limit = max(0, limit)
    page = episodes[bounded_offset : bounded_offset + bounded_limit]

    return {
        "traces": [_prod_trace_summary(store, episode) for episode in page],
        "total": total,
    }


def episode_session_conversation(store: Any, episode_id: str) -> dict[str, Any]:
    """Render the whole session containing ``episode_id`` as a chat transcript.

    Each episode is one user turn (``user_text``) plus the gateway's reply
    (``gateway_final`` text) with a compact list of the tool calls it made.
    Works for any prod/gold episode id; the anchor episode is flagged so the UI
    can highlight it. Raises ``KeyError`` when the episode is unknown.
    """
    anchor = store.get_episode(episode_id)
    if anchor is None:
        raise KeyError(episode_id)

    session_id = anchor.session_id or ""
    episode_ids = (
        store.list_session_episode_ids(session_id) if session_id else [episode_id]
    )
    if episode_id not in episode_ids:
        episode_ids = [*episode_ids, episode_id]

    messages: list[dict[str, Any]] = []
    for eid in episode_ids:
        episode = store.get_episode(eid)
        if episode is None:
            continue
        events = list(store.iter_events(eid))
        created_ms = _epoch_ms(episode.created_at)
        if episode.user_text:
            messages.append(
                {
                    "role": "user",
                    "text": episode.user_text,
                    "ts": created_ms,
                    "episodeId": eid,
                    "toolCalls": [],
                    "status": "success",
                }
            )
        tool_summaries = _episode_tool_summaries(eid, events)
        final_text = _gateway_final_text(events)
        end_ms = _epoch_ms(events[-1].timestamp) if events else created_ms
        if final_text or tool_summaries:
            messages.append(
                {
                    "role": "assistant",
                    "text": final_text or "",
                    "ts": end_ms,
                    "episodeId": eid,
                    "toolCalls": tool_summaries,
                    "status": "error"
                    if any(summary["status"] == "error" for summary in tool_summaries)
                    else "success",
                }
            )

    return {
        "title": anchor.user_text or session_id or episode_id,
        "kind": "session",
        "sessionId": session_id,
        "anchorEpisodeId": episode_id,
        "messages": messages,
    }


def _episode_tool_summaries(episode_id: str, events: list[EvalEvent]) -> list[dict[str, str]]:
    return [
        {"name": span["title"], "status": span["status"]}
        for span in _tool_spans(episode_id, events)
    ]


def category_for(tool_name: str | None) -> str:
    """Map an ohmo tool name to the trace-viewer span category."""
    name = (tool_name or "").strip()
    if "google_search" in name or name == "web_fetch":
        return "retrieval"
    if name == "todo_write":
        return "event"
    if name.startswith(("bash", "mcp__")) or name:
        return "tool_execution"
    return "unknown"


def _child_spans(episode_id: str, events: list[EvalEvent]) -> list[dict[str, Any]]:
    child_spans = [
        *_model_call_spans(episode_id, events),
        *_tool_spans(episode_id, events),
        *_decision_trace_spans(episode_id, events),
    ]
    return sorted(child_spans, key=lambda span: _int_value(span.get("startTimeMs")))


def _tool_spans(episode_id: str, events: list[EvalEvent]) -> list[dict[str, Any]]:
    pending: dict[str, list[tuple[int, EvalEvent]]] = {}
    spans: list[dict[str, Any]] = []

    for index, event in enumerate(events):
        if event.kind == "tool_started":
            key = _tool_call_key(event, index)
            pending.setdefault(key, []).append((index, event))
            continue

        if event.kind not in TOOL_COMPLETED_KINDS:
            continue

        key = _tool_call_key(event, index)
        starts = pending.get(key)
        if not starts:
            continue
        start_index, start_event = starts.pop(0)
        if not starts:
            pending.pop(key, None)
        spans.append(_tool_span(episode_id, start_index, start_event, event))

    return spans


def _tool_span(
    episode_id: str,
    start_index: int,
    start_event: EvalEvent,
    completed_event: EvalEvent,
) -> dict[str, Any]:
    input_value = _tool_span_input(start_event)
    output_value = (completed_event.payload or {}).get("output")
    start_ms = _epoch_ms(start_event.timestamp)
    end_ms = _epoch_ms(completed_event.timestamp)
    tool_call_id = start_event.tool_call_id or completed_event.tool_call_id
    tool_name = start_event.tool_name or completed_event.tool_name

    return {
        "id": tool_call_id or f"{episode_id}:tool:{start_index}",
        "title": effective_tool_label(tool_name or "", input_value or {}),
        "startTimeMs": start_ms,
        "endTimeMs": end_ms,
        "durationMs": max(0, end_ms - start_ms),
        "type": category_for(tool_name),
        "status": "error" if completed_event.is_error else "success",
        "input": _json_or_none(input_value),
        "output": _text_or_json(output_value),
        "raw": _json_dumps(
            {
                "started": _safe_event_dump(start_event),
                "completed": _safe_event_dump(completed_event),
            }
        ),
        "attributes": _tool_attributes(tool_call_id, tool_name),
        "children": [],
    }


def _model_call_spans(episode_id: str, events: list[EvalEvent]) -> list[dict[str, Any]]:
    if not events:
        return []

    spans: list[dict[str, Any]] = []
    first_event = events[0]
    previous_event: EvalEvent | None = None
    ordinal = 0
    for event in events:
        if event.kind == "model_call":
            ordinal += 1
            spans.append(
                _model_call_span(
                    episode_id,
                    ordinal,
                    previous_event or first_event,
                    event,
                )
            )
        previous_event = event
    return spans


def _model_call_span(
    episode_id: str,
    index: int,
    start_event: EvalEvent,
    event: EvalEvent,
) -> dict[str, Any]:
    payload = event.payload or {}
    model = _string_value(payload.get("model")) or f"model call {index}"
    input_tokens = _int_value(payload.get("input_tokens"))
    output_tokens = _int_value(payload.get("output_tokens"))
    start_ms = _epoch_ms(start_event.timestamp)
    end_ms = _epoch_ms(event.timestamp)
    return {
        "id": f"{episode_id}:llm:{index}",
        "title": model,
        "startTimeMs": start_ms,
        "endTimeMs": end_ms,
        "durationMs": max(0, end_ms - start_ms),
        "type": "llm_call",
        "status": "success",
        "tokensCount": input_tokens + output_tokens,
        "input": None,
        "output": None,
        "raw": _json_dumps(_model_dump(event)),
        "attributes": _model_call_attributes(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        "children": [],
    }


def _decision_trace_spans(episode_id: str, events: list[EvalEvent]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.kind not in DECISION_TRACE_EVENT_KINDS:
            continue
        spans.append(_decision_trace_span(episode_id, index, event))
    return spans


def _decision_trace_span(
    episode_id: str,
    index: int,
    event: EvalEvent,
) -> dict[str, Any]:
    payload_metadata = _decision_trace_payload_metadata(event)
    trace_event_id = _string_value(payload_metadata.get("trace_event_id"))
    timestamp_ms = _epoch_ms(event.timestamp)
    return {
        "id": f"{episode_id}:decision_trace:{trace_event_id or index}",
        "title": event.kind,
        "startTimeMs": timestamp_ms,
        "endTimeMs": timestamp_ms,
        "durationMs": 0,
        "type": "decision_trace",
        "status": "error" if event.kind == TRACE_MISSING_REQUIRED or event.is_error else "success",
        "input": None,
        "output": None,
        "raw": _json_dumps(
            {
                "kind": event.kind,
                "timestamp": event.timestamp.isoformat(),
                "tool_name": event.tool_name,
                "tool_call_id": event.tool_call_id,
                "is_error": event.is_error,
                "payload_metadata": payload_metadata,
            }
        ),
        "attributes": _decision_trace_attributes(event, payload_metadata),
        "children": [],
    }


def _prod_trace_summary(store: Any, episode: EvalEpisode) -> dict[str, Any]:
    events = list(store.iter_events(episode.episode_id))
    child_spans = _child_spans(episode.episode_id, events)
    start_ms, end_ms = _trace_bounds_ms(episode, events)
    return {
        "id": episode.episode_id,
        "name": episode.user_text or "(turn)",
        "kind": "prod",
        "createdAt": _epoch_ms(episode.created_at),
        "status": "error" if any(span["status"] == "error" for span in child_spans) else "success",
        "spansCount": 1 + len(child_spans),
        "durationMs": max(0, end_ms - start_ms),
    }


def _root_attributes(
    episode: EvalEpisode,
    decision_trace_summary: dict[str, Any],
) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "model", episode.metadata.get("model"))
    _append_attribute(attributes, "cwd", episode.metadata.get("cwd"))
    _append_attributes(attributes, decision_trace_summary)
    return attributes


def _decision_trace_attributes(
    event: EvalEvent,
    payload_metadata: dict[str, Any],
) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "kind", event.kind)
    for key in _DECISION_TRACE_ATTRIBUTE_KEYS:
        _append_attribute(attributes, key, payload_metadata.get(key))
    _append_attribute(
        attributes,
        "unsupported_claim_count",
        payload_metadata.get("unsupported_claim_count", 0),
    )
    _append_attribute(attributes, "uncertainty_status", payload_metadata.get("uncertainty_status"))
    return attributes


def _tool_attributes(
    tool_call_id: str | None,
    tool_name: str | None,
) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "tool_call_id", tool_call_id)
    _append_attribute(attributes, "tool_name", tool_name)
    return attributes


def _model_call_attributes(
    *,
    input_tokens: int,
    output_tokens: int,
) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "input_tokens", input_tokens)
    _append_attribute(attributes, "output_tokens", output_tokens)
    return attributes


def _append_attributes(
    attributes: list[dict[str, dict[str, str] | str]],
    values: dict[str, Any],
) -> None:
    for key, value in values.items():
        _append_attribute(attributes, key, value)


def _append_attribute(
    attributes: list[dict[str, dict[str, str] | str]],
    key: str,
    value: Any,
) -> None:
    string_value = _string_value(value)
    if string_value is None:
        return
    attributes.append({"key": key, "value": {"stringValue": string_value}})


def _decision_trace_payload_metadata(event: EvalEvent) -> dict[str, Any]:
    payload = event.payload or {}
    metadata: dict[str, Any] = {}
    for key in _DECISION_TRACE_ATTRIBUTE_KEYS:
        if key in payload:
            metadata[key] = payload[key]
    if "schema_version" in payload:
        metadata["schema_version"] = payload["schema_version"]
    metadata["unsupported_claim_count"] = unsupported_claim_count_from_payload(payload)
    if event.kind == TRACE_UNCERTAINTY:
        metadata["uncertainty_status"] = _string_value(payload.get("uncertainty_status")) or "present"
    elif "uncertainty_status" in payload:
        metadata["uncertainty_status"] = payload["uncertainty_status"]
    return metadata


def _tool_span_input(event: EvalEvent) -> Any:
    input_value = (event.payload or {}).get("input")
    if _is_legacy_trace_tool_started(event):
        return _legacy_trace_tool_input_metadata(input_value)
    return input_value


def _legacy_trace_tool_input_metadata(input_value: Any) -> dict[str, Any]:
    if not isinstance(input_value, dict):
        return {}

    metadata: dict[str, Any] = {}
    kind = _string_value(input_value.get("kind"))
    if kind is not None:
        metadata["kind"] = kind

    raw_payload = input_value.get("payload")
    payload = raw_payload if isinstance(raw_payload, dict) else input_value
    for key in _DECISION_TRACE_ATTRIBUTE_KEYS:
        if key in payload:
            metadata[key] = payload[key]
    if "schema_version" in payload:
        metadata["schema_version"] = payload["schema_version"]

    metadata["unsupported_claim_count"] = unsupported_claim_count_from_payload(payload)
    if kind == TRACE_UNCERTAINTY:
        metadata["uncertainty_status"] = _string_value(payload.get("uncertainty_status")) or "present"
    elif "uncertainty_status" in payload:
        metadata["uncertainty_status"] = payload["uncertainty_status"]
    return metadata


def _is_legacy_trace_tool_started(event: EvalEvent) -> bool:
    return event.kind == "tool_started" and event.tool_name == _TRACE_TOOL_NAME


def _safe_event_dump(event: EvalEvent) -> dict[str, Any]:
    dumped = _model_dump(event)
    if event.kind in DECISION_TRACE_EVENT_KINDS:
        dumped["payload"] = _decision_trace_payload_metadata(event)
    elif _is_legacy_trace_tool_started(event):
        dumped["payload"] = {
            "input": _legacy_trace_tool_input_metadata((event.payload or {}).get("input"))
        }
    elif event.kind == STRUCTURAL_ASSISTANT_FINAL:
        payload = event.payload or {}
        dumped["payload"] = {
            key: payload[key]
            for key in _ASSISTANT_FINAL_TRACE_METADATA_KEYS
            if key in payload
        }
    return dumped


def _trace_bounds_ms(episode: EvalEpisode, events: list[EvalEvent]) -> tuple[int, int]:
    if not events:
        created_ms = _epoch_ms(episode.created_at)
        return created_ms, created_ms
    return _epoch_ms(events[0].timestamp), _epoch_ms(events[-1].timestamp)


def _gateway_final_text(events: list[EvalEvent]) -> str | None:
    final_text: str | None = None
    for event in events:
        if event.kind == "gateway_final":
            final_text = _text_or_json((event.payload or {}).get("text"))
    return final_text


def _tool_call_key(event: EvalEvent, index: int) -> str:
    if event.tool_call_id:
        return event.tool_call_id
    return f"missing-tool-call-id:{index}"


def _model_call_total_tokens(events: list[EvalEvent]) -> int:
    return sum(
        _model_call_tokens(event)
        for event in events
        if event.kind == "model_call"
    )


def _model_call_tokens(event: EvalEvent) -> int:
    payload = event.payload or {}
    return _int_value(payload.get("input_tokens")) + _int_value(payload.get("output_tokens"))


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _int_value(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return _json_dumps(value)


def _text_or_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _model_dump(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json")
