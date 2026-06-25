"""Adapters from ohmo eval episodes to trace-viewer DTOs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from openharness.evals import EvalEpisode, EvalEvent
from openharness.evals.tool_labels import effective_tool_label

TOOL_COMPLETED_KINDS = {"tool_completed", "tool_completed_error"}


def episode_to_trace_viewer_data(store: Any, episode_id: str) -> dict[str, Any]:
    """Return one episode as the JSON-friendly trace viewer DTO."""
    episode = store.get_episode(episode_id)
    if episode is None:
        raise KeyError(episode_id)

    events = list(store.iter_events(episode_id))
    child_spans = _tool_spans(episode_id, events)
    start_ms, end_ms = _trace_bounds_ms(episode, events)
    final_output = _gateway_final_text(events)
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
                "events": [_model_dump(event) for event in events],
            }
        ),
        "attributes": _root_attributes(episode),
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
    input_value = (start_event.payload or {}).get("input")
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
                "started": _model_dump(start_event),
                "completed": _model_dump(completed_event),
            }
        ),
        "attributes": _tool_attributes(tool_call_id, tool_name),
        "children": [],
    }


def _prod_trace_summary(store: Any, episode: EvalEpisode) -> dict[str, Any]:
    events = list(store.iter_events(episode.episode_id))
    tool_spans = _tool_spans(episode.episode_id, events)
    start_ms, end_ms = _trace_bounds_ms(episode, events)
    return {
        "id": episode.episode_id,
        "name": episode.user_text or "(turn)",
        "kind": "prod",
        "createdAt": _epoch_ms(episode.created_at),
        "status": "error" if any(span["status"] == "error" for span in tool_spans) else "success",
        "spansCount": 1 + len(tool_spans),
        "durationMs": max(0, end_ms - start_ms),
    }


def _root_attributes(episode: EvalEpisode) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "model", episode.metadata.get("model"))
    _append_attribute(attributes, "cwd", episode.metadata.get("cwd"))
    return attributes


def _tool_attributes(
    tool_call_id: str | None,
    tool_name: str | None,
) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "tool_call_id", tool_call_id)
    _append_attribute(attributes, "tool_name", tool_name)
    return attributes


def _append_attribute(
    attributes: list[dict[str, dict[str, str] | str]],
    key: str,
    value: Any,
) -> None:
    string_value = _string_value(value)
    if string_value is None:
        return
    attributes.append({"key": key, "value": {"stringValue": string_value}})


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


def _epoch_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


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
