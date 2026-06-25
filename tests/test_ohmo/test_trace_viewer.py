from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from openharness.evals import EvalEpisode, EvalEvent
from ohmo.evals import get_eval_store
from ohmo.evals.viewer import episode_to_trace_viewer_data, list_prod_traces


def test_episode_to_trace_viewer_data_maps_successful_tool_turn(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _append_trace_episode(store, base=base)

    data = episode_to_trace_viewer_data(store, "ep-1")

    assert data["traceRecord"]["id"] == "ep-1"
    assert data["traceRecord"]["spansCount"] == 2
    assert data["traceRecord"]["durationMs"] == 500
    assert data["traceRecord"]["startTimeMs"] == int(base.timestamp() * 1000)

    spans = data["spans"]
    assert len(spans) == 1
    root = spans[0]
    assert root["type"] == "agent_invocation"
    assert root["input"] == "посмотри отзывы"
    assert root["output"] == "вот отзывы"
    assert root["status"] == "success"

    children = root["children"]
    assert len(children) == 1
    child = children[0]
    assert child["title"].startswith("bash:maps-cli")
    assert child["type"] == "tool_execution"
    assert child["status"] == "success"
    assert child["durationMs"] > 0
    assert child["durationMs"] == 200
    assert child["output"] == "..."


def test_episode_to_trace_viewer_data_propagates_tool_error_status(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _append_trace_episode(store, episode_id="ep-error", base=base, is_error=True)

    data = episode_to_trace_viewer_data(store, "ep-error")

    root = data["spans"][0]
    child = root["children"][0]
    assert child["status"] == "error"
    assert root["status"] == "error"


def test_list_prod_traces_returns_prod_items_and_filters_by_query(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _append_trace_episode(store, episode_id="ep-1", base=base, user_text="посмотри отзывы")
    _append_trace_episode(
        store,
        episode_id="ep-2",
        base=base + timedelta(seconds=1),
        user_text="проверь погоду",
    )

    all_items = list_prod_traces(store)
    assert all_items["total"] == 2
    assert all_items["traces"][0]["id"] == "ep-2"
    assert all_items["traces"][0]["kind"] == "prod"

    filtered = list_prod_traces(store, q="ОТЗЫВЫ")
    assert filtered["total"] == 1
    assert filtered["traces"][0]["id"] == "ep-1"
    assert filtered["traces"][0]["kind"] == "prod"


def _append_trace_episode(
    store,
    *,
    episode_id: str = "ep-1",
    base: datetime,
    user_text: str = "посмотри отзывы",
    is_error: bool = False,
) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id="session-1",
            created_at=base,
            user_text=user_text,
            metadata={"model": "gpt-test", "cwd": "/tmp/project"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="inbound_message",
            timestamp=base,
            payload={"user_text": user_text},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_started",
            timestamp=base + timedelta(milliseconds=100),
            payload={"input": {"command": 'maps-cli reviews "Xander"'}},
            tool_name="bash",
            tool_call_id="c1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_completed",
            timestamp=base + timedelta(milliseconds=300),
            payload={"output": "..."},
            tool_name="bash",
            tool_call_id="c1",
            is_error=is_error,
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            timestamp=base + timedelta(milliseconds=500),
            payload={"text": "вот отзывы"},
        )
    )
