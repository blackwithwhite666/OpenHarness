from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openharness.evals import EvalEpisode, EvalEvent, EvalStore


def test_eval_store_creates_hybrid_layout(tmp_path: Path):
    root = tmp_path / "evals"

    store = EvalStore(root)

    assert store.root == root.resolve()
    assert (root / "evals.sqlite").is_file()
    for dirname in (
        "episodes",
        "states",
        "graph",
        "embeddings",
        "candidates",
        "cases",
        "packs",
        "reports",
    ):
        assert (root / dirname).is_dir()


def test_eval_store_appends_jsonl_and_indexes_records(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    episode = EvalEpisode(
        episode_id="ep-1",
        source="gateway",
        app="ohmo",
        session_id="session-1",
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        user_goal="answer a direct message",
        user_text="Can you check this?",
        tags=["dm", "smoke"],
        privacy="redacted",
        status="open",
        metadata={"channel": "telegram"},
    )
    event = EvalEvent(
        episode_id="ep-1",
        kind="tool_call",
        timestamp=datetime(2026, 1, 2, 3, 4, 6, tzinfo=timezone.utc),
        payload={"arguments": {"text": "hello"}},
        tool_name="send_message",
        tool_call_id="toolu-1",
        is_error=False,
    )

    store.append_episode(episode)
    store.append_event(event)

    episodes_path = store.root / "episodes" / "episodes.jsonl"
    events_path = store.root / "episodes" / "events.jsonl"
    episode_rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    event_rows = [json.loads(line) for line in events_path.read_text().splitlines()]

    assert episode_rows == [
        {
            "episode_id": "ep-1",
            "source": "gateway",
            "app": "ohmo",
            "session_id": "session-1",
            "created_at": "2026-01-02T03:04:05Z",
            "user_goal": "answer a direct message",
            "user_text": "Can you check this?",
            "tags": ["dm", "smoke"],
            "privacy": "redacted",
            "status": "open",
            "metadata": {"channel": "telegram"},
        }
    ]
    assert event_rows == [
        {
            "episode_id": "ep-1",
            "kind": "tool_call",
            "timestamp": "2026-01-02T03:04:06Z",
            "payload": {"arguments": {"text": "hello"}},
            "tool_name": "send_message",
            "tool_call_id": "toolu-1",
            "is_error": False,
        }
    ]

    assert store.count_episodes() == 1
    assert store.count_events() == 1
    assert store.count_events("ep-1") == 1
    assert store.list_episode_ids() == ["ep-1"]
    assert store.get_episode("ep-1") == episode
    assert list(store.iter_events("ep-1")) == [event]

    with sqlite3.connect(store.database_path) as connection:
        episode_count = connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        event_row = connection.execute(
            """
            SELECT kind, tool_name, tool_call_id, is_error
            FROM events
            WHERE episode_id = ?
            """,
            ("ep-1",),
        ).fetchone()
    assert episode_count == 1
    assert event_row == ("tool_call", "send_message", "toolu-1", 0)


def test_eval_store_rejects_event_without_episode(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")

    with pytest.raises(ValueError, match="episode does not exist"):
        store.append_event(EvalEvent(episode_id="missing", kind="message"))

    assert store.count_events() == 0
    assert not (store.root / "episodes" / "events.jsonl").exists()
