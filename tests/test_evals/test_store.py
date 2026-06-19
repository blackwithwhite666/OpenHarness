from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openharness.evals import (
    EvalEmbeddingManifest,
    EvalEmbeddingRecord,
    EvalEpisode,
    EvalEvent,
    EvalStore,
    EvalTextFacet,
)


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


def test_eval_store_rehydrates_missing_sqlite_from_artifacts(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    episode = _episode()
    event = _event()
    embedding = _embedding_record()

    store.append_episode(episode)
    store.append_event(event)
    embedding_jsonl_path = _write_embedding_artifacts(store, embedding)
    store.replace_embedding_index([embedding], jsonl_path=embedding_jsonl_path)

    store.database_path.unlink()

    assert store.list_episode_ids() == ["ep-1"]
    assert store.count_embedding_records() == 1
    assert store.list_embedding_facet_ids() == {"facet-1"}

    rehydrated = EvalStore(store.root)

    assert rehydrated.count_episodes() == 1
    assert rehydrated.count_events() == 1
    assert rehydrated.count_events("ep-1") == 1
    assert rehydrated.list_episode_ids() == ["ep-1"]
    assert rehydrated.get_episode("ep-1") == episode
    assert list(rehydrated.iter_events("ep-1")) == [event]
    assert rehydrated.count_embedding_records() == 1
    assert rehydrated.list_embedding_facet_ids() == {"facet-1"}

    with sqlite3.connect(rehydrated.database_path) as connection:
        embedding_row = connection.execute(
            """
            SELECT facet_id, episode_id, facet_kind, source_path, jsonl_path
            FROM embedding_records
            """
        ).fetchone()

    assert embedding_row == (
        "facet-1",
        "ep-1",
        "user_request",
        "episode.user_text",
        "embeddings/embedding_records.jsonl",
    )


def test_eval_store_rehydrates_empty_sqlite_file_from_jsonl(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    episode = _episode()
    event = _event()

    store.append_episode(episode)
    store.append_event(event)
    store.database_path.write_bytes(b"")

    rehydrated = EvalStore(store.root)

    assert rehydrated.list_episode_ids() == ["ep-1"]
    assert rehydrated.get_episode("ep-1") == episode
    assert list(rehydrated.iter_events("ep-1")) == [event]


def _episode() -> EvalEpisode:
    return EvalEpisode(
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


def _event() -> EvalEvent:
    return EvalEvent(
        episode_id="ep-1",
        kind="tool_call",
        timestamp=datetime(2026, 1, 2, 3, 4, 6, tzinfo=timezone.utc),
        payload={"arguments": {"text": "hello"}},
        tool_name="send_message",
        tool_call_id="toolu-1",
        is_error=False,
    )


def _embedding_record() -> EvalEmbeddingRecord:
    return EvalEmbeddingRecord(
        facet=EvalTextFacet(
            facet_id="facet-1",
            episode_id="ep-1",
            facet_kind="user_request",
            source_path="episode.user_text",
            text_hash="hash-1",
            text_length=19,
        ),
        model="BAAI/bge-m3",
        dimensions=2,
        vector=[0.1, 0.2],
        created_at=datetime(2026, 1, 2, 3, 4, 7, tzinfo=timezone.utc),
    )


def _write_embedding_artifacts(store: EvalStore, record: EvalEmbeddingRecord) -> str:
    records_relative_path = "embeddings/embedding_records.jsonl"
    records_path = store.root / records_relative_path
    records_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    manifest = EvalEmbeddingManifest(
        model=record.model,
        dimensions=record.dimensions,
        records_path=records_relative_path,
        facet_count=1,
        embedding_count=1,
    )
    (store.root / "embeddings" / "embedding_manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return records_relative_path
