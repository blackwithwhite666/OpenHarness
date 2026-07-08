from __future__ import annotations

from pathlib import Path

from openharness.evals import (
    EvalEpisode,
    EvalStore,
    segment_sessions_into_conversations,
)


def test_segment_sessions_into_conversations_splits_on_gap(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    for episode_id, created_at in (
        ("ep-1", "2026-01-01T12:00:00Z"),
        ("ep-2", "2026-01-01T12:03:00Z"),
        ("ep-3", "2026-01-01T12:06:00Z"),
        ("ep-4", "2026-01-01T12:45:00Z"),
        ("ep-5", "2026-01-01T12:49:00Z"),
    ):
        _append_episode(
            store,
            episode_id=episode_id,
            session_id="session-1",
            created_at=created_at,
        )

    conversations = segment_sessions_into_conversations(store, gap_minutes=30)

    assert [conversation.episode_ids for conversation in conversations] == [
        ("ep-1", "ep-2", "ep-3"),
        ("ep-4", "ep-5"),
    ]


def test_segment_sessions_into_conversations_min_turns_drops_singletons(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    for episode_id, created_at in (
        ("ep-1", "2026-01-01T12:00:00Z"),
        ("ep-2", "2026-01-01T12:02:00Z"),
        ("ep-3", "2026-01-01T12:04:00Z"),
        ("ep-4", "2026-01-01T12:40:00Z"),
    ):
        _append_episode(
            store,
            episode_id=episode_id,
            session_id="session-1",
            created_at=created_at,
        )

    conversations = segment_sessions_into_conversations(
        store,
        gap_minutes=30,
        min_turns=2,
    )

    assert [conversation.episode_ids for conversation in conversations] == [
        ("ep-1", "ep-2", "ep-3"),
    ]


def test_segment_sessions_into_conversations_keeps_session_ids_separate(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    for episode_id, session_id, created_at in (
        ("a-1", "session-a", "2026-01-01T12:00:00Z"),
        ("b-1", "session-b", "2026-01-01T12:01:00Z"),
        ("a-2", "session-a", "2026-01-01T12:02:00Z"),
        ("b-2", "session-b", "2026-01-01T12:03:00Z"),
    ):
        _append_episode(
            store,
            episode_id=episode_id,
            session_id=session_id,
            created_at=created_at,
        )

    conversations = segment_sessions_into_conversations(store, gap_minutes=30)

    assert {
        conversation.session_id: conversation.episode_ids
        for conversation in conversations
    } == {
        "session-a": ("a-1", "a-2"),
        "session-b": ("b-1", "b-2"),
    }


def _append_episode(
    store: EvalStore,
    *,
    episode_id: str,
    session_id: str,
    created_at: str,
) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id=session_id,
            created_at=created_at,
            user_text=f"private request {episode_id}",
        )
    )
