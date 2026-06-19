"""Filesystem and SQLite storage for eval/flywheel data."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Iterator, TypeVar

from pydantic import BaseModel

from openharness.evals.models import EvalEmbeddingRecord, EvalEpisode, EvalEvent


ModelT = TypeVar("ModelT", bound=BaseModel)


class EvalStore:
    """Append JSONL replay data and maintain a small SQLite lookup index."""

    LAYOUT_DIRS = (
        "episodes",
        "states",
        "graph",
        "embeddings",
        "candidates",
        "cases",
        "packs",
        "reports",
    )
    EPISODES_JSONL = "episodes.jsonl"
    EVENTS_JSONL = "events.jsonl"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.database_path = self.root / "evals.sqlite"
        self.episodes_dir = self.root / "episodes"
        self._ensure_layout()

    def append_episode(self, episode: EvalEpisode) -> None:
        """Append one episode metadata record and index it by episode id."""
        if self.get_episode(episode.episode_id) is not None:
            raise ValueError(f"episode already exists: {episode.episode_id}")

        jsonl_path = self.episodes_dir / self.EPISODES_JSONL
        offset = self._append_jsonl(jsonl_path, episode)
        record = episode.model_dump(mode="json")

        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO episodes (
                        episode_id,
                        source,
                        app,
                        session_id,
                        created_at,
                        privacy,
                        status,
                        jsonl_path,
                        jsonl_offset
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode.episode_id,
                        episode.source,
                        episode.app,
                        episode.session_id,
                        record["created_at"],
                        episode.privacy,
                        episode.status,
                        self._relative_jsonl_path(jsonl_path),
                        offset,
                    ),
                )
        finally:
            connection.close()

    def append_event(self, event: EvalEvent) -> None:
        """Append one episode event and index it for replay lookup."""
        if self.get_episode(event.episode_id) is None:
            raise ValueError(f"episode does not exist: {event.episode_id}")

        jsonl_path = self.episodes_dir / self.EVENTS_JSONL
        offset = self._append_jsonl(jsonl_path, event)
        record = event.model_dump(mode="json")

        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    INSERT INTO events (
                        episode_id,
                        kind,
                        timestamp,
                        tool_name,
                        tool_call_id,
                        is_error,
                        jsonl_path,
                        jsonl_offset
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.episode_id,
                        event.kind,
                        record["timestamp"],
                        event.tool_name,
                        event.tool_call_id,
                        int(event.is_error),
                        self._relative_jsonl_path(jsonl_path),
                        offset,
                    ),
                )
        finally:
            connection.close()

    def get_episode(self, episode_id: str) -> EvalEpisode | None:
        """Return an episode record by id, or None when it has not been indexed."""
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT jsonl_path, jsonl_offset
                FROM episodes
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return self._read_jsonl_model(EvalEpisode, row["jsonl_path"], row["jsonl_offset"])

    def iter_events(self, episode_id: str) -> Iterator[EvalEvent]:
        """Yield indexed events for an episode in append order."""
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT jsonl_path, jsonl_offset
                FROM events
                WHERE episode_id = ?
                ORDER BY id
                """,
                (episode_id,),
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            yield self._read_jsonl_model(EvalEvent, row["jsonl_path"], row["jsonl_offset"])

    def list_episode_ids(self) -> list[str]:
        """List indexed episode ids in creation order."""
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT episode_id
                FROM episodes
                ORDER BY created_at, episode_id
                """
            ).fetchall()
        finally:
            connection.close()
        return [row["episode_id"] for row in rows]

    def count_episodes(self) -> int:
        """Return the number of indexed episode records."""
        connection = self._connect()
        try:
            row = connection.execute("SELECT COUNT(*) AS count FROM episodes").fetchone()
        finally:
            connection.close()
        return int(row["count"])

    def count_events(self, episode_id: str | None = None) -> int:
        """Return the number of indexed event records, optionally scoped to one episode."""
        connection = self._connect()
        try:
            if episode_id is None:
                row = connection.execute("SELECT COUNT(*) AS count FROM events").fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM events WHERE episode_id = ?",
                    (episode_id,),
                ).fetchone()
        finally:
            connection.close()
        return int(row["count"])

    def replace_embedding_index(
        self,
        records: Sequence[EvalEmbeddingRecord],
        *,
        jsonl_path: str,
    ) -> None:
        """Replace the SQLite lookup rows for a generated embedding JSONL file."""
        connection = self._connect()
        try:
            with connection:
                connection.execute("DELETE FROM embedding_records")
                connection.executemany(
                    """
                    INSERT INTO embedding_records (
                        facet_id,
                        episode_id,
                        facet_kind,
                        source_path,
                        text_hash,
                        text_length,
                        model,
                        dimensions,
                        jsonl_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            record.facet.facet_id,
                            record.facet.episode_id,
                            record.facet.facet_kind,
                            record.facet.source_path,
                            record.facet.text_hash,
                            record.facet.text_length,
                            record.model,
                            record.dimensions,
                            jsonl_path,
                        )
                        for record in records
                    ],
                )
        finally:
            connection.close()

    def count_embedding_records(self) -> int:
        """Return the number of dense embedding records indexed in SQLite."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM embedding_records"
            ).fetchone()
        finally:
            connection.close()
        return int(row["count"])

    def _ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for dirname in self.LAYOUT_DIRS:
            (self.root / dirname).mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS episodes (
                        episode_id TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        app TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        privacy TEXT NOT NULL,
                        status TEXT NOT NULL,
                        jsonl_path TEXT NOT NULL,
                        jsonl_offset INTEGER NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_episodes_session
                    ON episodes (session_id);

                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        episode_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        tool_name TEXT,
                        tool_call_id TEXT,
                        is_error INTEGER NOT NULL,
                        jsonl_path TEXT NOT NULL,
                        jsonl_offset INTEGER NOT NULL,
                        FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_episode
                    ON events (episode_id, id);

                    CREATE INDEX IF NOT EXISTS idx_events_kind
                    ON events (kind);

                    CREATE TABLE IF NOT EXISTS embedding_records (
                        facet_id TEXT PRIMARY KEY,
                        episode_id TEXT NOT NULL,
                        facet_kind TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        text_hash TEXT NOT NULL,
                        text_length INTEGER NOT NULL,
                        model TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        jsonl_path TEXT NOT NULL,
                        FOREIGN KEY (episode_id) REFERENCES episodes (episode_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_embedding_records_episode
                    ON embedding_records (episode_id);

                    CREATE INDEX IF NOT EXISTS idx_embedding_records_kind
                    ON embedding_records (facet_kind);
                    """
                )
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _append_jsonl(self, path: Path, model: BaseModel) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as handle:
            handle.seek(0, 2)
            offset = handle.tell()
            handle.write(model.model_dump_json())
            handle.write("\n")
        return offset

    def _read_jsonl_model(
        self,
        model_type: type[ModelT],
        relative_path: str,
        offset: int,
    ) -> ModelT:
        path = self.root / relative_path
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            line = handle.readline()
        return model_type.model_validate_json(line)

    def _relative_jsonl_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()
