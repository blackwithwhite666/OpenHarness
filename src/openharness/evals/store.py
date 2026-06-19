"""Filesystem and SQLite storage for eval/flywheel data."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Iterator, TypeVar

from pydantic import BaseModel, ValidationError

from openharness.evals.models import (
    EvalEmbeddingManifest,
    EvalEmbeddingRecord,
    EvalEpisode,
    EvalEvent,
)


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
    EMBEDDING_MANIFEST_JSON = "embedding_manifest.json"
    EMBEDDING_RECORDS_JSONL = "embedding_records.jsonl"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.database_path = self.root / "evals.sqlite"
        self.episodes_dir = self.root / "episodes"
        self.embeddings_dir = self.root / "embeddings"
        self._ensure_layout()

    def append_episode(self, episode: EvalEpisode) -> None:
        """Append one episode metadata record and index it by episode id."""
        if self.get_episode(episode.episode_id) is not None:
            raise ValueError(f"episode already exists: {episode.episode_id}")

        jsonl_path = self.episodes_dir / self.EPISODES_JSONL
        offset = self._append_jsonl(jsonl_path, episode)

        connection = self._connect()
        try:
            with connection:
                self._insert_episode_index(
                    connection,
                    episode,
                    jsonl_path=self._relative_jsonl_path(jsonl_path),
                    jsonl_offset=offset,
                )
        finally:
            connection.close()

    def append_event(self, event: EvalEvent) -> None:
        """Append one episode event and index it for replay lookup."""
        if self.get_episode(event.episode_id) is None:
            raise ValueError(f"episode does not exist: {event.episode_id}")

        jsonl_path = self.episodes_dir / self.EVENTS_JSONL
        offset = self._append_jsonl(jsonl_path, event)

        connection = self._connect()
        try:
            with connection:
                self._insert_event_index(
                    connection,
                    event,
                    jsonl_path=self._relative_jsonl_path(jsonl_path),
                    jsonl_offset=offset,
                )
        finally:
            connection.close()

    def get_episode(self, episode_id: str) -> EvalEpisode | None:
        """Return an episode record by id, or None when it has not been indexed."""
        self._ensure_lookup_index()
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
        self._ensure_lookup_index()
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
        self._ensure_lookup_index()
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
        self._ensure_lookup_index()
        connection = self._connect()
        try:
            row = connection.execute("SELECT COUNT(*) AS count FROM episodes").fetchone()
        finally:
            connection.close()
        return int(row["count"])

    def count_events(self, episode_id: str | None = None) -> int:
        """Return the number of indexed event records, optionally scoped to one episode."""
        self._ensure_lookup_index()
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
        self._ensure_lookup_index()
        connection = self._connect()
        try:
            with connection:
                connection.execute("DELETE FROM embedding_records")
                for record in records:
                    self._insert_embedding_record_index(
                        connection,
                        record,
                        jsonl_path=jsonl_path,
                    )
        finally:
            connection.close()

    def count_embedding_records(self) -> int:
        """Return the number of dense embedding records indexed in SQLite."""
        self._ensure_lookup_index()
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM embedding_records"
            ).fetchone()
        finally:
            connection.close()
        return int(row["count"])

    def list_embedding_facet_ids(self) -> set[str]:
        """Return facet ids that have dense embedding lookup rows."""
        self._ensure_lookup_index()
        connection = self._connect()
        try:
            rows = connection.execute("SELECT facet_id FROM embedding_records").fetchall()
        finally:
            connection.close()
        return {str(row["facet_id"]) for row in rows}

    def _ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for dirname in self.LAYOUT_DIRS:
            (self.root / dirname).mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._rehydrate_empty_lookup_tables()

    def _ensure_lookup_index(self) -> None:
        if not self.database_path.exists() or self.database_path.stat().st_size == 0:
            self._init_database()
        self._rehydrate_empty_lookup_tables()

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

    def _rehydrate_empty_lookup_tables(self) -> None:
        connection = self._connect()
        try:
            with connection:
                if connection.execute("SELECT 1 FROM episodes LIMIT 1").fetchone() is None:
                    self._rehydrate_episode_index(connection)
                if connection.execute("SELECT 1 FROM events LIMIT 1").fetchone() is None:
                    self._rehydrate_event_index(connection)
                if (
                    connection.execute("SELECT 1 FROM embedding_records LIMIT 1").fetchone()
                    is None
                ):
                    self._rehydrate_embedding_index(connection)
        finally:
            connection.close()

    def _rehydrate_episode_index(self, connection: sqlite3.Connection) -> None:
        jsonl_path = self.episodes_dir / self.EPISODES_JSONL
        relative_path = self._relative_jsonl_path(jsonl_path)
        for offset, episode in self._iter_jsonl_model_offsets(EvalEpisode, jsonl_path):
            self._insert_episode_index(
                connection,
                episode,
                jsonl_path=relative_path,
                jsonl_offset=offset,
            )

    def _rehydrate_event_index(self, connection: sqlite3.Connection) -> None:
        jsonl_path = self.episodes_dir / self.EVENTS_JSONL
        relative_path = self._relative_jsonl_path(jsonl_path)
        for offset, event in self._iter_jsonl_model_offsets(EvalEvent, jsonl_path):
            self._insert_event_index(
                connection,
                event,
                jsonl_path=relative_path,
                jsonl_offset=offset,
            )

    def _rehydrate_embedding_index(self, connection: sqlite3.Connection) -> None:
        relative_path = self._embedding_records_relative_path()
        if relative_path is None:
            return

        records_path = self._resolve_store_relative_path(relative_path)
        if not records_path.exists():
            raise FileNotFoundError(f"embedding records artifact not found: {relative_path}")
        for _, record in self._iter_jsonl_model_offsets(EvalEmbeddingRecord, records_path):
            self._insert_embedding_record_index(connection, record, jsonl_path=relative_path)

    def _embedding_records_relative_path(self) -> str | None:
        manifest_path = self.embeddings_dir / self.EMBEDDING_MANIFEST_JSON
        if manifest_path.exists():
            try:
                manifest = EvalEmbeddingManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            except ValidationError as exc:
                raise ValueError(
                    f"invalid {self._relative_jsonl_path(manifest_path)}"
                ) from exc
            return manifest.records_path

        records_path = self.embeddings_dir / self.EMBEDDING_RECORDS_JSONL
        if records_path.exists():
            return self._relative_jsonl_path(records_path)
        return None

    def _insert_episode_index(
        self,
        connection: sqlite3.Connection,
        episode: EvalEpisode,
        *,
        jsonl_path: str,
        jsonl_offset: int,
    ) -> None:
        record = episode.model_dump(mode="json")
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
                jsonl_path,
                jsonl_offset,
            ),
        )

    def _insert_event_index(
        self,
        connection: sqlite3.Connection,
        event: EvalEvent,
        *,
        jsonl_path: str,
        jsonl_offset: int,
    ) -> None:
        record = event.model_dump(mode="json")
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
                jsonl_path,
                jsonl_offset,
            ),
        )

    def _insert_embedding_record_index(
        self,
        connection: sqlite3.Connection,
        record: EvalEmbeddingRecord,
        *,
        jsonl_path: str,
    ) -> None:
        connection.execute(
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
            ),
        )

    def _iter_jsonl_model_offsets(
        self,
        model_type: type[ModelT],
        path: Path,
    ) -> Iterator[tuple[int, ModelT]]:
        if not path.exists():
            return

        relative_path = self._relative_jsonl_path(path)
        with path.open("r", encoding="utf-8") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                try:
                    yield offset, model_type.model_validate_json(line)
                except ValidationError as exc:
                    raise ValueError(f"invalid {relative_path} row {line_number}") from exc

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
        path = self._resolve_store_relative_path(relative_path)
        with path.open("r", encoding="utf-8") as handle:
            handle.seek(offset)
            line = handle.readline()
        return model_type.model_validate_json(line)

    def _resolve_store_relative_path(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"artifact path must stay under eval store: {relative_path}") from exc
        return path

    def _relative_jsonl_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()
