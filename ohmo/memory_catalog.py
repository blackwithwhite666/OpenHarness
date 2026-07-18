"""Transactional SQLite catalog for ohmo personal memory.

The catalog is a workspace-scoped authority that lives alongside the legacy
Markdown store during migration.  Every mutation is serialized with
``BEGIN IMMEDIATE``; validation, deduplication, and budget checks therefore see
the same database state that is committed by the mutation.
"""

from __future__ import annotations

import builtins
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, cast

from ohmo.memory_store import (
    DEFAULT_ENTRY_CHAR_LIMIT,
    DEFAULT_STORE_CHAR_BUDGET,
    MemoryOpResult,
    slugify,
)
from ohmo.threat_patterns import first_threat_message
from ohmo.workspace import get_memory_dir

_SCHEMA_VERSION = 1
_MAX_TITLE_CHARS = 256
_BUSY_TIMEOUT_MS = 5_000
_RESERVED_NAMES = {"memory.md"}
_SOURCES = {"curated", "derived"}
_ARCHIVE_STATUSES = {"active", "archived"}


@dataclass(frozen=True)
class CatalogRecord:
    """One complete row from the memory catalog."""

    slug: str
    title: str
    content: str
    size: int
    usage: int
    pinned: int
    generation: int
    source: str
    archive_status: str
    honcho_conclusion_ids: str
    outbox_state: str | None
    created_at: str
    updated_at: str


class MemoryCatalog:
    """Workspace-scoped transactional SQLite memory catalog."""

    def __init__(
        self,
        workspace: str | Path | None = None,
        *,
        db_path: str | Path | None = None,
        entry_char_limit: int | None = None,
        store_char_budget: int | None = None,
    ) -> None:
        self._memory_dir = get_memory_dir(workspace)
        self._db_path = (
            Path(db_path).expanduser()
            if db_path is not None
            else self._memory_dir / "catalog.sqlite3"
        )
        self._entry_char_limit = (
            entry_char_limit
            if entry_char_limit is not None
            else _env_int("OHMO_MEMORY_ENTRY_CHARS", DEFAULT_ENTRY_CHAR_LIMIT)
        )
        self._store_char_budget = (
            store_char_budget
            if store_char_budget is not None
            else _env_int("OHMO_MEMORY_STORE_CHARS", DEFAULT_STORE_CHAR_BUDGET)
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @property
    def db_path(self) -> Path:
        """The SQLite file backing this catalog."""
        return self._db_path

    # -- connection and schema ---------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._db_path,
            timeout=_BUSY_TIMEOUT_MS / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _write_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        # The write lock covers the version check and all migration DDL, so two
        # processes constructing a catalog cannot both apply a migration.
        with self._write_connection() as connection:
            version = cast(int, connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _SCHEMA_VERSION:
                raise RuntimeError(
                    f"catalog schema version {version} is newer than supported "
                    f"version {_SCHEMA_VERSION}"
                )
            if version < 1:
                self._migrate_to_v1(connection)
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")

    @staticmethod
    def _migrate_to_v1(connection: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE memories (
                slug TEXT PRIMARY KEY,
                title TEXT,
                content TEXT NOT NULL,
                size INTEGER NOT NULL CHECK (size = length(content)),
                usage INTEGER NOT NULL DEFAULT 0 CHECK (usage >= 0),
                pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1)),
                generation INTEGER NOT NULL DEFAULT 1 CHECK (generation >= 1),
                source TEXT NOT NULL DEFAULT 'curated'
                    CHECK (source IN ('curated', 'derived')),
                archive_status TEXT NOT NULL DEFAULT 'active'
                    CHECK (archive_status IN ('active', 'archived')),
                honcho_conclusion_ids TEXT DEFAULT '[]',
                outbox_state TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                slug UNINDEXED,
                title,
                content
            )
            """,
            """
            CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, slug, title, content)
                VALUES (new.rowid, new.slug, new.title, new.content);
            END
            """,
            """
            CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE rowid = old.rowid;
            END
            """,
            """
            CREATE TRIGGER memories_fts_update
            AFTER UPDATE OF title, content ON memories BEGIN
                DELETE FROM memories_fts WHERE rowid = old.rowid;
                INSERT INTO memories_fts(rowid, slug, title, content)
                VALUES (new.rowid, new.slug, new.title, new.content);
            END
            """,
        )
        for statement in statements:
            connection.execute(statement)

    # -- writes -------------------------------------------------------------
    def add(self, title: str, content: str, *, source: str = "curated") -> MemoryOpResult:
        clean_title = (title or "").strip()
        clean_content = (content or "").strip()

        with self._write_connection() as connection:
            if not clean_title:
                return MemoryOpResult(False, "A title is required.")
            if not clean_content:
                return MemoryOpResult(False, "Content cannot be empty.")
            if len(clean_title) > _MAX_TITLE_CHARS:
                return MemoryOpResult(
                    False,
                    f"Title is too long ({len(clean_title)} chars; max {_MAX_TITLE_CHARS}).",
                )
            if len(clean_content) > self._entry_char_limit:
                return MemoryOpResult(
                    False,
                    f"Entry is {len(clean_content):,} chars, over the "
                    f"{self._entry_char_limit:,}-char per-entry limit. "
                    "Split it into focused entries or shorten it.",
                )
            if source not in _SOURCES:
                return MemoryOpResult(False, f"Invalid memory source {source!r}.")

            threat = first_threat_message(f"{clean_title}\n{clean_content}", scope="strict")
            if threat:
                return MemoryOpResult(False, threat)

            slug = slugify(clean_title)
            name = f"{slug}.md"
            if name.lower() in _RESERVED_NAMES:
                return MemoryOpResult(
                    False,
                    "That title is reserved for the memory index — choose a more specific title.",
                )

            duplicate = connection.execute(
                "SELECT slug FROM memories WHERE content = ? ORDER BY slug LIMIT 1",
                (clean_content,),
            ).fetchone()
            if duplicate is not None:
                return MemoryOpResult(
                    True,
                    f"Already remembered (matches {duplicate['slug']}.md); nothing added.",
                )

            collision = connection.execute(
                "SELECT 1 FROM memories WHERE slug = ?",
                (slug,),
            ).fetchone()
            if collision is not None:
                return MemoryOpResult(
                    False,
                    f"An entry {name!r} already exists with different content. "
                    f"Use action='update' (name={slug!r}) to change it, or choose a more "
                    "specific title so it gets its own file.",
                )

            current_total = self._total_chars(connection, active_only=True)
            new_total = current_total + len(clean_content)
            if new_total > self._store_char_budget:
                existing = self._list_records(connection, include_archived=False)
                return MemoryOpResult(
                    False,
                    f"Memory at {current_total:,}/{self._store_char_budget:,} chars. "
                    f"Adding {clean_title!r} ({len(clean_content):,} chars) would exceed "
                    "the budget. Consolidate now — use action='update' to merge overlapping "
                    "entries into shorter ones, or action='remove' to drop stale/less-important "
                    "ones (see entries below), then retry this add — all in this turn.",
                    entries=cast(Any, tuple(existing)),
                )

            timestamp = _utc_timestamp()
            connection.execute(
                """
                INSERT INTO memories (
                    slug, title, content, size, source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    slug,
                    clean_title,
                    clean_content,
                    len(clean_content),
                    source,
                    timestamp,
                    timestamp,
                ),
            )
            return MemoryOpResult(True, f"Saved memory {name}.")

    def import_entry(
        self,
        slug: str,
        title: str,
        content: str,
        *,
        source: str = "curated",
        archive_status: str = "active",
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> MemoryOpResult:
        """Import one trusted legacy entry without applying model-write limits.

        This low-level migration primitive deliberately bypasses threat scanning,
        per-entry limits, deduplication by content, and the active-store budget.
        Callers must threat-scan untrusted content before invoking it. A repeated
        import is a no-op only when the existing row at ``slug`` has identical
        content; conflicting content is never overwritten.
        """
        clean_slug = _slug_reference(slug)

        with self._write_connection() as connection:
            if not clean_slug:
                return MemoryOpResult(False, "An import slug is required.")
            if source not in _SOURCES:
                return MemoryOpResult(False, f"Invalid memory source {source!r}.")
            if archive_status not in _ARCHIVE_STATUSES:
                return MemoryOpResult(
                    False,
                    f"Invalid memory archive status {archive_status!r}.",
                )

            existing = connection.execute(
                "SELECT content FROM memories WHERE slug = ?",
                (clean_slug,),
            ).fetchone()
            if existing is not None:
                if existing["content"] == content:
                    return MemoryOpResult(
                        True,
                        f"Already imported {clean_slug}.md; nothing changed.",
                    )
                return MemoryOpResult(
                    False,
                    f"Import conflict for {clean_slug}.md: existing content differs; "
                    "nothing changed.",
                )

            timestamp = _utc_timestamp()
            effective_created_at = timestamp if created_at is None else created_at
            effective_updated_at = timestamp if updated_at is None else updated_at
            connection.execute(
                """
                INSERT INTO memories (
                    slug, title, content, size, source, archive_status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_slug,
                    title,
                    content,
                    len(content),
                    source,
                    archive_status,
                    effective_created_at,
                    effective_updated_at,
                ),
            )
            return MemoryOpResult(True, f"Imported memory {clean_slug}.md.")

    def update(self, slug: str, content: str, *, title: str | None = None) -> MemoryOpResult:
        clean_slug = _slug_reference(slug)
        clean_content = (content or "").strip()
        clean_title = (title or "").strip() if title is not None else None

        with self._write_connection() as connection:
            if not clean_content:
                return MemoryOpResult(False, "Content cannot be empty.")
            if clean_title and len(clean_title) > _MAX_TITLE_CHARS:
                return MemoryOpResult(
                    False,
                    f"Title is too long ({len(clean_title)} chars; max {_MAX_TITLE_CHARS}).",
                )
            if len(clean_content) > self._entry_char_limit:
                return MemoryOpResult(
                    False,
                    f"Entry is {len(clean_content):,} chars, over the "
                    f"{self._entry_char_limit:,}-char per-entry limit. "
                    "Shorten it or split into focused entries.",
                )

            threat = first_threat_message(f"{clean_title or ''}\n{clean_content}", scope="strict")
            if threat:
                return MemoryOpResult(False, threat)

            row = connection.execute(
                "SELECT * FROM memories WHERE slug = ?",
                (clean_slug,),
            ).fetchone()
            if row is None:
                return MemoryOpResult(
                    False,
                    f"No memory entry {slug!r}. Use action='add' to create it.",
                )

            new_total = self._total_chars(connection, active_only=True)
            if row["archive_status"] == "active":
                new_total = new_total - cast(int, row["size"]) + len(clean_content)
            if new_total > self._store_char_budget:
                existing = self._list_records(connection, include_archived=False)
                return MemoryOpResult(
                    False,
                    f"Memory would be {new_total:,}/{self._store_char_budget:,} chars after "
                    "this update. Trim this entry or remove stale ones first (see entries "
                    "below), then retry — this turn.",
                    entries=cast(Any, tuple(existing)),
                )

            effective_title = clean_title or cast(str, row["title"])
            connection.execute(
                """
                UPDATE memories
                SET title = ?, content = ?, size = ?, generation = generation + 1,
                    updated_at = ?
                WHERE slug = ?
                """,
                (effective_title, clean_content, len(clean_content), _utc_timestamp(), clean_slug),
            )
            return MemoryOpResult(True, f"Updated memory {clean_slug}.md.")

    def remove(self, slug: str) -> MemoryOpResult:
        clean_slug = _slug_reference(slug)
        with self._write_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE memories
                SET archive_status = 'archived', updated_at = ?
                WHERE slug = ? AND archive_status = 'active'
                """,
                (_utc_timestamp(), clean_slug),
            )
            if cursor.rowcount == 0:
                return MemoryOpResult(False, f"No memory entry {slug!r}.")
            return MemoryOpResult(True, f"Archived memory {clean_slug}.md.")

    def record_use(self, slug: str) -> None:
        clean_slug = _slug_reference(slug)
        with self._write_connection() as connection:
            connection.execute(
                """
                UPDATE memories
                SET usage = usage + 1, updated_at = ?
                WHERE slug = ?
                """,
                (_utc_timestamp(), clean_slug),
            )

    # -- reads --------------------------------------------------------------
    def get(self, slug: str) -> CatalogRecord | None:
        clean_slug = _slug_reference(slug)
        with self._read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM memories WHERE slug = ?",
                (clean_slug,),
            ).fetchone()
        return _record_from_row(row) if row is not None else None

    def list(self, *, include_archived: bool = False) -> list[CatalogRecord]:
        with self._read_connection() as connection:
            return self._list_records(connection, include_archived=include_archived)

    @staticmethod
    def _list_records(
        connection: sqlite3.Connection,
        *,
        include_archived: bool,
    ) -> builtins.list[CatalogRecord]:
        where = "" if include_archived else "WHERE archive_status = 'active'"
        rows = connection.execute(
            f"""
            SELECT * FROM memories
            {where}
            ORDER BY pinned DESC, usage DESC, slug ASC
            """
        ).fetchall()
        return [_record_from_row(row) for row in rows]

    def total_chars(self, *, active_only: bool = True) -> int:
        with self._read_connection() as connection:
            return self._total_chars(connection, active_only=active_only)

    @staticmethod
    def _total_chars(connection: sqlite3.Connection, *, active_only: bool) -> int:
        where = "WHERE archive_status = 'active'" if active_only else ""
        row = connection.execute(
            f"SELECT COALESCE(SUM(size), 0) AS total FROM memories {where}"
        ).fetchone()
        return cast(int, row["total"])

    def search(self, query: str, top_k: int) -> builtins.list[CatalogRecord]:
        """Return FTS5 hits across active and archived rows.

        FTS5's lower-is-better BM25 score (title weight 2, content weight 1) is
        blended with a bounded usage boost of ``0.25 * usage / (usage + 4)``.
        The bounded boost lets frequently injected entries win close matches
        without allowing an unbounded counter to erase textual relevance.
        Slug is the final deterministic tie-breaker.
        """
        clean_query = (query or "").strip()
        if not clean_query or top_k <= 0:
            return []
        statement = """
            SELECT m.*,
                   bm25(memories_fts, 0.0, 2.0, 1.0)
                       - (0.25 * CAST(m.usage AS REAL) / (m.usage + 4.0))
                       AS blended_score
            FROM memories_fts
            JOIN memories AS m ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ?
            ORDER BY blended_score ASC, m.usage DESC, m.slug ASC
            LIMIT ?
        """
        with self._read_connection() as connection:
            try:
                rows = connection.execute(statement, (clean_query, top_k)).fetchall()
            except sqlite3.OperationalError as error:
                if "fts5: syntax error" not in str(error).lower():
                    raise
                # Natural-language callers need not understand FTS5 quoting.
                # Preserve valid advanced MATCH expressions, but retry malformed
                # punctuation as an AND of literal Unicode word tokens.
                literal_query = _literal_fts_query(clean_query)
                if not literal_query:
                    return []
                rows = connection.execute(statement, (literal_query, top_k)).fetchall()
        return [_record_from_row(row) for row in rows]


def _record_from_row(row: sqlite3.Row) -> CatalogRecord:
    return CatalogRecord(
        slug=cast(str, row["slug"]),
        title=cast(str, row["title"]),
        content=cast(str, row["content"]),
        size=cast(int, row["size"]),
        usage=cast(int, row["usage"]),
        pinned=cast(int, row["pinned"]),
        generation=cast(int, row["generation"]),
        source=cast(str, row["source"]),
        archive_status=cast(str, row["archive_status"]),
        honcho_conclusion_ids=cast(str, row["honcho_conclusion_ids"]),
        outbox_state=cast(str | None, row["outbox_state"]),
        created_at=cast(str, row["created_at"]),
        updated_at=cast(str, row["updated_at"]),
    )


def _slug_reference(slug: str) -> str:
    value = (slug or "").strip()
    return value[:-3] if value.lower().endswith(".md") else value


def _literal_fts_query(query: str) -> str:
    return " AND ".join(f'"{term}"' for term in re.findall(r"\w+", query))


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


__all__ = ["CatalogRecord", "MemoryCatalog"]
