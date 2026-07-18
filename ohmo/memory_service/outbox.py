"""At-least-once Honcho mirror for authoritative curated catalog memory."""

from __future__ import annotations

import json
import sqlite3
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol, Sequence, cast

from ohmo.memory_service.honcho_client import HonchoClient

_LEASE_SECONDS = 30.0


class _OutboxCatalogMixin:
    """SQLite outbox operations mixed into ``MemoryCatalog`` to keep it compact."""

    def _write_connection(self) -> AbstractContextManager[sqlite3.Connection]:
        raise NotImplementedError

    @staticmethod
    def _migrate_to_v2(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY,
                op_type TEXT NOT NULL CHECK (op_type IN ('add', 'update', 'remove')),
                slug TEXT NOT NULL,
                content TEXT,
                old_conclusion_ids TEXT NOT NULL DEFAULT '[]',
                state TEXT NOT NULL DEFAULT 'pending'
                    CHECK (state IN ('pending', 'leased', 'done', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                lease_expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _enqueue_outbox(
        connection: sqlite3.Connection,
        op_type: str,
        slug: str,
        content: str | None,
        *,
        old_conclusion_ids: str | None = "[]",
        timestamp: str | None = None,
    ) -> None:
        now = timestamp or _timestamp()
        connection.execute(
            """
            INSERT INTO outbox (
                op_type, slug, content, old_conclusion_ids, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (op_type, slug, content, old_conclusion_ids or "[]", now, now),
        )

    def lease_outbox(self, limit: int, lease_seconds: float) -> list[sqlite3.Row]:
        """Atomically lease pending or expired operations in creation order."""
        if limit <= 0:
            return []
        now = _timestamp()
        expires = _timestamp(timedelta(seconds=max(0.0, lease_seconds)))
        with self._write_connection() as connection:
            id_rows = connection.execute(
                """
                SELECT id FROM outbox
                WHERE state = 'pending'
                   OR (state = 'leased' AND lease_expires_at <= ?)
                ORDER BY id LIMIT ?
                """,
                (now, limit),
            ).fetchall()
            ids = [cast(int, row["id"]) for row in id_rows]
            if not ids:
                return []
            placeholders = ", ".join("?" for _ in ids)
            connection.execute(
                f"""
                UPDATE outbox SET state = 'leased', lease_expires_at = ?, updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (expires, now, *ids),
            )
            return connection.execute(
                f"SELECT * FROM outbox WHERE id IN ({placeholders}) ORDER BY id", ids
            ).fetchall()

    def mark_outbox_done(self, outbox_id: int) -> None:
        with self._write_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE outbox SET state = 'done', lease_expires_at = NULL, updated_at = ?
                WHERE id = ? AND state = 'leased'
                """,
                (_timestamp(), outbox_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"outbox operation {outbox_id} is not leased")

    def mark_outbox_retry(self, outbox_id: int) -> None:
        with self._write_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE outbox SET state = 'pending', attempts = attempts + 1,
                    lease_expires_at = NULL, updated_at = ?
                WHERE id = ? AND state = 'leased'
                """,
                (_timestamp(), outbox_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"outbox operation {outbox_id} is not leased")

    def set_conclusion_ids(self, slug: str, conclusion_ids: Sequence[str]) -> None:
        with self._write_connection() as connection:
            cursor = connection.execute(
                "UPDATE memories SET honcho_conclusion_ids = ? WHERE slug = ?",
                (json.dumps(list(conclusion_ids)), _slug_reference(slug)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown memory {_slug_reference(slug)!r}")

    def append_conclusion_id(self, slug: str, conclusion_id: str) -> None:
        clean_slug = _slug_reference(slug)
        with self._write_connection() as connection:
            row = connection.execute(
                "SELECT honcho_conclusion_ids FROM memories WHERE slug = ?", (clean_slug,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown memory {clean_slug!r}")
            conclusion_ids = _decode_ids(row["honcho_conclusion_ids"])
            if conclusion_id not in conclusion_ids:
                conclusion_ids.append(conclusion_id)
            connection.execute(
                "UPDATE memories SET honcho_conclusion_ids = ? WHERE slug = ?",
                (json.dumps(conclusion_ids), clean_slug),
            )

    def reconcile_outbox(self) -> dict[str, int]:
        """Requeue operations whose drainer lease has expired."""
        now = _timestamp()
        with self._write_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE outbox SET state = 'pending', lease_expires_at = NULL, updated_at = ?
                WHERE state = 'leased' AND lease_expires_at <= ?
                """,
                (now, now),
            )
        return {"requeued": cursor.rowcount}


class _OutboxCatalog(Protocol):
    def lease_outbox(self, limit: int, lease_seconds: float) -> list[sqlite3.Row]: ...

    def mark_outbox_done(self, outbox_id: int) -> None: ...

    def mark_outbox_retry(self, outbox_id: int) -> None: ...

    def set_conclusion_ids(self, slug: str, conclusion_ids: Sequence[str]) -> None: ...

    def append_conclusion_id(self, slug: str, conclusion_id: str) -> None: ...

    def reconcile_outbox(self) -> dict[str, int]: ...


@dataclass(frozen=True, slots=True)
class DrainReport:
    """Outcome counts for one bounded drain pass."""

    mirrored: int = 0
    retried: int = 0
    failed: int = 0


async def drain_once(
    catalog: _OutboxCatalog,
    honcho_client: HonchoClient,
    *,
    batch: int = 100,
) -> DrainReport:
    """Lease and deliver one batch without putting Honcho on the write path."""
    mirrored = retried = failed = 0
    for operation in catalog.lease_outbox(batch, _LEASE_SECONDS):
        outbox_id = cast(int, operation["id"])
        op_type = cast(str, operation["op_type"])
        slug = cast(str, operation["slug"])
        try:
            new_id = None
            if op_type in {"add", "update"}:
                content = operation["content"]
                if not isinstance(content, str):
                    raise ValueError(f"outbox operation {outbox_id} has no content")
                acknowledgements = await honcho_client.create_conclusions(
                    [
                        {
                            "content": content,
                            "observer_id": "ohmo-curated",
                            "observed_id": "owner",
                        }
                    ]
                )
                if not acknowledgements or not acknowledgements[0].id:
                    raise RuntimeError("Honcho did not acknowledge the conclusion")
                new_id = acknowledgements[0].id
        except Exception:
            try:
                catalog.mark_outbox_retry(outbox_id)
            except Exception:
                failed += 1
            else:
                retried += 1
            continue

        try:
            old_ids = _decode_ids(operation["old_conclusion_ids"])
            if op_type == "add":
                assert new_id is not None
                catalog.append_conclusion_id(slug, new_id)
            elif op_type == "update":
                assert new_id is not None
                await _delete_best_effort(honcho_client, old_ids)
                catalog.set_conclusion_ids(slug, [new_id])
            elif op_type == "remove":
                await _delete_best_effort(honcho_client, old_ids)
            else:
                raise ValueError(f"unknown outbox operation {op_type!r}")
            catalog.mark_outbox_done(outbox_id)
        except Exception:
            # An acknowledged remote write must never be converted to pending:
            # leave the lease in place so expiry causes an at-least-once replay.
            failed += 1
        else:
            mirrored += 1
    return DrainReport(mirrored=mirrored, retried=retried, failed=failed)


def reconcile_outbox(catalog: _OutboxCatalog) -> dict[str, int]:
    """Run the synchronous expired-lease reconciliation pass."""
    return catalog.reconcile_outbox()


async def _delete_best_effort(honcho_client: HonchoClient, conclusion_ids: Sequence[str]) -> None:
    for conclusion_id in conclusion_ids:
        try:
            await honcho_client.delete_conclusion(conclusion_id)
        except Exception:
            pass


def _decode_ids(raw: object) -> list[str]:
    if not isinstance(raw, str):
        raise ValueError("conclusion ids must be encoded as JSON text")
    decoded = json.loads(raw)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ValueError("conclusion ids must be a JSON array of strings")
    return decoded


def _slug_reference(slug: str) -> str:
    value = (slug or "").strip()
    return value[:-3] if value.lower().endswith(".md") else value


def _timestamp(offset: timedelta = timedelta()) -> str:
    return (datetime.now(timezone.utc) + offset).isoformat().replace("+00:00", "Z")


__all__ = ["DrainReport", "drain_once", "reconcile_outbox"]
