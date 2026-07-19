"""Tests for the at-least-once curated-memory Honcho mirror."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service.outbox import DrainReport, drain_once


class FakeHoncho:
    def __init__(self, *, unreachable: bool = False) -> None:
        self.unreachable = unreachable
        self.created: list[list[dict[str, object]]] = []
        self.deleted: list[str] = []

    async def create_conclusions(self, conclusions: list[dict[str, object]]) -> list[Any]:
        self.created.append(conclusions)
        if self.unreachable:
            raise OSError("honcho is unreachable")
        return [SimpleNamespace(id=f"conclusion-{len(self.created)}")]

    async def delete_conclusion(self, conclusion_id: str) -> None:
        self.deleted.append(conclusion_id)


def _outbox_rows(catalog: MemoryCatalog) -> list[sqlite3.Row]:
    connection = sqlite3.connect(catalog.db_path)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute("SELECT * FROM outbox ORDER BY id").fetchall()
    finally:
        connection.close()


def _expire_leases(catalog: MemoryCatalog) -> None:
    with sqlite3.connect(catalog.db_path) as connection:
        connection.execute(
            "UPDATE outbox SET lease_expires_at = '2000-01-01T00:00:00Z' WHERE state = 'leased'"
        )


def test_schema_v1_migrates_to_latest_idempotently(tmp_path: Path):
    db_path = tmp_path / "catalog.sqlite3"
    with sqlite3.connect(db_path, isolation_level=None) as connection:
        MemoryCatalog._migrate_to_v1(connection)
        connection.execute("PRAGMA user_version = 1")

    MemoryCatalog(db_path=db_path)
    MemoryCatalog(db_path=db_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(outbox)").fetchall()
        }
        embedding_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(memory_embeddings)").fetchall()
        }
    assert columns == {
        "id",
        "tenant_id",
        "op_type",
        "slug",
        "content",
        "old_conclusion_ids",
        "state",
        "attempts",
        "lease_expires_at",
        "created_at",
        "updated_at",
    }
    assert embedding_columns == {
        "tenant_id",
        "slug",
        "model",
        "dim",
        "vector",
        "generation",
        "updated_at",
    }


def test_curated_add_and_enqueue_are_atomic_and_do_not_call_honcho(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    honcho = FakeHoncho(unreachable=True)

    assert catalog.add("owner", "Timezone", "User lives in Moscow.").ok

    [operation] = _outbox_rows(catalog)
    assert (operation["op_type"], operation["slug"], operation["content"]) == (
        "add",
        "timezone",
        "User lives in Moscow.",
    )
    assert (operation["state"], operation["attempts"]) == ("pending", 0)
    assert honcho.created == []

    def fail_enqueue(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("outbox unavailable")

    monkeypatch.setattr(catalog, "_enqueue_outbox", fail_enqueue)
    with pytest.raises(sqlite3.OperationalError, match="outbox unavailable"):
        catalog.add("owner", "Rolled back", "This row must roll back.")
    assert catalog.get("owner", "rolled_back") is None


async def test_drain_add_uses_curated_peer_pair_records_ack_and_marks_done(tmp_path: Path):
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    honcho = FakeHoncho()
    assert catalog.add("owner", "Timezone", "User lives in Moscow.").ok

    report = await drain_once(catalog, honcho)

    assert report == DrainReport(mirrored=1)
    assert honcho.created == [
        [
            {
                "content": "User lives in Moscow.",
                "observer_id": "ohmo-curated",
                "observed_id": "owner",
            }
        ]
    ]
    record = catalog.get("owner", "timezone")
    assert record is not None
    assert json.loads(record.honcho_conclusion_ids) == ["conclusion-1"]
    assert _outbox_rows(catalog)[0]["state"] == "done"


async def test_ack_followed_by_local_failure_is_replayed_at_least_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    honcho = FakeHoncho()
    assert catalog.add("owner", "Editor", "User prefers Neovim.").ok
    original_append = catalog.append_conclusion_id

    def fail_local_record(slug: str, conclusion_id: str) -> None:
        raise sqlite3.OperationalError("disk unavailable after ack")

    monkeypatch.setattr(catalog, "append_conclusion_id", fail_local_record)
    assert await drain_once(catalog, honcho) == DrainReport(failed=1)
    assert _outbox_rows(catalog)[0]["state"] == "leased"
    _expire_leases(catalog)
    monkeypatch.setattr(catalog, "append_conclusion_id", original_append)

    assert await drain_once(catalog, honcho) == DrainReport(mirrored=1)
    assert len(honcho.created) == 2
    record = catalog.get("owner", "editor")
    assert record is not None
    assert json.loads(record.honcho_conclusion_ids) == ["conclusion-2"]
    assert _outbox_rows(catalog)[0]["state"] == "done"


async def test_update_replaces_conclusion_and_remove_deletes_current_id(tmp_path: Path):
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    honcho = FakeHoncho()
    assert catalog.add("owner", "Editor", "User prefers Vim.").ok
    assert await drain_once(catalog, honcho) == DrainReport(mirrored=1)

    assert catalog.update("owner", "editor", "User prefers Neovim.").ok
    update_operation = _outbox_rows(catalog)[1]
    assert json.loads(update_operation["old_conclusion_ids"]) == ["conclusion-1"]
    assert await drain_once(catalog, honcho) == DrainReport(mirrored=1)
    assert honcho.deleted == ["conclusion-1"]
    record = catalog.get("owner", "editor")
    assert record is not None
    assert json.loads(record.honcho_conclusion_ids) == ["conclusion-2"]

    assert catalog.remove("owner", "editor").ok
    remove_operation = _outbox_rows(catalog)[2]
    assert json.loads(remove_operation["old_conclusion_ids"]) == ["conclusion-2"]
    assert await drain_once(catalog, honcho) == DrainReport(mirrored=1)
    assert honcho.deleted == ["conclusion-1", "conclusion-2"]
    assert _outbox_rows(catalog)[2]["state"] == "done"


def test_reconcile_requeues_an_expired_lease(tmp_path: Path):
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    assert catalog.add("owner", "Shell", "User prefers zsh.").ok
    assert len(catalog.lease_outbox(1, 60)) == 1
    _expire_leases(catalog)

    assert catalog.reconcile_outbox() == {"requeued": 1}
    operation = _outbox_rows(catalog)[0]
    assert operation["state"] == "pending"
    assert operation["lease_expires_at"] is None


async def test_honcho_error_retries_without_affecting_committed_catalog_write(tmp_path: Path):
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    honcho = FakeHoncho(unreachable=True)
    assert catalog.add("owner", "Shell", "User prefers zsh.").ok
    assert catalog.get("owner", "shell") is not None

    assert await drain_once(catalog, honcho) == DrainReport(retried=1)

    operation = _outbox_rows(catalog)[0]
    assert (operation["state"], operation["attempts"]) == ("pending", 1)
    assert catalog.get("owner", "shell") is not None


def test_imported_and_derived_writes_do_not_enqueue(tmp_path: Path):
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")

    assert catalog.import_entry("owner", "legacy", "Legacy", "Imported curated memory.").ok
    assert catalog.add("owner", "Derived", "Synthesized memory.", source="derived").ok
    assert catalog.update("owner", "derived", "Updated synthesized memory.").ok
    assert catalog.remove("owner", "derived").ok

    assert _outbox_rows(catalog) == []
