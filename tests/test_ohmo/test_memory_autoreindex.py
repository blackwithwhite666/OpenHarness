"""Tests for best-effort memory index refreshes."""

from __future__ import annotations

from pathlib import Path

from ohmo.document_search import reindex
from ohmo.memory_store import MemoryOpResult, MemoryStore


def test_add_reindexes_new_file(monkeypatch, tmp_path: Path):
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "ohmo.memory_store.reindex",
        lambda path, *, collection: calls.append((path, collection)),
    )

    result = MemoryStore(tmp_path).add("Timezone", "User prefers UTC.")

    assert result.ok
    assert calls == [(tmp_path / "memory" / "timezone.md", "memory")]


def test_update_reindexes_entry(monkeypatch, tmp_path: Path):
    store = MemoryStore(tmp_path)
    assert store.add("Timezone", "User prefers UTC.").ok
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "ohmo.memory_store.reindex",
        lambda path, *, collection: calls.append((path, collection)),
    )

    result = store.update("timezone", "User prefers Moscow time.")

    assert result.ok
    assert calls == [(tmp_path / "memory" / "timezone.md", "memory")]


def test_remove_reindexes_archived_file(monkeypatch, tmp_path: Path):
    store = MemoryStore(tmp_path)
    assert store.add("Timezone", "User prefers UTC.").ok
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "ohmo.memory_store.reindex",
        lambda path, *, collection: calls.append((path, collection)),
    )

    result = store.remove("timezone")

    assert result.ok
    assert calls == [(tmp_path / "memory" / "archive" / "timezone.md", "archive")]


def test_failed_mutation_does_not_reindex(monkeypatch, tmp_path: Path):
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "ohmo.memory_store.reindex",
        lambda path, *, collection: calls.append((path, collection)),
    )

    result = MemoryStore(tmp_path).add("Timezone", "")

    assert not result.ok
    assert calls == []


def test_reindex_failure_does_not_change_memory_result(monkeypatch, tmp_path: Path):
    def fail_reindex(path: str | Path, *, collection: str) -> None:
        raise RuntimeError("index spawn failed")

    monkeypatch.setattr("ohmo.memory_store.reindex", fail_reindex)

    result = MemoryStore(tmp_path).add("Timezone", "User prefers UTC.")

    assert result == MemoryOpResult(True, "Saved memory timezone.md.")
    assert (tmp_path / "memory" / "timezone.md").read_text(encoding="utf-8") == (
        "User prefers UTC.\n"
    )


def test_reindex_disabled_does_not_spawn(monkeypatch, tmp_path: Path):
    def unexpected_popen(*args, **kwargs):
        raise AssertionError("disabled auto-indexing must not spawn")

    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    monkeypatch.setattr("ohmo.document_search.subprocess.Popen", unexpected_popen)

    assert reindex(tmp_path / "timezone.md") is None


def test_reindex_missing_cli_is_silent(monkeypatch, tmp_path: Path):
    def missing_popen(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "1")
    monkeypatch.setattr("ohmo.document_search.subprocess.Popen", missing_popen)

    assert reindex(tmp_path / "timezone.md") is None


def test_real_mutations_never_spawn_under_test_env(monkeypatch, tmp_path: Path):
    # Regression: a real add/remove (reindex NOT mocked) must NOT spawn the
    # document_search CLI under the test env — the autouse conftest fixture keeps
    # OHMO_MEMORY_AUTOINDEX=0. Without it, pytest on a host that has the CLI (the
    # self-hosted CI runner) polluted the SHARED ~/.document_search index.
    import ohmo.document_search as ds

    calls: list = []
    monkeypatch.setattr(ds.subprocess, "Popen", lambda *a, **k: calls.append(a))
    store = MemoryStore(tmp_path)
    assert store.add("Timezone", "User prefers UTC.").ok
    assert store.remove("timezone").ok
    assert calls == []  # auto-index disabled by the conftest fixture -> no CLI spawn
