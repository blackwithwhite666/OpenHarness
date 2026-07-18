"""Tests for the async adapter over ohmo's file memory implementation."""

from __future__ import annotations

import json
from pathlib import Path

import ohmo.memory_tool as memory_tool_module
from ohmo.memory import load_memory_prompt
from ohmo.memory_backend import FileMemoryBackend, MemoryHit
from ohmo.memory_store import MemoryOpResult, MemoryStore


class _FakeSearchProcess:
    def __init__(self, stdout: bytes) -> None:
        self.stdout = stdout
        self.returncode = 0

    async def communicate(self):
        return self.stdout, b""


async def test_file_memory_backend_crud_list_and_get_match_store(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    store = MemoryStore(tmp_path)
    backend = FileMemoryBackend(store)

    assert await backend.list() == store.list() == []
    assert await backend.get("missing") == store.get("missing") is None

    assert await backend.add("Timezone", "User prefers UTC.") == MemoryOpResult(
        True, "Saved memory timezone.md."
    )
    assert await backend.list() == store.list()
    assert await backend.get("timezone") == store.get("timezone")

    assert await backend.update("timezone", "User prefers MSK.") == MemoryOpResult(
        True, "Updated memory timezone.md."
    )
    assert await backend.get("timezone.md") == store.get("timezone.md")

    assert await backend.remove("timezone") == MemoryOpResult(True, "Archived memory timezone.md.")
    assert await backend.list() == store.list() == []
    assert await backend.get("timezone") == store.get("timezone") is None


async def test_file_memory_backend_search_matches_memory_tool_helper(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    store = MemoryStore(tmp_path)
    assert store.add("Timezone", "User prefers UTC timestamps.").ok
    backend = FileMemoryBackend(store)
    raw_hits = [
        {
            "source_path": "/home/me/.ohmo/memory/timezone.md",
            "collection": "memory",
            "score": 0.923,
            "snippet": "User prefers UTC timestamps.",
        },
        {
            "source_path": "/home/me/.ohmo/memory/archive/editor.md",
            "collection": "archive",
            "score": 0.801,
            "snippet": "User used Vim for editing.",
        },
    ]
    calls: list[tuple[str, ...]] = []

    async def fake_exec(*args, **kwargs):
        del kwargs
        calls.append(args)
        return _FakeSearchProcess(json.dumps(raw_hits).encode("utf-8"))

    monkeypatch.setattr(memory_tool_module.asyncio, "create_subprocess_exec", fake_exec)

    direct = await memory_tool_module._search_memory("timezone and editor", 2)
    hits = await backend.search("timezone and editor", 2)

    assert direct.output == (
        "2 memory hits for 'timezone and editor':\n"
        "- timezone (score 0.92): User prefers UTC timestamps.\n"
        "- editor (score 0.80): User used Vim for editing."
    )
    assert hits == [
        MemoryHit(
            name="timezone",
            title="Timezone",
            snippet="User prefers UTC timestamps.",
            rank=1,
        ),
        MemoryHit(
            name="editor",
            title="editor",
            snippet="User used Vim for editing.",
            rank=2,
        ),
    ]
    expected_argv = (
        memory_tool_module._DOCUMENT_SEARCH_CLI,
        "search",
        "timezone and editor",
        "--collection",
        memory_tool_module._MEMORY_SEARCH_COLLECTIONS,
        "--top-k",
        "2",
    )
    assert calls == [expected_argv, expected_argv]


async def test_file_memory_backend_render_prompt_matches_sync_renderer(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    store = MemoryStore(tmp_path)
    assert store.add("Alpha", "alpha body").ok
    assert store.add("Bravo", "bravo body").ok
    backend = FileMemoryBackend(store)

    rendered = await backend.render_prompt(budget=10)
    direct = load_memory_prompt(tmp_path, max_chars=10)

    assert rendered == direct
    assert "## alpha.md" in rendered
    assert "## bravo.md" not in rendered


async def test_file_memory_backend_append_turn_is_noop(tmp_path: Path):
    store = MemoryStore(tmp_path)
    assert store.add("Timezone", "User prefers UTC.").ok
    backend = FileMemoryBackend(store)
    before = store.list()

    assert await backend.append_turn("user", "Please remember this turn.") is None

    assert store.list() == before
