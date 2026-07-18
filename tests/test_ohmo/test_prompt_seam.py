"""Tests for fresh per-turn ohmo memory prompt assembly."""

from __future__ import annotations

from pathlib import Path

from ohmo.memory_backend import FileMemoryBackend
from ohmo.memory_store import MemoryStore
from ohmo.prompt_seam import compose_runtime_prompt, prepare_turn
from ohmo.prompts import build_ohmo_system_prompt


async def test_prompt_seam_is_byte_identical_to_legacy_owner_path(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    workspace = tmp_path / ".ohmo-home"
    store = MemoryStore(workspace)
    assert store.add("Timezone", "User prefers UTC.").ok
    assert store.add("Editor", "User prefers Vim.").ok
    backend = FileMemoryBackend(store)

    legacy = build_ohmo_system_prompt(
        tmp_path, workspace=workspace, include_ohmo_memory=True
    )
    memory_free = build_ohmo_system_prompt(
        tmp_path, workspace=workspace, include_ohmo_memory=False
    )
    snapshot = await prepare_turn(backend)

    assert compose_runtime_prompt(memory_free, snapshot) == legacy


def test_memory_free_persona_has_no_memory_or_workspace_paths(tmp_path: Path):
    workspace = tmp_path / ".ohmo-home"

    prompt = build_ohmo_system_prompt(
        tmp_path, workspace=workspace, include_ohmo_memory=False
    )

    assert "# ohmo Memory" not in prompt
    assert "# ohmo Workspace" not in prompt
    assert "Personal workspace root:" not in prompt
    assert "Personal memory directory:" not in prompt
    assert str(workspace) not in prompt


async def test_compose_runtime_prompt_injects_exactly_one_memory_block(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    workspace = tmp_path / ".ohmo-home"
    backend = FileMemoryBackend(MemoryStore(workspace))
    memory_free = build_ohmo_system_prompt(
        tmp_path, workspace=workspace, include_ohmo_memory=False
    )
    snapshot = await prepare_turn(backend)

    composed = compose_runtime_prompt(memory_free, snapshot)

    assert composed.count("# ohmo Memory") == 1
    assert compose_runtime_prompt(composed, snapshot) == composed


async def test_prepare_turn_reads_backend_writes(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")
    workspace = tmp_path / ".ohmo-home"
    backend = FileMemoryBackend(MemoryStore(workspace))
    memory_free = build_ohmo_system_prompt(
        tmp_path, workspace=workspace, include_ohmo_memory=False
    )
    frozen_snapshot = await prepare_turn(backend)
    frozen_prompt = compose_runtime_prompt(memory_free, frozen_snapshot)

    added = await backend.add("Timezone", "User prefers UTC timestamps.")
    fresh_snapshot = await prepare_turn(backend)

    assert added.ok
    assert "User prefers UTC timestamps." not in frozen_prompt
    assert "User prefers UTC timestamps." in fresh_snapshot
