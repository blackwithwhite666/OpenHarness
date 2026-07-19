"""Catalog-backed owner commands and first-boot migration tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import openharness.commands.registry as registry_module
from openharness.commands import CommandContext, create_default_command_registry
from openharness.config.settings import load_settings
from openharness.engine.query_engine import QueryEngine
from openharness.permissions import PermissionChecker
from openharness.tools import create_default_tool_registry

import ohmo.memory as memory_module
from ohmo.memory import create_memory_command_backend, ensure_catalog_migrated
from ohmo.memory_backend import CatalogMemoryBackend, make_memory_backend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_store import MemoryStore
from ohmo.gateway.models import GatewayConfig
from ohmo.workspace import get_memory_dir


class _FakeApiClient:
    async def stream_message(self, request):
        del request
        raise AssertionError("stream_message should not be called in command tests")


def _make_context(cwd: Path, workspace: Path, *, backend_kind: str) -> CommandContext:
    tool_registry = create_default_tool_registry()
    return CommandContext(
        engine=QueryEngine(
            api_client=_FakeApiClient(),
            tool_registry=tool_registry,
            permission_checker=PermissionChecker(load_settings().permission),
            cwd=cwd,
            model="claude-test",
            system_prompt="system",
        ),
        cwd=str(cwd),
        tool_registry=tool_registry,
        memory_backend=create_memory_command_backend(
            workspace,
            backend_kind=backend_kind,
        ),
        include_project_memory=False,
    )


async def test_catalog_memory_commands_curate_catalog_without_markdown(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("EDITOR", "fake-editor")
    workspace = tmp_path / "workspace"
    registry = create_default_command_registry()
    context = _make_context(tmp_path, workspace, backend_kind="catalog")

    add_command, add_args = registry.lookup("/memory add Notes :: remember this")
    add_result = await add_command.handler(add_args, context)

    assert add_result.message == "Added memory entry notes.md"
    catalog = MemoryCatalog(workspace)
    note = catalog.get("owner", "notes")
    assert note is not None
    assert note.content == "remember this"

    show_command, show_args = registry.lookup("/memory show notes")
    show_result = await show_command.handler(show_args, context)
    assert show_result.message == "remember this"

    list_command, list_args = registry.lookup("/memory list")
    list_result = await list_command.handler(list_args, context)
    assert list_result.message == "notes.md"

    def fake_editor(argv, *, cwd, check):
        del cwd, check
        Path(argv[1]).write_text("edited in catalog\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(registry_module.subprocess, "run", fake_editor)
    edit_command, edit_args = registry.lookup("/memory edit notes")
    edit_result = await edit_command.handler(edit_args, context)
    assert edit_result.message == "Edited catalog memory entry: notes"
    edited = catalog.get("owner", "notes")
    assert edited is not None
    assert edited.content == "edited in catalog"

    remove_command, remove_args = registry.lookup("/memory remove notes")
    remove_result = await remove_command.handler(remove_args, context)
    assert remove_result.message == "Removed memory entry notes"
    archived = catalog.get("owner", "notes")
    assert archived is not None
    assert archived.archive_status == "archived"

    assert (await list_command.handler(list_args, context)).message == "No memory files."
    assert (await show_command.handler(show_args, context)).message == (
        "Memory entry not found: notes"
    )
    assert list(get_memory_dir(workspace).rglob("*.md")) == []


async def test_file_memory_commands_keep_exact_existing_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    workspace = tmp_path / "workspace"
    registry = create_default_command_registry()
    context = _make_context(tmp_path, workspace, backend_kind="file")

    add_command, add_args = registry.lookup("/memory add Notes :: remember this")
    add_result = await add_command.handler(add_args, context)
    assert add_result.message == "Added memory entry notes.md"

    memory_dir = get_memory_dir(workspace)
    assert (memory_dir / "notes.md").read_bytes() == b"remember this\n"
    assert (memory_dir / "MEMORY.md").read_bytes() == (b"# Memory Index\n- [Notes](notes.md)\n")

    list_command, list_args = registry.lookup("/memory list")
    assert (await list_command.handler(list_args, context)).message == "notes.md"
    show_command, show_args = registry.lookup("/memory show notes")
    assert (await show_command.handler(show_args, context)).message == "remember this\n"

    remove_command, remove_args = registry.lookup("/memory remove notes")
    assert (await remove_command.handler(remove_args, context)).message == (
        "Removed memory entry notes"
    )
    assert (await list_command.handler(list_args, context)).message == "No memory files."
    assert (await show_command.handler(show_args, context)).message == (
        "Memory entry not found: notes"
    )


async def test_dream_is_not_applicable_to_catalog_without_file_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    workspace = tmp_path / "workspace"
    registry = create_default_command_registry()
    context = _make_context(tmp_path, workspace, backend_kind="catalog")
    before = {
        path.relative_to(workspace): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }

    dream_command, dream_args = registry.lookup("/dream")
    result = await dream_command.handler(dream_args, context)

    assert result.message == "/dream is not applicable on the catalog backend."
    after = {
        path.relative_to(workspace): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_ensure_catalog_migrated_is_copy_only_and_noops_when_populated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    store = MemoryStore(workspace)
    store.add_legacy("Home timezone", "User lives in Moscow.")
    store.add_legacy("Editor preference", "User prefers Neovim.")
    memory_dir = get_memory_dir(workspace)
    markdown_before = {
        path.relative_to(memory_dir): path.read_bytes() for path in memory_dir.rglob("*.md")
    }

    catalog = ensure_catalog_migrated(workspace)

    assert {
        record.slug: (record.title, record.content)
        for record in catalog.list("owner", include_archived=True)
    } == {
        "editor_preference": ("Editor preference", "User prefers Neovim."),
        "home_timezone": ("Home timezone", "User lives in Moscow."),
    }

    def fail_migrate(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a populated catalog must not be migrated again")

    monkeypatch.setattr(memory_module, "migrate", fail_migrate)
    second = ensure_catalog_migrated(workspace)

    assert second.list("owner", include_archived=True) == catalog.list("owner", include_archived=True)
    assert {
        path.relative_to(memory_dir): path.read_bytes() for path in memory_dir.rglob("*.md")
    } == markdown_before


async def test_catalog_backend_factory_migrates_before_first_recall(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = MemoryStore(workspace)
    store.add_legacy("Profile", "User prefers concise answers.")
    markdown_before = {
        path.relative_to(get_memory_dir(workspace)): path.read_bytes()
        for path in get_memory_dir(workspace).rglob("*.md")
    }

    backend = make_memory_backend(GatewayConfig(memory_backend="catalog"), workspace)

    assert isinstance(backend, CatalogMemoryBackend)
    assert [(entry.name, entry.content) for entry in await backend.list()] == [
        ("profile.md", "User prefers concise answers."),
    ]
    assert {
        path.relative_to(get_memory_dir(workspace)): path.read_bytes()
        for path in get_memory_dir(workspace).rglob("*.md")
    } == markdown_before
