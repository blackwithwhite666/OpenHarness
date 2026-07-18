"""Tests for backend-aware ``/memory`` and ``/dream`` dispatch."""

from __future__ import annotations

from pathlib import Path

from openharness.commands import CommandContext, create_default_command_registry
from openharness.config.settings import load_settings
from openharness.engine.query_engine import QueryEngine
from openharness.permissions import PermissionChecker
from openharness.tools import create_default_tool_registry

from ohmo.memory import create_memory_command_backend
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


async def test_file_backend_memory_command_keeps_existing_behavior(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    workspace = tmp_path / "workspace"
    registry = create_default_command_registry()
    context = _make_context(tmp_path, workspace, backend_kind="file")

    add_command, add_args = registry.lookup("/memory add Notes :: remember this")
    add_result = await add_command.handler(add_args, context)

    assert add_result.message == "Added memory entry notes.md"
    assert (get_memory_dir(workspace) / "notes.md").read_text(encoding="utf-8") == "remember this\n"

    show_command, show_args = registry.lookup("/memory show notes")
    show_result = await show_command.handler(show_args, context)

    assert show_result.message == "remember this\n"


async def test_honcho_backend_disables_memory_and_dream_without_touching_workspace(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "marker.txt"
    marker.write_text("unchanged\n", encoding="utf-8")
    registry = create_default_command_registry()
    context = _make_context(tmp_path, workspace, backend_kind="honcho")
    before = {path.relative_to(workspace): path.read_bytes() for path in workspace.rglob("*")}

    memory_payloads = (
        "/memory show notes",
        "/memory add Notes :: must not be written",
        "/memory remove notes",
        "/memory edit notes",
        "/memory migrate --apply",
    )
    for payload in memory_payloads:
        command, args = registry.lookup(payload)
        result = await command.handler(args, context)
        assert result.message == "/memory is not supported on the honcho backend."

    dream_command, dream_args = registry.lookup("/dream")
    dream_result = await dream_command.handler(dream_args, context)

    assert dream_result.message == "/dream is not supported on the honcho backend."
    after = {path.relative_to(workspace): path.read_bytes() for path in workspace.rglob("*")}
    assert after == before
