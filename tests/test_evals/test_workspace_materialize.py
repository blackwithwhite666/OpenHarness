from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals.executor import EvalToolFixture, ReplayToolInput
from openharness.evals.workspace_materialize import (
    LiveLocalReadTool,
    _safe_join,
    materialize_read_fixtures,
    remap_in,
)
from openharness.tools import create_default_tool_registry
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


def test_safe_join_and_remap_in_confine_paths_to_sandbox(tmp_path: Path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    mapped = _safe_join(sandbox, "/home/u/x.md")

    assert mapped == sandbox.resolve() / "home" / "u" / "x.md"
    assert remap_in("/home/u/x.md", sandbox) == str(mapped)
    assert _safe_join(sandbox, "../../etc/passwd") is None

    outside = tmp_path / "outside"
    outside.mkdir()
    escape = sandbox / "escape"
    try:
        escape.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not available on this filesystem")
    assert _safe_join(sandbox, "/escape/passwd") is None


def test_materialize_read_fixtures_writes_only_safe_read_file_outputs(tmp_path: Path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    fixtures = (
        EvalToolFixture(
            tool_name="read_file",
            call_key_hash="safe",
            input_text='{"path": "/home/u/a.md"}',
            output_text="CONTENT",
        ),
        EvalToolFixture(
            tool_name="read_file",
            call_key_hash="escape",
            input_text='{"path": "../../etc/passwd"}',
            output_text="SECRET",
        ),
        EvalToolFixture(
            tool_name="glob",
            call_key_hash="glob",
            input_text='{"root": "/home/u", "pattern": "*.md"}',
            output_text="a.md",
        ),
    )

    count = materialize_read_fixtures(fixtures, sandbox)

    assert count == 1
    assert (sandbox / "home" / "u" / "a.md").read_text(encoding="utf-8") == "CONTENT"
    assert not (tmp_path / "etc" / "passwd").exists()


@pytest.mark.asyncio
async def test_live_local_read_tool_reads_materialized_file_and_falls_back(
    tmp_path: Path,
):
    real_tool = create_default_tool_registry().get("read_file")
    assert real_tool is not None
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = _safe_join(sandbox, "/home/u/a.md")
    assert target is not None
    target.parent.mkdir(parents=True)
    target.write_text("CONTENT\n", encoding="utf-8")
    empty_target = _safe_join(sandbox, "/home/u/empty.md")
    assert empty_target is not None
    empty_target.write_text("", encoding="utf-8")
    mock_tool = _RecordingMockTool("read_file", real_tool.input_model, output="MOCKED")
    tool = LiveLocalReadTool(
        real_tool=real_tool,
        mock_tool=mock_tool,
        sandbox=sandbox,
    )

    result = await tool.execute(
        real_tool.input_model(path="/home/u/a.md", limit=20),
        ToolExecutionContext(cwd=tmp_path),
    )
    empty_result = await tool.execute(
        real_tool.input_model(path="/home/u/empty.md", limit=20),
        ToolExecutionContext(cwd=tmp_path),
    )
    fallback_result = await tool.execute(
        real_tool.input_model(path="../../etc/passwd"),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert "CONTENT" in result.output
    assert result.metadata["lane"] == "live-local"
    assert result.metadata["tool"] == "read_file"
    assert "/home/u/empty.md" in empty_result.output
    assert str(sandbox.resolve()) not in result.output
    assert str(sandbox.resolve()) not in empty_result.output
    assert fallback_result.output == "MOCKED"
    assert mock_tool.calls == [{"path": "../../etc/passwd", "offset": 0, "limit": 200}]


@pytest.mark.asyncio
async def test_live_local_read_tool_glob_uses_remapped_root(tmp_path: Path):
    real_tool = create_default_tool_registry().get("glob")
    assert real_tool is not None
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = _safe_join(sandbox, "/home/u/a.md")
    assert target is not None
    target.parent.mkdir(parents=True)
    target.write_text("CONTENT\n", encoding="utf-8")
    mock_tool = _RecordingMockTool("glob", real_tool.input_model, output="MOCKED")
    tool = LiveLocalReadTool(
        real_tool=real_tool,
        mock_tool=mock_tool,
        sandbox=sandbox,
    )

    result = await tool.execute(
        real_tool.input_model(pattern="*.md", root="/home/u", limit=20),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "a.md"
    assert str(sandbox.resolve()) not in result.output
    assert mock_tool.calls == []


class _RecordingMockTool(BaseTool):
    description = "recording mock tool"
    input_model = ReplayToolInput

    def __init__(
        self,
        name: str,
        input_model: type[ReplayToolInput],
        *,
        output: str,
    ) -> None:
        self.name = name
        self.input_model = input_model
        self.output = output
        self.calls: list[dict[str, object]] = []

    async def execute(
        self,
        arguments: ReplayToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        self.calls.append(arguments.model_dump())
        return ToolResult(output=self.output, metadata={"replayed": True})

    def is_read_only(self, arguments: ReplayToolInput) -> bool:
        del arguments
        return True
