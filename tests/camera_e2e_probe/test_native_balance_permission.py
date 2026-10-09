"""Exercise default-mode admission through a real engine and MCP adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from native_balance_config import WELLNESS_TOOL, admit_native_wellness_read
from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.config.settings import PermissionSettings, Settings
from openharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import ToolExecutionCompleted
from openharness.mcp.client import McpToolCallResult
from openharness.mcp.types import McpToolInfo
from openharness.permissions import PermissionChecker, PermissionMode
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from openharness.tools.mcp_tool import McpToolAdapter


class _ScriptedClient:
    def __init__(self, calls: list[str]):
        self.calls = calls
        self.turn = 0

    async def stream_message(self, _request):
        self.turn += 1
        if self.turn == 1:
            content = [ToolUseBlock(id=f"call-{name}", name=name, input={}) for name in self.calls]
        else:
            content = [TextBlock(text="Finished the offline admission check.")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason=None,
        )


class _Manager:
    def __init__(self):
        self.calls = []

    async def call_tool_result(self, server_name, tool_name, arguments):
        self.calls.append((server_name, tool_name, arguments))
        return McpToolCallResult(output="fixture wellness read")


class _Args(BaseModel):
    pass


class _MutatingTool(BaseTool):
    name = "probe_mutation"
    description = "A mutation that must remain blocked"
    input_model = _Args

    def __init__(self):
        self.calls = 0

    async def execute(self, _arguments: BaseModel, _context: ToolExecutionContext) -> ToolResult:
        self.calls += 1
        return ToolResult(output="mutated")


async def _run_engine(tmp_path, monkeypatch, denied_tools, calls):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    original = PermissionSettings(mode=PermissionMode.DEFAULT, denied_tools=denied_tools)
    manager = _Manager()
    mutator = _MutatingTool()
    registry = ToolRegistry()
    registry.register(McpToolAdapter(manager, McpToolInfo(
        server_name="worfalomey", name="get_wellness_data",
        description="wellness fixture", input_schema={"type": "object", "properties": {}},
    )))
    registry.register(mutator)
    engine = QueryEngine(
        api_client=_ScriptedClient(calls), tool_registry=registry,
        permission_checker=PermissionChecker(original), cwd=tmp_path,
        model="offline-test", system_prompt="offline permission test", max_turns=3,
    )
    bundle = SimpleNamespace(engine=engine, current_settings=lambda: Settings(permission=original))
    admit_native_wellness_read(bundle)
    events = [event async for event in engine.submit_message("check permission boundary")]
    return manager, mutator, events


@pytest.mark.asyncio
async def test_only_wellness_adapter_runs_in_default_mode(tmp_path, monkeypatch):
    manager, mutator, events = await _run_engine(
        tmp_path, monkeypatch, [], [WELLNESS_TOOL, "probe_mutation"],
    )
    results = {event.tool_name: event for event in events if isinstance(event, ToolExecutionCompleted)}
    assert manager.calls == [("worfalomey", "get_wellness_data", {})]
    assert mutator.calls == 0
    assert results[WELLNESS_TOOL].is_error is False
    assert results["probe_mutation"].is_error is True
    assert "confirmation" in results["probe_mutation"].output.lower()


@pytest.mark.asyncio
async def test_explicit_deny_wins_over_task_local_wellness_allow(tmp_path, monkeypatch):
    manager, mutator, events = await _run_engine(
        tmp_path, monkeypatch, [WELLNESS_TOOL], [WELLNESS_TOOL],
    )
    results = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    assert manager.calls == [] and mutator.calls == 0
    assert len(results) == 1 and results[0].is_error is True
    assert "explicitly denied" in results[0].output
