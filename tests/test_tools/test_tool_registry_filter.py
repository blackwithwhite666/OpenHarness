"""Tests for ToolRegistry.apply_tool_filter (sub-agent toolset partition).

The deep-research routing (ADR adrs/deep-research-openharness.md §3) relies on a
spawned sub-agent's worker restricting its tool registry to the def's
``tools`` / ``disallowed_tools``. ``apply_tool_filter`` is the in-place filter
the worker applies. Offline, no network.
"""

from __future__ import annotations

from pydantic import BaseModel

from openharness.tools.base import (
    BaseTool,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
)


class _StubTool(BaseTool):
    """Minimal tool with a settable name for registry tests."""

    input_model = BaseModel

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"stub {name}"

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:  # noqa: ARG002
        return ToolResult(output="")


def _registry(*names: str) -> ToolRegistry:
    reg = ToolRegistry()
    for name in names:
        reg.register(_StubTool(name))
    return reg


def _names(reg: ToolRegistry) -> set[str]:
    return {t.name for t in reg.list_tools()}


def test_allowlist_restricts_to_named_tools():
    reg = _registry("web_fetch", "bash", "agent", "file_write")
    reg.apply_tool_filter(["web_fetch", "bash"], None)
    assert _names(reg) == {"web_fetch", "bash"}


def test_allowlist_star_is_all_tools():
    reg = _registry("web_fetch", "bash", "file_write")
    reg.apply_tool_filter(["*"], None)
    assert _names(reg) == {"web_fetch", "bash", "file_write"}


def test_allowlist_none_is_all_tools():
    reg = _registry("web_fetch", "bash")
    reg.apply_tool_filter(None, None)
    assert _names(reg) == {"web_fetch", "bash"}


def test_denylist_removes_named_tools():
    reg = _registry("read_file", "web_fetch", "bash", "agent", "file_write")
    reg.apply_tool_filter(None, ["agent", "file_write"])
    assert _names(reg) == {"read_file", "web_fetch", "bash"}


def test_allowlist_then_denylist_compose():
    # research-verification semantics: allow read/fetch/bash/serper, deny mutators.
    reg = _registry(
        "read_file",
        "web_fetch",
        "bash",
        "mcp__google_search__search",
        "agent",
        "file_write",
        "file_edit",
        "notebook_edit",
    )
    reg.apply_tool_filter(
        ["read_file", "web_fetch", "bash", "mcp__google_search__search"],
        ["agent", "exit_plan_mode", "file_edit", "file_write", "notebook_edit"],
    )
    assert _names(reg) == {"read_file", "web_fetch", "bash", "mcp__google_search__search"}


def test_mcp_tool_name_survives_allowlist():
    # The Serper MCP tool name (mcp__google_search__search) must match exactly.
    reg = _registry("mcp__google_search__search", "web_fetch", "mcp__other__thing")
    reg.apply_tool_filter(["mcp__google_search__search", "web_fetch"], None)
    assert _names(reg) == {"mcp__google_search__search", "web_fetch"}


def test_deep_research_def_toolset_round_trips_through_filter():
    """The actual deep-research def's tools resolve to themselves under the filter."""
    from openharness.coordinator.agent_definitions import get_agent_definition

    agent = get_agent_definition("deep-research")
    assert agent is not None and agent.tools is not None
    # Registry has the def's tools plus some that must be filtered out.
    reg = _registry(*agent.tools, "file_write", "edit_file", "mcp__other__thing")
    reg.apply_tool_filter(agent.tools, agent.disallowed_tools)
    assert _names(reg) == set(agent.tools)
