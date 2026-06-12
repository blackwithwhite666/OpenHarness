"""Tool abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from openharness.hooks.executor import HookExecutor


@dataclass
class ToolExecutionContext:
    """Shared execution context for tool invocations."""

    cwd: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    hook_executor: HookExecutor | None = None


@dataclass(frozen=True)
class ToolResult:
    """Normalized tool execution result."""

    output: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """Base class for all OpenHarness tools."""

    name: str
    description: str
    input_model: type[BaseModel]

    @abstractmethod
    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        """Execute the tool."""

    def is_read_only(self, arguments: BaseModel) -> bool:
        """Return whether the invocation is read-only."""
        del arguments
        return False

    def to_api_schema(self) -> dict[str, Any]:
        """Return the tool schema expected by the Anthropic Messages API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


class ToolRegistry:
    """Map tool names to implementations."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Return a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def apply_tool_filter(
        self,
        allowed: list[str] | None = None,
        disallowed: list[str] | None = None,
    ) -> None:
        """Restrict the registry to a sub-agent's partitioned toolset, in place.

        Mirrors the ``AgentDefinition`` semantics:

        * ``allowed`` — allowlist. ``None`` or ``["*"]`` means "all tools" (no
          allowlist restriction). Otherwise only tools whose name is in the list
          survive. Names match exactly, including MCP tool names of the form
          ``mcp__<server>__<tool>`` produced by :class:`McpToolAdapter`.
        * ``disallowed`` — denylist, applied after the allowlist. Any tool whose
          name is in this list is removed.

        Used at the subprocess-teammate boundary so a spawned sub-agent only
        sees the tools its definition declares (the def's ``tools`` /
        ``disallowed_tools`` are otherwise dropped when crossing the worker
        subprocess — see ``swarm/spawn_utils.build_inherited_cli_flags``).
        """
        if allowed is not None and allowed != ["*"]:
            allow_set = set(allowed)
            self._tools = {
                name: tool for name, tool in self._tools.items() if name in allow_set
            }
        if disallowed:
            deny_set = set(disallowed)
            self._tools = {
                name: tool for name, tool in self._tools.items() if name not in deny_set
            }

    def to_api_schema(self) -> list[dict[str, Any]]:
        """Return all tool schemas in API format."""
        return [tool.to_api_schema() for tool in self._tools.values()]
