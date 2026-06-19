"""Transient eval executor contract and deterministic replay executor."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from openharness.api.client import SupportsStreamingMessages
from openharness.config.settings import PermissionSettings
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import (
    AssistantTurnComplete,
    ErrorEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.evals.facets import EvalTextFacetInput
from openharness.evals.models import EvalEpisode, EvalEvent, EvalRunPack, EvalRunPackCase
from openharness.evals.store import EvalStore
from openharness.permissions.checker import PermissionChecker
from openharness.permissions.modes import PermissionMode
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult


class ReplayToolInput(BaseModel):
    """Permissive replay-tool input model used for captured tool schemas."""

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True)
class EvalToolFixture:
    """Transient replay fixture for one captured tool call."""

    tool_name: str
    call_key_hash: str
    started: bool = False
    completed: bool = False
    is_error: bool = False
    start_event_index: int | None = None
    complete_event_index: int | None = None
    input_text: str = ""
    output_text: str = ""
    input_summary_length: int = 0
    output_summary_length: int = 0


@dataclass(frozen=True)
class EvalExecutionContext:
    """Transient context passed to eval executors.

    This object may contain raw facet and event text. It must never be
    persisted directly; reports are built from explicit metadata-only fields.
    """

    store: EvalStore
    pack: EvalRunPack
    case: EvalRunPackCase
    episode: EvalEpisode
    events: tuple[EvalEvent, ...]
    input_facets: tuple[EvalTextFacetInput, ...]
    expected_facets: tuple[EvalTextFacetInput, ...]
    tool_fixtures: tuple[EvalToolFixture, ...]
    primary_prompt: str
    expected_final_text: str
    resource_snapshot_status: str
    scratch_dir: Path | None = None


@dataclass(frozen=True)
class EvalExecutorResult:
    """Transient result returned by an eval executor."""

    final_text: str = ""
    tool_path: tuple[str, ...] = ()
    event_kind_path: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalAgentRunner(Protocol):
    """Runs an eval prompt against a registry of replay-only tools."""

    name: str

    def run(
        self,
        *,
        prompt: str,
        tool_registry: ToolRegistry,
        context: EvalExecutionContext,
    ) -> EvalExecutorResult:
        """Run one prompt and return transient observed behavior."""


class EvalExecutor(Protocol):
    """Synchronous executor contract for one eval case."""

    name: str

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        """Execute one eval case and return a transient observed result."""


class ReplayToolsExecutor:
    """Executor that runs prompts with replay-only tool fixtures."""

    name = "replay-tools"

    def __init__(self, *, agent_runner: EvalAgentRunner | None = None) -> None:
        self._agent_runner = agent_runner or ReplayScriptAgentRunner()

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        return self._agent_runner.run(
            prompt=context.primary_prompt,
            tool_registry=build_replay_tool_registry(context.tool_fixtures),
            context=context,
        )


class ReplayScriptAgentRunner:
    """Offline runner that replays captured tool calls in order."""

    name = "replay-script"

    def run(
        self,
        *,
        prompt: str,
        tool_registry: ToolRegistry,
        context: EvalExecutionContext,
    ) -> EvalExecutorResult:
        del prompt
        return asyncio.run(_run_scripted_replay(tool_registry, context))


class QueryEngineEvalAgentRunner:
    """Run eval prompts through QueryEngine using replay-only tools."""

    name = "query-engine"

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        model: str,
        system_prompt: str = "You are running a replay-only eval.",
        cwd: str | Path | None = None,
        max_turns: int = 8,
        max_tokens: int = 4096,
    ) -> None:
        self._api_client = api_client
        self._model = model
        self._system_prompt = system_prompt
        self._cwd = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
        self._max_turns = max_turns
        self._max_tokens = max_tokens

    def run(
        self,
        *,
        prompt: str,
        tool_registry: ToolRegistry,
        context: EvalExecutionContext,
    ) -> EvalExecutorResult:
        return asyncio.run(
            _run_query_engine_replay(
                api_client=self._api_client,
                model=self._model,
                system_prompt=self._system_prompt,
                cwd=self._cwd,
                max_turns=self._max_turns,
                max_tokens=self._max_tokens,
                prompt=prompt,
                tool_registry=tool_registry,
                context=context,
            )
        )


class ReplayFixtureTool(BaseTool):
    """Replay-only tool that returns captured outputs and has no side effects."""

    description = "Replay-only eval tool backed by captured outputs."
    input_model = ReplayToolInput

    def __init__(self, *, tool_name: str, fixtures: tuple[EvalToolFixture, ...]) -> None:
        self.name = tool_name
        self._fixtures = fixtures
        self._next_index = 0

    async def execute(
        self,
        arguments: ReplayToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del arguments, context
        if self._next_index >= len(self._fixtures):
            return ToolResult(
                output=f"No replay fixture available for {self.name}.",
                is_error=True,
            )
        fixture = self._fixtures[self._next_index]
        self._next_index += 1
        return ToolResult(
            output=fixture.output_text,
            is_error=fixture.is_error,
            metadata={
                "replayed": True,
                "call_key_hash": fixture.call_key_hash,
            },
        )

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True


def build_replay_tool_registry(fixtures: tuple[EvalToolFixture, ...]) -> ToolRegistry:
    """Build a replay-only registry from captured tool fixtures."""
    registry = ToolRegistry()
    by_name: dict[str, list[EvalToolFixture]] = {}
    for fixture in fixtures:
        by_name.setdefault(fixture.tool_name, []).append(fixture)
    for tool_name, tool_fixtures in by_name.items():
        registry.register(
            ReplayFixtureTool(
                tool_name=tool_name,
                fixtures=tuple(tool_fixtures),
            )
        )
    return registry


async def _run_scripted_replay(
    tool_registry: ToolRegistry,
    context: EvalExecutionContext,
) -> EvalExecutorResult:
    tool_path: list[str] = []
    event_kind_path = ["execution_started"]
    for fixture in context.tool_fixtures:
        tool = tool_registry.get(fixture.tool_name)
        if tool is None:
            event_kind_path.append("tool_missing")
            continue
        event_kind_path.append("tool_started")
        result = await tool.execute(
            ReplayToolInput(),
            ToolExecutionContext(cwd=context.store.root),
        )
        tool_path.append(fixture.tool_name)
        event_kind_path.append("tool_completed_error" if result.is_error else "tool_completed")
    event_kind_path.append("execution_completed")
    return EvalExecutorResult(
        final_text=context.expected_final_text,
        tool_path=tuple(tool_path),
        event_kind_path=tuple(event_kind_path),
        metadata={
            "agent_runner": ReplayScriptAgentRunner.name,
            "fixture_count": len(context.tool_fixtures),
        },
    )


async def _run_query_engine_replay(
    *,
    api_client: SupportsStreamingMessages,
    model: str,
    system_prompt: str,
    cwd: Path,
    max_turns: int,
    max_tokens: int,
    prompt: str,
    tool_registry: ToolRegistry,
    context: EvalExecutionContext,
) -> EvalExecutorResult:
    engine = QueryEngine(
        api_client=api_client,
        tool_registry=tool_registry,
        permission_checker=PermissionChecker(
            PermissionSettings(mode=PermissionMode.FULL_AUTO)
        ),
        cwd=cwd,
        model=model,
        system_prompt=system_prompt,
        max_turns=max_turns,
        max_tokens=max_tokens,
    )
    tool_path: list[str] = []
    event_kind_path: list[str] = ["execution_started"]
    final_text = ""
    async for event in engine.submit_message(prompt):
        if isinstance(event, ToolExecutionStarted):
            tool_path.append(event.tool_name)
            event_kind_path.append("tool_started")
        elif isinstance(event, ToolExecutionCompleted):
            event_kind_path.append(
                "tool_completed_error" if event.is_error else "tool_completed"
            )
        elif isinstance(event, AssistantTurnComplete):
            final_text = event.message.text
            event_kind_path.append("assistant_turn_complete")
        elif isinstance(event, ErrorEvent):
            event_kind_path.append("execution_error")
    event_kind_path.append("execution_completed")
    return EvalExecutorResult(
        final_text=final_text,
        tool_path=tuple(tool_path),
        event_kind_path=tuple(event_kind_path),
        metadata={
            "agent_runner": QueryEngineEvalAgentRunner.name,
            "engine_message_count": len(engine.messages),
            "source_event_count": len(context.events),
        },
    )
