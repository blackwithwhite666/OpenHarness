"""Transient eval executor contract and deterministic replay executor."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

from openharness.api.client import SupportsStreamingMessages
from openharness.config.settings import PermissionSettings
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.engine.query import MaxTurnsExceeded
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import (
    AssistantTurnComplete,
    ErrorEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.evals.facets import EvalTextFacetInput
from openharness.evals.models import EvalEpisode, EvalEvent, EvalRunPack, EvalRunPackCase
from openharness.evals.replay_matching import _fixture_input_key
from openharness.evals.state import compute_state_delta
from openharness.evals.store import EvalStore
from openharness.permissions.checker import PermissionChecker
from openharness.permissions.modes import PermissionMode
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult

ResultT = TypeVar("ResultT")


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
    input_key: str = ""


@dataclass(frozen=True)
class SynthContext:
    """Auxiliary LLM context for synthesized replay fixture codegen."""

    api_client: SupportsStreamingMessages
    model: str


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
    conversation_history: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EvalObservedCall:
    """One observed tool call from an eval run.

    Transient: carries the raw arguments the run actually passed so trace/
    policy oracles can inspect them. Must never be persisted directly — the
    execution report only keeps sanitized, metadata-only tool labels.
    """

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


@dataclass(frozen=True)
class EvalExecutorResult:
    """Transient result returned by an eval executor."""

    final_text: str = ""
    tool_path: tuple[str, ...] = ()
    event_kind_path: tuple[str, ...] = ()
    tool_calls: tuple[EvalObservedCall, ...] = ()
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

    def __init__(
        self,
        *,
        agent_runner: EvalAgentRunner | None = None,
        match_mode: str = "order",
        synth_context: SynthContext | None = None,
    ) -> None:
        _validate_replay_match_mode(match_mode)
        self._agent_runner = agent_runner or ReplayScriptAgentRunner()
        self._match_mode = match_mode
        self._synth_context = synth_context

    @property
    def fixture_match_mode(self) -> str:
        return self._match_mode

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        return self._agent_runner.run(
            prompt=context.primary_prompt,
            tool_registry=build_replay_tool_registry(
                context.tool_fixtures,
                match_mode=self._match_mode,
                synth_context=self._synth_context,
            ),
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
        return _run_eval_coroutine(_run_scripted_replay(tool_registry, context))


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
        return _run_eval_coroutine(
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


class SandboxMutatingAgentRunner:
    """Run eval prompts with selected real tools bound to a throwaway sandbox."""

    name = "sandbox"

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        model: str,
        sandbox_tool_factory: Callable[[Path], Sequence[BaseTool]],
        sandbox_state_fn: Callable[[Path], dict[str, Any]],
        system_prompt: str = "You are running a sandboxed eval.",
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
        self._sandbox_tool_factory = sandbox_tool_factory
        self._sandbox_state_fn = sandbox_state_fn

    def run(
        self,
        *,
        prompt: str,
        tool_registry: ToolRegistry,
        context: EvalExecutionContext,
    ) -> EvalExecutorResult:
        sandbox = Path(tempfile.mkdtemp(prefix="openharness-eval-sandbox-")).resolve()
        try:
            before = self._sandbox_state_fn(sandbox)
            for tool in self._sandbox_tool_factory(sandbox):
                tool_registry.register(tool)
            result = _run_eval_coroutine(
                _run_query_engine_replay(
                    api_client=self._api_client,
                    model=self._model,
                    system_prompt=self._system_prompt,
                    cwd=sandbox,
                    max_turns=self._max_turns,
                    max_tokens=self._max_tokens,
                    prompt=prompt,
                    tool_registry=tool_registry,
                    context=context,
                )
            )
            after = self._sandbox_state_fn(sandbox)
            delta = compute_state_delta(before, after)
            return EvalExecutorResult(
                final_text=result.final_text,
                tool_path=result.tool_path,
                event_kind_path=result.event_kind_path,
                tool_calls=result.tool_calls,
                metadata={
                    **result.metadata,
                    "agent_runner": self.name,
                    "sandbox_state_delta": delta,
                },
            )
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)


class ReplayFixtureTool(BaseTool):
    """Replay-only tool; exact arg matching may become a tunable policy later."""

    description = "Replay-only eval tool backed by captured outputs."
    input_model = ReplayToolInput

    def __init__(
        self,
        *,
        tool_name: str,
        fixtures: tuple[EvalToolFixture, ...],
        match_mode: str = "order",
    ) -> None:
        _validate_replay_match_mode(match_mode)
        self.name = tool_name
        self._fixtures = fixtures
        self._match_mode = match_mode
        self._next_index = 0
        self._used_indexes: set[int] = set()

    async def execute(
        self,
        arguments: ReplayToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if self._match_mode == "order":
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

        del context
        requested_key = _fixture_input_key(arguments.model_dump())
        for index, fixture in enumerate(self._fixtures):
            if index in self._used_indexes or fixture.input_key != requested_key:
                continue
            self._used_indexes.add(index)
            return ToolResult(
                output=fixture.output_text,
                is_error=fixture.is_error,
                metadata={
                    "replayed": True,
                    "match": "arguments",
                    "call_key_hash": fixture.call_key_hash,
                },
            )
        return ToolResult(
            output=f"No replay fixture for {self.name} with these arguments.",
            is_error=True,
            metadata={
                "replayed": False,
                "match": "miss",
                "requested_key": requested_key,
            },
        )

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True


def build_replay_tool_registry(
    fixtures: tuple[EvalToolFixture, ...],
    *,
    match_mode: str = "order",
    synth_context: SynthContext | None = None,
) -> ToolRegistry:
    """Build a replay-only registry from captured tool fixtures."""
    _validate_replay_match_mode(match_mode)
    if match_mode == "synth" and synth_context is None:
        raise ValueError("synth fixture match requires a SynthContext")
    registry = ToolRegistry()
    by_name: dict[str, list[EvalToolFixture]] = {}
    for fixture in fixtures:
        by_name.setdefault(fixture.tool_name, []).append(fixture)
    for tool_name, tool_fixtures in by_name.items():
        fixtures_tuple = tuple(tool_fixtures)
        if match_mode == "synth":
            assert synth_context is not None
            from openharness.evals.synth_fixture import (  # noqa: PLC0415
                SynthesizedFixtureTool,
            )

            registry.register(
                SynthesizedFixtureTool(
                    tool_name=tool_name,
                    fixtures=fixtures_tuple,
                    api_client=synth_context.api_client,
                    model=synth_context.model,
                    fallback_match_mode="order",
                )
            )
        else:
            registry.register(
                ReplayFixtureTool(
                    tool_name=tool_name,
                    fixtures=fixtures_tuple,
                    match_mode=match_mode,
                )
            )
    return registry


def _validate_replay_match_mode(match_mode: str) -> None:
    if match_mode not in {"order", "arguments", "synth"}:
        raise ValueError("fixture match mode must be one of: order, arguments, synth")


def _run_eval_coroutine(
    coro: Coroutine[Any, Any, ResultT],
) -> ResultT:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, ResultT] = {}
    errors: list[BaseException] = []

    def run_in_thread() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=run_in_thread,
        name="openharness-eval-runner",
        daemon=True,
    )
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result["value"]


async def _run_scripted_replay(
    tool_registry: ToolRegistry,
    context: EvalExecutionContext,
) -> EvalExecutorResult:
    tool_path: list[str] = []
    observed_calls: list[EvalObservedCall] = []
    event_kind_path = ["execution_started"]
    for fixture in context.tool_fixtures:
        tool = tool_registry.get(fixture.tool_name)
        if tool is None:
            event_kind_path.append("tool_missing")
            continue
        event_kind_path.append("tool_started")
        result = await tool.execute(
            ReplayToolInput.model_validate(_fixture_arguments(context, fixture)),
            ToolExecutionContext(cwd=context.store.root),
        )
        tool_path.append(fixture.tool_name)
        observed_calls.append(
            EvalObservedCall(
                tool_name=fixture.tool_name,
                arguments=_fixture_arguments(context, fixture),
                is_error=result.is_error,
            )
        )
        event_kind_path.append("tool_completed_error" if result.is_error else "tool_completed")
    event_kind_path.append("execution_completed")
    return EvalExecutorResult(
        final_text=context.expected_final_text,
        tool_path=tuple(tool_path),
        event_kind_path=tuple(event_kind_path),
        tool_calls=tuple(observed_calls),
        metadata={
            "agent_runner": ReplayScriptAgentRunner.name,
            "fixture_count": len(context.tool_fixtures),
        },
    )


def _fixture_arguments(
    context: EvalExecutionContext, fixture: EvalToolFixture
) -> dict[str, Any]:
    """Recover the structured tool input recorded for a replayed fixture.

    For the scripted runner the observed call equals the captured one, so the
    arguments come from the originating ``tool_started`` event payload.
    """
    idx = fixture.start_event_index
    if idx is None or idx < 0 or idx >= len(context.events):
        return {}
    payload = context.events[idx].payload or {}
    arguments = payload.get("input")
    return dict(arguments) if isinstance(arguments, dict) else {}


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
    conversation_history = getattr(context, "conversation_history", ())
    if conversation_history:
        engine.load_messages(
            [
                _conversation_history_message(role, text)
                for role, text in conversation_history
            ]
        )
    tool_path: list[str] = []
    observed_calls: list[dict[str, Any]] = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    event_kind_path: list[str] = ["execution_started"]
    final_text = ""
    max_turns_exceeded = False
    try:
        async for event in engine.submit_message(prompt):
            if isinstance(event, ToolExecutionStarted):
                tool_path.append(event.tool_name)
                entry = {
                    "tool_name": event.tool_name,
                    "arguments": dict(event.tool_input or {}),
                    "is_error": False,
                }
                observed_calls.append(entry)
                if event.tool_call_id:
                    calls_by_id[event.tool_call_id] = entry
                event_kind_path.append("tool_started")
            elif isinstance(event, ToolExecutionCompleted):
                entry = calls_by_id.get(event.tool_call_id)
                if entry is not None:
                    entry["is_error"] = event.is_error
                event_kind_path.append(
                    "tool_completed_error" if event.is_error else "tool_completed"
                )
            elif isinstance(event, AssistantTurnComplete):
                final_text = event.message.text
                event_kind_path.append("assistant_turn_complete")
            elif isinstance(event, ErrorEvent):
                event_kind_path.append("execution_error")
    except MaxTurnsExceeded:
        max_turns_exceeded = True
        final_text = ""
        event_kind_path.append("max_turns_exceeded")
    event_kind_path.append("execution_completed")
    metadata = {
        "agent_runner": QueryEngineEvalAgentRunner.name,
        "engine_message_count": len(engine.messages),
        "source_event_count": len(context.events),
        "seeded_history_message_count": len(conversation_history),
    }
    if max_turns_exceeded:
        metadata["max_turns_exceeded"] = True
    return EvalExecutorResult(
        final_text=final_text,
        tool_path=tuple(tool_path),
        event_kind_path=tuple(event_kind_path),
        tool_calls=tuple(EvalObservedCall(**entry) for entry in observed_calls),
        metadata=metadata,
    )


def _conversation_history_message(role: str, text: str) -> ConversationMessage:
    if role == "user":
        return ConversationMessage.from_user_text(text)
    if role == "assistant":
        return ConversationMessage(role="assistant", content=[TextBlock(text=text)])
    raise ValueError(f"unsupported conversation history role: {role}")
