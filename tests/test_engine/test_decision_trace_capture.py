from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.config.settings import PermissionSettings
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.engine.query import DECISION_TRACE_RECORDER_METADATA_KEY
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import AssistantTurnComplete
from openharness.evals import (
    STRUCTURAL_ASSISTANT_FINAL,
    STRUCTURAL_MODEL_CALL,
    STRUCTURAL_TOOL_COMPLETED,
    STRUCTURAL_TOOL_PERMISSION,
    STRUCTURAL_TOOL_STARTED,
    STRUCTURAL_TURN_CONTINUED,
    STRUCTURAL_TURN_STARTED,
    TRACE_DECISION,
    TRACE_FINALIZATION,
    TRACE_MISSING_REQUIRED,
    DecisionTraceRecorder,
    EvalEpisode,
    EvalStore,
)
from openharness.permissions import PermissionChecker, PermissionMode
from openharness.hooks import HookEvent
from openharness.hooks.types import AggregatedHookResult
from openharness.tools import TraceTool
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult


@dataclass
class _FakeResponse:
    message: ConversationMessage
    usage: UsageSnapshot


class _ScriptedApiClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.requests = []
        self._responses = list(responses)

    async def stream_message(self, request):
        self.requests.append(request)
        response = self._responses.pop(0)
        yield ApiMessageCompleteEvent(
            message=response.message,
            usage=response.usage,
        )


class _TraceInput(BaseModel):
    text: str
    path: str = "notes.txt"
    command: str = "trace-tool inspect notes.txt"


class _TraceAwareTool(BaseTool):
    name = "trace_echo"
    description = "Echo input and report whether decision trace metadata is available."
    input_model = _TraceInput

    def __init__(self, expected_recorder: DecisionTraceRecorder) -> None:
        self.expected_recorder = expected_recorder
        self.saw_recorder = False

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        parsed = _TraceInput.model_validate(arguments)
        self.saw_recorder = (
            context.metadata.get(DECISION_TRACE_RECORDER_METADATA_KEY)
            is self.expected_recorder
        )
        return ToolResult(
            output=f"echoed {parsed.text}\nmetadata recorder visible: {self.saw_recorder}",
        )


class _AlwaysSignalDecisionTraceRecorder:
    def __init__(
        self,
        recorder: DecisionTraceRecorder,
        *,
        signals: tuple[str, ...] = ("ohmo_nutrition_request",),
    ) -> None:
        self._recorder = recorder
        self._signals = signals

    def trace_requirement_signals(self, final_text: str) -> tuple[str, ...]:
        del final_text
        return self._signals

    def record(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> object | None:
        return self._recorder.record(
            kind,
            payload,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )

    def record_structural(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> object | None:
        return self._recorder.record_structural(
            kind,
            payload,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )


class _EmptyHookExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[HookEvent, dict[str, object]]] = []

    async def execute(self, event: HookEvent, payload: dict[str, object]) -> AggregatedHookResult:
        self.calls.append((event, payload))
        return AggregatedHookResult([])


def _store_and_recorder(
    tmp_path: Path,
    *,
    episode_id: str = "ep-trace",
    enabled: bool = True,
) -> tuple[EvalStore, DecisionTraceRecorder]:
    store = EvalStore(tmp_path / "evals")
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="test",
            app="openharness",
            session_id="session-1",
            user_text="trace capture",
        )
    )
    return store, DecisionTraceRecorder(store=store, episode_id=episode_id, enabled=enabled)


def _trace_payload(trace_event_id: str, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "trace_event_id": trace_event_id,
    }
    payload.update(overrides)
    return payload


def _engine(
    *,
    tmp_path: Path,
    api_client: _ScriptedApiClient,
    recorder: DecisionTraceRecorder,
    tool_registry: ToolRegistry | None = None,
    hook_executor: Any | None = None,
) -> QueryEngine:
    return QueryEngine(
        api_client=api_client,
        tool_registry=tool_registry or ToolRegistry(),
        permission_checker=PermissionChecker(
            PermissionSettings(mode=PermissionMode.FULL_AUTO)
        ),
        cwd=tmp_path,
        model="trace-model",
        system_prompt="system",
        decision_trace_recorder=recorder,
        hook_executor=hook_executor,
    )


@pytest.mark.asyncio
async def test_tool_turn_records_compact_structural_trace(tmp_path: Path) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    tool = _TraceAwareTool(expected_recorder=recorder)
    registry = ToolRegistry()
    registry.register(tool)
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        TextBlock(text="I will call the tool."),
                        ToolUseBlock(
                            id="tool-call-1",
                            name="trace_echo",
                            input={
                                "text": "alpha beta gamma",
                                "path": "notes.txt",
                                "command": "trace-tool inspect notes.txt",
                            },
                        ),
                    ],
                ),
                usage=UsageSnapshot(input_tokens=11, output_tokens=7),
            ),
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="Done.")],
                ),
                usage=UsageSnapshot(input_tokens=13, output_tokens=5),
            ),
        ]
    )
    engine = _engine(
        tmp_path=tmp_path,
        api_client=api_client,
        recorder=recorder,
        tool_registry=registry,
    )

    _ = [event async for event in engine.submit_message("please use the tool")]

    events = list(store.iter_events("ep-trace"))
    assert [event.kind for event in events] == [
        STRUCTURAL_TURN_STARTED,
        STRUCTURAL_MODEL_CALL,
        STRUCTURAL_TOOL_STARTED,
        STRUCTURAL_TOOL_PERMISSION,
        STRUCTURAL_TOOL_COMPLETED,
        STRUCTURAL_MODEL_CALL,
        STRUCTURAL_ASSISTANT_FINAL,
    ]
    assert {event.episode_id for event in events} == {"ep-trace"}
    assert tool.saw_recorder is True

    first_model_call = events[1]
    assert first_model_call.payload["input_tokens"] == 11
    assert first_model_call.payload["output_tokens"] == 7
    assert first_model_call.payload["tool_use_count"] == 1

    tool_started, tool_permission, tool_completed = events[2:5]
    assert [event.tool_call_id for event in events[2:5]] == [
        "tool-call-1",
        "tool-call-1",
        "tool-call-1",
    ]
    assert [event.tool_name for event in events[2:5]] == [
        "trace_echo",
        "trace_echo",
        "trace_echo",
    ]
    assert tool_started.payload["input_keys"] == ["command", "path", "text"]
    assert "input_summary" in tool_started.payload
    assert "input_length" in tool_started.payload
    assert "input_sha256" in tool_started.payload
    assert tool_permission.payload["allowed"] is True
    assert tool_permission.payload["requires_confirmation"] is False
    assert tool_permission.payload["read_only"] is False
    assert "command_summary" in tool_permission.payload
    assert "path_sha256" in tool_permission.payload
    assert tool_completed.payload["is_error"] is False
    assert isinstance(tool_completed.payload["duration_ms"], (int, float))
    assert tool_completed.payload["duration_ms"] >= 0
    assert "output_summary" in tool_completed.payload
    assert "output_length" in tool_completed.payload
    assert "output_sha256" in tool_completed.payload


@pytest.mark.asyncio
async def test_no_tool_final_answer_records_turn_model_and_final(
    tmp_path: Path,
) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="Final answer.")],
                ),
                usage=UsageSnapshot(input_tokens=3, output_tokens=2),
            )
        ]
    )
    engine = _engine(tmp_path=tmp_path, api_client=api_client, recorder=recorder)

    _ = [event async for event in engine.submit_message("answer directly")]

    events = list(store.iter_events("ep-trace"))
    assert [event.kind for event in events] == [
        STRUCTURAL_TURN_STARTED,
        STRUCTURAL_MODEL_CALL,
        STRUCTURAL_ASSISTANT_FINAL,
    ]
    assert events[0].payload["user_text_summary"] == "answer directly"
    assert events[1].payload["input_tokens"] == 3
    assert events[2].payload["assistant_text_summary"] == "Final answer."


@pytest.mark.asyncio
async def test_disabled_recorder_writes_no_structural_events(tmp_path: Path) -> None:
    store, recorder = _store_and_recorder(tmp_path, enabled=False)
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="Nothing recorded.")],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
        ]
    )
    engine = _engine(tmp_path=tmp_path, api_client=api_client, recorder=recorder)

    _ = [event async for event in engine.submit_message("hello")]

    assert store.count_events("ep-trace") == 0
    assert list(store.iter_events("ep-trace")) == []


@pytest.mark.asyncio
async def test_continue_pending_records_turn_continued(tmp_path: Path) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="Continued final.")],
                ),
                usage=UsageSnapshot(input_tokens=4, output_tokens=3),
            )
        ]
    )
    engine = _engine(tmp_path=tmp_path, api_client=api_client, recorder=recorder)
    engine.load_messages(
        [
            ConversationMessage.from_user_text("run a tool"),
            ConversationMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="tool-call-pending",
                        name="trace_echo",
                        input={"text": "pending"},
                    )
                ],
            ),
            ConversationMessage(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="tool-call-pending",
                        content="pending result",
                    )
                ],
            ),
        ]
    )

    _ = [event async for event in engine.continue_pending()]

    events = list(store.iter_events("ep-trace"))
    assert [event.kind for event in events] == [
        STRUCTURAL_TURN_CONTINUED,
        STRUCTURAL_MODEL_CALL,
        STRUCTURAL_ASSISTANT_FINAL,
    ]
    assert events[0].payload["pending_tool_result_count"] == 1
    assert events[0].payload["pending_tool_result_ids"] == ["tool-call-pending"]


@pytest.mark.asyncio
async def test_trace_required_happy_path_uses_existing_model_trace(
    tmp_path: Path,
) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    registry = ToolRegistry()
    registry.register(TraceTool())
    final_answer = (
        "Based on the available result, the implementation should stay scoped to "
        "the trace recorder path and keep the original final response unchanged."
    )
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="trace-call-1",
                            name="trace",
                            input={
                                "kind": TRACE_FINALIZATION,
                                "payload": _trace_payload(
                                    "trace-1",
                                    answer_claims=[
                                        {
                                            "claim": "scope stays on the trace recorder path",
                                            "supported_by": ["toolu_obs_1"],
                                        }
                                    ],
                                ),
                            },
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=8, output_tokens=4),
            ),
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text=final_answer)],
                ),
                usage=UsageSnapshot(input_tokens=12, output_tokens=7),
            ),
        ]
    )
    engine = _engine(
        tmp_path=tmp_path,
        api_client=api_client,
        recorder=recorder,
        tool_registry=registry,
    )

    _ = [event async for event in engine.submit_message("answer with trace coverage")]

    # A proactive trace_finalization closes the requirement: no repair turn.
    assert len(api_client.requests) == 2
    recorded_events = list(store.iter_events("ep-trace"))
    assert any(event.kind == TRACE_FINALIZATION for event in recorded_events)
    assert not any(event.kind == TRACE_MISSING_REQUIRED for event in recorded_events)


@pytest.mark.asyncio
async def test_proactive_non_finalization_trace_still_triggers_finalization_repair(
    tmp_path: Path,
) -> None:
    # Fix: a proactive trace_decision (or observation) must NOT close the
    # finalization requirement — the repair still fires to demand a
    # trace_finalization that maps the answer to evidence.
    store, recorder = _store_and_recorder(tmp_path)
    registry = ToolRegistry()
    registry.register(TraceTool())
    final_answer = (
        "Based on the available result, the implementation should stay scoped to "
        "the trace recorder path and keep the original final response unchanged."
    )
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="trace-call-1",
                            name="trace",
                            input={
                                "kind": TRACE_DECISION,
                                "payload": _trace_payload(
                                    "trace-1",
                                    decision="read the recorder path before answering",
                                ),
                            },
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=8, output_tokens=4),
            ),
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text=final_answer)],
                ),
                usage=UsageSnapshot(input_tokens=12, output_tokens=7),
            ),
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="repair-trace-1",
                            name="trace",
                            input={
                                "kind": TRACE_FINALIZATION,
                                "payload": _trace_payload(
                                    "repair-trace-1",
                                    reason="substantive final answer",
                                    answer_claims=[
                                        {"claim": "scope stays on recorder path",
                                         "supported_by": ["toolu_obs_1"]}
                                    ],
                                ),
                            },
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=3, output_tokens=2),
            ),
        ]
    )
    engine = _engine(
        tmp_path=tmp_path,
        api_client=api_client,
        recorder=recorder,
        tool_registry=registry,
    )

    _ = [event async for event in engine.submit_message("answer with trace coverage")]

    # 3 requests: decision turn, final-text turn, finalization repair turn.
    assert len(api_client.requests) == 3
    assert [tool["name"] for tool in api_client.requests[2].tools] == ["trace"]
    recorded_events = list(store.iter_events("ep-trace"))
    assert any(event.kind == TRACE_DECISION for event in recorded_events)
    assert any(event.kind == TRACE_FINALIZATION for event in recorded_events)
    assert not any(event.kind == TRACE_MISSING_REQUIRED for event in recorded_events)


@pytest.mark.asyncio
async def test_trace_required_repair_records_trace_without_yielding_extra_turn(
    tmp_path: Path,
) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    registry = ToolRegistry()
    registry.register(TraceTool())
    final_answer = (
        "Based on the repository state, the correct response is to preserve the "
        "user-visible answer while recording a compact finalization breadcrumb."
    )
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text=final_answer)],
                ),
                usage=UsageSnapshot(input_tokens=5, output_tokens=6),
            ),
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="repair-trace-1",
                            name="trace",
                            input={
                                "kind": TRACE_FINALIZATION,
                                "payload": _trace_payload(
                                    "repair-trace-1",
                                    reason="substantive final answer",
                                    final_answer_summary="preserve answer and record breadcrumb",
                                ),
                            },
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=3, output_tokens=2),
            ),
        ]
    )
    engine = _engine(
        tmp_path=tmp_path,
        api_client=api_client,
        recorder=recorder,
        tool_registry=registry,
    )

    stream_events = [event async for event in engine.submit_message("answer directly")]

    assistant_turns = [
        event for event in stream_events if isinstance(event, AssistantTurnComplete)
    ]
    assert [event.message.text for event in assistant_turns] == [final_answer]
    assert len(api_client.requests) == 2
    assert [tool["name"] for tool in api_client.requests[1].tools] == ["trace"]

    recorded_events = list(store.iter_events("ep-trace"))
    assert any(event.kind == TRACE_FINALIZATION for event in recorded_events)
    assert not any(event.kind == TRACE_MISSING_REQUIRED for event in recorded_events)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_empty_hook", [False, True])
async def test_short_answer_trace_repair_is_triggered_by_recorder_signal(
    tmp_path: Path,
    with_empty_hook: bool,
) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    registry = ToolRegistry()
    registry.register(TraceTool())
    hook_executor = _EmptyHookExecutor() if with_empty_hook else None

    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="OK.")],
                ),
                usage=UsageSnapshot(input_tokens=3, output_tokens=1),
            ),
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="repair-trace-1",
                            name="trace",
                            input={
                                "kind": TRACE_FINALIZATION,
                                "payload": _trace_payload(
                                    "repair-trace-1",
                                    reason="ohmo_nutrition_request",
                                    answer_claims=[],
                                ),
                            },
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=4, output_tokens=2),
            ),
        ]
    )
    engine = _engine(
        tmp_path=tmp_path,
        api_client=api_client,
        recorder=_AlwaysSignalDecisionTraceRecorder(recorder),
        tool_registry=registry,
        hook_executor=hook_executor,
    )

    stream_events = [event async for event in engine.submit_message("нежно короткий ответ")]

    assert len(api_client.requests) == 2
    assert [tool["name"] for tool in api_client.requests[1].tools] == ["trace"]
    assert "Signals: ohmo_nutrition_request" in api_client.requests[1].messages[-1].text

    assistant_turns = [
        event for event in stream_events if isinstance(event, AssistantTurnComplete)
    ]
    assert [event.message.text for event in assistant_turns] == ["OK."]

    recorded_events = list(store.iter_events("ep-trace"))
    assert any(event.kind == TRACE_FINALIZATION for event in recorded_events)

    if with_empty_hook:
        assert hook_executor is not None
        assert len(hook_executor.calls) >= 1
        event, payload = next(
            (event_payload for event_payload in hook_executor.calls if event_payload[0] == HookEvent.STOP),
            (None, {}),
        )
        assert event == HookEvent.STOP
        assert payload["stop_reason"] == "tool_uses_empty"


@pytest.mark.asyncio
async def test_short_answer_trace_repair_is_not_triggered_by_empty_recorder_signal(
    tmp_path: Path,
) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="OK.")],
                ),
                usage=UsageSnapshot(input_tokens=3, output_tokens=1),
            )
        ]
    )
    engine = _engine(
        tmp_path=tmp_path,
        api_client=api_client,
        recorder=_AlwaysSignalDecisionTraceRecorder(recorder, signals=()),
        tool_registry=ToolRegistry(),
    )

    _ = [event async for event in engine.submit_message("нужен короткий ответ")]

    assert len(api_client.requests) == 1
    assert not any(
        event.kind == TRACE_MISSING_REQUIRED for event in store.iter_events("ep-trace")
    )


@pytest.mark.asyncio
async def test_trace_required_repair_failure_records_diagnostic(
    tmp_path: Path,
) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    registry = ToolRegistry()
    registry.register(TraceTool())
    final_answer = (
        "Based on the completed checks, the final response should remain visible "
        "even when a hidden repair attempt cannot record the required trace."
    )
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text=final_answer)],
                ),
                usage=UsageSnapshot(input_tokens=5, output_tokens=6),
            ),
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="No trace call was made.")],
                ),
                usage=UsageSnapshot(input_tokens=3, output_tokens=2),
            ),
        ]
    )
    engine = _engine(
        tmp_path=tmp_path,
        api_client=api_client,
        recorder=recorder,
        tool_registry=registry,
    )

    stream_events = [event async for event in engine.submit_message("answer directly")]

    assistant_turns = [
        event for event in stream_events if isinstance(event, AssistantTurnComplete)
    ]
    assert [event.message.text for event in assistant_turns] == [final_answer]
    assert len(api_client.requests) == 2

    diagnostics = [
        event
        for event in store.iter_events("ep-trace")
        if event.kind == TRACE_MISSING_REQUIRED
    ]
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.is_error is True
    assert diagnostic.payload["missing"] == ["model_authored_trace"]
    assert diagnostic.payload["reason"] == "substantive_final_answer"
    assert diagnostic.payload["final_answer_length"] == len(final_answer)
    assert diagnostic.payload["final_answer_summary"]


@pytest.mark.asyncio
async def test_trivial_final_answer_does_not_trigger_trace_repair(
    tmp_path: Path,
) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    registry = ToolRegistry()
    registry.register(TraceTool())
    api_client = _ScriptedApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="OK.")],
                ),
                usage=UsageSnapshot(input_tokens=3, output_tokens=1),
            )
        ]
    )
    engine = _engine(
        tmp_path=tmp_path,
        api_client=api_client,
        recorder=recorder,
        tool_registry=registry,
    )

    _ = [event async for event in engine.submit_message("acknowledge")]

    assert len(api_client.requests) == 1
    assert not any(
        event.kind == TRACE_MISSING_REQUIRED for event in store.iter_events("ep-trace")
    )
