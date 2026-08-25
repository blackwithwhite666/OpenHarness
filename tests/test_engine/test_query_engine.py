"""Tests for the query engine."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from openharness.api.client import ApiMessageCompleteEvent, ApiRetryEvent, ApiTextDeltaEvent
from openharness.api.errors import RequestFailure
from openharness.api.usage import UsageSnapshot
from openharness.config.settings import PermissionSettings, Settings
from openharness.engine.messages import (
    AttachmentRefBlock,
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.engine.query import (
    MaxTurnsExceeded,
    QueryContext,
    _execute_tool_call,
    _is_prompt_too_long_error,
)
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    CompactProgressEvent,
    ErrorEvent,
    StatusEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.hooks import HookEvent, HookExecutionContext, HookExecutor
from openharness.hooks.loader import HookRegistry
from openharness.hooks.schemas import PromptHookDefinition
from openharness.permissions import PermissionChecker, PermissionMode
from openharness.prompts.context import build_runtime_system_prompt
from openharness.tasks import get_task_manager
from openharness.tools import create_default_tool_registry
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from openharness.tools.glob_tool import GlobTool
from openharness.tools.grep_tool import GrepTool
from openharness.tools.image_to_text_tool import ImageToTextTool


@dataclass
class _FakeResponse:
    message: ConversationMessage
    usage: UsageSnapshot


class FakeApiClient:
    """Deterministic streaming client used by query tests."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)

    async def stream_message(self, request):
        del request
        response = self._responses.pop(0)
        for block in response.message.content:
            if isinstance(block, TextBlock) and block.text:
                yield ApiTextDeltaEvent(text=block.text)
        yield ApiMessageCompleteEvent(
            message=response.message,
            usage=response.usage,
            stop_reason=None,
        )


class StaticApiClient:
    """Fake client that always returns one fixed assistant message."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def stream_message(self, request):
        del request
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=self._text)]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason=None,
        )


class RetryThenSuccessApiClient:
    async def stream_message(self, request):
        del request
        yield ApiRetryEvent(message="rate limited", attempt=1, max_attempts=4, delay_seconds=1.5)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="after retry")]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason=None,
        )


class PromptTooLongThenSuccessApiClient:
    def __init__(self) -> None:
        self._calls = 0

    async def stream_message(self, request):
        self._calls += 1
        if self._calls == 1:
            raise RequestFailure("prompt too long")
        if self._calls == 2:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(role="assistant", content=[TextBlock(text="<summary>compressed</summary>")]),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason=None,
            )
            return
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="after reactive compact")]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason=None,
        )


class RecordingApiClient:
    def __init__(self, text: str = "ok") -> None:
        self.requests = []
        self._text = text

    async def stream_message(self, request):
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=self._text)]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason=None,
        )


class FailingApiClient:
    async def stream_message(self, request):
        del request
        raise RuntimeError("provider failed")
        yield  # pragma: no cover


def _externalize_test_images(
    messages: list[ConversationMessage],
) -> list[ConversationMessage]:
    durable: list[ConversationMessage] = []
    for message in messages:
        blocks = [
            AttachmentRefBlock(
                attachment_id="a" * 64,
                media_type=block.media_type,
                byte_size=3,
                label="test.png",
            )
            if isinstance(block, ImageBlock)
            else block
            for block in message.content
        ]
        durable.append(message.model_copy(update={"content": blocks}))
    return durable


@pytest.mark.asyncio
async def test_query_engine_same_event_id_retries_without_duplicate_user_turn(
    tmp_path: Path,
) -> None:
    client = RecordingApiClient()
    engine = QueryEngine(
        api_client=client,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        supports_native_images=True,
        durable_message_transform=_externalize_test_images,
    )
    inbound = ConversationMessage(
        role="user",
        event_id="event-123",
        content=[TextBlock(text="meal"), ImageBlock(media_type="image/png", data="YWJj")],
    )

    for _ in range(5):
        _ = [event async for event in engine.submit_message(inbound)]

    users = [message for message in engine.messages if message.role == "user"]
    assert len(users) == 1
    assert users[0].event_id == "event-123"
    assert isinstance(users[0].content[-1], AttachmentRefBlock)
    assert all(
        any(isinstance(block, ImageBlock) for block in request.messages[0].content)
        for request in client.requests
    )


@pytest.mark.asyncio
async def test_query_engine_text_turn_has_ref_but_zero_historical_images(
    tmp_path: Path,
) -> None:
    client = RecordingApiClient()
    engine = QueryEngine(
        api_client=client,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        supports_native_images=True,
        durable_message_transform=_externalize_test_images,
    )

    _ = [
        event
        async for event in engine.submit_message(
            ConversationMessage(
                role="user",
                event_id="image-event",
                content=[ImageBlock(media_type="image/png", data="YWJj")],
            )
        )
    ]
    _ = [
        event
        async for event in engine.submit_message(
            ConversationMessage.from_user_text("what next?").model_copy(
                update={"event_id": "text-event"}
            )
        )
    ]

    second_request = client.requests[1]
    assert all(
        not isinstance(block, ImageBlock)
        for message in second_request.messages
        for block in message.content
    )
    assert any(
        isinstance(block, AttachmentRefBlock)
        for message in second_request.messages
        for block in message.content
    )


@pytest.mark.asyncio
async def test_query_engine_error_path_externalizes_active_image(tmp_path: Path) -> None:
    engine = QueryEngine(
        api_client=FailingApiClient(),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        supports_native_images=True,
        durable_message_transform=_externalize_test_images,
    )

    events = [
        event
        async for event in engine.submit_message(
            ConversationMessage(
                role="user",
                event_id="failed-image-event",
                content=[ImageBlock(media_type="image/png", data="YWJj")],
            )
        )
    ]

    assert any(isinstance(event, ErrorEvent) for event in events)
    assert all(
        not isinstance(block, ImageBlock)
        for message in engine.messages
        for block in message.content
    )
    assert isinstance(engine.messages[0].content[0], AttachmentRefBlock)


@pytest.mark.asyncio
async def test_query_engine_rejects_stale_event_retry_after_later_user_turn(
    tmp_path: Path,
) -> None:
    engine = QueryEngine(
        api_client=RecordingApiClient(),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    _ = [
        event
        async for event in engine.submit_message(
            ConversationMessage.from_user_text("first").model_copy(update={"event_id": "one"})
        )
    ]
    _ = [
        event
        async for event in engine.submit_message(
            ConversationMessage.from_user_text("second").model_copy(update={"event_id": "two"})
        )
    ]

    with pytest.raises(ValueError, match="stale event_id"):
        _ = [
            event
            async for event in engine.submit_message(
                ConversationMessage.from_user_text("first retry").model_copy(
                    update={"event_id": "one"}
                )
            )
        ]


class MaxTokensTooLargeThenSuccessApiClient:
    def __init__(self) -> None:
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            raise RequestFailure(
                "max_tokens is too large: 120000. This model supports at most "
                "32000 completion tokens, whereas you provided 120000."
            )
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="after token clamp")]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason=None,
        )


class EmptyAssistantApiClient:
    async def stream_message(self, request):
        del request
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason=None,
        )


class CoordinatorLoopApiClient:
    def __init__(self) -> None:
        self.requests = []
        self._calls = 0

    async def stream_message(self, request):
        self.requests.append(request)
        self._calls += 1
        if self._calls == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        TextBlock(text="Launching a worker."),
                        ToolUseBlock(
                            id="toolu_agent_1",
                            name="agent",
                            input={
                                "description": "inspect coordinator wiring",
                                "prompt": "check whether coordinator mode is active",
                                "subagent_type": "worker",
                                "mode": "in_process_teammate",
                            },
                        ),
                    ],
                ),
                usage=UsageSnapshot(input_tokens=2, output_tokens=2),
                stop_reason=None,
            )
            return
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="Worker launched; coordinator mode is active.")]),
            usage=UsageSnapshot(input_tokens=2, output_tokens=2),
            stop_reason=None,
        )


class _NoopApiClient:
    async def stream_message(self, request):
        del request
        if False:
            yield None


def test_query_prompt_too_long_detection_handles_llama_cpp_errors():
    assert _is_prompt_too_long_error(
        RequestFailure("exceed_context_size_error: prompt exceeds the available context size")
    )


def test_query_prompt_too_long_detection_handles_openai_context_length_errors():
    assert _is_prompt_too_long_error(
        RequestFailure(
            "Input tokens exceed the configured limit of 922000 tokens. "
            "Your messages resulted in 3591869 tokens. Please reduce the length of the messages. "
            "code='context_length_exceeded'"
        )
    )


@pytest.mark.asyncio
async def test_query_engine_plain_text_reply(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="Hello from the model.")],
                    ),
                    usage=UsageSnapshot(input_tokens=10, output_tokens=5),
                )
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )

    events = [event async for event in engine.submit_message("hello")]

    assert isinstance(events[0], AssistantTextDelta)
    assert events[0].text == "Hello from the model."
    assert isinstance(events[-1], AssistantTurnComplete)
    assert engine.total_usage.input_tokens == 10
    assert engine.total_usage.output_tokens == 5
    assert len(engine.messages) == 2


@pytest.mark.asyncio
async def test_query_engine_internal_message_is_not_retained_as_external_user_turn(
    tmp_path: Path,
):
    engine = QueryEngine(
        api_client=StaticApiClient("accepted internal result"),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    engine.load_messages([ConversationMessage.from_user_text("original user request")])

    events = [
        event
        async for event in engine.submit_internal_message(
            "Internal reconciliation instruction that must not be external"
        )
    ]

    assert isinstance(events[-1], AssistantTurnComplete)
    assert [message.text for message in engine.messages] == [
        "original user request",
        "accepted internal result",
    ]


@pytest.mark.asyncio
async def test_query_engine_internal_message_keeps_provider_valid_tool_trace(
    tmp_path: Path,
):
    registry = ToolRegistry()
    registry.register(_OkTool())
    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(id="toolu_internal", name="ok_tool", input={}),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="accepted reconciliation")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    engine.load_messages([ConversationMessage.from_user_text("original user request")])

    events = [event async for event in engine.submit_internal_message("private reconciliation")]

    assert isinstance(events[-1], AssistantTurnComplete)
    assert all("private reconciliation" not in message.text for message in engine.messages)
    assert [message.role for message in engine.messages] == ["user", "assistant", "user", "assistant"]
    assert isinstance(engine.messages[1].content[0], ToolUseBlock)
    assert isinstance(engine.messages[2].content[0], ToolResultBlock)
    assert engine.messages[-1].text == "accepted reconciliation"


@pytest.mark.asyncio
async def test_query_engine_internal_message_removes_rebuilt_prompt_after_history_replacement(
    tmp_path: Path, monkeypatch
):
    engine = QueryEngine(
        api_client=StaticApiClient("unused"),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    engine.load_messages([ConversationMessage.from_user_text("original request")])

    async def fake_run_query(_context, messages):
        messages[:] = [
            ConversationMessage.from_user_text("[compact boundary]"),
            ConversationMessage.from_user_text("private reconciliation"),
            ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(id="toolu_rebuilt", name="ok_tool", input={})],
            ),
            ConversationMessage(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="toolu_rebuilt", content="accepted tool result")
                ],
            ),
            ConversationMessage(
                role="assistant", content=[TextBlock(text="accepted after replacement")]
            ),
        ]
        yield AssistantTurnComplete(
            message=messages[-3], usage=UsageSnapshot(input_tokens=1, output_tokens=1)
        ), UsageSnapshot(input_tokens=1, output_tokens=1)
        yield AssistantTurnComplete(
            message=messages[-1], usage=UsageSnapshot(input_tokens=1, output_tokens=1)
        ), UsageSnapshot(input_tokens=1, output_tokens=1)

    monkeypatch.setattr("openharness.engine.query_engine.run_query", fake_run_query)
    events = [event async for event in engine.submit_internal_message("private reconciliation")]

    assert isinstance(events[-1], AssistantTurnComplete)
    assert [message.text for message in engine.messages] == [
        "[compact boundary]",
        "",
        "",
        "accepted after replacement",
    ]
    assert isinstance(engine.messages[1].content[0], ToolUseBlock)
    assert isinstance(engine.messages[2].content[0], ToolResultBlock)
    assert all(message.text != "private reconciliation" for message in engine.messages)


@pytest.mark.asyncio
async def test_query_engine_internal_message_cancellation_restores_base_history(
    tmp_path: Path,
):
    registry = ToolRegistry()
    registry.register(_OkTool())
    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[ToolUseBlock(id="toolu_cancel", name="ok_tool", input={})],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant", content=[TextBlock(text="hidden final")]
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    base = ConversationMessage.from_user_text("original user request")
    engine.load_messages([base])
    stream = engine.submit_internal_message("temporary instruction")

    while True:
        event = await anext(stream)
        if isinstance(event, AssistantTurnComplete):
            break
    await stream.aclose()

    assert engine.messages == [base]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["error", "max_turns"])
async def test_query_engine_internal_message_failure_restores_exact_base_history(
    tmp_path: Path, monkeypatch, outcome: str
):
    engine = QueryEngine(
        api_client=StaticApiClient("unused"),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    base = ConversationMessage.from_user_text("original user request")
    engine.load_messages([base])

    async def fake_run_query(_context, _messages):
        if outcome == "max_turns":
            raise MaxTurnsExceeded(1)
        yield ErrorEvent(message="provider failed"), None

    monkeypatch.setattr("openharness.engine.query_engine.run_query", fake_run_query)
    if outcome == "max_turns":
        with pytest.raises(MaxTurnsExceeded):
            _ = [event async for event in engine.submit_internal_message("temporary instruction")]
    else:
        _ = [event async for event in engine.submit_internal_message("temporary instruction")]

    assert engine.messages == [base]


@pytest.mark.asyncio
async def test_run_query_surfaces_quota_exceeded_as_explicit_error(tmp_path: Path):
    # A provider quota error must surface as a distinct, greppable
    # "Provider quota exceeded: ..." ErrorEvent instead of the generic
    # "API error: ..." dump.
    from openharness.api.errors import QuotaExceededError
    from openharness.engine.query import QueryContext, run_query

    class _QuotaExceededApiClient:
        async def stream_message(self, request):
            raise QuotaExceededError(
                "You've reached your usage limit for this billing cycle."
            )
            yield  # unreachable: marks this function as an async generator

    ctx = QueryContext(
        api_client=_QuotaExceededApiClient(),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="k3",
        system_prompt="system",
        max_tokens=64,
    )
    messages = [ConversationMessage.from_user_text("hi")]

    events = [event async for event, _usage in run_query(ctx, messages)]

    error_events = [event for event in events if isinstance(event, ErrorEvent)]
    assert error_events, "expected an ErrorEvent for the quota failure"
    assert error_events[0].message.startswith("Provider quota exceeded: ")
    assert "usage limit" in error_events[0].message


@pytest.mark.asyncio
async def test_query_engine_passes_native_images_without_fallback_tool_schema(
    tmp_path: Path,
) -> None:
    client = RecordingApiClient()
    engine = QueryEngine(
        api_client=client,
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="system",
    )
    prompt = ConversationMessage(
        role="user",
        content=[
            TextBlock(text="Inspect this image"),
            ImageBlock(media_type="image/png", data="YWJj"),
        ],
    )

    events = [event async for event in engine.submit_message(prompt)]

    assert isinstance(events[-1], AssistantTurnComplete)
    request = client.requests[0]
    assert any(isinstance(block, ImageBlock) for block in request.messages[0].content)
    assert "image_to_text" not in {schema["name"] for schema in request.tools}


@pytest.mark.asyncio
async def test_query_engine_converts_images_internally_for_text_only_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute(self, arguments, context):
        del self, arguments, context
        return ToolResult(output="internal image description")

    monkeypatch.setattr(ImageToTextTool, "execute", fake_execute)
    client = RecordingApiClient()
    engine = QueryEngine(
        api_client=client,
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="system",
        supports_native_images=False,
        tool_metadata={
            "vision_model_config": {
                "model": "vision-model",
                "api_key": "vision-key",
            }
        },
    )
    prompt = ConversationMessage(
        role="user",
        content=[ImageBlock(media_type="image/png", data="YWJj")],
    )

    events = [event async for event in engine.submit_message(prompt)]

    assert any(isinstance(event, StatusEvent) for event in events)
    assert isinstance(events[-1], AssistantTurnComplete)
    request = client.requests[0]
    assert not any(
        isinstance(block, ImageBlock)
        for message in request.messages
        for block in message.content
    )
    assert any(
        isinstance(block, TextBlock) and "internal image description" in block.text
        for message in request.messages
        for block in message.content
    )
    assert "image_to_text" not in {schema["name"] for schema in request.tools}


@pytest.mark.asyncio
async def test_query_engine_rejects_unhandled_images_before_provider_call(
    tmp_path: Path,
) -> None:
    client = RecordingApiClient()
    engine = QueryEngine(
        api_client=client,
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="text-only-model",
        system_prompt="system",
        supports_native_images=False,
        tool_metadata={"vision_model_config": {"model": "vision-model"}},
    )
    prompt = ConversationMessage(
        role="user",
        content=[ImageBlock(media_type="image/png", data="YWJj")],
    )

    events = [event async for event in engine.submit_message(prompt)]

    assert client.requests == []
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].recoverable is False
    assert "cannot process image input" in events[0].message
    assert "vision" in events[0].message


@pytest.mark.asyncio
async def test_query_engine_clamps_oversized_max_tokens_before_request(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    client = RecordingApiClient()
    engine = QueryEngine(
        api_client=client,
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="openai-compatible-model",
        system_prompt="system",
        max_tokens=400_000,
    )

    events = [event async for event in engine.submit_message("hello")]

    assert client.requests[0].max_tokens == 128_000
    assert any(isinstance(event, StatusEvent) and "safe per-request output cap" in event.message for event in events)
    assert isinstance(events[-1], AssistantTurnComplete)


@pytest.mark.asyncio
async def test_query_engine_retries_with_provider_completion_token_limit(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    client = MaxTokensTooLargeThenSuccessApiClient()
    engine = QueryEngine(
        api_client=client,
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="openai-compatible-model",
        system_prompt="system",
        max_tokens=120_000,
        max_turns=1,
    )

    events = [event async for event in engine.submit_message("hello")]

    assert [request.max_tokens for request in client.requests] == [120_000, 32_000]
    assert any(isinstance(event, StatusEvent) and "provider limit 32000" in event.message for event in events)
    assert isinstance(events[-1], AssistantTurnComplete)


@pytest.mark.asyncio
async def test_query_engine_executes_tool_calls(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    sample = tmp_path / "hello.txt"
    sample.write_text("alpha\nbeta\n", encoding="utf-8")

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            TextBlock(text="I will inspect the file."),
                            ToolUseBlock(
                                id="toolu_123",
                                name="read_file",
                                input={"path": str(sample), "offset": 0, "limit": 2},
                            ),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=4, output_tokens=3),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="The file contains alpha and beta.")],
                    ),
                    usage=UsageSnapshot(input_tokens=8, output_tokens=6),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )

    events = [event async for event in engine.submit_message("read the file")]

    assert any(isinstance(event, ToolExecutionStarted) for event in events)
    tool_results = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    assert len(tool_results) == 1
    assert "alpha" in tool_results[0].output
    assert isinstance(events[-1], AssistantTurnComplete)
    assert "alpha and beta" in events[-1].message.text
    assert len(engine.messages) == 4


@pytest.mark.asyncio
async def test_query_engine_coordinator_mode_uses_coordinator_prompt_and_runs_agent_loop(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLAUDE_CODE_COORDINATOR_MODE", "1")

    api_client = CoordinatorLoopApiClient()
    system_prompt = build_runtime_system_prompt(Settings(), cwd=tmp_path, latest_user_prompt="investigate issue")
    engine = QueryEngine(
        api_client=api_client,
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings()),
        cwd=tmp_path,
        model="claude-test",
        system_prompt=system_prompt,
    )

    events = [event async for event in engine.submit_message("investigate issue")]

    assert len(api_client.requests) == 2
    assert "You are a **coordinator**." in api_client.requests[0].system_prompt
    assert "Coordinator User Context" not in api_client.requests[0].system_prompt
    coordinator_context_messages = [
        msg for msg in api_client.requests[0].messages if msg.role == "user" and "Coordinator User Context" in msg.text
    ]
    assert len(coordinator_context_messages) == 1
    assert "Workers spawned via the agent tool have access to these tools" in coordinator_context_messages[0].text
    assert any(isinstance(event, ToolExecutionStarted) and event.tool_name == "agent" for event in events)
    agent_results = [event for event in events if isinstance(event, ToolExecutionCompleted) and event.tool_name == "agent"]
    assert len(agent_results) == 1
    assert isinstance(events[-1], AssistantTurnComplete)
    assert "coordinator mode is active" in events[-1].message.text


@pytest.mark.asyncio
async def test_query_engine_allows_unbounded_turns_when_max_turns_is_none(tmp_path: Path):
    sample = tmp_path / "hello.txt"
    sample.write_text("alpha\nbeta\n", encoding="utf-8")

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            TextBlock(text="I will inspect the file."),
                            ToolUseBlock(
                                id="toolu_123",
                                name="read_file",
                                input={"path": str(sample), "offset": 0, "limit": 2},
                            ),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=4, output_tokens=3),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="The file contains alpha and beta.")],
                    ),
                    usage=UsageSnapshot(input_tokens=8, output_tokens=6),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        max_turns=None,
    )

    events = [event async for event in engine.submit_message("read the file")]

    assert isinstance(events[-1], AssistantTurnComplete)
    assert "alpha and beta" in events[-1].message.text
    assert engine.max_turns is None


@pytest.mark.asyncio
async def test_query_engine_surfaces_retry_status_events(tmp_path: Path):
    engine = QueryEngine(
        api_client=RetryThenSuccessApiClient(),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings()),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )

    events = [event async for event in engine.submit_message("hello")]

    assert any(isinstance(event, StatusEvent) and "retrying in 1.5s" in event.message for event in events)
    assert isinstance(events[-1], AssistantTurnComplete)


@pytest.mark.asyncio
async def test_query_engine_emits_compact_progress_before_reply(tmp_path: Path, monkeypatch):
    long_text = "alpha " * 50000
    monkeypatch.setattr("openharness.services.compact.try_session_memory_compaction", lambda *args, **kwargs: None)
    monkeypatch.setattr("openharness.services.compact.should_autocompact", lambda *args, **kwargs: True)
    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(role="assistant", content=[TextBlock(text="<summary>trimmed</summary>")]),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(role="assistant", content=[TextBlock(text="after compact")]),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings()),
        cwd=tmp_path,
        model="claude-sonnet-4-6",
        system_prompt="system",
    )
    engine.load_messages(
        [
            ConversationMessage(role="user", content=[TextBlock(text=long_text)]),
            ConversationMessage(role="assistant", content=[TextBlock(text=long_text)]),
            ConversationMessage(role="user", content=[TextBlock(text=long_text)]),
            ConversationMessage(role="assistant", content=[TextBlock(text=long_text)]),
            ConversationMessage(role="user", content=[TextBlock(text=long_text)]),
            ConversationMessage(role="assistant", content=[TextBlock(text=long_text)]),
            ConversationMessage(role="user", content=[TextBlock(text=long_text)]),
            ConversationMessage(role="assistant", content=[TextBlock(text=long_text)]),
        ]
    )

    events = [event async for event in engine.submit_message("hello")]

    hooks_start_index = next(i for i, event in enumerate(events) if isinstance(event, CompactProgressEvent) and event.phase == "hooks_start")
    compact_start_index = next(i for i, event in enumerate(events) if isinstance(event, CompactProgressEvent) and event.phase == "compact_start")
    final_index = next(i for i, event in enumerate(events) if isinstance(event, AssistantTurnComplete))
    assert hooks_start_index < compact_start_index
    assert compact_start_index < final_index
    assert any(isinstance(event, CompactProgressEvent) and event.phase == "compact_end" for event in events)


@pytest.mark.asyncio
async def test_query_engine_reactive_compacts_after_prompt_too_long(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("openharness.services.compact.try_session_memory_compaction", lambda *args, **kwargs: None)
    monkeypatch.setattr("openharness.services.compact.should_autocompact", lambda *args, **kwargs: False)
    engine = QueryEngine(
        api_client=PromptTooLongThenSuccessApiClient(),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings()),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    engine.load_messages(
        [
            ConversationMessage(role="user", content=[TextBlock(text="one")]),
            ConversationMessage(role="assistant", content=[TextBlock(text="two")]),
            ConversationMessage(role="user", content=[TextBlock(text="three")]),
            ConversationMessage(role="assistant", content=[TextBlock(text="four")]),
            ConversationMessage(role="user", content=[TextBlock(text="five")]),
            ConversationMessage(role="assistant", content=[TextBlock(text="six")]),
            ConversationMessage(role="user", content=[TextBlock(text="seven")]),
            ConversationMessage(role="assistant", content=[TextBlock(text="eight")]),
        ]
    )

    events = [event async for event in engine.submit_message("nine")]

    assert any(
        isinstance(event, CompactProgressEvent)
        and event.trigger == "reactive"
        and event.phase == "compact_start"
        for event in events
    )
    assert isinstance(events[-1], AssistantTurnComplete)
    assert events[-1].message.text == "after reactive compact"


@pytest.mark.asyncio
async def test_query_engine_tracks_recent_read_files_and_skills(tmp_path: Path):
    sample = tmp_path / "hello.txt"
    sample.write_text("alpha\nbeta\n", encoding="utf-8")
    registry = create_default_tool_registry()
    skill_tool = registry.get("skill")
    assert skill_tool is not None

    async def _fake_skill_execute(arguments, context):
        del context
        return ToolResult(output=f"Loaded skill: {arguments.name}")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(skill_tool, "execute", _fake_skill_execute)

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(name="read_file", input={"path": str(sample)}),
                            ToolUseBlock(name="skill", input={"name": "demo-skill"}),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings()),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        tool_metadata={},
    )

    try:
        events = [event async for event in engine.submit_message("track context")]
    finally:
        monkeypatch.undo()

    assert isinstance(events[-1], AssistantTurnComplete)
    read_state = engine._tool_metadata.get("read_file_state")
    assert isinstance(read_state, list) and read_state
    assert read_state[-1]["path"] == str(sample.resolve())
    assert "alpha" in read_state[-1]["preview"]
    task_focus = engine.tool_metadata.get("task_focus_state")
    assert isinstance(task_focus, dict)
    assert "track context" in task_focus.get("goal", "")
    assert str(sample.resolve()) in task_focus.get("active_artifacts", [])
    invoked_skills = engine._tool_metadata.get("invoked_skills")
    assert isinstance(invoked_skills, list)
    assert invoked_skills[-1] == "demo-skill"
    verified = engine.tool_metadata.get("recent_verified_work")
    assert isinstance(verified, list)
    assert any("Inspected file" in entry for entry in verified)
    assert any("Loaded skill demo-skill" in entry for entry in verified)


@pytest.mark.asyncio
async def test_query_engine_tracks_async_agent_activity(tmp_path: Path, monkeypatch):
    registry = create_default_tool_registry()
    agent_tool = registry.get("agent")
    assert agent_tool is not None

    async def _fake_execute(arguments, context):
        del arguments, context
        return ToolResult(output="Spawned agent worker@team (task_id=task_123, backend=subprocess)")

    monkeypatch.setattr(agent_tool, "execute", _fake_execute)
    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                name="agent",
                                input={"description": "Inspect CI", "prompt": "Inspect CI"},
                            )
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(role="assistant", content=[TextBlock(text="spawned")]),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        tool_metadata={},
    )

    events = [event async for event in engine.submit_message("spawn helper")]

    assert isinstance(events[-1], AssistantTurnComplete)
    async_state = engine._tool_metadata.get("async_agent_state")
    assert isinstance(async_state, list)
    assert async_state[-1].startswith("Spawned async agent")
    async_tasks = engine._tool_metadata.get("async_agent_tasks")
    assert isinstance(async_tasks, list)
    assert async_tasks[-1]["agent_id"] == "worker@team"
    assert async_tasks[-1]["task_id"] == "task_123"
    assert async_tasks[-1]["notification_sent"] is False


@pytest.mark.asyncio
async def test_query_engine_respects_pre_tool_hook_blocks(tmp_path: Path):
    sample = tmp_path / "hello.txt"
    sample.write_text("alpha\n", encoding="utf-8")
    registry = HookRegistry()
    registry.register(
        HookEvent.PRE_TOOL_USE,
        PromptHookDefinition(prompt="reject", matcher="read_file"),
    )

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="toolu_999",
                                name="read_file",
                                input={"path": str(sample)},
                            )
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="blocked")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings()),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        hook_executor=HookExecutor(
            registry,
            HookExecutionContext(
                cwd=tmp_path,
                api_client=StaticApiClient('{"ok": false, "reason": "no reading"}'),
                default_model="claude-test",
            ),
        ),
    )

    events = [event async for event in engine.submit_message("read file")]

    tool_results = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    assert tool_results
    assert tool_results[0].is_error is True
    assert "no reading" in tool_results[0].output


class _RecordingHookExecutor:
    """Duck-typed hook executor that records every fired event + payload."""

    def __init__(self) -> None:
        self.calls: list[tuple[HookEvent, dict]] = []

    async def execute(self, event: HookEvent, payload: dict):
        from openharness.hooks.types import AggregatedHookResult

        self.calls.append((event, dict(payload)))
        return AggregatedHookResult(results=[])


@pytest.mark.asyncio
async def test_user_prompt_submit_hook_fires(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    recorder = _RecordingHookExecutor()
    engine = QueryEngine(
        api_client=StaticApiClient("done"),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        hook_executor=recorder,  # type: ignore[arg-type]
    )

    _ = [event async for event in engine.submit_message("hello world")]

    user_prompt_calls = [c for c in recorder.calls if c[0] == HookEvent.USER_PROMPT_SUBMIT]
    assert len(user_prompt_calls) == 1
    assert user_prompt_calls[0][1]["event"] == "user_prompt_submit"
    assert user_prompt_calls[0][1]["prompt"] == "hello world"


@pytest.mark.asyncio
async def test_stop_hook_fires_on_clean_turn(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    recorder = _RecordingHookExecutor()
    engine = QueryEngine(
        api_client=StaticApiClient("all done"),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings()),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        hook_executor=recorder,  # type: ignore[arg-type]
    )

    _ = [event async for event in engine.submit_message("hi")]

    stop_calls = [c for c in recorder.calls if c[0] == HookEvent.STOP]
    assert len(stop_calls) == 1
    assert stop_calls[0][1]["event"] == "stop"
    assert stop_calls[0][1]["stop_reason"] == "tool_uses_empty"


@pytest.mark.asyncio
async def test_stop_hook_does_not_fire_when_tool_uses_present(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    sample = tmp_path / "hello.txt"
    sample.write_text("alpha\n", encoding="utf-8")
    recorder = _RecordingHookExecutor()
    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="toolu_1",
                                name="read_file",
                                input={"path": str(sample), "offset": 0, "limit": 1},
                            )
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="wrapped up")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings()),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        hook_executor=recorder,  # type: ignore[arg-type]
    )

    _ = [event async for event in engine.submit_message("read the file")]

    stop_calls = [c for c in recorder.calls if c[0] == HookEvent.STOP]
    # STOP fires exactly once — at the end of the second turn (no tool_uses),
    # NOT after the first turn that contained a tool_use.
    assert len(stop_calls) == 1


@pytest.mark.asyncio
async def test_notification_hook_fires_on_permission_prompt(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    recorder = _RecordingHookExecutor()
    prompt_tool_calls: list[tuple[str, str]] = []

    async def _permission_prompt(tool_name: str, reason: str) -> bool:
        prompt_tool_calls.append((tool_name, reason))
        # Assert the NOTIFICATION hook fired before this callback was invoked.
        notif = [c for c in recorder.calls if c[0] == HookEvent.NOTIFICATION]
        assert notif, "notification hook must fire before permission prompt"
        return False  # deny — keeps the turn short

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="toolu_bash_1",
                                name="bash",
                                input={"command": "echo hi"},
                            )
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="denied")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.DEFAULT)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        permission_prompt=_permission_prompt,
        hook_executor=recorder,  # type: ignore[arg-type]
    )

    _ = [event async for event in engine.submit_message("run something")]

    notification_calls = [c for c in recorder.calls if c[0] == HookEvent.NOTIFICATION]
    assert len(notification_calls) == 1
    payload = notification_calls[0][1]
    assert payload["event"] == "notification"
    assert payload["notification_type"] == "permission_prompt"
    assert payload["tool_name"] == "bash"
    # The permission prompt callback was invoked (confirms the hook fired on the
    # correct branch, not on the silently-denied branch).
    assert prompt_tool_calls


@pytest.mark.asyncio
async def test_subagent_stop_hook_fires_when_spawned_agent_finishes(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    recorder = _RecordingHookExecutor()
    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="toolu_agent_1",
                                name="agent",
                                input={
                                    "description": "quick worker run",
                                    "prompt": "ready",
                                    "subagent_type": "worker",
                                    "mode": "local_agent",
                                    "command": 'python -u -c "import sys; print(sys.stdin.readline().strip())"',
                                },
                            )
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=2, output_tokens=2),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="worker done")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        hook_executor=recorder,  # type: ignore[arg-type]
    )

    _ = [event async for event in engine.submit_message("run a worker")]

    manager = get_task_manager()
    deadline = asyncio.get_running_loop().time() + 2.0
    while asyncio.get_running_loop().time() < deadline:
        subagent_stop_calls = [c for c in recorder.calls if c[0] == HookEvent.SUBAGENT_STOP]
        if subagent_stop_calls:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("subagent_stop hook did not fire")

    subagent_stop_calls = [c for c in recorder.calls if c[0] == HookEvent.SUBAGENT_STOP]
    assert len(subagent_stop_calls) == 1
    payload = subagent_stop_calls[0][1]
    assert payload["event"] == "subagent_stop"
    assert payload["agent_id"] == "worker@default"
    assert payload["subagent_type"] == "worker"
    assert payload["mode"] == "local_agent"
    assert payload["status"] == "completed"
    assert payload["return_code"] == 0

    task = manager.get_task(payload["task_id"])
    assert task is not None
    assert task.status == "completed"


def _tool_context(tmp_path: Path, registry: ToolRegistry, settings: PermissionSettings) -> QueryContext:
    return QueryContext(
        api_client=_NoopApiClient(),
        tool_registry=registry,
        permission_checker=PermissionChecker(settings),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        max_tokens=1,
        max_turns=1,
    )


@pytest.mark.asyncio
async def test_execute_tool_call_blocks_sensitive_directory_roots(tmp_path: Path):
    sensitive_dir = tmp_path / ".ssh"
    sensitive_dir.mkdir()
    (sensitive_dir / "id_rsa").write_text("PRIVATE KEY MATERIAL\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(GrepTool())

    result = await _execute_tool_call(
        _tool_context(tmp_path, registry, PermissionSettings(mode=PermissionMode.DEFAULT)),
        "grep",
        "toolu_grep",
        {"pattern": "PRIVATE", "root": str(sensitive_dir), "file_glob": "*"},
    )

    assert result.is_error is True
    assert "sensitive credential path" in result.content


@pytest.mark.asyncio
async def test_execute_tool_call_applies_path_rules_to_directory_roots(tmp_path: Path):
    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()
    (blocked_dir / "secret.txt").write_text("classified\n", encoding="utf-8")

    registry = ToolRegistry()
    registry.register(GlobTool())

    result = await _execute_tool_call(
        _tool_context(
            tmp_path,
            registry,
            PermissionSettings(
                mode=PermissionMode.DEFAULT,
                path_rules=[{"pattern": str(blocked_dir) + "/*", "allow": False}],
            ),
        ),
        "glob",
        "toolu_glob",
        {"pattern": "*", "root": str(blocked_dir)},
    )

    assert result.is_error is True
    assert str(blocked_dir) in result.content


@pytest.mark.asyncio
async def test_execute_tool_call_returns_actionable_reason_when_user_denies_confirmation(tmp_path: Path):
    async def _deny(_tool_name: str, _reason: str) -> bool:
        return False

    result = await _execute_tool_call(
        QueryContext(
            api_client=_NoopApiClient(),
            tool_registry=create_default_tool_registry(),
            permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.DEFAULT)),
            cwd=tmp_path,
            model="claude-test",
            system_prompt="system",
            max_tokens=1,
            max_turns=1,
            permission_prompt=_deny,
        ),
        "bash",
        "toolu_bash",
        {"command": "mkdir -p scratch-dir"},
    )

    assert result.is_error is True
    assert "Mutating tools require user confirmation" in result.content
    assert "/permissions full_auto" in result.content


@pytest.mark.asyncio
async def test_query_engine_executes_ask_user_tool(tmp_path: Path):
    async def _answer(question: str) -> str:
        assert question == "Which color?"
        return "green"

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="toolu_ask",
                                name="ask_user_question",
                                input={"question": "Which color?"},
                            ),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="Picked green.")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings()),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        ask_user_prompt=_answer,
    )

    events = [event async for event in engine.submit_message("pick a color")]

    tool_results = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    assert tool_results
    assert tool_results[0].output == "green"
    assert isinstance(events[-1], AssistantTurnComplete)
    assert events[-1].message.text == "Picked green."


@pytest.mark.asyncio
async def test_query_engine_applies_path_rules_to_relative_read_file_targets(tmp_path: Path):
    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()
    secret = blocked_dir / "secret.txt"
    secret.write_text("top-secret\n", encoding="utf-8")

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="toolu_blocked_read",
                                name="read_file",
                                input={"path": "blocked/secret.txt", "offset": 0, "limit": 1},
                            )
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="blocked")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(
            PermissionSettings(
                mode=PermissionMode.DEFAULT,
                path_rules=[{"pattern": str((blocked_dir / "*").resolve()), "allow": False}],
            )
        ),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )

    events = [event async for event in engine.submit_message("read blocked file")]

    tool_results = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    assert tool_results
    assert tool_results[0].is_error is True
    assert "matches deny rule" in tool_results[0].output


@pytest.mark.asyncio
async def test_query_engine_applies_path_rules_to_write_file_targets_in_full_auto(tmp_path: Path):
    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()
    target = blocked_dir / "output.txt"

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="toolu_blocked_write",
                                name="write_file",
                                input={"path": "blocked/output.txt", "content": "poc"},
                            )
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="blocked")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(
            PermissionSettings(
                mode=PermissionMode.FULL_AUTO,
                path_rules=[{"pattern": str((blocked_dir / "*").resolve()), "allow": False}],
            )
        ),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )

    events = [event async for event in engine.submit_message("write blocked file")]

    tool_results = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    assert tool_results
    assert tool_results[0].is_error is True
    assert "matches deny rule" in tool_results[0].output
    assert target.exists() is False


class _OkInput(BaseModel):
    pass


class _OkTool(BaseTool):
    name = "ok_tool"
    description = "Returns success."
    input_model = _OkInput

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(output="ok", metadata={"sentinel": "metadata"})


class _BoomTool(BaseTool):
    name = "boom_tool"
    description = "Always raises."
    input_model = _OkInput

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        raise RuntimeError("boom")


class _LoadConversationImageTool(BaseTool):
    name = "load_conversation_image"
    description = "Test-only trusted conversation image loader."
    input_model = _OkInput

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(
            output="Loaded conversation image aaaaaaaaaaaa (image/png, 3 bytes).",
            metadata={
                "attachment_id": "a" * 64,
                "media_type": "image/png",
                "byte_size": 3,
                "_openharness_transient_image": ImageBlock(
                    media_type="image/png",
                    data="YWJj",
                ),
            },
        )


@pytest.mark.asyncio
async def test_exact_conversation_image_tool_adds_transient_image_for_next_request_only(
    tmp_path: Path,
) -> None:
    client = FakeApiClient(
        [
            _FakeResponse(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="toolu_image",
                            name="load_conversation_image",
                            input={},
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            ),
            _FakeResponse(
                message=ConversationMessage.from_user_text("loaded").model_copy(
                    update={"role": "assistant"}
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            ),
        ]
    )
    requests = []
    original_stream = client.stream_message

    async def recording_stream(request):
        requests.append(request)
        async for event in original_stream(request):
            yield event

    client.stream_message = recording_stream
    registry = ToolRegistry()
    registry.register(_LoadConversationImageTool())
    engine = QueryEngine(
        api_client=client,
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        supports_native_images=True,
        durable_message_transform=_externalize_test_images,
    )
    engine.load_messages(
        [
            ConversationMessage(
                role="user",
                content=[
                    AttachmentRefBlock(
                        attachment_id="a" * 64,
                        media_type="image/png",
                        byte_size=3,
                        label="test.png",
                    )
                ],
            )
        ]
    )

    _ = [event async for event in engine.submit_message("please reopen it")]

    assert len(requests) == 2
    first_images = [
        block
        for message in requests[0].messages
        for block in message.content
        if isinstance(block, ImageBlock)
    ]
    second_images = [
        block
        for message in requests[1].messages
        for block in message.content
        if isinstance(block, ImageBlock)
    ]
    assert first_images == []
    assert len(second_images) == 1
    assert second_images[0].data == "YWJj"
    assert all(
        not isinstance(block, ImageBlock)
        for message in engine.messages
        for block in message.content
    )
    assert "YWJj" not in json.dumps(
        [message.model_dump(mode="json") for message in engine.messages]
    )


@pytest.mark.asyncio
async def test_query_engine_synthesizes_tool_result_when_single_tool_raises(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(_BoomTool())

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            TextBlock(text="Running one tool."),
                            ToolUseBlock(id="toolu_boom", name="boom_tool", input={}),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="Recovered from the failure.")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )

    events = [event async for event in engine.submit_message("run one tool")]

    completed = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    assert len(completed) == 1
    assert completed[0].tool_name == "boom_tool"
    assert completed[0].is_error is True
    assert "RuntimeError" in completed[0].output
    assert "boom" in completed[0].output

    user_tool_messages = [
        msg
        for msg in engine.messages
        if msg.role == "user" and any(isinstance(block, ToolResultBlock) for block in msg.content)
    ]
    assert len(user_tool_messages) == 1
    result_blocks = [
        block for block in user_tool_messages[0].content if isinstance(block, ToolResultBlock)
    ]
    assert result_blocks[0].tool_use_id == "toolu_boom"

    assert isinstance(events[-1], AssistantTurnComplete)
    assert events[-1].message.text == "Recovered from the failure."


class _LargeOutputTool(BaseTool):
    name = "mcp__playwright__browser_snapshot"
    description = "Returns a large browser snapshot."
    input_model = _OkInput

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        del arguments, context
        return ToolResult(output="snapshot-line\n" * 40)


@pytest.mark.asyncio
async def test_query_engine_persists_compacted_tool_turn_history(tmp_path: Path, monkeypatch):
    """Compaction must not make a completed tool turn disappear from engine history."""

    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    monkeypatch.setattr("openharness.services.compact.try_session_memory_compaction", lambda *args, **kwargs: None)
    should_calls = {"count": 0}

    def _should_compact_once(*args, **kwargs):
        del args, kwargs
        should_calls["count"] += 1
        return should_calls["count"] == 1

    monkeypatch.setattr("openharness.services.compact.should_autocompact", _should_compact_once)

    registry = ToolRegistry()
    registry.register(_OkTool())
    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="<summary>Earlier setup was completed.</summary>")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            TextBlock(text="I will verify with a tool."),
                            ToolUseBlock(id="toolu_ok_after_compact", name="ok_tool", input={}),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="Tool finished after compact.")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    engine.load_messages(
        [
            ConversationMessage.from_user_text(f"historical user request {index}")
            if index % 2 == 0
            else ConversationMessage(role="assistant", content=[TextBlock(text=f"historical answer {index}")])
            for index in range(8)
        ]
    )

    events = [event async for event in engine.submit_message("new request after compact")]

    assert any(isinstance(event, CompactProgressEvent) and event.phase == "compact_end" for event in events)
    assert any("This session is being continued" in message.text for message in engine.messages)
    assert any(
        isinstance(block, ToolUseBlock) and block.id == "toolu_ok_after_compact"
        for message in engine.messages
        for block in message.content
    )
    assert any(
        isinstance(block, ToolResultBlock) and block.tool_use_id == "toolu_ok_after_compact"
        for message in engine.messages
        for block in message.content
    )
    assert engine.messages[-1].text == "Tool finished after compact."


@pytest.mark.asyncio
async def test_query_engine_synthesizes_tool_result_when_parallel_tool_raises(tmp_path: Path):
    """Parallel tool calls must each yield a tool_result even when one tool raises.

    Regression for the case where ``asyncio.gather`` (without
    ``return_exceptions=True``) propagated the first exception, abandoned the
    sibling coroutines, and left the conversation with un-replied ``tool_use``
    blocks — Anthropic's API then rejects the next request on the session.
    """

    registry = ToolRegistry()
    registry.register(_OkTool())
    registry.register(_BoomTool())

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            TextBlock(text="Running two tools."),
                            ToolUseBlock(id="toolu_ok", name="ok_tool", input={}),
                            ToolUseBlock(id="toolu_boom", name="boom_tool", input={}),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[TextBlock(text="Recovered from the failure.")],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )

    events = [event async for event in engine.submit_message("run both tools")]

    completed = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    completed_by_name = {event.tool_name: event for event in completed}
    assert set(completed_by_name) == {"ok_tool", "boom_tool"}
    assert completed_by_name["ok_tool"].is_error is False
    assert completed_by_name["ok_tool"].output == "ok"
    assert completed_by_name["ok_tool"].metadata == {"sentinel": "metadata"}
    assert completed_by_name["boom_tool"].is_error is True
    assert "RuntimeError" in completed_by_name["boom_tool"].output
    assert "boom" in completed_by_name["boom_tool"].output

    user_tool_messages = [
        msg for msg in engine.messages if msg.role == "user" and any(isinstance(block, ToolResultBlock) for block in msg.content)
    ]
    assert len(user_tool_messages) == 1
    result_blocks = [block for block in user_tool_messages[0].content if isinstance(block, ToolResultBlock)]
    assert {block.tool_use_id for block in result_blocks} == {"toolu_ok", "toolu_boom"}

    assert isinstance(events[-1], AssistantTurnComplete)
    assert events[-1].message.text == "Recovered from the failure."


@pytest.mark.asyncio
async def test_query_engine_sanitizes_dangling_tool_use_before_new_prompt(tmp_path: Path):
    engine = QueryEngine(
        api_client=StaticApiClient("fresh reply"),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    engine.load_messages([
        ConversationMessage.from_user_text("previous request"),
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="call_missing_output", name="ok_tool", input={})],
        ),
    ])

    events = [event async for event in engine.submit_message("new prompt")]

    assert isinstance(events[-1], AssistantTurnComplete)
    assert events[-1].message.text == "fresh reply"
    assert not any(
        isinstance(block, ToolUseBlock) and block.id == "call_missing_output"
        for message in engine.messages
        for block in message.content
    )


@pytest.mark.asyncio
async def test_query_engine_continue_pending_sanitizes_dangling_tool_use(tmp_path: Path):
    engine = QueryEngine(
        api_client=StaticApiClient("continued reply"),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )
    engine.load_messages([
        ConversationMessage.from_user_text("previous request"),
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="call_missing_output", name="ok_tool", input={})],
        ),
    ])

    events = [event async for event in engine.continue_pending()]

    assert isinstance(events[-1], AssistantTurnComplete)
    assert events[-1].message.text == "continued reply"
    assert not any(
        isinstance(block, ToolUseBlock) and block.id == "call_missing_output"
        for message in engine.messages
        for block in message.content
    )


@pytest.mark.asyncio
async def test_query_engine_offloads_large_tool_result_outputs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENHARNESS_TOOL_OUTPUT_INLINE_CHARS", "256")
    monkeypatch.setenv("OPENHARNESS_TOOL_OUTPUT_PREVIEW_CHARS", "128")
    registry = ToolRegistry()
    registry.register(_LargeOutputTool())

    engine = QueryEngine(
        api_client=FakeApiClient(
            [
                _FakeResponse(
                    message=ConversationMessage(
                        role="assistant",
                        content=[
                            ToolUseBlock(
                                id="toolu_snapshot",
                                name="mcp__playwright__browser_snapshot",
                                input={},
                            ),
                        ],
                    ),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
                _FakeResponse(
                    message=ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                ),
            ]
        ),
        tool_registry=registry,
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
        tool_metadata={},
    )

    events = [event async for event in engine.submit_message("snapshot")]

    completed = [event for event in events if isinstance(event, ToolExecutionCompleted)]
    assert len(completed) == 1
    assert completed[0].output.startswith("[Tool output truncated]")
    assert "snapshot-line" in completed[0].output

    user_tool_messages = [
        msg for msg in engine.messages if msg.role == "user" and any(isinstance(block, ToolResultBlock) for block in msg.content)
    ]
    result_blocks = [block for block in user_tool_messages[0].content if isinstance(block, ToolResultBlock)]
    inline = result_blocks[0].content
    assert "Full output saved to:" in inline
    assert "Original size:" in inline
    assert inline.count("snapshot-line") < 40
    artifact_line = next(line for line in inline.splitlines() if line.startswith("Full output saved to:"))
    artifact_path = Path(artifact_line.removeprefix("Full output saved to:").strip())
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == "snapshot-line\n" * 40
    assert str(artifact_path) in engine.tool_metadata["task_focus_state"]["active_artifacts"]


@pytest.mark.asyncio
async def test_query_engine_drops_empty_assistant_messages(tmp_path: Path):
    engine = QueryEngine(
        api_client=EmptyAssistantApiClient(),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="claude-test",
        system_prompt="system",
    )

    events = [event async for event in engine.submit_message("hello")]

    assert any(isinstance(event, ErrorEvent) for event in events)
    assert not any(isinstance(event, AssistantTurnComplete) for event in events)
    assert len(engine.messages) == 1
    assert engine.messages[0].role == "user"


@pytest.mark.asyncio
async def test_submit_message_repairs_dangling_tool_use_from_interrupt(tmp_path: Path, monkeypatch):
    """A turn cancelled by a newer user message can leave a dangling assistant
    tool_use in the *live* in-memory history (the persisted snapshot is
    sanitized, the live one was not). submit_message must repair it before
    querying, else the provider rejects the next request with
    'No tool output found for function call ...'."""
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)
    client = RecordingApiClient()
    engine = QueryEngine(
        api_client=client,
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        cwd=tmp_path,
        model="openai-compatible-model",
        system_prompt="system",
        max_tokens=120_000,
        max_turns=1,
    )
    # Interrupted turn: assistant tool_use with no following tool_result.
    engine._messages = [
        ConversationMessage.from_user_text("remember this about me"),
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="call_dangling", name="read_file", input={"path": "x"})],
        ),
    ]

    _ = [event async for event in engine.submit_message("a newer message")]

    def _has_dangling(messages):
        return any(
            isinstance(b, ToolUseBlock) and b.id == "call_dangling"
            for m in messages
            for b in m.content
        )

    assert not _has_dangling(engine.messages)          # repaired in the live history
    assert not _has_dangling(client.requests[0].messages)  # and never sent to the provider


@pytest.mark.asyncio
async def test_execute_tool_call_unknown_tool_returns_error(tmp_path: Path):
    result = await _execute_tool_call(
        _tool_context(tmp_path, ToolRegistry(), PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        "no_such_tool", "id1", {},
    )
    assert result.is_error is True
    assert "Unknown tool" in result.content


@pytest.mark.asyncio
async def test_execute_tool_call_invalid_input_returns_error(tmp_path: Path):
    registry = ToolRegistry()
    registry.register(GrepTool())
    result = await _execute_tool_call(
        _tool_context(tmp_path, registry, PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        "grep", "id2", {},  # missing required 'pattern'
    )
    assert result.is_error is True
    assert "Invalid input" in result.content


@pytest.mark.asyncio
async def test_execute_tool_call_swallows_tool_exception_as_error_result(tmp_path: Path):
    """Poison-safety: a raising tool yields an is_error result, never propagates —
    else the single-tool path leaves a dangling tool_use and poisons the session."""
    registry = ToolRegistry()
    registry.register(_BoomTool())
    result = await _execute_tool_call(
        _tool_context(tmp_path, registry, PermissionSettings(mode=PermissionMode.FULL_AUTO)),
        "boom_tool", "id3", {},
    )
    assert result.is_error is True
    assert result.tool_use_id == "id3"
    assert "boom" in result.content or "RuntimeError" in result.content
