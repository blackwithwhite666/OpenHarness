from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from openharness.api.client import ApiMessageRequest, ApiMessageCompleteEvent, ApiRetryEvent, ApiTextDeltaEvent
from openharness.api.codex_client import (
    MAX_RETRIES,
    CodexApiClient,
    StreamStalled,
    _convert_messages_to_codex,
    _format_codex_stream_error,
    _resolve_codex_url,
)
from openharness.engine.messages import ConversationMessage, ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock


class _FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        lines: list[str] | None = None,
        body: str = "",
        hang_after_lines: bool = False,
        pause_after_line: int | None = None,
        pause_gate: asyncio.Event | None = None,
        line_delays: dict[int, float] | None = None,
        hang_on_enter: bool = False,
    ) -> None:
        self.status_code = status_code
        self._lines = lines or []
        self._body = body.encode("utf-8")
        self._hang_after_lines = hang_after_lines
        self._pause_after_line = pause_after_line
        self._pause_gate = pause_gate
        self._line_delays = line_delays or {}
        self._hang_on_enter = hang_on_enter

    async def __aenter__(self) -> "_FakeStreamResponse":
        if self._hang_on_enter:
            await asyncio.Event().wait()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def aread(self) -> bytes:
        return self._body

    async def aiter_lines(self):
        for index, line in enumerate(self._lines):
            delay = self._line_delays.get(index)
            if delay is not None:
                await asyncio.sleep(delay)
            yield line
            if index == self._pause_after_line and self._pause_gate is not None:
                await self._pause_gate.wait()
        if self._hang_after_lines:
            await asyncio.Event().wait()


class _SlowDripStreamResponse(_FakeStreamResponse):
    def __init__(self, *, interval: float, initial_lines: list[str] | None = None) -> None:
        super().__init__()
        self._interval = interval
        self._initial_lines = initial_lines or []

    async def aiter_lines(self):
        for line in self._initial_lines:
            yield line
        while True:
            await asyncio.sleep(self._interval)
            yield 'data: {"type":"response.in_progress","response":{"status":"in_progress"}}'
            yield ""


class _FakeAsyncClient:
    def __init__(self, response: _FakeStreamResponse, sink: dict[str, Any]) -> None:
        self._response = response
        self._sink = sink

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, Any]):
        self._sink["method"] = method
        self._sink["url"] = url
        self._sink["headers"] = headers
        self._sink["json"] = json
        return self._response


class _FakeAsyncClientSequence:
    def __init__(self, responses: list[_FakeStreamResponse], sink: dict[str, Any]) -> None:
        self._responses = iter(responses)
        self._sink = sink
        self.attempts = 0

    def __call__(self, *args, **kwargs) -> _FakeAsyncClient:
        self.attempts += 1
        return _FakeAsyncClient(next(self._responses), self._sink)


def _b64url(data: dict[str, object]) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    import base64

    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _fake_codex_token() -> str:
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_test"}}
    return f"{_b64url({'alg': 'none', 'typ': 'JWT'})}.{_b64url(payload)}.sig"


def _codex_request() -> ApiMessageRequest:
    return ApiMessageRequest(
        model="gpt-5.5",
        messages=[ConversationMessage.from_user_text("hi")],
        system_prompt="Be helpful.",
    )


async def _collect_stream(client: CodexApiClient, request: ApiMessageRequest) -> list[Any]:
    return [event async for event in client.stream_message(request)]


def _successful_text_lines(*deltas: str) -> list[str]:
    text = "".join(deltas)
    lines: list[str] = []
    for delta in deltas:
        lines.extend([
            f'data: {{"type":"response.output_text.delta","delta":"{delta}"}}',
            "",
        ])
    lines.extend([
        (
            'data: {"type":"response.output_item.done","item":{"id":"msg_1",'
            f'"type":"message","content":[{{"type":"output_text","text":"{text}"}}]}}'
        ),
        "",
        (
            'data: {"type":"response.completed","response":{"status":"completed",'
            '"usage":{"input_tokens":2,"output_tokens":1}}}'
        ),
        "",
    ])
    return lines


def _disable_retry_delays(monkeypatch) -> None:
    monkeypatch.setattr("openharness.api.codex_client.BASE_DELAY_SECONDS", 0.0)
    monkeypatch.setattr("openharness.api.codex_client.MAX_DELAY_SECONDS", 0.0)


def test_convert_messages_to_codex():
    messages = [
        ConversationMessage.from_user_text("Inspect file"),
        ConversationMessage(
            role="assistant",
            content=[
                TextBlock(text="I'll inspect it."),
                ToolUseBlock(id="call_123", name="read_file", input={"path": "README.md"}),
            ],
        ),
        ConversationMessage(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_123", content="hello", is_error=False)],
        ),
    ]

    converted = _convert_messages_to_codex(messages)

    assert converted[0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Inspect file"}],
    }
    assert converted[1]["type"] == "message"
    assert converted[1]["role"] == "assistant"
    assert converted[2]["type"] == "function_call"
    assert converted[2]["call_id"] == "call_123"
    assert json.loads(converted[2]["arguments"]) == {"path": "README.md"}
    assert converted[3] == {
        "type": "function_call_output",
        "call_id": "call_123",
        "output": "hello",
    }


def test_convert_user_message_with_tool_result_before_text_to_codex():
    messages = [
        ConversationMessage(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="call_123", content="done", is_error=False),
                TextBlock(text="next request"),
            ],
        )
    ]

    converted = _convert_messages_to_codex(messages)

    assert converted == [
        {"type": "function_call_output", "call_id": "call_123", "output": "done"},
        {"role": "user", "content": [{"type": "input_text", "text": "next request"}]},
    ]



def test_convert_multimodal_user_message_to_codex():
    messages = [
        ConversationMessage(
            role="user",
            content=[
                TextBlock(text="What is in this image?"),
                ImageBlock(media_type="image/png", data="YWJj", source_path="/tmp/example.png"),
            ],
        )
    ]

    converted = _convert_messages_to_codex(messages)

    assert converted == [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What is in this image?"},
            {"type": "input_image", "image_url": "data:image/png;base64,YWJj"},
        ],
    }]


def test_resolve_codex_url_ignores_unrelated_base_url():
    assert _resolve_codex_url("https://api.moonshot.cn/anthropic") == "https://chatgpt.com/backend-api/codex/responses"


def test_format_codex_stream_error_includes_code_and_request_id():
    message = _format_codex_stream_error(
        {
            "type": "error",
            "message": "Upstream overloaded",
            "code": "overloaded",
            "request_id": "req_123",
        },
        fallback="Codex error",
    )
    assert message == "Upstream overloaded (code=overloaded) [request_id=req_123]"


@pytest.mark.asyncio
async def test_codex_client_streams_text(monkeypatch):
    sink: dict[str, Any] = {}
    response = _FakeStreamResponse(
        lines=[
            'event: response.output_item.added',
            'data: {"type":"response.output_item.added","item":{"id":"msg_1","type":"message","content":[],"role":"assistant"}}',
            "",
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"CODE"}',
            "",
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"X_OK"}',
            "",
            'event: response.output_item.done',
            'data: {"type":"response.output_item.done","item":{"id":"msg_1","type":"message","content":[{"type":"output_text","text":"CODEX_OK","annotations":[]}]}}',
            "",
            'event: response.completed',
            'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":12,"output_tokens":3}}}',
            "",
        ]
    )
    monkeypatch.setattr(
        "openharness.api.codex_client.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(response, sink),
    )

    client = CodexApiClient(_fake_codex_token())
    request = ApiMessageRequest(
        model="gpt-5.5",
        messages=[ConversationMessage.from_user_text("hi")],
        system_prompt="Be helpful.",
        effort="xhigh",
    )
    events = [event async for event in client.stream_message(request)]

    assert [event.text for event in events if isinstance(event, ApiTextDeltaEvent)] == ["CODE", "X_OK"]
    complete = next(event for event in events if isinstance(event, ApiMessageCompleteEvent))
    assert complete.message.text == "CODEX_OK"
    assert complete.usage.input_tokens == 12
    assert complete.usage.output_tokens == 3
    assert sink["url"].endswith("/codex/responses")
    assert sink["json"]["instructions"] == "Be helpful."
    assert sink["json"]["model"] == "gpt-5.5"
    assert sink["json"]["reasoning"] == {"effort": "xhigh"}
    assert sink["headers"]["OpenAI-Beta"] == "responses=experimental"


@pytest.mark.asyncio
async def test_codex_client_retries_response_start_hang_then_succeeds(monkeypatch):
    sink: dict[str, Any] = {}
    stalled = _FakeStreamResponse(hang_after_lines=True)
    succeeded = _FakeStreamResponse(lines=_successful_text_lines("clean ", "answer"))
    client_factory = _FakeAsyncClientSequence([stalled, succeeded], sink)
    monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", client_factory)
    _disable_retry_delays(monkeypatch)

    client = CodexApiClient(
        _fake_codex_token(),
        stall_timeout_seconds=0.02,
        attempt_timeout_seconds=0.2,
    )
    started = time.monotonic()
    events = await asyncio.wait_for(
        _collect_stream(client, _codex_request()),
        timeout=0.5,
    )

    assert time.monotonic() - started < 0.15
    assert client_factory.attempts == 2
    retry_events = [event for event in events if isinstance(event, ApiRetryEvent)]
    assert len(retry_events) == 1
    assert "inactivity timeout" in retry_events[0].message
    assert [event.text for event in events if isinstance(event, ApiTextDeltaEvent)] == [
        "clean ",
        "answer",
    ]
    complete_events = [event for event in events if isinstance(event, ApiMessageCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].message.text == "clean answer"


@pytest.mark.asyncio
async def test_codex_client_total_timeout_stops_slow_drip(monkeypatch):
    sink: dict[str, Any] = {}
    response = _SlowDripStreamResponse(interval=0.01)
    succeeded = _FakeStreamResponse(lines=_successful_text_lines("recovered"))
    client_factory = _FakeAsyncClientSequence([response, succeeded], sink)
    monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", client_factory)
    _disable_retry_delays(monkeypatch)

    client = CodexApiClient(
        _fake_codex_token(),
        stall_timeout_seconds=0.03,
        attempt_timeout_seconds=0.06,
    )
    started = time.monotonic()
    events = await asyncio.wait_for(
        _collect_stream(client, _codex_request()),
        timeout=0.3,
    )

    assert time.monotonic() - started < 0.2
    assert client_factory.attempts == 2
    retry_events = [event for event in events if isinstance(event, ApiRetryEvent)]
    assert len(retry_events) == 1
    assert "attempt timeout" in retry_events[0].message
    assert [event.text for event in events if isinstance(event, ApiTextDeltaEvent)] == ["recovered"]


@pytest.mark.asyncio
async def test_codex_client_total_timeout_stops_connect_headers_hang(monkeypatch):
    sink: dict[str, Any] = {}
    response = _FakeStreamResponse(hang_on_enter=True)
    succeeded = _FakeStreamResponse(lines=_successful_text_lines("connected"))
    client_factory = _FakeAsyncClientSequence([response, succeeded], sink)
    monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", client_factory)
    _disable_retry_delays(monkeypatch)

    client = CodexApiClient(
        _fake_codex_token(),
        stall_timeout_seconds=0.01,
        attempt_timeout_seconds=0.04,
    )
    started = time.monotonic()
    events = await asyncio.wait_for(
        _collect_stream(client, _codex_request()),
        timeout=0.2,
    )

    assert time.monotonic() - started < 0.15
    assert client_factory.attempts == 2
    retry_events = [event for event in events if isinstance(event, ApiRetryEvent)]
    assert len(retry_events) == 1
    assert "attempt timeout" in retry_events[0].message
    assert [event.text for event in events if isinstance(event, ApiTextDeltaEvent)] == ["connected"]


@pytest.mark.asyncio
async def test_codex_client_gives_up_after_repeated_stalls(monkeypatch):
    sink: dict[str, Any] = {}
    stalled_responses = [
        _FakeStreamResponse(
            lines=['data: {"type":"response.in_progress"}', ""],
            hang_after_lines=True,
        )
        for _ in range(MAX_RETRIES + 1)
    ]
    client_factory = _FakeAsyncClientSequence(stalled_responses, sink)
    monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", client_factory)
    _disable_retry_delays(monkeypatch)

    client = CodexApiClient(_fake_codex_token(), stall_timeout_seconds=0.01)
    events: list[Any] = []

    async def consume() -> None:
        async for event in client.stream_message(_codex_request()):
            events.append(event)

    started = time.monotonic()
    with pytest.raises(StreamStalled) as error:
        await asyncio.wait_for(consume(), timeout=0.5)

    assert time.monotonic() - started < 0.5
    assert client_factory.attempts == MAX_RETRIES + 1
    assert len([event for event in events if isinstance(event, ApiRetryEvent)]) == MAX_RETRIES
    assert "Codex stream stalled" in str(error.value)
    assert f"gave up after {MAX_RETRIES + 1} attempts" in str(error.value)
    assert client._is_retryable(error.value)


@pytest.mark.asyncio
async def test_codex_client_does_not_retry_stall_after_text_delta(monkeypatch):
    sink: dict[str, Any] = {}
    partial = _FakeStreamResponse(
        lines=['data: {"type":"response.output_text.delta","delta":"partial"}', ""],
        hang_after_lines=True,
    )
    unused_retry = _FakeStreamResponse(lines=_successful_text_lines("duplicate"))
    client_factory = _FakeAsyncClientSequence([partial, unused_retry], sink)
    monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", client_factory)
    _disable_retry_delays(monkeypatch)

    client = CodexApiClient(_fake_codex_token(), stall_timeout_seconds=0.01)
    events: list[Any] = []

    async def consume() -> None:
        async for event in client.stream_message(_codex_request()):
            events.append(event)

    with pytest.raises(StreamStalled):
        await asyncio.wait_for(consume(), timeout=0.2)

    assert client_factory.attempts == 1
    assert [event.text for event in events if isinstance(event, ApiTextDeltaEvent)] == ["partial"]
    assert not any(isinstance(event, ApiRetryEvent) for event in events)


@pytest.mark.asyncio
async def test_codex_client_does_not_retry_total_timeout_after_text_delta(monkeypatch):
    sink: dict[str, Any] = {}
    partial = _SlowDripStreamResponse(
        interval=0.01,
        initial_lines=['data: {"type":"response.output_text.delta","delta":"partial"}', ""],
    )
    unused_retry = _FakeStreamResponse(lines=_successful_text_lines("duplicate"))
    client_factory = _FakeAsyncClientSequence([partial, unused_retry], sink)
    monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", client_factory)
    _disable_retry_delays(monkeypatch)

    client = CodexApiClient(
        _fake_codex_token(),
        stall_timeout_seconds=0.03,
        attempt_timeout_seconds=0.06,
    )
    events: list[Any] = []

    async def consume() -> None:
        async for event in client.stream_message(_codex_request()):
            events.append(event)

    with pytest.raises(StreamStalled, match="attempt timeout"):
        await asyncio.wait_for(consume(), timeout=0.3)

    assert client_factory.attempts == 1
    assert [event.text for event in events if isinstance(event, ApiTextDeltaEvent)] == ["partial"]
    assert not any(isinstance(event, ApiRetryEvent) for event in events)


@pytest.mark.asyncio
async def test_codex_client_happy_path_deltas_remain_live_and_ordered(monkeypatch):
    sink: dict[str, Any] = {}
    release_rest = asyncio.Event()
    response = _FakeStreamResponse(
        lines=_successful_text_lines("first", "second"),
        pause_after_line=1,
        pause_gate=release_rest,
    )
    client_factory = _FakeAsyncClientSequence([response], sink)
    monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", client_factory)

    client = CodexApiClient(
        _fake_codex_token(),
        stall_timeout_seconds=0.2,
        attempt_timeout_seconds=0.5,
    )
    stream = client.stream_message(_codex_request()).__aiter__()
    first_event = await asyncio.wait_for(stream.__anext__(), timeout=0.1)

    assert isinstance(first_event, ApiTextDeltaEvent)
    assert first_event.text == "first"
    assert not release_rest.is_set()

    release_rest.set()
    remaining = [event async for event in stream]
    assert [
        event.text
        for event in [first_event, *remaining]
        if isinstance(event, ApiTextDeltaEvent)
    ] == ["first", "second"]
    assert len([event for event in remaining if isinstance(event, ApiMessageCompleteEvent)]) == 1
    assert client_factory.attempts == 1


@pytest.mark.asyncio
async def test_codex_client_disabled_timeouts_allow_slow_stream(monkeypatch):
    sink: dict[str, Any] = {}
    lines = [
        'data: {"type":"response.in_progress"}',
        "",
        *_successful_text_lines("slow", " stream"),
    ]
    response = _FakeStreamResponse(lines=lines, line_delays={2: 0.04})
    client_factory = _FakeAsyncClientSequence([response], sink)
    monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", client_factory)

    client = CodexApiClient(
        _fake_codex_token(),
        stall_timeout_seconds=None,
        attempt_timeout_seconds=None,
    )
    events = await asyncio.wait_for(
        _collect_stream(client, _codex_request()),
        timeout=0.2,
    )

    assert [event.text for event in events if isinstance(event, ApiTextDeltaEvent)] == [
        "slow",
        " stream",
    ]
    assert len([event for event in events if isinstance(event, ApiMessageCompleteEvent)]) == 1
    assert client_factory.attempts == 1


@pytest.mark.asyncio
async def test_codex_client_emits_tool_use(monkeypatch):
    sink: dict[str, Any] = {}
    response = _FakeStreamResponse(
        lines=[
            'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","arguments":"","call_id":"call_abc","name":"glob"}}',
            "",
            'data: {"type":"response.output_item.done","item":{"id":"fc_1","type":"function_call","arguments":"{\\"pattern\\":\\"src/**/*.py\\"}","call_id":"call_abc","name":"glob"}}',
            "",
            'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":7,"output_tokens":2}}}',
            "",
        ]
    )
    monkeypatch.setattr(
        "openharness.api.codex_client.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(response, sink),
    )

    client = CodexApiClient(_fake_codex_token())
    request = ApiMessageRequest(
        model="gpt-5.4",
        messages=[ConversationMessage.from_user_text("glob")],
        system_prompt="Use tools.",
        tools=[{"name": "glob", "description": "find files", "input_schema": {"type": "object"}}],
    )
    events = [event async for event in client.stream_message(request)]

    complete = next(event for event in events if isinstance(event, ApiMessageCompleteEvent))
    assert complete.stop_reason == "tool_use"
    assert len(complete.message.tool_uses) == 1
    tool_use = complete.message.tool_uses[0]
    assert tool_use.id == "call_abc"
    assert tool_use.name == "glob"
    assert tool_use.input == {"pattern": "src/**/*.py"}
    assert sink["json"]["tools"][0]["name"] == "glob"


# ---- token refresh (long-running gateway self-heals expired codex tokens) ----
def test_codex_refresh_client_auth_updates_token():
    client = CodexApiClient("stale", auth_token_resolver=lambda: "fresh")
    client._refresh_client_auth()
    assert client._auth_token == "fresh"


def test_codex_refresh_client_auth_is_best_effort_on_resolver_error():
    def boom() -> str:
        raise RuntimeError("resolver down")

    client = CodexApiClient("stale", auth_token_resolver=boom)
    client._refresh_client_auth()
    assert client._auth_token == "stale"  # keeps the previous token, never raises


def test_codex_no_resolver_keeps_captured_token():
    client = CodexApiClient("tok")
    client._refresh_client_auth()
    assert client._auth_token == "tok"


@pytest.mark.asyncio
async def test_codex_client_sends_resolved_token_not_captured(monkeypatch):
    sink: dict[str, Any] = {}
    response = _FakeStreamResponse(
        lines=[
            'event: response.output_item.done',
            'data: {"type":"response.output_item.done","item":{"id":"m","type":"message","content":[{"type":"output_text","text":"ok","annotations":[]}]}}',
            "",
            'event: response.completed',
            'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}',
            "",
        ]
    )
    monkeypatch.setattr(
        "openharness.api.codex_client.httpx.AsyncClient",
        lambda *args, **kwargs: _FakeAsyncClient(response, sink),
    )
    fresh = _fake_codex_token()
    client = CodexApiClient("stale-captured-token", auth_token_resolver=lambda: fresh)
    request = ApiMessageRequest(
        model="gpt-5.4",
        messages=[ConversationMessage.from_user_text("hi")],
        system_prompt="x",
    )
    [event async for event in client.stream_message(request)]
    # the request carried the resolver's fresh token, not the stale captured one
    assert sink["headers"]["Authorization"] == f"Bearer {fresh}"
