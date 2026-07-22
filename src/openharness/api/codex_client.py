"""OpenAI Codex subscription client backed by chatgpt.com Codex Responses."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import platform
import random
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any, AsyncIterator, TypeVar

import httpx

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiRetryEvent,
    ApiStreamEvent,
    ApiTextDeltaEvent,
)
from openharness.api.errors import AuthenticationFailure, OpenHarnessApiError, RateLimitFailure, RequestFailure
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, ImageBlock, TextBlock, ToolResultBlock, ToolUseBlock

log = logging.getLogger(__name__)

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
MAX_RETRIES = 5
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 30.0

_T = TypeVar("_T")


class StreamStalled(RequestFailure):
    """Raised when a Codex SSE stream stops producing events."""


def _extract_account_id(token: str) -> str:
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthenticationFailure("Codex access token is not a valid JWT.")
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)).decode("utf-8")
        )
    except Exception as exc:
        raise AuthenticationFailure("Could not decode Codex access token payload.") from exc
    auth_claim = payload.get(JWT_CLAIM_PATH)
    if not isinstance(auth_claim, dict):
        raise AuthenticationFailure("Codex access token is missing account metadata.")
    account_id = auth_claim.get("chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id:
        raise AuthenticationFailure("Codex access token is missing chatgpt_account_id.")
    return account_id


def _resolve_codex_url(base_url: str | None) -> str:
    trimmed = (base_url or "").strip()
    if trimmed and "chatgpt.com/backend-api" not in trimmed:
        trimmed = ""
    raw = (trimmed or DEFAULT_CODEX_BASE_URL).rstrip("/")
    if raw.endswith("/codex/responses"):
        return raw
    if raw.endswith("/codex"):
        return f"{raw}/responses"
    return f"{raw}/codex/responses"


def _build_codex_headers(token: str, *, session_id: str | None = None) -> dict[str, str]:
    account_id = _extract_account_id(token)
    headers = {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "originator": "openharness",
        "User-Agent": f"openharness ({platform.system().lower()} {platform.machine() or 'unknown'})",
        "OpenAI-Beta": "responses=experimental",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    if session_id:
        headers["session_id"] = session_id
    return headers


def _convert_messages_to_codex(messages: list[ConversationMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "user":
            # Responses API requires function_call_output items to appear before
            # any following user input.  A ConversationMessage can contain both
            # tool results and user text, so emit the tool outputs first to keep
            # every prior function_call immediately satisfied.
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    result.append({
                        "type": "function_call_output",
                        "call_id": block.tool_use_id,
                        "output": block.content,
                    })
            user_content: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text.strip():
                    user_content.append({"type": "input_text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    user_content.append({
                        "type": "input_image",
                        "image_url": f"data:{block.media_type};base64,{block.data}",
                    })
            if user_content:
                result.append({"role": "user", "content": user_content})
            continue

        assistant_text = "".join(block.text for block in msg.content if isinstance(block, TextBlock))
        if assistant_text:
            result.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text, "annotations": []}],
            })
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                result.append({
                    "type": "function_call",
                    "id": f"fc_{block.id[:58]}",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": json.dumps(block.input, separators=(",", ":")),
                })
    return result


def _convert_tools_to_codex(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        }
        for tool in tools
    ]


def _build_codex_body(request: ApiMessageRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "store": False,
        "stream": True,
        "instructions": request.system_prompt or "You are OpenHarness.",
        "input": _convert_messages_to_codex(request.messages),
        "text": {"verbosity": "medium"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    if request.cache_key:
        body["prompt_cache_key"] = request.cache_key
    if request.tools:
        body["tools"] = _convert_tools_to_codex(request.tools)
    effort = _normalize_reasoning_effort(request.effort)
    if effort:
        body["reasoning"] = {"effort": effort}
    return body


def _normalize_reasoning_effort(effort: str | None) -> str | None:
    normalized = (effort or "").strip().lower()
    if normalized == "max":
        return "xhigh"
    if normalized in {"low", "medium", "high", "xhigh"}:
        return normalized
    return None


def _usage_from_response(response: dict[str, Any]) -> UsageSnapshot:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return UsageSnapshot()
    input_tokens_details = usage.get("input_tokens_details")
    cached_input_tokens = (
        input_tokens_details.get("cached_tokens", 0)
        if isinstance(input_tokens_details, dict)
        else 0
    )
    return UsageSnapshot(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_input_tokens=int(cached_input_tokens or 0),
    )


def _stop_reason_from_response(response: dict[str, Any], *, has_tool_calls: bool) -> str | None:
    status = response.get("status")
    if has_tool_calls and status == "completed":
        return "tool_use"
    if status == "completed":
        return "stop"
    if status == "incomplete":
        return "length"
    if status in {"failed", "cancelled"}:
        return "error"
    return None


def _format_error_message(status_code: int, payload: str) -> str:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message
        detail = parsed.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail
    text = payload.strip()
    if text:
        return text
    return f"Codex request failed with status {status_code}"


def _format_codex_stream_error(event: dict[str, Any], *, fallback: str) -> str:
    error = event.get("error")
    payload = error if isinstance(error, dict) else event
    message = payload.get("message") if isinstance(payload, dict) else None
    code = payload.get("code") if isinstance(payload, dict) else None
    request_id = (
        (payload.get("request_id") if isinstance(payload, dict) else None)
        or event.get("request_id")
    )

    parts: list[str] = []
    if isinstance(message, str) and message.strip():
        parts.append(message.strip())
    elif isinstance(code, str) and code.strip():
        parts.append(code.strip())
    else:
        parts.append(fallback)

    if isinstance(code, str) and code.strip():
        parts.append(f"(code={code.strip()})")
    if isinstance(request_id, str) and request_id.strip():
        parts.append(f"[request_id={request_id.strip()}]")
    return " ".join(parts)


def _translate_status_error(status_code: int, message: str) -> OpenHarnessApiError:
    if status_code in {401, 403}:
        return AuthenticationFailure(message)
    if status_code == 429:
        return RateLimitFailure(message)
    return RequestFailure(message)


class CodexApiClient:
    """Client for ChatGPT/Codex subscription-backed Codex Responses."""

    def __init__(
        self,
        auth_token: str,
        *,
        base_url: str | None = None,
        auth_token_resolver: Callable[[], str] | None = None,
        stall_timeout_seconds: float | None = 30.0,
        attempt_timeout_seconds: float | None = 120.0,
    ) -> None:
        self._auth_token = auth_token
        self._base_url = base_url
        self._url = _resolve_codex_url(base_url)
        self._auth_token_resolver = auth_token_resolver
        self._stall_timeout_seconds = stall_timeout_seconds
        self._attempt_timeout_seconds = attempt_timeout_seconds

    def _refresh_client_auth(self) -> None:
        """Re-resolve the access token before a request so a long-running client
        picks up a refreshed/rotated ChatGPT/Codex token (the resolver re-reads
        ~/.codex/auth.json and refreshes it when expired) instead of sending a
        stale captured token and 401-ing until process restart. Best-effort: a
        resolver failure leaves the previous token in place."""
        if self._auth_token_resolver is None:
            return
        try:
            next_token = self._auth_token_resolver()
        except Exception as exc:
            # Best-effort: keep the previous token (degrade to an eventual 401 the
            # next call self-heals) rather than crash. Log so a genuinely dead
            # refresh chain ("run codex login") is visible, not just a bare 401.
            log.warning("codex token refresh failed, using previous token: %s", exc)
            return
        if next_token and next_token != self._auth_token:
            self._auth_token = next_token

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        last_error: Exception | None = None
        stream_event_yielded = False
        for attempt in range(MAX_RETRIES + 1):
            try:
                # Off the event loop: the resolver may do a blocking HTTPS refresh.
                await asyncio.to_thread(self._refresh_client_auth)
                async for event in self._stream_once(request):
                    stream_event_yielded = True
                    yield event
                return
            except Exception as exc:
                last_error = exc
                if stream_event_yielded or not self._is_retryable(exc):
                    raise self._translate_error(exc) from exc
                if attempt >= MAX_RETRIES:
                    if isinstance(exc, StreamStalled):
                        raise StreamStalled(
                            f"Codex stream stalled; gave up after {attempt + 1} attempts: {exc}"
                        ) from exc
                    raise self._translate_error(exc) from exc
                base_delay = min(BASE_DELAY_SECONDS * (2 ** attempt), MAX_DELAY_SECONDS)
                delay = base_delay + random.uniform(0, base_delay * 0.25)
                yield ApiRetryEvent(
                    message=str(exc),
                    attempt=attempt + 1,
                    max_attempts=MAX_RETRIES + 1,
                    delay_seconds=delay,
                )
                await asyncio.sleep(delay)
        if last_error is not None:
            raise self._translate_error(last_error) from last_error

    async def _stream_once(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + self._attempt_timeout_seconds
            if self._attempt_timeout_seconds and self._attempt_timeout_seconds > 0
            else None
        )
        body = _build_codex_body(request)

        content: list[TextBlock | ToolUseBlock] = []
        current_text_parts: list[str] = []
        completed_response: dict[str, Any] | None = None

        headers = _build_codex_headers(self._auth_token)
        async with AsyncExitStack() as client_stack:
            client = await self._await_before_attempt_deadline(
                client_stack.enter_async_context(
                    httpx.AsyncClient(timeout=60.0, follow_redirects=True)
                ),
                deadline=deadline,
            )
            async with AsyncExitStack() as stack:
                response = await self._await_before_attempt_deadline(
                    stack.enter_async_context(
                        client.stream("POST", self._url, headers=headers, json=body)
                    ),
                    deadline=deadline,
                )
                if response.status_code >= 400:
                    payload = await self._await_before_attempt_deadline(
                        response.aread(),
                        deadline=deadline,
                    )
                    message = _format_error_message(response.status_code, payload.decode("utf-8", "replace"))
                    raise httpx.HTTPStatusError(message, request=response.request, response=response)

                event_iterator = self._iter_sse_events(response).__aiter__()
                while True:
                    try:
                        event = await self._next_sse_event(event_iterator, deadline=deadline)
                    except StopAsyncIteration:
                        break

                    event_type = event.get("type")
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            current_text_parts.append(delta)
                            yield ApiTextDeltaEvent(text=delta)
                    elif event_type == "response.output_item.done":
                        item = event.get("item")
                        if not isinstance(item, dict):
                            continue
                        item_type = item.get("type")
                        if item_type == "message":
                            text = ""
                            raw_content = item.get("content")
                            if isinstance(raw_content, list):
                                parts = []
                                for block in raw_content:
                                    if isinstance(block, dict):
                                        if block.get("type") == "output_text":
                                            parts.append(str(block.get("text", "")))
                                        elif block.get("type") == "refusal":
                                            parts.append(str(block.get("refusal", "")))
                                text = "".join(parts)
                            if text:
                                content.append(TextBlock(text=text))
                        elif item_type == "function_call":
                            arguments = item.get("arguments")
                            parsed_arguments: dict[str, Any]
                            if isinstance(arguments, str) and arguments:
                                try:
                                    loaded = json.loads(arguments)
                                except json.JSONDecodeError:
                                    loaded = {}
                            else:
                                loaded = {}
                            parsed_arguments = loaded if isinstance(loaded, dict) else {}
                            call_id = item.get("call_id")
                            name = item.get("name")
                            if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
                                content.append(ToolUseBlock(id=call_id, name=name, input=parsed_arguments))
                    elif event_type == "response.completed":
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            completed_response = response_payload
                    elif event_type == "response.failed":
                        response_payload = event.get("response")
                        if isinstance(response_payload, dict):
                            raise RequestFailure(
                                _format_codex_stream_error(
                                    response_payload,
                                    fallback="Codex response failed",
                                )
                            )
                        raise RequestFailure("Codex response failed")
                    elif event_type == "error":
                        raise RequestFailure(
                            _format_codex_stream_error(event, fallback="Codex error")
                        )

        if current_text_parts and not any(isinstance(block, TextBlock) for block in content):
            content.insert(0, TextBlock(text="".join(current_text_parts)))

        final_message = ConversationMessage(role="assistant", content=content)
        usage = _usage_from_response(completed_response or {})
        stop_reason = _stop_reason_from_response(
            completed_response or {},
            has_tool_calls=bool(final_message.tool_uses),
        )
        yield ApiMessageCompleteEvent(
            message=final_message,
            usage=usage,
            stop_reason=stop_reason,
        )

    async def _await_before_attempt_deadline(
        self,
        awaitable: Awaitable[_T],
        *,
        deadline: float | None,
    ) -> _T:
        if deadline is None:
            return await awaitable

        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        try:
            return await asyncio.wait_for(awaitable, timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise self._attempt_timeout_error() from exc

    async def _next_sse_event(
        self,
        event_iterator: AsyncIterator[dict[str, Any]],
        *,
        deadline: float | None,
    ) -> dict[str, Any]:
        wait_timeout: float | None = None
        deadline_is_limit = False
        if self._stall_timeout_seconds and self._stall_timeout_seconds > 0:
            wait_timeout = self._stall_timeout_seconds

        if deadline is not None:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            if wait_timeout is None or remaining <= wait_timeout:
                wait_timeout = remaining
                deadline_is_limit = True

        try:
            if wait_timeout is None:
                return await event_iterator.__anext__()
            return await asyncio.wait_for(event_iterator.__anext__(), timeout=wait_timeout)
        except asyncio.TimeoutError as exc:
            if deadline_is_limit:
                raise self._attempt_timeout_error() from exc
            raise StreamStalled(
                "Codex stream stalled after "
                f"{self._stall_timeout_seconds:g}s without an SSE event "
                "(inactivity timeout)"
            ) from exc

    def _attempt_timeout_error(self) -> StreamStalled:
        timeout_seconds = self._attempt_timeout_seconds
        assert timeout_seconds is not None and timeout_seconds > 0
        return StreamStalled(
            "Codex attempt timeout after "
            f"{timeout_seconds:g}s total wall-clock time"
        )

    async def _iter_sse_events(self, response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if line == "":
                if data_lines:
                    payload = "\n".join(data_lines).strip()
                    data_lines = []
                    if payload and payload != "[DONE]":
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict):
                            yield event
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            payload = "\n".join(data_lines).strip()
            if payload and payload != "[DONE]":
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    return
                if isinstance(event, dict):
                    yield event

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, StreamStalled):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {429, 500, 502, 503, 504}
        if isinstance(exc, RateLimitFailure):
            return True
        if isinstance(exc, RequestFailure):
            message = str(exc).lower()
            return any(
                term in message
                for term in [
                    "timeout",
                    "connect",
                    "network",
                    "rate",
                    "overloaded",
                    "you can retry",
                    "try again",
                    "temporarily",
                    "processing your request",
                    "internal error",
                    "server_is_overloaded",
                    "peer closed",
                    "incomplete",
                    "connection reset",
                    "reset by peer",
                    "closed connection",
                    "protocol",
                    "broken pipe",
                    "aborted",
                    "eof",
                    "unavailable",
                    "capacity",
                    "500",
                    "502",
                    "503",
                    "504",
                    "429",
                ]
            )
        # httpx.TransportError covers timeouts, network errors AND protocol
        # errors (e.g. RemoteProtocolError "peer closed connection") -- all transient.
        if isinstance(exc, httpx.TransportError):
            return True
        return False

    @staticmethod
    def _translate_error(exc: Exception) -> OpenHarnessApiError:
        if isinstance(exc, OpenHarnessApiError):
            return exc
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return _translate_status_error(status, str(exc))
        if isinstance(exc, httpx.HTTPError):
            return RequestFailure(str(exc))
        return RequestFailure(str(exc))
