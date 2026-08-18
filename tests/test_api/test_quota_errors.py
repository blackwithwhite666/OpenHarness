"""Quota-exceeded classification: detect, log at ERROR, and never retry.

A provider "usage limit"/quota error (Codex 429 "The usage limit has been
reached", Kimi 403 "usage limit ... billing cycle", OpenAI 429
insufficient_quota) is subscription-scoped: retrying with a short backoff can
never succeed, and the retry storm used to leave only INFO-level "retrying"
noise — or a swallowed error for suppressed reminder turns — instead of a
clear ERROR per request.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from openharness.api.client import ApiMessageRequest, ApiRetryEvent
from openharness.api.codex_client import CodexApiClient, _quota_aware_request_failure
from openharness.api.errors import QuotaExceededError, RequestFailure, is_quota_error_message
from openharness.api.openai_client import OpenAICompatibleClient
from openharness.engine.messages import ConversationMessage


def _b64url(data: dict[str, object]) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _fake_codex_token() -> str:
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct_test"}}
    return f"{_b64url({'alg': 'none', 'typ': 'JWT'})}.{_b64url(payload)}.sig"


def _request() -> ApiMessageRequest:
    return ApiMessageRequest(
        model="k3",
        messages=[ConversationMessage.from_user_text("hi")],
    )


class TestIsQuotaErrorMessage:
    def test_codex_usage_limit(self):
        assert is_quota_error_message("The usage limit has been reached")

    def test_kimi_billing_cycle(self):
        assert is_quota_error_message(
            "Error code: 403 - {'error': {'message': \"You've reached your usage "
            "limit for this billing cycle. Your quota will be refreshed in the "
            "next cycle.\"}}"
        )

    def test_openai_insufficient_quota(self):
        assert is_quota_error_message(
            "Error code: 429 - You exceeded your current quota, please check "
            "your plan and billing details."
        )

    def test_transient_rate_limit_is_not_quota(self):
        assert not is_quota_error_message("Rate limit exceeded. Please retry later.")

    def test_kimi_agent_header_403_is_not_quota(self):
        # The other kimi 403: missing KimiCLI headers ("only available for
        # Coding Agents") is a client misconfiguration, not a quota.
        assert not is_quota_error_message("only available for Coding Agents")

    def test_generic_server_error_is_not_quota(self):
        assert not is_quota_error_message("Error code: 500 - internal error")

    def test_empty_message(self):
        assert not is_quota_error_message("")


def test_quota_aware_request_failure_classifies_by_message():
    quota = _quota_aware_request_failure("The usage limit has been reached")
    assert isinstance(quota, QuotaExceededError)

    transient = _quota_aware_request_failure("Upstream overloaded (code=overloaded)")
    assert isinstance(transient, RequestFailure)
    assert not isinstance(transient, QuotaExceededError)


class _FakeStreamResponse:
    def __init__(self, *, status_code: int = 200, lines: list[str] | None = None, body: str = "") -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.request = httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses")
        self._lines = lines or []
        self._body = body.encode("utf-8")

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def aread(self) -> bytes:
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _CountingAsyncClient:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response
        self.attempts = 0

    async def __aenter__(self) -> "_CountingAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def stream(self, method: str, url: str, *, headers: dict[str, str], json: dict[str, Any]):
        self.attempts += 1
        return self._response


class TestCodexQuotaFailFast:
    @pytest.mark.asyncio
    async def test_usage_limit_429_is_not_retried(self, monkeypatch):
        monkeypatch.setattr("openharness.api.codex_client.BASE_DELAY_SECONDS", 0.0)
        monkeypatch.setattr("openharness.api.codex_client.MAX_DELAY_SECONDS", 0.0)
        response = _FakeStreamResponse(
            status_code=429,
            body=json.dumps({"error": {"message": "The usage limit has been reached"}}),
        )
        fake = _CountingAsyncClient(response)
        monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", lambda *a, **k: fake)

        client = CodexApiClient(_fake_codex_token())
        events = []
        with pytest.raises(QuotaExceededError) as error:
            async for event in client.stream_message(_request()):
                events.append(event)

        assert fake.attempts == 1  # failed fast, no retry storm
        assert "usage limit" in str(error.value).lower()
        assert not any(isinstance(event, ApiRetryEvent) for event in events)

    @pytest.mark.asyncio
    async def test_usage_limit_sse_error_is_not_retried(self, monkeypatch):
        response = _FakeStreamResponse(
            lines=[
                'data: {"type":"error","message":"The usage limit has been reached","code":"usage_limit_reached"}',
                "",
            ]
        )
        fake = _CountingAsyncClient(response)
        monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", lambda *a, **k: fake)

        client = CodexApiClient(_fake_codex_token())
        with pytest.raises(QuotaExceededError):
            async for _ in client.stream_message(_request()):
                pass

        assert fake.attempts == 1

    @pytest.mark.asyncio
    async def test_transient_429_is_still_retried(self, monkeypatch):
        monkeypatch.setattr("openharness.api.codex_client.BASE_DELAY_SECONDS", 0.0)
        monkeypatch.setattr("openharness.api.codex_client.MAX_DELAY_SECONDS", 0.0)
        response = _FakeStreamResponse(
            status_code=429,
            body=json.dumps({"error": {"message": "Rate limit exceeded. Please retry later."}}),
        )
        fake = _CountingAsyncClient(response)
        monkeypatch.setattr("openharness.api.codex_client.httpx.AsyncClient", lambda *a, **k: fake)

        client = CodexApiClient(_fake_codex_token())
        with pytest.raises(Exception) as error:
            async for _ in client.stream_message(_request()):
                pass

        assert not isinstance(error.value, QuotaExceededError)
        assert fake.attempts > 1  # retry loop still engages for transient 429


class TestOpenAICompatibleQuotaFailFast:
    @staticmethod
    def _client_with_transport(handler) -> tuple[OpenAICompatibleClient, httpx.AsyncClient]:
        transport = httpx.MockTransport(handler)
        http_client = httpx.AsyncClient(transport=transport)
        client = OpenAICompatibleClient(api_key="test-key", base_url="https://api.kimi.com/coding/v1")
        client._client._client = http_client
        return client, http_client

    @pytest.mark.asyncio
    async def test_kimi_403_usage_limit_is_not_retried(self):
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            return httpx.Response(
                403,
                json={
                    "error": {
                        "message": (
                            "You've reached your usage limit for this billing "
                            "cycle. Your quota will be refreshed in the next "
                            "cycle. Try again later."
                        )
                    }
                },
            )

        client, http_client = self._client_with_transport(handler)
        try:
            events = []
            with pytest.raises(QuotaExceededError) as error:
                async for event in client.stream_message(_request()):
                    events.append(event)

            assert calls["count"] == 1  # failed fast, no retry storm
            assert "usage limit" in str(error.value).lower()
            assert not any(isinstance(event, ApiRetryEvent) for event in events)
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_kimi_403_agent_headers_error_is_authentication_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={"error": {"message": "only available for Coding Agents"}},
            )

        client, http_client = self._client_with_transport(handler)
        try:
            with pytest.raises(Exception) as error:
                async for _ in client.stream_message(_request()):
                    pass

            from openharness.api.errors import AuthenticationFailure

            assert isinstance(error.value, AuthenticationFailure)
            assert not isinstance(error.value, QuotaExceededError)
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_transient_429_is_still_retried(self, monkeypatch):
        # The openai SDK retries 429s internally, so drive the retry loop with
        # a fake SDK client whose first create() raises a transient 429 — the
        # wrapper must still engage its own retry loop for it.
        monkeypatch.setattr("openharness.api.openai_client.BASE_DELAY", 0.0)
        monkeypatch.setattr("openharness.api.openai_client.MAX_DELAY", 0.0)

        class _Transient429(Exception):
            status_code = 429

        class _FlakyCompletions:
            def __init__(self) -> None:
                self.calls = 0

            async def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise _Transient429("Rate limit exceeded. Please retry later.")

                async def _stream():
                    yield SimpleNamespace(choices=[], usage=None)

                return _stream()

        class _FlakyChat:
            def __init__(self) -> None:
                self.completions = _FlakyCompletions()

        class _FlakySDK:
            def __init__(self) -> None:
                self.chat = _FlakyChat()

        client = OpenAICompatibleClient(api_key="test-key")
        fake_sdk = _FlakySDK()
        client._client = fake_sdk

        events = [event async for event in client.stream_message(_request())]

        assert fake_sdk.chat.completions.calls == 2  # one transient failure, one retry
        assert any(isinstance(event, ApiRetryEvent) for event in events)
