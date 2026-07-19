"""Tests for web fetch and search tools."""

from __future__ import annotations

import time

import httpx
import pytest

from openharness.tools.base import ToolExecutionContext
from openharness.untrusted import UNTRUSTED_BANNER
from openharness.tools.web_fetch_tool import WebFetchTool, WebFetchToolInput, _html_to_text
from openharness.tools.web_search_tool import WebSearchTool, WebSearchToolInput
from openharness.utils.network_guard import fetch_public_http_response


@pytest.mark.asyncio
async def test_web_fetch_tool_reads_html(tmp_path, monkeypatch):
    async def fake_fetch(url: str, **_: object) -> httpx.Response:
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html><body><h1>OpenHarness Test</h1><p>web fetch works</p></body></html>",
            request=request,
        )

    monkeypatch.setitem(WebFetchTool.execute.__globals__, "fetch_public_http_response", fake_fetch)

    tool = WebFetchTool()
    result = await tool.execute(
        WebFetchToolInput(url="https://example.com/"),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is False
    assert result.output == (
        "URL: https://example.com/\n"
        "Status: 200\n"
        "Content-Type: text/html; charset=utf-8\n\n"
        f"{UNTRUSTED_BANNER}\n\n"
        "OpenHarness Test web fetch works"
    )


@pytest.mark.asyncio
async def test_web_search_tool_reads_results(tmp_path, monkeypatch):
    async def fake_fetch(url: str, **kwargs: object) -> httpx.Response:
        query = (kwargs.get("params") or {}).get("q", "")
        request = httpx.Request("GET", url, params=kwargs.get("params"))
        body = (
            "<html><body>"
            '<a class="result__a" href="https://example.com/docs">OpenHarness Docs</a>'
            '<div class="result__snippet">Search query was %s and docs were found.</div>'
            "</body></html>"
        ) % query
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=body,
            request=request,
        )

    monkeypatch.setitem(WebSearchTool.execute.__globals__, "fetch_public_http_response", fake_fetch)

    tool = WebSearchTool()
    result = await tool.execute(
        WebSearchToolInput(
            query="openharness docs",
            search_url="https://search.example.com/html",
        ),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is False
    assert result.output.startswith(f"{UNTRUSTED_BANNER}\n\nSearch results for:")
    assert result.output.count(UNTRUSTED_BANNER) == 1
    assert "OpenHarness Docs" in result.output
    assert "https://example.com/docs" in result.output
    assert "openharness docs" in result.output


def test_html_to_text_handles_large_html_quickly():
    html = "<html><head><style>.x{color:red}</style><script>var x=1;</script></head><body>"
    html += ("<div><span>Issue item</span><a href='/x'>link</a></div>" * 6000)
    html += "</body></html>"

    started = time.time()
    text = _html_to_text(html)
    elapsed = time.time() - started

    assert "Issue item" in text
    assert "var x=1" not in text
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_web_fetch_tool_rejects_embedded_credentials(tmp_path):
    tool = WebFetchTool()
    result = await tool.execute(
        WebFetchToolInput(url="https://user:pass@example.com/"),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is True
    assert "embedded credentials" in result.output


@pytest.mark.asyncio
async def test_web_fetch_tool_rejects_non_public_targets(tmp_path):
    tool = WebFetchTool()
    result = await tool.execute(
        WebFetchToolInput(url="http://127.0.0.1:8080/"),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is True
    assert UNTRUSTED_BANNER not in result.output
    assert "non-public" in result.output


@pytest.mark.asyncio
async def test_web_search_tool_uses_env_search_url(tmp_path, monkeypatch):
    calls = []

    async def fake_fetch(url: str, **kwargs: object) -> httpx.Response:
        calls.append((url, kwargs))
        request = httpx.Request("GET", url, params=kwargs.get("params"))
        body = (
            "<html><body>"
            '<a class="result__a" href="https://example.com/docs">OpenHarness Docs</a>'
            '<div class="result__snippet">Found through configured search.</div>'
            "</body></html>"
        )
        return httpx.Response(200, text=body, request=request)

    monkeypatch.setenv("OPENHARNESS_WEB_SEARCH_URL", "https://search.example.com/html")
    monkeypatch.setitem(WebSearchTool.execute.__globals__, "fetch_public_http_response", fake_fetch)

    tool = WebSearchTool()
    result = await tool.execute(WebSearchToolInput(query="openharness docs"), ToolExecutionContext(cwd=tmp_path))

    assert result.is_error is False
    assert calls[0][0] == "https://search.example.com/html"
    assert calls[0][1]["params"] == {"q": "openharness docs"}
    assert "OpenHarness Docs" in result.output


@pytest.mark.asyncio
async def test_fetch_public_http_response_uses_openharness_web_proxy(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, **kwargs: object) -> httpx.Response:
            request = httpx.Request("GET", url, params=kwargs.get("params"))
            return httpx.Response(200, text="ok", request=request)

    monkeypatch.setenv("OPENHARNESS_WEB_PROXY", "http://proxy.example.com:7890")
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    async def fake_ensure_public_http_url(url: str) -> None:
        return None

    monkeypatch.setattr("openharness.utils.network_guard.ensure_public_http_url", fake_ensure_public_http_url)

    response = await fetch_public_http_response("https://example.com/")

    assert response.status_code == 200
    assert seen["trust_env"] is False
    assert seen["proxy"] == "http://proxy.example.com:7890"


@pytest.mark.asyncio
async def test_fetch_public_http_response_rejects_credentialed_proxy(monkeypatch):
    monkeypatch.setenv("OPENHARNESS_WEB_PROXY", "http://user:pass@proxy.example.com:7890")

    with pytest.raises(ValueError, match="embedded credentials"):
        await fetch_public_http_response("https://example.com/")


@pytest.mark.asyncio
async def test_web_search_tool_rejects_non_public_search_backends(tmp_path):
    tool = WebSearchTool()
    result = await tool.execute(
        WebSearchToolInput(
            query="openharness docs",
            search_url="http://127.0.0.1:8080/search",
        ),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is True
    assert UNTRUSTED_BANNER not in result.output
    assert "non-public" in result.output


@pytest.mark.asyncio
async def test_get_capped_truncates_large_body():
    # An unbounded download would buffer the whole body into RAM (OOM on small
    # hosts). _get_capped must stop after max_bytes, yet still expose .text/.content.
    from openharness.utils.network_guard import _get_capped

    async def _chunks():
        for _ in range(100):  # 100 x 64KB = 6.4MB delivered in chunks (like a socket)
            yield b"x" * 64_000

    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=_chunks()))
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await _get_capped(client, "http://example.com/big", None, None, 1_000_000)
    assert len(resp.content) == 1_000_000  # capped at max_bytes, not the full 6.4MB
    assert resp.text  # .text works after the manual _content materialisation


@pytest.mark.asyncio
async def test_get_capped_redirect_body_not_read():
    # A redirect must be returned WITHOUT reading the body so the caller can follow
    # the Location header (the body of a 3xx is useless and may be large).
    from openharness.utils.network_guard import _get_capped

    transport = httpx.MockTransport(
        lambda req: httpx.Response(302, headers={"location": "http://example.com/next"})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        resp = await _get_capped(client, "http://example.com/r", None, None, 1000)
    assert resp.has_redirect_location
    assert resp.headers["location"] == "http://example.com/next"
