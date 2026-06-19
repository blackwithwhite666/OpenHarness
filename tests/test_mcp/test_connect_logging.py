"""MCP connect-lifecycle logging.

A failed connect used to be invisible in journald — the reason lived only in
the in-memory ``McpConnectionStatus.detail`` (surfaced via ``mcp_status``). These
tests pin that connect outcomes (failure + success) are now logged, so e.g. an
expired OAuth bearer 401-ing at initialize is a one-line ``journalctl`` find
instead of a silent "no tools".
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import httpx
import pytest

import openharness.mcp.client as client_module
from openharness.mcp.client import McpClientManager
from openharness.mcp.types import McpHttpServerConfig, McpOAuthConfig, McpStdioServerConfig

_LOGGER = "openharness.mcp.client"


@pytest.mark.asyncio
async def test_failed_connect_logs_warning(caplog):
    manager = McpClientManager(
        {"broken": McpStdioServerConfig(command="/nonexistent/mcp-binary-xyz", args=[])}
    )
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await manager.connect_all()
    try:
        assert manager.list_statuses()[0].state == "failed"
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("broken" in m and "failed to connect" in m for m in msgs), msgs
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_successful_connect_logs_info(caplog):
    server_script = Path(__file__).resolve().parents[1] / "fixtures" / "fake_mcp_server.py"
    manager = McpClientManager(
        {"fixture": McpStdioServerConfig(command=sys.executable, args=[str(server_script)])}
    )
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        await manager.connect_all()
    try:
        assert manager.list_statuses()[0].state == "connected"
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert any("fixture" in m and "connected" in m for m in msgs), msgs
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_http_oauth_connect_prewarms_bearer_before_transport_and_sets_timeout(
    monkeypatch,
):
    calls: list[str] = []
    timeout_holder: dict[str, httpx.Timeout | None] = {}

    def fake_ensure_bearer(oauth, **kwargs):
        calls.append("prewarm")
        return "tok"

    def fake_transport(url, http_client=None):
        calls.append("transport")
        raise RuntimeError("stop")

    real_async_client = httpx.AsyncClient

    def spy_async_client(*args, **kwargs):
        timeout_holder["timeout"] = kwargs.get("timeout")
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("openharness.mcp.oauth.ensure_bearer", fake_ensure_bearer)
    monkeypatch.setattr(client_module, "streamable_http_client", fake_transport)
    monkeypatch.setattr(client_module.httpx, "AsyncClient", spy_async_client)

    cfg = McpHttpServerConfig(
        type="http",
        url="https://example.test/mcp",
        oauth=McpOAuthConfig(
            token_url="https://example.test/token",
            client_id="c",
            token_file="/tmp/nonexistent-mcp-token.json",
        ),
    )
    manager = McpClientManager({"w": cfg})

    try:
        await manager.connect_all()

        assert calls.index("prewarm") < calls.index("transport")
        assert timeout_holder["timeout"] is not None
    finally:
        await manager.close()
