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

import pytest

from openharness.mcp.client import McpClientManager
from openharness.mcp.types import McpStdioServerConfig

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
