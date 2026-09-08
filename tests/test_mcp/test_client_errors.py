"""Tests for MCP client error handling on disconnected servers."""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from mcp.types import CallToolResult, TextContent

try:
    BaseExceptionGroup
except NameError:  # pragma: no cover - Python < 3.11 compatibility
    from exceptiongroup import BaseExceptionGroup

from openharness.mcp.client import (
    McpClientManager,
    McpServerNotConnectedError,
    McpToolTimeoutError,
)
from openharness.mcp.types import McpConnectionStatus, McpStdioServerConfig, McpToolInfo
from openharness.tools.base import ToolExecutionContext
from openharness.tools.mcp_tool import McpToolAdapter
from openharness.tools.read_mcp_resource_tool import ReadMcpResourceTool
from openharness.untrusted import UNTRUSTED_BANNER

_LOGGER = "openharness.mcp.client"


class _AsyncContextManager:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def _install_owned_session(
    manager: McpClientManager, name: str, session: AsyncMock
) -> asyncio.Event:
    """Install a session with a minimal owner task matching manager lifecycle."""
    shutdown = asyncio.Event()
    closed = asyncio.Event()

    async def _owner() -> None:
        await shutdown.wait()
        closed.set()

    manager._sessions[name] = session
    manager._shutdown_events[name] = shutdown
    manager._conn_tasks[name] = asyncio.create_task(_owner())
    await asyncio.sleep(0)
    return closed


def _text_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


# --- McpClientManager.call_tool ---


@pytest.mark.asyncio
async def test_call_tool_raises_when_server_never_connected():
    manager = McpClientManager({})
    with pytest.raises(McpServerNotConnectedError, match="not connected"):
        await manager.call_tool("missing", "some_tool", {})


@pytest.mark.asyncio
async def test_call_tool_raises_when_server_failed_to_connect():
    config = McpStdioServerConfig(command="false", args=[])
    manager = McpClientManager({"bad": config})
    manager._statuses["bad"] = McpConnectionStatus(
        name="bad", state="failed", detail="Connection refused",
    )
    with pytest.raises(McpServerNotConnectedError, match="Connection refused"):
        await manager.call_tool("bad", "tool", {})


@pytest.mark.asyncio
async def test_call_tool_raises_when_session_errors():
    manager = McpClientManager({})
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = RuntimeError("transport closed")
    manager._sessions["flaky"] = mock_session

    with pytest.raises(McpServerNotConnectedError, match="transport closed"):
        await manager.call_tool("flaky", "tool", {})


@pytest.mark.asyncio
async def test_call_tool_surfaces_http_status():
    class _HttpFailure(Exception):
        response = MagicMock(status_code=401)

    manager = McpClientManager({})
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = _HttpFailure()
    manager._sessions["auth"] = mock_session

    with pytest.raises(McpServerNotConnectedError) as exc_info:
        await manager.call_tool("auth", "tool", {})

    assert "HTTP 401" in str(exc_info.value)
    assert not str(exc_info.value).endswith("call failed:")


@pytest.mark.asyncio
async def test_call_tool_empty_exc_uses_type_name():
    class _Blank(Exception):
        def __str__(self):
            return ""

    manager = McpClientManager({})
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = _Blank()
    manager._sessions["blank"] = mock_session

    with pytest.raises(McpServerNotConnectedError) as exc_info:
        await manager.call_tool("blank", "tool", {})

    assert "_Blank" in str(exc_info.value)
    assert not str(exc_info.value).endswith("call failed:")


@pytest.mark.asyncio
async def test_call_tool_empty_exception_group_uses_first_inner():
    class _Blank(Exception):
        def __str__(self):
            return ""

    manager = McpClientManager({})
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = BaseExceptionGroup("", [_Blank()])
    manager._sessions["blank-group"] = mock_session

    with pytest.raises(McpServerNotConnectedError) as exc_info:
        await manager.call_tool("blank-group", "tool", {})

    assert "_Blank" in str(exc_info.value)
    assert not str(exc_info.value).endswith("call failed:")


@pytest.mark.asyncio
async def test_call_tool_times_out_when_session_hangs(monkeypatch):
    """A hung MCP backend must raise (not hang forever) so the turn gets a result."""
    monkeypatch.setenv("OPENHARNESS_MCP_TOOL_TIMEOUT", "0.05")
    manager = McpClientManager({})
    mock_session = AsyncMock()

    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    mock_session.call_tool.side_effect = _hang
    manager._sessions["slow"] = mock_session

    with pytest.raises(McpToolTimeoutError, match="timed out"):
        await manager.call_tool("slow", "tool", {})
    # subclass of McpServerNotConnectedError so existing handlers catch it too
    assert issubclass(McpToolTimeoutError, McpServerNotConnectedError)


@pytest.mark.asyncio
async def test_call_tool_timeout_reconnects_and_retries_once(monkeypatch):
    monkeypatch.setenv("OPENHARNESS_MCP_TOOL_TIMEOUT", "0.01")
    manager = McpClientManager(
        {"slow": McpStdioServerConfig(command="unused", args=[])}
    )
    stale_session = AsyncMock()

    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    stale_session.call_tool.side_effect = _hang
    closed = await _install_owned_session(manager, "slow", stale_session)
    replacement = AsyncMock()
    replacement.call_tool.return_value = _text_result("recovered")

    async def _connect(name, _config):
        manager._sessions[name] = replacement
        return replacement

    connect = AsyncMock(side_effect=_connect)
    monkeypatch.setattr(manager, "_connect_server", connect)

    assert await manager.call_tool("slow", "tool", {}) == "recovered"
    assert closed.is_set()
    assert stale_session.call_tool.await_count == 1
    assert replacement.call_tool.await_count == 1
    connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_tool_closed_transport_reconnects_and_retries(monkeypatch, caplog):
    manager = McpClientManager(
        {"flaky": McpStdioServerConfig(command="unused", args=[])}
    )
    stale_session = AsyncMock()
    stale_session.call_tool.side_effect = anyio.ClosedResourceError()
    await _install_owned_session(manager, "flaky", stale_session)
    replacement = AsyncMock()
    replacement.call_tool.return_value = _text_result("ok")

    async def _connect(name, _config):
        manager._sessions[name] = replacement
        return replacement

    connect = AsyncMock(side_effect=_connect)
    monkeypatch.setattr(manager, "_connect_server", connect)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert await manager.call_tool("flaky", "tool", {}) == "ok"

    assert stale_session.call_tool.await_count == 1
    assert replacement.call_tool.await_count == 1
    connect.assert_awaited_once()
    reconnect_logs = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "reconnecting" in record.getMessage()
    ]
    assert reconnect_logs == ["MCP server 'flaky' tool call failed; reconnecting"]


@pytest.mark.asyncio
async def test_call_tool_retry_failure_refreshes_for_future_without_third_call(monkeypatch):
    manager = McpClientManager(
        {"flaky": McpStdioServerConfig(command="unused", args=[])}
    )
    stale_session = AsyncMock()
    stale_session.call_tool.side_effect = RuntimeError("first transport closed")
    await _install_owned_session(manager, "flaky", stale_session)
    retry_session = AsyncMock()
    retry_session.call_tool.side_effect = RuntimeError("retry transport closed")
    future_session = AsyncMock()
    replacements = iter((retry_session, future_session))

    async def _connect(name, _config):
        replacement = next(replacements)
        manager._sessions[name] = replacement
        return replacement

    connect = AsyncMock(side_effect=_connect)
    monkeypatch.setattr(manager, "_connect_server", connect)

    with pytest.raises(McpServerNotConnectedError, match="retry transport closed"):
        await manager.call_tool("flaky", "tool", {})

    assert stale_session.call_tool.await_count == 1
    assert retry_session.call_tool.await_count == 1
    assert future_session.call_tool.await_count == 0
    assert connect.await_count == 2
    assert manager._sessions["flaky"] is future_session


@pytest.mark.asyncio
async def test_concurrent_stale_failures_share_one_reconnect(monkeypatch, caplog):
    manager = McpClientManager(
        {"flaky": McpStdioServerConfig(command="unused", args=[])}
    )
    stale_session = AsyncMock()
    stale_session.call_tool.side_effect = RuntimeError("transport closed")
    await _install_owned_session(manager, "flaky", stale_session)
    replacement = AsyncMock()
    replacement.call_tool.return_value = _text_result("ok")

    async def _connect(name, _config):
        await asyncio.sleep(0)
        manager._sessions[name] = replacement
        return replacement

    connect = AsyncMock(side_effect=_connect)
    monkeypatch.setattr(manager, "_connect_server", connect)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        outcomes = await asyncio.gather(
            manager.call_tool("flaky", "one", {}),
            manager.call_tool("flaky", "two", {}),
        )

    assert outcomes == ["ok", "ok"]
    assert stale_session.call_tool.await_count == 2
    assert replacement.call_tool.await_count == 2
    connect.assert_awaited_once()
    reconnect_logs = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "reconnecting" in record.getMessage()
    ]
    assert reconnect_logs == ["MCP server 'flaky' tool call failed; reconnecting"]


@pytest.mark.asyncio
async def test_call_tool_timeout_disabled_with_zero(monkeypatch):
    """OPENHARNESS_MCP_TOOL_TIMEOUT<=0 disables the timeout (call completes)."""
    monkeypatch.setenv("OPENHARNESS_MCP_TOOL_TIMEOUT", "0")
    manager = McpClientManager({})
    mock_session = AsyncMock()
    result = MagicMock()
    text_item = MagicMock()
    text_item.type = "text"
    text_item.text = "ok"
    result.content = [text_item]
    result.structuredContent = None
    mock_session.call_tool.return_value = result
    manager._sessions["s"] = mock_session

    assert await manager.call_tool("s", "tool", {}) == "ok"


@pytest.mark.asyncio
async def test_mcp_tool_adapter_returns_error_result_on_timeout(monkeypatch):
    """End-to-end poison-safety: a hung tool surfaces as an is_error result."""
    monkeypatch.setenv("OPENHARNESS_MCP_TOOL_TIMEOUT", "0.05")
    manager = McpClientManager({})
    mock_session = AsyncMock()

    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    mock_session.call_tool.side_effect = _hang
    manager._sessions["slow"] = mock_session
    tool_info = McpToolInfo(
        server_name="slow",
        name="hello",
        description="test",
        input_schema={"type": "object", "properties": {}},
    )
    adapter = McpToolAdapter(manager, tool_info)
    result = await adapter.execute(
        adapter.input_model.model_validate({}),
        ToolExecutionContext(cwd=Path(".")),
    )
    assert result.is_error is True
    assert "timed out" in result.output


@pytest.mark.asyncio
async def test_call_tool_includes_unknown_server_detail_for_unconfigured():
    """When the server name is not even in _statuses, detail says 'unknown server'."""
    manager = McpClientManager({})
    with pytest.raises(McpServerNotConnectedError, match="unknown server"):
        await manager.call_tool("ghost", "tool", {})


# --- McpClientManager.call_tool_result: tool-declared isError preservation ---


def _manager_with_result(result: CallToolResult) -> McpClientManager:
    manager = McpClientManager({})
    mock_session = AsyncMock()
    mock_session.call_tool.return_value = result
    manager._sessions["srv"] = mock_session
    return manager


@pytest.mark.asyncio
async def test_call_tool_result_success_is_not_error():
    manager = _manager_with_result(
        CallToolResult(content=[TextContent(type="text", text="payload")], isError=False)
    )

    outcome = await manager.call_tool_result("srv", "tool", {})

    assert outcome.is_error is False
    assert outcome.output == "payload"


@pytest.mark.asyncio
async def test_call_tool_result_preserves_tool_declared_error_body():
    manager = _manager_with_result(
        CallToolResult(
            content=[TextContent(type="text", text="interval must not exceed 31 days")],
            isError=True,
        )
    )

    outcome = await manager.call_tool_result("srv", "tool", {})

    assert outcome.is_error is True
    assert outcome.output == "interval must not exceed 31 days"


@pytest.mark.asyncio
async def test_call_tool_still_returns_body_string_for_direct_callers():
    manager = _manager_with_result(
        CallToolResult(
            content=[TextContent(type="text", text="interval must not exceed 31 days")],
            isError=True,
        )
    )

    assert await manager.call_tool("srv", "tool", {}) == "interval must not exceed 31 days"


@pytest.mark.asyncio
async def test_call_tool_result_raises_on_disconnected_server():
    manager = McpClientManager({})
    with pytest.raises(McpServerNotConnectedError, match="not connected"):
        await manager.call_tool_result("missing", "some_tool", {})


@pytest.mark.asyncio
async def test_call_tool_result_times_out_when_session_hangs(monkeypatch):
    monkeypatch.setenv("OPENHARNESS_MCP_TOOL_TIMEOUT", "0.05")
    manager = McpClientManager({})
    mock_session = AsyncMock()

    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(30)

    mock_session.call_tool.side_effect = _hang
    manager._sessions["slow"] = mock_session

    with pytest.raises(McpToolTimeoutError, match="timed out"):
        await manager.call_tool_result("slow", "tool", {})


@pytest.mark.asyncio
async def test_call_tool_result_raises_on_transport_error_not_misclassified():
    manager = McpClientManager({})
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = RuntimeError("transport closed")
    manager._sessions["flaky"] = mock_session

    with pytest.raises(McpServerNotConnectedError, match="transport closed"):
        await manager.call_tool_result("flaky", "tool", {})


# --- McpClientManager.read_resource ---


@pytest.mark.asyncio
async def test_read_resource_raises_when_server_never_connected():
    manager = McpClientManager({})
    with pytest.raises(McpServerNotConnectedError, match="not connected"):
        await manager.read_resource("missing", "res://data")


@pytest.mark.asyncio
async def test_read_resource_raises_when_session_errors():
    manager = McpClientManager({})
    mock_session = AsyncMock()
    mock_session.read_resource.side_effect = OSError("broken pipe")
    manager._sessions["flaky"] = mock_session

    with pytest.raises(McpServerNotConnectedError, match="broken pipe"):
        await manager.read_resource("flaky", "res://data")


@pytest.mark.asyncio
async def test_read_resource_surfaces_http_status():
    class _HttpFailure(Exception):
        response = MagicMock(status_code=401)

    manager = McpClientManager({})
    mock_session = AsyncMock()
    mock_session.read_resource.side_effect = _HttpFailure()
    manager._sessions["auth"] = mock_session

    with pytest.raises(McpServerNotConnectedError) as exc_info:
        await manager.read_resource("auth", "res://data")

    assert "HTTP 401" in str(exc_info.value)
    assert not str(exc_info.value).endswith("resource read failed:")


@pytest.mark.asyncio
async def test_register_connected_session_tolerates_missing_resources_list():
    manager = McpClientManager({})
    session = AsyncMock()
    session.initialize.return_value = None
    session.list_tools.return_value.tools = []
    session.list_resources.side_effect = RuntimeError("Method not found")
    stack = AsyncExitStack()
    await stack.__aenter__()
    stack.enter_async_context = AsyncMock(return_value=session)

    await manager._register_connected_session(
        name="context7",
        config=McpStdioServerConfig(command="npx", args=[]),
        stack=stack,
        read_stream=object(),
        write_stream=object(),
        auth_configured=False,
    )

    assert manager._statuses["context7"].state == "connected"
    assert manager._statuses["context7"].resources == []


@pytest.mark.asyncio
async def test_close_signals_owner_tasks_and_clears_state():
    """close() must set each connection's shutdown event, await its owner task
    (so the AsyncExitStack is closed in the task that opened it), and clear all
    per-connection state."""
    manager = McpClientManager({})
    shutdown = asyncio.Event()
    closed: dict[str, bool] = {}

    async def fake_owner() -> None:
        await shutdown.wait()
        closed["done"] = True

    task = asyncio.create_task(fake_owner())
    await asyncio.sleep(0)  # let the owner task start and block on the event
    manager._shutdown_events["context7"] = shutdown
    manager._conn_tasks["context7"] = task
    manager._sessions["context7"] = AsyncMock()

    await manager.close()

    assert closed.get("done") is True  # owner task ran to completion (in-task close)
    assert manager._conn_tasks == {}
    assert manager._shutdown_events == {}
    assert manager._sessions == {}


@pytest.mark.asyncio
async def test_close_failed_stack_suppresses_cross_task_runtime_error():
    """The cancel-scope RuntimeError (now raised inside the owner task during
    aclose) must be swallowed, never crash the gateway."""
    manager = McpClientManager({})
    stack = MagicMock()
    stack.aclose = AsyncMock(
        side_effect=RuntimeError(
            "Attempted to exit cancel scope in a different task than it was entered in"
        )
    )
    await manager._close_failed_stack(stack)  # must not raise


@pytest.mark.asyncio
async def test_close_failed_stack_suppresses_cancelled_error():
    manager = McpClientManager({})
    stack = MagicMock()
    stack.aclose = AsyncMock(side_effect=asyncio.CancelledError())
    await manager._close_failed_stack(stack)  # must not raise


@pytest.mark.asyncio
async def test_close_failed_stack_suppresses_base_exception_group_cleanup_error():
    manager = McpClientManager({})
    stack = MagicMock()
    stack.aclose = AsyncMock(
        side_effect=BaseExceptionGroup(
            "cleanup failed",
            [asyncio.CancelledError()],
        )
    )

    await manager._close_failed_stack(stack)


@pytest.mark.asyncio
async def test_connect_all_marks_http_server_failed_when_initialize_is_cancelled(monkeypatch):
    import openharness.mcp.client as client_module
    from openharness.mcp.types import McpHttpServerConfig

    manager = McpClientManager(
        {
            "broken-http": McpHttpServerConfig(
                url="http://127.0.0.1:9999/mcp",
                headers={},
            )
        }
    )

    monkeypatch.setattr(
        client_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: _AsyncContextManager(AsyncMock()),
    )
    monkeypatch.setattr(
        client_module,
        "streamable_http_client",
        lambda *args, **kwargs: _AsyncContextManager((object(), object(), AsyncMock())),
    )
    manager._register_connected_session = AsyncMock(
        side_effect=asyncio.CancelledError("simulated cancellation")
    )

    await manager.connect_all()

    status = manager.list_statuses()[0]
    assert status.name == "broken-http"
    assert status.state == "failed"
    assert "simulated cancellation" in status.detail


# --- McpToolAdapter catches error and returns ToolResult(is_error=True) ---


@pytest.mark.asyncio
async def test_mcp_tool_adapter_returns_error_result_on_disconnected_server():
    manager = McpClientManager({})
    tool_info = McpToolInfo(
        server_name="gone",
        name="hello",
        description="test",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    adapter = McpToolAdapter(manager, tool_info)
    result = await adapter.execute(
        adapter.input_model.model_validate({"x": "1"}),
        ToolExecutionContext(cwd=Path(".")),
    )
    assert result.is_error is True
    assert UNTRUSTED_BANNER not in result.output
    assert "not connected" in result.output


# --- ReadMcpResourceTool catches error and returns ToolResult(is_error=True) ---


@pytest.mark.asyncio
async def test_read_mcp_resource_tool_returns_error_result_on_disconnected_server():
    manager = McpClientManager({})
    tool = ReadMcpResourceTool(manager)
    result = await tool.execute(
        tool.input_model.model_validate({"server": "gone", "uri": "res://x"}),
        ToolExecutionContext(cwd=Path(".")),
    )
    assert result.is_error is True
    assert UNTRUSTED_BANNER not in result.output
    assert "not connected" in result.output
