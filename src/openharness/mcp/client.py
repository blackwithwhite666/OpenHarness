"""MCP client manager."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, ReadResourceResult

from openharness.mcp.types import (
    McpConnectionStatus,
    McpHttpServerConfig,
    McpResourceInfo,
    McpStdioServerConfig,
    McpToolInfo,
)

log = logging.getLogger(__name__)

# A slow / hung MCP backend must never block a tool call forever: an unanswered
# tool call leaves a dangling tool_use in the conversation and poisons the whole
# session (the model API then rejects every subsequent turn with "No tool output
# found for function call ..."). Bound every call with a timeout. Configurable
# via OPENHARNESS_MCP_TOOL_TIMEOUT (seconds); <= 0 disables.
_DEFAULT_MCP_TOOL_TIMEOUT = 120.0


def _describe_mcp_exc(exc: BaseException) -> str:
    """Human-readable failure detail that is never empty.

    Prefer an HTTP status code when the exception carries an HTTP response;
    otherwise fall back to its message and finally its type name.
    """
    try:
        type_name = type(exc).__name__ or "BaseException"
    except BaseException:
        type_name = "BaseException"

    try:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    except BaseException:
        status = None

    try:
        text = (str(exc) or "").strip()
    except BaseException:
        text = ""

    if status is not None:
        try:
            status_text = str(status).strip() or type(status).__name__
        except BaseException:
            status_text = "unknown"
        return f"HTTP {status_text}: {text or type_name}"

    try:
        inner_exceptions = getattr(exc, "exceptions", ())
    except BaseException:
        inner_exceptions = ()
    if isinstance(inner_exceptions, tuple) and inner_exceptions:
        # ExceptionGroup.__str__ adds only a count when its message is empty;
        # that summary is no more useful than an empty exception message.
        try:
            group_message = getattr(exc, "message", None)
            group_message_is_empty = group_message is not None and not str(
                group_message
            ).strip()
        except BaseException:
            group_message_is_empty = False
        if not text or text == type_name or group_message_is_empty:
            try:
                inner = inner_exceptions[0]
                if isinstance(inner, BaseException) and inner is not exc:
                    return _describe_mcp_exc(inner)
            except BaseException:
                pass

    return text or type_name


def _mcp_tool_timeout() -> float | None:
    raw = os.environ.get("OPENHARNESS_MCP_TOOL_TIMEOUT")
    if raw is None or raw.strip() == "":
        return _DEFAULT_MCP_TOOL_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_MCP_TOOL_TIMEOUT
    return None if value <= 0 else value


def _auth_configured_for(config: object) -> bool:
    """Whether a server config carries any auth (for status reporting)."""
    if isinstance(config, McpStdioServerConfig):
        return bool(config.env)
    return bool(getattr(config, "headers", None) or getattr(config, "oauth", None))


class McpServerNotConnectedError(Exception):
    """Raised when an MCP server is not connected or its session has been lost."""


@dataclass(frozen=True)
class McpToolCallResult:
    """Stringified MCP tool output plus the tool-declared error flag.

    ``is_error`` mirrors ``CallToolResult.isError``: the call succeeded at the
    transport level but the tool itself declared failure (e.g. argument
    validation). Transport failures still raise ``McpServerNotConnectedError``.
    """

    output: str
    is_error: bool = False


class McpToolTimeoutError(McpServerNotConnectedError):
    """Raised when an MCP tool call exceeds the configured timeout."""


class _OAuthBearerAuth(httpx.Auth):
    """Inject a fresh OAuth bearer on every request.

    The MCP streamable-HTTP transport opens one long-lived connection, so a
    bearer set once at connect time goes stale when the (often short-lived)
    access token expires — every later request then 401s until the process
    restarts. Refreshing per request (a cheap file read while the token is still
    valid; a lock-serialized refresh near expiry) keeps the connection usable
    across token rotations without reconnecting. A 401 also force-refreshes the
    bearer and retries the request once.
    """

    def __init__(self, oauth) -> None:
        self._oauth = oauth

    def sync_auth_flow(self, request):
        from openharness.mcp.oauth import ensure_bearer

        token = ensure_bearer(self._oauth)
        request.headers[self._oauth.header] = f"Bearer {token}"
        response = yield request
        if response.status_code == 401:
            token = ensure_bearer(self._oauth, force=True, stale_token=token)
            request.headers[self._oauth.header] = f"Bearer {token}"
            yield request

    async def async_auth_flow(self, request):
        from openharness.mcp.oauth import ensure_bearer

        token = await asyncio.to_thread(ensure_bearer, self._oauth)
        request.headers[self._oauth.header] = f"Bearer {token}"
        response = yield request
        if response.status_code == 401:
            token = await asyncio.to_thread(
                lambda: ensure_bearer(self._oauth, force=True, stale_token=token)
            )
            request.headers[self._oauth.header] = f"Bearer {token}"
            yield request


class McpClientManager:
    """Manage MCP connections and expose tools/resources."""

    def __init__(self, server_configs: dict[str, object]) -> None:
        self._server_configs = server_configs
        self._statuses: dict[str, McpConnectionStatus] = {
            name: McpConnectionStatus(
                name=name,
                state="pending",
                transport=getattr(config, "type", "unknown"),
            )
            for name, config in server_configs.items()
        }
        self._sessions: dict[str, ClientSession] = {}
        # Each connection's transport (stdio/http) and ClientSession open an anyio
        # task group / cancel scope, which anyio requires to be entered and exited
        # in the SAME task. So each connection is owned by one dedicated task that
        # enters the stack, waits on its shutdown event, then closes the stack —
        # never across tasks. The old design closed stacks from whatever task ran
        # close()/interrupt (or the async-generator GC), raising "Attempted to exit
        # cancel scope in a different task" and crashing the whole gateway.
        self._conn_tasks: dict[str, asyncio.Task] = {}
        self._shutdown_events: dict[str, asyncio.Event] = {}

    async def connect_all(self) -> None:
        """Connect all configured MCP servers supported by the current build."""
        for name, config in self._server_configs.items():
            if isinstance(config, (McpStdioServerConfig, McpHttpServerConfig)):
                shutdown = asyncio.Event()
                ready = asyncio.Event()
                self._shutdown_events[name] = shutdown
                self._conn_tasks[name] = asyncio.create_task(
                    self._serve(name, config, shutdown, ready),
                    name=f"mcp-conn:{name}",
                )
                # Wait for this connection's connect attempt to finish (success or
                # failure) before the next, preserving sequential-connect order.
                await ready.wait()
            else:
                detail = f"Unsupported MCP transport in current build: {config.type}"
                log.warning("MCP server %r not connected: %s", name, detail)
                self._statuses[name] = McpConnectionStatus(
                    name=name,
                    state="failed",
                    transport=config.type,
                    auth_configured=bool(getattr(config, "headers", None)),
                    detail=detail,
                )

    async def reconnect_all(self) -> None:
        """Reconnect all configured servers."""
        await self.close()
        self._statuses = {
            name: McpConnectionStatus(name=name, state="pending", transport=getattr(config, "type", "unknown"))
            for name, config in self._server_configs.items()
        }
        await self.connect_all()

    def update_server_config(self, name: str, config: object) -> None:
        """Replace one server config in memory."""
        self._server_configs[name] = config

    def get_server_config(self, name: str) -> object | None:
        """Return one configured server object if present."""
        return self._server_configs.get(name)

    async def _close_failed_stack(self, stack: AsyncExitStack) -> None:
        """Best-effort cleanup for a connection attempt that never finished."""
        try:
            await stack.aclose()
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise

    def _mark_connection_failed(
        self,
        name: str,
        config: object,
        *,
        auth_configured: bool,
        exc: BaseException,
    ) -> None:
        """Record one MCP connection failure without aborting startup."""
        detail = str(exc) or exc.__class__.__name__
        # Log it: the status detail is otherwise in-memory only (surfaced via
        # mcp_status), so a failed connect — e.g. an expired OAuth bearer 401-ing
        # at initialize — is invisible in journald. One warning turns a silent
        # "no tools" into a one-line diagnosis.
        log.warning(
            "MCP server %r failed to connect (%s): %s",
            name,
            getattr(config, "type", "unknown"),
            detail,
        )
        self._statuses[name] = McpConnectionStatus(
            name=name,
            state="failed",
            transport=getattr(config, "type", "unknown"),
            auth_configured=auth_configured,
            detail=detail,
        )

    async def close(self) -> None:
        """Close all active MCP sessions.

        Signal every owner task to shut down, then await them so each
        ``AsyncExitStack`` is closed inside the same task that opened it.
        """
        for event in self._shutdown_events.values():
            event.set()
        tasks = list(self._conn_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._conn_tasks.clear()
        self._shutdown_events.clear()
        self._sessions.clear()

    def list_statuses(self) -> list[McpConnectionStatus]:
        """Return statuses for all configured servers."""
        return [self._statuses[name] for name in sorted(self._statuses)]

    def list_tools(self) -> list[McpToolInfo]:
        """Return all connected MCP tools."""
        tools: list[McpToolInfo] = []
        for status in self.list_statuses():
            tools.extend(status.tools)
        return tools

    def list_resources(self) -> list[McpResourceInfo]:
        """Return all connected MCP resources."""
        resources: list[McpResourceInfo] = []
        for status in self.list_statuses():
            resources.extend(status.resources)
        return resources

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Invoke one MCP tool and stringify the result.

        Returns the textual body whether or not the tool declared an error;
        use ``call_tool_result`` when the ``isError`` flag matters.
        """
        return (await self.call_tool_result(server_name, tool_name, arguments)).output

    async def call_tool_result(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> McpToolCallResult:
        """Invoke one MCP tool, preserving the tool-level ``isError`` flag."""
        session = self._sessions.get(server_name)
        if session is None:
            status = self._statuses.get(server_name)
            detail = status.detail if status else "unknown server"
            raise McpServerNotConnectedError(
                f"MCP server '{server_name}' is not connected: {detail}"
            )
        timeout = _mcp_tool_timeout()
        try:
            if timeout is None:
                result: CallToolResult = await session.call_tool(tool_name, arguments)
            else:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments), timeout=timeout
                )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise McpToolTimeoutError(
                f"MCP server '{server_name}' tool '{tool_name}' timed out after "
                f"{timeout:.0f}s (set OPENHARNESS_MCP_TOOL_TIMEOUT to adjust)"
            ) from exc
        except Exception as exc:
            raise McpServerNotConnectedError(
                f"MCP server '{server_name}' call failed: {_describe_mcp_exc(exc)}"
            ) from exc
        parts: list[str] = []
        for item in result.content:
            if getattr(item, "type", None) == "text":
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(item.model_dump_json())
        if result.structuredContent and not parts:
            parts.append(str(result.structuredContent))
        if not parts:
            parts.append("(no output)")
        # `is True` (not bool()): mocked sessions return a truthy MagicMock for
        # unset isError, while the real CallToolResult field is always a bool.
        return McpToolCallResult(
            output="\n".join(parts).strip(),
            is_error=getattr(result, "isError", False) is True,
        )

    async def read_resource(self, server_name: str, uri: str) -> str:
        """Read one MCP resource and stringify the response."""
        session = self._sessions.get(server_name)
        if session is None:
            status = self._statuses.get(server_name)
            detail = status.detail if status else "unknown server"
            raise McpServerNotConnectedError(
                f"MCP server '{server_name}' is not connected: {detail}"
            )
        try:
            result: ReadResourceResult = await session.read_resource(uri)
        except Exception as exc:
            raise McpServerNotConnectedError(
                f"MCP server '{server_name}' resource read failed: "
                f"{_describe_mcp_exc(exc)}"
            ) from exc
        parts: list[str] = []
        for item in result.contents:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(getattr(item, "blob", "")))
        return "\n".join(parts).strip()

    async def _serve(
        self,
        name: str,
        config: object,
        shutdown: asyncio.Event,
        ready: asyncio.Event,
    ) -> None:
        """Own one MCP connection for its whole lifetime in a single task.

        Opens the transport + ClientSession, signals ``ready``, then holds the
        stack open until ``shutdown`` is set — so each anyio task group / cancel
        scope is entered and exited in this one task, never across tasks.
        """
        stack = AsyncExitStack()
        try:
            if isinstance(config, McpStdioServerConfig):
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=config.command,
                            args=config.args,
                            env=config.env,
                            cwd=config.cwd,
                        )
                    )
                )
                auth_configured = bool(config.env)
            else:  # McpHttpServerConfig
                headers = dict(config.headers or {})
                # OAuth bearer is injected per request (auth=) rather than as a
                # static header, so it auto-refreshes on the long-lived connection
                # instead of going stale and 401-ing after the access token's TTL.
                oauth = getattr(config, "oauth", None)
                auth = _OAuthBearerAuth(oauth) if oauth else None
                if oauth is not None:
                    # Pre-warm the bearer BEFORE opening the transport so the
                    # connect's first request (initialize) never blocks on an
                    # inline token refresh. The access token is short-lived (~1h);
                    # a connect that coincided with its hourly refresh otherwise ran
                    # the refresh inline during initialize and blew the client's
                    # connect timeout -> "Cancelled via cancel scope", leaving the
                    # server unconnected for the bundle's lifetime. Best-effort: on
                    # failure, log and let the per-request auth flow surface the real
                    # error at initialize.
                    from openharness.mcp.oauth import ensure_bearer

                    try:
                        await asyncio.to_thread(ensure_bearer, oauth)
                    except Exception as exc:  # noqa: BLE001 - never abort connect here
                        log.warning("MCP server %r OAuth bearer pre-warm failed: %s", name, exc)
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=headers or None,
                        auth=auth,
                        # The MCP SDK ignores its own timeout/sse_read_timeout once a
                        # custom http_client is passed, so set an explicit, generous
                        # timeout here (httpx's 5s default is too tight for a connect
                        # that may still refresh a token). Tool calls are separately
                        # bounded by OPENHARNESS_MCP_TOOL_TIMEOUT.
                        timeout=httpx.Timeout(30.0, connect=10.0),
                    )
                )
                read_stream, write_stream, _get_session_id = await stack.enter_async_context(
                    streamable_http_client(config.url, http_client=http_client)
                )
                auth_configured = bool(config.headers or oauth)
            await self._register_connected_session(
                name=name,
                config=config,
                stack=stack,
                read_stream=read_stream,
                write_stream=write_stream,
                auth_configured=auth_configured,
            )
        except (KeyboardInterrupt, SystemExit):
            await self._close_failed_stack(stack)
            ready.set()
            raise
        except BaseException as exc:
            self._mark_connection_failed(
                name,
                config,
                auth_configured=_auth_configured_for(config),
                exc=exc,
            )
            await self._close_failed_stack(stack)
            ready.set()
            return
        # Connected. Unblock connect_all, then keep the stack open in THIS task
        # until shutdown so the transport/session cancel scopes exit where they
        # were entered.
        ready.set()
        try:
            await shutdown.wait()
        finally:
            await self._close_failed_stack(stack)

    async def _register_connected_session(
        self,
        *,
        name: str,
        config: object,
        stack: AsyncExitStack,
        read_stream: Any,
        write_stream: Any,
        auth_configured: bool,
    ) -> None:
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        tool_result = await session.list_tools()
        resource_result = None
        try:
            resource_result = await session.list_resources()
        except Exception as exc:
            if "Method not found" not in str(exc):
                raise
        tools = [
            McpToolInfo(
                server_name=name,
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema or {"type": "object", "properties": {}}),
            )
            for tool in tool_result.tools
        ]
        resources = [
            McpResourceInfo(
                server_name=name,
                name=resource.name or str(resource.uri),
                uri=str(resource.uri),
                description=resource.description or "",
            )
            for resource in (resource_result.resources if resource_result is not None else [])
        ]
        self._sessions[name] = session
        self._statuses[name] = McpConnectionStatus(
            name=name,
            state="connected",
            transport=getattr(config, "type", "unknown"),
            auth_configured=auth_configured,
            tools=tools,
            resources=resources,
        )
        log.info(
            "MCP server %r connected (%s): %d tools, %d resources",
            name,
            getattr(config, "type", "unknown"),
            len(tools),
            len(resources),
        )
