"""MCP tool adapters."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable

from pydantic import BaseModel, Field, create_model

from openharness.mcp.client import McpClientManager, McpServerNotConnectedError
from openharness.mcp.types import McpToolInfo
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from openharness.untrusted import UNTRUSTED_BANNER


class McpToolAdapter(BaseTool):
    """Expose one MCP tool as a normal OpenHarness tool."""

    def __init__(self, manager: McpClientManager, tool_info: McpToolInfo) -> None:
        self._manager = manager
        self._tool_info = tool_info
        server_segment = _sanitize_tool_segment(tool_info.server_name)
        tool_segment = _sanitize_tool_segment(tool_info.name)
        self.name = f"mcp__{server_segment}__{tool_segment}"
        self.description = tool_info.description or f"MCP tool {tool_info.name}"
        self.input_model = _input_model_from_schema(self.name, tool_info.input_schema)

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        del context
        return await self._execute_payload(arguments.model_dump(mode="json", exclude_none=True))

    async def _execute_payload(self, payload: dict[str, object]) -> ToolResult:
        try:
            call_typed = getattr(self._manager, "call_tool_result", None)
            if call_typed is not None:
                outcome = await call_typed(
                    self._tool_info.server_name,
                    self._tool_info.name,
                    payload,
                )
                output = outcome.output
                tool_is_error = bool(getattr(outcome, "is_error", False))
            else:
                output = await self._manager.call_tool(
                    self._tool_info.server_name,
                    self._tool_info.name,
                    payload,
                )
                tool_is_error = False
        except McpServerNotConnectedError as exc:
            return ToolResult(output=str(exc), is_error=True)
        if not isinstance(output, str) or not output.strip():
            return ToolResult(output=output, is_error=tool_is_error)
        return ToolResult(output=f"{UNTRUSTED_BANNER}\n\n{output}", is_error=tool_is_error)


class WellnessUserIdInjectingAdapter(BaseTool):
    """OHMO-scoped wrapper for the worfalomey ``get_wellness_data`` MCP tool.

    Hides ``params.user_id`` and ``params.health_types`` from the
    model-visible schema (the normal model contract stays the interval and
    the remaining optional filters) and enforces both at execution time:
    the gateway-resolved wellness identity overrides any model-supplied
    selector, and any model-supplied ``health_types`` is stripped so
    Telegent falls back to the linked device's observed types. Fails
    closed — explicit error, no MCP request — when no tenant is bound for
    the current turn, so Telegent's configured owner default is never used
    for an unmapped Telegram principal.
    """

    _HIDDEN_PARAMS = ("user_id", "health_types")

    def __init__(self, delegate: McpToolAdapter) -> None:
        self._delegate = delegate
        self.name = delegate.name
        self.description = delegate.description
        self.input_model = _input_model_from_schema(
            self.name,
            _schema_without_nested_params(delegate._tool_info.input_schema, self._HIDDEN_PARAMS),
        )
        self._tenant: str | None = None

    def set_tenant(self, tenant: str | None) -> None:
        """Bind (or clear, with ``None``) the wellness identity for this turn."""
        normalized = (tenant or "").strip()
        self._tenant = normalized or None

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        del context
        tenant = self._tenant
        if tenant is None:
            return ToolResult(
                output=(
                    "wellness data is unavailable: no wellness identity is resolved "
                    "for the current turn"
                ),
                is_error=True,
            )
        payload = arguments.model_dump(mode="json", exclude_none=True)
        params = payload.get("params")
        injected = dict(params) if isinstance(params, dict) else {}
        injected.pop("health_types", None)
        injected["user_id"] = tenant
        payload["params"] = injected
        return await self._delegate._execute_payload(payload)


_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_without_nested_params(
    schema: dict[str, object], keys: Iterable[str]
) -> dict[str, object]:
    """Return a copy of the tool schema with nested ``params.<key>`` entries removed."""
    hidden = set(keys)
    scrubbed = copy.deepcopy(schema)
    properties = scrubbed.get("properties")
    if not isinstance(properties, dict):
        return scrubbed
    params = properties.get("params")
    if not isinstance(params, dict):
        return scrubbed
    param_properties = params.get("properties")
    if isinstance(param_properties, dict):
        for key in hidden:
            param_properties.pop(key, None)
    required = params.get("required")
    if isinstance(required, list):
        params["required"] = [item for item in required if item not in hidden]
    return scrubbed


def _input_model_from_schema(tool_name: str, schema: dict[str, object]) -> type[BaseModel]:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return create_model(f"{tool_name.title()}Input")

    fields = {}
    required = set(schema.get("required", [])) if isinstance(schema.get("required", []), list) else set()
    for key in properties:
        prop = properties[key] if isinstance(properties[key], dict) else {}
        py_type = _JSON_TYPE_MAP.get(str(prop.get("type", "")), object)
        if key in required:
            fields[key] = (py_type, Field(default=...))
        else:
            fields[key] = (py_type | None, Field(default=None))
    return create_model(f"{tool_name.title().replace('-', '_')}Input", **fields)


def _sanitize_tool_segment(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    if not sanitized:
        return "tool"
    if not sanitized[0].isalpha():
        return f"mcp_{sanitized}"
    return sanitized
