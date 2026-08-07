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

    Hides the legacy ``params.user_id`` and ``params.health_types`` fields
    from the model-visible schema. The trusted gateway binds a Telegram
    principal for every turn. An owner may additionally select a numeric
    ``params.participant_id``; a family turn is always pinned to its own
    principal. Telegent's response is passed through without translating
    participant ids into names or tenants.
    """

    _HIDDEN_PARAMS = ("user_id", "health_types")

    def __init__(self, delegate: McpToolAdapter) -> None:
        self._delegate = delegate
        self.name = delegate.name
        self.description = delegate.description
        schema = _schema_without_nested_params(
            delegate._tool_info.input_schema, self._HIDDEN_PARAMS
        )
        _make_participant_optional(schema)
        self.input_model = _input_model_from_schema(self.name, schema)
        self._trusted_principal: str | None = None
        self._trusted_channel: str | None = None
        self._owner_turn = False
        self._family_turn = False

    def set_trusted_principal(
        self,
        principal: str | int | None,
        *,
        channel: str = "telegram",
        owner_turn: bool = False,
        family_turn: bool = False,
    ) -> None:
        """Bind the immutable principal and turn role for the next call."""
        value = str(principal).strip() if principal is not None else ""
        self._trusted_principal = value or None
        self._trusted_channel = str(channel).strip().lower() or None
        self._owner_turn = owner_turn is True
        self._family_turn = family_turn is True

    def set_tenant(self, tenant: str | None) -> None:
        """Reject the removed tenant-shaped binding and clear prior identity."""
        del tenant
        self.set_trusted_principal(None, channel="")

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        del context
        if self._trusted_principal is None:
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
        injected.pop("user_id", None)
        if self._trusted_principal is not None:
            if self._trusted_channel != "telegram" or not self._trusted_principal.isdigit():
                return ToolResult(
                    output="wellness data is unavailable: trusted Telegram principal is invalid",
                    is_error=True,
                )
            if self._family_turn:
                selected = self._trusted_principal
            elif self._owner_turn:
                selected = injected.get("participant_id")
                if selected is None:
                    selected = int(self._trusted_principal)
                if (
                    isinstance(selected, bool)
                    or not isinstance(selected, int)
                    or selected <= 0
                ):
                    return ToolResult(
                        output=(
                            "wellness data is unavailable: participant_id must be "
                            "a positive integer"
                        ),
                        is_error=True,
                    )
            else:
                return ToolResult(
                    output="wellness data is unavailable: no authorized wellness role",
                    is_error=True,
                )
            injected["participant_id"] = int(selected)
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
    scrubbed = _resolve_schema_refs(copy.deepcopy(schema))
    scrubbed.pop("$defs", None)
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


def _resolve_schema_refs(schema: dict[str, object]) -> dict[str, object]:
    """Inline visible local ``$defs`` references, rejecting unsupported refs."""

    def resolve(value: object, stack: tuple[str, ...] = ()) -> object:
        if isinstance(value, list):
            return [resolve(item, stack) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            ref = value["$ref"]
            if not isinstance(ref, str):
                raise ValueError("unsupported JSON Schema reference: $ref must be a string")
            parts = ref.split("/")
            if len(parts) != 3 or parts[:2] != ["#", "$defs"] or not parts[2]:
                raise ValueError(f"unsupported JSON Schema reference: {ref!r}")
            if ref in stack:
                raise ValueError(f"cyclic JSON Schema reference: {ref!r}")

            definitions = schema.get("$defs")
            if not isinstance(definitions, dict):
                raise ValueError(f"broken JSON Schema reference: {ref!r}")
            definition_name = parts[2].replace("~1", "/").replace("~0", "~")
            target = definitions.get(definition_name)
            if not isinstance(target, dict):
                raise ValueError(f"broken JSON Schema reference: {ref!r}")

            resolved = copy.deepcopy(target)
            resolved.update({key: item for key, item in value.items() if key != "$ref"})
            return resolve(resolved, (*stack, ref))
        return {
            key: value if key == "$defs" else resolve(value, stack)
            for key, value in value.items()
        }

    resolved = resolve(schema)
    if not isinstance(resolved, dict):
        raise TypeError("invalid JSON Schema root")
    return resolved


def _make_participant_optional(schema: dict[str, object]) -> None:
    """Make the owner-selectable participant selector omission-safe."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    params = properties.get("params")
    if not isinstance(params, dict):
        return
    required = params.get("required")
    if isinstance(required, list):
        params["required"] = [item for item in required if item != "participant_id"]


def _input_model_from_schema(tool_name: str, schema: dict[str, object]) -> type[BaseModel]:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return create_model(f"{tool_name.title()}Input")

    fields = {}
    required = (
        set(schema.get("required", []))
        if isinstance(schema.get("required", []), list)
        else set()
    )
    for key in properties:
        prop = properties[key] if isinstance(properties[key], dict) else {}
        py_type = _python_type_from_schema(f"{tool_name}_{key}", prop)
        if key in required:
            fields[key] = (py_type, Field(default=...))
        else:
            fields[key] = (py_type | None, Field(default=None))
    return create_model(f"{tool_name.title().replace('-', '_')}Input", **fields)


def _python_type_from_schema(name: str, schema: dict[str, object]) -> type:
    """Build the small nested Pydantic shape used by MCP tool arguments."""
    for keyword in ("anyOf", "oneOf"):
        alternatives = schema.get(keyword)
        if isinstance(alternatives, list):
            non_null = [
                item
                for item in alternatives
                if isinstance(item, dict) and item.get("type") != "null"
            ]
            if len(non_null) == 1:
                return _python_type_from_schema(name, non_null[0])

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null_types = [item for item in schema_type if item != "null"]
        if len(non_null_types) == 1:
            schema_type = non_null_types[0]
    if schema_type != "object" or not isinstance(schema.get("properties"), dict):
        return _JSON_TYPE_MAP.get(str(schema_type or ""), object)
    properties = schema["properties"]
    required = (
        set(schema.get("required", []))
        if isinstance(schema.get("required", []), list)
        else set()
    )
    fields = {}
    for key, value in properties.items():
        prop = value if isinstance(value, dict) else {}
        py_type = _python_type_from_schema(f"{name}_{key}", prop)
        fields[key] = (py_type, Field(default=... if key in required else None))
        if key not in required:
            fields[key] = (py_type | None, Field(default=None))
    return create_model(f"{name.title().replace('-', '_')}Input", **fields)


def _sanitize_tool_segment(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", value)
    if not sanitized:
        return "tool"
    if not sanitized[0].isalpha():
        return f"mcp_{sanitized}"
    return sanitized
