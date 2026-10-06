"""MCP tool adapters."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable

from pydantic import BaseModel, Field, create_model

from openharness.mcp.client import McpClientManager, McpServerNotConnectedError
from openharness.mcp.types import McpToolInfo
from openharness.mcp.wellness_delegation import (
    META_KEY,
    TrustedWellnessActor,
    WellnessDelegationConfig,
    sign_wellness_call,
)
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
        return await self._execute_payload(
            arguments.model_dump(mode="json", exclude_none=True), context=context
        )

    async def _execute_payload(
        self, payload: dict[str, object], *, context: ToolExecutionContext | None = None,
        meta: dict[str, str] | None = None,
    ) -> ToolResult:
        # Freeze the exact final arguments and metadata before entering the async manager.
        frozen_payload = copy.deepcopy(payload)
        frozen_meta = dict(meta) if meta is not None else None
        try:
            call_typed = getattr(self._manager, "call_tool_result", None)
            if call_typed is not None:
                outcome = await call_typed(
                    self._tool_info.server_name,
                    self._tool_info.name,
                    frozen_payload,
                    **({"meta": frozen_meta} if frozen_meta is not None else {}),
                )
                output = outcome.output
                tool_is_error = bool(getattr(outcome, "is_error", False))
            else:
                if frozen_meta is not None:
                    return ToolResult(output="wellness data is unavailable: signed metadata transport is unavailable", is_error=True)
                output = await self._manager.call_tool(
                    self._tool_info.server_name,
                    self._tool_info.name,
                    frozen_payload,
                )
                tool_is_error = False
        except McpServerNotConnectedError as exc:
            error_text = str(exc)
            signed_token = frozen_meta.get(META_KEY) if frozen_meta is not None else None
            if isinstance(signed_token, str):
                error_text = error_text.replace(signed_token, "[REDACTED-JWT]")
            return ToolResult(output=error_text, is_error=True)
        except TypeError:
            if frozen_meta is not None:
                return ToolResult(
                    output="wellness data is unavailable: signed metadata transport is unavailable",
                    is_error=True,
                )
            raise
        signed_token = frozen_meta.get(META_KEY) if frozen_meta is not None else None
        if isinstance(output, str) and isinstance(signed_token, str):
            output = output.replace(signed_token, "[REDACTED-JWT]")
        if not isinstance(output, str) or not output.strip():
            return ToolResult(output=output, is_error=tool_is_error)
        return ToolResult(output=f"{UNTRUSTED_BANNER}\n\n{output}", is_error=tool_is_error)


class WellnessLoginInjectingAdapter(BaseTool):
    """Sign final wellness arguments with the admitted Telegram reader identity.

    The server owns self versus delegated-reader authorization. Mutable adapter
    setters are retained only for compatibility and never establish authority.
    """

    _HIDDEN_PARAMS = ("user_id", "health_types", "participant_id")

    def __init__(self, delegate: McpToolAdapter) -> None:
        self._delegate = delegate
        self.name = delegate.name
        self.description = delegate.description
        schema = _schema_without_nested_params(
            delegate._tool_info.input_schema, self._HIDDEN_PARAMS
        )
        _make_login_optional(schema)
        self.input_model = _input_model_from_schema(self.name, schema)
        try:
            self._delegation_config: WellnessDelegationConfig | None = WellnessDelegationConfig.from_env()
        except ValueError:
            self._delegation_config = None

    def set_trusted_principal(
        self,
        principal: str | int | None,
        *,
        trusted_login: str | None = None,
        channel: str = "telegram",
        owner_turn: bool = False,
        family_turn: bool = False,
    ) -> None:
        """Bind the immutable principal and turn role for the next call."""
        # Kept for source compatibility. Mutable adapter state is never authority.
        del principal, trusted_login, channel, owner_turn, family_turn

    def set_tenant(self, tenant: str | None) -> None:
        """Reject the removed tenant-shaped binding and clear prior identity."""
        del tenant
        self.set_trusted_principal(None, channel="")

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        metadata = getattr(context, "metadata", None)
        actor = metadata.get("wellness_trusted_actor") if isinstance(metadata, dict) else None
        if type(actor) is not TrustedWellnessActor:
            return ToolResult(
                output="wellness data is unavailable: no admitted reader identity is present",
                is_error=True,
            )
        if self._delegation_config is None:
            return ToolResult(output="wellness data is unavailable: delegation is not configured", is_error=True)
        # Preserve omission versus explicit null for the signed final wire body.
        payload = arguments.model_dump(mode="json", exclude_unset=True)
        params = payload.get("params")
        injected = dict(params) if isinstance(params, dict) else {}
        injected.pop("user_id", None)
        injected.pop("health_types", None)
        injected.pop("participant_id", None)
        payload["params"] = injected
        try:
            meta = sign_wellness_call(self._delegation_config, actor, payload)
        except (TypeError, ValueError):
            return ToolResult(output="wellness data is unavailable: delegation request is invalid", is_error=True)
        return await self._delegate._execute_payload(payload, context=context, meta=meta)


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


def _make_login_optional(schema: dict[str, object]) -> None:
    """Make the owner-selectable login omission-safe."""
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    params = properties.get("params")
    if not isinstance(params, dict):
        return
    required = params.get("required")
    if isinstance(required, list):
        params["required"] = [item for item in required if item != "login"]


def _normalize_login(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    normalized = normalized.removeprefix("@")
    normalized = normalized.lower()
    return normalized if re.fullmatch(r"[a-z0-9_]{1,64}", normalized) else None


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
