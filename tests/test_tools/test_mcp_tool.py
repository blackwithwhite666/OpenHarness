"""Tests for MCP tool adapters — input model generation and argument serialization."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from openharness.mcp.client import (
    McpServerNotConnectedError,
    McpToolCallResult,
    McpToolTimeoutError,
)
from openharness.mcp.types import McpResourceInfo, McpToolInfo
from openharness.tools.base import ToolExecutionContext
from openharness.tools.list_mcp_resources_tool import ListMcpResourcesTool
from openharness.tools.mcp_tool import (
    McpToolAdapter,
    WellnessLoginInjectingAdapter,
    _input_model_from_schema,
)
from openharness.tools.read_mcp_resource_tool import ReadMcpResourceTool
from openharness.untrusted import UNTRUSTED_BANNER


class _FakeMcpManager:
    def __init__(
        self,
        *,
        tool_output: str = "",
        resource_output: str = "",
        resources: list[McpResourceInfo] | None = None,
    ) -> None:
        self.tool_output = tool_output
        self.resource_output = resource_output
        self.resources = resources or []

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        del server_name, tool_name, arguments
        return self.tool_output

    async def read_resource(self, server_name: str, uri: str) -> str:
        del server_name, uri
        return self.resource_output

    def list_resources(self) -> list[McpResourceInfo]:
        return self.resources


class _TypedFakeMcpManager:
    """Manager double exposing the typed call_tool_result path."""

    def __init__(
        self,
        *,
        outcome: McpToolCallResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.outcome = outcome
        self.error = error

    async def call_tool_result(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> McpToolCallResult:
        del server_name, tool_name, arguments
        if self.error is not None:
            raise self.error
        assert self.outcome is not None
        return self.outcome


def _demo_adapter(manager) -> McpToolAdapter:
    return McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="demo",
            name="hello",
            description="test",
            input_schema={"type": "object", "properties": {}},
        ),
    )


@pytest.mark.asyncio
async def test_mcp_tool_adapter_typed_success_is_fenced_and_not_error():
    manager = _TypedFakeMcpManager(outcome=McpToolCallResult(output="server supplied output"))
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is False
    assert result.output == f"{UNTRUSTED_BANNER}\n\nserver supplied output"


@pytest.mark.asyncio
async def test_mcp_tool_adapter_tool_declared_error_preserves_body():
    manager = _TypedFakeMcpManager(
        outcome=McpToolCallResult(
            output="interval must not exceed 31 days",
            is_error=True,
        )
    )
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is True
    assert "interval must not exceed 31 days" in result.output


@pytest.mark.asyncio
async def test_mcp_tool_adapter_tool_declared_error_with_empty_body_stays_error():
    manager = _TypedFakeMcpManager(outcome=McpToolCallResult(output="", is_error=True))
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is True
    assert result.output == ""


@pytest.mark.asyncio
async def test_mcp_tool_adapter_timeout_remains_error_on_typed_path():
    manager = _TypedFakeMcpManager(
        error=McpToolTimeoutError("MCP server 'demo' tool 'hello' timed out after 1s")
    )
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is True
    assert "timed out" in result.output


@pytest.mark.asyncio
async def test_mcp_tool_adapter_disconnection_remains_error_on_typed_path():
    manager = _TypedFakeMcpManager(
        error=McpServerNotConnectedError("MCP server 'demo' is not connected: boom")
    )
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is True
    assert "not connected" in result.output


@pytest.mark.asyncio
async def test_mcp_tool_adapter_fences_nonempty_success_output():
    manager = _FakeMcpManager(tool_output="server supplied output")
    adapter = McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="demo",
            name="hello",
            description="test",
            input_schema={"type": "object", "properties": {}},
        ),
    )

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is False
    assert result.output == f"{UNTRUSTED_BANNER}\n\nserver supplied output"


@pytest.mark.asyncio
async def test_mcp_tool_adapter_does_not_fence_empty_output():
    manager = _FakeMcpManager(tool_output="")
    adapter = McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="demo",
            name="hello",
            description="test",
            input_schema={"type": "object", "properties": {}},
        ),
    )

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.output == ""
    assert UNTRUSTED_BANNER not in result.output


@pytest.mark.asyncio
async def test_read_mcp_resource_fences_success_output():
    tool = ReadMcpResourceTool(_FakeMcpManager(resource_output="resource body"))

    result = await tool.execute(
        tool.input_model(server="demo", uri="demo://readme"),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is False
    assert result.output == f"{UNTRUSTED_BANNER}\n\nresource body"


@pytest.mark.asyncio
async def test_list_mcp_resources_fences_server_supplied_descriptions():
    manager = _FakeMcpManager(
        resources=[
            McpResourceInfo(
                server_name="demo",
                name="Readme",
                uri="demo://readme",
                description="server supplied description",
            )
        ]
    )
    tool = ListMcpResourcesTool(manager)

    result = await tool.execute(tool.input_model(), ToolExecutionContext(cwd=Path(".")))

    assert result.is_error is False
    assert result.output == (
        f"{UNTRUSTED_BANNER}\n\n"
        "demo:demo://readme server supplied description"
    )


class _RecordingMcpManager:
    def __init__(self, tool_output: str = "wellness payload") -> None:
        self.tool_output = tool_output
        self.calls: list[tuple[str, str, dict]] = []

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        self.calls.append((server_name, tool_name, arguments))
        return self.tool_output


def _wellness_delegate(manager: _RecordingMcpManager) -> McpToolAdapter:
    return McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="worfalomey",
            name="get_wellness_data",
            description="wellness",
            input_schema={
                "type": "object",
                "properties": {
                    "params": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "health_types": {"type": "array"},
                            "login": {"type": "string"},
                            "interval": {"type": "string"},
                            "start": {"type": "string"},
                            "end": {"type": "string"},
                            "include_health": {"type": "boolean"},
                        },
                    }
                },
            },
        ),
    )


def _wellness_ref_delegate(manager: _RecordingMcpManager) -> McpToolAdapter:
    return McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="worfalomey",
            name="get_wellness_data",
            description="wellness",
            input_schema={
                "type": "object",
                "properties": {
                    "params": {"$ref": "#/$defs/WellnessParams"},
                },
                "required": ["params"],
                "$defs": {
                    "WellnessParams": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "health_types": {
                                "anyOf": [
                                    {"type": "array", "items": {"type": "string"}},
                                    {"type": "null"},
                                ],
                                "default": None,
                            },
                            "participant_id": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                                "default": None,
                            },
                            "login": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "interval": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "start": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "end": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                        },
                    }
                },
            },
        ),
    )


class TestWellnessLoginInjectingAdapter:
    """OHMO-scoped wellness login selector and trusted identity binding."""

    def test_legacy_identity_fields_are_hidden_from_model_schema(self):
        delegate = _wellness_delegate(_RecordingMcpManager())
        adapter = WellnessLoginInjectingAdapter(delegate)

        schema = adapter.input_model.model_json_schema()
        serialized = json.dumps(schema)
        assert "login" in serialized
        for field in ("user_id", "health_types", "participant_id"):
            assert field not in serialized
            assert field not in json.dumps(adapter.to_api_schema())

    async def test_current_schema_omitted_login_has_no_stale_extra(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal("116870365", owner_turn=True)

        result = await adapter.execute(
            adapter.input_model(params={"start": "2026-08-09", "end": "2026-08-09"}),
            ToolExecutionContext(cwd=Path(".")),
        )

        assert result.is_error is False
        assert manager.calls[0][2]["params"] == {
            "start": "2026-08-09",
            "end": "2026-08-09",
        }

    async def test_owner_normalizes_selected_login(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal("116870365", owner_turn=True)

        await adapter.execute(
            adapter.input_model(params={"login": "  @Marina_Lipina  ", "interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )

        assert manager.calls[-1][2]["params"]["login"] == "marina_lipina"

    async def test_owner_rejects_invalid_selected_login_without_mcp_call(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal(
            "116870365", trusted_login="dmitry_owner", owner_turn=True
        )

        result = await adapter.execute(
            adapter.input_model(params={"login": "Marina-Lipina", "interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )

        assert result.is_error is True
        assert "params.login" in result.output
        assert manager.calls == []

    async def test_owner_omitted_login_uses_trusted_contact_username(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal(
            "116870365", trusted_login="Dmitry_Example", owner_turn=True
        )

        await adapter.execute(
            adapter.input_model(params={"interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )

        assert manager.calls[-1][2]["params"]["login"] == "dmitry_example"

    async def test_owner_omitted_login_without_contact_username_is_safe(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal("116870365", owner_turn=True)

        await adapter.execute(
            adapter.input_model(params={"interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )

        assert manager.calls[-1][2]["params"] == {"interval": "7d"}

    async def test_family_overrides_model_login_with_trusted_contact(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal(
            "200", trusted_login="Marina_Lipina", family_turn=True
        )

        await adapter.execute(
            adapter.input_model(params={"login": "mallory", "interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )

        assert manager.calls[-1][2]["params"]["login"] == "marina_lipina"

    async def test_family_without_trusted_username_fails_closed(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal("200", family_turn=True)

        result = await adapter.execute(
            adapter.input_model(params={"login": "mallory", "interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )

        assert result.is_error is True
        assert "no usable username" in result.output
        assert manager.calls == []

    async def test_family_with_invalid_trusted_username_fails_closed(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal(
            "200", trusted_login="Marina-Lipina", family_turn=True
        )

        result = await adapter.execute(
            adapter.input_model(params={"interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )

        assert result.is_error is True
        assert "no usable username" in result.output
        assert manager.calls == []

    async def test_strips_all_legacy_identity_fields(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_ref_delegate(manager))
        adapter.set_trusted_principal("100", owner_turn=True)

        class LegacyArguments(BaseModel):
            params: dict[str, object]

        arguments = LegacyArguments(
            params={
                "interval": "7d",
                "participant_id": 116870365,
                "user_id": "mallory",
                "health_types": ["weight"],
            }
        )
        result = await adapter.execute(arguments, ToolExecutionContext(cwd=Path(".")))

        assert result.is_error is False
        assert manager.calls[-1][2]["params"] == {"interval": "7d"}

    async def test_missing_identity_clears_stale_binding(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal("100", owner_turn=True)
        await adapter.execute(
            adapter.input_model(params={"interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )
        adapter.set_trusted_principal(None)

        result = await adapter.execute(
            adapter.input_model(params={"interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )
        assert result.is_error is True
        assert len(manager.calls) == 1

    async def test_non_telegram_principal_fails_closed(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal("100", channel="feishu", owner_turn=True)

        result = await adapter.execute(
            adapter.input_model(params={"interval": "7d"}),
            ToolExecutionContext(cwd=Path(".")),
        )

        assert result.is_error is True
        assert manager.calls == []

    def test_resolves_fastmcp_params_ref_without_mutating_delegate_schema(self):
        delegate = _wellness_ref_delegate(_RecordingMcpManager())
        original_schema = copy.deepcopy(delegate._tool_info.input_schema)
        adapter = WellnessLoginInjectingAdapter(delegate)

        serialized = json.dumps(adapter.input_model.model_json_schema())
        assert "user_id" not in serialized
        assert "health_types" not in serialized
        assert "participant_id" not in serialized
        assert delegate._tool_info.input_schema == original_schema

    @pytest.mark.parametrize(
        "ref,definitions",
        [
            ("#/components/schemas/WellnessParams", {"WellnessParams": {"type": "object"}}),
            ("#/$defs/MissingParams", {}),
        ],
    )
    def test_rejects_unsupported_or_broken_params_ref(self, ref, definitions):
        manager = _RecordingMcpManager()
        delegate = _wellness_ref_delegate(manager)
        delegate._tool_info.input_schema["properties"]["params"] = {"$ref": ref}
        delegate._tool_info.input_schema["$defs"] = definitions

        with pytest.raises(ValueError, match="JSON Schema reference"):
            WellnessLoginInjectingAdapter(delegate)

    async def test_ref_schema_owner_selection_and_family_override(self):
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_ref_delegate(manager))
        adapter.set_trusted_principal("100", owner_turn=True)

        await adapter.execute(
            adapter.input_model(params={"interval": "7d", "login": "@Marina_Lipina"}),
            ToolExecutionContext(cwd=Path(".")),
        )
        assert manager.calls[-1][2]["params"] == {
            "interval": "7d",
            "login": "marina_lipina",
        }


class TestInputModelFromSchema:
    """Verify _input_model_from_schema maps JSON Schema types correctly."""

    def test_required_string_rejects_none(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        Model = _input_model_from_schema("search", schema)
        with pytest.raises(ValidationError):
            Model(query=None)

    def test_required_string_accepts_value(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        Model = _input_model_from_schema("search", schema)
        m = Model(query="zigzag")
        assert m.query == "zigzag"

    def test_optional_string_defaults_to_none(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "wing": {"type": "string"},
            },
            "required": ["query"],
        }
        Model = _input_model_from_schema("search", schema)
        m = Model(query="test")
        assert m.wing is None

    def test_exclude_none_omits_optional_keeps_required(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "wing": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }
        Model = _input_model_from_schema("search", schema)
        m = Model(query="test")
        dumped = m.model_dump(mode="json", exclude_none=True)
        assert dumped == {"query": "test"}

    def test_all_json_types_mapped(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
                "tags": {"type": "array"},
                "meta": {"type": "object"},
            },
            "required": ["name", "count", "score", "active", "tags", "meta"],
        }
        Model = _input_model_from_schema("full", schema)
        m = Model(name="x", count=1, score=0.5, active=True, tags=["a"], meta={"k": "v"})
        dumped = m.model_dump(mode="json")
        assert dumped == {
            "name": "x", "count": 1, "score": 0.5,
            "active": True, "tags": ["a"], "meta": {"k": "v"},
        }

    def test_empty_schema_creates_valid_model(self):
        Model = _input_model_from_schema("empty", {"type": "object"})
        m = Model()
        assert m.model_dump(mode="json") == {}

    def test_model_rejects_null_for_required_integer(self):
        schema = {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": ["limit"],
        }
        Model = _input_model_from_schema("limited", schema)
        with pytest.raises(ValidationError):
            Model(limit=None)
