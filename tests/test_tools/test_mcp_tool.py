"""Tests for MCP tool adapters — input model generation and argument serialization."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from openharness.mcp.types import McpResourceInfo, McpToolInfo
from openharness.tools.base import ToolExecutionContext
from openharness.tools.list_mcp_resources_tool import ListMcpResourcesTool
from openharness.tools.mcp_tool import McpToolAdapter, _input_model_from_schema
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
