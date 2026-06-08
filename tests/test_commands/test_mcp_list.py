from __future__ import annotations

from openharness import cli
from openharness.mcp.types import McpHttpServerConfig, McpStdioServerConfig


def test_mcp_list_handles_pydantic_model_configs(monkeypatch, capsys):
    cfgs = {
        "a": McpStdioServerConfig(command="python3", args=["x.py"]),
        "b": McpHttpServerConfig(url="https://x/mcp"),
    }
    monkeypatch.setattr("openharness.config.load_settings", lambda: object())
    monkeypatch.setattr("openharness.plugins.load_plugins", lambda *a, **k: [])
    monkeypatch.setattr("openharness.mcp.config.load_mcp_server_configs", lambda *a, **k: cfgs)

    cli.mcp_list()  # must not raise (regression: cfg.get on a pydantic model)

    out = capsys.readouterr().out
    assert "a: stdio (python3)" in out
    assert "b: http (https://x/mcp)" in out


def test_mcp_list_empty(monkeypatch, capsys):
    monkeypatch.setattr("openharness.config.load_settings", lambda: object())
    monkeypatch.setattr("openharness.plugins.load_plugins", lambda *a, **k: [])
    monkeypatch.setattr("openharness.mcp.config.load_mcp_server_configs", lambda *a, **k: {})
    cli.mcp_list()
    assert "No MCP servers configured." in capsys.readouterr().out
