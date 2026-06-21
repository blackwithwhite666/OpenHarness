from __future__ import annotations

from types import SimpleNamespace

from openharness.evals.tool_labels import (
    command_capabilities,
    effective_tool_label,
    effective_tool_path,
    extract_command_binaries,
    tool_call_binaries,
)


def test_extract_command_binaries_basic():
    assert extract_command_binaries("weather-cli forecast 'СПб' --json") == ["weather-cli"]
    assert extract_command_binaries("cat a.txt | jq .x") == ["jq"]
    assert extract_command_binaries("FOO=bar maps-cli search x") == ["maps-cli"]
    assert extract_command_binaries("cd /tmp && weather-cli now") == ["weather-cli"]
    assert extract_command_binaries("/usr/local/bin/maps-cli reviews 1") == ["maps-cli"]
    assert extract_command_binaries("sudo -v") == []


def test_extract_command_binaries_ignores_heredoc_body():
    cmd = "python3.12 - <<'PY'\nimport os\nprint(os.getcwd())\nPY"
    assert extract_command_binaries(cmd) == ["python3.12"]


def test_command_capabilities_adds_subcommand_for_cli_tools():
    assert command_capabilities("maps-cli reviews 9089 --json") == ["maps-cli reviews"]
    assert command_capabilities("maps-cli search 'x'") == ["maps-cli search"]
    assert command_capabilities("weather-cli forecast 'СПб' --days 1") == ["weather-cli forecast"]
    assert command_capabilities("git --no-pager status") == ["git status"]
    assert command_capabilities("git commit -m x") == ["git commit"]
    # python is not a -cli/known-subcommand tool -> no subcommand appended
    assert command_capabilities("python3.12 - <<'PY'\nprint(1)\nPY") == ["python3.12"]
    assert command_capabilities("python3.12 script.py") == ["python3.12"]


def test_command_capabilities_skip_flags_and_plumbing():
    assert command_capabilities("sudo -v") == []
    assert effective_tool_label("bash", {"command": "sudo -v"}) == "bash"

    for command in (
        "set -e",
        "mkdir -p x",
        "apt-get update",
        "chmod +x f",
        "cp a b",
        "ls -la /tmp",
    ):
        assert command_capabilities(command) == []
        assert effective_tool_label("bash", {"command": command}) == "bash"

    assert command_capabilities("mkdir -p d && maps-cli search x") == ["maps-cli search"]


def test_effective_tool_label_shell_vs_typed():
    assert effective_tool_label("bash", {"command": "maps-cli reviews x"}) == "bash:maps-cli reviews"
    assert effective_tool_label("bash", {"command": "python3.12 -"}) == "bash:python3.12"
    assert effective_tool_label("bash", {"command": ""}) == "bash"
    assert effective_tool_label("bash", None) == "bash"
    # Typed tools are unchanged -> name-based analysis still works.
    assert effective_tool_label("web_fetch", {"url": "https://x"}) == "web_fetch"
    assert effective_tool_label("remind_create", {"due": "x"}) == "remind_create"


def test_tool_call_binaries_only_for_shell():
    assert tool_call_binaries("bash", {"command": "maps-cli reviews x"}) == ["maps-cli"]
    assert tool_call_binaries("web_fetch", {"url": "https://x"}) == []


def _ev(kind, tool_name=None, call_id=None, command=None):
    payload = {"input": {"command": command}} if command is not None else {}
    return SimpleNamespace(kind=kind, tool_name=tool_name, tool_call_id=call_id, payload=payload)


def test_effective_tool_path_lifts_capability_and_dedupes_calls():
    events = [
        _ev("inbound_message"),
        _ev("tool_started", "bash", "c1", "weather-cli forecast 'СПб'"),
        _ev("tool_completed", "bash", "c1", None),
        _ev("tool_started", "bash", "c2", "maps-cli reviews 9089"),
        _ev("tool_completed", "bash", "c2", None),
        _ev("gateway_final"),
    ]
    assert effective_tool_path(events) == ["bash:weather-cli forecast", "bash:maps-cli reviews"]
