from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals import (
    EvalEvent,
    EvalToolFixture,
    ReplayFixtureTool,
    ReplayToolInput,
    SynthContext,
    SynthesizedFixtureTool,
    build_replay_tool_registry,
)
from openharness.evals.executor import _validate_replay_match_mode
from openharness.evals.execution import _fixture_input_key, _tool_fixtures
from openharness.tools.base import ToolExecutionContext


def test_fixture_input_key_normalizes_order_whitespace_and_none_values():
    left = _fixture_input_key(
        {
            "b": "hello \n  world",
            "a": None,
            "nested": {"z": None, "a": "  keep\tspacing "},
        }
    )
    right = _fixture_input_key(
        {
            "nested": {"a": "keep spacing"},
            "b": "hello world",
        }
    )

    assert left == right
    assert len(left) == 16
    assert left != _fixture_input_key({"b": "different"})
    assert _fixture_input_key("not structured") == ""


def test_tool_fixtures_capture_input_key_only_for_structured_input():
    structured = _tool_fixtures(
        (
            EvalEvent(
                episode_id="ep-1",
                kind="tool_started",
                tool_name="bash",
                tool_call_id="tool-1",
                payload={"input": {"command": "echo   hi"}},
            ),
        )
    )
    unstructured = _tool_fixtures(
        (
            EvalEvent(
                episode_id="ep-1",
                kind="tool_started",
                tool_name="bash",
                tool_call_id="tool-1",
                payload={"input": "echo hi"},
            ),
        )
    )

    assert structured[0].input_key == _fixture_input_key({"command": "echo hi"})
    assert unstructured[0].input_key == ""


@pytest.mark.asyncio
async def test_replay_fixture_tool_arguments_mode_matches_by_input_key_not_order(
    tmp_path: Path,
):
    first = _fixture("bash", "first", {"command": "first command"})
    second = _fixture("bash", "second", {"command": "second command"})
    tool = ReplayFixtureTool(
        tool_name="bash",
        fixtures=(first, second),
        match_mode="arguments",
    )

    result = await tool.execute(
        ReplayToolInput.model_validate({"command": " second \n command "}),
        ToolExecutionContext(cwd=tmp_path),
    )
    miss = await tool.execute(
        ReplayToolInput.model_validate({"command": "missing command"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "second"
    assert result.metadata == {
        "replayed": True,
        "match": "arguments",
        "call_key_hash": "hash-second",
    }
    assert miss.is_error is False
    assert miss.output == ""
    assert miss.metadata["replayed"] is False
    assert miss.metadata["replay_miss"] is True
    assert miss.metadata["requested_key"] == _fixture_input_key(
        {"command": "missing command"}
    )


@pytest.mark.asyncio
async def test_replay_fixture_tool_arguments_mode_consumes_duplicate_matches_in_order(
    tmp_path: Path,
):
    args = {"command": "same command"}
    tool = ReplayFixtureTool(
        tool_name="bash",
        fixtures=(
            _fixture("bash", "first duplicate", args),
            _fixture("bash", "second duplicate", args),
        ),
        match_mode="arguments",
    )

    first = await tool.execute(
        ReplayToolInput.model_validate(args),
        ToolExecutionContext(cwd=tmp_path),
    )
    second = await tool.execute(
        ReplayToolInput.model_validate(args),
        ToolExecutionContext(cwd=tmp_path),
    )
    miss = await tool.execute(
        ReplayToolInput.model_validate(args),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert first.output == "first duplicate"
    assert second.output == "second duplicate"
    assert miss.is_error is False
    assert miss.output == ""
    assert miss.metadata["replay_miss"] is True


@pytest.mark.asyncio
async def test_replay_fixture_tool_order_mode_remains_default(tmp_path: Path):
    tool = ReplayFixtureTool(
        tool_name="bash",
        fixtures=(
            _fixture("bash", "first", {"command": "first command"}),
            _fixture("bash", "second", {"command": "second command"}),
        ),
    )

    result = await tool.execute(
        ReplayToolInput.model_validate({"command": "second command"}),
        ToolExecutionContext(cwd=tmp_path),
    )
    second = await tool.execute(
        ReplayToolInput.model_validate({"command": "first command"}),
        ToolExecutionContext(cwd=tmp_path),
    )
    overflow = await tool.execute(
        ReplayToolInput.model_validate({"command": "extra command"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "first"
    assert result.metadata == {"replayed": True, "call_key_hash": "hash-first"}
    assert second.output == "second"
    assert overflow.is_error is False
    assert overflow.output == ""
    assert overflow.metadata == {"replayed": False, "replay_overflow": True}


@pytest.mark.asyncio
async def test_build_replay_tool_registry_wires_arguments_mode(tmp_path: Path):
    registry = build_replay_tool_registry(
        (
            _fixture("bash", "first", {"command": "first command"}),
            _fixture("bash", "second", {"command": "second command"}),
        ),
        match_mode="arguments",
    )
    tool = registry.get("bash")
    assert tool is not None

    result = await tool.execute(
        ReplayToolInput.model_validate({"command": "second command"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "second"
    assert result.metadata["match"] == "arguments"


def test_build_replay_tool_registry_synth_requires_context():
    with pytest.raises(ValueError, match="requires a SynthContext"):
        build_replay_tool_registry(
            (_fixture("bash", "first", {"command": "first command"}),),
            match_mode="synth",
        )


def test_build_replay_tool_registry_wires_synth_mode():
    registry = build_replay_tool_registry(
        (_fixture("bash", "first", {"command": "first command"}),),
        match_mode="synth",
        synth_context=SynthContext(api_client=object(), model="codegen-model"),
    )

    assert isinstance(registry.get("bash"), SynthesizedFixtureTool)


def test_validate_replay_match_mode_accepts_synth():
    _validate_replay_match_mode("synth")


def _fixture(
    tool_name: str,
    output: str,
    args: dict[str, object],
) -> EvalToolFixture:
    return EvalToolFixture(
        tool_name=tool_name,
        call_key_hash=f"hash-{output.split()[0]}",
        output_text=output,
        input_key=_fixture_input_key(args),
    )
