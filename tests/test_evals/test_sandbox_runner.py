from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel, Field

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.evals import (
    EvalToolFixture,
    SandboxMutatingAgentRunner,
    build_replay_tool_registry,
)
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult


def test_sandbox_mutating_agent_runner_runs_real_factory_tool_and_cleans_up(
    tmp_path: Path,
):
    sandbox_paths: list[Path] = []
    api_client = _ToolThenFinalApiClient(
        tool_name="write_note",
        tool_input={"name": "created"},
    )
    runner = SandboxMutatingAgentRunner(
        api_client=api_client,
        model="eval-model",
        sandbox_tool_factory=lambda sandbox: _factory(sandbox, sandbox_paths),
        sandbox_state_fn=_file_state,
        cwd=tmp_path,
    )

    result = runner.run(
        prompt="write a note",
        tool_registry=ToolRegistry(),
        context=SimpleNamespace(events=()),
    )

    assert result.final_text == "sandbox final"
    assert result.tool_path == ("write_note",)
    assert result.metadata["agent_runner"] == "sandbox"
    assert result.metadata["sandbox_state_delta"]["changed"] is True
    assert result.metadata["sandbox_state_delta"]["memory"]["added_keys"] == [
        "created.txt"
    ]
    assert sandbox_paths
    assert not sandbox_paths[0].exists()


def test_sandbox_mutating_agent_runner_keeps_bash_on_replay_fixture(
    tmp_path: Path,
):
    side_effect = tmp_path / "should-not-exist"
    api_client = _ToolThenFinalApiClient(
        tool_name="bash",
        tool_input={"command": f"touch {side_effect}"},
    )
    runner = SandboxMutatingAgentRunner(
        api_client=api_client,
        model="eval-model",
        sandbox_tool_factory=lambda sandbox: (),
        sandbox_state_fn=_file_state,
        cwd=tmp_path,
    )
    registry = build_replay_tool_registry(
        (
            EvalToolFixture(
                tool_name="bash",
                call_key_hash="fixture-bash",
                output_text="replayed bash output",
            ),
        )
    )

    result = runner.run(
        prompt="run bash",
        tool_registry=registry,
        context=SimpleNamespace(events=()),
    )

    assert result.tool_path == ("bash",)
    assert result.tool_calls[0].tool_name == "bash"
    assert result.tool_calls[0].is_error is False
    assert result.metadata["sandbox_state_delta"]["changed"] is False
    assert not side_effect.exists()


class _WriteNoteInput(BaseModel):
    name: str = Field(default="note")


class _WriteNoteTool(BaseTool):
    name = "write_note"
    description = "Write a note file in the current workspace."
    input_model = _WriteNoteInput

    async def execute(
        self,
        arguments: _WriteNoteInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        path = context.cwd / f"{arguments.name}.txt"
        path.write_text("created by sandbox", encoding="utf-8")
        return ToolResult(output=f"wrote {path.name}")


class _ToolThenFinalApiClient:
    def __init__(self, *, tool_name: str, tool_input: dict[str, object]) -> None:
        self._tool_name = tool_name
        self._tool_input = tool_input
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        last_message = request.messages[-1]
        if any(isinstance(block, ToolResultBlock) for block in last_message.content):
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="sandbox final")],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return

        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id=f"toolu-{self._tool_name}",
                        name=self._tool_name,
                        input=self._tool_input,
                    )
                ],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def _factory(sandbox: Path, sandbox_paths: list[Path]):
    sandbox_paths.append(sandbox)
    return (_WriteNoteTool(),)


def _file_state(sandbox: Path) -> dict[str, object]:
    keys = sorted(path.name for path in sandbox.glob("*.txt"))
    return {
        "reminders": {
            "entry_keys": [],
            "count": 0,
            "status_counts": {},
        },
        "memory": {
            "entry_keys": keys,
            "count": len(keys),
        },
        "todos": {
            "entry_keys": [],
            "count": 0,
        },
    }
