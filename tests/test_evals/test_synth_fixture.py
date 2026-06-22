from __future__ import annotations

import json
from pathlib import Path

import pytest

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.evals import (
    EvalToolFixture,
    ReplayToolInput,
    SynthesizedFixtureTool,
)
from openharness.tools.base import ToolExecutionContext


class _StaticCodegenApiClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=self._text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _RaisingCodegenApiClient:
    def __init__(self) -> None:
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        raise RuntimeError("codegen failed")
        yield


@pytest.mark.asyncio
async def test_synth_fixture_tool_executes_generated_respond(tmp_path: Path):
    api_client = _StaticCodegenApiClient(
        """
def respond(arguments, captured):
    return "synth:" + arguments["query"]
""".strip()
    )
    tool = _tool(api_client=api_client)

    result = await tool.execute(
        ReplayToolInput.model_validate({"query": "alpha"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "synth:alpha"
    assert result.is_error is False
    assert result.metadata == {"synth": True, "lane": "synth", "tool": "bash"}
    assert len(api_client.requests) == 1
    assert api_client.requests[0].tools == []


@pytest.mark.asyncio
async def test_synth_fixture_tool_can_filter_grounded_captured_output(
    tmp_path: Path,
):
    api_client = _StaticCodegenApiClient(
        """
def respond(arguments, captured):
    needle = arguments.get("needle", "")
    lines = captured[0]["output"].splitlines()
    return "\\n".join(line for line in lines if needle in line)
""".strip()
    )
    tool = _tool(
        api_client=api_client,
        fixtures=(
            _fixture(
                output="red apple\nblue berry\nred cherry",
                input_text='{"query": "colors"}',
            ),
        ),
    )

    result = await tool.execute(
        ReplayToolInput.model_validate({"needle": "red"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "red apple\nred cherry"
    assert result.metadata["lane"] == "synth"


@pytest.mark.asyncio
async def test_synth_fixture_tool_caches_codegen_once(tmp_path: Path):
    api_client = _StaticCodegenApiClient(
        """
def respond(arguments, captured):
    return captured[0]["output"] + ":" + arguments["suffix"]
""".strip()
    )
    tool = _tool(api_client=api_client)

    first = await tool.execute(
        ReplayToolInput.model_validate({"suffix": "one"}),
        ToolExecutionContext(cwd=tmp_path),
    )
    second = await tool.execute(
        ReplayToolInput.model_validate({"suffix": "two"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert first.output == "frozen output:one"
    assert second.output == "frozen output:two"
    assert len(api_client.requests) == 1


@pytest.mark.asyncio
async def test_synth_fixture_tool_falls_back_when_codegen_has_no_respond(
    tmp_path: Path,
):
    tool = _tool(api_client=_StaticCodegenApiClient("print('no function')"))

    result = await tool.execute(
        ReplayToolInput.model_validate({"query": "alpha"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "frozen output"
    assert result.metadata["lane"] == "synth-fallback"
    assert result.metadata["synth"] is False


@pytest.mark.asyncio
async def test_synth_fixture_tool_falls_back_when_codegen_raises(tmp_path: Path):
    tool = _tool(api_client=_RaisingCodegenApiClient())

    result = await tool.execute(
        ReplayToolInput.model_validate({"query": "alpha"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "frozen output"
    assert result.metadata["lane"] == "synth-fallback"


@pytest.mark.asyncio
async def test_synth_fixture_tool_falls_back_when_generated_code_raises(
    tmp_path: Path,
):
    api_client = _StaticCodegenApiClient(
        """
def respond(arguments, captured):
    raise RuntimeError("boom")
""".strip()
    )
    tool = _tool(api_client=api_client)

    result = await tool.execute(
        ReplayToolInput.model_validate({"query": "alpha"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "frozen output"
    assert result.metadata["lane"] == "synth-fallback"


@pytest.mark.asyncio
async def test_synth_fixture_tool_falls_back_on_timeout(tmp_path: Path):
    api_client = _StaticCodegenApiClient(
        """
def respond(arguments, captured):
    import time
    time.sleep(1)
    return "too late"
""".strip()
    )
    tool = _tool(api_client=api_client, exec_timeout=0.1)

    result = await tool.execute(
        ReplayToolInput.model_validate({"query": "alpha"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "frozen output"
    assert result.metadata["lane"] == "synth-fallback"


@pytest.mark.asyncio
async def test_synth_fixture_subprocess_uses_isolated_exec(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}

    class _FakeProcess:
        returncode = 0

        async def communicate(self, payload: bytes):
            captured["payload"] = json.loads(payload.decode("utf-8"))
            return b'{"ok": true, "output": "safe"}', b""

        def kill(self) -> None:
            captured["killed"] = True

        async def wait(self) -> None:
            captured["waited"] = True

    async def fake_create_subprocess_exec(*argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(
        "openharness.evals.synth_fixture.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    tool = _tool(
        api_client=_StaticCodegenApiClient(
            """
def respond(arguments, captured):
    return "safe"
""".strip()
        )
    )

    result = await tool.execute(
        ReplayToolInput.model_validate({"query": "alpha"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "safe"
    argv = captured["argv"]
    assert "-I" in argv
    assert "-S" in argv
    kwargs = captured["kwargs"]
    assert kwargs.get("shell") is not True
    assert "shell" not in kwargs
    env = kwargs["env"]
    assert set(env) <= {"PATH"}
    assert all(
        marker not in key.upper()
        for key in env
        for marker in ("KEY", "TOKEN", "SECRET")
    )
    assert captured["payload"]["arguments"] == {"query": "alpha"}
    assert captured["payload"]["captured"][0]["output"] == "frozen output"


def _tool(
    *,
    api_client,
    fixtures: tuple[EvalToolFixture, ...] | None = None,
    exec_timeout: float = 20.0,
) -> SynthesizedFixtureTool:
    return SynthesizedFixtureTool(
        tool_name="bash",
        fixtures=fixtures or (_fixture(),),
        api_client=api_client,
        model="codegen-model",
        exec_timeout=exec_timeout,
    )


def _fixture(
    *,
    output: str = "frozen output",
    input_text: str = '{"command": "captured"}',
) -> EvalToolFixture:
    return EvalToolFixture(
        tool_name="bash",
        call_key_hash="hash-1",
        input_text=input_text,
        output_text=output,
        input_key="input-key",
    )
