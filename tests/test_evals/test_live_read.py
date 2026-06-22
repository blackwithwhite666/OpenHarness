from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    LiveReadAgentRunner,
    LiveReadBashTool,
    ReplayToolInput,
    build_replay_tool_registry,
    classify_bash_command,
)
from openharness.evals.workspace_materialize import LiveLocalReadTool
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult


@pytest.mark.parametrize(
    ("command", "expected_lane", "expected_tokens"),
    [
        ("maps-cli search 'СПб'", "live", ["maps-cli", "search", "СПб"]),
        ("afisha-cli events --city spb", "live", ["afisha-cli", "events", "--city", "spb"]),
        ("weather-cli forecast x", "live", ["weather-cli", "forecast", "x"]),
        ("rm -rf /", "mock", []),
        ("maps-cli search x && rm y", "mock", []),
        ("maps-cli search x | grep y", "mock", []),
        ("echo $(whoami)", "mock", []),
        ("python3 x.py", "mock", []),
        ("", "mock", []),
    ],
)
def test_classify_bash_command(command, expected_lane, expected_tokens):
    lane, tokens = classify_bash_command(command)

    assert lane == expected_lane
    assert tokens == expected_tokens


@pytest.mark.asyncio
async def test_live_read_bash_tool_executes_allowlisted_command(tmp_path: Path, monkeypatch):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def fake_exec(*args, **kwargs):
        calls.append((args, kwargs))
        return _FakeProcess(stdout=b"live stdout\n")

    monkeypatch.setattr(
        "openharness.evals.live_read.asyncio.create_subprocess_exec",
        fake_exec,
    )
    mock_tool = _MockBashTool()
    tool = LiveReadBashTool(mock_tool=mock_tool, cwd=tmp_path)

    result = await tool.execute(
        ReplayToolInput.model_validate({"command": "maps-cli search x"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "live stdout\n"
    assert result.is_error is False
    assert result.metadata["lane"] == "live"
    assert result.metadata["binary"] == "maps-cli"
    assert result.metadata["returncode"] == 0
    assert mock_tool.calls == []
    assert calls[0][0][:3] == ("maps-cli", "search", "x")
    assert calls[0][1]["cwd"] == str(tmp_path.resolve())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    [
        "python3 x.py",
        "maps-cli search x && rm y",
        "maps-cli search x; rm y",
        "maps-cli search x | grep y",
        "maps-cli search $(whoami)",
    ],
)
async def test_live_read_bash_tool_falls_back_to_mock_for_unsafe_commands(
    tmp_path: Path,
    monkeypatch,
    command: str,
):
    async def fail_exec(*args, **kwargs):
        raise AssertionError(f"subprocess should not run: {args} {kwargs}")

    monkeypatch.setattr(
        "openharness.evals.live_read.asyncio.create_subprocess_exec",
        fail_exec,
    )
    mock_tool = _MockBashTool(output="mocked replay")
    tool = LiveReadBashTool(mock_tool=mock_tool, cwd=tmp_path)

    result = await tool.execute(
        ReplayToolInput.model_validate({"command": command}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "mocked replay"
    assert result.metadata["lane"] == "mock"
    assert mock_tool.calls == [command]


@pytest.mark.asyncio
async def test_live_read_bash_tool_timeout_kills_process(tmp_path: Path, monkeypatch):
    fake_process = _FakeProcess(delay=60.0)

    async def fake_exec(*args, **kwargs):
        return fake_process

    monkeypatch.setattr(
        "openharness.evals.live_read.asyncio.create_subprocess_exec",
        fake_exec,
    )
    tool = LiveReadBashTool(
        mock_tool=_MockBashTool(),
        cwd=tmp_path,
        timeout=0.001,
    )

    result = await tool.execute(
        ReplayToolInput.model_validate({"command": "weather-cli forecast x"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "<timeout>"
    assert result.is_error is True
    assert result.metadata["lane"] == "live"
    assert result.metadata["binary"] == "weather-cli"
    assert fake_process.killed is True


def test_live_read_agent_runner_uses_live_bash_and_cleans_temp_cwd(
    tmp_path: Path,
    monkeypatch,
):
    live_cwds: list[Path] = []

    async def fake_exec(*args, **kwargs):
        live_cwds.append(Path(kwargs["cwd"]))
        return _FakeProcess(stdout=b"LIVE_RESULT")

    monkeypatch.setattr(
        "openharness.evals.live_read.asyncio.create_subprocess_exec",
        fake_exec,
    )
    api_client = _LiveReadModelApiClient()
    runner = LiveReadAgentRunner(
        api_client=api_client,
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
    )
    registry = build_replay_tool_registry(
        (
            EvalToolFixture(
                tool_name="bash",
                call_key_hash="fixture-bash",
                output_text="MOCK_RESULT",
            ),
        )
    )

    result = runner.run(
        prompt="private live read request",
        tool_registry=registry,
        context=SimpleNamespace(events=()),
    )

    assert result.final_text == "final saw LIVE_RESULT"
    assert result.tool_path == ("bash",)
    assert result.metadata["agent_runner"] == "query-engine-live-read"
    assert live_cwds
    assert live_cwds[0].name.startswith("openharness-eval-live-read-")
    assert live_cwds[0].exists() is False


def test_live_read_agent_runner_overrides_google_search_mcp_and_closes(
    tmp_path: Path,
    monkeypatch,
):
    managers: list[_FakeMcpClientManager] = []
    google_cfg = object()

    def fake_create_default_tool_registry(manager):
        assert manager.connected is True
        registry = ToolRegistry()
        registry.register(_FakeGoogleSearchTool(output="LIVE_GOOGLE_RESULT"))
        return registry

    monkeypatch.setattr("openharness.evals.live_read.load_settings", lambda: object())
    monkeypatch.setattr(
        "openharness.evals.live_read.load_mcp_server_configs",
        lambda settings, plugins: {"google_search": google_cfg},
    )
    monkeypatch.setattr(
        "openharness.evals.live_read.McpClientManager",
        lambda cfg: _FakeMcpClientManager(cfg, managers),
    )
    monkeypatch.setattr(
        "openharness.evals.live_read.create_default_tool_registry",
        fake_create_default_tool_registry,
    )

    runner = LiveReadAgentRunner(
        api_client=_GoogleSearchModelApiClient(
            expected_tool_result="LIVE_GOOGLE_RESULT",
            forbidden_tool_result="REPLAY_GOOGLE_RESULT",
            final_text="final saw LIVE_GOOGLE_RESULT",
        ),
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
        live_mcp_server_names=("google_search",),
    )
    registry = _google_search_replay_registry()

    result = runner.run(
        prompt="search live",
        tool_registry=registry,
        context=SimpleNamespace(events=()),
    )

    assert result.final_text == "final saw LIVE_GOOGLE_RESULT"
    assert result.tool_path == ("mcp__google_search__google_search",)
    assert result.metadata["agent_runner"] == "query-engine-live-read"
    assert result.metadata["live_mcp_servers"] == ["google_search"]
    assert managers
    assert managers[0].server_configs == {"google_search": google_cfg}
    assert managers[0].closed is True


def test_live_read_agent_runner_wraps_typed_read_tools_and_cleans_temp_cwd(
    tmp_path: Path,
):
    api_client = _LiveReadFileModelApiClient()
    runner = LiveReadAgentRunner(
        api_client=api_client,
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
        live_typed_read_tool_names=("read_file",),
    )
    read_fixture = EvalToolFixture(
        tool_name="read_file",
        call_key_hash="fixture-read",
        input_text='{"path": "/home/u/a.md", "offset": 0, "limit": 20}',
        output_text="CAPTURED_CONTENT",
    )
    write_fixture = EvalToolFixture(
        tool_name="write_file",
        call_key_hash="fixture-write",
        input_text='{"path": "/home/u/a.md", "content": "MUTATION"}',
        output_text="WRITE_REPLAY",
    )
    registry = build_replay_tool_registry((read_fixture, write_fixture))

    result = runner.run(
        prompt="private live file read request",
        tool_registry=registry,
        context=SimpleNamespace(events=(), tool_fixtures=(read_fixture, write_fixture)),
    )

    read_tool = registry.get("read_file")
    write_tool = registry.get("write_file")
    assert isinstance(read_tool, LiveLocalReadTool)
    assert not isinstance(write_tool, LiveLocalReadTool)
    assert read_tool._sandbox.exists() is False
    assert result.final_text == "final saw live local read"
    assert result.tool_path == ("read_file",)
    assert result.metadata["agent_runner"] == "query-engine-live-read"
    assert result.metadata["live_local_read_tools"] == ["read_file"]
    assert result.metadata["materialized_file_count"] == 1


def test_live_read_agent_runner_keeps_replay_google_search_when_mcp_connect_fails(
    tmp_path: Path,
    monkeypatch,
):
    managers: list[_FakeMcpClientManager] = []
    google_cfg = object()

    monkeypatch.setattr("openharness.evals.live_read.load_settings", lambda: object())
    monkeypatch.setattr(
        "openharness.evals.live_read.load_mcp_server_configs",
        lambda settings, plugins: {"google_search": google_cfg},
    )
    monkeypatch.setattr(
        "openharness.evals.live_read.McpClientManager",
        lambda cfg: _FakeMcpClientManager(
            cfg,
            managers,
            connect_error=RuntimeError("boom"),
        ),
    )
    monkeypatch.setattr(
        "openharness.evals.live_read.create_default_tool_registry",
        lambda manager: pytest.fail(
            "MCP registry should not be created after connect failure"
        ),
    )

    runner = LiveReadAgentRunner(
        api_client=_GoogleSearchModelApiClient(
            expected_tool_result="REPLAY_GOOGLE_RESULT",
            forbidden_tool_result="LIVE_GOOGLE_RESULT",
            final_text="final saw REPLAY_GOOGLE_RESULT",
        ),
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
        live_mcp_server_names=("google_search",),
    )

    result = runner.run(
        prompt="search replay",
        tool_registry=_google_search_replay_registry(),
        context=SimpleNamespace(events=()),
    )

    assert result.final_text == "final saw REPLAY_GOOGLE_RESULT"
    assert result.tool_path == ("mcp__google_search__google_search",)
    assert result.metadata["live_mcp_servers"] == ["google_search"]
    assert managers
    assert managers[0].closed is True


class _MockBashTool(BaseTool):
    name = "bash"
    description = "mock bash"
    input_model = ReplayToolInput

    def __init__(self, output: str = "mocked") -> None:
        self.output = output
        self.calls: list[str] = []

    async def execute(
        self,
        arguments: ReplayToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        payload = arguments.model_dump()
        command = payload.get("command") or payload.get("cmd") or ""
        self.calls.append(command if isinstance(command, str) else "")
        return ToolResult(output=self.output, metadata={"replayed": True})

    def is_read_only(self, arguments: ReplayToolInput) -> bool:
        del arguments
        return True


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        delay: float = 0.0,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._delay = delay
        self.killed = False

    async def communicate(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self):
        return self.returncode


class _LiveReadModelApiClient:
    def __init__(self) -> None:
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="toolu-live-bash",
                            name="bash",
                            input={"command": "maps-cli search x"},
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return

        tool_results = [
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert any("LIVE_RESULT" in block.content for block in tool_results)
        assert not any("MOCK_RESULT" in block.content for block in tool_results)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text="final saw LIVE_RESULT")],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _LiveReadFileModelApiClient:
    def __init__(self) -> None:
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="toolu-live-file",
                            name="read_file",
                            input={"path": "/home/u/a.md", "offset": 0, "limit": 20},
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return

        tool_results = [
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert any("\tCAPTURED_CONTENT" in block.content for block in tool_results)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text="final saw live local read")],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def _google_search_replay_registry() -> ToolRegistry:
    return build_replay_tool_registry(
        (
            EvalToolFixture(
                tool_name="mcp__google_search__google_search",
                call_key_hash="fixture-google-search",
                output_text="REPLAY_GOOGLE_RESULT",
            ),
        )
    )


class _FakeGoogleSearchTool(BaseTool):
    name = "mcp__google_search__google_search"
    description = "fake live google search"
    input_model = ReplayToolInput

    def __init__(self, output: str) -> None:
        self.output = output

    async def execute(
        self,
        arguments: ReplayToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        assert arguments.model_dump()["query"] == "openharness live"
        return ToolResult(output=self.output, metadata={"lane": "live"})

    def is_read_only(self, arguments: ReplayToolInput) -> bool:
        del arguments
        return True


class _FakeMcpClientManager:
    def __init__(
        self,
        server_configs: dict[str, object],
        managers: list["_FakeMcpClientManager"],
        *,
        connect_error: Exception | None = None,
    ) -> None:
        self.server_configs = server_configs
        self.connected = False
        self.closed = False
        self._connect_error = connect_error
        managers.append(self)

    async def connect_all(self) -> None:
        if self._connect_error is not None:
            raise self._connect_error
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    def list_tools(self):
        return [
            SimpleNamespace(
                server_name="google_search",
                name="google_search",
                description="fake google search",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]


class _GoogleSearchModelApiClient:
    def __init__(
        self,
        *,
        expected_tool_result: str,
        forbidden_tool_result: str,
        final_text: str,
    ) -> None:
        self.requests = []
        self._expected_tool_result = expected_tool_result
        self._forbidden_tool_result = forbidden_tool_result
        self._final_text = final_text

    async def stream_message(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="toolu-google-search",
                            name="mcp__google_search__google_search",
                            input={"query": "openharness live"},
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return

        tool_results = [
            block
            for message in request.messages
            for block in message.content
            if isinstance(block, ToolResultBlock)
        ]
        assert any(self._expected_tool_result in block.content for block in tool_results)
        assert not any(
            self._forbidden_tool_result in block.content for block in tool_results
        )
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=self._final_text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )
