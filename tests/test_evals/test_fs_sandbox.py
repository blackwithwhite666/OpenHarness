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
from openharness.evals import EvalToolFixture, ReplayToolInput, build_replay_tool_registry
from openharness.evals.fs_sandbox import (
    FsSandboxAgentRunner,
    FsSandboxBashTool,
    SandboxFsTool,
    assemble_fs,
    build_bwrap_argv,
)
from openharness.tools import create_default_tool_registry
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


def test_build_bwrap_argv_net_modes_and_binds(tmp_path: Path):
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    ro = tmp_path / "ro"
    rw = tmp_path / "rw"
    dest = tmp_path / "dest"
    for path in (home, cwd, ro, rw):
        path.mkdir()

    argv = build_bwrap_argv(
        sandbox_root=tmp_path / "sandbox",
        cwd=cwd,
        home=home,
        ro_binds=(ro,),
        rw_binds=((rw, dest),),
        net_mode="none",
    )

    assert argv[0] == "bwrap"
    assert "--unshare-net" in argv
    assert "--unshare-user" not in argv
    assert "sudo" not in argv
    assert "--clearenv" not in argv
    assert _contains_subsequence(argv, ["--ro-bind", str(ro), str(ro)])
    assert _contains_subsequence(argv, ["--bind", str(rw), str(dest)])
    assert _contains_subsequence(argv, ["--setenv", "HOME", str(home)])
    assert _contains_subsequence(argv, ["--setenv", "PATH", "/usr/bin:/bin"])
    assert _contains_subsequence(argv, ["--chdir", str(cwd)])

    netns_argv = build_bwrap_argv(
        sandbox_root=tmp_path / "sandbox",
        cwd=cwd,
        home=home,
        ro_binds=(ro,),
        rw_binds=((rw, dest),),
        net_mode="netns:evalns",
        uid=123,
        gid=456,
        proxy_url="http://10.77.0.1:3128",
    )

    assert netns_argv[:5] == ["sudo", "ip", "netns", "exec", "evalns"]
    assert "bwrap" in netns_argv
    assert _contains_subsequence(
        netns_argv,
        ["--unshare-user", "--uid", "123", "--gid", "456"],
    )
    assert "--unshare-net" not in netns_argv
    assert _contains_subsequence(
        netns_argv,
        ["--setenv", "HTTPS_PROXY", "http://10.77.0.1:3128"],
    )
    assert _contains_subsequence(
        netns_argv,
        ["--setenv", "HTTP_PROXY", "http://10.77.0.1:3128"],
    )


def test_build_bwrap_argv_browser_socket_bind_and_env(tmp_path: Path):
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    tmp_bind = tmp_path / "tmp"
    for path in (home, cwd, tmp_bind):
        path.mkdir()

    argv = build_bwrap_argv(
        sandbox_root=tmp_path / "sandbox",
        cwd=cwd,
        home=home,
        ro_binds=(),
        rw_binds=((tmp_bind, Path("/tmp")),),
        net_mode="none",
        browser_socket="/tmp/browser-cli-ohmo.sock",
        browser_cli_name="ohmo",
    )

    assert _contains_subsequence(
        argv,
        ["--bind", "/tmp/browser-cli-ohmo.sock", "/tmp/browser-cli-ohmo.sock"],
    )
    assert _contains_subsequence(argv, ["--setenv", "BROWSER_CLI_NAME", "ohmo"])
    tmp_bind_index = _subsequence_index(argv, ["--bind", str(tmp_bind), "/tmp"])
    socket_bind_index = _subsequence_index(
        argv,
        ["--bind", "/tmp/browser-cli-ohmo.sock", "/tmp/browser-cli-ohmo.sock"],
    )
    assert tmp_bind_index < socket_bind_index

    omitted = build_bwrap_argv(
        sandbox_root=tmp_path / "sandbox",
        cwd=cwd,
        home=home,
        ro_binds=(),
        rw_binds=((tmp_bind, Path("/tmp")),),
        net_mode="none",
    )

    assert not _contains_subsequence(
        omitted,
        ["--bind", "/tmp/browser-cli-ohmo.sock", "/tmp/browser-cli-ohmo.sock"],
    )
    assert not _contains_subsequence(
        omitted,
        ["--setenv", "BROWSER_CLI_NAME", "ohmo"],
    )


def test_assemble_fs_copies_mutable_dirs_and_builds_remap(tmp_path: Path):
    root = tmp_path / "sandbox"
    home = tmp_path / "home"
    cwd = tmp_path / "case"
    memory = home / ".ohmo" / "memory"
    skills = home / ".ohmo" / "skills"
    user_file = home / ".ohmo" / "user.md"
    memory.mkdir(parents=True)
    skills.mkdir(parents=True)
    cwd.mkdir()
    (memory / "fact.md").write_text("remembered\n", encoding="utf-8")
    user_file.write_text("profile\n", encoding="utf-8")

    plan = assemble_fs(
        root,
        home=home,
        mutable_dirs=("memory", "todos", "user.md"),
        ro_source_dirs=(skills, home / ".ohmo" / "missing"),
        cwd=cwd,
    )

    assert plan.work == root.resolve() / "work"
    assert plan.tmp == root.resolve() / "tmp"
    assert (plan.state_root / "memory" / "fact.md").read_text(encoding="utf-8") == (
        "remembered\n"
    )
    assert (plan.state_root / "user.md").read_text(encoding="utf-8") == "profile\n"
    assert (plan.state_root / "todos").is_dir()
    assert plan.remap[(home / ".ohmo" / "memory").resolve()] == (
        plan.state_root / "memory"
    )
    assert plan.remap[user_file.resolve()] == plan.state_root / "user.md"
    assert plan.remap[Path("/tmp")] == plan.tmp
    assert plan.remap[cwd.resolve()] == plan.work
    assert (plan.state_root / "memory", (home / ".ohmo" / "memory").resolve()) in (
        plan.rw_binds
    )
    assert (plan.state_root / "user.md", user_file.resolve()) in plan.rw_binds
    assert (plan.tmp, Path("/tmp")) in plan.rw_binds
    assert plan.ro_binds == (skills.resolve(),)
    assert plan.mutable_copies == ("memory", "todos", "user.md")


def test_assemble_fs_persists_cwd_state_with_persist_cwd(tmp_path: Path):
    root = tmp_path / "sandbox"
    home = tmp_path / "home"
    cwd = tmp_path / "case"
    cwd.mkdir(parents=True)
    home.mkdir()

    first_plan = assemble_fs(
        root,
        home=home,
        mutable_dirs=(),
        ro_source_dirs=(),
        cwd=cwd,
        persist_cwd=True,
    )
    assert first_plan.work == cwd.resolve()
    first_plan.work.mkdir(parents=True, exist_ok=True)
    (first_plan.work / "state.txt").write_text("persisted", encoding="utf-8")

    second_plan = assemble_fs(
        tmp_path / "sandbox-next",
        home=home,
        mutable_dirs=(),
        ro_source_dirs=(),
        cwd=cwd,
        persist_cwd=True,
    )
    assert second_plan.work == cwd.resolve()
    assert (second_plan.work / "state.txt").read_text(encoding="utf-8") == "persisted"

    disposable_plan = assemble_fs(
        tmp_path / "sandbox-disposable",
        home=home,
        mutable_dirs=(),
        ro_source_dirs=(),
        cwd=cwd,
        persist_cwd=False,
    )
    assert disposable_plan.work != cwd
    assert not (disposable_plan.work / "state.txt").exists()


@pytest.mark.asyncio
async def test_sandbox_fs_tool_round_trips_writes_and_confines_live_paths(
    tmp_path: Path,
):
    real_cwd = tmp_path / "real"
    sandbox_work = tmp_path / "sandbox" / "work"
    real_cwd.mkdir()
    sandbox_work.mkdir(parents=True)
    live_file = tmp_path / "live.txt"
    live_file.write_text("LIVE\n", encoding="utf-8")
    registry = create_default_tool_registry()
    real_write = registry.get("write_file")
    real_read = registry.get("read_file")
    assert real_write is not None
    assert real_read is not None
    remap = {real_cwd.resolve(): sandbox_work.resolve()}
    context = ToolExecutionContext(cwd=real_cwd)
    write_tool = SandboxFsTool(
        real_tool=real_write,
        mock_tool=_RecordingMockTool("write_file", real_write.input_model),
        remap=remap,
        cwd=sandbox_work,
    )
    read_tool = SandboxFsTool(
        real_tool=real_read,
        mock_tool=_RecordingMockTool("read_file", real_read.input_model),
        remap=remap,
        cwd=sandbox_work,
    )

    write_result = await write_tool.execute(
        real_write.input_model(path=str(real_cwd / "report.txt"), content="COPY\n"),
        context,
    )
    read_result = await read_tool.execute(
        real_read.input_model(path=str(real_cwd / "report.txt"), limit=20),
        context,
    )
    relative_result = await write_tool.execute(
        real_write.input_model(path="relative.txt", content="REL\n"),
        context,
    )
    live_result = await read_tool.execute(
        real_read.input_model(path=str(live_file), limit=20),
        context,
    )
    refused_result = await write_tool.execute(
        real_write.input_model(path=str(live_file), content="CHANGED\n"),
        context,
    )

    assert write_result.is_error is False
    assert "COPY" in read_result.output
    assert (sandbox_work / "report.txt").read_text(encoding="utf-8") == "COPY\n"
    assert not (real_cwd / "report.txt").exists()
    assert "Wrote" in relative_result.output
    assert (sandbox_work / "relative.txt").read_text(encoding="utf-8") == "REL\n"
    assert "LIVE" in live_result.output
    assert refused_result.is_error is True
    assert "Write outside sandbox refused" in refused_result.output
    assert live_file.read_text(encoding="utf-8") == "LIVE\n"


@pytest.mark.asyncio
async def test_fs_sandbox_bash_tool_runs_bwrap_argv_and_falls_back(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[tuple[object, ...]] = []

    async def fake_exec(*args, **kwargs):
        del kwargs
        calls.append(args)
        return _FakeProcess(stdout=b"sandbox stdout\n")

    monkeypatch.setattr(
        "openharness.evals.fs_sandbox.asyncio.create_subprocess_exec",
        fake_exec,
    )
    mock_tool = _MockBashTool(output="mocked replay")
    tool = FsSandboxBashTool(
        mock_tool=mock_tool,
        sandbox_root=tmp_path / "sandbox",
        cwd=tmp_path,
        home=tmp_path / "home",
        ro_binds=(),
        rw_binds=(),
    )

    result = await tool.execute(
        ReplayToolInput.model_validate({"command": "echo hi"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.output == "sandbox stdout\n"
    assert result.is_error is False
    assert result.metadata["lane"] == "fs-sandbox"
    assert calls
    assert calls[0][0] == "bwrap"
    assert "--unshare-net" in calls[0]
    assert calls[0][-3:] == ("bash", "-c", "echo hi")
    assert mock_tool.calls == []

    async def missing_exec(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("bwrap")

    monkeypatch.setattr(
        "openharness.evals.fs_sandbox.asyncio.create_subprocess_exec",
        missing_exec,
    )

    fallback = await tool.execute(
        ReplayToolInput.model_validate({"command": "echo hi"}),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert fallback.output == "mocked replay"
    assert fallback.metadata["lane"] == "mock"
    assert mock_tool.calls == ["echo hi"]


def test_fs_sandbox_agent_runner_overrides_tools_and_cleans_up(tmp_path: Path):
    api_client = _WriteReadApiClient()
    runner = FsSandboxAgentRunner(
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
                output_text="REPLAY_BASH",
            ),
        )
    )

    result = runner.run(
        prompt="write and read",
        tool_registry=registry,
        context=SimpleNamespace(events=()),
    )

    assert result.final_text == "final saw sandbox file"
    assert result.tool_path == ("write_file", "read_file")
    assert result.metadata["agent_runner"] == "fs-sandbox"
    assert result.metadata["sandbox_net_mode"] == "none"
    assert result.metadata["mutable_copies"] == ["memory", "todos", "reminders", "user.md"]
    assert isinstance(registry.get("bash"), FsSandboxBashTool)
    for name in ("read_file", "write_file", "edit_file", "glob", "grep"):
        assert isinstance(registry.get(name), SandboxFsTool)

    bash_tool = registry.get("bash")
    assert isinstance(bash_tool, FsSandboxBashTool)
    assert bash_tool._browser_socket is None
    assert bash_tool._browser_cli_name is None
    home_bin = (Path.home() / "bin").resolve()
    if home_bin.exists():
        assert home_bin in bash_tool._ro_binds
    assert bash_tool._sandbox_root.exists() is False
    assert not (tmp_path / "note.txt").exists()


def test_fs_sandbox_agent_runner_threads_netns_proxy_browser_to_bash_tool(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr("openharness.evals.fs_sandbox.load_settings", lambda: object())
    monkeypatch.setattr(
        "openharness.evals.fs_sandbox.load_mcp_server_configs",
        lambda settings, cli_configs: {},
    )
    api_client = _WriteReadApiClient()
    runner = FsSandboxAgentRunner(
        api_client=api_client,
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
        net_mode="netns:evalns",
        proxy_url="http://10.77.0.1:3128",
        browser_socket="/tmp/browser-cli-ohmo.sock",
        browser_cli_name="ohmo",
        live_mcp_server_names=("google_search",),
    )
    registry = build_replay_tool_registry(())

    result = runner.run(
        prompt="write and read",
        tool_registry=registry,
        context=SimpleNamespace(events=()),
    )

    bash_tool = registry.get("bash")
    assert isinstance(bash_tool, FsSandboxBashTool)
    assert result.metadata["sandbox_net_mode"] == "netns:evalns"
    assert result.metadata["live_mcp_servers"] == ["google_search"]
    argv = build_bwrap_argv(
        sandbox_root=bash_tool._sandbox_root,
        cwd=bash_tool._cwd,
        home=bash_tool._home,
        ro_binds=bash_tool._ro_binds,
        rw_binds=bash_tool._rw_binds,
        net_mode=bash_tool._net_mode,
        proxy_url=bash_tool._proxy_url,
        browser_socket=bash_tool._browser_socket,
        browser_cli_name=bash_tool._browser_cli_name,
        uid=123,
        gid=456,
    ) + ["bash", "-lc", "echo ok"]
    assert argv[:5] == ["sudo", "ip", "netns", "exec", "evalns"]
    assert _contains_subsequence(
        argv,
        ["--unshare-user", "--uid", "123", "--gid", "456"],
    )
    assert "--unshare-net" not in argv
    assert _contains_subsequence(
        argv,
        ["--setenv", "HTTPS_PROXY", "http://10.77.0.1:3128"],
    )
    assert _contains_subsequence(
        argv,
        ["--bind", "/tmp/browser-cli-ohmo.sock", "/tmp/browser-cli-ohmo.sock"],
    )
    assert _contains_subsequence(argv, ["--setenv", "BROWSER_CLI_NAME", "ohmo"])
    assert argv[-3:] == ["bash", "-lc", "echo ok"]


def _contains_subsequence(items: list[str], expected: list[str]) -> bool:
    width = len(expected)
    return any(items[index : index + width] == expected for index in range(len(items)))


def _subsequence_index(items: list[str], expected: list[str]) -> int:
    width = len(expected)
    for index in range(len(items)):
        if items[index : index + width] == expected:
            return index
    raise AssertionError(f"{expected!r} not found in {items!r}")


class _RecordingMockTool(BaseTool):
    description = "recording mock tool"
    input_model = ReplayToolInput

    def __init__(self, name: str, input_model: type[ReplayToolInput]) -> None:
        self.name = name
        self.input_model = input_model
        self.calls: list[dict[str, object]] = []

    async def execute(
        self,
        arguments: ReplayToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        self.calls.append(arguments.model_dump())
        return ToolResult(output="MOCKED", metadata={"replayed": True})


class _MockBashTool(BaseTool):
    name = "bash"
    description = "mock bash"
    input_model = ReplayToolInput

    def __init__(self, output: str) -> None:
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


class _WriteReadApiClient:
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
                            id="toolu-write",
                            name="write_file",
                            input={"path": "note.txt", "content": "sandbox file\n"},
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return
        if len(self.requests) == 2:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="toolu-read",
                            name="read_file",
                            input={"path": "note.txt", "limit": 20},
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
        assert any("sandbox file" in block.content for block in tool_results)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text="final saw sandbox file")],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def test_build_bwrap_argv_bin_dirs_on_path_and_ro_bound(tmp_path):
    from openharness.evals.fs_sandbox import build_bwrap_argv

    bindir = tmp_path / "skillbin"
    bindir.mkdir()
    argv = build_bwrap_argv(
        sandbox_root=tmp_path, cwd=tmp_path, home=tmp_path,
        ro_binds=[], rw_binds=[], net_mode="host", bin_dirs=[bindir],
    )
    resolved = str(bindir.resolve())
    assert resolved in argv  # ro-bound
    assert argv[argv.index("PATH") + 1] == f"{resolved}:/usr/bin:/bin"  # prepended
    assert "--unshare-net" not in argv  # host net


def test_build_sandbox_skill_bin_mocks_publisher(tmp_path):
    from ohmo.evals.runner import _build_sandbox_skill_bin
    from ohmo.workspace import get_skills_dir

    assert _build_sandbox_skill_bin(None, live_skill=False) == ()
    ws = tmp_path / "ws"
    skills = get_skills_dir(ws)
    (skills / "static_publisher").mkdir(parents=True)
    (skills / "maps").mkdir(parents=True)
    (skills / "static_publisher" / "static_publisher-cli").write_text("real")
    (skills / "maps" / "maps-cli").write_text("real")
    dirs = _build_sandbox_skill_bin(ws, live_skill=True)
    mock_pub = dirs[0] / "static_publisher-cli"
    assert mock_pub.exists() and (mock_pub.stat().st_mode & 0o111)
    assert "mock" in mock_pub.read_text()
    assert (dirs[1] / "maps-cli").exists()  # flat symlink to the real nested CLI


def test_build_bwrap_argv_bin_dirs_bound_after_tmp(tmp_path):
    # Regression: a skill-bin dir lives under /tmp (mkdtemp), and /tmp is itself an
    # rw-bind to the disposable sandbox tmp. The bin dir must be ro-bound AFTER the
    # /tmp bind, else /tmp shadows it and every mocked/flat skill CLI becomes
    # "command not found" in the jail.
    from openharness.evals.fs_sandbox import build_bwrap_argv

    tmp_bind = tmp_path / "sandbox-tmp"
    tmp_bind.mkdir()
    bindir = tmp_path / "skillbin"
    bindir.mkdir()
    argv = build_bwrap_argv(
        sandbox_root=tmp_path,
        cwd=tmp_path,
        home=tmp_path,
        ro_binds=[],
        rw_binds=((tmp_bind, Path("/tmp")),),
        net_mode="host",
        bin_dirs=[bindir],
    )
    resolved = str(bindir.resolve())
    tmp_idx = _subsequence_index(argv, ["--bind", str(tmp_bind), "/tmp"])
    bin_idx = _subsequence_index(argv, ["--ro-bind", resolved, resolved])
    assert bin_idx > tmp_idx  # bin dir bound after /tmp -> not shadowed


def test_build_sandbox_skill_bin_mocks_dropbox(tmp_path):
    from ohmo.evals.runner import _build_sandbox_skill_bin
    from ohmo.workspace import get_skills_dir

    ws = tmp_path / "ws"
    get_skills_dir(ws).mkdir(parents=True)
    dirs = _build_sandbox_skill_bin(ws, live_skill=True)
    mock_dropbox = dirs[0] / "dropbox"  # shadows the real ~/bin/dropbox on PATH
    assert mock_dropbox.exists() and (mock_dropbox.stat().st_mode & 0o111)
    body = mock_dropbox.read_text()
    assert "sharelink" in body and "dropbox.com/s/" in body


def test_fs_sandbox_default_ro_source_dirs_include_attachments():
    runner = FsSandboxAgentRunner(api_client=_WriteReadApiClient(), model="m")
    tails = [str(p) for p in runner._ro_source_dirs]
    assert any(t.endswith(".ohmo/attachments") for t in tails)
    assert any(t.endswith(".ohmo/skills") for t in tails)


def test_fs_sandbox_extra_ro_source_dirs_appended_without_dropping_defaults():
    runner = FsSandboxAgentRunner(
        api_client=_WriteReadApiClient(),
        model="m",
        extra_ro_source_dirs=("/data/tickets", "~/fixtures/pdfs"),
    )
    tails = [str(p) for p in runner._ro_source_dirs]
    # extras are present...
    assert any(t.endswith("/data/tickets") for t in tails)
    assert any(t.endswith("/fixtures/pdfs") for t in tails)  # ~ expanded
    assert not any(t.startswith("~") for t in tails)
    # ...and the built-in defaults are still there.
    assert any(t.endswith(".ohmo/attachments") for t in tails)
