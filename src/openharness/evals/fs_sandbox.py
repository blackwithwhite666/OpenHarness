"""Disposable real filesystem sandbox for eval agent runners."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from openharness.api.client import SupportsStreamingMessages
from openharness.config import load_settings
from openharness.evals.executor import (
    EvalExecutionContext,
    EvalExecutorResult,
    ReplayFixtureTool,
    ReplayToolInput,
    _run_eval_coroutine,
    _run_query_engine_replay,
)
from openharness.evals.live_read import _process_output, _truncate_output
from openharness.mcp.client import McpClientManager
from openharness.mcp.config import load_mcp_server_configs
from openharness.tools import create_default_tool_registry
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

_SYSTEM_RO_BINDS = (
    "/usr",
    "/bin",
    "/lib",
    "/lib64",
    "/etc/ssl",
    "/etc/resolv.conf",
)
_FS_TOOL_NAMES = ("read_file", "write_file", "edit_file", "glob", "grep")
_WRITE_TOOL_NAMES = frozenset({"write_file", "edit_file"})
_PATH_FIELDS_BY_TOOL = {
    "read_file": ("path",),
    "write_file": ("path",),
    "edit_file": ("path",),
    "glob": ("root", "pattern"),
    "grep": ("root",),
}


@dataclass(frozen=True)
class SandboxPlan:
    """Filesystem plan shared by bwrap bash and in-process typed tools."""

    root: Path
    state_root: Path
    tmp: Path
    work: Path
    remap: dict[Path, Path]
    ro_binds: tuple[Path, ...]
    rw_binds: tuple[tuple[Path, Path], ...]
    cwd: Path
    home: Path
    mutable_copies: tuple[str, ...]


def build_bwrap_argv(
    *,
    sandbox_root: str | Path,
    cwd: str | Path,
    home: str | Path,
    ro_binds: Iterable[str | Path],
    rw_binds: Iterable[tuple[str | Path, str | Path]],
    net_mode: str = "none",
    uid: int | None = None,
    gid: int | None = None,
    proxy_url: str | None = None,
    browser_socket: str | None = None,
    browser_cli_name: str | None = None,
    bin_dirs: Iterable[str | Path] = (),
) -> list[str]:
    """Build the bubblewrap argv used for real bash execution.

    ``bin_dirs`` are ro-bound AND prepended to PATH, so skill CLIs (which live
    nested under ``skills/<name>/<name>-cli`` and would otherwise be present-but-
    unreachable — bare invocations resolve as "command not found") become
    runnable in the jail.
    """
    del sandbox_root
    bwrap_argv = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        # A minimal /dev (null, zero, urandom, tty, ...); without it skill CLIs
        # (and their `2>/dev/null` / randomness) fail on a missing device node.
        "--dev",
        "/dev",
    ]
    netns_name: str | None = None
    if net_mode == "none":
        bwrap_argv.append("--new-session")
        bwrap_argv.append("--unshare-net")
    elif net_mode == "host":
        bwrap_argv.append("--new-session")
    elif net_mode.startswith("netns:") and net_mode.removeprefix("netns:"):
        netns_name = net_mode.removeprefix("netns:")
        bwrap_argv.extend(
            [
                "--unshare-user",
                "--uid",
                str(os.getuid() if uid is None else uid),
                "--gid",
                str(os.getgid() if gid is None else gid),
                "--new-session",
            ]
        )
    else:
        raise ValueError(f"unsupported sandbox net_mode: {net_mode}")

    for path in _SYSTEM_RO_BINDS:
        bind_path = Path(path)
        if bind_path.exists():
            bwrap_argv.extend(["--ro-bind", path, path])

    for raw_path in ro_binds:
        path = Path(raw_path).expanduser()
        if path.exists():
            text = str(path)
            bwrap_argv.extend(["--ro-bind", text, text])

    for raw_src, raw_dest in rw_binds:
        src = Path(raw_src).expanduser()
        dest = Path(raw_dest).expanduser()
        bwrap_argv.extend(["--bind", str(src), str(dest)])

    if browser_socket:
        socket_path = str(Path(browser_socket).expanduser())
        bwrap_argv.extend(["--bind", socket_path, socket_path])

    # bin_dirs are bound LAST, after the rw_binds above. The skill-bin dir is
    # created under the host /tmp (mkdtemp) and /tmp is itself an rw-bind to the
    # disposable sandbox tmp; binding the bin dir BEFORE that /tmp bind let the
    # /tmp bind shadow it, so every mocked / flat skill CLI silently dropped to
    # "command not found" in the jail. Binding after /tmp keeps them reachable.
    sandbox_bin_paths: list[str] = []
    for raw_bin in bin_dirs:
        bin_path = Path(raw_bin).expanduser()
        if bin_path.is_dir():
            text = str(bin_path.resolve())
            if text not in sandbox_bin_paths:
                bwrap_argv.extend(["--ro-bind", text, text])
                sandbox_bin_paths.append(text)

    bwrap_argv.extend(
        [
            "--setenv",
            "HOME",
            str(Path(home).expanduser().resolve()),
            "--setenv",
            "PATH",
            ":".join([*sandbox_bin_paths, "/usr/bin", "/bin"]),
        ]
    )
    if proxy_url:
        bwrap_argv.extend(
            [
                "--setenv",
                "HTTPS_PROXY",
                proxy_url,
                "--setenv",
                "HTTP_PROXY",
                proxy_url,
            ]
        )
    if browser_cli_name:
        bwrap_argv.extend(["--setenv", "BROWSER_CLI_NAME", browser_cli_name])
    bwrap_argv.extend(
        [
            "--chdir",
            str(Path(cwd).expanduser().resolve()),
        ]
    )

    if netns_name is not None:
        return ["sudo", "ip", "netns", "exec", netns_name, *bwrap_argv]
    return bwrap_argv


def assemble_fs(
    sandbox_root: str | Path,
    *,
    home: str | Path,
    mutable_dirs: Iterable[str | Path],
    ro_source_dirs: Iterable[str | Path],
    cwd: str | Path | None = None,
    cwd_name: str = "work",
    persist_cwd: bool = False,
) -> SandboxPlan:
    """Create disposable copies and path mappings for the eval filesystem."""
    root = Path(sandbox_root).expanduser().resolve()
    home_path = Path(home).expanduser().resolve()
    real_cwd = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    work = real_cwd if persist_cwd else root / cwd_name
    state_root = (
        real_cwd / ".openharness-eval-fs-state"
        if persist_cwd
        else root / "state"
    )
    tmp = root / "tmp"

    work.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    ohmo_root = home_path / ".ohmo"
    remap: dict[Path, Path] = {}
    rw_binds: list[tuple[Path, Path]] = []
    mutable_copies: list[str] = []
    for raw_entry in mutable_dirs:
        entry = Path(raw_entry).expanduser()
        name = entry.name
        real_path = entry.resolve() if entry.is_absolute() else (ohmo_root / name).resolve()
        copy_path = state_root / name
        if real_path.exists():
            if persist_cwd:
                if not copy_path.exists():
                    if real_path.is_dir():
                        shutil.copytree(real_path, copy_path, dirs_exist_ok=True)
                    elif real_path.is_file():
                        copy_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(real_path, copy_path)
            else:
                if real_path.is_dir():
                    shutil.copytree(real_path, copy_path, dirs_exist_ok=True)
                elif real_path.is_file():
                    copy_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(real_path, copy_path)
        elif not copy_path.exists():
            copy_path.mkdir(parents=True, exist_ok=True)
        remap[real_path] = copy_path.resolve()
        rw_binds.append((copy_path.resolve(), real_path))
        mutable_copies.append(name)

        if name == "reminders":
            real_reminders_file = (ohmo_root / "reminders.json").resolve()
            if real_reminders_file.exists():
                if persist_cwd and copy_path.exists():
                    (state_root / "reminders.json").parent.mkdir(parents=True, exist_ok=True)
                if persist_cwd and (state_root / "reminders.json").exists():
                    continue
                shutil.copy2(real_reminders_file, state_root / "reminders.json")

    remap[Path("/tmp")] = tmp.resolve()
    resolved_tmp = Path("/tmp").resolve()
    if resolved_tmp != Path("/tmp"):
        remap[resolved_tmp] = tmp.resolve()
    remap[real_cwd] = work.resolve()
    rw_binds.extend(((tmp.resolve(), Path("/tmp")),))
    rw_binds.append((work.resolve(), real_cwd))

    ro_binds: list[Path] = []
    seen_ro: set[Path] = set()
    for raw_path in ro_source_dirs:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or path in seen_ro:
            continue
        ro_binds.append(path)
        seen_ro.add(path)

    return SandboxPlan(
        root=root,
        state_root=state_root.resolve(),
        tmp=tmp.resolve(),
        work=work.resolve(),
        remap=remap,
        ro_binds=tuple(ro_binds),
        rw_binds=tuple(rw_binds),
        cwd=real_cwd,
        home=home_path,
        mutable_copies=tuple(mutable_copies),
    )


class FsSandboxBashTool(BaseTool):
    """Run every bash command inside bwrap with replay fallback on jail failure."""

    name = "bash"
    description = "Eval bash tool executed inside a disposable filesystem sandbox."
    input_model = ReplayToolInput

    def __init__(
        self,
        *,
        mock_tool: BaseTool,
        sandbox_root: str | Path,
        cwd: str | Path,
        home: str | Path,
        ro_binds: Iterable[str | Path],
        rw_binds: Iterable[tuple[str | Path, str | Path]],
        net_mode: str = "none",
        proxy_url: str | None = None,
        browser_socket: str | None = None,
        browser_cli_name: str | None = None,
        bin_dirs: Iterable[str | Path] = (),
        timeout: float = 120.0,
    ) -> None:
        self._mock_tool = mock_tool
        self._sandbox_root = Path(sandbox_root).expanduser().resolve()
        self._cwd = Path(cwd).expanduser().resolve()
        self._home = Path(home).expanduser().resolve()
        self._ro_binds = tuple(Path(path).expanduser().resolve() for path in ro_binds)
        self._rw_binds = tuple(
            (Path(src).expanduser().resolve(), Path(dest).expanduser().resolve())
            for src, dest in rw_binds
        )
        self._net_mode = net_mode
        self._proxy_url = proxy_url
        self._browser_socket = browser_socket
        self._browser_cli_name = browser_cli_name
        self._bin_dirs = tuple(bin_dirs)
        self._timeout = timeout

    async def execute(
        self,
        arguments: ReplayToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        payload = arguments.model_dump()
        raw_command = payload.get("command") or payload.get("cmd") or ""
        command = raw_command if isinstance(raw_command, str) else ""
        # Guarantee the skill-bin is on PATH for the command itself, and use a
        # non-login shell: a login shell (-l) can re-source a profile that
        # clobbers the jail's --setenv PATH, intermittently dropping skill CLIs
        # to "command not found". Export inside the command is bwrap-env-proof.
        if self._bin_dirs:
            bins = ":".join(
                str(Path(b).expanduser().resolve()) for b in self._bin_dirs
            )
            command = f'export PATH="{bins}:$PATH"; {command}'
        argv = build_bwrap_argv(
            sandbox_root=self._sandbox_root,
            cwd=self._cwd,
            home=self._home,
            ro_binds=self._ro_binds,
            rw_binds=self._rw_binds,
            net_mode=self._net_mode,
            proxy_url=self._proxy_url,
            browser_socket=self._browser_socket,
            browser_cli_name=self._browser_cli_name,
            bin_dirs=self._bin_dirs,
        ) + ["bash", "-c", command]
        metadata = {"lane": "fs-sandbox", "net_mode": self._net_mode}
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self._sandbox_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._timeout,
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    output="<timeout>",
                    is_error=True,
                    metadata={**metadata, "timeout": self._timeout},
                )
        except OSError:
            result = await self._mock_tool.execute(arguments, context)
            return ToolResult(
                output=result.output,
                is_error=result.is_error,
                metadata={**result.metadata, "lane": "mock"},
            )

        return ToolResult(
            output=_truncate_output(
                _process_output(
                    stdout=stdout,
                    stderr=stderr,
                    returncode=process.returncode,
                )
            ),
            is_error=process.returncode != 0,
            metadata={**metadata, "returncode": process.returncode},
        )

    def is_read_only(self, arguments: ReplayToolInput) -> bool:
        del arguments
        return True


class SandboxFsTool(BaseTool):
    """Run a real typed filesystem tool against remapped disposable paths."""

    def __init__(
        self,
        *,
        real_tool: BaseTool,
        mock_tool: BaseTool,
        remap: Mapping[str | Path, str | Path],
        cwd: str | Path,
        writable_prefixes: Iterable[str | Path] | None = None,
        path_fields: tuple[str, ...] | None = None,
    ) -> None:
        self.name = real_tool.name
        self.description = real_tool.description
        self.input_model = real_tool.input_model
        self._real_tool = real_tool
        self._mock_tool = mock_tool
        self._cwd = Path(cwd).expanduser().resolve()
        self._remap = {
            Path(src).expanduser().resolve(): Path(dst).expanduser().resolve()
            for src, dst in remap.items()
        }
        self._path_fields = path_fields or _PATH_FIELDS_BY_TOOL.get(
            real_tool.name,
            ("path", "root"),
        )
        writable = writable_prefixes if writable_prefixes is not None else self._remap.values()
        self._writable_prefixes = tuple(
            Path(path).expanduser().resolve() for path in writable
        )

    async def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        try:
            data = arguments.model_dump()
            for field in self._path_fields:
                value = data.get(field)
                if not isinstance(value, str) or not value:
                    continue
                if self.name == "glob" and field == "pattern" and not _is_absolute(value):
                    continue
                data[field] = self._remap_path(value)

            if self.name in _WRITE_TOOL_NAMES:
                forbidden = self._forbidden_write_path(data)
                if forbidden is not None:
                    return ToolResult(
                        output=f"Write outside sandbox refused: {forbidden}",
                        is_error=True,
                        metadata={"lane": "fs-sandbox", "tool": self.name},
                    )

            remapped_args = self._real_tool.input_model(**data)
            result = await self._real_tool.execute(
                remapped_args,
                ToolExecutionContext(
                    cwd=self._cwd,
                    metadata=context.metadata,
                    hook_executor=context.hook_executor,
                ),
            )
        except Exception:
            return await self._mock_tool.execute(arguments, context)

        return ToolResult(
            output=self._remap_output(result.output),
            is_error=result.is_error,
            metadata={**result.metadata, "lane": "fs-sandbox", "tool": self.name},
        )

    def is_read_only(self, arguments: BaseModel) -> bool:
        if self.name in _WRITE_TOOL_NAMES:
            return False
        return self._real_tool.is_read_only(arguments)

    def _remap_path(self, value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            return value
        resolved = path.resolve()
        for source, target in _sorted_remap_items(self._remap):
            if _is_under(resolved, source):
                return str(target / resolved.relative_to(source))
        return value

    def _forbidden_write_path(self, data: Mapping[str, object]) -> Path | None:
        for field in self._path_fields:
            value = data.get(field)
            if not isinstance(value, str) or not value:
                continue
            path = _resolve_against(self._cwd, value)
            if any(_is_under(path, prefix) for prefix in self._writable_prefixes):
                continue
            return path
        return None

    def _remap_output(self, output: str) -> str:
        text = output
        replacements = sorted(
            ((target, source) for source, target in self._remap.items()),
            key=lambda item: len(str(item[0])),
            reverse=True,
        )
        for target, source in replacements:
            text = text.replace(str(target), str(source))
        return text


class FsSandboxAgentRunner:
    """Run QueryEngine evals with real local FS work in a disposable sandbox."""

    name = "fs-sandbox"

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        model: str,
        system_prompt: str = "You are running an fs-sandbox eval.",
        cwd: str | Path | None = None,
        max_turns: int = 8,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        net_mode: str = "none",
        proxy_url: str | None = None,
        browser_socket: str | None = None,
        browser_cli_name: str | None = None,
        persist_cwd: bool = False,
        live_mcp_server_names: tuple[str, ...] = (),
        mutable_dirs: Iterable[str | Path] = ("memory", "todos", "reminders", "user.md"),
        ro_source_dirs: Iterable[str | Path] | None = None,
        extra_ro_source_dirs: Iterable[str | Path] = (),
        sandbox_bin_dirs: Iterable[str | Path] = (),
    ) -> None:
        self._api_client = api_client
        self._model = model
        self._system_prompt = system_prompt
        self._cwd = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._net_mode = net_mode
        self._proxy_url = proxy_url
        self._browser_socket = browser_socket
        self._browser_cli_name = browser_cli_name
        self._persist_cwd = persist_cwd
        self._live_mcp_server_names = tuple(live_mcp_server_names)
        self._mutable_dirs = tuple(mutable_dirs)
        self._ro_source_dirs = (
            tuple(ro_source_dirs)
            if ro_source_dirs is not None
            else (
                Path.home() / ".ohmo" / "skills",
                # Message attachments (voice .ogg, PDFs, images) the agent is
                # asked to open. Read-only: without this the jail can't see them
                # and cases fail with "file not found" though the file exists.
                Path.home() / ".ohmo" / "attachments",
                Path.home() / "bin",
                Path(sys.prefix),
            )
        )
        # Caller-supplied extra read-only mounts (e.g. a Dropbox subfolder holding
        # ticket PDFs a case asks the agent to open). Appended so the built-in
        # defaults above are never dropped. Personal paths stay out of this repo —
        # they come from the private eval driver via --sandbox-ro-dir.
        self._ro_source_dirs = self._ro_source_dirs + tuple(
            Path(p).expanduser() for p in extra_ro_source_dirs
        )
        self._sandbox_bin_dirs = tuple(sandbox_bin_dirs)
        self._home = Path.home().resolve()

    def run(
        self,
        *,
        prompt: str,
        tool_registry: ToolRegistry,
        context: EvalExecutionContext,
    ) -> EvalExecutorResult:
        async def _run_async() -> EvalExecutorResult:
            sandbox_root = Path(
                tempfile.mkdtemp(prefix="openharness-eval-fs-sandbox-")
            ).resolve()
            mcp: McpClientManager | None = None
            mcp_connected = False
            try:
                plan = assemble_fs(
                    sandbox_root,
                    home=self._home,
                    mutable_dirs=self._mutable_dirs,
                    ro_source_dirs=self._ro_source_dirs,
                    cwd=self._cwd,
                    persist_cwd=self._persist_cwd,
                )
                real_registry: ToolRegistry | None = None
                if self._live_mcp_server_names:
                    try:
                        settings = load_settings()
                        all_cfg = load_mcp_server_configs(settings, [])
                        cfg = {
                            name: all_cfg[name]
                            for name in self._live_mcp_server_names
                            if name in all_cfg
                        }
                        if cfg:
                            mcp = McpClientManager(cfg)
                            await mcp.connect_all()
                            mcp_connected = True
                            real_registry = create_default_tool_registry(mcp)
                            for tool in real_registry.list_tools():
                                if tool.name.startswith("mcp__"):
                                    tool_registry.register(tool)
                    except Exception:
                        logger.warning(
                            "fs-sandbox MCP connect failed; %s stays replay",
                            self._live_mcp_server_names,
                            exc_info=True,
                        )

                if real_registry is None:
                    real_registry = create_default_tool_registry(
                        mcp if mcp_connected else None
                    )
                for name in _FS_TOOL_NAMES:
                    real_tool = real_registry.get(name)
                    if real_tool is None:
                        continue
                    mock_tool = tool_registry.get(name) or ReplayFixtureTool(
                        tool_name=name,
                        fixtures=(),
                    )
                    tool_registry.register(
                        SandboxFsTool(
                            real_tool=real_tool,
                            mock_tool=mock_tool,
                            remap=plan.remap,
                            cwd=plan.work,
                        )
                    )

                mock_bash = tool_registry.get("bash") or ReplayFixtureTool(
                    tool_name="bash",
                    fixtures=(),
                )
                tool_registry.register(
                    FsSandboxBashTool(
                        mock_tool=mock_bash,
                        sandbox_root=plan.root,
                        cwd=plan.cwd,
                        home=plan.home,
                        ro_binds=plan.ro_binds,
                        rw_binds=plan.rw_binds,
                        net_mode=self._net_mode,
                        proxy_url=self._proxy_url,
                        browser_socket=self._browser_socket,
                        browser_cli_name=self._browser_cli_name,
                        bin_dirs=self._sandbox_bin_dirs,
                        timeout=self._timeout,
                    )
                )
                local_state_tools = _rebind_local_state_tools(tool_registry, plan.state_root)

                result = await _run_query_engine_replay(
                    api_client=self._api_client,
                    model=self._model,
                    system_prompt=self._system_prompt,
                    cwd=plan.work,
                    max_turns=self._max_turns,
                    max_tokens=self._max_tokens,
                    prompt=prompt,
                    tool_registry=tool_registry,
                    context=context,
                )
                return EvalExecutorResult(
                    final_text=result.final_text,
                    tool_path=result.tool_path,
                    event_kind_path=result.event_kind_path,
                    tool_calls=result.tool_calls,
                    model_calls=result.model_calls,
                    metadata={
                        **result.metadata,
                        "agent_runner": self.name,
                        "sandbox_net_mode": self._net_mode,
                        "live_mcp_servers": list(self._live_mcp_server_names),
                        "mutable_copies": list(plan.mutable_copies),
                        "local_state_tools": list(local_state_tools),
                    },
                )
            finally:
                if mcp is not None:
                    try:
                        await mcp.close()
                    except Exception:
                        logger.warning(
                            "fs-sandbox MCP close failed; continuing eval cleanup",
                            exc_info=True,
                        )
                shutil.rmtree(sandbox_root, ignore_errors=True)

        return _run_eval_coroutine(_run_async())


def _rebind_local_state_tools(tool_registry: ToolRegistry, state_root: Path) -> tuple[str, ...]:
    registered: list[str] = []
    try:
        if tool_registry.get("memory") is not None:
            from ohmo.memory_store import MemoryStore  # noqa: PLC0415
            from ohmo.memory_tool import OhmoMemoryTool  # noqa: PLC0415

            tool_registry.register(OhmoMemoryTool(MemoryStore(state_root)))
            registered.append("memory")

        if tool_registry.get("todo_write") is not None:
            from ohmo.todo_store import TodoStore  # noqa: PLC0415
            from ohmo.todo_write_tool import OhmoTodoWriteTool  # noqa: PLC0415

            tool_registry.register(
                OhmoTodoWriteTool(TodoStore(state_root), lambda: "eval-sandbox")
            )
            registered.append("todo_write")

        reminder_names = ("remind_create", "remind_list", "remind_cancel")
        if any(tool_registry.get(name) is not None for name in reminder_names):
            registered.extend(_rebind_reminder_tools(tool_registry, state_root))
    except Exception:
        logger.warning("fs-sandbox local state tool setup failed", exc_info=True)
    return tuple(registered)


def _rebind_reminder_tools(tool_registry: ToolRegistry, state_root: Path) -> tuple[str, ...]:
    from ohmo.reminders.store import ReminderStore  # noqa: PLC0415
    from ohmo.reminders.tool import (  # noqa: PLC0415
        RemindCancelTool,
        RemindCreateTool,
        RemindListTool,
    )

    default_tz = "Europe/Moscow"
    max_per_chat = 50
    try:
        from ohmo.gateway.runtime import (  # noqa: PLC0415
            DEFAULT_REMINDER_MAX_PER_CHAT,
            DEFAULT_REMINDER_TZ,
        )

        default_tz = DEFAULT_REMINDER_TZ
        max_per_chat = DEFAULT_REMINDER_MAX_PER_CHAT
    except Exception:
        pass

    store = ReminderStore(workspace=state_root)
    lock = asyncio.Lock()
    metadata = {
        "ohmo_reminder_ctx": {
            "channel": "eval",
            "chat_id": "eval-sandbox",
            "session_key": "eval-sandbox",
            "sender_id": "eval-sandbox",
            "chat_type": "private",
            "is_group": False,
            "tz": default_tz,
        }
    }
    rebound: list[str] = []
    if tool_registry.get("remind_create") is not None:
        tool_registry.register(
            _ToolContextMetadataWrapper(
                RemindCreateTool(
                    store,
                    lock,
                    default_tz=default_tz,
                    max_per_chat=max_per_chat,
                ),
                metadata=metadata,
            )
        )
        rebound.append("remind_create")
    if tool_registry.get("remind_list") is not None:
        tool_registry.register(
            _ToolContextMetadataWrapper(
                RemindListTool(store, lock, default_tz=default_tz),
                metadata=metadata,
            )
        )
        rebound.append("remind_list")
    if tool_registry.get("remind_cancel") is not None:
        tool_registry.register(
            _ToolContextMetadataWrapper(
                RemindCancelTool(store, lock),
                metadata=metadata,
            )
        )
        rebound.append("remind_cancel")
    return tuple(rebound)


class _ToolContextMetadataWrapper(BaseTool):
    def __init__(self, tool: BaseTool, *, metadata: Mapping[str, object]) -> None:
        self.name = tool.name
        self.description = tool.description
        self.input_model = tool.input_model
        self._tool = tool
        self._metadata = dict(metadata)

    async def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        return await self._tool.execute(
            arguments,
            ToolExecutionContext(
                cwd=context.cwd,
                metadata={**context.metadata, **self._metadata},
                hook_executor=context.hook_executor,
            ),
        )

    def is_read_only(self, arguments: BaseModel) -> bool:
        return self._tool.is_read_only(arguments)


def _sorted_remap_items(remap: Mapping[Path, Path]) -> list[tuple[Path, Path]]:
    return sorted(remap.items(), key=lambda item: len(str(item[0])), reverse=True)


def _resolve_against(cwd: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_absolute(value: str) -> bool:
    return Path(value).expanduser().is_absolute()
