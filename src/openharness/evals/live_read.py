"""Live read lane for evals.

Only ``bash`` calls that are single-segment invocations of explicitly
allowlisted read-only skill CLIs are executed for real. All other tools,
including typed web tools, remain replay fixtures in this slice unless a
specific MCP server is enabled as a live read lane.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import shutil
import tempfile
from collections.abc import Collection
from pathlib import Path

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
from openharness.evals.tool_labels import _binary_of, _command_segments
from openharness.mcp.client import McpClientManager
from openharness.mcp.config import load_mcp_server_configs
from openharness.tools import create_default_tool_registry
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

READ_LIVE_BASH_ALLOWLIST = frozenset(
    {
        "maps-cli",
        "afisha-cli",
        "weather-cli",
        "travel-cli",
        "calendar-cli",
        "pdf-extract-ocr",
    }
)
_SHELL_OPERATORS = re.compile(r"[;|&<>`]|\$\(|&&|\|\||\n")
_OUTPUT_CHAR_LIMIT = 20_000
_TRUNCATED_SUFFIX = "\n[truncated]"


def classify_bash_command(
    command: str,
    allowlist: Collection[str] = READ_LIVE_BASH_ALLOWLIST,
) -> tuple[str, list[str]]:
    """Classify a bash command as live-safe or replay-only."""
    if not isinstance(command, str) or not command.strip():
        return "mock", []
    if _SHELL_OPERATORS.search(command):
        return "mock", []
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "mock", []
    if not tokens:
        return "mock", []
    segments = _command_segments(command)
    if len(segments) != 1 or segments[0] != tokens:
        return "mock", []
    if _binary_of(tokens[0]) not in allowlist:
        return "mock", []
    return "live", tokens


class LiveReadBashTool(BaseTool):
    """Run allowlisted read-only bash commands live and replay everything else."""

    name = "bash"
    description = "Eval bash tool with a live read-only lane and replay fallback."
    input_model = ReplayToolInput

    def __init__(
        self,
        *,
        mock_tool: BaseTool,
        allowlist: Collection[str] = READ_LIVE_BASH_ALLOWLIST,
        cwd: Path,
        timeout: float = 120.0,
    ) -> None:
        self._mock_tool = mock_tool
        self._allowlist = frozenset(allowlist)
        self._cwd = Path(cwd).resolve()
        self._timeout = timeout

    async def execute(
        self,
        arguments: ReplayToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        payload = arguments.model_dump()
        raw_command = payload.get("command") or payload.get("cmd") or ""
        command = raw_command if isinstance(raw_command, str) else ""
        lane, tokens = classify_bash_command(command, self._allowlist)
        if lane != "live":
            result = await self._mock_tool.execute(arguments, context)
            return ToolResult(
                output=result.output,
                is_error=result.is_error,
                metadata={**result.metadata, "lane": "mock"},
            )
        return await self._execute_live(tokens)

    async def _execute_live(self, tokens: list[str]) -> ToolResult:
        binary = tokens[0]
        metadata = {"lane": "live", "binary": binary}
        try:
            process = await asyncio.create_subprocess_exec(
                binary,
                *tokens[1:],
                cwd=str(self._cwd),
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
        except OSError as exc:
            return ToolResult(
                output=str(exc),
                is_error=True,
                metadata=metadata,
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


class LiveReadAgentRunner:
    """Run QueryEngine evals with selected live read lanes."""

    name = "query-engine-live-read"

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        model: str,
        system_prompt: str = "You are running a live-read eval.",
        cwd: str | Path | None = None,
        max_turns: int = 8,
        max_tokens: int = 4096,
        allowlist: Collection[str] = READ_LIVE_BASH_ALLOWLIST,
        timeout: float = 120.0,
        live_mcp_server_names: tuple[str, ...] = (),
    ) -> None:
        self._api_client = api_client
        self._model = model
        self._system_prompt = system_prompt
        self._cwd = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._allowlist = frozenset(allowlist)
        self._timeout = timeout
        self._live_mcp_server_names = tuple(live_mcp_server_names)

    def run(
        self,
        *,
        prompt: str,
        tool_registry: ToolRegistry,
        context: EvalExecutionContext,
    ) -> EvalExecutorResult:
        async def _run_async() -> EvalExecutorResult:
            live_cwd = Path(
                tempfile.mkdtemp(prefix="openharness-eval-live-read-")
            ).resolve()
            mcp: McpClientManager | None = None
            try:
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
                            real_registry = create_default_tool_registry(mcp)
                            for tool in real_registry.list_tools():
                                if tool.name.startswith("mcp__"):
                                    tool_registry.register(tool)
                    except Exception:
                        logger.warning(
                            "live-read MCP connect failed; %s stays replay",
                            self._live_mcp_server_names,
                            exc_info=True,
                        )

                mock_bash = tool_registry.get("bash") or ReplayFixtureTool(
                    tool_name="bash",
                    fixtures=(),
                )
                tool_registry.register(
                    LiveReadBashTool(
                        mock_tool=mock_bash,
                        allowlist=self._allowlist,
                        cwd=live_cwd,
                        timeout=self._timeout,
                    )
                )
                result = await _run_query_engine_replay(
                    api_client=self._api_client,
                    model=self._model,
                    system_prompt=self._system_prompt,
                    cwd=live_cwd,
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
                    metadata={
                        **result.metadata,
                        "agent_runner": self.name,
                        "live_mcp_servers": list(self._live_mcp_server_names),
                    },
                )
            finally:
                if mcp is not None:
                    try:
                        await mcp.close()
                    except Exception:
                        logger.warning(
                            "live-read MCP close failed; continuing eval cleanup",
                            exc_info=True,
                        )
                shutil.rmtree(live_cwd, ignore_errors=True)

        return _run_eval_coroutine(_run_async())


def _process_output(*, stdout: bytes, stderr: bytes, returncode: int | None) -> str:
    output = stdout.decode("utf-8", errors="replace")
    if returncode and stderr:
        if output and not output.endswith("\n"):
            output += "\n"
        output += stderr.decode("utf-8", errors="replace")
    return output


def _truncate_output(output: str) -> str:
    if len(output) <= _OUTPUT_CHAR_LIMIT:
        return output
    keep = _OUTPUT_CHAR_LIMIT - len(_TRUNCATED_SUFFIX)
    return output[:keep] + _TRUNCATED_SUFFIX
