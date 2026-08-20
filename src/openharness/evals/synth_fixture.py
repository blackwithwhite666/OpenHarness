"""LLM-synthesized replay fixture tools.

The generated code is a pure data transform over the captured fixture data,
produced by our own grounded codegen (not adversarial input). It is executed:
  - via ``asyncio.create_subprocess_exec(sys.executable, "-I", "-S", "-c", RUNNER)``
    -- never ``shell=True``;
  - cwd = a throwaway temp dir (rmtree in finally);
  - with a wall-clock timeout (default 20s; kill on timeout);
  - with a minimal env (do not inherit the parent env / secrets) -- pass only
    what python needs to start (e.g. ``env={"PATH": os.environ.get("PATH", "")}``);
  - the subprocess receives only ``{code, arguments, captured}`` as JSON on
    stdin -- no api keys, no tokens, no files;
  - output is truncated to a sane cap (20,000 chars).

It is never executed by the live gateway -- only by an explicit
``--fixture-match synth`` eval run. This is process-level isolation
proportionate to a buggy-code (not malicious-code) threat model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
from typing import Any

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    SupportsStreamingMessages,
)
from openharness.engine.messages import ConversationMessage
from openharness.evals.executor import (
    EvalToolFixture,
    ReplayFixtureTool,
    ReplayToolInput,
    SynthContext,
)
from openharness.tools.base import ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)

_PROMPT_EXAMPLE_CHAR_LIMIT = 4_000
_OUTPUT_CHAR_LIMIT = 20_000
_TRUNCATED_SUFFIX = "\n[truncated]"

SYNTH_CODEGEN_SYSTEM_PROMPT = """\
You write deterministic Python replay fixture adapters.

Output ONLY the source of:

def respond(arguments, captured):
    ...

Contract:
- arguments is a dict containing the agent's tool arguments.
- captured is a list of dicts {"input": str, "output": str, "is_error": bool}
  containing the real captured examples for this tool.
- respond returns ONE string: the tool output.

Rules:
- Ground every returned fact in captured; you may filter, sort, slice, reformat,
  or select from captured outputs to fit arguments.
- Do not fabricate facts (names, numbers, URLs, results) absent from captured.
  If arguments ask for something not present, return the most relevant captured
  output or a short honest "no data" string -- never invent.
- Standard library only; pure function; no network/file/OS access; no I/O.
- Output ONLY the function source (no markdown fences, no prose).
"""

RUNNER = r"""
import json
import sys

try:
    payload = json.loads(sys.stdin.read())
    namespace = {}
    exec(payload["code"], namespace)
    respond = namespace["respond"]
    result = respond(payload.get("arguments") or {}, payload.get("captured") or [])
    print(json.dumps({"ok": True, "output": str(result)}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": repr(exc)}))
"""

SYNTH_STATE_CODEGEN_SYSTEM_PROMPT = """\
You write deterministic Python replay fixture adapters WITH STATE.

Output ONLY the source of:

def respond(arguments, captured, state):
    ...

Contract:
- arguments is a dict containing the agent's tool arguments for THIS call.
- captured is a list of dicts {"input": str, "output": str, "is_error": bool}
  containing the real captured examples for this tool.
- state is a dict that PERSISTS across calls to THIS tool within one eval case.
  It starts as {} on the first call. Mutate it IN PLACE to remember anything you
  need across calls. It must stay JSON-serializable.
- respond returns ONE string: the tool output.

Use state to model call-order-dependent behavior a stateless function cannot:
- Recovery: if captured shows a transient error then a success for the same
  request, return the error on the first matching call and the success on the
  retry -- track a per-request attempt counter in state.
- Pagination / progression: advance a cursor kept in state.
- Effects of earlier calls to THIS tool (e.g. an item added earlier shows up in
  a later listing).

Rules:
- Ground every returned FACT in captured; never invent names/numbers/URLs/results
  absent from captured. state holds only control info (counters, cursors, seen
  keys/ids) and values you already grounded in captured.
- If arguments ask for something not present, return the most relevant captured
  output or a short honest "no data" string -- never invent.
- Standard library only; deterministic given (arguments, captured, state); no
  network/file/OS access; no I/O; no randomness; no wall-clock time.
- Output ONLY the function source (no markdown fences, no prose).
"""

RUNNER_STATE = r"""
import json
import sys

try:
    payload = json.loads(sys.stdin.read())
    namespace = {}
    exec(payload["code"], namespace)
    respond = namespace["respond"]
    state = payload.get("state") or {}
    result = respond(payload.get("arguments") or {}, payload.get("captured") or [], state)
    output = result
    new_state = state
    if isinstance(result, (list, tuple)) and len(result) == 2 and isinstance(result[1], dict):
        output, new_state = result[0], result[1]
    print(json.dumps({"ok": True, "output": str(output), "state": new_state}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": repr(exc)}))
"""


def _codegen_prompt(
    tool_name: str,
    fixtures: tuple[EvalToolFixture, ...],
    *,
    stateful: bool = False,
) -> str:
    examples = [
        {
            "input": _truncate_for_prompt(fixture.input_text),
            "output": _truncate_for_prompt(fixture.output_text),
            "is_error": fixture.is_error,
        }
        for fixture in fixtures
    ]
    signature = (
        "respond(arguments, captured, state)"
        if stateful
        else "respond(arguments, captured)"
    )
    return (
        f"Write a deterministic {signature} function for this replay tool. Use "
        "only the captured examples below as factual ground.\n\n"
        f"Tool name: {tool_name}\n\n"
        "Captured examples (input/output truncated for prompt budget only; the "
        "runtime receives full captured values):\n"
        f"{json.dumps(examples, ensure_ascii=False, indent=2)}"
    )


class SynthesizedFixtureTool(ReplayFixtureTool):
    """Replay fixture tool with a synthesized, grounded transform lane."""

    def __init__(
        self,
        *,
        tool_name: str,
        fixtures: tuple[EvalToolFixture, ...],
        api_client: SupportsStreamingMessages,
        model: str,
        fallback_match_mode: str = "order",
        system_prompt: str | None = None,
        codegen_max_tokens: int = 2048,
        exec_timeout: float = 20.0,
        stateful: bool = False,
    ) -> None:
        super().__init__(
            tool_name=tool_name,
            fixtures=fixtures,
            match_mode=fallback_match_mode,
        )
        self._api_client = api_client
        self._model = model
        self._stateful = stateful
        self._system_prompt = system_prompt or (
            SYNTH_STATE_CODEGEN_SYSTEM_PROMPT if stateful else SYNTH_CODEGEN_SYSTEM_PROMPT
        )
        self._codegen_max_tokens = codegen_max_tokens
        self._exec_timeout = exec_timeout
        self._code: str | None = None
        self._codegen_attempted = False
        # Per-tool state that persists across calls within one eval case. Only
        # used in stateful mode; lets the synthesized adapter model recovery
        # (error-then-retry), pagination, and effects of earlier same-tool calls.
        self._state: dict[str, Any] = {}

    async def execute(
        self,
        arguments: ReplayToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        try:
            result = await self._synth_execute(arguments)
            if result is not None:
                return result
        except Exception:
            logger.warning(
                "synth fixture execution failed for %s; falling back",
                self.name,
                exc_info=True,
            )

        fb = await super().execute(arguments, context)
        return ToolResult(
            output=fb.output,
            is_error=fb.is_error,
            metadata={**fb.metadata, "synth": False, "lane": "synth-fallback"},
        )

    async def _synth_execute(self, arguments: ReplayToolInput) -> ToolResult | None:
        code = await self._ensure_code()
        if code is None:
            return None
        output = await self._run_code(code, dict(arguments.model_dump()))
        if output is None:
            return None
        return ToolResult(
            output=output,
            is_error=False,
            metadata={"synth": True, "lane": "synth", "tool": self.name},
        )

    async def _ensure_code(self) -> str | None:
        if self._codegen_attempted:
            return self._code
        self._codegen_attempted = True
        try:
            text = await self._complete(
                _codegen_prompt(self.name, self._fixtures, stateful=self._stateful)
            )
            text = _strip_code_fences(text)
            if "def respond(" not in text:
                self._code = None
                return None
            self._code = text
            return self._code
        except Exception:
            logger.warning(
                "synth fixture codegen failed for %s; falling back",
                self.name,
                exc_info=True,
            )
            self._code = None
            return None

    async def _complete(self, prompt: str) -> str:
        text = ""
        async for event in self._api_client.stream_message(
            ApiMessageRequest(
                model=self._model,
                messages=[ConversationMessage.from_user_text(prompt)],
                system_prompt=self._system_prompt,
                max_tokens=self._codegen_max_tokens,
                tools=[],
            )
        ):
            if isinstance(event, ApiMessageCompleteEvent):
                text = event.message.text.strip()
        return text

    async def _run_code(self, code: str, arguments: dict[str, Any]) -> str | None:
        captured = [
            {
                "input": fixture.input_text,
                "output": fixture.output_text,
                "is_error": fixture.is_error,
            }
            for fixture in self._fixtures
        ]
        request: dict[str, Any] = {
            "code": code,
            "arguments": arguments,
            "captured": captured,
        }
        if self._stateful:
            request["state"] = self._state
        payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
        tmp = tempfile.mkdtemp(prefix="openharness-synth-fixture-")
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-S",
                "-c",
                RUNNER_STATE if self._stateful else RUNNER,
                cwd=tmp,
                env={"PATH": os.environ.get("PATH", "")},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _stderr = await asyncio.wait_for(
                    process.communicate(payload),
                    timeout=self._exec_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return None
            if process.returncode not in (0, None):
                return None
            try:
                result = json.loads(stdout.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                return None
            if not isinstance(result, dict) or result.get("ok") is not True:
                return None
            if self._stateful and isinstance(result.get("state"), dict):
                # Carry the mutated state forward to the next call to this tool.
                self._state = result["state"]
            return _truncate_output(str(result.get("output", "")))
        except OSError:
            logger.warning(
                "synth fixture subprocess failed for %s; falling back",
                self.name,
                exc_info=True,
            )
            return None
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _truncate_for_prompt(text: str) -> str:
    if len(text) <= _PROMPT_EXAMPLE_CHAR_LIMIT:
        return text
    keep = _PROMPT_EXAMPLE_CHAR_LIMIT - len(_TRUNCATED_SUFFIX)
    return text[:keep] + _TRUNCATED_SUFFIX


def _truncate_output(output: str) -> str:
    if len(output) <= _OUTPUT_CHAR_LIMIT:
        return output
    keep = _OUTPUT_CHAR_LIMIT - len(_TRUNCATED_SUFFIX)
    return output[:keep] + _TRUNCATED_SUFFIX


__all__ = [
    "RUNNER",
    "RUNNER_STATE",
    "SYNTH_CODEGEN_SYSTEM_PROMPT",
    "SYNTH_STATE_CODEGEN_SYSTEM_PROMPT",
    "SynthContext",
    "SynthesizedFixtureTool",
    "_codegen_prompt",
]
