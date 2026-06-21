"""Effective tool / capability labels for eval trajectories.

Agents that route capabilities through a shell/exec tool (e.g. ohmo runs
everything via ``bash``) collapse the trajectory: arithmetic and a weather
lookup both look like ``["bash"]``. The real capability is encoded in the
command string. These helpers derive an *effective tool label* —
``bash:<binary>``, and for CLI-style tools ``bash:<binary> <subcommand>``
(e.g. ``bash:maps-cli reviews``) — so a trajectory/motif distinguishes
capabilities even when they are argument-encoded. For typed tools the label is
just the tool name, so name-based analysis is unaffected.

This is a methodology primitive, not an ohmo quirk: any agent that wraps real
capabilities behind a generic exec tool needs the capability lifted out of the
arguments before a trajectory means anything.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from typing import Any

SHELL_TOOL_NAMES = frozenset({"bash", "sh", "shell", "zsh", "shell_exec", "run_shell"})
_COMMAND_KEYS = ("command", "cmd", "script")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1.*?^\s*\2\s*$", re.DOTALL | re.MULTILINE)
_SEPARATORS = re.compile(r"\|\||&&|[;\n|&]")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_WRAPPERS = frozenset(
    {"sudo", "env", "nohup", "time", "exec", "command", "builtin", "nice",
     "ionice", "stdbuf", "xargs", "watch", "then", "do", "else", "timeout"}
)
_SKIP = frozenset({"cd", "pushd", "popd", ":", "true", "false", "export", "source", "."})
_PLUMBING = frozenset(
    {
        "set",
        "mkdir",
        "rmdir",
        "rm",
        "mv",
        "cp",
        "ln",
        "touch",
        "chmod",
        "chown",
        "cat",
        "echo",
        "ls",
        "test",
        "[",
        "head",
        "tail",
        "printf",
        "kill",
        "sleep",
        "pwd",
        "which",
        "mkfifo",
        "apt",
        "apt-get",
        "brew",
        "dpkg",
        "yum",
        "dnf",
        "snap",
    }
)
_SHELL_KEYWORDS = frozenset(
    {
        "for",
        "while",
        "until",
        "if",
        "elif",
        "then",
        "else",
        "fi",
        "case",
        "esac",
        "select",
        "function",
        "do",
        "done",
        "in",
        "time",
    }
)
# Binaries whose first positional token is a meaningful subcommand (capability).
# Skill CLIs follow the ``*-cli`` convention; a few common multi-command tools
# are listed explicitly.
_SUBCOMMAND_BINARIES = frozenset(
    {"git", "docker", "kubectl", "npm", "yarn", "pnpm", "cargo", "go", "pip",
     "pip3", "poetry", "gh", "ya", "systemctl"}
)
_SUBCOMMAND_RE = re.compile(r"^[a-z][a-z0-9][a-z0-9-]*$")


def _command_segments(command: str) -> list[list[str]]:
    """Tokenized command segments, each starting at the invoked binary.

    Heredoc bodies are stripped, the command is split on shell separators, and
    leading ``VAR=val`` assignments / wrappers (``sudo``/``timeout``…) skipped.
    """
    if not command:
        return []
    cleaned = _HEREDOC.sub(" ", command)
    segments: list[list[str]] = []
    for segment in _SEPARATORS.split(cleaned):
        stripped = segment.strip()
        if not stripped:
            continue
        line = stripped.splitlines()[0].strip()
        if not line:
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()
        i = 0
        while i < len(tokens) and (
            _ASSIGN.match(tokens[i])
            or tokens[i] in _WRAPPERS
            or tokens[i].startswith("-")
        ):
            i += 1
        if i < len(tokens):
            segments.append(tokens[i:])
    return segments


def _binary_of(token: str) -> str:
    return token.rsplit("/", 1)[-1]


def _is_noncapability_binary(binary: str) -> bool:
    return (
        not binary
        or binary.startswith("-")
        or binary in _SKIP
        or binary in _PLUMBING
        or binary in _SHELL_KEYWORDS
        or bool(_ASSIGN.match(binary))
    )


def _segment_capability(tokens: list[str]) -> str | None:
    """``binary`` or ``binary subcommand`` for one command segment."""
    binary = _binary_of(tokens[0])
    if _is_noncapability_binary(binary):
        return None
    if binary.endswith("-cli") or binary in _SUBCOMMAND_BINARIES:
        for token in tokens[1:]:
            if token.startswith("-"):
                continue  # a flag — the subcommand may still follow
            if _SUBCOMMAND_RE.match(token):
                return f"{binary} {token}"
            break  # first positional isn't a subcommand-like word
    return binary


def extract_command_binaries(command: str) -> list[str]:
    """Ordered, de-duplicated list of executables a shell command invokes."""
    out: list[str] = []
    for tokens in _command_segments(command):
        binary = _binary_of(tokens[0])
        if _is_noncapability_binary(binary):
            continue
        if binary not in out:
            out.append(binary)
    return out


def command_capabilities(command: str) -> list[str]:
    """Ordered, de-duplicated capabilities (``binary`` / ``binary subcommand``)."""
    out: list[str] = []
    for tokens in _command_segments(command):
        cap = _segment_capability(tokens)
        if cap and cap not in out:
            out.append(cap)
    return out


def _command_text(arguments: Any) -> str:
    if isinstance(arguments, Mapping):
        for key in _COMMAND_KEYS:
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    if isinstance(arguments, str):
        return arguments
    return ""


def tool_call_binaries(tool_name: str, arguments: Any) -> list[str]:
    """Binaries invoked by a shell tool call (empty for non-shell tools)."""
    if (tool_name or "").strip() in SHELL_TOOL_NAMES:
        return extract_command_binaries(_command_text(arguments))
    return []


def effective_tool_label(tool_name: str, arguments: Any = None) -> str:
    """Capability-aware label: ``<shell>:<binary> [subcommand]`` for shell tools, else the name."""
    name = (tool_name or "").strip()
    if name in SHELL_TOOL_NAMES:
        caps = command_capabilities(_command_text(arguments))
        if caps:
            return f"{name}:{caps[0]}"
    return name


def effective_tool_path(events: Sequence[Any]) -> list[str]:
    """Effective-label tool path for EvalEvents, de-duplicated per tool call."""
    path: list[str] = []
    seen: set[tuple[str, str]] = set()
    started: set[tuple[str, str]] = set()
    for index, event in enumerate(events):
        tool_name = getattr(event, "tool_name", None)
        if not tool_name:
            continue
        call_id = getattr(event, "tool_call_id", None)
        call_key = (tool_name, call_id or f"event-{index}")
        if getattr(event, "kind", None) == "tool_started":
            started.add(call_key)
        elif call_id and call_key in started:
            continue
        if call_key in seen:
            continue
        seen.add(call_key)
        arguments = (getattr(event, "payload", None) or {}).get("input")
        path.append(effective_tool_label(tool_name, arguments))
    return path
