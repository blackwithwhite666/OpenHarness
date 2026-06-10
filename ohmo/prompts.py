"""Prompt assembly for ohmo persona and workspace context."""

from __future__ import annotations

from pathlib import Path

from openharness.memory import load_memory_prompt as load_project_memory_prompt
from openharness.prompts.system_prompt import get_base_system_prompt

from ohmo.memory import load_memory_prompt as load_ohmo_memory_prompt
from ohmo.workspace import (
    get_bootstrap_path,
    get_identity_path,
    get_soul_path,
    get_user_path,
    get_workspace_root,
)


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    return content or None


def build_ohmo_system_prompt(
    cwd: str | Path,
    *,
    workspace: str | Path | None = None,
    extra_prompt: str | None = None,
    include_project_memory: bool = False,
) -> str:
    """Build the custom base prompt for ohmo sessions."""
    root = get_workspace_root(workspace)
    sections = [get_base_system_prompt()]

    if extra_prompt:
        sections.extend(["# Additional Instructions", extra_prompt.strip()])

    soul = _read_text(get_soul_path(root))
    if soul:
        sections.extend(["# ohmo Soul", soul])

    identity = _read_text(get_identity_path(root))
    if identity:
        sections.extend(["# ohmo Identity", identity])

    user = _read_text(get_user_path(root))
    if user:
        sections.extend(["# User Profile", user])

    bootstrap = _read_text(get_bootstrap_path(root))
    if bootstrap:
        sections.extend(["# First-Run Bootstrap", bootstrap])

    sections.extend(
        [
            "# Staying on track (multi-step work)",
            (
                "For any task that needs more than ~2 tool calls (browser flows, "
                "multi-train checks, research, multi-file edits): FIRST call the "
                "`todo_write` tool to lay out the concrete steps, then mark each one "
                "`completed` as you finish it, keeping exactly one `in_progress`. The "
                "todo list is your working memory across tool calls — it stops you "
                "from losing the plan, re-doing steps, sequencing wrong, or stopping "
                "early. Re-read it before deciding the next action. Skip it only for "
                "trivial one-shot answers."
            ),
        ]
    )

    sections.extend(
        [
            "# ohmo Workspace",
            f"- Personal workspace root: {root}",
            "- Personal memory and sessions live under the shared ohmo workspace root.",
            "- Resume only within ohmo sessions; do not assume interoperability with plain OpenHarness sessions.",
        ]
    )

    if ohmo_memory := load_ohmo_memory_prompt(root):
        sections.append(ohmo_memory)

    if include_project_memory:
        project_memory = load_project_memory_prompt(cwd)
        if project_memory:
            sections.append(project_memory)

    return "\n\n".join(section for section in sections if section and section.strip())
