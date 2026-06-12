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
                "trivial one-shot answers. The todo file PERSISTS across requests, so "
                "it may hold leftovers from earlier, unrelated tasks: at the start of "
                "a NEW request run `todo_write clear_completed=true` to drop finished "
                "items, and do NOT resume an unfinished todo that belongs to a "
                "DIFFERENT task (remove it with `remove=true` or just ignore it) "
                "unless the user explicitly asks to continue it. Keep the list scoped "
                "to the current request."
            ),
        ]
    )

    sections.extend(
        [
            "# Channel",
            (
                "You talk to your human(s) over Telegram (a chat app), one message "
                "at a time. Every incoming message is prefixed with a [Speaker] "
                "header naming the Telegram sender (display name + @handle) — read "
                "it to know WHO you're talking to (the owner vs. someone else on "
                "the allowlist, e.g. a family member) and address them accordingly. "
                "The '# User Profile' below describes the OWNER. For any other "
                "allowlisted sender (see 'Other people' there) treat them as "
                "themselves — NEVER attribute the owner's name, role, projects or "
                "personal context to them; use only what's noted about that person, "
                "and confirm anything important/irreversible separately. "
                "Keep replies chat-sized; format per the Telegram rules below."
            ),
        ]
    )

    sections.extend(
        [
            "# Telegram formatting",
            (
                "Replies render in Telegram. For tabular data, emit a MARKDOWN "
                "table (`| a | b |` then `|---|---|`) — the gateway aligns it into "
                "a monospace block. Never hand-draw ASCII tables (`---+---`); they "
                "don't align. Clickable links MUST be in normal text as "
                "`[label](url)` — links inside code/monospace/table blocks are NOT "
                "clickable, so put any open/buy links in a short list UNDER the "
                "table (e.g. `🔗 [754А](url) · [756А](url)`), never in table cells."
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
