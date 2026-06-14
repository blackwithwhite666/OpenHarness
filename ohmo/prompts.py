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
                "trivial one-shot answers. Each chat has its OWN list. When you move "
                "on to a NEW, UNRELATED task, call `todo_write new_list=true` to start "
                "a clean list (the previous one is archived to its own file) instead of "
                "carrying another task's items forward; within a single task, "
                "`clear_completed=true` prunes finished items and `remove=true` drops a "
                "single one. Keep the list scoped to the current request."
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
            "# Attaching files",
            (
                "You CAN send a file (an HTML report, image, PDF, any generated "
                "artifact) as a Telegram attachment — you do NOT need a special "
                "tool for it. Write the file to disk, then put a marker on its own "
                "line in your reply: `[[attach: /absolute/path/to/file]]`. The "
                "gateway strips that marker out of the visible text and sends the "
                "file as a Telegram document. Use an ABSOLUTE path on THIS machine "
                "(where you run your tools). One marker per file; repeat the marker "
                "for several files. This is the normal, working way to deliver a "
                "generated report/image — DO it instead of claiming you can't "
                "attach files or offering to copy them to Dropbox."
            ),
        ]
    )

    sections.extend(
        [
            "# Asking with buttons",
            (
                "When you need the user to pick between a few concrete options "
                "(a choice, a confirmation), END your reply with a marker on its "
                "own line: `[[ask: <your question> | <option 1> | <option 2> | "
                "…]]`. The gateway shows the question with TAPPABLE buttons, and "
                "the tap arrives as the user's next message — so just continue "
                "from their choice on your next turn. Put the question only in the "
                "marker (don't also repeat it in prose), keep each option SHORT "
                "(a few words), and give 2–6 options. Use it for genuine forks "
                "(`[[ask: Куда едем в выходные? | Питер | Москва | Дома]]`), not "
                "for open-ended questions — for those just ask in plain text."
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

    sections.extend(
        [
            "# Reminders",
            (
                "You can set persistent proactive reminders that fire even after a "
                "restart, using `remind_create` / `remind_list` / `remind_cancel`. "
                "When the user asks to be reminded ('напомни … в 18:00', 'каждый "
                "будний день в 9 присылай погоду'), YOU resolve the natural-language "
                "time into an absolute tz-aware `dtstart` and, for repeats, an iCal "
                "`RRULE` — the tool does not parse free-form text. Ground relative "
                "phrases ('через 2 часа', 'завтра в 9') against the current time, "
                "which every reminder-tool result echoes back. Use `mode='static'` "
                "to deliver the text verbatim, or `mode='agentic'` when the reminder "
                "should run a full agent action at fire time (e.g. fetch and send "
                "the weather). Never pass chat_id — delivery targets this chat "
                "automatically."
            ),
        ]
    )

    if ohmo_memory := load_ohmo_memory_prompt(root):
        sections.append(ohmo_memory)

    if include_project_memory:
        project_memory = load_project_memory_prompt(cwd)
        if project_memory:
            sections.append(project_memory)

    return "\n\n".join(section for section in sections if section and section.strip())
