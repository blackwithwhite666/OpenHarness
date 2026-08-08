"""Prompt assembly for ohmo persona and workspace context."""

from __future__ import annotations

from pathlib import Path

from ohmo.memory import load_memory_prompt as load_ohmo_memory_prompt
from ohmo.threat_patterns import scan_for_threats
from ohmo.workspace import (
    get_bootstrap_path,
    get_identity_path,
    get_soul_path,
    get_user_path,
    get_workspace_root,
)
from openharness.memory import load_memory_prompt as load_project_memory_prompt
from openharness.prompts.system_prompt import get_base_system_prompt


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    return content or None


def _read_text_scanned(path: Path, label: str) -> str | None:
    """Read a persona/context file, replacing it with a placeholder if it carries
    an injection/exfil payload — these go verbatim into the always-on system
    prompt, so a file poisoned on disk (compromised tool / sister session) must
    not pass through. Scope "all" is the low-FP subset, so legitimate persona
    phrasings ("you are ohmo, …") are not falsely blanked. The file is left intact.
    """
    content = _read_text(path)
    if content is None:
        return None
    findings = scan_for_threats(content, scope="all")
    if findings:
        return (
            f"[BLOCKED: {label} contained threat pattern(s): {', '.join(findings)}. "
            f"Edit the file to remove it.]"
        )
    return content


def build_ohmo_system_prompt(
    cwd: str | Path,
    *,
    workspace: str | Path | None = None,
    extra_prompt: str | None = None,
    include_project_memory: bool = False,
    include_ohmo_memory: bool = True,
    include_ohmo_workspace: bool | None = None,
) -> str:
    """Build the custom base prompt for ohmo sessions."""
    root = get_workspace_root(workspace)
    if include_ohmo_workspace is None:
        include_ohmo_workspace = include_ohmo_memory
    sections = [get_base_system_prompt()]

    if extra_prompt:
        sections.extend(["# Additional Instructions", extra_prompt.strip()])

    soul = _read_text_scanned(get_soul_path(root), "soul.md")
    if soul:
        sections.extend(["# ohmo Soul", soul])

    identity = _read_text_scanned(get_identity_path(root), "identity.md")
    if identity:
        sections.extend(["# ohmo Identity", identity])

    user = _read_text_scanned(get_user_path(root), "user.md")
    if user:
        sections.extend(["# User Profile", user])

    bootstrap = _read_text_scanned(get_bootstrap_path(root), "BOOTSTRAP.md")
    if bootstrap:
        sections.extend(["# First-Run Bootstrap", bootstrap])

    sections.extend(
        [
            "# Staying on track (multi-step work)",
            (
                "For any task that needs more than ~2 tool calls (browser flows, "
                "multi-train checks, research, multi-file edits): FIRST call the "
                "`todo_write` with the COMPLETE desired snapshot in `todos`, then "
                "resubmit the full snapshot whenever an item changes. Each item has "
                "`content` and one status: `pending`, `in_progress`, `completed`, or "
                "`blocked`; keep exactly one `in_progress` item at most. Include a "
                "non-empty `blocked_reason` only for blocked items, and use "
                "`completed` for finished work. The todo list is your working memory "
                "across tool calls — it stops you from losing the plan, re-doing steps, "
                "sequencing wrong, or stopping early. Re-read it before deciding the "
                "next action. Skip it only for trivial one-shot answers. Each chat has "
                "its OWN list. Treat every submission as replacement of the entire "
                "list; send `todos=[]` to clear it. Keep the list scoped to the current "
                "request."
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

    if include_ohmo_workspace:
        sections.extend(_build_ohmo_workspace_sections(root))

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

    sections.extend(
        [
            "# Wellness and nutrition data",
            (
                "For health, nutrition, wellness, or activity questions about a "
                "configured participant, use `get_wellness_data`. Treat its response "
                "as authoritative: identify the participant only by the returned "
                "`login`, never by a memory guess or by mapping an id yourself. "
                "The gateway supplies the trusted default participant and may expose "
                "an owner-only numeric participant selector; never use `user_id` or "
                "`health_types` parameters."
            ),
            "# Nutrition finalization annotations",
            (
                "If the user asks for calorie/macronutrient estimates (including from an "
                "image), include `annotations.nutrition` in `trace_finalization`."
            ),
            (
                "Required shape (schema v2):\n"
                "```\n"
                "{\n"
                '  "schema_version": 2,\n'
                '  "record_type": "meal_observation",\n'
                '  "basis": ["image"],\n'
                '  "consumption_status": "unknown",\n'
                '  "meal_date": null,\n'
                '  "meal_at": null,\n'
                '  "is_estimate": true,\n'
                '  "energy_kcal_min": 200,\n'
                '  "energy_kcal_max": 300,\n'
                '  "energy_kcal_best": 250,\n'
                '  "protein_g": null,\n'
                '  "fat_g": null,\n'
                '  "carbohydrate_g": null,\n'
                '  "confidence": "medium",\n'
                '  "items": [\n'
                "    {\n"
                '      "name": "food_name",\n'
                '      "quantity_text": "1 portion",\n'
                '      "energy_kcal_min": 100,\n'
                '      "energy_kcal_max": 120,\n'
                '      "energy_kcal_best": 110\n'
                "    }\n"
                "  ],\n"
                '  "changed_fields": [],\n'
                '  "summary_date": null,\n'
                '  "assumptions": [],\n'
                '  "warnings": []\n'
                "}\n"
                "```\n"
                "At least one total energy field (`energy_kcal_min|max|best`) is required "
                "only for `meal_observation`. "
                "Enforce ordering constraints whenever values are present: "
                "`energy_kcal_min <= energy_kcal_max`, `energy_kcal_min <= "
                "energy_kcal_best`, and `energy_kcal_best <= energy_kcal_max`; "
                "non-finite and negative values "
                "are invalid. Keep `items` flat, not nested. `assumptions` and `warnings` "
                "must be bounded short strings."
            ),
            (
                "`record_type` is one of `meal_observation`, `meal_correction`, "
                "`meal_deletion`, `day_summary`. Use `meal_observation` for a new possible "
                'consumption event. When the user CORRECTS an earlier meal ("that was '
                'breakfast on 1 August", "it was 300 kcal, not 500"), emit '
                "`meal_correction`: list ONLY the fields being changed in `changed_fields` "
                "and provide replacement values just for those fields — a correction is "
                "never another meal, and a date-only correction does not repeat the calorie "
                "estimate. When the user says a logged meal must not count, emit "
                "`meal_deletion` with no nutrient values. A daily report you calculate "
                "from already-recorded meals is `day_summary` with totals plus "
                "`summary_date` — it is a non-countable summary, NEVER a new meal."
            ),
            (
                "`meal_date` is an ISO calendar date (`YYYY-MM-DD`) for date-only language: "
                '"breakfast on 1 August" sets `meal_date=2026-08-01` and keeps '
                "`meal_at=null`. `meal_at` is optional and may be emitted only when the "
                "user explicitly states the meal or consumption time precisely enough. "
                "Never infer or copy either field from a forwarded source timestamp, "
                "receive timestamp, image metadata, or a model guess. For an image without "
                "explicit consumption language, keep `meal_date=null`, `meal_at=null` and "
                "`consumption_status=unknown`."
            ),
            (
                "Set `explicit_new_consumption` to true ONLY when the user explicitly "
                "states that the same food or an already-sent photo is a new, separate "
                'consumption ("I ate the same thing again today"). In every other case '
                "keep it false: a resent or reused photo without that explicit statement "
                "is a duplicate, not another meal."
            ),
            (
                "Never emit identity or provenance fields (`meal_id`, `source_message_id`, "
                "`reply_to_source_message_id`, `attachment_fingerprints`, tenant or session "
                "ids) — the trusted gateway attaches them and rejects model-authored values."
            ),
        ]
    )

    if include_ohmo_memory and (ohmo_memory := load_ohmo_memory_prompt(root)):
        sections.append(ohmo_memory)

    if include_project_memory:
        project_memory = load_project_memory_prompt(cwd)
        if project_memory:
            sections.append(project_memory)

    return "\n\n".join(section for section in sections if section and section.strip())


def _build_ohmo_workspace_sections(root: str | Path) -> tuple[str, ...]:
    return (
        "# ohmo Workspace",
        f"- Personal workspace root: {root}",
        "- Personal memory and sessions live under the shared ohmo workspace root.",
        (
            "- When a needed fact is not visible in the injected memory index, use "
            "the memory tool's `search` action for semantic recall."
        ),
        "- Resume only within ohmo sessions; do not assume interoperability with plain OpenHarness sessions.",
    )
