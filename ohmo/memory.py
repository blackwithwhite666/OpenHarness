"""Personal memory helpers for ``.ohmo``."""

from __future__ import annotations

import os
from pathlib import Path

from openharness.commands import MemoryCommandBackend

from ohmo.memory_store import MemoryStore
from ohmo.threat_patterns import scan_for_threats
from ohmo.workspace import get_memory_dir, get_memory_index_path

# Always-on memory injection budget: inject entry bodies until their cumulative
# size exceeds this, so a small corpus is fully in-context (every rule visible)
# and a large one is capped — overflow stays in the index, read on demand via the
# memory tool. Env-tunable. Replaces the old fixed first-5-entries cap, which left
# alphabetically-late entries (e.g. weather-rules) un-injected so the agent never
# applied them unless it happened to read the file.
DEFAULT_MEMORY_INJECT_CHARS = 12000
_MEMORY_ENTRY_RENDER_CHARS = 4000  # per-entry body truncation in the prompt


def _inject_char_budget() -> int:
    raw = os.environ.get("OHMO_MEMORY_INJECT_CHARS")
    if raw is None:
        return DEFAULT_MEMORY_INJECT_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MEMORY_INJECT_CHARS
    return value if value > 0 else DEFAULT_MEMORY_INJECT_CHARS


def list_memory_files(workspace: str | Path | None = None) -> list[Path]:
    """List ``.ohmo`` memory markdown files (excludes the index + symlinks)."""
    return MemoryStore(workspace).entry_paths()


def add_memory_entry(workspace: str | Path | None, title: str, content: str) -> Path:
    """Create/overwrite a personal memory file and upsert it in ``MEMORY.md``.

    Routed through :class:`MemoryStore` so the ``/memory`` slash command + CLI
    share its discipline: unicode-safe slug, per-line index upsert (no naive
    substring dedup), and no clobbering of the reserved ``MEMORY.md`` index.
    """
    return MemoryStore(workspace).add_legacy(title, content)


def remove_memory_entry(workspace: str | Path | None, name: str) -> bool:
    """Delete a memory file and remove its index entry."""
    return MemoryStore(workspace).remove(name).ok


def load_memory_prompt(
    workspace: str | Path | None = None,
    *,
    max_files: int | None = None,
    max_chars: int | None = None,
) -> str | None:
    """Return a prompt section describing personal memory.

    Entry bodies are injected until their cumulative size exceeds a char budget
    (``max_chars`` or ``OHMO_MEMORY_INJECT_CHARS``, default 12000) — so a small
    corpus is injected whole (every rule always in-context) and a large one is
    capped (overflow stays listed in the index, read on demand via the memory
    tool). ``max_files`` optionally also caps the count (default: no count cap).
    """
    memory_dir = get_memory_dir(workspace)
    index_path = get_memory_index_path(workspace)
    lines = [
        "# ohmo Memory",
        f"- Personal memory directory: {memory_dir}",
        "- Use this memory for stable user preferences and durable personal context.",
        "- Curate it with the `memory` tool (add/update/remove/list/get) — do NOT write "
        'memory files by hand. Save DECLARATIVE facts ("User prefers UTC"), not '
        "self-instructions; skip transient progress, raw data dumps (paths/listings), and secrets.",
    ]

    if index_path.exists():
        # Render scope is "all" (classic injection + exfil): it is <= every write
        # scope (model tool=strict, /memory=all), so an accepted entry always
        # renders (no accepted-but-hidden), while persona-style phrasings that
        # only trip the broader "strict"/"context" sets are not falsely blanked.
        index_lines = [
            "[BLOCKED: index line contained a threat pattern]"
            if scan_for_threats(ln, scope="all")
            else ln
            for ln in index_path.read_text(encoding="utf-8").splitlines()[:200]
        ]
        lines.extend(["", "## MEMORY.md", "```md", *index_lines, "```"])

    # Snapshot-time defense-in-depth: a memory entry written on disk by a
    # compromised tool / sister session (bypassing the write-time scan) must not
    # be injected verbatim. Replace a tripped entry with a placeholder; the
    # on-disk file is left intact so the agent can read/remove it via the tool.
    budget = max_chars if max_chars is not None else _inject_char_budget()
    # Rank by access frequency (then name, for stability): the facts the agent
    # actually pulls (recorded on `memory action='get'`) are injected ahead of cold
    # ones, so when the corpus overflows the char budget the important entries stay
    # in-context and the unused tail drops to the index. A small corpus fits whole,
    # so the order is invisible; it only matters once memory exceeds the budget.
    usage = MemoryStore(workspace).usage()
    entries = sorted(
        list_memory_files(workspace),
        key=lambda p: (-usage.get(p.name, 0), p.name),
    )
    used = 0
    shown = 0
    for index, path in enumerate(entries):
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if not content:
            continue
        findings = scan_for_threats(content, scope="all")
        if findings:
            body = (
                f"[BLOCKED: {path.name} contained threat pattern(s): "
                f"{', '.join(findings)}. Removed from the prompt; use the memory tool "
                f"(action='get'/'remove') to inspect or delete it.]"
            )
        else:
            body = content[:_MEMORY_ENTRY_RENDER_CHARS]
        # Stop once the char budget or the optional count cap is hit — but always
        # inject at least one body. Remaining entries stay listed in the index.
        over_budget = shown > 0 and used + len(body) > budget
        over_count = max_files is not None and shown >= max_files
        if over_budget or over_count:
            remaining = sum(
                1
                for p in entries[index:]
                if p.read_text(encoding="utf-8", errors="replace").strip()
            )
            if remaining:
                lines.append("")
                lines.append(
                    f"_({remaining} more memory entr{'y' if remaining == 1 else 'ies'} in the index "
                    f"above — read one with memory(action='get', name='<name>'))._"
                )
            break
        lines.extend(["", f"## {path.name}", "```md", body, "```"])
        used += len(body)
        shown += 1

    return "\n".join(lines)


def create_memory_command_backend(workspace: str | Path | None = None) -> MemoryCommandBackend:
    """Return a ``/memory`` backend bound to ohmo's personal memory store."""

    return MemoryCommandBackend(
        label="ohmo personal memory",
        get_memory_dir=lambda: get_memory_dir(workspace),
        get_entrypoint=lambda: get_memory_index_path(workspace),
        list_files=lambda: list_memory_files(workspace),
        add_entry=lambda title, content: add_memory_entry(workspace, title, content),
        remove_entry=lambda name: remove_memory_entry(workspace, name),
    )
