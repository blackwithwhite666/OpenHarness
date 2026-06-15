"""Personal memory helpers for ``.ohmo``."""

from __future__ import annotations

from pathlib import Path

from openharness.commands import MemoryCommandBackend

from ohmo.memory_store import MemoryStore
from ohmo.threat_patterns import scan_for_threats
from ohmo.workspace import get_memory_dir, get_memory_index_path


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


def load_memory_prompt(workspace: str | Path | None = None, *, max_files: int = 5) -> str | None:
    """Return a prompt section describing personal memory."""
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
    for path in list_memory_files(workspace)[:max_files]:
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
            body = content[:4000]
        lines.extend(["", f"## {path.name}", "```md", body, "```"])

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
