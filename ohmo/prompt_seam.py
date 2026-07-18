"""Per-turn assembly seam for ohmo's standing memory prompt."""

from __future__ import annotations

from pathlib import Path

from ohmo.memory_backend import MemoryBackend
from ohmo.prompts import _build_ohmo_workspace_sections

_MEMORY_HEADING = "# ohmo Memory"
_MEMORY_DIRECTORY_PREFIX = "- Personal memory directory: "
_REMINDERS_SECTION = "# Reminders"


async def prepare_turn(backend: MemoryBackend, *, budget: int | None = None) -> str:
    """Read a fresh backend-rendered memory snapshot for one submitted turn."""
    return await backend.render_prompt(budget)


def compose_runtime_prompt(memory_free_base: str, snapshot: str) -> str:
    """Compose one snapshot into a memory-free ohmo persona.

    The file renderer carries the memory directory in its scaffold. That lets
    this function restore the legacy workspace section at its original position
    while keeping the stored base free of both memory and workspace paths. An
    already-composed input is returned unchanged rather than duplicating the
    standing memory block.
    """
    if _has_memory_block(memory_free_base):
        return memory_free_base
    if not snapshot or not snapshot.strip():
        return memory_free_base

    base = memory_free_base
    if workspace_root := _workspace_root_from_snapshot(snapshot):
        reminders_marker = f"\n\n{_REMINDERS_SECTION}\n"
        if reminders_marker in base:
            before, after = base.rsplit(reminders_marker, 1)
            workspace_section = "\n\n".join(
                _build_ohmo_workspace_sections(workspace_root)
            )
            base = (
                f"{before}\n\n{workspace_section}"
                f"{reminders_marker}{after}"
            )

    return "\n\n".join(
        section for section in (base, snapshot) if section and section.strip()
    )


def _has_memory_block(prompt: str) -> bool:
    return any(line == _MEMORY_HEADING for line in prompt.splitlines())


def _workspace_root_from_snapshot(snapshot: str) -> Path | None:
    for line in snapshot.splitlines():
        if line.startswith(_MEMORY_DIRECTORY_PREFIX):
            memory_dir = line.removeprefix(_MEMORY_DIRECTORY_PREFIX)
            return Path(memory_dir).parent
    return None
