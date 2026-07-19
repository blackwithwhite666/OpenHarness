"""Per-turn assembly seam for ohmo's standing memory prompt."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ohmo.gateway.memory_gate import GateDecision, evaluate_memory_gate
from ohmo.memory_backend import MemoryBackend
from ohmo.prompts import _build_ohmo_workspace_sections

if TYPE_CHECKING:
    from ohmo.gateway.turn_context import TurnContext

_MEMORY_HEADING = "# ohmo Memory"
_MEMORY_DIRECTORY_PREFIX = "- Personal memory directory: "
_REMINDERS_SECTION = "# Reminders"


class TurnSnapshot(str):
    """Rendered memory block plus the decision governing future recall.

    This remains a ``str`` so the current prompt-injection path stays byte-for-
    byte compatible. The later visible-recall step can inspect
    ``gate_decision`` without changing today's rendering behavior.
    """

    gate_decision: GateDecision

    def __new__(
        cls,
        memory_block: str,
        *,
        gate_decision: GateDecision,
    ) -> TurnSnapshot:
        snapshot = super().__new__(cls, memory_block)
        snapshot.gate_decision = gate_decision
        return snapshot

    @property
    def memory_block(self) -> str:
        """Return the unchanged backend-rendered prompt text."""
        return str(self)


async def prepare_turn(
    backend: MemoryBackend,
    *,
    budget: int | None = None,
    turn_ctx: TurnContext | None = None,
    tools_confined: bool | None = None,
    principal_isolated: bool | None = None,
) -> TurnSnapshot:
    """Read a fresh backend-rendered memory snapshot for one submitted turn.

    The gate decision is deliberately metadata only in this step: the backend
    block is still rendered and injected exactly as before.
    """
    gate_decision = evaluate_memory_gate(
        turn_ctx,
        tools_confined=tools_confined,
        principal_isolated=principal_isolated,
    )
    return TurnSnapshot(
        await backend.render_prompt(budget),
        gate_decision=gate_decision,
    )


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
