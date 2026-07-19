"""Per-turn assembly seam for ohmo's standing memory prompt."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ohmo.gateway.memory_gate import GateDecision, evaluate_memory_gate, memory_engaged
from ohmo.memory_backend import MemoryBackend, ShadowMemoryBackend, _inject_char_budget
from ohmo.prompts import _build_ohmo_workspace_sections

if TYPE_CHECKING:
    from ohmo.gateway.turn_context import TurnContext

_MEMORY_HEADING = "# ohmo Memory"
_WORKSPACE_HEADING = "# ohmo Workspace"
_MEMORY_DIRECTORY_PREFIX = "- Personal memory directory: "
_REMINDERS_SECTION = "# Reminders"
_DERIVED_RECALL_TIMEOUT_SECONDS = 0.5


class TurnSnapshot(str):
    """Rendered memory block plus the decision governing future recall.

    This remains a ``str`` so the current prompt-injection path stays byte-for-
    byte compatible. Callers can inspect ``gate_decision`` without parsing
    prompt text.
    """

    gate_decision: GateDecision
    memory_engaged: bool

    def __new__(
        cls,
        memory_block: str,
        *,
        gate_decision: GateDecision,
        memory_engaged: bool,
    ) -> TurnSnapshot:
        snapshot = super().__new__(cls, memory_block)
        snapshot.gate_decision = gate_decision
        snapshot.memory_engaged = memory_engaged
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
    principal_isolated: bool | None = None,
    visible_recall: bool = False,
    latest_user_prompt: str | None = None,
    derived_recall_timeout: float = _DERIVED_RECALL_TIMEOUT_SECONDS,
    owner_principals: tuple[str, ...] = (),
    memory_engaged_override: bool | None = None,
    derived_backend: ShadowMemoryBackend | None = None,
) -> TurnSnapshot:
    """Read a fresh backend-rendered memory snapshot for one submitted turn.

    With no configured owners, the authoritative catalog block is rendered as
    before. Once owners are configured, it is rendered only after a green
    confidentiality gate. Honcho derived recall additionally requires explicit
    opt-in. Any Honcho failure leaves the catalog-only block byte-for-byte
    unchanged.
    """
    gate_decision = evaluate_memory_gate(
        turn_ctx,
        principal_isolated=principal_isolated,
    )
    engaged = (
        memory_engaged(owner_principals, gate_decision)
        if memory_engaged_override is None
        else memory_engaged_override
    )
    memory_block = await backend.render_prompt(budget) if engaged else ""
    recall_backend = derived_backend
    if recall_backend is None and isinstance(backend, ShadowMemoryBackend):
        recall_backend = backend
    if (
        visible_recall is True
        and gate_decision.allowed
        and recall_backend is not None
        and isinstance(latest_user_prompt, str)
    ):
        separator = "\n\n" if memory_block.strip() else ""
        composite_budget = budget if budget is not None else _inject_char_budget()
        derived_budget = composite_budget - len(memory_block) - len(separator)
        derived_block = await recall_backend.derived_recall_block(
            latest_user_prompt,
            budget=derived_budget,
            timeout=derived_recall_timeout,
        )
        if derived_block is not None:
            memory_block = f"{memory_block}{separator}{derived_block}"
    return TurnSnapshot(
        memory_block,
        gate_decision=gate_decision,
        memory_engaged=engaged,
    )


def compose_runtime_prompt(
    memory_free_base: str,
    snapshot: str,
    *,
    memory_engaged: bool = True,
) -> str:
    """Compose one snapshot into a memory-free ohmo persona.

    The file renderer carries the memory directory in its scaffold. That lets
    this function restore the legacy workspace section at its original position
    while keeping the stored base free of both memory and workspace paths. An
    already-composed input is returned unchanged rather than duplicating the
    standing memory block.
    """
    if not memory_engaged:
        return _strip_memory_surfaces(memory_free_base)
    if _has_memory_block(memory_free_base):
        return memory_free_base
    if not snapshot or not snapshot.strip():
        return memory_free_base

    base = memory_free_base
    if not _has_workspace_section(base) and (
        workspace_root := _workspace_root_from_snapshot(snapshot)
    ):
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


def _has_workspace_section(prompt: str) -> bool:
    return any(line == _WORKSPACE_HEADING for line in prompt.splitlines())


def _strip_memory_surfaces(prompt: str) -> str:
    """Remove authoritative memory and its workspace instructions from a base."""
    workspace_marker = f"\n\n{_WORKSPACE_HEADING}\n"
    reminders_marker = f"\n\n{_REMINDERS_SECTION}\n"
    if workspace_marker in prompt:
        before, after_workspace = prompt.split(workspace_marker, 1)
        if reminders_marker in f"\n\n{after_workspace}":
            _, after = f"\n\n{after_workspace}".split(reminders_marker, 1)
            prompt = f"{before}{reminders_marker}{after}"

    memory_marker = f"\n\n{_MEMORY_HEADING}\n"
    if memory_marker in prompt:
        prompt = prompt.split(memory_marker, 1)[0]
    return prompt


def _workspace_root_from_snapshot(snapshot: str) -> Path | None:
    for line in snapshot.splitlines():
        if line.startswith(_MEMORY_DIRECTORY_PREFIX):
            memory_dir = line.removeprefix(_MEMORY_DIRECTORY_PREFIX)
            return Path(memory_dir).parent
    return None
