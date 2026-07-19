"""Fail-closed confidentiality gate for model-facing memory recall."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from ohmo.gateway.turn_context import TurnContext

_OWNER_CONJUNCT = "is_canonical_owner"
_PRIVATE_CONJUNCT = "is_trusted_private_chat"
_ISOLATED_CONJUNCT = "principal_isolated_session"


@dataclass(frozen=True)
class GateDecision:
    """Result of evaluating every confidentiality-gate conjunct."""

    allowed: bool
    reasons: tuple[str, ...]


def memory_engaged(
    owner_principals: Collection[str],
    gate_decision: GateDecision,
) -> bool:
    """Return whether authoritative memory is available for this turn.

    An empty owner list is the legacy single-user configuration, where memory
    remains unconditionally available. Once owners are configured, every
    model-facing memory surface requires a green confidentiality gate.
    """
    return not owner_principals or gate_decision.allowed


def evaluate_memory_gate(
    turn_ctx: TurnContext | None,
    *,
    principal_isolated: bool | None,
) -> GateDecision:
    """Allow recall only when every positively attested conjunct is true.

    Exact ``True`` checks make malformed, absent, or unknown inputs deny recall
    rather than accidentally becoming truthy.
    """
    conjuncts = (
        (_OWNER_CONJUNCT, getattr(turn_ctx, "is_owner", False) is True),
        (_PRIVATE_CONJUNCT, getattr(turn_ctx, "is_private", False) is True),
        (_ISOLATED_CONJUNCT, principal_isolated is True),
    )
    reasons = tuple(name for name, passed in conjuncts if not passed)
    return GateDecision(allowed=not reasons, reasons=reasons)


def principal_isolated_session(
    turn_ctx: TurnContext | None,
    session_owner_principal: str | None,
) -> bool:
    """Return whether the session is positively bound to this exact principal."""
    if turn_ctx is None or not isinstance(session_owner_principal, str):
        return False
    return bool(
        turn_ctx.principal
        and session_owner_principal
        and session_owner_principal == turn_ctx.principal
    )
