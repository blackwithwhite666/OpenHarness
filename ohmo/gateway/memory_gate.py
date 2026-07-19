"""Fail-closed confidentiality gate for model-facing memory recall."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.turn_context import TurnContext, canonical_principal
from ohmo.memory_audit import memory_audit_event

_OWNER_CONJUNCT = "is_canonical_owner"
_PRIVATE_CONJUNCT = "is_trusted_private_chat"
_ISOLATED_CONJUNCT = "principal_isolated_session"
_TURN_CONTEXT_CONJUNCT = "turn_context_present"
_PRINCIPAL_CONJUNCT = "canonical_principal_present"
_RECOGNIZED_CONJUNCT = "principal_has_memory_tenant"
_ENABLED_CONJUNCT = "tenant_memory_enabled"


@dataclass(frozen=True)
class GateDecision:
    """Result of evaluating every confidentiality-gate conjunct."""

    allowed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MemoryScope:
    """Private and shared memory tenants authorized for one turn."""

    private_tenant: str
    shared_tenants: tuple[str, ...]


def resolve_memory_scope(
    cfg: GatewayConfig,
    turn_ctx: TurnContext | None,
    *,
    principal_isolated: bool | None,
) -> MemoryScope | None:
    """Resolve the turn's authorized memory audience, denying by default."""
    if turn_ctx is None:
        _audit_scope_denial("unknown", (_TURN_CONTEXT_CONJUNCT,))
        return None

    principal = canonical_principal(turn_ctx.channel, turn_ctx.principal)
    if not principal:
        _audit_scope_denial("unknown", (_PRINCIPAL_CONJUNCT,))
        return None
    owner_principals = {
        canonical_principal(turn_ctx.channel, owner) for owner in cfg.owner_principals
    }
    tenant: str | None
    if principal in owner_principals:
        tenant = "owner"
    else:
        tenant = cfg.family_principals.get(principal)

    if tenant is None:
        _audit_scope_denial("unrecognized", (_RECOGNIZED_CONJUNCT,))
        return None

    failing_conjuncts = []
    if turn_ctx.is_private is not True:
        failing_conjuncts.append(_PRIVATE_CONJUNCT)
    if principal_isolated is not True:
        failing_conjuncts.append(_ISOLATED_CONJUNCT)

    legacy_owner_mode = not cfg.family_principals and not cfg.enabled_memory_tenants
    if legacy_owner_mode and tenant != "owner":
        failing_conjuncts.append(_OWNER_CONJUNCT)
    elif not legacy_owner_mode and tenant not in cfg.enabled_memory_tenants:
        failing_conjuncts.append(_ENABLED_CONJUNCT)

    if failing_conjuncts:
        _audit_scope_denial(
            tenant,
            tuple(failing_conjuncts),
            cross_scope=tenant != "owner",
        )
        return None

    if legacy_owner_mode:
        scope = MemoryScope(private_tenant="owner", shared_tenants=())
    else:
        scope = MemoryScope(
            private_tenant=tenant,
            shared_tenants=cfg.shared_tenants,
        )
    memory_audit_event(
        "memory_gate",
        tenant_id=tenant,
        requester_tenant=tenant,
        outcome="allow",
    )
    return scope


def _audit_scope_denial(
    requester_tenant: str,
    failing_conjuncts: tuple[str, ...],
    *,
    cross_scope: bool = False,
) -> None:
    memory_audit_event(
        "memory_gate",
        tenant_id=requester_tenant,
        requester_tenant=requester_tenant,
        outcome="deny",
        failing_conjuncts=failing_conjuncts,
        cross_scope_denial=cross_scope,
    )


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
