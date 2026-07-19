"""Content-free observability for tenant-scoped memory operations."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from threading import Lock
from typing import Literal


logger = logging.getLogger(__name__)

AuditEvent = Literal["memory_gate", "catalog_op", "honcho_request"]

_OPAQUE_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]+")
_COUNTERS: Counter[tuple[str, str, str]] = Counter()
_COUNTER_LOCK = Lock()


def memory_audit_event(
    event: AuditEvent,
    *,
    tenant_id: str | None = None,
    requester_tenant: str | None = None,
    outcome: str | None = None,
    failing_conjuncts: tuple[str, ...] = (),
    operation: str | None = None,
    workspace: str | None = None,
    cross_scope_denial: bool = False,
) -> dict[str, object]:
    """Emit one structured event containing opaque routing metadata only.

    The deliberately narrow signature makes it impossible for callers to attach
    titles, queries, slugs, message text, or memory content accidentally.
    """
    scope = _identifier(tenant_id, "tenant_id") if tenant_id is not None else None
    requester = (
        _identifier(requester_tenant, "requester_tenant")
        if requester_tenant is not None
        else None
    )
    workspace_name = (
        _identifier(workspace, "workspace") if workspace is not None else None
    )
    result = _identifier(outcome, "outcome") if outcome is not None else None
    op = _identifier(operation, "operation") if operation is not None else None
    conjuncts = tuple(
        _identifier(conjunct, "failing_conjunct") for conjunct in failing_conjuncts
    )

    counter_scope = scope or requester or workspace_name or "unknown"
    counter_dimension = result or op or "total"
    with _COUNTER_LOCK:
        _COUNTERS[(event, counter_scope, counter_dimension)] += 1
        count = _COUNTERS[(event, counter_scope, counter_dimension)]
        cross_scope_count = None
        if cross_scope_denial:
            _COUNTERS[("cross_scope_denial", requester or counter_scope, "deny")] += 1
            cross_scope_count = _COUNTERS[
                ("cross_scope_denial", requester or counter_scope, "deny")
            ]

    payload: dict[str, object] = {"event": event, "count": count}
    if scope is not None:
        payload["tenant_id"] = scope
    if requester is not None:
        payload["requester_tenant"] = requester
    if result is not None:
        payload["outcome"] = result
    if conjuncts:
        payload["failing_conjuncts"] = list(conjuncts)
    if op is not None:
        payload["operation"] = op
    if workspace_name is not None:
        payload["workspace"] = workspace_name
    if cross_scope_count is not None:
        payload["cross_scope_denial_count"] = cross_scope_count

    logger.info(
        "memory_audit %s",
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        extra={"memory_audit": payload},
    )
    return payload


def get_memory_audit_counters() -> dict[str, int]:
    """Return a deterministic snapshot of the in-process audit counters."""
    with _COUNTER_LOCK:
        return {
            ":".join(key): value
            for key, value in sorted(_COUNTERS.items())
        }


def reset_memory_audit_counters() -> None:
    """Reset counters for process lifecycle boundaries and isolated tests."""
    with _COUNTER_LOCK:
        _COUNTERS.clear()


def _identifier(value: str, description: str) -> str:
    clean = (value or "").strip()
    if not clean or _OPAQUE_IDENTIFIER.fullmatch(clean) is None:
        raise ValueError(f"{description} must be an opaque identifier")
    return clean


__all__ = [
    "get_memory_audit_counters",
    "memory_audit_event",
    "reset_memory_audit_counters",
]
