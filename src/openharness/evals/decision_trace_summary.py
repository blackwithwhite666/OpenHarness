"""Metadata-only summaries for decision-trace eval events."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from openharness.evals.decision_trace import (
    DECISION_TRACE_DIAGNOSTIC_EVENT_KINDS,
    DECISION_TRACE_EVENT_KINDS,
    DECISION_TRACE_MODEL_EVENT_KINDS,
    STRUCTURAL_ASSISTANT_FINAL,
    TRACE_FINALIZATION,
    TRACE_MISSING_REQUIRED,
    TRACE_UNCERTAINTY,
)

DECISION_TRACE_SUMMARY_KEYS = (
    "decision_trace_event_count",
    "decision_trace_model_event_count",
    "decision_trace_diagnostic_event_count",
    "decision_trace_missing_required_count",
    "decision_trace_required_count",
    "decision_trace_recorded_count",
    "decision_trace_coverage_status",
    "decision_trace_finalization_count",
    "decision_trace_claim_count",
    "decision_trace_supported_claim_count",
    "decision_trace_evidence_linked_finalization_count",
    "decision_trace_evidence_status",
    "unsupported_claim_count",
    "uncertainty_trace_count",
    "uncertainty_status",
    "decision_trace_sensitivity_labels",
    "decision_trace_max_sensitivity",
)

_SENSITIVITY_RANK = {
    "public": 0,
    "personal": 1,
    "private": 2,
    "secret": 3,
}
_UNSUPPORTED_CLAIM_COUNT_KEYS = (
    "unsupported_claim_count",
    "unsupported_claims_count",
)


def summarize_decision_trace(events: Sequence[Any]) -> dict[str, Any]:
    """Return JSON-serializable, metadata-only decision-trace summary fields."""
    event_count = 0
    model_event_count = 0
    diagnostic_event_count = 0
    missing_required_count = 0
    required_count = 0
    recorded_count = 0
    finalization_count = 0
    claim_count = 0
    supported_claim_count = 0
    evidence_linked_finalization_count = 0
    unsupported_claim_count = 0
    uncertainty_trace_count = 0
    sensitivity_labels: set[str] = set()

    for event in events:
        kind = _string_value(getattr(event, "kind", None))
        payload = _mapping(getattr(event, "payload", None))

        if kind in DECISION_TRACE_MODEL_EVENT_KINDS:
            event_count += 1
            model_event_count += 1
        elif kind in DECISION_TRACE_DIAGNOSTIC_EVENT_KINDS:
            event_count += 1
            diagnostic_event_count += 1

        if kind == TRACE_FINALIZATION:
            finalization_count += 1
            claims, supported = _claim_evidence_counts(payload)
            claim_count += claims
            supported_claim_count += supported
            if supported > 0:
                evidence_linked_finalization_count += 1

        if kind == TRACE_MISSING_REQUIRED:
            missing_required_count += 1
        if kind == STRUCTURAL_ASSISTANT_FINAL:
            if payload.get("trace_required") is True:
                required_count += 1
            if payload.get("model_trace_recorded") is True:
                recorded_count += 1
        if kind in DECISION_TRACE_EVENT_KINDS:
            unsupported_claim_count += unsupported_claim_count_from_payload(payload)
            if kind == TRACE_UNCERTAINTY:
                uncertainty_trace_count += 1
            sensitivity = payload.get("sensitivity")
            if isinstance(sensitivity, str) and sensitivity in _SENSITIVITY_RANK:
                sensitivity_labels.add(sensitivity)

    sorted_sensitivity_labels = sorted(
        sensitivity_labels,
        key=lambda label: (_SENSITIVITY_RANK[label], label),
    )
    return {
        "decision_trace_event_count": event_count,
        "decision_trace_model_event_count": model_event_count,
        "decision_trace_diagnostic_event_count": diagnostic_event_count,
        "decision_trace_missing_required_count": missing_required_count,
        "decision_trace_required_count": required_count,
        "decision_trace_recorded_count": recorded_count,
        "decision_trace_coverage_status": _coverage_status(
            required_count=required_count,
            recorded_count=recorded_count,
            missing_required_count=missing_required_count,
        ),
        "decision_trace_finalization_count": finalization_count,
        "decision_trace_claim_count": claim_count,
        "decision_trace_supported_claim_count": supported_claim_count,
        "decision_trace_evidence_linked_finalization_count": (
            evidence_linked_finalization_count
        ),
        "decision_trace_evidence_status": _evidence_status(
            finalization_count=finalization_count,
            evidence_linked_finalization_count=evidence_linked_finalization_count,
        ),
        "unsupported_claim_count": unsupported_claim_count,
        "uncertainty_trace_count": uncertainty_trace_count,
        "uncertainty_status": "present" if uncertainty_trace_count else "absent",
        "decision_trace_sensitivity_labels": sorted_sensitivity_labels,
        "decision_trace_max_sensitivity": (
            sorted_sensitivity_labels[-1] if sorted_sensitivity_labels else ""
        ),
    }


def copy_decision_trace_summary_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only known decision-trace summary fields from a metadata mapping."""
    return {key: metadata[key] for key in DECISION_TRACE_SUMMARY_KEYS if key in metadata}


def unsupported_claim_count_from_payload(payload: Mapping[str, Any]) -> int:
    """Derive a count from safe numeric/list metadata fields only."""
    counts: list[int] = []
    for key in _UNSUPPORTED_CLAIM_COUNT_KEYS:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            counts.append(max(0, int(value)))
    unsupported_claims = payload.get("unsupported_claims")
    if isinstance(unsupported_claims, list):
        counts.append(len(unsupported_claims))
    return max(counts, default=0)


_EVIDENCE_LINK_KEYS = ("supported_by", "evidence_id", "evidence_ids")


def _claim_evidence_counts(payload: Mapping[str, Any]) -> tuple[int, int]:
    """Return (total claims, claims linked to at least one evidence id)."""
    claims = payload.get("answer_claims")
    if not isinstance(claims, list):
        return (0, 0)
    total = 0
    supported = 0
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        total += 1
        if _claim_has_evidence(claim):
            supported += 1
    return (total, supported)


def _claim_has_evidence(claim: Mapping[str, Any]) -> bool:
    for key in _EVIDENCE_LINK_KEYS:
        value = claim.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, tuple)) and any(
            isinstance(item, str) and item.strip() for item in value
        ):
            return True
    return False


def _evidence_status(
    *,
    finalization_count: int,
    evidence_linked_finalization_count: int,
) -> str:
    """Distinguish 'has a trace' from 'has an evidence-linked trace'.

    coverage_status can be ``complete`` while evidence_status is ``thin`` — that
    gap is the signal that the repair path is emitting reference-only finalizations
    instead of grounded answer_claims (D5).
    """
    if finalization_count == 0:
        return "none"
    if evidence_linked_finalization_count > 0:
        return "linked"
    return "thin"


def _coverage_status(
    *,
    required_count: int,
    recorded_count: int,
    missing_required_count: int,
) -> str:
    if required_count == 0:
        return "not_required"
    if recorded_count >= required_count and missing_required_count == 0:
        return "complete"
    if recorded_count > 0:
        return "partial"
    return "missing"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None
