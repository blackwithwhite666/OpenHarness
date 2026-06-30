from __future__ import annotations

from openharness.evals import (
    EvalEvent,
    TRACE_ABSENCE,
    TRACE_FINALIZATION,
    TRACE_MISSING_REQUIRED,
    summarize_decision_trace,
)


def _final(trace_event_id: str, **payload: object) -> EvalEvent:
    base = {"schema_version": 1, "trace_event_id": trace_event_id}
    base.update(payload)
    return EvalEvent(episode_id="ep-1", kind=TRACE_FINALIZATION, payload=base)


def test_evidence_status_none_without_finalization() -> None:
    summary = summarize_decision_trace([])
    assert summary["decision_trace_evidence_status"] == "none"
    assert summary["decision_trace_finalization_count"] == 0
    assert summary["decision_trace_supported_claim_count"] == 0


def test_evidence_status_thin_when_claims_unlinked() -> None:
    # A reference-only finalization (what the old repair path produced): a claim
    # with no supported_by must NOT count as evidence-linked.
    summary = summarize_decision_trace(
        [
            _final(
                "f-1",
                final_answer_summary="did the thing",
                answer_claims=[{"claim": "did the thing"}],
            )
        ]
    )
    assert summary["decision_trace_finalization_count"] == 1
    assert summary["decision_trace_claim_count"] == 1
    assert summary["decision_trace_supported_claim_count"] == 0
    assert summary["decision_trace_evidence_linked_finalization_count"] == 0
    assert summary["decision_trace_evidence_status"] == "thin"


def test_evidence_status_thin_when_summary_only_no_claims() -> None:
    summary = summarize_decision_trace([_final("f-1", final_answer_summary="x")])
    assert summary["decision_trace_evidence_status"] == "thin"
    assert summary["decision_trace_claim_count"] == 0


def test_evidence_status_linked_when_claim_has_supported_by() -> None:
    summary = summarize_decision_trace(
        [
            _final(
                "f-1",
                answer_claims=[
                    {"claim": "closed today", "supported_by": ["toolu_maps_1"]},
                    {"claim": "phone is X", "supported_by": []},
                ],
            )
        ]
    )
    assert summary["decision_trace_claim_count"] == 2
    assert summary["decision_trace_supported_claim_count"] == 1
    assert summary["decision_trace_evidence_linked_finalization_count"] == 1
    assert summary["decision_trace_evidence_status"] == "linked"


def test_absence_event_counts_separately_from_missing_required() -> None:
    # A model-authored trace_absence must not inflate the engine's
    # coverage-failure metric (missing_required_count).
    events = [
        EvalEvent(
            episode_id="ep-1",
            kind=TRACE_ABSENCE,
            payload={"schema_version": 1, "trace_event_id": "abs-1", "needed": "x"},
        ),
        EvalEvent(
            episode_id="ep-1",
            kind=TRACE_MISSING_REQUIRED,
            payload={"schema_version": 1, "trace_event_id": "miss-1"},
        ),
    ]
    summary = summarize_decision_trace(events)
    assert summary["decision_trace_absence_count"] == 1
    assert summary["decision_trace_missing_required_count"] == 1
    # trace_absence is a model event; trace_missing_required is a diagnostic.
    assert summary["decision_trace_model_event_count"] == 1
    assert summary["decision_trace_diagnostic_event_count"] == 1


def test_evidence_link_accepts_evidence_id_string() -> None:
    summary = summarize_decision_trace(
        [_final("f-1", answer_claims=[{"claim": "c", "evidence_id": "E1"}])]
    )
    assert summary["decision_trace_supported_claim_count"] == 1
    assert summary["decision_trace_evidence_status"] == "linked"
