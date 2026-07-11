"""Grounding-failure attribution helpers for faithful session reports."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from openharness.api.client import SupportsStreamingMessages
from openharness.evals.executor import _run_eval_coroutine
from openharness.evals.judge import _default_grounding_search, _verify_grounding
from openharness.evals.meta_judge import MetaJudgeAttributor, summarize_attributions
from openharness.evals.models import EvalSessionReport
from openharness.evals.grounding_trace import read_session_trace, trace_text_fields

MAX_REFUTED_FACT_CLAIMS = 6
MAX_CLAIM_CHARS = 240
MAX_EVIDENCE_CHARS = 240

_BOUNDARY_STATUSES = {
    "sandbox_blocked",
    "retrieval_error",
    "retrieval_empty",
    "search_error",
    "no_retrieval_results",
}
_RETRIEVAL_EMPTY_MARKERS = (
    "search error",
    "retrieval error",
    "returned nothing",
    "no relevant results",
    "no results",
    "0 results",
)


def grounding_report_metadata(grounding: object) -> dict[str, object]:
    """Return bounded report-case metadata extracted from a grounding result."""
    if not isinstance(grounding, dict):
        grounding = {}
    return {
        "grounding_status": str(grounding.get("status") or ""),
        "grounding_score": grounding.get("score"),
        "grounding_refuted_fact_claims": _refuted_fact_claims(grounding),
    }


def read_faithful_session_report(path: str | Path) -> EvalSessionReport:
    return EvalSessionReport.model_validate_json(
        Path(path).expanduser().read_text(encoding="utf-8")
    )


def attribute_faithful_grounding_report(
    report: EvalSessionReport,
    *,
    attributor: MetaJudgeAttributor | None,
    api_client: SupportsStreamingMessages | None = None,
    model: str = "",
    trace_root: Path | None = None,
    search: Callable[..., Awaitable[str]] = _default_grounding_search,
) -> dict[str, object]:
    """Attribute failed faithful grounding checks to model or harness causes."""
    attributions: list[dict[str, object]] = []
    for case in report.cases:
        if case.checks.get("grounding_ok") is not False:
            continue
        detail = _case_grounding_detail(case.metadata)
        trace = read_session_trace(trace_root, case.session_id) if trace_root else {}
        if not detail["has_detail"]:
            if api_client is None:
                raise ValueError(
                    f"grounding detail missing for {case.session_id}; "
                    "provide api_client/model and trace data to re-derive it"
                )
            detail = _derive_grounding_detail(
                api_client=api_client,
                model=model,
                trace=trace,
                metadata=case.metadata,
                search=search,
                session_id=case.session_id,
            )

        shortcut = _boundary_shortcut(detail)
        if shortcut is not None:
            attribution = shortcut
        else:
            if attributor is None:
                raise ValueError(
                    "meta-judge attributor is required for non-boundary grounding failures"
                )
            fields = trace_text_fields(trace, case.metadata)
            attribution = attributor.attribute(
                task=fields["task"],
                rubric={
                    "grounding": "Final answer factual claims must be supported by retrieval evidence."
                },
                answer=fields["answer"],
                trajectory=_grounding_trajectory(detail),
                aspect_scores={
                    "grounding": detail.get("score"),
                    "status": detail.get("status"),
                },
                gate_failures=["grounding_ok"],
            )

        item = {
            "session_id": case.session_id,
            "grounding_status": detail.get("status") or "",
            "grounding_score": detail.get("score"),
            "gate_failures": ["grounding_ok"],
        }
        item.update(attribution)
        attributions.append(item)

    summary = summarize_attributions(attributions)
    summary["grounding_harness_debt_pct"] = summary["harness_debt_pct"]
    return {
        "report_id": report.report_id,
        "lane": "faithful",
        "check": "grounding",
        "failed_count": len(attributions),
        "attributions": attributions,
        "summary": summary,
    }


def default_grounding_blame_output_path(report_path: str | Path) -> Path:
    path = Path(report_path).expanduser()
    return path.with_name(f"{path.stem}.grounding_blame.json")


def _refuted_fact_claims(grounding: dict[str, object]) -> list[dict[str, str]]:
    claims = grounding.get("claims")
    if not isinstance(claims, list):
        return []
    out: list[dict[str, str]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if str(claim.get("kind") or "fact") != "fact":
            continue
        if claim.get("verdict") != "refuted":
            continue
        out.append(
            {
                "claim": _truncate_right(str(claim.get("claim") or ""), MAX_CLAIM_CHARS),
                "evidence": _truncate_right(
                    str(claim.get("evidence") or ""), MAX_EVIDENCE_CHARS
                ),
            }
        )
        if len(out) >= MAX_REFUTED_FACT_CLAIMS:
            break
    return out


def _case_grounding_detail(metadata: dict[str, Any]) -> dict[str, object]:
    claims = metadata.get("grounding_refuted_fact_claims")
    if not isinstance(claims, list):
        claims = []
    has_detail = any(
        key in metadata
        for key in ("grounding_status", "grounding_score", "grounding_refuted_fact_claims")
    )
    return {
        "has_detail": has_detail,
        "status": str(metadata.get("grounding_status") or ""),
        "score": metadata.get("grounding_score"),
        "refuted_fact_claims": [item for item in claims if isinstance(item, dict)][
            :MAX_REFUTED_FACT_CLAIMS
        ],
    }


def _derive_grounding_detail(
    *,
    api_client: SupportsStreamingMessages,
    model: str,
    trace: dict[str, object],
    metadata: dict[str, Any],
    search: Callable[..., Awaitable[str]],
    session_id: str,
) -> dict[str, object]:
    fields = trace_text_fields(trace, metadata)
    if not fields["answer"] or not fields["task"]:
        raise ValueError(
            f"grounding detail missing for {session_id} and trace lacks final_text/intent"
        )
    grounding = _run_eval_coroutine(
        _verify_grounding(
            api_client,
            model,
            task=fields["task"],
            answer=fields["answer"],
            trajectory=fields["answer"],
            checklist_items=[fields["task"]],
            search=search,
        )
    )
    detail = _case_grounding_detail(grounding_report_metadata(grounding))
    detail["has_detail"] = True
    return detail


def _boundary_shortcut(detail: dict[str, object]) -> dict[str, object] | None:
    status = str(detail.get("status") or "")
    if status in _BOUNDARY_STATUSES:
        return {
            "blame": "harness_boundary",
            "subtype": status,
            "confidence": 1.0,
            "evidence": f"grounding verifier status is {status}.",
            "votes": 0,
        }
    claims = detail.get("refuted_fact_claims")
    if isinstance(claims, list) and claims:
        evidence_texts = [
            str(item.get("evidence") or "") for item in claims if isinstance(item, dict)
        ]
        if evidence_texts and all(
            _evidence_is_retrieval_empty(text) for text in evidence_texts
        ):
            return {
                "blame": "harness_boundary",
                "subtype": "retrieval_empty",
                "confidence": 1.0,
                "evidence": "grounding evidence indicates retrieval errored or returned nothing.",
                "votes": 0,
            }
    return None


def _grounding_trajectory(detail: dict[str, object]) -> list[dict[str, object]]:
    claims = detail.get("refuted_fact_claims")
    if not isinstance(claims, list):
        return []
    trajectory: list[dict[str, object]] = []
    for item in claims[:MAX_REFUTED_FACT_CLAIMS]:
        if not isinstance(item, dict):
            continue
        evidence = str(item.get("evidence") or "")
        trajectory.append(
            {
                "claim": str(item.get("claim") or ""),
                "kind": "fact",
                "verdict": "refuted",
                "evidence": evidence,
                "is_error": _evidence_is_retrieval_empty(evidence),
            }
        )
    return trajectory


def _evidence_is_retrieval_empty(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _RETRIEVAL_EMPTY_MARKERS)


def _truncate_right(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "...[truncated]"
    if limit <= len(marker):
        return text[:limit]
    return f"{text[:limit - len(marker)]}{marker}"
