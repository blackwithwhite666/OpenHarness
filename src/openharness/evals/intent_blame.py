"""Intent-failure attribution helpers for faithful session reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openharness.evals.meta_judge import MetaJudgeAttributor, summarize_attributions
from openharness.evals.models import EvalSessionReport


def attribute_faithful_intent_report(
    report: EvalSessionReport,
    *,
    attributor: MetaJudgeAttributor | None,
) -> dict[str, object]:
    """Attribute failed faithful intent checks to model or harness causes."""
    attributions: list[dict[str, object]] = []
    for case in report.cases:
        if case.checks.get("intent_met") is not False:
            continue

        intent = _metadata_text(case.metadata, "grounding_task")
        answer = _metadata_text(case.metadata, "grounding_answer")
        intent_evidence = _metadata_text(case.metadata, "intent_evidence")
        if not intent:
            raise ValueError(
                f"intent attribution requires grounding_task for {case.session_id}"
            )
        if not answer:
            raise ValueError(
                f"intent attribution requires grounding_answer for {case.session_id}"
            )
        if attributor is None:
            raise ValueError("meta-judge attributor is required for intent failures")

        attribution = attributor.attribute(
            task=intent,
            rubric=intent_attribution_rubric(intent),
            answer=answer,
            trajectory=[{"evidence": intent_evidence}],
            aspect_scores={"intent_met": False},
            gate_failures=["intent_met"],
        )
        item: dict[str, object] = {
            "session_id": case.session_id,
            "intent": intent,
        }
        item.update(attribution)
        attributions.append(item)

    summary = summarize_attributions(attributions)
    summary["intent_harness_debt_pct"] = summary["harness_debt_pct"]
    return {
        "report_id": report.report_id,
        "lane": "faithful",
        "check": "intent",
        "failed_count": len(attributions),
        "attributions": attributions,
        "summary": summary,
    }


def intent_attribution_rubric(intent: str) -> dict[str, object]:
    return {
        "task_completion": {
            "text": f"The agent must ACHIEVE this user intent: {intent}"
        }
    }


def default_intent_blame_output_path(report_path: str | Path) -> Path:
    path = Path(report_path).expanduser()
    return path.with_name(f"{path.stem}.intent_blame.json")


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    return "" if value is None else str(value).strip()
