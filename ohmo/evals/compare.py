"""Ohmo helpers for execution report comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openharness.evals import (
    EvalReportComparisonWrite,
    write_execution_report_comparison,
)

from ohmo.evals.adapter import get_eval_store


@dataclass(frozen=True)
class OhmoEvalCompareResult:
    """Summary returned after comparing two Ohmo execution reports."""

    write: EvalReportComparisonWrite
    report_only: bool


def compare_ohmo_eval_reports(
    *,
    workspace: str | Path | None = None,
    baseline_report: str | Path,
    candidate_report: str | Path = "eval_report.json",
    report_filename: str = "eval_compare.json",
    score_tolerance: float = 0.0,
    report_only: bool = False,
) -> OhmoEvalCompareResult:
    """Compare two execution reports under an Ohmo eval workspace."""
    store = get_eval_store(workspace)
    write = write_execution_report_comparison(
        store,
        baseline_report_path=_resolve_report_path(store.root, baseline_report),
        candidate_report_path=_resolve_report_path(store.root, candidate_report),
        report_filename=report_filename,
        score_tolerance=score_tolerance,
    )
    return OhmoEvalCompareResult(write=write, report_only=report_only)


def _resolve_report_path(store_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = store_root / "reports" / path
    return path.resolve()
