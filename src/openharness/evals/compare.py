"""Metadata-only comparison for execution eval reports."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from openharness.evals.models import (
    EvalExecutionReport,
    EvalExecutionReportCase,
    EvalReportComparison,
    EvalReportComparisonCase,
)
from openharness.evals.store import EvalStore
from openharness.utils.fs import atomic_write_text

_STATUS_SEVERITY = {
    "passed": 0,
    "failed": 1,
    "blocked": 2,
    "error": 3,
}


@dataclass(frozen=True)
class EvalReportComparisonWrite:
    """Summary returned after writing an execution report comparison."""

    report: EvalReportComparison
    path: Path
    relative_path: str


def read_execution_report(path: str | Path) -> EvalExecutionReport:
    """Read an execution report JSON file."""
    report_path = Path(path).expanduser().resolve()
    try:
        report = EvalExecutionReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
    except ValidationError as exc:
        raise ValueError(f"invalid execution report: {report_path}") from exc
    if report.report_kind != "execution_report":
        raise ValueError(f"report is not an execution_report: {report_path}")
    return report


def compare_execution_reports(
    baseline: EvalExecutionReport,
    candidate: EvalExecutionReport,
    *,
    score_tolerance: float = 0.0,
) -> EvalReportComparison:
    """Compare two execution reports and classify regressions/improvements."""
    if score_tolerance < 0:
        raise ValueError("score_tolerance must be non-negative")
    if baseline.report_kind != "execution_report":
        raise ValueError("baseline report must be an execution_report")
    if candidate.report_kind != "execution_report":
        raise ValueError("candidate report must be an execution_report")

    baseline_by_key = {_case_key(case): case for case in baseline.cases}
    candidate_by_key = {_case_key(case): case for case in candidate.cases}
    duplicate_keys = _duplicate_keys(baseline.cases) + _duplicate_keys(candidate.cases)
    if duplicate_keys:
        raise ValueError(f"duplicate report case ids: {', '.join(sorted(set(duplicate_keys)))}")

    cases: list[EvalReportComparisonCase] = []
    for key in sorted(set(baseline_by_key) | set(candidate_by_key)):
        baseline_case = baseline_by_key.get(key)
        candidate_case = candidate_by_key.get(key)
        cases.append(
            _compare_case(
                key,
                baseline_case,
                candidate_case,
                score_tolerance=score_tolerance,
            )
        )

    unchanged_count = sum(1 for case in cases if case.status == "unchanged")
    improvement_count = sum(1 for case in cases if case.status == "improved")
    regression_count = sum(1 for case in cases if case.status == "regressed")
    added_count = sum(1 for case in cases if case.status == "added")
    removed_count = sum(1 for case in cases if case.status == "removed")
    compared_count = sum(
        1
        for case in cases
        if case.status in {"unchanged", "improved", "regressed"}
    )
    baseline_score = sum(case.score for case in baseline.cases)
    candidate_score = sum(case.score for case in candidate.cases)
    return EvalReportComparison(
        report_id=_stable_id(
            "eval-compare",
            baseline.report_id,
            candidate.report_id,
            f"{score_tolerance:.12f}",
        ),
        baseline_report_id=_safe_ref("report", baseline.report_id),
        candidate_report_id=_safe_ref("report", candidate.report_id),
        baseline_pack_id=_safe_ref("pack", baseline.pack_id),
        candidate_pack_id=_safe_ref("pack", candidate.pack_id),
        case_count=len(cases),
        compared_count=compared_count,
        unchanged_count=unchanged_count,
        improvement_count=improvement_count,
        regression_count=regression_count + removed_count,
        added_count=added_count,
        removed_count=removed_count,
        baseline_passed_count=baseline.passed_count,
        candidate_passed_count=candidate.passed_count,
        baseline_non_passed_count=(
            baseline.failed_count + baseline.blocked_count + baseline.error_count
        ),
        candidate_non_passed_count=(
            candidate.failed_count + candidate.blocked_count + candidate.error_count
        ),
        score_delta=candidate_score - baseline_score,
        cases=cases,
        metadata={
            "privacy": "metadata_only",
            "mode": "execution_report_compare",
            "score_tolerance": score_tolerance,
            "baseline_case_count": baseline.case_count,
            "candidate_case_count": candidate.case_count,
            "removed_counts_as_regression": True,
        },
    )


def write_execution_report_comparison(
    store: EvalStore,
    *,
    baseline_report_path: str | Path,
    candidate_report_path: str | Path,
    report_filename: str = "eval_compare.json",
    score_tolerance: float = 0.0,
) -> EvalReportComparisonWrite:
    """Compare two execution report files and write a comparison report."""
    baseline = read_execution_report(baseline_report_path)
    candidate = read_execution_report(candidate_report_path)
    report = compare_execution_reports(
        baseline,
        candidate,
        score_tolerance=score_tolerance,
    )
    path = _report_output_path(store, report_filename)
    atomic_write_text(path, report.model_dump_json(indent=2) + "\n")
    return EvalReportComparisonWrite(
        report=report,
        path=path,
        relative_path=path.relative_to(store.root).as_posix(),
    )


def _compare_case(
    key: str,
    baseline: EvalExecutionReportCase | None,
    candidate: EvalExecutionReportCase | None,
    *,
    score_tolerance: float,
) -> EvalReportComparisonCase:
    if baseline is None:
        assert candidate is not None
        return EvalReportComparisonCase(
            gold_case_id=candidate.gold_case_id,
            case_id=candidate.case_id,
            status="added",
            candidate_status=candidate.status,
            candidate_score=candidate.score,
            score_delta=candidate.score,
            metadata=_case_metadata(key, baseline, candidate),
        )
    if candidate is None:
        return EvalReportComparisonCase(
            gold_case_id=baseline.gold_case_id,
            case_id=baseline.case_id,
            status="removed",
            baseline_status=baseline.status,
            baseline_score=baseline.score,
            score_delta=-baseline.score,
            warnings=["missing_in_candidate"],
            metadata=_case_metadata(key, baseline, candidate),
        )

    baseline_severity = _status_severity(baseline.status)
    candidate_severity = _status_severity(candidate.status)
    score_delta = candidate.score - baseline.score
    warnings: list[str] = []
    if candidate_severity > baseline_severity:
        status = "regressed"
        warnings.append("status_regressed")
    elif candidate_severity < baseline_severity:
        status = "improved"
    elif score_delta < -score_tolerance:
        status = "regressed"
        warnings.append("score_regressed")
    elif score_delta > score_tolerance:
        status = "improved"
    else:
        status = "unchanged"

    return EvalReportComparisonCase(
        gold_case_id=candidate.gold_case_id or baseline.gold_case_id,
        case_id=candidate.case_id or baseline.case_id,
        status=status,
        baseline_status=baseline.status,
        candidate_status=candidate.status,
        baseline_score=baseline.score,
        candidate_score=candidate.score,
        score_delta=score_delta,
        warnings=warnings,
        metadata=_case_metadata(key, baseline, candidate),
    )


def _case_key(case: EvalExecutionReportCase) -> str:
    return case.gold_case_id or case.case_id


def _duplicate_keys(cases: list[EvalExecutionReportCase]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for case in cases:
        key = _case_key(case)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def _case_metadata(
    key: str,
    baseline: EvalExecutionReportCase | None,
    candidate: EvalExecutionReportCase | None,
) -> dict[str, object]:
    return {
        "case_key": key,
        "baseline_warning_count": len(baseline.warnings) if baseline else 0,
        "candidate_warning_count": len(candidate.warnings) if candidate else 0,
        "baseline_check_count": len(baseline.checks) if baseline else 0,
        "candidate_check_count": len(candidate.checks) if candidate else 0,
    }


def _status_severity(status: str) -> int:
    return _STATUS_SEVERITY.get(status, max(_STATUS_SEVERITY.values()) + 1)


def _safe_ref(prefix: str, value: str) -> str:
    text = str(value)
    if text and len(text) <= 128 and all(_is_safe_label_char(char) for char in text):
        return text
    return _stable_id(prefix, text)


def _is_safe_label_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in {"_", "-", ".", ":", "/"})


def _report_output_path(store: EvalStore, filename: str) -> Path:
    output_dir = store.root / "reports"
    if output_dir.is_symlink():
        raise ValueError("eval report directory must not be a symlink")
    output_path = (output_dir / filename).resolve()
    if not _is_relative_to(output_path, output_dir.resolve()):
        raise ValueError("eval compare filename must stay under store.root/reports")
    return output_path


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\n".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:24]}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True
