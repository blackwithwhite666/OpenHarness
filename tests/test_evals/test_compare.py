from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals import (
    EvalExecutionReport,
    EvalExecutionReportCase,
    EvalStore,
    compare_execution_reports,
    read_execution_report,
    write_execution_report_comparison,
)


def test_compare_execution_reports_classifies_regressions_and_improvements():
    baseline = _report(
        "baseline",
        [
            _case("gold-1", "case-1", "passed", 1.0),
            _case("gold-2", "case-2", "failed", 0.4),
            _case("gold-3", "case-3", "passed", 0.8),
        ],
    )
    candidate = _report(
        "candidate",
        [
            _case("gold-1", "case-1", "failed", 0.5),
            _case("gold-2", "case-2", "passed", 1.0),
            _case("gold-4", "case-4", "passed", 1.0),
        ],
    )

    comparison = compare_execution_reports(baseline, candidate)

    assert comparison.report_kind == "execution_comparison_report"
    assert comparison.case_count == 4
    assert comparison.compared_count == 2
    assert comparison.regression_count == 2
    assert comparison.improvement_count == 1
    assert comparison.added_count == 1
    assert comparison.removed_count == 1
    by_case = {case.case_id: case for case in comparison.cases}
    assert by_case["case-1"].status == "regressed"
    assert by_case["case-1"].warnings == ["status_regressed"]
    assert by_case["case-2"].status == "improved"
    assert by_case["case-3"].status == "removed"
    assert by_case["case-3"].warnings == ["missing_in_candidate"]
    assert by_case["case-4"].status == "added"


def test_compare_execution_reports_honors_score_tolerance():
    baseline = _report("baseline", [_case("gold-1", "case-1", "passed", 0.90)])
    candidate = _report("candidate", [_case("gold-1", "case-1", "passed", 0.85)])

    tolerant = compare_execution_reports(
        baseline,
        candidate,
        score_tolerance=0.10,
    )
    strict = compare_execution_reports(
        baseline,
        candidate,
        score_tolerance=0.01,
    )

    assert tolerant.cases[0].status == "unchanged"
    assert tolerant.regression_count == 0
    assert strict.cases[0].status == "regressed"
    assert strict.cases[0].warnings == ["score_regressed"]
    assert strict.regression_count == 1


def test_write_execution_report_comparison_is_metadata_only_and_path_safe(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_report(
        baseline_path,
        _report("baseline private text must not leak", [_case("gold-1", "case-1", "passed", 1.0)]),
    )
    _write_report(
        candidate_path,
        _report("candidate private text must not leak", [_case("gold-1", "case-1", "failed", 0.5)]),
    )

    write = write_execution_report_comparison(
        store,
        baseline_report_path=baseline_path,
        candidate_report_path=candidate_path,
    )

    assert write.relative_path == "reports/eval_compare.json"
    assert write.report.regression_count == 1
    serialized = write.path.read_text(encoding="utf-8")
    assert "private text must not leak" not in serialized
    with pytest.raises(ValueError, match="store.root/reports"):
        write_execution_report_comparison(
            store,
            baseline_report_path=baseline_path,
            candidate_report_path=candidate_path,
            report_filename="../compare.json",
        )


def test_read_execution_report_rejects_non_execution_report(tmp_path: Path):
    path = tmp_path / "wrong.json"
    report = _report("report", [_case("gold-1", "case-1", "passed", 1.0)])
    path.write_text(
        report.model_copy(update={"report_kind": "smoke_report"}).model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not an execution_report"):
        read_execution_report(path)


def test_compare_execution_reports_rejects_duplicate_case_ids():
    baseline = _report(
        "baseline",
        [
            _case("gold-1", "case-1", "passed", 1.0),
            _case("gold-1", "case-duplicate", "passed", 1.0),
        ],
    )
    candidate = _report("candidate", [_case("gold-1", "case-1", "passed", 1.0)])

    with pytest.raises(ValueError, match="duplicate report case ids"):
        compare_execution_reports(baseline, candidate)


def _case(
    gold_case_id: str,
    case_id: str,
    status: str,
    score: float,
) -> EvalExecutionReportCase:
    return EvalExecutionReportCase(
        gold_case_id=gold_case_id,
        case_id=case_id,
        status=status,
        score=score,
        checks={"ok": status == "passed"},
        warnings=[] if status == "passed" else ["ok"],
    )


def _report(report_id: str, cases: list[EvalExecutionReportCase]) -> EvalExecutionReport:
    return EvalExecutionReport(
        report_id=report_id,
        pack_id="pack-1",
        case_count=len(cases),
        passed_count=sum(1 for case in cases if case.status == "passed"),
        failed_count=sum(1 for case in cases if case.status == "failed"),
        blocked_count=sum(1 for case in cases if case.status == "blocked"),
        error_count=sum(1 for case in cases if case.status == "error"),
        cases=cases,
        metadata={"privacy": "metadata_only"},
    )


def _write_report(path: Path, report: EvalExecutionReport) -> None:
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
