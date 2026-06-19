from __future__ import annotations

from pathlib import Path

from openharness.evals import EvalExecutionReport, EvalExecutionReportCase
from ohmo.evals import compare_ohmo_eval_reports


def test_compare_ohmo_eval_reports_resolves_relative_report_paths(tmp_path: Path):
    workspace = tmp_path / "workspace"
    reports_dir = workspace / "evals" / "reports"
    reports_dir.mkdir(parents=True)
    _write_report(
        reports_dir / "baseline.json",
        _report("baseline", [_case("gold-1", "case-1", "passed", 1.0)]),
    )
    _write_report(
        reports_dir / "candidate.json",
        _report("candidate", [_case("gold-1", "case-1", "failed", 0.5)]),
    )

    result = compare_ohmo_eval_reports(
        workspace=workspace,
        baseline_report="baseline.json",
        candidate_report="candidate.json",
        report_only=True,
    )

    assert result.report_only is True
    assert result.write.path == reports_dir / "eval_compare.json"
    assert result.write.report.regression_count == 1


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
    )


def _report(report_id: str, cases: list[EvalExecutionReportCase]) -> EvalExecutionReport:
    return EvalExecutionReport(
        report_id=report_id,
        pack_id="pack-1",
        case_count=len(cases),
        passed_count=sum(1 for case in cases if case.status == "passed"),
        failed_count=sum(1 for case in cases if case.status == "failed"),
        blocked_count=0,
        error_count=0,
        cases=cases,
        metadata={"privacy": "metadata_only"},
    )


def _write_report(path: Path, report: EvalExecutionReport) -> None:
    path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
