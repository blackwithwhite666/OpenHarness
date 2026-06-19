from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals import EvalExecutionReport, EvalExecutionReportCase
from ohmo.evals import (
    compare_ohmo_eval_reports,
    list_ohmo_eval_baselines,
    save_ohmo_eval_baseline,
)


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


def test_save_ohmo_eval_baseline_validates_and_lists_baselines(tmp_path: Path):
    workspace = tmp_path / "workspace"
    reports_dir = workspace / "evals" / "reports"
    reports_dir.mkdir(parents=True)
    _write_report(
        reports_dir / "eval_report.json",
        _report("candidate", [_case("gold-1", "case-1", "passed", 1.0)]),
    )

    saved = save_ohmo_eval_baseline(
        workspace=workspace,
        source_report="eval_report.json",
        name="main",
    )
    listed = list_ohmo_eval_baselines(workspace=workspace)

    assert saved.name == "main"
    assert saved.relative_path == "reports/baselines/main.json"
    assert saved.path == reports_dir / "baselines" / "main.json"
    assert saved.case_count == 1
    assert saved.passed_count == 1
    assert saved.non_passed_count == 0
    assert len(listed.baselines) == 1
    assert listed.baselines[0].name == "main"
    assert listed.baselines[0].relative_path == "reports/baselines/main.json"


def test_save_ohmo_eval_baseline_rejects_overwrite_and_bad_names(tmp_path: Path):
    workspace = tmp_path / "workspace"
    reports_dir = workspace / "evals" / "reports"
    reports_dir.mkdir(parents=True)
    _write_report(
        reports_dir / "eval_report.json",
        _report("candidate", [_case("gold-1", "case-1", "passed", 1.0)]),
    )
    save_ohmo_eval_baseline(workspace=workspace, name="main")

    with pytest.raises(ValueError, match="already exists"):
        save_ohmo_eval_baseline(workspace=workspace, name="main")
    with pytest.raises(ValueError, match="baseline name"):
        save_ohmo_eval_baseline(workspace=workspace, name="../bad")

    replaced = save_ohmo_eval_baseline(
        workspace=workspace,
        name="main.json",
        overwrite=True,
    )

    assert replaced.name == "main"


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
