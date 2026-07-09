from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from ohmo.cli import app


def test_meta_report_selects_failed_cases_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / ".ohmo-home"
    report_path = tmp_path / "eval_report.json"
    traces_dir = tmp_path / "traces"
    rubrics_path = tmp_path / "rubrics.json"
    output_path = tmp_path / "meta_report.json"
    traces_dir.mkdir()

    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "report_kind": "execution_report",
                "report_id": "exec-1",
                "pack_id": "pack-1",
                "case_count": 2,
                "passed_count": 1,
                "failed_count": 1,
                "cases": [
                    {
                        "gold_case_id": "gold-pass",
                        "case_id": "case:pass",
                        "status": "passed",
                        "observed_trace": {"metadata": {}},
                    },
                    {
                        "gold_case_id": "gold-fail",
                        "case_id": "case:fail",
                        "status": "failed",
                        "observed_trace": {
                            "final_text_hash": "hash-final",
                            "metadata": {
                                "aspect.task_completion": 0.25,
                                "aspect.grounding": 0.5,
                                "rubric_gate_failures": ["task_completion"],
                            },
                        },
                    },
                ],
                "metadata": {"execution_id": "exec-1"},
            }
        ),
        encoding="utf-8",
    )
    (traces_dir / "case:fail-0.json").write_text(
        json.dumps(
            {
                "prompt": "Fix the failing deployment.",
                "final_text": "I could not inspect the deployment.",
                "tool_calls": [
                    {
                        "tool_name": "read_file",
                        "is_error": True,
                        "output": "mount failed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    rubrics_path.write_text(
        json.dumps(
            {
                "case:fail": {
                    "task_completion": [{"id": "tc1", "text": "Inspect deployment"}],
                    "grounding": [{"id": "g1", "text": "Use observed tool output"}],
                }
            }
        ),
        encoding="utf-8",
    )

    instances = []

    class _FakeMetaJudgeAttributor:
        def __init__(self, *, api_client, model: str, votes: int = 1) -> None:
            self.api_client = api_client
            self.model = model
            self.votes = votes
            self.calls = []
            instances.append(self)

        def attribute(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "blame": "harness_boundary",
                "subtype": "missing_mount",
                "confidence": 0.9,
                "evidence": "The trace shows the required mount was unavailable.",
                "votes": self.votes,
            }

    monkeypatch.setattr(
        "ohmo.cli._build_agent_runner_config",
        lambda *args, **kwargs: SimpleNamespace(
            api_client=object(),
            model=kwargs.get("model") or "fake-meta-model",
        ),
    )
    monkeypatch.setattr("ohmo.cli.MetaJudgeAttributor", _FakeMetaJudgeAttributor)

    result = CliRunner().invoke(
        app,
        [
            "evals",
            "meta-report",
            "--workspace",
            str(workspace),
            "--report",
            str(report_path),
            "--traces-dir",
            str(traces_dir),
            "--rubrics-file",
            str(rubrics_path),
            "--model",
            "meta-model",
            "--meta-votes",
            "2",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(instances) == 1
    assert instances[0].model == "meta-model"
    assert instances[0].votes == 2
    assert len(instances[0].calls) == 1
    call = instances[0].calls[0]
    assert call["task"] == "Fix the failing deployment."
    assert call["answer"] == "I could not inspect the deployment."
    assert call["aspect_scores"] == {"task_completion": 0.25, "grounding": 0.5}
    assert call["gate_failures"] == ["task_completion"]
    assert call["rubric"]["task_completion"][0]["id"] == "tc1"
    assert call["trajectory"][0]["is_error"] is True

    assert "case:fail | 0.25 | 0.5 | harness_boundary | missing_mount" in result.output
    assert "case:pass" not in result.output
    assert "summary:" in result.output
    assert "harness_debt_pct: 100.0" in result.output
    assert "model_signal_pct: 0.0" in result.output

    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["failed_count"] == 1
    assert output["attributions"][0]["case_id"] == "case:fail"
    assert output["summary"]["counts"]["harness_boundary"] == 1
