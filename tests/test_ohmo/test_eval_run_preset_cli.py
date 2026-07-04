"""CLI tests for ``ohmo evals run --preset/--gate`` (spec-driven runs)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from ohmo.cli import app

_SPEC = {
    "version": 1,
    "defaults": {
        "pack": "eval_pack.json",
        "fixture_match": "args_then_order",
        "samples": 1,
    },
    "presets": {
        "inner": {
            "agent_runner": "query-engine",
            "scorer": "freezing_judge",
            "model": "gpt-5.5",
            "system_prompt_file": "system_prompt.txt",
            "histories_file": "histories.json",
            "live_skill": False,
            "cache": {"completions": "completions.json", "mode": "strict"},
            "gate": {"hit_floor": 0.98, "passed_baseline": 46},
        },
        "faithful": {
            "agent_runner": "fs-sandbox",
            "scorer": "grounding_judge_v1",
            "model": "gpt-5.5",
            "live_skill": True,
            "cache": {"mode": "off"},
        },
    },
}


def _make_bundle(workspace: Path) -> Path:
    evals = workspace / "evals"
    evals.mkdir(parents=True, exist_ok=True)
    (evals / "spec.json").write_text(json.dumps(_SPEC), encoding="utf-8")
    (evals / "system_prompt.txt").write_text("PINNED PROMPT", encoding="utf-8")
    (evals / "histories.json").write_text("{}", encoding="utf-8")
    (evals / "completions.json").write_text("{}", encoding="utf-8")
    return evals


def _fake_report(*, hit_rate: float = 1.0, passed: int = 46, total: int = 59):
    def fake_run_ohmo_eval_report(**kwargs):
        fake_run_ohmo_eval_report.calls.append(kwargs)
        return SimpleNamespace(
            report_only=kwargs.get("report_only", False),
            write=SimpleNamespace(
                path=Path(kwargs["workspace"]) / "evals" / "reports" / "r.json",
                report=SimpleNamespace(
                    case_count=total,
                    passed_count=passed,
                    failed_count=total - passed,
                    metadata={
                        "completion_cache": {
                            "hit_rate": hit_rate,
                            "hits": int(round(hit_rate * 751)),
                            "misses": int(round((1 - hit_rate) * 751)),
                        }
                    },
                ),
            ),
        )

    fake_run_ohmo_eval_report.calls = []
    return fake_run_ohmo_eval_report


def test_preset_projects_frozen_inputs(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    evals = _make_bundle(workspace)
    fake = _fake_report(passed=2, total=2)  # no gate here: full-pass so exit 0
    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake)

    result = CliRunner().invoke(
        app,
        ["evals", "run", "--workspace", str(workspace), "--preset", "inner"],
    )

    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["agent_runner_name"] == "query-engine"
    assert call["scorer"] == "freezing_judge"
    assert call["model"] == "gpt-5.5"
    assert call["fixture_match"] == "args_then_order"
    assert call["samples"] == 1
    assert call["pack_filename"] == "eval_pack.json"
    assert call["system_prompt"] == "PINNED PROMPT"
    assert call["cache_strict"] is True
    assert call["live_skill"] is False
    assert call["cache_completions"] == str(evals / "completions.json")
    assert call["histories_file"] == str(evals / "histories.json")


def test_explicit_flag_overrides_preset(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    _make_bundle(workspace)
    fake = _fake_report(passed=2, total=2)  # no gate here: full-pass so exit 0
    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", fake)

    result = CliRunner().invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--preset",
            "inner",
            "--model",
            "gpt-6",
            "--samples",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    call = fake.calls[0]
    assert call["model"] == "gpt-6"
    assert call["samples"] == 4
    assert call["agent_runner_name"] == "query-engine"  # untouched -> preset


def test_gate_passes_within_thresholds(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    _make_bundle(workspace)
    monkeypatch.setattr(
        "ohmo.cli.run_ohmo_eval_report", _fake_report(hit_rate=1.0, passed=46)
    )

    result = CliRunner().invoke(
        app,
        ["evals", "run", "--workspace", str(workspace), "--preset", "inner", "--gate"],
    )

    assert result.exit_code == 0, result.output
    assert "inner gate: OK" in result.output


def test_gate_fails_on_scored_regression(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    _make_bundle(workspace)
    monkeypatch.setattr(
        "ohmo.cli.run_ohmo_eval_report", _fake_report(hit_rate=1.0, passed=45)
    )

    result = CliRunner().invoke(
        app,
        ["evals", "run", "--workspace", str(workspace), "--preset", "inner", "--gate"],
    )

    assert result.exit_code == 1
    assert "inner gate: FAILED" in result.output


def test_gate_fails_on_cache_miss(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    _make_bundle(workspace)
    monkeypatch.setattr(
        "ohmo.cli.run_ohmo_eval_report", _fake_report(hit_rate=0.5, passed=46)
    )

    result = CliRunner().invoke(
        app,
        ["evals", "run", "--workspace", str(workspace), "--preset", "inner", "--gate"],
    )

    assert result.exit_code == 1
    assert "inner gate: FAILED" in result.output


def test_spec_requires_preset(tmp_path: Path):
    workspace = tmp_path / ".ohmo-home"
    evals = _make_bundle(workspace)
    result = CliRunner().invoke(
        app,
        ["evals", "run", "--workspace", str(workspace), "--spec", str(evals / "spec.json")],
    )
    assert result.exit_code == 1
    assert "--spec requires --preset" in result.output


def test_gate_without_gate_preset_errors(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".ohmo-home"
    _make_bundle(workspace)
    monkeypatch.setattr("ohmo.cli.run_ohmo_eval_report", _fake_report())
    result = CliRunner().invoke(
        app,
        [
            "evals",
            "run",
            "--workspace",
            str(workspace),
            "--preset",
            "faithful",
            "--gate",
        ],
    )
    assert result.exit_code == 1
    assert "gate thresholds" in result.output
