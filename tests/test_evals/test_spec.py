"""Unit tests for eval spec + preset resolution (``ohmo.evals.spec``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ohmo.evals.spec import (
    GateThresholds,
    SpecError,
    default_spec_path,
    evaluate_gate,
    load_spec,
    resolve_run,
)

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
            "scorer": "trajectory_judge_v1",
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


def _write_spec(dir_path: Path, spec: dict | None = None) -> Path:
    spec_path = dir_path / "spec.json"
    spec_path.write_text(json.dumps(spec or _SPEC), encoding="utf-8")
    return spec_path


def test_resolve_inner_pins_identity_and_resolves_paths_relative_to_spec(tmp_path):
    spec_path = _write_spec(tmp_path)
    run = resolve_run(spec_path, "inner")

    assert run.agent_runner_name == "query-engine"
    assert run.scorer == "trajectory_judge_v1"
    assert run.model == "gpt-5.5"
    assert run.fixture_match == "args_then_order"  # from defaults
    assert run.samples == 1
    assert run.live_skill is False
    assert run.cache_mode == "strict"
    # File references resolve against the spec file's directory (portable bundle).
    assert run.system_prompt_file == str(tmp_path / "system_prompt.txt")
    assert run.histories_file == str(tmp_path / "histories.json")
    assert run.cache_completions == str(tmp_path / "completions.json")
    assert run.gate == GateThresholds(hit_floor=0.98, passed_baseline=46)


def test_resolve_faithful_has_no_cache_and_no_gate(tmp_path):
    spec_path = _write_spec(tmp_path)
    run = resolve_run(spec_path, "faithful")

    assert run.agent_runner_name == "fs-sandbox"
    assert run.cache_mode == "off"
    assert run.cache_completions is None
    assert run.live_skill is True
    assert run.gate is None


def test_overrides_win_over_preset(tmp_path):
    spec_path = _write_spec(tmp_path)
    run = resolve_run(
        spec_path,
        "inner",
        overrides={"model": "gpt-6", "cache_mode": "record", "samples": 5},
    )
    assert run.model == "gpt-6"
    assert run.cache_mode == "record"
    assert run.samples == 5
    # untouched fields keep the preset value
    assert run.agent_runner_name == "query-engine"


def test_override_of_unknown_field_is_rejected(tmp_path):
    spec_path = _write_spec(tmp_path)
    with pytest.raises(SpecError, match="cannot override"):
        resolve_run(spec_path, "inner", overrides={"bogus": 1})


def test_unknown_preset_lists_available(tmp_path):
    spec_path = _write_spec(tmp_path)
    with pytest.raises(SpecError, match="faithful, inner"):
        resolve_run(spec_path, "nope")


def test_strict_cache_without_file_is_rejected(tmp_path):
    bad = {
        "version": 1,
        "presets": {"x": {"agent_runner": "query-engine", "cache": {"mode": "strict"}}},
    }
    spec_path = _write_spec(tmp_path, bad)
    with pytest.raises(SpecError, match="needs a cache file"):
        resolve_run(spec_path, "x")


def test_unsupported_version_is_rejected(tmp_path):
    spec_path = _write_spec(tmp_path, {"version": 99, "presets": {"x": {}}})
    with pytest.raises(SpecError, match="unsupported version"):
        load_spec(spec_path)


def test_missing_spec_file_raises(tmp_path):
    with pytest.raises(SpecError, match="not found"):
        load_spec(tmp_path / "absent.json")


def test_default_spec_path_finds_bundle_spec(tmp_path):
    assert default_spec_path(tmp_path) is None
    evals = tmp_path / "evals"
    evals.mkdir()
    _write_spec(evals)
    assert default_spec_path(tmp_path) == evals / "spec.json"


def test_evaluate_gate_passes_within_thresholds():
    result = evaluate_gate(
        GateThresholds(0.98, 46),
        hit_rate=1.0,
        hits=751,
        misses=0,
        passed=46,
        total=59,
    )
    assert result.ok is True
    assert result.reasons == ()
    assert "gate: OK" in result.describe("inner")


def test_evaluate_gate_fails_on_hit_rate_drop():
    result = evaluate_gate(
        GateThresholds(0.98, 46),
        hit_rate=0.50,
        hits=30,
        misses=29,
        passed=46,
        total=59,
    )
    assert result.ok is False
    assert any("hit-rate" in reason for reason in result.reasons)
    assert "gate: FAILED" in result.describe("inner")


def test_evaluate_gate_fails_on_scored_regression():
    result = evaluate_gate(
        GateThresholds(0.98, 46),
        hit_rate=1.0,
        hits=751,
        misses=0,
        passed=45,
        total=59,
    )
    assert result.ok is False
    assert len(result.reasons) == 1
    assert "baseline" in result.reasons[0]
