"""Offline tests for the paired M0-vs-M1 comparison (ADR §1, §5).

Zero network. Builds tiny runner-shaped JSONL files in ``tmp_path`` and checks the
McNemar exact math + the M1 gate (significance AND no L1+L2 regression).
"""

from __future__ import annotations

import json

from tests.eval.gaia.compare import binom_lower_tail, compare, mcnemar_exact


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _row(task_id, score, level):
    return {"task_id": task_id, "score": score, "level": level}


def test_binom_lower_tail_known_values():
    assert binom_lower_tail(0, 8) == 1 / 256  # C(8,0) * .5^8
    assert binom_lower_tail(8, 8) == 1.0  # whole mass
    assert binom_lower_tail(-1, 8) == 0.0
    assert binom_lower_tail(0, 0) == 1.0


def test_mcnemar_candidate_clearly_better():
    # baseline broke 0, candidate fixed 8 -> one-sided p = 1/256 << 0.05.
    p_two, p_one = mcnemar_exact(b=0, c=8)
    assert p_one < 0.05
    assert p_two == min(1.0, 2 / 256)


def test_mcnemar_no_discordant_is_null():
    assert mcnemar_exact(0, 0) == (1.0, 1.0)


def test_compare_gate_pass_significant_no_regression(tmp_path):
    # 10 tasks. Baseline fails t0..t5 (L1/L2), candidate passes all -> b=0, c=6.
    base = [_row(f"t{i}", 0.0 if i < 6 else 1.0, 1 if i % 2 else 2) for i in range(10)]
    cand = [_row(f"t{i}", 1.0, 1 if i % 2 else 2) for i in range(10)]
    _write_jsonl(tmp_path / "m0.jsonl", base)
    _write_jsonl(tmp_path / "m1.jsonl", cand)

    res = compare(tmp_path / "m0.jsonl", tmp_path / "m1.jsonl")
    assert res["only_baseline_pass_b"] == 0
    assert res["only_candidate_pass_c"] == 6
    assert res["candidate_significant_at_05"] is True
    assert res["l12_no_regression"] is True


def test_compare_identical_is_not_significant(tmp_path):
    rows = [_row(f"t{i}", 1.0 if i % 2 else 0.0, 1) for i in range(10)]
    _write_jsonl(tmp_path / "a.jsonl", rows)
    _write_jsonl(tmp_path / "b.jsonl", rows)
    res = compare(tmp_path / "a.jsonl", tmp_path / "b.jsonl")
    assert res["only_baseline_pass_b"] == res["only_candidate_pass_c"] == 0
    assert res["candidate_significant_at_05"] is False  # gate FAIL on a tie


def test_compare_flags_l12_regression(tmp_path):
    # Candidate fixes one L3 but breaks two L1 -> significant-ish, but regresses L1+L2.
    base = [_row("a", 1.0, 1), _row("b", 1.0, 1), _row("c", 0.0, 3)]
    cand = [_row("a", 0.0, 1), _row("b", 0.0, 1), _row("c", 1.0, 3)]
    _write_jsonl(tmp_path / "m0.jsonl", base)
    _write_jsonl(tmp_path / "m1.jsonl", cand)
    res = compare(tmp_path / "m0.jsonl", tmp_path / "m1.jsonl")
    assert res["l12_pass_baseline"] == 2
    assert res["l12_pass_candidate"] == 0
    assert res["l12_no_regression"] is False  # gate FAIL on regression


def test_compare_partial_candidate_uses_common_tasks(tmp_path):
    # Baseline = 4 tasks; candidate only completed the first 2 (crashed sweep).
    # acc/l12 must be over the 2 COMMON tasks, not 2-vs-4 denominators (which would
    # read a partial run as a regression).
    base = [_row("t0", 1.0, 1), _row("t1", 0.0, 1), _row("t2", 1.0, 2), _row("t3", 1.0, 2)]
    cand = [_row("t0", 1.0, 1), _row("t1", 0.0, 1)]
    _write_jsonl(tmp_path / "m0.jsonl", base)
    _write_jsonl(tmp_path / "m1.jsonl", cand)
    res = compare(tmp_path / "m0.jsonl", tmp_path / "m1.jsonl")
    assert res["n_common"] == 2
    assert res["partial"] is True
    # On the 2 common tasks both pass t0 and fail t1 -> identical, no regression.
    assert res["baseline_acc"] == res["candidate_acc"] == 0.5
    assert res["l12_pass_baseline"] == res["l12_pass_candidate"] == 1
    assert res["l12_no_regression"] is True
    assert res["only_baseline_pass_b"] == res["only_candidate_pass_c"] == 0
