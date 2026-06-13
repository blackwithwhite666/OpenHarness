"""Paired milestone-over-milestone comparison for the GAIA gate split (ADR §1, §5).

Two milestones (e.g. M0 baseline vs M1 deep-research) run the **same** gate tasks,
so the correct test is **paired** — McNemar's exact test on per-task pass/fail — not
a comparison of two independent Wilson CIs. At n=30 the independent CIs are far too
wide (±~0.15) to detect a realistic single-milestone gain; McNemar conditions only on
the tasks the two milestones *disagree* on and is much more powerful.

Gate (ADR §5, M1): McNemar one-sided p < 0.05 in the new milestone's favour AND no
L1+L2 regression (L1+L2 passes must not drop).

Usage:
    python -m tests.eval.gaia.compare <baseline.jsonl> <candidate.jsonl>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

__all__ = ["load_scores", "binom_lower_tail", "mcnemar_exact", "compare"]


def load_scores(path: Path | str) -> dict[str, tuple[bool, int]]:
    """Read a runner JSONL into ``task_id -> (passed, level)``.

    A task "passes" when its aggregated ``score`` (K-run majority) is >= 0.5.
    """
    out: dict[str, tuple[bool, int]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[d["task_id"]] = (float(d["score"]) >= 0.5, int(d["level"]))
    return out


def binom_lower_tail(k: int, n: int) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, 0.5)`` — exact, no scipy."""
    if n == 0:
        return 1.0
    if k < 0:
        return 0.0
    return sum(math.comb(n, i) for i in range(0, min(k, n) + 1)) * (0.5**n)


def mcnemar_exact(b: int, c: int) -> tuple[float, float]:
    """Exact McNemar on the discordant counts.

    ``b`` = #tasks only the BASELINE passed; ``c`` = #tasks only the CANDIDATE passed.
    Under H0 each discordant pair favours either side with p=0.5, so ``b ~ Bin(b+c, .5)``.

    Returns ``(p_two_sided, p_one_sided_candidate_better)``. The candidate is better
    when ``c`` is large / ``b`` is small, so the one-sided p is the lower tail at ``b``.
    """
    n = b + c
    if n == 0:
        return 1.0, 1.0
    p_two = min(1.0, 2.0 * binom_lower_tail(min(b, c), n))
    p_one_candidate = binom_lower_tail(b, n)
    return p_two, p_one_candidate


def _l12_passes(scores: dict[str, tuple[bool, int]]) -> int:
    return sum(1 for passed, level in scores.values() if passed and level in (1, 2))


def compare(baseline_path: Path | str, candidate_path: Path | str) -> dict:
    """Pair baseline vs candidate by ``task_id`` and run the M1 gate."""
    base = load_scores(baseline_path)
    cand = load_scores(candidate_path)
    common = sorted(set(base) & set(cand))
    b = sum(1 for t in common if base[t][0] and not cand[t][0])  # only baseline pass
    c = sum(1 for t in common if not base[t][0] and cand[t][0])  # only candidate pass
    both = sum(1 for t in common if base[t][0] and cand[t][0])
    neither = sum(1 for t in common if not base[t][0] and not cand[t][0])
    p_two, p_one = mcnemar_exact(b, c)
    l12_base, l12_cand = _l12_passes(base), _l12_passes(cand)
    return {
        "n_common": len(common),
        "baseline_acc": round(sum(p for p, _ in base.values()) / max(len(base), 1), 3),
        "candidate_acc": round(sum(p for p, _ in cand.values()) / max(len(cand), 1), 3),
        "only_baseline_pass_b": b,
        "only_candidate_pass_c": c,
        "both_pass": both,
        "neither_pass": neither,
        "mcnemar_p_two_sided": round(p_two, 4),
        "mcnemar_p_one_sided_candidate_better": round(p_one, 4),
        "candidate_significant_at_05": p_one < 0.05,
        "l12_pass_baseline": l12_base,
        "l12_pass_candidate": l12_cand,
        "l12_no_regression": l12_cand >= l12_base,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: compare <baseline.jsonl> <candidate.jsonl>", file=sys.stderr)
        return 2
    res = compare(argv[0], argv[1])
    print(json.dumps(res, indent=2))
    gate = res["candidate_significant_at_05"] and res["l12_no_regression"]
    print(
        f"\nM1 GATE: {'PASS' if gate else 'FAIL'} "
        f"(McNemar one-sided p={res['mcnemar_p_one_sided_candidate_better']}; "
        f"discordant b={res['only_baseline_pass_b']} c={res['only_candidate_pass_c']}; "
        f"L1+L2 {res['l12_pass_candidate']} vs {res['l12_pass_baseline']})"
    )
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
