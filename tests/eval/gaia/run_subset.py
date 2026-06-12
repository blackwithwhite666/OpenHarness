"""GAIA subset runner — K=3 spawn + score + telemetry (PARTIAL SCAFFOLD).

Layer 3 of the deep-research architecture (ADR §2½, §4, component #10): the
runner sets up a per-task ``cwd`` + attachment, spawns the (future)
``deep-research`` sub-agent via OpenHarness's subprocess agent boundary, scores
the extracted answer with ``scorer.question_scorer``, and records JSONL +
``REPORT.md`` with Wilson 95% CIs.

What is REAL here (offline, unit-tested in test_scorer.py):
  * ``wilson_ci(k, n)`` — Wilson score-interval helper;
  * ``aggregate(rows)`` — per-level + overall pass counts;
  * ``write_jsonl`` / ``write_report`` — the telemetry writers;
  * ``score_run`` — extract + score a single transcript string.

What is STUBBED (needs the agent + HF dataset, not available in PR1):
  * ``_spawn_deep_research`` — the SAME subprocess boundary the main agent uses
    (``agent_tool.py`` -> ``registry.get_executor("subprocess").spawn(config)``,
    cwd + ``subagent_type="deep-research"``). The ``deep-research``
    AgentDefinition does not exist yet (ships in PR3 / M1), so the spawn call is
    a clearly-marked TODO that raises ``NotImplementedError``.
  * ``run_subset`` — the K=3 orchestration loop wiring loader -> spawn -> score.

Dependency direction (ADR §2½): this module *imports* OpenHarness; OpenHarness
never imports this. The eval harness is not part of the running bot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tests.eval.gaia.scorer import extract_answer, question_scorer

# K=3 per ADR §1 (every task run 3x; per-task score = mean) — live web flakes.
DEFAULT_K = 3

# Wilson 95% CI z-score (two-sided 0.95).
_Z_95 = 1.959963984540054


# --------------------------------------------------------------------------- #
# Statistics (REAL, offline, tested)
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k`` successes / ``n``.

    Used for per-level + overall accuracy bounds in ``REPORT.md``. A milestone
    gate passes only when the *lower* bound clears the bar (ADR §1) — never a
    point estimate. Returns ``(0.0, 0.0)`` for ``n == 0``.
    """
    if n < 0 or k < 0:
        raise ValueError(f"k and n must be non-negative, got k={k}, n={n}")
    if k > n:
        raise ValueError(f"k must not exceed n, got k={k} > n={n}")
    if n == 0:
        return (0.0, 0.0)

    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (lo, hi)


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[Any, dict[str, Any]]:
    """Group scored task rows into per-level + overall pass counts.

    A row's ``score`` is treated as a pass if ``>= 0.5`` (mean-of-K rounding:
    2/3 or 3/3 of K=3 runs correct counts as a task pass). Returns a mapping
    ``{level: {n, k, acc}, ..., "overall": {...}}``.
    """
    buckets: dict[Any, dict[str, Any]] = {}

    def _bump(key: Any, passed: bool) -> None:
        b = buckets.setdefault(key, {"n": 0, "k": 0})
        b["n"] += 1
        if passed:
            b["k"] += 1

    for row in rows:
        passed = float(row.get("score", 0.0)) >= 0.5
        _bump(row.get("level", "unknown"), passed)
        _bump("overall", passed)

    for b in buckets.values():
        b["acc"] = (b["k"] / b["n"]) if b["n"] else 0.0
    return buckets


# --------------------------------------------------------------------------- #
# Telemetry writers (REAL, offline, tested)
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Append-friendly JSONL writer (one task-result object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            fh.write("\n")
    return path


def write_report(
    path: Path | str,
    rows: Sequence[Mapping[str, Any]],
    *,
    git_sha: str,
    prev_sha: str | None = None,
) -> Path:
    """Render the rolled-up ``REPORT.md`` (per-level acc + Wilson CIs + overall).

    Levels are gated/tracked per ADR §1 (gate on L1+L2; L3 tracked, never
    gated). Extraction-/infra-failure are recorded distinctly from wrong-answer
    upstream and surfaced here when present on the rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agg = aggregate(rows)

    extraction_failures = sum(1 for r in rows if r.get("extraction_failure"))
    infra_failures = sum(1 for r in rows if r.get("infra_failure"))
    n_tasks = len(rows)

    lines: list[str] = []
    lines.append(f"# GAIA subset report — `{git_sha}`")
    lines.append("")
    if prev_sha:
        lines.append(f"_Delta vs previous sha `{prev_sha}`._")
        lines.append("")
    lines.append("| Level | n | passes | accuracy | Wilson 95% CI |")
    lines.append("|---|---|---|---|---|")

    def _row(label: str, bucket: Mapping[str, Any]) -> str:
        n, k = bucket["n"], bucket["k"]
        lo, hi = wilson_ci(k, n)
        return f"| {label} | {n} | {k} | {bucket['acc']:.3f} | [{lo:.3f}, {hi:.3f}] |"

    for level in (1, 2, 3):
        if level in agg:
            lines.append(_row(f"Level {level}", agg[level]))
    # Any non-numeric levels (e.g. "unknown") after the canonical three.
    for key in sorted(k for k in agg if k not in (1, 2, 3, "overall")):
        lines.append(_row(f"Level {key}", agg[key]))
    if "overall" in agg:
        lines.append(_row("Overall", agg["overall"]))

    lines.append("")
    lines.append(f"- Tasks scored: {n_tasks}")
    lines.append(f"- Extraction failures: {extraction_failures}")
    lines.append(f"- Infra failures (search 403 / fetch timeout / captcha): {infra_failures}")
    lines.append("")
    lines.append(
        "> Gate on L1+L2 (lower CI bound clears the bar). L3 tracked, never gated."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Single-transcript scoring (REAL, offline)
# --------------------------------------------------------------------------- #
def score_run(
    transcript: str | None, ground_truth: str, *, strict: bool = True
) -> dict[str, Any]:
    """Extract the final answer from one transcript and score it.

    Returns ``{answer, score, extraction_failure}``. An extraction miss
    (``answer is None``) is recorded as ``extraction_failure=True`` and scores
    0.0, kept distinct from a wrong-answer so a formatting bug is never read as a
    capability gap (ADR §1, §4).
    """
    answer = extract_answer(transcript)
    extraction_failure = answer is None
    correct = question_scorer(answer, ground_truth, strict=strict)
    return {
        "answer": answer,
        "score": 1.0 if correct else 0.0,
        "extraction_failure": extraction_failure,
    }


# --------------------------------------------------------------------------- #
# Spawn boundary (STUB — needs the deep-research AgentDefinition, PR3/M1)
# --------------------------------------------------------------------------- #
@dataclass
class TaskRun:
    """One GAIA task to execute (loader output shape)."""

    task_id: str
    question: str
    ground_truth: str
    level: int
    cwd: Path
    file_name: str | None = None
    attachments: list[Path] = field(default_factory=list)


async def _spawn_deep_research(
    task: TaskRun,
    *,
    model: str,
    timeout_s: float = 600.0,
) -> str:
    """Spawn the deep-research sub-agent on (question, cwd) -> transcript string.

    STUB — this is the single intersection with OpenHarness (ADR §2½): the SAME
    subprocess boundary the main agent uses to spawn sub-agents. The real
    implementation will mirror ``agent_tool.py``::

        from openharness.swarm.registry import get_backend_registry
        from openharness.swarm.types import TeammateSpawnConfig
        from openharness.coordinator.agent_definitions import get_agent_definition

        agent_def = get_agent_definition("deep-research")   # ships in PR3 / M1
        executor = get_backend_registry().get_executor("subprocess")
        config = TeammateSpawnConfig(
            name="deep-research",
            team="gaia-eval",
            prompt=task.question,
            cwd=str(task.cwd),                 # attachment contract (ADR §2(c))
            parent_session_id="gaia-eval",
            model=model,                       # PINNED — never "inherit" (ADR §7)
            system_prompt=agent_def.system_prompt,
            permissions=agent_def.permissions,
            task_type="local_agent",
        )
        result = await executor.spawn(config)
        # then poll task_get(result.task_id) to a final answer + transcript.

    TODO(PR3/M1): implement once the ``deep-research`` AgentDefinition exists.
    Until then this is the deliberate seam between the (shippable now) offline
    scorer and the (not-yet-built) agent.
    """
    raise NotImplementedError(
        "deep-research agent spawn is stubbed until PR3/M1: the 'deep-research' "
        "AgentDefinition does not exist yet. See _spawn_deep_research docstring "
        "for the exact subprocess-boundary call to wire up."
    )


async def run_subset(  # pragma: no cover - orchestration stub
    tasks: Sequence[TaskRun],
    *,
    model: str,
    k: int = DEFAULT_K,
    results_dir: Path | str,
    git_sha: str,
    strict: bool = True,
) -> Path:
    """K=3 orchestration: spawn -> score -> JSONL + REPORT (SKELETON).

    STUB body — depends on ``_spawn_deep_research`` (and a loader to build
    ``tasks``). The shape is fixed and the writers it calls are real/tested:

        for task in tasks:
            scores = []
            for _ in range(k):
                transcript = await _spawn_deep_research(task, model=model)
                scores.append(score_run(transcript, task.ground_truth, strict=strict))
            row = {... mean score, tokens, latency, extraction/infra flags ...}
            rows.append(row)
        write_jsonl(results_dir / f"{git_sha}.jsonl", rows)
        write_report(results_dir / "REPORT.md", rows, git_sha=git_sha)

    TODO(PR2/PR3): wire loader + spawn; K=3 mean; tokens/latency telemetry;
    tag extraction-failure vs infra-failure distinctly (ADR §1).
    """
    raise NotImplementedError(
        "run_subset orchestration is a skeleton: it depends on the loader "
        "(HF-gated dataset) and _spawn_deep_research (PR3/M1 agent). The score "
        "+ telemetry helpers it calls (score_run / write_jsonl / write_report / "
        "wilson_ci) are real and unit-tested."
    )
