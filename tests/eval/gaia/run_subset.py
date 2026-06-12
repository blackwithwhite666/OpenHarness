"""GAIA subset runner — K=3 spawn + score + telemetry.

Layer 3 of the deep-research architecture (ADR §2½, §4, component #10): the
runner sets up a per-task ``cwd`` + attachment, spawns the agent via
OpenHarness's subprocess agent boundary (the SAME boundary the main agent uses
to spawn sub-agents — ``agent_tool.py``), scores the extracted answer with
``scorer.question_scorer``, and records JSONL + ``REPORT.md`` with Wilson 95%
CIs.

The spawn path mirrors ``agent_tool.AgentTool.execute`` — see
:func:`_spawn_agent` for the exact resolve-def → build-config → spawn → poll →
read sequence. ``model`` is PINNED by the runner (ADR §7) — never ``"inherit"``
/ ``None`` — so the eval is deterministic. For M0 ``--agent`` is the CURRENT
general worker (``general-purpose``); the dedicated ``deep-research``
AgentDefinition ships in PR3 / M1.

Offline-testable seams: the spawn helper (:func:`_spawn_agent`) and the
poll/read helpers are split so the unit tests monkeypatch a canned transcript
provider and never touch a real subprocess or the network.

Dependency direction (ADR §2½): this module *imports* OpenHarness; OpenHarness
never imports this. The eval harness is not part of the running bot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Sequence

from tests.eval.gaia.scorer import extract_answer, question_scorer

# K=3 per ADR §1 (every task run 3x; per-task score = mean) — live web flakes.
DEFAULT_K = 3

# Wilson 95% CI z-score (two-sided 0.95).
_Z_95 = 1.959963984540054

# Default pinned model for the eval path (ADR §7: never "inherit").
DEFAULT_MODEL = "claude-opus-4-8"

# How long to wait for one spawned agent before tagging an infra failure.
DEFAULT_TIMEOUT_S = 600.0
# read_task_output tail cap — large so long transcripts aren't truncated before
# extract_answer runs (the manager default is 12000).
_READ_MAX_BYTES = 2_000_000


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
    upstream and surfaced here (counts + rates). Median tokens & latency are
    rendered when present on the rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    agg = aggregate(rows)

    extraction_failures = sum(1 for r in rows if r.get("extraction_failure"))
    infra_failures = sum(1 for r in rows if r.get("infra_failure"))
    n_tasks = len(rows)

    def _median(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    median_tokens = _median("tokens")
    median_latency = _median("latency_s")

    def _rate(count: int) -> str:
        return f"{(count / n_tasks):.3f}" if n_tasks else "0.000"

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
    for key in sorted(str(k) for k in agg if k not in (1, 2, 3, "overall")):
        lines.append(_row(f"Level {key}", agg[key]))
    if "overall" in agg:
        lines.append(_row("Overall", agg["overall"]))

    lines.append("")
    lines.append(f"- Tasks scored: {n_tasks}")
    lines.append(
        f"- Extraction failures: {extraction_failures} (rate {_rate(extraction_failures)})"
    )
    lines.append(
        "- Infra failures (spawn error / timeout / search 403 / fetch timeout / "
        f"captcha): {infra_failures} (rate {_rate(infra_failures)})"
    )
    if median_tokens is not None:
        lines.append(f"- Median tokens/task: {median_tokens:.0f}")
    if median_latency is not None:
        lines.append(f"- Median latency/task: {median_latency:.1f}s")
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
# Task model + per-task working dir / attachment contract (ADR §2(c))
# --------------------------------------------------------------------------- #
@dataclass
class TaskRun:
    """One GAIA task to execute (loader output, post per-task-dir setup)."""

    task_id: str
    question: str
    ground_truth: str
    level: int
    cwd: Path
    file_name: str | None = None
    attachments: list[Path] = field(default_factory=list)


# A spawn function: (TaskRun, model) -> transcript string. Raises on infra
# failure (spawn error / timeout). The default is the real subprocess boundary;
# the unit tests inject a canned-transcript stand-in here so nothing touches a
# subprocess or the network.
SpawnFn = Callable[["TaskRun", str], Awaitable[str]]


class InfraFailure(RuntimeError):
    """Raised when a spawn errors / times out (tagged distinctly from wrong)."""


def prepare_task_dir(
    *,
    task_id: str,
    question: str,
    ground_truth: str,
    level: int,
    source_file: Path | None,
    work_root: Path,
) -> TaskRun:
    """Make a per-task working dir and copy the attachment in (ADR §2(c)).

    The sub-agent is spawned with this dir as ``cwd``; the prompt declares the
    input files are in cwd. ~50% of validation tasks carry a file — a broken
    attachment path silently zeroes them (ADR §7), so we copy + record the
    landed path and let the caller assert presence.
    """
    task_dir = Path(work_root) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    attachments: list[Path] = []
    file_name: str | None = None
    if source_file is not None:
        src = Path(source_file)
        dest = task_dir / src.name
        shutil.copy2(src, dest)
        attachments.append(dest)
        file_name = src.name
    return TaskRun(
        task_id=task_id,
        question=question,
        ground_truth=ground_truth,
        level=level,
        cwd=task_dir,
        file_name=file_name,
        attachments=attachments,
    )


def build_prompt(task: TaskRun) -> str:
    """Augment the question with the cwd-attachment contract (OWL-adapted).

    The sub-agent prompt declares that input files for this task are in its cwd,
    lists the explicit attached file path(s), and asks for a single final answer
    wrapped in ``<final_answer>…</final_answer>`` (our sentinel; the scorer also
    has a tolerant fallback).
    """
    parts = [task.question.strip(), ""]
    if task.attachments:
        listed = "\n".join(f"  - {p.name}" for p in task.attachments)
        parts.append(
            "Input files for this task are in your working directory "
            f"({task.cwd}). List it first; the attached file(s):\n{listed}"
        )
        parts.append("")
    parts.append(
        "When done, output the single, exact final answer wrapped in "
        "<final_answer>…</final_answer> with no commentary inside the tags."
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Spawn boundary (REAL — the SAME subprocess boundary the main agent uses)
# --------------------------------------------------------------------------- #
async def _wait_terminal(
    manager: Any,
    task_id: str,
    *,
    timeout_s: float,
    poll_interval_s: float = 0.5,
) -> str:
    """Poll ``manager.get_task(task_id).status`` until terminal, return status.

    Terminal statuses are ``completed | failed | killed`` (manager.py). Raises
    :class:`InfraFailure` on timeout.
    """
    deadline = time.monotonic() + timeout_s
    terminal = {"completed", "failed", "killed"}
    while True:
        record = manager.get_task(task_id)
        status = getattr(record, "status", None) if record is not None else None
        if status in terminal:
            return status
        if time.monotonic() >= deadline:
            raise InfraFailure(
                f"agent task {task_id} did not finish within {timeout_s}s"
            )
        await asyncio.sleep(poll_interval_s)


async def _spawn_agent(
    task: TaskRun,
    model: str,
    *,
    subagent_type: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Run one agent on (prompt, cwd) -> transcript string.

    The single intersection with OpenHarness (ADR §2½): the SAME subprocess
    boundary the main agent uses to spawn sub-agents. Mirrors
    ``agent_tool.AgentTool.execute`` — resolve the def, copy its
    ``system_prompt`` / ``permissions`` into the config, PIN the model.

    Raises :class:`InfraFailure` on spawn error / timeout so the caller tags it
    distinctly from a wrong answer (ADR §1).
    """
    # Imported lazily so the offline tests (which monkeypatch the spawn) don't
    # drag in the OpenHarness subprocess stack.
    from openharness.coordinator.agent_definitions import get_agent_definition
    from openharness.swarm.registry import get_backend_registry
    from openharness.swarm.types import TeammateSpawnConfig
    from openharness.tasks import get_task_manager

    agent_def = get_agent_definition(subagent_type)

    executor = get_backend_registry().get_executor("subprocess")
    config = TeammateSpawnConfig(
        name=subagent_type,
        team="gaia-eval",
        prompt=build_prompt(task),
        cwd=str(task.cwd),
        parent_session_id="gaia-eval",
        model=model,  # PINNED — never "inherit" / None (ADR §7).
        system_prompt=agent_def.system_prompt if agent_def else None,
        permissions=agent_def.permissions if agent_def else [],
        # Forward the def's tool partition (deep-research: Serper-MCP + fetch +
        # bash + agent) so the worker is restricted to its declared toolset.
        allowed_tools=agent_def.tools if agent_def else None,
        disallowed_tools=agent_def.disallowed_tools if agent_def else None,
        task_type="local_agent",
    )

    result = await executor.spawn(config)
    if not result.success:
        raise InfraFailure(result.error or f"spawn failed for {subagent_type}")

    manager = get_task_manager()
    await _wait_terminal(manager, result.task_id, timeout_s=timeout_s)
    return manager.read_task_output(result.task_id, max_bytes=_READ_MAX_BYTES)


# --------------------------------------------------------------------------- #
# K=3 orchestration (REAL — wiring spawn -> score -> JSONL + REPORT)
# --------------------------------------------------------------------------- #
async def _run_one_task(
    task: TaskRun,
    *,
    spawn_fn: SpawnFn,
    model: str,
    k: int,
    strict: bool,
) -> dict[str, Any]:
    """Run one task K times, mean-score it, and tag failure modes.

    Per ADR §1: per-task score = mean of K; extraction-failure (no parseable
    answer) and infra-failure (spawn error / timeout) are recorded SEPARATELY
    from wrong-answer so a formatting/infra bug is never read as a capability
    gap. A run that raises :class:`InfraFailure` contributes a 0.0 score and
    flags the task as an infra failure.
    """
    scores: list[float] = []
    extraction_failures = 0
    infra_failures = 0
    answers: list[str | None] = []
    latencies: list[float] = []

    for _ in range(k):
        t0 = time.monotonic()
        try:
            transcript = await spawn_fn(task, model)
        except InfraFailure:
            infra_failures += 1
            scores.append(0.0)
            answers.append(None)
            latencies.append(time.monotonic() - t0)
            continue
        latencies.append(time.monotonic() - t0)
        scored = score_run(transcript, task.ground_truth, strict=strict)
        scores.append(scored["score"])
        answers.append(scored["answer"])
        if scored["extraction_failure"]:
            extraction_failures += 1

    mean_score = statistics.mean(scores) if scores else 0.0
    return {
        "task_id": task.task_id,
        "level": task.level,
        "k": k,
        "score": mean_score,
        "scores": scores,
        "answers": answers,
        # A task is flagged when the failure mode dominates its K runs (>= half),
        # so a single transient flake doesn't mislabel an otherwise-scored task.
        "extraction_failure": extraction_failures * 2 >= k,
        "infra_failure": infra_failures * 2 >= k,
        "extraction_failure_runs": extraction_failures,
        "infra_failure_runs": infra_failures,
        "latency_s": statistics.median(latencies) if latencies else None,
        "needs_file": bool(task.attachments),
    }


async def run_subset(
    tasks: Sequence[TaskRun],
    *,
    model: str = DEFAULT_MODEL,
    k: int = DEFAULT_K,
    results_dir: Path | str,
    git_sha: str,
    spawn_fn: SpawnFn | None = None,
    subagent_type: str = "general-purpose",
    strict: bool = True,
    prev_sha: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Path:
    """K=3 orchestration: spawn -> score -> JSONL + REPORT.

    For each task, run ``k`` spawns, mean-score, and tag extraction/infra
    failures distinctly (ADR §1). Writes ``<results_dir>/<git_sha>.jsonl`` and
    ``<results_dir>/REPORT.md`` (per-level acc + Wilson CIs + overall + median
    tokens/latency + extraction/infra rates). Returns the REPORT path.

    ``spawn_fn`` defaults to the real subprocess boundary (:func:`_spawn_agent`
    bound to ``subagent_type`` + ``model``); the unit tests inject a
    canned-transcript stand-in so the orchestration is exercised offline.
    """
    if spawn_fn is None:

        async def spawn_fn(task: TaskRun, model: str) -> str:  # noqa: F811
            return await _spawn_agent(
                task, model, subagent_type=subagent_type, timeout_s=timeout_s
            )

    rows: list[dict[str, Any]] = []
    for task in tasks:
        row = await _run_one_task(
            task, spawn_fn=spawn_fn, model=model, k=k, strict=strict
        )
        rows.append(row)

    results_dir = Path(results_dir)
    write_jsonl(results_dir / f"{git_sha}.jsonl", rows)
    return write_report(
        results_dir / "REPORT.md", rows, git_sha=git_sha, prev_sha=prev_sha
    )


# --------------------------------------------------------------------------- #
# CLI: python -m tests.eval.gaia.run_subset --split {dev,gate} --agent ... --k 3
# --------------------------------------------------------------------------- #
def _load_manifest_tasks(
    *,
    split: str,
    snapshot_root: Path,
    work_root: Path,
    manifest_dir: Path,
) -> list[TaskRun]:
    """Resolve a split manifest -> per-task dirs (with attachments copied in).

    Reads ``<manifest_dir>/<split>.yaml`` for the fixed ``task_id`` set, joins
    it against the loaded GAIA validation tasks, and prepares a per-task working
    dir for each. HF-gated (the loader needs the dataset) — not exercised by the
    offline tests.
    """
    import yaml  # noqa: PLC0415

    from tests.eval.gaia.loader import load_validation_tasks  # noqa: PLC0415

    manifest = yaml.safe_load((manifest_dir / f"{split}.yaml").read_text())
    wanted_ids = [row["task_id"] for row in manifest.get("tasks", [])]
    by_id = {t.task_id: t for t in load_validation_tasks(snapshot_root)}

    runs: list[TaskRun] = []
    for task_id in wanted_ids:
        if task_id == "TODO" or task_id not in by_id:
            continue  # placeholder rows / not-yet-frozen manifests.
        t = by_id[task_id]
        runs.append(
            prepare_task_dir(
                task_id=t.task_id,
                question=t.question,
                ground_truth=t.answer,
                level=t.level,
                source_file=t.file_path,
                work_root=work_root,
            )
        )
    return runs


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """Run the GAIA subset end-to-end (HF_TOKEN + live model required).

    Example::

        python -m tests.eval.gaia.run_subset --split gate \\
            --agent general-purpose --k 3
    """
    parser = argparse.ArgumentParser(description="Run a GAIA subset + score it.")
    parser.add_argument("--split", choices=["dev", "gate"], default="dev")
    parser.add_argument(
        "--agent",
        default="general-purpose",
        help="subagent_type to spawn (M0: the current general worker; "
        "'deep-research' ships in PR3/M1).",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="PINNED eval model.")
    parser.add_argument(
        "--snapshot", default=None, help="GAIA snapshot root (else download)."
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parent / "results"),
    )
    parser.add_argument("--git-sha", default="working-tree")
    parser.add_argument("--prev-sha", default=None)
    parser.add_argument(
        "--work-root",
        default=None,
        help="Where to make per-task dirs (default: a temp dir).",
    )
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)

    from tests.eval.gaia.loader import download_gaia_snapshot  # noqa: PLC0415

    snapshot_root = (
        Path(args.snapshot) if args.snapshot else download_gaia_snapshot()
    )
    if args.work_root:
        work_root = Path(args.work_root)
    else:
        import tempfile  # noqa: PLC0415

        work_root = Path(tempfile.mkdtemp(prefix="gaia-eval-"))

    manifest_dir = Path(__file__).resolve().parent
    tasks = _load_manifest_tasks(
        split=args.split,
        snapshot_root=snapshot_root,
        work_root=work_root,
        manifest_dir=manifest_dir,
    )
    if not tasks:
        print(
            f"No runnable tasks for split={args.split}: the manifest still has "
            "TODO placeholders. Run build_manifests.py to freeze real task_ids."
        )
        return 1

    report = asyncio.run(
        run_subset(
            tasks,
            model=args.model,
            k=args.k,
            results_dir=args.results_dir,
            git_sha=args.git_sha,
            subagent_type=args.agent,
            prev_sha=args.prev_sha,
            timeout_s=args.timeout_s,
        )
    )
    print(f"wrote {report}")
    return 0


# Bare sys.exit(main()) — entry-point friendly (ADR project conventions).
if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
