"""Cross-backend memory-recall sweep, report, and go/no-go gate."""

from __future__ import annotations

import argparse
import asyncio
import inspect
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import cast

from openharness.api.resolver import ApiClientResolutionError, resolve_api_client_from_settings
from openharness.config import load_settings

from ohmo.evals.memory.benchmark import MemoryCase, load_cases
from ohmo.evals.memory.provisioning import BackendKind, provision_backend
from ohmo.evals.memory.recall_judge import (
    CaseScore,
    CompleterLike,
    make_live_completer,
    score_case,
)

_SUPPORTED_KINDS = ("file", "catalog", "shadow")
_GATE_TOLERANCE = 1e-9


@dataclass
class BackendSummary:
    """Aggregate metrics for one backend in a sweep."""

    kind: str
    n_cases: int
    mean_score: float
    recall_rate: float
    use_rate: float
    grounded_rate: float
    per_category: dict[str, float]
    mean_latency_ms: float


@dataclass
class SweepResult:
    """Aggregates and raw case scores produced by a cross-backend sweep."""

    by_kind: dict[str, BackendSummary]
    baseline_kind: str = "file"
    raw_scores: dict[str, list[CaseScore]] = field(default_factory=dict)

    @property
    def case_scores(self) -> dict[str, list[CaseScore]]:
        """Compatibility-friendly descriptive alias for ``raw_scores``."""
        return self.raw_scores


@dataclass
class GateResult:
    """Machine-readable outcome of comparing candidates with the baseline."""

    passed: bool
    reasons: list[str]
    deltas: dict[str, float]


async def run_sweep(
    cases: Sequence[MemoryCase],
    kinds: Sequence[str],
    *,
    complete: CompleterLike,
    honcho_base_url: str | None = None,
    honcho_admin_jwt: str | None = None,
    run: str = "p4",
    samples: int = 1,
) -> SweepResult:
    """Run every case/sample against a newly provisioned instance of each backend."""
    normalized_kinds = _validate_sweep_inputs(
        cases,
        kinds,
        honcho_base_url=honcho_base_url,
        honcho_admin_jwt=honcho_admin_jwt,
        samples=samples,
    )
    raw_scores: dict[str, list[CaseScore]] = {kind: [] for kind in normalized_kinds}
    latencies: dict[str, list[float]] = {kind: [] for kind in normalized_kinds}

    for kind in normalized_kinds:
        for case in cases:
            for sample in range(samples):
                started = perf_counter()
                provisioned = await provision_backend(
                    cast(BackendKind, kind),
                    run=run,
                    case=case.id,
                    sample=sample,
                    seed_entries=case.seed_entries,
                    honcho_base_url=honcho_base_url,
                    honcho_admin_jwt=honcho_admin_jwt,
                )
                provision_latency_ms = (perf_counter() - started) * 1_000
                try:
                    raw_scores[kind].append(
                        await score_case(case, provisioned, complete=complete)
                    )
                    latencies[kind].append(provision_latency_ms)
                finally:
                    await provisioned.teardown()

    return SweepResult(
        by_kind={
            kind: _summarize(kind, raw_scores[kind], latencies[kind])
            for kind in normalized_kinds
        },
        raw_scores=raw_scores,
    )


def evaluate_gate(sweep: SweepResult) -> GateResult:
    """Compare every candidate with the baseline's score and grounding rate."""
    baseline = sweep.by_kind.get(sweep.baseline_kind)
    if baseline is None:
        return GateResult(
            passed=False,
            reasons=[f"baseline backend {sweep.baseline_kind!r} is missing"],
            deltas={},
        )

    reasons: list[str] = []
    deltas: dict[str, float] = {}
    for kind, candidate in sweep.by_kind.items():
        if kind == sweep.baseline_kind:
            continue

        score_delta = candidate.mean_score - baseline.mean_score
        grounding_delta = candidate.grounded_rate - baseline.grounded_rate
        deltas[f"{kind}.mean_score"] = score_delta
        deltas[f"{kind}.grounded_rate"] = grounding_delta

        if score_delta < -_GATE_TOLERANCE:
            reasons.append(
                f"{kind} recall score is worse than {sweep.baseline_kind}: "
                f"{candidate.mean_score:.4f} < {baseline.mean_score:.4f} "
                f"(delta {score_delta:+.4f})"
            )
        if grounding_delta < -_GATE_TOLERANCE:
            reasons.append(
                f"{kind} grounding regressed versus {sweep.baseline_kind}: "
                f"{candidate.grounded_rate:.4f} < {baseline.grounded_rate:.4f} "
                f"(delta {grounding_delta:+.4f})"
            )

    return GateResult(passed=not reasons, reasons=reasons, deltas=deltas)


def render_report(sweep: SweepResult, gate: GateResult) -> str:
    """Render a human-readable Markdown summary and gate verdict."""
    lines = [
        "# Memory Backend A/B Report",
        "",
        "## Backend summary",
        "",
        "| Backend | Cases | Mean score | Recall | Use | Grounded | Mean latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, summary in sweep.by_kind.items():
        lines.append(
            f"| {_cell(kind)} | {summary.n_cases} | {summary.mean_score:.4f} | "
            f"{summary.recall_rate:.4f} | {summary.use_rate:.4f} | "
            f"{summary.grounded_rate:.4f} | {summary.mean_latency_ms:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-category mean score",
            "",
            "| Category | " + " | ".join(_cell(kind) for kind in sweep.by_kind) + " |",
            "|---|" + "---:|" * len(sweep.by_kind),
        ]
    )
    categories = sorted(
        {
            category
            for summary in sweep.by_kind.values()
            for category in summary.per_category
        }
    )
    if categories:
        for category in categories:
            values = [
                _format_optional(sweep.by_kind[kind].per_category.get(category))
                for kind in sweep.by_kind
            ]
            lines.append(f"| {_cell(category)} | " + " | ".join(values) + " |")
    else:
        lines.append("| _(no categories)_ | " + " | ".join("—" for _ in sweep.by_kind) + " |")

    lines.extend(["", "## Gate", "", f"GATE: {'PASS' if gate.passed else 'FAIL'}", ""])
    if gate.reasons:
        lines.append("Reasons:")
        lines.extend(f"- {reason}" for reason in gate.reasons)
    else:
        lines.append(
            f"All candidate backends meet or exceed the `{sweep.baseline_kind}` baseline "
            "without a grounding regression."
        )

    if gate.deltas:
        lines.extend(["", f"Deltas versus `{sweep.baseline_kind}`:", ""])
        for name, delta in gate.deltas.items():
            lines.append(f"- `{name}`: {delta:+.4f}")

    return "\n".join(lines) + "\n"


def _validate_sweep_inputs(
    cases: Sequence[MemoryCase],
    kinds: Sequence[str],
    *,
    honcho_base_url: str | None,
    honcho_admin_jwt: str | None,
    samples: int,
) -> list[str]:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not cases:
        raise ValueError("at least one memory case is required")
    if not kinds:
        raise ValueError("at least one backend kind is required")

    normalized = [str(kind).strip().casefold() for kind in kinds]
    unsupported = [kind for kind in normalized if kind not in _SUPPORTED_KINDS]
    if unsupported:
        raise ValueError(f"unsupported eval memory backend: {unsupported[0]!r}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("backend kinds must not contain duplicates")
    if "shadow" in normalized:
        if not honcho_base_url or not honcho_base_url.strip():
            raise ValueError("shadow eval backend requires explicit honcho_base_url")
        if not honcho_admin_jwt:
            raise ValueError("shadow eval backend requires explicit honcho_admin_jwt")
    return normalized


def _summarize(kind: str, scores: list[CaseScore], latencies: list[float]) -> BackendSummary:
    category_scores: defaultdict[str, list[float]] = defaultdict(list)
    for score in scores:
        category_scores[score.category].append(score.mean_score)
    return BackendSummary(
        kind=kind,
        n_cases=len(scores),
        mean_score=_mean([score.mean_score for score in scores]),
        recall_rate=_mean([score.recall_rate for score in scores]),
        use_rate=_mean([score.use_rate for score in scores]),
        grounded_rate=_mean([score.grounded_rate for score in scores]),
        per_category={
            category: _mean(values) for category, values in sorted(category_scores.items())
        },
        mean_latency_ms=_mean(latencies),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _format_optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Path to memory benchmark JSONL cases.")
    parser.add_argument(
        "--kinds",
        required=True,
        help="Comma-separated backends: file,catalog[,shadow].",
    )
    parser.add_argument("--honcho-base-url", help="Honcho URL for shadow eval provisioning.")
    parser.add_argument(
        "--honcho-admin-jwt",
        help="Explicit Honcho admin JWT for temporary shadow workspaces.",
    )
    parser.add_argument("--samples", type=int, default=1, help="Samples per backend/case.")
    parser.add_argument("--out", help="Optional path for the Markdown report.")
    parser.add_argument("--model", help="Optional completion model override.")
    parser.add_argument("--profile", help="Optional configured provider profile.")
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
    settings = load_settings().merge_cli_overrides(
        model=args.model,
        active_profile=args.profile,
    )
    settings = settings.materialize_active_profile()
    try:
        api_client = resolve_api_client_from_settings(settings)
    except (ApiClientResolutionError, SystemExit) as error:
        raise ValueError("memory A/B report requires configured API authentication") from error

    try:
        sweep = await run_sweep(
            load_cases(args.cases),
            kinds,
            complete=make_live_completer(api_client, str(settings.model)),
            honcho_base_url=args.honcho_base_url,
            honcho_admin_jwt=args.honcho_admin_jwt,
            samples=args.samples,
        )
    finally:
        close = getattr(api_client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    gate = evaluate_gate(sweep)
    report = render_report(sweep, gate)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if gate.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report CLI and return its process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run_cli(args))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BackendSummary",
    "GateResult",
    "SweepResult",
    "evaluate_gate",
    "main",
    "render_report",
    "run_sweep",
]
