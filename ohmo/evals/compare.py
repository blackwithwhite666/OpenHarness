"""Ohmo helpers for execution report comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openharness.evals import (
    EvalReportComparisonWrite,
    read_execution_report,
    write_execution_report_comparison,
)
from openharness.utils.fs import atomic_write_text

from ohmo.evals.adapter import get_eval_store


@dataclass(frozen=True)
class OhmoEvalCompareResult:
    """Summary returned after comparing two Ohmo execution reports."""

    write: EvalReportComparisonWrite
    report_only: bool


@dataclass(frozen=True)
class OhmoEvalBaselineSaveResult:
    """Summary returned after saving an execution report baseline."""

    name: str
    path: Path
    relative_path: str
    report_id: str
    case_count: int
    passed_count: int
    non_passed_count: int


@dataclass(frozen=True)
class OhmoEvalBaselineListItem:
    """Metadata-only row for one saved eval baseline."""

    name: str
    path: Path
    relative_path: str
    report_id: str
    pack_id: str
    case_count: int
    passed_count: int
    failed_count: int
    blocked_count: int
    error_count: int


@dataclass(frozen=True)
class OhmoEvalBaselineListResult:
    """Summary returned after listing saved eval baselines."""

    baselines: list[OhmoEvalBaselineListItem]


def compare_ohmo_eval_reports(
    *,
    workspace: str | Path | None = None,
    baseline_report: str | Path,
    candidate_report: str | Path = "eval_report.json",
    report_filename: str = "eval_compare.json",
    score_tolerance: float = 0.0,
    report_only: bool = False,
) -> OhmoEvalCompareResult:
    """Compare two execution reports under an Ohmo eval workspace."""
    store = get_eval_store(workspace)
    write = write_execution_report_comparison(
        store,
        baseline_report_path=_resolve_report_path(store.root, baseline_report),
        candidate_report_path=_resolve_report_path(store.root, candidate_report),
        report_filename=report_filename,
        score_tolerance=score_tolerance,
    )
    return OhmoEvalCompareResult(write=write, report_only=report_only)


def save_ohmo_eval_baseline(
    *,
    workspace: str | Path | None = None,
    source_report: str | Path = "eval_report.json",
    name: str = "main",
    overwrite: bool = False,
) -> OhmoEvalBaselineSaveResult:
    """Save an execution report under ``evals/reports/baselines``."""
    store = get_eval_store(workspace)
    baseline_name = _normalize_baseline_name(name)
    source_path = _resolve_report_path(store.root, source_report)
    report = read_execution_report(source_path)
    output_path = _baseline_output_path(store.root, baseline_name)
    if output_path.exists() and not overwrite:
        raise ValueError(
            f"eval baseline already exists: {baseline_name}. Use --overwrite to replace it."
        )
    atomic_write_text(output_path, report.model_dump_json(indent=2) + "\n")
    return OhmoEvalBaselineSaveResult(
        name=baseline_name,
        path=output_path,
        relative_path=output_path.relative_to(store.root).as_posix(),
        report_id=report.report_id,
        case_count=report.case_count,
        passed_count=report.passed_count,
        non_passed_count=report.failed_count + report.blocked_count + report.error_count,
    )


def list_ohmo_eval_baselines(
    *,
    workspace: str | Path | None = None,
) -> OhmoEvalBaselineListResult:
    """List saved execution report baselines."""
    store = get_eval_store(workspace)
    baseline_dir = _baseline_dir(store.root)
    if not baseline_dir.exists():
        return OhmoEvalBaselineListResult(baselines=[])
    baselines: list[OhmoEvalBaselineListItem] = []
    for path in sorted(baseline_dir.glob("*.json")):
        resolved_path = path.resolve()
        report = read_execution_report(resolved_path)
        baselines.append(
            OhmoEvalBaselineListItem(
                name=path.stem,
                path=resolved_path,
                relative_path=resolved_path.relative_to(store.root).as_posix(),
                report_id=report.report_id,
                pack_id=report.pack_id,
                case_count=report.case_count,
                passed_count=report.passed_count,
                failed_count=report.failed_count,
                blocked_count=report.blocked_count,
                error_count=report.error_count,
            )
        )
    return OhmoEvalBaselineListResult(baselines=baselines)


def _resolve_report_path(store_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = store_root / "reports" / path
    return path.resolve()


def _baseline_output_path(store_root: Path, name: str) -> Path:
    baseline_dir = _baseline_dir(store_root)
    output_path = (baseline_dir / f"{name}.json").resolve()
    if not _is_relative_to(output_path, baseline_dir.resolve()):
        raise ValueError("baseline name must stay under store.root/reports/baselines")
    return output_path


def _baseline_dir(store_root: Path) -> Path:
    path = store_root / "reports" / "baselines"
    if path.is_symlink():
        raise ValueError("eval baseline directory must not be a symlink")
    return path


def _normalize_baseline_name(name: str) -> str:
    normalized = name.strip()
    if normalized.endswith(".json"):
        normalized = normalized[:-5]
    if not normalized:
        raise ValueError("baseline name must not be empty")
    if len(normalized) > 96 or any(not _is_safe_baseline_char(char) for char in normalized):
        raise ValueError(
            "baseline name may contain only letters, numbers, '.', '_', and '-'"
        )
    return normalized


def _is_safe_baseline_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in {".", "_", "-"})


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True
