"""Ohmo helpers for runnable eval packs and smoke reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openharness.evals import (
    EvalRunPack,
    EvalRunPackWrite,
    EvalSmokeReportWrite,
    read_gold_cases,
    read_run_pack,
    run_smoke_report,
    write_run_pack,
)

from ohmo.evals.adapter import get_eval_store


@dataclass(frozen=True)
class OhmoEvalPackResult:
    """Summary returned after building a runnable Ohmo eval pack."""

    write: EvalRunPackWrite
    case_count: int
    gold_case_count: int


@dataclass(frozen=True)
class OhmoEvalSmokeResult:
    """Summary returned after running an Ohmo eval smoke report."""

    write: EvalSmokeReportWrite
    report_only: bool


def build_ohmo_eval_pack(
    *,
    workspace: str | Path | None = None,
    pack_filename: str = "eval_pack.json",
) -> OhmoEvalPackResult:
    """Build a runnable eval pack from reviewed Ohmo gold cases."""
    store = get_eval_store(workspace)
    gold_case_count = len(read_gold_cases(store))
    write = write_run_pack(store, pack_filename=pack_filename)
    return OhmoEvalPackResult(
        write=write,
        case_count=len(write.pack.cases),
        gold_case_count=gold_case_count,
    )


def run_ohmo_eval_smoke(
    *,
    workspace: str | Path | None = None,
    pack_filename: str = "eval_pack.json",
    report_filename: str = "smoke_report.json",
    limit: int | None = None,
    report_only: bool = False,
) -> OhmoEvalSmokeResult:
    """Run smoke/report-only validation over an Ohmo eval pack."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    store = get_eval_store(workspace)
    pack = read_run_pack(store, pack_filename=pack_filename)
    if limit is not None:
        pack = _limited_pack(pack, limit)
    write = run_smoke_report(store, pack=pack, report_filename=report_filename)
    return OhmoEvalSmokeResult(write=write, report_only=report_only)


def _limited_pack(pack: EvalRunPack, limit: int) -> EvalRunPack:
    metadata = dict(pack.metadata)
    metadata["limit"] = limit
    metadata["source_case_count"] = len(pack.cases)
    return pack.model_copy(update={"cases": pack.cases[:limit], "metadata": metadata})
