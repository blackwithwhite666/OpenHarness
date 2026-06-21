"""Runnable eval pack assembly and report-only smoke runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from openharness.evals.models import (
    EvalGoldCase,
    EvalRunPack,
    EvalRunPackCase,
    EvalSmokeReport,
    EvalSmokeReportCase,
)
from openharness.evals.review import read_gold_cases
from openharness.evals.store import EvalStore
from openharness.utils.fs import atomic_write_text


@dataclass(frozen=True)
class EvalRunPackWrite:
    """Summary returned after writing a runnable eval pack."""

    pack: EvalRunPack
    path: Path
    relative_path: str


@dataclass(frozen=True)
class EvalSmokeReportWrite:
    """Summary returned after writing a smoke/report-only eval report."""

    report: EvalSmokeReport
    path: Path
    relative_path: str


def build_run_pack(
    store: EvalStore,
    *,
    case_ids: list[str] | None = None,
    gold_records_filename: str = "gold_cases.jsonl",
) -> EvalRunPack:
    """Build a runnable metadata-only pack from reviewed gold cases."""
    gold_cases = read_gold_cases(store, records_filename=gold_records_filename)
    _ensure_unique_gold_case_ids(gold_cases)
    selected = _select_gold_cases(gold_cases, case_ids)
    if not selected:
        raise ValueError("no gold cases available for eval pack")

    cases = [_pack_case(gold) for gold in selected]
    pack_id = _stable_id("pack", _canonical_json([case.model_dump(mode="json") for case in cases]))
    return EvalRunPack(
        pack_id=pack_id,
        source_records_path=f"cases/{gold_records_filename}",
        cases=cases,
        metadata={
            "privacy": "metadata_only",
            "case_count": len(cases),
        },
    )


def write_run_pack(
    store: EvalStore,
    pack: EvalRunPack | None = None,
    *,
    case_ids: list[str] | None = None,
    pack_filename: str = "eval_pack.json",
) -> EvalRunPackWrite:
    """Write a runnable eval pack under ``store.root/packs``."""
    payload = pack or build_run_pack(store, case_ids=case_ids)
    path = _pack_output_path(store, pack_filename)
    atomic_write_text(path, payload.model_dump_json(indent=2) + "\n")
    return EvalRunPackWrite(
        pack=payload,
        path=path,
        relative_path=path.relative_to(store.root).as_posix(),
    )


def read_run_pack(
    store: EvalStore,
    *,
    pack_filename: str = "eval_pack.json",
) -> EvalRunPack:
    """Read a runnable eval pack from ``store.root/packs``."""
    path = _pack_output_path(store, pack_filename)
    try:
        return EvalRunPack.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(f"invalid {path.name}") from exc


def run_smoke_report(
    store: EvalStore,
    *,
    pack: EvalRunPack | None = None,
    pack_filename: str = "eval_pack.json",
    report_filename: str = "smoke_report.json",
) -> EvalSmokeReportWrite:
    """Run metadata-only smoke checks for a runnable eval pack."""
    payload = pack or read_run_pack(store, pack_filename=pack_filename)
    if not payload.cases:
        raise ValueError("eval pack must contain cases")
    report_cases = [_smoke_case(case) for case in payload.cases]
    passed_count = sum(1 for case in report_cases if case.status == "passed")
    failed_count = len(report_cases) - passed_count
    report = EvalSmokeReport(
        report_id=_stable_id("smoke", payload.pack_id, *(case.case_id for case in payload.cases)),
        pack_id=payload.pack_id,
        case_count=len(report_cases),
        passed_count=passed_count,
        failed_count=failed_count,
        cases=report_cases,
        metadata={
            "privacy": "metadata_only",
            "mode": "smoke_report_only",
            "pack_case_count": len(payload.cases),
        },
    )
    path = _report_output_path(store, report_filename)
    atomic_write_text(path, report.model_dump_json(indent=2) + "\n")
    return EvalSmokeReportWrite(
        report=report,
        path=path,
        relative_path=path.relative_to(store.root).as_posix(),
    )


def _pack_case(gold: EvalGoldCase) -> EvalRunPackCase:
    metadata = {
        "review_status": gold.review_status,
        "source_review_status": gold.metadata.get("source_review_status", ""),
        "candidate_score": gold.metadata.get("candidate_score", 0),
        "signals": gold.metadata.get("signals", []),
        "event_count": gold.metadata.get("event_count", 0),
    }
    if "state_delta" in gold.metadata:
        metadata["state_delta"] = gold.metadata.get("state_delta")
    return EvalRunPackCase(
        gold_case_id=gold.gold_case_id,
        case_id=gold.case_id,
        episode_id=gold.episode_id,
        case_kind=gold.case_kind,
        input_facet_ids=gold.input_facet_ids,
        expected_facet_ids=gold.expected_facet_ids,
        tool_names=gold.tool_names,
        capability_path=gold.capability_path,
        rubric=gold.rubric,
        scorer=gold.scorer,
        metadata=metadata,
    )


def _smoke_case(case: EvalRunPackCase) -> EvalSmokeReportCase:
    checks = {
        "has_gold_case_id": bool(case.gold_case_id),
        "has_case_id": bool(case.case_id),
        "has_input_facets": bool(case.input_facet_ids),
        "has_expected_facets": bool(case.expected_facet_ids),
        "has_rubric": bool(case.rubric),
    }
    warnings = [name for name, passed in checks.items() if not passed]
    return EvalSmokeReportCase(
        gold_case_id=case.gold_case_id,
        case_id=case.case_id,
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
        warnings=warnings,
        metadata={
            "case_kind": case.case_kind,
            "tool_count": len(case.tool_names),
        },
    )


def _select_gold_cases(
    gold_cases: list[EvalGoldCase],
    case_ids: list[str] | None,
) -> list[EvalGoldCase]:
    if case_ids is None:
        return sorted(gold_cases, key=lambda gold: (gold.episode_id, gold.case_id))
    requested = list(dict.fromkeys(case_ids))
    if not requested:
        raise ValueError("case_ids must not be empty")
    by_id = {gold.case_id: gold for gold in gold_cases}
    missing = [case_id for case_id in requested if case_id not in by_id]
    if missing:
        raise ValueError(f"gold case not found: {', '.join(missing)}")
    return [by_id[case_id] for case_id in requested]


def _ensure_unique_gold_case_ids(gold_cases: list[EvalGoldCase]) -> None:
    seen_case_ids: set[str] = set()
    seen_gold_case_ids: set[str] = set()
    duplicate_case_ids: list[str] = []
    duplicate_gold_case_ids: list[str] = []
    for gold in gold_cases:
        if gold.case_id in seen_case_ids:
            duplicate_case_ids.append(gold.case_id)
        if gold.gold_case_id in seen_gold_case_ids:
            duplicate_gold_case_ids.append(gold.gold_case_id)
        seen_case_ids.add(gold.case_id)
        seen_gold_case_ids.add(gold.gold_case_id)
    if duplicate_case_ids:
        raise ValueError(f"duplicate gold case ids: {', '.join(sorted(set(duplicate_case_ids)))}")
    if duplicate_gold_case_ids:
        raise ValueError(
            "duplicate gold case record ids: "
            + ", ".join(sorted(set(duplicate_gold_case_ids)))
        )


def _pack_output_path(store: EvalStore, filename: str) -> Path:
    output_dir = store.root / "packs"
    output_path = (output_dir / filename).resolve()
    if not _is_relative_to(output_path, output_dir.resolve()):
        raise ValueError("pack output filename must stay under store.root/packs")
    return output_path


def _report_output_path(store: EvalStore, filename: str) -> Path:
    output_dir = store.root / "reports"
    output_path = (output_dir / filename).resolve()
    if not _is_relative_to(output_path, output_dir.resolve()):
        raise ValueError("report output filename must stay under store.root/reports")
    return output_path


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\n".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:24]}"


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True
