"""Metadata replay runner for runnable eval packs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openharness.evals.facets import collect_text_facets
from openharness.evals.models import (
    EvalEpisode,
    EvalEvent,
    EvalReplayContext,
    EvalReplayReport,
    EvalReplayReportCase,
    EvalRunPack,
    EvalRunPackCase,
    EvalTextFacet,
)
from openharness.evals.pack import read_run_pack
from openharness.evals.store import EvalStore
from openharness.utils.fs import atomic_write_text


@dataclass(frozen=True)
class EvalReplayReportWrite:
    """Summary returned after writing a replay-runner eval report."""

    report: EvalReplayReport
    path: Path
    relative_path: str


def run_replay_report(
    store: EvalStore,
    *,
    pack: EvalRunPack | None = None,
    pack_filename: str = "eval_pack.json",
    report_filename: str = "eval_report.json",
    limit: int | None = None,
) -> EvalReplayReportWrite:
    """Run metadata-only replay checks over a runnable eval pack."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    payload = pack or read_run_pack(store, pack_filename=pack_filename)
    cases = payload.cases[:limit] if limit is not None else payload.cases
    if not cases:
        raise ValueError("eval pack must contain cases")

    facets_by_id = {item.facet.facet_id: item.facet for item in collect_text_facets(store)}
    report_cases = [_run_case(store, case, facets_by_id) for case in cases]
    passed_count = sum(1 for case in report_cases if case.status == "passed")
    failed_count = len(report_cases) - passed_count
    report = EvalReplayReport(
        report_id=_stable_id("eval", payload.pack_id, *(case.case_id for case in cases)),
        pack_id=payload.pack_id,
        case_count=len(report_cases),
        passed_count=passed_count,
        failed_count=failed_count,
        cases=report_cases,
        metadata={
            "privacy": "metadata_only",
            "mode": "metadata_replay",
            "pack_case_count": len(payload.cases),
            "limit": limit or 0,
        },
    )
    path = _report_output_path(store, report_filename)
    atomic_write_text(path, report.model_dump_json(indent=2) + "\n")
    return EvalReplayReportWrite(
        report=report,
        path=path,
        relative_path=path.relative_to(store.root).as_posix(),
    )


def _run_case(
    store: EvalStore,
    case: EvalRunPackCase,
    facets_by_id: dict[str, EvalTextFacet],
) -> EvalReplayReportCase:
    episode = store.get_episode(case.episode_id)
    events = list(store.iter_events(case.episode_id)) if episode is not None else []
    tool_path = _tool_path(events)
    input_facets = [facets_by_id.get(facet_id) for facet_id in case.input_facet_ids]
    expected_facets = [facets_by_id.get(facet_id) for facet_id in case.expected_facet_ids]
    input_facet_kinds = [facet.facet_kind for facet in input_facets if facet is not None]
    expected_facet_kinds = [facet.facet_kind for facet in expected_facets if facet is not None]

    checks = {
        "episode_exists": episode is not None,
        "has_events": bool(events),
        "input_facets_resolve": bool(case.input_facet_ids)
        and all(facet is not None for facet in input_facets),
        "expected_facets_resolve": bool(case.expected_facet_ids)
        and all(facet is not None for facet in expected_facets),
        "tool_refs_observed": all(tool_name in tool_path for tool_name in case.tool_names),
        "has_rubric": bool(case.rubric),
    }
    warnings = [name for name, passed in checks.items() if not passed]
    context = None
    if episode is not None:
        context = _metadata_context(
            episode=episode,
            events=events,
            tool_path=tool_path,
            input_facet_kinds=input_facet_kinds,
            expected_facet_kinds=expected_facet_kinds,
            case=case,
        )

    return EvalReplayReportCase(
        gold_case_id=case.gold_case_id,
        case_id=case.case_id,
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
        warnings=warnings,
        context=context,
        metadata={
            "case_kind": case.case_kind,
            "tool_count": len(case.tool_names),
            "rubric_count": len(case.rubric),
        },
    )


def _metadata_context(
    *,
    episode: EvalEpisode,
    events: Sequence[EvalEvent],
    tool_path: Sequence[str],
    input_facet_kinds: Sequence[str],
    expected_facet_kinds: Sequence[str],
    case: EvalRunPackCase,
) -> EvalReplayContext:
    return EvalReplayContext(
        episode_id=episode.episode_id,
        source=episode.source,
        app=episode.app,
        status=episode.status,
        privacy=episode.privacy,
        event_kind_path=[event.kind for event in events],
        tool_path=list(tool_path),
        event_count=len(events),
        error_count=sum(1 for event in events if event.is_error),
        input_facet_kinds=list(input_facet_kinds),
        expected_facet_kinds=list(expected_facet_kinds),
        metadata={
            "case_kind": case.case_kind,
            "session_id_present": bool(episode.session_id),
        },
    )


def _tool_path(events: Sequence[EvalEvent]) -> list[str]:
    path: list[str] = []
    seen_calls: set[tuple[str, str]] = set()
    started_calls: set[tuple[str, str]] = set()
    for index, event in enumerate(events):
        if not event.tool_name:
            continue
        call_key = (
            event.tool_name,
            event.tool_call_id or f"event-{index}",
        )
        if event.kind == "tool_started":
            started_calls.add(call_key)
        elif event.tool_call_id and call_key in started_calls:
            continue
        if call_key in seen_calls:
            continue
        seen_calls.add(call_key)
        path.append(event.tool_name)
    return path


def _report_output_path(store: EvalStore, filename: str) -> Path:
    output_dir = store.root / "reports"
    output_path = (output_dir / filename).resolve()
    if not _is_relative_to(output_path, output_dir.resolve()):
        raise ValueError("eval report filename must stay under store.root/reports")
    return output_path


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\n".join(parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(raw).hexdigest()[:24]}"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return False
    return True
