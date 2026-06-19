"""Executor-based eval runner for runnable eval packs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from openharness.evals.facets import EvalTextFacetInput, collect_text_facets
from openharness.evals.executor import (
    EvalExecutionContext,
    EvalExecutor,
    EvalExecutorResult,
    EvalToolFixture,
    ReplayToolsExecutor,
)
from openharness.evals.models import (
    EvalEpisode,
    EvalEvent,
    EvalExecutionReport,
    EvalExecutionReportCase,
    EvalObservedToolCall,
    EvalObservedTrace,
    EvalReplayContext,
    EvalResourceSnapshot,
    EvalRunPack,
    EvalRunPackCase,
)
from openharness.evals.pack import read_run_pack
from openharness.evals.store import EvalStore
from openharness.utils.fs import atomic_write_text

_EXECUTION_SCORE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvalExecutionReportWrite:
    """Summary returned after writing an executor-based eval report."""

    report: EvalExecutionReport
    path: Path
    relative_path: str


@dataclass(frozen=True)
class EvalExecutionScorerResult:
    """Metadata-only scoring outcome for one eval execution."""

    passed: bool
    score: float
    scorer_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalExecutionScorer(Protocol):
    """Scores transient executor output against an eval execution context."""

    name: str

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        """Return a metadata-only score result."""


class ExactMatchEvalScorer:
    """Default deterministic scorer for replay-style evals."""

    name = "exact-final-text"

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        passed = _texts_match(
            executor_result.final_text,
            context.expected_final_text,
        )
        return EvalExecutionScorerResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            scorer_name=self.name,
            metadata={
                "expected_text_length": len(_normalize_text(context.expected_final_text)),
                "observed_text_length": len(_normalize_text(executor_result.final_text)),
            },
        )


def run_execution_report(
    store: EvalStore,
    *,
    executor: EvalExecutor | None = None,
    scorer: EvalExecutionScorer | None = None,
    pack: EvalRunPack | None = None,
    pack_filename: str = "eval_pack.json",
    report_filename: str = "eval_report.json",
    limit: int | None = None,
) -> EvalExecutionReportWrite:
    """Run executor-based checks over a runnable eval pack."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    selected_executor = executor or ReplayToolsExecutor()
    selected_scorer = scorer or ExactMatchEvalScorer()
    payload = pack or read_run_pack(store, pack_filename=pack_filename)
    cases = payload.cases[:limit] if limit is not None else payload.cases
    if not cases:
        raise ValueError("eval pack must contain cases")

    facet_inputs_by_id = {
        item.facet.facet_id: item for item in collect_text_facets(store)
    }
    report_cases = [
        _execute_case(
            store,
            payload,
            case,
            facet_inputs_by_id,
            selected_executor,
            selected_scorer,
        )
        for case in cases
    ]
    passed_count = sum(1 for case in report_cases if case.status == "passed")
    blocked_count = sum(1 for case in report_cases if case.status == "blocked")
    error_count = sum(1 for case in report_cases if case.status == "error")
    failed_count = sum(1 for case in report_cases if case.status == "failed")
    report = EvalExecutionReport(
        report_id=_stable_id(
            "eval-exec",
            selected_executor.name,
            payload.pack_id,
            *(case.case_id for case in cases),
        ),
        pack_id=payload.pack_id,
        case_count=len(report_cases),
        passed_count=passed_count,
        failed_count=failed_count,
        blocked_count=blocked_count,
        error_count=error_count,
        cases=report_cases,
        metadata={
            "privacy": "metadata_only",
            "mode": "execution_replay",
            "executor_name": selected_executor.name,
            "scorer_name": selected_scorer.name,
            "score_schema_version": _EXECUTION_SCORE_SCHEMA_VERSION,
            "pack_case_count": len(payload.cases),
            "limit": limit or 0,
        },
    )
    path = _report_output_path(store, report_filename)
    atomic_write_text(path, report.model_dump_json(indent=2) + "\n")
    return EvalExecutionReportWrite(
        report=report,
        path=path,
        relative_path=path.relative_to(store.root).as_posix(),
    )


def _execute_case(
    store: EvalStore,
    pack: EvalRunPack,
    case: EvalRunPackCase,
    facet_inputs_by_id: dict[str, EvalTextFacetInput],
    executor: EvalExecutor,
    scorer: EvalExecutionScorer,
) -> EvalExecutionReportCase:
    episode = store.get_episode(case.episode_id)
    events = list(store.iter_events(case.episode_id)) if episode is not None else []
    input_facets = [facet_inputs_by_id.get(facet_id) for facet_id in case.input_facet_ids]
    expected_facets = [
        facet_inputs_by_id.get(facet_id) for facet_id in case.expected_facet_ids
    ]
    resolved_inputs = tuple(facet for facet in input_facets if facet is not None)
    resolved_expected = tuple(facet for facet in expected_facets if facet is not None)
    input_facet_kinds = [item.facet.facet_kind for item in resolved_inputs]
    expected_facet_kinds = [item.facet.facet_kind for item in resolved_expected]
    tool_fixtures = _tool_fixtures(events)
    source_tool_path = [fixture.tool_name for fixture in tool_fixtures]
    resource_snapshot_status = _resource_snapshot_status(store, events)
    checks = {
        "episode_exists": episode is not None,
        "has_events": bool(events),
        "input_facets_resolve": bool(case.input_facet_ids)
        and all(facet is not None for facet in input_facets),
        "expected_facets_resolve": bool(case.expected_facet_ids)
        and all(facet is not None for facet in expected_facets),
        "tool_fixtures_resolve": all(
            tool_name in source_tool_path for tool_name in case.tool_names
        ),
        "tool_trace_complete": _tool_trace_complete(tool_fixtures, case.tool_names),
        "resource_snapshot_valid_or_absent": resource_snapshot_status
        in {"absent", "valid"},
        "has_rubric": bool(case.rubric),
    }
    context = None
    if episode is not None:
        context = _metadata_context(
            episode=episode,
            events=events,
            tool_path=source_tool_path,
            input_facet_kinds=input_facet_kinds,
            expected_facet_kinds=expected_facet_kinds,
            case=case,
        )
    if not all(checks.values()) or episode is None:
        return _blocked_execution_case(
            case,
            checks,
            context,
            resource_snapshot_status=resource_snapshot_status,
            executor_name=executor.name,
        )

    execution_context = EvalExecutionContext(
        store=store,
        pack=pack,
        case=case,
        episode=episode,
        events=tuple(events),
        input_facets=resolved_inputs,
        expected_facets=resolved_expected,
        tool_fixtures=tuple(tool_fixtures),
        primary_prompt=_select_facet_text(resolved_inputs, ("user_goal", "user_request")),
        expected_final_text=_select_facet_text(
            resolved_expected,
            ("assistant_final", "gateway_error", "tool_output"),
        ),
        resource_snapshot_status=resource_snapshot_status,
    )

    try:
        executor_result = executor.run_case(execution_context)
    except Exception as exc:
        return _error_execution_case(
            case=case,
            checks=checks,
            context=context,
            execution_context=execution_context,
            executor_name=executor.name,
            resource_snapshot_status=resource_snapshot_status,
            exc=exc,
        )

    observed_tool_path = [_sanitize_label(value, prefix="tool") for value in executor_result.tool_path]
    scorer_result = scorer.score(
        context=execution_context,
        executor_result=executor_result,
    )
    behavior_checks = {
        "execution_completed": True,
        "tool_sequence_matches": list(executor_result.tool_path) == list(case.tool_names),
        "final_output_matches": scorer_result.passed,
        "privacy_report_metadata_only": True,
    }
    all_checks = {**checks, **behavior_checks}
    observed_trace = _observed_trace(
        executor_name=executor.name,
        case=case,
        context=execution_context,
        event_kind_path=_sanitize_labels(
            executor_result.event_kind_path or ("execution_completed",),
            prefix="event",
        ),
        tool_path=observed_tool_path,
        final_text=executor_result.final_text,
        error_type="",
        error_hash="",
        metadata={
            "tool_calls_source": "replay_fixtures",
            "final_output_match_score": scorer_result.score,
            "scorer_name": scorer_result.scorer_name,
            "scorer_metadata_key_count": len(scorer_result.metadata),
            "executor_metadata_key_count": len(executor_result.metadata),
        },
    )
    return EvalExecutionReportCase(
        gold_case_id=case.gold_case_id,
        case_id=case.case_id,
        status="passed" if all(all_checks.values()) else "failed",
        score=_score(all_checks),
        max_score=1.0,
        checks=all_checks,
        warnings=[name for name, passed in all_checks.items() if not passed],
        context=context,
        observed_trace=observed_trace,
        metadata=_execution_case_metadata(
            case,
            executor_name=executor.name,
            scorer_name=scorer_result.scorer_name,
            resource_snapshot_status=resource_snapshot_status,
        ),
    )


def _error_execution_case(
    *,
    case: EvalRunPackCase,
    checks: dict[str, bool],
    context: EvalReplayContext | None,
    execution_context: EvalExecutionContext,
    executor_name: str,
    resource_snapshot_status: str,
    exc: Exception,
) -> EvalExecutionReportCase:
    error_hash = _hash_text(f"{type(exc).__name__}\n{exc}")
    observed_trace = _observed_trace(
        executor_name=executor_name,
        case=case,
        context=execution_context,
        event_kind_path=("execution_started", "execution_error"),
        tool_path=(),
        final_text="",
        error_type=type(exc).__name__,
        error_hash=error_hash,
        metadata={"tool_calls_source": "replay_fixtures"},
    )
    error_checks = {
        **checks,
        "execution_completed": False,
        "tool_sequence_matches": False,
        "final_output_matches": False,
        "privacy_report_metadata_only": True,
    }
    return EvalExecutionReportCase(
        gold_case_id=case.gold_case_id,
        case_id=case.case_id,
        status="error",
        score=_score(error_checks),
        max_score=1.0,
        checks=error_checks,
        warnings=[name for name, passed in error_checks.items() if not passed],
        context=context,
        observed_trace=observed_trace,
        metadata=_execution_case_metadata(
            case,
            executor_name=executor_name,
            scorer_name="exact-final-text",
            resource_snapshot_status=resource_snapshot_status,
        ),
    )


def _blocked_execution_case(
    case: EvalRunPackCase,
    checks: dict[str, bool],
    context: EvalReplayContext | None,
    *,
    resource_snapshot_status: str,
    executor_name: str,
) -> EvalExecutionReportCase:
    return EvalExecutionReportCase(
        gold_case_id=case.gold_case_id,
        case_id=case.case_id,
        status="blocked",
        score=_score(checks),
        max_score=1.0,
        checks=checks,
        warnings=[name for name, passed in checks.items() if not passed],
        context=context,
        observed_trace=None,
        metadata=_execution_case_metadata(
            case,
            executor_name=executor_name,
            scorer_name="",
            resource_snapshot_status=resource_snapshot_status,
        ),
    )


def _execution_case_metadata(
    case: EvalRunPackCase,
    *,
    executor_name: str,
    scorer_name: str,
    resource_snapshot_status: str,
) -> dict[str, Any]:
    return {
        "case_kind": case.case_kind,
        "tool_count": len(case.tool_names),
        "rubric_count": len(case.rubric),
        "executor_name": executor_name,
        "scorer_name": scorer_name,
        "resource_snapshot_status": resource_snapshot_status,
        "score_schema_version": _EXECUTION_SCORE_SCHEMA_VERSION,
    }


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


def _observed_trace(
    *,
    executor_name: str,
    case: EvalRunPackCase,
    context: EvalExecutionContext,
    event_kind_path: Sequence[str],
    tool_path: Sequence[str],
    final_text: str,
    error_type: str,
    error_hash: str,
    metadata: dict[str, Any],
) -> EvalObservedTrace:
    return EvalObservedTrace(
        trace_id=_stable_id(
            "trace",
            executor_name,
            case.case_id,
            context.episode.episode_id,
            "|".join(event_kind_path),
            "|".join(tool_path),
        ),
        executor_name=executor_name,
        episode_id=context.episode.episode_id,
        event_kind_path=list(event_kind_path),
        tool_path=list(tool_path),
        event_count=len(event_kind_path),
        error_count=1 if error_type else 0,
        tool_calls=[
            EvalObservedToolCall(
                tool_name=fixture.tool_name,
                call_key_hash=fixture.call_key_hash,
                started=fixture.started,
                completed=fixture.completed,
                is_error=fixture.is_error,
                start_event_index=fixture.start_event_index,
                complete_event_index=fixture.complete_event_index,
                input_summary_length=fixture.input_summary_length,
                output_summary_length=fixture.output_summary_length,
                metadata={"source": "replay_fixture"},
            )
            for fixture in context.tool_fixtures
        ],
        final_text_hash=_hash_text(_normalize_text(final_text)) if final_text else "",
        final_text_length=len(_normalize_text(final_text)),
        error_type=error_type,
        error_hash=error_hash,
        metadata=metadata,
    )


def _tool_fixtures(events: Sequence[EvalEvent]) -> list[EvalToolFixture]:
    fixtures: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for index, event in enumerate(events):
        if not event.tool_name:
            continue
        call_key = (event.tool_name, event.tool_call_id or f"event-{index}")
        if call_key not in fixtures:
            fixtures[call_key] = {
                "tool_name": event.tool_name,
                "call_key_hash": _stable_id("toolcall", *call_key),
                "started": False,
                "completed": False,
                "is_error": False,
                "start_event_index": None,
                "complete_event_index": None,
                "input_text": "",
                "output_text": "",
                "input_summary_length": 0,
                "output_summary_length": 0,
            }
            order.append(call_key)
        fixture = fixtures[call_key]
        if event.kind == "tool_started":
            fixture["started"] = True
            fixture["start_event_index"] = index
            fixture["input_text"] = _payload_text(event.payload, ("input", "input_summary"))
            fixture["input_summary_length"] = len(
                _normalize_text(_payload_text(event.payload, ("input_summary",)))
            )
        elif event.kind == "tool_completed":
            fixture["completed"] = True
            fixture["complete_event_index"] = index
            fixture["is_error"] = bool(event.is_error)
            fixture["output_text"] = _payload_text(
                event.payload,
                ("output", "output_summary"),
            )
            fixture["output_summary_length"] = len(
                _normalize_text(_payload_text(event.payload, ("output_summary",)))
            )
        else:
            fixture["is_error"] = bool(fixture["is_error"] or event.is_error)

    return [EvalToolFixture(**fixtures[key]) for key in order]


def _tool_trace_complete(
    fixtures: Sequence[EvalToolFixture],
    expected_tool_names: Sequence[str],
) -> bool:
    if not expected_tool_names:
        return True
    remaining = list(fixtures)
    for tool_name in expected_tool_names:
        match_index = next(
            (
                index
                for index, fixture in enumerate(remaining)
                if fixture.tool_name == tool_name
            ),
            None,
        )
        if match_index is None:
            return False
        fixture = remaining.pop(match_index)
        if not fixture.started or not fixture.completed:
            return False
    return True


def _resource_snapshot_status(store: EvalStore, events: Sequence[EvalEvent]) -> str:
    statuses = [
        _read_resource_snapshot_status(store, event)
        for event in events
        if event.kind == "resource_snapshot"
    ]
    if not statuses:
        return "absent"
    if "corrupt" in statuses:
        return "corrupt"
    if "missing" in statuses:
        return "missing"
    return "valid"


def _read_resource_snapshot_status(store: EvalStore, event: EvalEvent) -> str:
    path_value = event.payload.get("path")
    if not isinstance(path_value, str) or not path_value:
        return "missing"
    path = (store.root / path_value).resolve()
    if not _is_relative_to(path, store.root):
        return "corrupt"
    try:
        snapshot = EvalResourceSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "missing"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError):
        return "corrupt"
    if snapshot.episode_id != event.episode_id:
        return "corrupt"
    if not _resource_paths_are_safe(snapshot):
        return "corrupt"
    return "valid"


def _resource_paths_are_safe(snapshot: EvalResourceSnapshot) -> bool:
    for resource in snapshot.resources:
        if not resource.path:
            continue
        resource_path = Path(resource.path)
        if resource_path.is_absolute():
            return False
        try:
            (Path("resources") / resource_path).resolve().relative_to(
                Path("resources").resolve()
            )
        except ValueError:
            return False
    return True


def _select_facet_text(
    facets: Sequence[EvalTextFacetInput],
    preferred_kinds: Sequence[str],
) -> str:
    for facet_kind in preferred_kinds:
        for item in facets:
            if item.facet.facet_kind == facet_kind:
                return item.text
    return facets[0].text if facets else ""


def _payload_text(payload: dict[str, Any], field_names: Sequence[str]) -> str:
    for field_name in field_names:
        if field_name not in payload:
            continue
        return _text_value(payload[field_name])
    return ""


def _text_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError):
        return str(type(value).__name__)


def _texts_match(observed: str, expected: str) -> bool:
    expected_text = _normalize_text(expected)
    observed_text = _normalize_text(observed)
    return bool(expected_text) and observed_text == expected_text


def _score(checks: dict[str, bool]) -> float:
    if not checks:
        return 0.0
    return sum(1 for passed in checks.values() if passed) / len(checks)


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _sanitize_labels(values: Sequence[str], *, prefix: str) -> list[str]:
    return [_sanitize_label(value, prefix=prefix) for value in values]


def _sanitize_label(value: str, *, prefix: str) -> str:
    text = str(value)
    if text and len(text) <= 96 and all(_is_safe_label_char(char) for char in text):
        return text
    return _stable_id(prefix, text)


def _is_safe_label_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in {"_", "-", ".", ":", "/"})


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _report_output_path(store: EvalStore, filename: str) -> Path:
    output_dir = store.root / "reports"
    if output_dir.is_symlink():
        raise ValueError("eval report directory must not be a symlink")
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
