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
from openharness.evals.replay_matching import _fixture_input_key
from openharness.evals.state import compute_episode_state_delta
from openharness.evals.store import EvalStore
from openharness.evals.tool_labels import effective_tool_label
from openharness.utils.fs import atomic_write_text

_EXECUTION_SCORE_SCHEMA_VERSION = 1
_INCIDENTAL_CAPABILITIES = frozenset(
    {"todo_write", "sleep", "task_get", "task_output", "task_stop", "tool_search"}
)
_CAPABILITY_METADATA_KEYS = (
    "observed_capabilities",
    "expected_capabilities",
    "missing_capabilities",
    "unexpected_capabilities",
)
_STATE_RESOURCE_NAMES = ("reminders", "memory", "todos")


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
    """Scores transient executor output against an eval execution context.

    Scorers may set ``requires_exact_tool_sequence`` to ``False`` to make
    exact tool-path matching advisory; absent external attrs default to True.
    """

    name: str
    requires_exact_tool_sequence: bool

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
    requires_exact_tool_sequence = True

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


class ToolTraceOracleV1:
    """Capability-agnostic trace/policy oracle.

    Scores a run by whether its tool trace is well-formed against the case's
    expected tool set — independent of the final wording or any world state.
    This is the one oracle family that works for *every* capability, including
    read-only / browser cases that have no ``world_after``: it asks "did the
    agent use tools sanely", not "is some store correct". Checks:

    - ``no_unexpected_tools``: every observed tool is in the case's expected set;
    - ``no_tool_errors``: no observed tool call returned an error;
    - ``within_call_budget``: total calls stay within a loop-guard budget;
    - ``used_tools_when_expected``: when the case expects tools, the run made at
      least one call (no answer-without-retrieval / hallucinated grounding).

    Under the scripted runner the observed trace equals the recorded one, so
    this acts as a golden-sanity check (e.g. flags a promoted case whose
    captured turn contained a tool error); it becomes a model-regression gate
    under the query-engine runner.
    """

    name = "tool_trace_oracle_v1"
    requires_exact_tool_sequence = True

    def __init__(self, *, max_calls_factor: int = 2, max_calls_floor: int = 3) -> None:
        self._max_calls_factor = max_calls_factor
        self._max_calls_floor = max_calls_floor

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        expected = list(context.case.tool_names)
        expected_set = set(expected)
        calls = list(executor_result.tool_calls)
        unexpected = sorted(
            {
                call.tool_name
                for call in calls
                if expected_set and call.tool_name not in expected_set
            }
        )
        error_count = sum(1 for call in calls if call.is_error)
        budget = max(len(expected) * self._max_calls_factor, self._max_calls_floor)
        checks = {
            "no_unexpected_tools": not unexpected,
            "no_tool_errors": error_count == 0,
            "within_call_budget": len(calls) <= budget,
            "used_tools_when_expected": (not expected) or bool(calls),
        }
        passed = all(checks.values())
        return EvalExecutionScorerResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            scorer_name=self.name,
            metadata={
                "observed_call_count": len(calls),
                "expected_tool_count": len(expected),
                "unexpected_tool_count": len(unexpected),
                "tool_error_count": error_count,
                "call_budget": budget,
                **{f"check.{name}": value for name, value in checks.items()},
            },
        )


class CapabilityTraceOracleV1:
    """Capability-aware trace/policy oracle for shell-routed tools.

    Scores the effective capability labels observed in a run instead of only
    the raw tool names. This keeps typed-tool behavior unchanged while making
    generic shell tools meaningful for agents that route capabilities through
    command arguments.
    """

    name = "capability_trace_oracle_v1"
    requires_exact_tool_sequence = True

    def __init__(self, *, max_calls_factor: int = 2, max_calls_floor: int = 3) -> None:
        self._max_calls_factor = max_calls_factor
        self._max_calls_floor = max_calls_floor

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        expected = list(context.case.capability_path)
        expected_set = set(expected)
        calls = list(executor_result.tool_calls)
        observed = [
            effective_tool_label(call.tool_name, call.arguments)
            for call in calls
        ]
        observed_set = set(observed)
        unexpected = sorted(
            {
                capability
                for capability in observed
                if expected_set and capability not in expected_set
            }
        )
        missing = sorted(
            {
                capability
                for capability in expected
                if capability not in observed_set
            }
        )
        error_count = sum(1 for call in calls if call.is_error)
        budget = max(len(expected) * self._max_calls_factor, self._max_calls_floor)
        checks = {
            "no_unexpected_capabilities": not unexpected,
            "no_tool_errors": error_count == 0,
            "within_call_budget": len(calls) <= budget,
            "expected_capabilities_present": not missing,
            "used_tools_when_expected": (not expected) or bool(calls),
        }
        passed = all(checks.values())
        return EvalExecutionScorerResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            scorer_name=self.name,
            metadata={
                "observed_call_count": len(calls),
                "expected_capability_count": len(expected),
                "unexpected_capability_count": len(unexpected),
                "missing_capability_count": len(missing),
                "tool_error_count": error_count,
                "call_budget": budget,
                **{f"check.{name}": value for name, value in checks.items()},
                **_capability_metadata(expected=expected, observed=observed),
            },
        )


class CapabilityCoverageOracleV1:
    """Capability-aware coverage oracle that ignores incidental bookkeeping."""

    name = "capability_coverage_oracle_v1"
    requires_exact_tool_sequence = False

    def __init__(
        self,
        *,
        incidental: frozenset[str] = _INCIDENTAL_CAPABILITIES,
    ) -> None:
        self._incidental = incidental

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        expected = list(context.case.capability_path)
        expected_core = [
            capability
            for capability in expected
            if capability not in self._incidental
        ]
        expected_core_set = set(expected_core)
        calls = list(executor_result.tool_calls)
        observed = [
            effective_tool_label(call.tool_name, call.arguments)
            for call in calls
        ]
        observed_core = [
            capability
            for capability in observed
            if capability not in self._incidental
        ]
        observed_core_set = set(observed_core)
        missing = sorted(
            {
                capability
                for capability in expected_core
                if capability not in observed_core_set
            }
        )
        error_count = sum(1 for call in calls if call.is_error)
        checks = {
            "no_tool_errors": error_count == 0,
            "expected_core_capabilities_covered": expected_core_set.issubset(
                observed_core_set
            ),
            "used_tools_when_expected": (not expected_core) or bool(observed_core),
        }
        passed = all(checks.values())
        return EvalExecutionScorerResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            scorer_name=self.name,
            metadata={
                "expected_core_count": len(expected_core),
                "observed_core_count": len(observed_core),
                "missing_core_count": len(missing),
                "tool_error_count": error_count,
                **{f"check.{name}": value for name, value in checks.items()},
                **_capability_metadata(
                    expected=expected_core,
                    observed=observed_core,
                ),
            },
        )


class StateOracleV1:
    """Metadata-only oracle for captured before/after world-state mutations."""

    name = "state_oracle_v1"
    requires_exact_tool_sequence = False

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        del executor_result
        observed = compute_episode_state_delta(
            context.store,
            context.episode.episode_id,
        )
        expected = context.case.metadata.get("state_delta")
        checks = {
            "world_after_captured": observed is not None,
            "state_changed": observed is not None and observed.get("changed") is True,
            "state_delta_matches_gold": expected is None
            or _state_delta_matches(observed, expected),
        }
        passed = all(checks.values())
        return EvalExecutionScorerResult(
            passed=passed,
            score=_score(checks),
            scorer_name=self.name,
            metadata={
                "observed_delta": observed,
                "expected_delta": expected,
                "observed_added_key_count": _state_delta_key_count(
                    observed,
                    "added_keys",
                ),
                "observed_removed_key_count": _state_delta_key_count(
                    observed,
                    "removed_keys",
                ),
                "expected_added_key_count": _state_delta_key_count(
                    expected,
                    "added_keys",
                ),
                "expected_removed_key_count": _state_delta_key_count(
                    expected,
                    "removed_keys",
                ),
                "observed_changed_resource_count": _state_delta_changed_resource_count(
                    observed
                ),
                "state_resource_count": len(_STATE_RESOURCE_NAMES),
                **{f"check.{name}": value for name, value in checks.items()},
            },
        )


class StateOutcomeOracleV1:
    """Drift-proof model gate for sandbox-mutated world state.

    This is the P1=A outcome oracle: it grades the observed sandbox state delta
    produced by a model run, not the exact trajectory or per-entry key hashes.
    Key identity can drift because the model re-decides tool arguments, so this
    oracle compares added/removed counts and reminder status-count shifts.
    """

    name = "state_outcome_oracle_v1"
    requires_exact_tool_sequence = False

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        observed = executor_result.metadata.get("sandbox_state_delta")
        expected = context.case.metadata.get("state_delta")
        checks = {
            "sandbox_executed": observed is not None,
            "state_mutated": observed is not None and observed.get("changed") is True,
            "outcome_matches_gold": expected is None
            or _count_delta_matches(observed, expected),
        }
        passed = all(checks.values())
        return EvalExecutionScorerResult(
            passed=passed,
            score=_score(checks),
            scorer_name=self.name,
            metadata={
                "observed_delta": observed,
                "expected_delta": expected,
                "observed_count_summary": _state_delta_count_summary(observed),
                "expected_count_summary": _state_delta_count_summary(expected),
                "observed_added_key_count": _state_delta_key_count(
                    observed,
                    "added_keys",
                ),
                "observed_removed_key_count": _state_delta_key_count(
                    observed,
                    "removed_keys",
                ),
                "expected_added_key_count": _state_delta_key_count(
                    expected,
                    "added_keys",
                ),
                "expected_removed_key_count": _state_delta_key_count(
                    expected,
                    "removed_keys",
                ),
                "observed_changed_resource_count": _state_delta_changed_resource_count(
                    observed
                ),
                "state_resource_count": len(_STATE_RESOURCE_NAMES),
                **{f"check.{name}": value for name, value in checks.items()},
            },
        )


EVAL_EXECUTION_SCORERS: dict[str, EvalExecutionScorer] = {
    ExactMatchEvalScorer.name: ExactMatchEvalScorer(),
    ToolTraceOracleV1.name: ToolTraceOracleV1(),
    CapabilityTraceOracleV1.name: CapabilityTraceOracleV1(),
    CapabilityCoverageOracleV1.name: CapabilityCoverageOracleV1(),
    StateOracleV1.name: StateOracleV1(),
    StateOutcomeOracleV1.name: StateOutcomeOracleV1(),
}


def resolve_execution_scorer(name: str) -> EvalExecutionScorer:
    """Resolve a registered execution scorer by name (raises on unknown)."""
    scorer = EVAL_EXECUTION_SCORERS.get(name)
    if scorer is None:
        supported = ", ".join(sorted(EVAL_EXECUTION_SCORERS))
        raise ValueError(f"unknown eval scorer: {name}. Supported scorers: {supported}")
    return scorer


def run_execution_report(
    store: EvalStore,
    *,
    executor: EvalExecutor | None = None,
    scorer: EvalExecutionScorer | None = None,
    pack: EvalRunPack | None = None,
    pack_filename: str = "eval_pack.json",
    report_filename: str = "eval_report.json",
    limit: int | None = None,
    samples: int = 1,
) -> EvalExecutionReportWrite:
    """Run executor-based checks over a runnable eval pack."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if samples < 1:
        raise ValueError("samples must be positive")

    selected_executor = executor or ReplayToolsExecutor()
    selected_scorer = scorer or ExactMatchEvalScorer()
    payload = pack or read_run_pack(store, pack_filename=pack_filename)
    cases = payload.cases[:limit] if limit is not None else payload.cases
    if not cases:
        raise ValueError("eval pack must contain cases")

    unknown_scorers = sorted(
        {
            case.scorer
            for case in cases
            if getattr(case, "scorer", None) and case.scorer not in EVAL_EXECUTION_SCORERS
        }
    )
    if unknown_scorers:
        supported = ", ".join(sorted(EVAL_EXECUTION_SCORERS))
        raise ValueError(
            f"unknown eval scorer(s): {', '.join(unknown_scorers)}. "
            f"Supported scorers: {supported}"
        )

    facet_inputs_by_id = {
        item.facet.facet_id: item for item in collect_text_facets(store)
    }
    report_cases = [
        _execute_case_sampled(
            store,
            payload,
            case,
            facet_inputs_by_id,
            selected_executor,
            selected_scorer,
            samples=samples,
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
            "fixture_match": getattr(selected_executor, "fixture_match_mode", "order"),
            "pack_case_count": len(payload.cases),
            "limit": limit or 0,
            "samples": samples,
        },
    )
    path = _report_output_path(store, report_filename)
    atomic_write_text(path, report.model_dump_json(indent=2) + "\n")
    return EvalExecutionReportWrite(
        report=report,
        path=path,
        relative_path=path.relative_to(store.root).as_posix(),
    )


def _execute_case_sampled(
    store: EvalStore,
    pack: EvalRunPack,
    case: EvalRunPackCase,
    facet_inputs_by_id: dict[str, EvalTextFacetInput],
    executor: EvalExecutor,
    default_scorer: EvalExecutionScorer,
    *,
    samples: int,
) -> EvalExecutionReportCase:
    first = _execute_case(
        store,
        pack,
        case,
        facet_inputs_by_id,
        executor,
        default_scorer,
    )
    if samples == 1 or first.status in {"blocked", "error"}:
        return first

    sample_cases = [first]
    for _ in range(samples - 1):
        sample_cases.append(
            _execute_case(
                store,
                pack,
                case,
                facet_inputs_by_id,
                executor,
                default_scorer,
            )
        )
    pass_count = sum(1 for sample_case in sample_cases if sample_case.status == "passed")
    return first.model_copy(
        update={
            "status": "passed" if pass_count * 2 > samples else "failed",
            "score": sum(sample_case.score for sample_case in sample_cases) / samples,
            "metadata": {
                **first.metadata,
                "sample_count": samples,
                "pass_count": pass_count,
                "pass_rate": pass_count / samples,
            },
        }
    )


def _execute_case(
    store: EvalStore,
    pack: EvalRunPack,
    case: EvalRunPackCase,
    facet_inputs_by_id: dict[str, EvalTextFacetInput],
    executor: EvalExecutor,
    default_scorer: EvalExecutionScorer,
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
    selected_scorer = (
        EVAL_EXECUTION_SCORERS.get(case.scorer, default_scorer)
        if getattr(case, "scorer", None)
        else default_scorer
    )
    scorer_result = selected_scorer.score(
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
    gating_checks = dict(all_checks)
    if not getattr(selected_scorer, "requires_exact_tool_sequence", True):
        gating_checks.pop("tool_sequence_matches", None)
    observed_trace_metadata = {
        "tool_calls_source": "replay_fixtures",
        "final_output_match_score": scorer_result.score,
        "scorer_name": scorer_result.scorer_name,
        "scorer_metadata_key_count": len(scorer_result.metadata),
        "executor_metadata_key_count": len(executor_result.metadata),
    }
    for key in _CAPABILITY_METADATA_KEYS:
        if key in scorer_result.metadata:
            observed_trace_metadata[key] = scorer_result.metadata[key]
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
        metadata=observed_trace_metadata,
    )
    return EvalExecutionReportCase(
        gold_case_id=case.gold_case_id,
        case_id=case.case_id,
        status="passed" if all(gating_checks.values()) else "failed",
        score=_score(gating_checks),
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
                "input_key": "",
            }
            order.append(call_key)
        fixture = fixtures[call_key]
        if event.kind == "tool_started":
            fixture["started"] = True
            fixture["start_event_index"] = index
            fixture["input_text"] = _payload_text(event.payload, ("input", "input_summary"))
            structured_input = event.payload.get("input")
            fixture["input_key"] = (
                _fixture_input_key(structured_input)
                if isinstance(structured_input, dict)
                else ""
            )
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


def _state_delta_matches(observed: Any, expected: Any) -> bool:
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    return _canonical_json(_state_delta_compare_payload(observed)) == _canonical_json(
        _state_delta_compare_payload(expected)
    )


def _count_delta_matches(observed: Any, expected: Any) -> bool:
    if not isinstance(observed, dict) or not isinstance(expected, dict):
        return False
    return _canonical_json(_state_delta_count_summary(observed)) == _canonical_json(
        _state_delta_count_summary(expected)
    )


def _state_delta_count_summary(delta: Any) -> dict[str, Any]:
    if not isinstance(delta, dict):
        return {}
    return {
        resource_name: _state_delta_resource_count_summary(
            delta.get(resource_name),
            resource_name,
        )
        for resource_name in _STATE_RESOURCE_NAMES
    }


def _state_delta_resource_count_summary(
    value: Any,
    resource_name: str,
) -> dict[str, Any]:
    item = value if isinstance(value, dict) else {}
    summary: dict[str, Any] = {
        "added_count": len(_sorted_string_list(item.get("added_keys"))),
        "removed_count": len(_sorted_string_list(item.get("removed_keys"))),
    }
    if resource_name == "reminders":
        summary["status_count_shift"] = _status_count_shift(
            item.get("status_counts_before"),
            item.get("status_counts_after"),
        )
    return summary


def _status_count_shift(before: Any, after: Any) -> dict[str, int]:
    before_counts = _status_counts(before)
    after_counts = _status_counts(after)
    return {
        key: after_counts.get(key, 0) - before_counts.get(key, 0)
        for key in sorted(set(before_counts) | set(after_counts))
        if after_counts.get(key, 0) - before_counts.get(key, 0) != 0
    }


def _state_delta_compare_payload(delta: dict[str, Any]) -> dict[str, Any]:
    return {
        resource_name: _state_delta_resource_payload(
            delta.get(resource_name),
            resource_name,
        )
        for resource_name in _STATE_RESOURCE_NAMES
    }


def _state_delta_resource_payload(value: Any, resource_name: str) -> dict[str, Any]:
    item = value if isinstance(value, dict) else {}
    payload: dict[str, Any] = {
        "added_keys": _sorted_string_list(item.get("added_keys")),
        "removed_keys": _sorted_string_list(item.get("removed_keys")),
        "count_before": _safe_int(item.get("count_before"), default=0),
        "count_after": _safe_int(item.get("count_after"), default=0),
    }
    if resource_name == "reminders":
        payload["status_counts_before"] = _status_counts(
            item.get("status_counts_before")
        )
        payload["status_counts_after"] = _status_counts(item.get("status_counts_after"))
    return payload


def _state_delta_key_count(delta: Any, key_name: str) -> int:
    if not isinstance(delta, dict):
        return 0
    total = 0
    for resource_name in _STATE_RESOURCE_NAMES:
        resource_delta = delta.get(resource_name)
        if not isinstance(resource_delta, dict):
            continue
        values = resource_delta.get(key_name)
        if isinstance(values, list):
            total += len([item for item in values if isinstance(item, str)])
    return total


def _state_delta_changed_resource_count(delta: Any) -> int:
    if not isinstance(delta, dict):
        return 0
    return sum(
        1
        for resource_name in _STATE_RESOURCE_NAMES
        if _state_resource_has_key_delta(delta.get(resource_name))
    )


def _state_resource_has_key_delta(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return bool(
        _sorted_string_list(value.get("added_keys"))
        or _sorted_string_list(value.get("removed_keys"))
    )


def _sorted_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(item for item in value if isinstance(item, str))


def _status_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): _safe_int(count, default=0)
        for key, count in sorted(value.items(), key=lambda item: str(item[0]))
    }


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _sanitize_labels(values: Sequence[str], *, prefix: str) -> list[str]:
    return [_sanitize_label(value, prefix=prefix) for value in values]


def _capability_metadata(
    *,
    expected: Sequence[str],
    observed: Sequence[str],
) -> dict[str, list[str]]:
    expected_set = set(expected)
    observed_set = set(observed)
    return {
        "observed_capabilities": _sorted_sanitized_capabilities(observed_set),
        "expected_capabilities": _sorted_sanitized_capabilities(expected_set),
        "missing_capabilities": _sorted_sanitized_capabilities(
            expected_set - observed_set
        ),
        "unexpected_capabilities": _sorted_sanitized_capabilities(
            observed_set - expected_set
        ),
    }


def _sorted_sanitized_capabilities(values: set[str]) -> list[str]:
    return sorted({_sanitize_capability_label(value) for value in values})


def _sanitize_capability_label(value: str) -> str:
    """Sanitize a capability label while keeping the ``binary subcommand`` form.

    Capability labels are structured (a typed tool name, or
    ``bash:<binary> [subcommand]``) and never carry raw arguments, so a single
    internal space is safe and worth preserving for readability — unlike
    ``_sanitize_label``, which hashes anything containing a space and would turn
    ``bash:maps-cli reviews`` into an opaque ``cap:<hash>``.
    """
    text = str(value)
    if text and len(text) <= 96 and all(_is_safe_capability_char(char) for char in text):
        return text
    return _stable_id("cap", text)


def _is_safe_capability_char(char: str) -> bool:
    return char == " " or _is_safe_label_char(char)


def _sanitize_label(value: str, *, prefix: str) -> str:
    text = str(value)
    if text and len(text) <= 96 and all(_is_safe_label_char(char) for char in text):
        return text
    return _stable_id(prefix, text)


def _is_safe_label_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in {"_", "-", ".", ":", "/"})


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


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
