"""Executor-based eval runner for runnable eval packs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    SupportsStreamingMessages,
)
from openharness.engine.messages import ConversationMessage
from openharness.evals.facets import EvalTextFacetInput, collect_text_facets
from openharness.evals.judge import FreezingJudgeScorer, RubricJudgeScorer
from openharness.evals.executor import (
    EvalExecutionContext,
    EvalExecutor,
    EvalExecutorResult,
    EvalToolFixture,
    ReplayToolsExecutor,
    _run_eval_coroutine,
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
from openharness.evals.decision_trace_summary import summarize_decision_trace
from openharness.evals.replay_matching import _fixture_input_key
from openharness.evals.replay_integrity import replay_integrity
from openharness.evals.state import compute_episode_state_delta
from openharness.evals.store import EvalStore
from openharness.evals.tool_labels import SHELL_TOOL_NAMES, effective_tool_label
from openharness.evals.trace_capture import write_eval_trace
from openharness.utils.fs import atomic_write_text

_EXECUTION_SCORE_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)
_INCIDENTAL_CAPABILITIES = frozenset(
    {"todo_write", "sleep", "task_get", "task_output", "task_stop", "tool_search"}
) | SHELL_TOOL_NAMES
_CAPABILITY_METADATA_KEYS = (
    "observed_capabilities",
    "expected_capabilities",
    "missing_capabilities",
    "unexpected_capabilities",
)
_JUDGE_METADATA_KEYS = (
    "judge_model",
    "verdict",
    "judge_votes",
    "judge_pass_votes",
    "reason_hash",
    "reason_length",
    "observed_capability_count",
    "had_tool_error",
)
_RUBRIC_JUDGE_METADATA_KEYS = (
    "judge_parsed_votes",
    "rubric_graded_score",
    "rubric_gate_failures",
    "rubric_used_checklist",
    "aspect.task_completion",
    "aspect.grounding",
    "aspect.tool_use",
    "aspect.answer_quality",
    "aspect.error_recovery",
    "aspect.efficiency",
    # verify-grounding (ADR ohmo-eval-verification-grounding)
    "grounding_mode",
    "grounding_status",
    "grounding_verified",
    "grounding_refuted",
    "grounding_claims",
)
_SCORER_REPORT_METADATA_KEYS = (
    _CAPABILITY_METADATA_KEYS + _JUDGE_METADATA_KEYS + _RUBRIC_JUDGE_METADATA_KEYS
)
_EXECUTOR_REPORT_METADATA_KEYS = (
    "seeded_history_message_count",
    "materialized_file_count",
)
_STATE_RESOURCE_NAMES = ("reminders", "memory", "todos")
_HISTORY_WINDOW_HOURS = 24
_HISTORY_WINDOW_MAX_TURNS = 40
_HISTORY_SEGMENT_TIMEOUT = 30.0
SESSION_SEGMENT_SYSTEM_PROMPT = (
    "These are consecutive turns in ONE chat, oldest first, followed by a "
    "TARGET turn. The user may have switched topics several times. Return the "
    "index of the EARLIEST prior turn that is part of the SAME ongoing "
    "conversation / task that the TARGET continues (a contiguous thread up to "
    "the target). If the TARGET starts a brand-new topic unrelated to all "
    'prior turns, there is no relevant history. Respond with ONLY JSON: '
    '{"start_index": <int>} (the earliest in-thread index), or '
    '{"start_index": null} if none relate. Be conservative: when unsure '
    "whether an older turn belongs, EXCLUDE it."
)
_HISTORY_SEGMENT_CACHE: dict[str, int | None] = {}


@dataclass(frozen=True)
class HistoryContext:
    """Auxiliary LLM context for scoping prior turns to a target conversation."""

    api_client: SupportsStreamingMessages
    model: str


@dataclass(frozen=True)
class _HistoryCandidateTurn:
    user_text: str
    assistant_text: str


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
    # Transient, recorder-only: the raw judge reason. NEVER serialized into the
    # metadata-only report (only reason_hash/length go to metadata); the D7
    # trace recorder reads this for the rich eval trace.
    raw_reason: str | None = None
    # When set, this graded [0,1] quality score becomes the case score (instead
    # of the pass-fraction of gating checks). Multi-aspect scorers set it;
    # binary scorers leave it None so the legacy case-score composition is
    # byte-identical (keeps the inner-loop gate baseline stable).
    graded_score: float | None = None


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

    Exact tool-sequence equality is advisory here: capability coverage gates,
    while trajectory order may legitimately drift.
    """

    name = "capability_trace_oracle_v1"
    requires_exact_tool_sequence = False

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
        core_expected = [
            capability
            for capability in expected
            if capability not in _INCIDENTAL_CAPABILITIES
        ]
        core_expected_set = set(core_expected)
        calls = list(executor_result.tool_calls)
        observed = [
            effective_tool_label(call.tool_name, call.arguments)
            for call in calls
        ]
        core_observed = [
            capability
            for capability in observed
            if capability not in _INCIDENTAL_CAPABILITIES
        ]
        core_observed_set = set(core_observed)
        unexpected = sorted(
            {
                capability
                for capability in core_observed
                if core_expected_set and capability not in core_expected_set
            }
        )
        missing = sorted(
            {
                capability
                for capability in core_expected
                if capability not in core_observed_set
            }
        )
        error_count = sum(1 for call in calls if call.is_error)
        budget = max(
            len(core_expected) * self._max_calls_factor,
            self._max_calls_floor,
        )
        checks = {
            "no_unexpected_capabilities": not unexpected,
            "no_tool_errors": error_count == 0,
            "within_call_budget": len(calls) <= budget,
            "expected_capabilities_present": not missing,
            "used_tools_when_expected": (not core_expected) or bool(calls),
        }
        passed = all(checks.values())
        return EvalExecutionScorerResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            scorer_name=self.name,
            metadata={
                "observed_call_count": len(calls),
                "expected_capability_count": len(core_expected),
                "unexpected_capability_count": len(unexpected),
                "missing_capability_count": len(missing),
                "incidental_capability_count": len(expected) - len(core_expected),
                "tool_error_count": error_count,
                "call_budget": budget,
                **{f"check.{name}": value for name, value in checks.items()},
                **_capability_metadata(
                    expected=core_expected,
                    observed=core_observed,
                ),
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


class _FreezingJudgeSentinel:
    name = FreezingJudgeScorer.name
    requires_exact_tool_sequence = False

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        del context, executor_result
        raise RuntimeError(
            "freezing_judge requires an api_client; run via "
            "'ohmo evals run --scorer freezing_judge'"
        )


class _RubricJudgeSentinel:
    name = RubricJudgeScorer.name
    requires_exact_tool_sequence = False

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        del context, executor_result
        raise RuntimeError(
            "rubric_judge requires an api_client; run via "
            "'ohmo evals run --scorer rubric_judge'"
        )


EVAL_EXECUTION_SCORERS: dict[str, EvalExecutionScorer] = {
    ExactMatchEvalScorer.name: ExactMatchEvalScorer(),
    ToolTraceOracleV1.name: ToolTraceOracleV1(),
    CapabilityTraceOracleV1.name: CapabilityTraceOracleV1(),
    CapabilityCoverageOracleV1.name: CapabilityCoverageOracleV1(),
    StateOracleV1.name: StateOracleV1(),
    StateOutcomeOracleV1.name: StateOutcomeOracleV1(),
    FreezingJudgeScorer.name: _FreezingJudgeSentinel(),
    RubricJudgeScorer.name: _RubricJudgeSentinel(),
    # Back-compat aliases for the pre-rename scorer names (trajectory_judge_v1 =
    # freezing_judge, v2 = rubric_judge) so a bundle spec.json or a recorded
    # report still on the old names keeps resolving. Drop once nothing uses them.
    "trajectory_judge_v1": _FreezingJudgeSentinel(),
    "trajectory_judge_v2": _RubricJudgeSentinel(),
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
    history_context: HistoryContext | None = None,
    pack: EvalRunPack | None = None,
    pack_filename: str = "eval_pack.json",
    report_filename: str = "eval_report.json",
    limit: int | None = None,
    samples: int = 1,
    max_turns: int | None = None,
    conversation_histories: dict[str, tuple[tuple[str, str], ...]] | None = None,
) -> EvalExecutionReportWrite:
    """Run executor-based checks over a runnable eval pack.

    ``conversation_histories`` (case_id -> ((role, text), ...)), when given,
    overrides the per-case session history that would otherwise be recomputed
    from the store. Recomputation reads the whole store's episodes + order, so it
    is not portable; baking the history makes a committed bundle replay the same
    turn-1 prefix on any machine.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if samples < 1:
        raise ValueError("samples must be positive")

    selected_executor = executor or ReplayToolsExecutor()
    scorer_override = scorer
    default_scorer = scorer or ExactMatchEvalScorer()
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
    report_id = _stable_id(
        "eval-exec",
        selected_executor.name,
        payload.pack_id,
        *(case.case_id for case in cases),
    )
    traces_root = store.root / "traces"
    path = _report_output_path(store, report_filename)
    total = len(cases)

    def _build_report(
        done_cases: list[EvalExecutionReportCase], *, complete: bool
    ) -> EvalExecutionReport:
        return EvalExecutionReport(
            report_id=report_id,
            pack_id=payload.pack_id,
            case_count=len(done_cases),
            passed_count=sum(1 for c in done_cases if c.status == "passed"),
            failed_count=sum(1 for c in done_cases if c.status == "failed"),
            blocked_count=sum(1 for c in done_cases if c.status == "blocked"),
            error_count=sum(1 for c in done_cases if c.status == "error"),
            cases=done_cases,
            metadata={
                "privacy": "metadata_only",
                "mode": "execution_replay",
                "executor_name": selected_executor.name,
                "scorer_name": default_scorer.name,
                "score_schema_version": _EXECUTION_SCORE_SCHEMA_VERSION,
                "fixture_match": getattr(selected_executor, "fixture_match_mode", "order"),
                "pack_case_count": len(payload.cases),
                "limit": limit or 0,
                "samples": samples,
                "max_turns": max_turns,
                # Progress markers: the report file is rewritten after every
                # case so an external observer can poll it mid-run. This run is
                # long and ~entirely network-bound (one model call per agent
                # turn + judge votes, sequential), so without intermediate
                # writes the report would only appear at the very end.
                "progress_done": len(done_cases),
                "progress_total": total,
                "progress_complete": complete,
            },
        )

    report_cases: list[EvalExecutionReportCase] = []
    for index, case in enumerate(cases, start=1):
        report_case = _execute_case_sampled(
            store,
            payload,
            case,
            facet_inputs_by_id,
            selected_executor,
            default_scorer,
            samples=samples,
            scorer_override=scorer_override,
            history_context=history_context,
            conversation_histories=conversation_histories,
            traces_root=traces_root,
            run_id=report_id,
        )
        report_cases.append(report_case)
        progress = (
            f"[eval] {index}/{total} "
            f"case={getattr(case, 'case_id', '?')} status={report_case.status}"
        )
        logger.info(progress)
        # Reliable progress in the run log even when INFO logging isn't wired
        # up to stdout (e.g. under nohup).
        print(progress, flush=True)
        # Intermediate write: rewrite the report after every case so progress
        # is observable mid-run.
        atomic_write_text(
            path,
            _build_report(report_cases, complete=False).model_dump_json(indent=2) + "\n",
        )

    report = _build_report(report_cases, complete=True)
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
    scorer_override: EvalExecutionScorer | None = None,
    history_context: HistoryContext | None = None,
    conversation_histories: dict[str, tuple[tuple[str, str], ...]] | None = None,
    traces_root: Path,
    run_id: str,
) -> EvalExecutionReportCase:
    first = _execute_case(
        store,
        pack,
        case,
        facet_inputs_by_id,
        executor,
        default_scorer,
        scorer_override=scorer_override,
        history_context=history_context,
        conversation_histories=conversation_histories,
        traces_root=traces_root,
        run_id=run_id,
        sample_index=0,
    )
    if samples == 1 or first.status in {"blocked", "error"}:
        return first

    sample_cases = [first]
    for sample_index in range(1, samples):
        sample_cases.append(
            _execute_case(
                store,
                pack,
                case,
                facet_inputs_by_id,
                executor,
                default_scorer,
                scorer_override=scorer_override,
                history_context=history_context,
                conversation_histories=conversation_histories,
                traces_root=traces_root,
                run_id=run_id,
                sample_index=sample_index,
            )
        )
    pass_count = sum(1 for sample_case in sample_cases if sample_case.status == "passed")
    # A split (neither unanimous pass nor unanimous fail) marks a flaky case:
    # the verdict is a coin-flip, not a stable signal. Surface it so the gate can
    # treat it as pass-with-warning rather than a hard pass/fail.
    flaky = 0 < pass_count < samples
    return first.model_copy(
        update={
            "status": "passed" if pass_count * 2 > samples else "failed",
            "score": sum(sample_case.score for sample_case in sample_cases) / samples,
            "metadata": {
                **first.metadata,
                "sample_count": samples,
                "pass_count": pass_count,
                "pass_rate": pass_count / samples,
                "flaky": flaky,
            },
        }
    )


def _session_conversation_history(
    store: EvalStore,
    episode: EvalEpisode,
    *,
    history_context: HistoryContext | None = None,
    max_messages: int = 16,
    max_chars: int = 12000,
) -> tuple[tuple[str, str], ...]:
    if not episode.session_id or max_messages <= 0 or max_chars <= 0:
        return ()

    indexed_episode_ids = list(enumerate(store.list_episode_ids()))
    target_order = next(
        (
            index
            for index, episode_id in indexed_episode_ids
            if episode_id == episode.episode_id
        ),
        len(indexed_episode_ids),
    )
    target_position = (episode.created_at, target_order)
    prior: list[tuple[int, EvalEpisode]] = []
    for index, episode_id in indexed_episode_ids:
        if episode_id == episode.episode_id:
            continue
        candidate = store.get_episode(episode_id)
        if candidate is None:
            continue
        if candidate.session_id != episode.session_id or candidate.app != episode.app:
            continue
        if (candidate.created_at, index) >= target_position:
            continue
        prior.append((index, candidate))

    candidates = [
        item
        for item in sorted(prior, key=lambda item: (item[1].created_at, item[0]))
        if _within_history_window(item[1], episode)
    ][-_HISTORY_WINDOW_MAX_TURNS:]
    if not candidates:
        return ()

    candidate_turns = [
        _HistoryCandidateTurn(
            user_text=prior_episode.user_text,
            assistant_text=_episode_gateway_final_text(store, prior_episode.episode_id),
        )
        for _, prior_episode in candidates
    ]
    # When an LLM history segmenter is configured, use it to find where the
    # relevant thread starts; otherwise fall back to the raw recent turns in the
    # window so the agent is never silently starved of conversation context (the
    # bare query-engine eval used to get no history at all without --history-*).
    if history_context is not None:
        start_index = _segment_conversation(history_context, candidate_turns, episode)
        if start_index is None:
            return ()
        candidate_turns = candidate_turns[start_index:]

    messages = _history_turn_messages(candidate_turns)

    kept = messages[-max_messages:]
    while kept and sum(len(text) for _, text in kept) > max_chars:
        kept = kept[1:]
    return tuple(kept)


def _within_history_window(candidate: EvalEpisode, target: EvalEpisode) -> bool:
    candidate_created_at = _parse_history_timestamp(candidate.created_at)
    target_created_at = _parse_history_timestamp(target.created_at)
    if candidate_created_at is None or target_created_at is None:
        return True
    return target_created_at - candidate_created_at <= timedelta(
        hours=_HISTORY_WINDOW_HOURS
    )


def _parse_history_timestamp(value: Any) -> datetime | None:
    try:
        if isinstance(value, datetime):
            timestamp = value
        elif isinstance(value, str):
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            return None
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _history_turn_messages(
    turns: Sequence[_HistoryCandidateTurn],
) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    for turn in turns:
        if turn.user_text.strip():
            messages.append(("user", turn.user_text))
        if turn.assistant_text.strip():
            messages.append(("assistant", turn.assistant_text))
    return messages


def _segment_conversation(
    history_context: HistoryContext,
    candidate_turns: Sequence[_HistoryCandidateTurn],
    target_episode: EvalEpisode,
) -> int | None:
    cache_key = target_episode.episode_id
    if cache_key in _HISTORY_SEGMENT_CACHE:
        cached = _HISTORY_SEGMENT_CACHE[cache_key]
        if cached is None or cached >= len(candidate_turns):
            return None
        return cached

    try:
        prompt = _history_segment_prompt(candidate_turns, target_episode)
        text = _run_eval_coroutine(_complete_history_segment(history_context, prompt))
        start_index = _parse_segment_start_index(text, len(candidate_turns))
    except Exception:
        start_index = None
    _HISTORY_SEGMENT_CACHE[cache_key] = start_index
    return start_index


async def _complete_history_segment(
    history_context: HistoryContext,
    prompt: str,
) -> str:
    request = ApiMessageRequest(
        model=history_context.model,
        messages=[ConversationMessage.from_user_text(prompt)],
        system_prompt=SESSION_SEGMENT_SYSTEM_PROMPT,
        # Output is a tiny JSON, but reasoning models can spend tokens before
        # it — keep headroom (matches the judge) so the JSON is never truncated.
        max_tokens=512,
        tools=[],
    )

    async def _collect() -> str:
        text = ""
        async for event in history_context.api_client.stream_message(request):
            if isinstance(event, ApiMessageCompleteEvent):
                text = event.message.text.strip()
        return text

    return await asyncio.wait_for(_collect(), timeout=_HISTORY_SEGMENT_TIMEOUT)


def _history_segment_prompt(
    candidate_turns: Sequence[_HistoryCandidateTurn],
    target_episode: EvalEpisode,
) -> str:
    lines: list[str] = []
    for index, turn in enumerate(candidate_turns):
        lines.append(
            f"[{index}] user: {_trim_history_text(turn.user_text, 300)}\n"
            f"    assistant: {_trim_history_text(turn.assistant_text, 300)}"
        )
    lines.append(f"TARGET user: {_trim_history_text(target_episode.user_text, 500)}")
    return "\n".join(lines)


def _trim_history_text(text: str, max_length: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_length:
        return clean
    return clean[: max_length - 3].rstrip() + "..."


def _parse_segment_start_index(text: str, candidate_count: int) -> int | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    start_index = payload.get("start_index")
    if start_index is None or isinstance(start_index, bool):
        return None
    if not isinstance(start_index, int):
        return None
    if start_index < 0 or start_index >= candidate_count:
        return None
    return start_index


def _episode_gateway_final_text(store: EvalStore, episode_id: str) -> str:
    text = ""
    for event in store.iter_events(episode_id):
        if event.kind != "gateway_final":
            continue
        event_text = _payload_text(
            event.payload,
            ("text", "assistant_final", "output", "final"),
        )
        if event_text:
            text = event_text
    return text


def build_gold_reference(
    store: EvalStore,
    case: EvalRunPackCase,
    facet_inputs_by_id: dict[str, EvalTextFacetInput],
    *,
    output_excerpt_chars: int = 400,
) -> tuple[str, str, str] | None:
    """Return ``(goal, gold_answer, gold_trajectory_json)`` for one case.

    Distilled from the recorded gold episode for offline rubric derivation (the
    reference solution the checklist is extracted from). Returns ``None`` when
    the episode is missing. This reads raw gold text and MUST only be used
    offline to author a committed rubric — never persisted into a report.
    """
    episode = store.get_episode(case.episode_id)
    if episode is None:
        return None
    events = list(store.iter_events(case.episode_id))
    resolved_inputs = tuple(
        facet
        for facet_id in case.input_facet_ids
        if (facet := facet_inputs_by_id.get(facet_id)) is not None
    )
    resolved_expected = tuple(
        facet
        for facet_id in case.expected_facet_ids
        if (facet := facet_inputs_by_id.get(facet_id)) is not None
    )
    goal = _select_facet_text(resolved_inputs, ("user_goal", "user_request"))
    gold_answer = _select_facet_text(
        resolved_expected, ("assistant_final", "gateway_error", "tool_output")
    )
    trajectory = json.dumps(
        [
            {
                "tool": fixture.tool_name,
                "is_error": fixture.is_error,
                "output": (fixture.output_text or "")[:output_excerpt_chars],
            }
            for fixture in _tool_fixtures(events)
        ],
        ensure_ascii=True,
    )
    return goal, gold_answer, trajectory


def _execute_case(
    store: EvalStore,
    pack: EvalRunPack,
    case: EvalRunPackCase,
    facet_inputs_by_id: dict[str, EvalTextFacetInput],
    executor: EvalExecutor,
    default_scorer: EvalExecutionScorer,
    *,
    scorer_override: EvalExecutionScorer | None = None,
    history_context: HistoryContext | None = None,
    conversation_histories: dict[str, tuple[tuple[str, str], ...]] | None = None,
    traces_root: Path,
    run_id: str,
    sample_index: int,
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
    replay_ok, unreplayable_reason = replay_integrity(episode, events, tool_fixtures)
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
        "replay_inputs_recoverable": replay_ok,
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
        if unreplayable_reason:
            context = context.model_copy(
                update={
                    "metadata": {
                        **context.metadata,
                        "unreplayable_reason": unreplayable_reason,
                    }
                }
            )
    if unreplayable_reason:
        logger.warning(
            "eval case %s unreplayable: %s",
            case.gold_case_id,
            unreplayable_reason,
        )
    if not all(checks.values()) or episode is None:
        return _blocked_execution_case(
            case,
            checks,
            context,
            resource_snapshot_status=resource_snapshot_status,
            executor_name=executor.name,
            unreplayable_reason=unreplayable_reason,
        )

    if conversation_histories is not None and case.case_id in conversation_histories:
        # Baked (portable) history — see run_execution_report. Avoids recomputing
        # from the store, whose episode set/order isn't portable to a slim bundle.
        conversation_history = conversation_histories[case.case_id]
    else:
        conversation_history = _session_conversation_history(
            store,
            episode,
            history_context=history_context,
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
        conversation_history=conversation_history,
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
    if scorer_override is not None:
        selected_scorer = scorer_override
    else:
        selected_scorer = (
            EVAL_EXECUTION_SCORERS.get(case.scorer, default_scorer)
            if getattr(case, "scorer", None)
            else default_scorer
        )
    scorer_result = selected_scorer.score(
        context=execution_context,
        executor_result=executor_result,
    )
    try:
        write_eval_trace(
            traces_root,
            run_id,
            case.case_id,
            sample_index,
            context=execution_context,
            executor_result=executor_result,
            scorer_result=scorer_result,
        )
    except Exception:
        logger.warning(
            "failed to write eval trace: run_id=%s case_id=%s sample_index=%s",
            run_id,
            case.case_id,
            sample_index,
            exc_info=True,
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
        **summarize_decision_trace(execution_context.events),
    }
    for key in _SCORER_REPORT_METADATA_KEYS:
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
    # Multi-aspect scorers (rubric_judge) supply a graded quality score;
    # binary scorers leave it None -> keep the legacy gating-fraction score.
    case_score = (
        scorer_result.graded_score
        if scorer_result.graded_score is not None
        else _score(gating_checks)
    )
    return EvalExecutionReportCase(
        gold_case_id=case.gold_case_id,
        case_id=case.case_id,
        status="passed" if all(gating_checks.values()) else "failed",
        score=case_score,
        max_score=1.0,
        checks=all_checks,
        warnings=[name for name, passed in all_checks.items() if not passed],
        context=context,
        observed_trace=observed_trace,
        metadata={
            **_execution_case_metadata(
                case,
                executor_name=executor.name,
                scorer_name=scorer_result.scorer_name,
                resource_snapshot_status=resource_snapshot_status,
            ),
            **_executor_report_metadata(executor_result.metadata),
        },
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
        metadata={
            "tool_calls_source": "replay_fixtures",
            **summarize_decision_trace(execution_context.events),
        },
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
    unreplayable_reason: str | None = None,
) -> EvalExecutionReportCase:
    metadata = _execution_case_metadata(
        case,
        executor_name=executor_name,
        scorer_name="",
        resource_snapshot_status=resource_snapshot_status,
    )
    if unreplayable_reason:
        metadata["unreplayable_reason"] = unreplayable_reason
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
        metadata=metadata,
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


def _executor_report_metadata(metadata: dict[str, Any]) -> dict[str, int]:
    report_metadata: dict[str, int] = {}
    for key in _EXECUTOR_REPORT_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            report_metadata[key] = value
    return report_metadata


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
