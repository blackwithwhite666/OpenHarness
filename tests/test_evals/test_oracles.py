from __future__ import annotations

from types import SimpleNamespace

import pytest

from openharness.evals import (
    EVAL_EXECUTION_SCORERS,
    EvalExecutorResult,
    EvalObservedCall,
    ExactMatchEvalScorer,
    ToolTraceOracleV1,
    resolve_execution_scorer,
)


def _ctx(tool_names):
    """Minimal duck-typed execution context: the oracle reads case.tool_names."""
    return SimpleNamespace(case=SimpleNamespace(tool_names=list(tool_names)))


def _result(*calls: EvalObservedCall) -> EvalExecutorResult:
    return EvalExecutorResult(
        tool_path=tuple(call.tool_name for call in calls),
        tool_calls=tuple(calls),
    )


def test_tool_trace_oracle_passes_clean_trace():
    out = ToolTraceOracleV1().score(
        context=_ctx(["remind_create"]),
        executor_result=_result(EvalObservedCall("remind_create", {"due": "x"}, False)),
    )
    assert out.passed is True
    assert out.score == 1.0
    assert out.scorer_name == "tool_trace_oracle_v1"
    assert out.metadata["tool_error_count"] == 0
    assert out.metadata["check.no_unexpected_tools"] is True


def test_tool_trace_oracle_flags_unexpected_tool():
    out = ToolTraceOracleV1().score(
        context=_ctx(["remind_create"]),
        executor_result=_result(
            EvalObservedCall("remind_create", {}, False),
            EvalObservedCall("bash", {"cmd": "rm -rf /"}, False),
        ),
    )
    assert out.passed is False
    assert out.metadata["unexpected_tool_count"] == 1
    assert out.metadata["check.no_unexpected_tools"] is False


def test_tool_trace_oracle_flags_tool_error():
    out = ToolTraceOracleV1().score(
        context=_ctx(["web_fetch"]),
        executor_result=_result(EvalObservedCall("web_fetch", {}, True)),
    )
    assert out.passed is False
    assert out.metadata["tool_error_count"] == 1
    assert out.metadata["check.no_tool_errors"] is False


def test_tool_trace_oracle_flags_loop_over_budget():
    calls = [EvalObservedCall("web_fetch", {}, False) for _ in range(6)]
    out = ToolTraceOracleV1(max_calls_factor=2, max_calls_floor=3).score(
        context=_ctx(["web_fetch"]),  # budget = max(1 * 2, 3) = 3
        executor_result=_result(*calls),
    )
    assert out.passed is False
    assert out.metadata["call_budget"] == 3
    assert out.metadata["check.within_call_budget"] is False


def test_tool_trace_oracle_flags_no_tools_when_expected():
    # Answered without ever calling the expected tool (no grounding / no fetch).
    out = ToolTraceOracleV1().score(
        context=_ctx(["web_fetch"]),
        executor_result=_result(),
    )
    assert out.passed is False
    assert out.metadata["check.used_tools_when_expected"] is False


def test_tool_trace_oracle_is_lenient_when_case_declares_no_tools():
    # No expected tool set -> nothing to judge -> vacuous pass.
    out = ToolTraceOracleV1().score(context=_ctx([]), executor_result=_result())
    assert out.passed is True


def test_resolve_execution_scorer_known_and_unknown():
    assert (
        resolve_execution_scorer("tool_trace_oracle_v1")
        is EVAL_EXECUTION_SCORERS["tool_trace_oracle_v1"]
    )
    assert isinstance(resolve_execution_scorer("exact-final-text"), ExactMatchEvalScorer)
    with pytest.raises(ValueError, match="unknown eval scorer"):
        resolve_execution_scorer("not-a-scorer")
