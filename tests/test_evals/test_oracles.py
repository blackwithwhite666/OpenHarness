from __future__ import annotations

from types import SimpleNamespace

import pytest

from openharness.evals import (
    CapabilityCoverageOracleV1,
    CapabilityTraceOracleV1,
    EVAL_EXECUTION_SCORERS,
    EvalExecutorResult,
    EvalObservedCall,
    ExactMatchEvalScorer,
    ToolTraceOracleV1,
    resolve_execution_scorer,
)


def _ctx(tool_names, capability_path=()):
    """Minimal duck-typed execution context: the oracles read case fields."""
    return SimpleNamespace(
        case=SimpleNamespace(
            tool_names=list(tool_names),
            capability_path=list(capability_path),
        )
    )


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


def test_capability_trace_oracle_passes_clean_shell_capability():
    out = CapabilityTraceOracleV1().score(
        context=_ctx(["bash"], ["bash:weather-cli forecast"]),
        executor_result=_result(
            EvalObservedCall(
                "bash",
                {"command": "weather-cli forecast 'СПб'"},
                False,
            )
        ),
    )

    assert out.passed is True
    assert out.score == 1.0
    assert out.scorer_name == "capability_trace_oracle_v1"
    assert out.metadata["observed_call_count"] == 1
    assert out.metadata["expected_capability_count"] == 1
    assert out.metadata["unexpected_capability_count"] == 0
    assert out.metadata["missing_capability_count"] == 0
    assert out.metadata["tool_error_count"] == 0
    assert out.metadata["check.no_unexpected_capabilities"] is True
    assert out.metadata["check.expected_capabilities_present"] is True
    assert "missing_capabilities" in out.metadata
    assert "observed_capabilities" in out.metadata


@pytest.mark.parametrize(
    ("capability_path", "calls", "failed_check", "count_key"),
    [
        (
            ["bash:weather-cli forecast", "bash:calendar-cli list"],
            (EvalObservedCall("bash", {"command": "weather-cli forecast 'СПб'"}, False),),
            "expected_capabilities_present",
            "missing_capability_count",
        ),
        (
            ["bash:weather-cli forecast"],
            (
                EvalObservedCall("bash", {"command": "weather-cli forecast 'СПб'"}, False),
                EvalObservedCall("bash", {"command": "calendar-cli list"}, False),
            ),
            "no_unexpected_capabilities",
            "unexpected_capability_count",
        ),
        (
            ["bash:weather-cli forecast"],
            (EvalObservedCall("bash", {"command": "weather-cli forecast 'СПб'"}, True),),
            "no_tool_errors",
            "tool_error_count",
        ),
        (
            ["bash:weather-cli forecast"],
            tuple(
                EvalObservedCall("bash", {"command": "weather-cli forecast 'СПб'"}, False)
                for _ in range(4)
            ),
            "within_call_budget",
            "observed_call_count",
        ),
    ],
)
def test_capability_trace_oracle_flags_trace_violations(
    capability_path,
    calls,
    failed_check,
    count_key,
):
    out = CapabilityTraceOracleV1(max_calls_factor=2, max_calls_floor=3).score(
        context=_ctx(["bash"], capability_path),
        executor_result=_result(*calls),
    )

    assert out.passed is False
    assert out.score == 0.0
    assert out.metadata[f"check.{failed_check}"] is False
    assert out.metadata[count_key] > 0


def test_capability_trace_oracle_exposes_capability_label_metadata():
    out = CapabilityTraceOracleV1().score(
        context=_ctx(["remind_create"], ["remind_create", "web_fetch"]),
        executor_result=_result(EvalObservedCall("remind_create", {}, False)),
    )

    assert out.passed is False
    assert out.metadata["observed_capabilities"] == ["remind_create"]
    assert out.metadata["expected_capabilities"] == ["remind_create", "web_fetch"]
    assert out.metadata["missing_capabilities"] == ["web_fetch"]
    assert out.metadata["unexpected_capabilities"] == []


def test_capability_trace_oracle_ignores_missing_incidental_capability():
    out = CapabilityTraceOracleV1().score(
        context=_ctx(["todo_write", "bash"], ["todo_write", "bash:weather-cli forecast"]),
        executor_result=_result(
            EvalObservedCall(
                "bash",
                {"command": "weather-cli forecast 'СПб'"},
                False,
            )
        ),
    )

    assert out.passed is True
    assert out.metadata["check.expected_capabilities_present"] is True
    assert out.metadata["missing_capability_count"] == 0
    assert out.metadata["incidental_capability_count"] == 1
    assert out.metadata["expected_capability_count"] == 1
    assert out.metadata["expected_capabilities"] == ["bash:weather-cli forecast"]
    assert out.metadata["missing_capabilities"] == []


def test_capability_oracles_ignore_expected_bare_shell_capability():
    context = _ctx(["bash"], ["bash", "bash:maps-cli search"])
    result = _result(
        EvalObservedCall(
            "bash",
            {"command": "maps-cli search 'x'"},
            False,
        )
    )

    trace = CapabilityTraceOracleV1().score(
        context=context,
        executor_result=result,
    )
    coverage = CapabilityCoverageOracleV1().score(
        context=context,
        executor_result=result,
    )

    assert trace.passed is True
    assert trace.metadata["check.expected_capabilities_present"] is True
    assert trace.metadata["missing_capability_count"] == 0
    assert trace.metadata["incidental_capability_count"] == 1
    assert trace.metadata["expected_capabilities"] == ["bash:maps-cli search"]
    assert trace.metadata["missing_capabilities"] == []
    assert coverage.passed is True
    assert coverage.metadata["check.expected_core_capabilities_covered"] is True
    assert coverage.metadata["missing_core_count"] == 0
    assert coverage.metadata["expected_capabilities"] == ["bash:maps-cli search"]
    assert coverage.metadata["missing_capabilities"] == []


def test_capability_trace_oracle_incidental_call_does_not_cover_core_capability():
    out = CapabilityTraceOracleV1().score(
        context=_ctx(["todo_write", "bash"], ["todo_write", "bash:weather-cli forecast"]),
        executor_result=_result(EvalObservedCall("todo_write", {}, False)),
    )

    assert out.passed is False
    assert out.metadata["missing_capability_count"] == 1
    assert out.metadata["check.expected_capabilities_present"] is False
    assert out.metadata["missing_capabilities"] == ["bash:weather-cli forecast"]
    assert out.metadata["observed_capabilities"] == []


def test_capability_trace_oracle_ignores_extra_incidental_capability():
    out = CapabilityTraceOracleV1().score(
        context=_ctx(["todo_write", "bash"], ["todo_write", "bash:weather-cli forecast"]),
        executor_result=_result(
            EvalObservedCall("todo_write", {}, False),
            EvalObservedCall(
                "bash",
                {"command": "weather-cli forecast 'СПб'"},
                False,
            ),
        ),
    )

    assert out.passed is True
    assert out.metadata["check.no_unexpected_capabilities"] is True
    assert out.metadata["unexpected_capability_count"] == 0
    assert out.metadata["unexpected_capabilities"] == []


def test_capability_trace_oracle_ignores_extra_bare_shell_call():
    out = CapabilityTraceOracleV1().score(
        context=_ctx(["bash"], ["bash:maps-cli search"]),
        executor_result=_result(
            EvalObservedCall(
                "bash",
                {"command": "maps-cli search 'x'"},
                False,
            ),
            EvalObservedCall("bash", {"command": "mkdir d"}, False),
        ),
    )

    assert out.passed is True
    assert out.metadata["check.no_unexpected_capabilities"] is True
    assert out.metadata["unexpected_capability_count"] == 0
    assert out.metadata["unexpected_capabilities"] == []
    assert out.metadata["observed_capabilities"] == ["bash:maps-cli search"]


def test_capability_metadata_preserves_subcommand_labels_readable():
    # bash:<binary> <subcommand> labels contain a space but no raw args, so they
    # must stay readable in the report — not hashed to cap:<hash>.
    out = CapabilityTraceOracleV1().score(
        context=_ctx(["bash"], ["bash:weather-cli forecast"]),
        executor_result=_result(
            EvalObservedCall("bash", {"command": "weather-cli forecast 'СПб'"}, False)
        ),
    )

    assert out.metadata["observed_capabilities"] == ["bash:weather-cli forecast"]
    assert out.metadata["expected_capabilities"] == ["bash:weather-cli forecast"]
    assert out.metadata["missing_capabilities"] == []


def test_capability_coverage_oracle_ignores_missing_incidental_capability():
    out = CapabilityCoverageOracleV1().score(
        context=_ctx(["bash"], ["todo_write", "bash:weather-cli forecast"]),
        executor_result=_result(
            EvalObservedCall(
                "bash",
                {"command": "weather-cli forecast 'СПб'"},
                False,
            )
        ),
    )

    assert out.passed is True
    assert out.score == 1.0
    assert out.scorer_name == "capability_coverage_oracle_v1"
    assert out.metadata["expected_core_count"] == 1
    assert out.metadata["observed_core_count"] == 1
    assert out.metadata["missing_core_count"] == 0
    assert out.metadata["check.no_tool_errors"] is True
    assert out.metadata["check.expected_core_capabilities_covered"] is True
    assert out.metadata["check.used_tools_when_expected"] is True
    assert "check.no_unexpected_capabilities" not in out.metadata
    assert "call_budget" not in out.metadata
    assert "todo_write" not in out.metadata["expected_capabilities"]
    assert out.metadata["missing_capabilities"] == []
    assert out.metadata["observed_capabilities"]


def test_capability_coverage_oracle_fails_when_core_capability_missing():
    out = CapabilityCoverageOracleV1().score(
        context=_ctx(["remind_create"], ["todo_write", "remind_create"]),
        executor_result=_result(EvalObservedCall("todo_write", {}, False)),
    )

    assert out.passed is False
    assert out.score == 0.0
    assert out.metadata["missing_core_count"] == 1
    assert out.metadata["check.expected_core_capabilities_covered"] is False
    assert out.metadata["check.used_tools_when_expected"] is False
    assert out.metadata["missing_capabilities"] == ["remind_create"]
    assert out.metadata["observed_capabilities"] == []


def test_capability_coverage_oracle_tolerates_extra_core_capability():
    out = CapabilityCoverageOracleV1().score(
        context=_ctx(["remind_create"], ["remind_create"]),
        executor_result=_result(
            EvalObservedCall("remind_create", {}, False),
            EvalObservedCall("web_fetch", {}, False),
        ),
    )

    assert out.passed is True
    assert out.metadata["missing_core_count"] == 0
    assert out.metadata["unexpected_capabilities"] == ["web_fetch"]
    assert out.metadata["check.expected_core_capabilities_covered"] is True


def test_resolve_execution_scorer_known_and_unknown():
    assert (
        resolve_execution_scorer("tool_trace_oracle_v1")
        is EVAL_EXECUTION_SCORERS["tool_trace_oracle_v1"]
    )
    assert (
        resolve_execution_scorer("capability_trace_oracle_v1")
        is EVAL_EXECUTION_SCORERS["capability_trace_oracle_v1"]
    )
    assert (
        resolve_execution_scorer("capability_coverage_oracle_v1")
        is EVAL_EXECUTION_SCORERS["capability_coverage_oracle_v1"]
    )
    assert isinstance(resolve_execution_scorer("exact-final-text"), ExactMatchEvalScorer)
    with pytest.raises(ValueError, match="unknown eval scorer"):
        resolve_execution_scorer("not-a-scorer")


def test_trajectory_judge_sentinel_requires_api_client():
    scorer = resolve_execution_scorer("trajectory_judge_v1")

    assert scorer is EVAL_EXECUTION_SCORERS["trajectory_judge_v1"]
    assert scorer.requires_exact_tool_sequence is False
    with pytest.raises(RuntimeError, match="trajectory_judge_v1 requires an api_client"):
        scorer.score(context=_ctx([]), executor_result=_result())
