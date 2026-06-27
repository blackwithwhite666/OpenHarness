"""Built-in tool for model-authored decision-trace breadcrumbs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from openharness.evals.decision_trace import (
    DECISION_TRACE_MODEL_EVENT_KINDS,
    DecisionTraceValidationError,
)
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

DECISION_TRACE_RECORDER_METADATA_KEY = "decision_trace_recorder"

TraceKind = Literal[
    "trace_intent",
    "trace_decision",
    "trace_observation",
    "trace_uncertainty",
    "trace_stop_condition",
    "trace_finalization",
]


class TraceToolInput(BaseModel):
    """Arguments for recording a decision-trace breadcrumb."""

    kind: TraceKind = Field(
        description=(
            "Model-authored decision trace event kind: trace_intent, "
            "trace_decision, trace_observation, trace_uncertainty, "
            "trace_stop_condition, or trace_finalization."
        )
    )
    payload: dict[str, Any] = Field(
        description=(
            "Structured JSON object to record. Keep it concise and do not include "
            "chain-of-thought, secrets, or private raw content."
        )
    )


class TraceTool(BaseTool):
    """Record a structured model-authored decision-trace event."""

    name = "trace"
    description = (
        "Record a concise structured decision-trace breadcrumb for intent, decisions, "
        "observations, uncertainty, stop conditions, or finalization."
    )
    input_model = TraceToolInput

    def is_read_only(self, arguments: TraceToolInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: TraceToolInput, context: ToolExecutionContext) -> ToolResult:
        recorder = context.metadata.get(DECISION_TRACE_RECORDER_METADATA_KEY)
        if recorder is None:
            return ToolResult(output="Decision trace recorder unavailable; no trace recorded.")

        if arguments.kind not in DECISION_TRACE_MODEL_EVENT_KINDS:
            return ToolResult(
                output=f"Unsupported model-authored decision trace kind: {arguments.kind}",
                is_error=True,
            )

        try:
            recorded = recorder.record(arguments.kind, arguments.payload)
        except DecisionTraceValidationError as exc:
            return ToolResult(
                output=f"Decision trace validation failed: {exc}",
                is_error=True,
            )

        if not recorded:
            return ToolResult(output="Decision trace recorder disabled; no trace recorded.")
        return ToolResult(output=f"Recorded decision trace event: {arguments.kind}")
