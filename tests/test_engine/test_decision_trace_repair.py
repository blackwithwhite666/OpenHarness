from __future__ import annotations

from openharness.engine.messages import ConversationMessage
from openharness.engine.query import (
    _TraceObservation,
    _TraceRequirement,
    _decision_trace_repair_instruction,
)


def _final(text: str) -> ConversationMessage:
    return ConversationMessage.from_user_text(text)


def test_repair_instruction_requests_evidence_linked_claims() -> None:
    req = _TraceRequirement(True, "substantive_final_answer", ("substantive_final_answer",))
    obs = [
        _TraceObservation("toolu_maps_1", "bash:maps-cli reviews", "closed today", False),
        _TraceObservation("toolu_web_2", "web_search", "holiday hours", False),
    ]
    instruction = _decision_trace_repair_instruction(_final("final answer"), req, obs)

    assert "trace_finalization" in instruction
    assert "answer_claims" in instruction
    assert "supported_by" in instruction
    # the actual observation ids must be offered for linking
    assert "toolu_maps_1" in instruction
    assert "toolu_web_2" in instruction
    assert "bash:maps-cli reviews" in instruction
    # no failed tool this turn -> no trace_observation step required
    assert "trace_observation" not in instruction


def test_repair_instruction_requires_failure_observation() -> None:
    req = _TraceRequirement(
        True,
        "failed_tool_result",
        ("failed_tool_result", "current_run_tool_use"),
    )
    obs = [_TraceObservation("toolu_bad_1", "bash:python3", "RuntimeError", True)]
    instruction = _decision_trace_repair_instruction(_final("final answer"), req, obs)

    assert "trace_observation" in instruction
    assert "related_tool_call_id" in instruction
    assert "[ERROR]" in instruction
    assert "toolu_bad_1" in instruction


def test_repair_instruction_handles_no_observations() -> None:
    req = _TraceRequirement(True, "substantive_final_answer", ("substantive_final_answer",))
    instruction = _decision_trace_repair_instruction(_final("final answer"), req, [])

    assert "No tool observations" in instruction
    assert "answer_claims" in instruction
