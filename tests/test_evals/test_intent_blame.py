from __future__ import annotations

import json

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.evals.intent_blame import (
    attribute_faithful_intent_report,
    intent_attribution_rubric,
)
from openharness.evals.meta_judge import MetaJudgeAttributor, signal_prefilter
from openharness.evals.models import EvalSessionReport, EvalSessionReportCase


class _SequenceJudgeApiClient:
    def __init__(self, payloads: list[dict]) -> None:
        self._texts = [json.dumps(payload) for payload in payloads]
        self.calls = 0
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def test_intent_blame_model_for_under_delivered_answer() -> None:
    intent = "Create a migration plan for the repository."
    answer = "I can describe general migration planning, but I did not inspect the repo."
    report = _faithful_report(
        "s-model",
        metadata={
            "grounding_task": intent,
            "grounding_answer": answer,
            "intent_evidence": "The answer did not produce the requested repository plan.",
        },
    )
    api_client = _SequenceJudgeApiClient(
        [
            {
                "blame": "model",
                "subtype": "under_delivered",
                "confidence": 0.88,
                "evidence": "The agent had the repo context but did not make the plan.",
            }
        ]
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attribute_faithful_intent_report(report, attributor=attributor)

    item = result["attributions"][0]
    assert item["session_id"] == "s-model"
    assert item["intent"] == intent
    assert item["blame"] == "model"
    assert item["subtype"] == "under_delivered"
    assert result["summary"]["intent_harness_debt_pct"] == 0.0
    assert api_client.calls == 1
    prompt = api_client.requests[0].messages[0].text
    assert intent_attribution_rubric(intent)["task_completion"]["text"] in prompt
    assert answer in prompt
    assert '"intent_met": false' in prompt


def test_intent_blame_harness_boundary_for_missing_required_tool() -> None:
    answer = "I could not edit the project because codex failed: command not found."
    evidence = "The required codex command was unavailable in the sandbox."
    report = _faithful_report(
        "s-boundary",
        metadata={
            "grounding_task": "Modify the local project with codex.",
            "grounding_answer": answer,
            "intent_evidence": evidence,
        },
    )
    api_client = _SequenceJudgeApiClient(
        [
            {
                "blame": "harness_boundary",
                "subtype": "missing_tool",
                "confidence": 0.94,
                "evidence": "The answer reports a required command was unavailable.",
            }
        ]
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attribute_faithful_intent_report(report, attributor=attributor)

    item = result["attributions"][0]
    assert item["blame"] == "harness_boundary"
    assert item["subtype"] == "missing_tool"
    assert result["summary"]["intent_harness_debt_pct"] == 100.0
    assert signal_prefilter([{"evidence": evidence}], answer) == "harness_boundary"
    prompt = api_client.requests[0].messages[0].text
    assert "Deterministic signal hint: harness_boundary" in prompt


def test_intent_blame_harness_rubric_for_under_specified_task() -> None:
    report = _faithful_report(
        "s-rubric",
        metadata={
            "grounding_task": "Do the requested thing.",
            "grounding_answer": "I could not determine which artifact or outcome was requested.",
            "intent_evidence": "The user intent omitted the target and success criteria.",
        },
    )
    api_client = _SequenceJudgeApiClient(
        [
            {
                "blame": "harness_rubric",
                "subtype": "under_specified_intent",
                "confidence": 0.9,
                "evidence": "The task lacks a concrete target or success criteria.",
            }
        ]
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attribute_faithful_intent_report(report, attributor=attributor)

    item = result["attributions"][0]
    assert item["blame"] == "harness_rubric"
    assert item["subtype"] == "under_specified_intent"
    assert item["intent"] == "Do the requested thing."
    assert result["summary"]["intent_harness_debt_pct"] == 100.0
    assert api_client.calls == 1


def _faithful_report(session_id: str, *, metadata: dict) -> EvalSessionReport:
    return EvalSessionReport(
        report_id="faithful-report",
        session_count=1,
        passed_count=0,
        failed_count=1,
        cases=[
            EvalSessionReportCase(
                session_id=session_id,
                status="failed",
                score=2 / 3,
                checks={
                    "intent_met": False,
                    "constraints_held": True,
                    "grounding_ok": True,
                },
                metadata=metadata,
            )
        ],
    )
