from __future__ import annotations

import json
from pathlib import Path

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.evals import (
    EvalEpisode,
    EvalSessionGroup,
    EvalSessionReport,
    EvalSessionReportCase,
    EvalSessionRunResult,
    FaithfulSessionRunner,
)
from openharness.evals.grounding_blame import (
    GROUNDING_ATTRIBUTION_RUBRIC,
    attribute_faithful_grounding_report,
)
from openharness.evals.meta_judge import MetaJudgeAttributor
from ohmo.evals import get_eval_store
import ohmo.evals.runner as runner_module


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


class _ConditionalJudgeApiClient:
    def __init__(self, *, required_task: str, required_rubric_text: str) -> None:
        self.required_task = required_task
        self.required_rubric_text = required_rubric_text
        self.calls = 0
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        self.calls += 1
        prompt = request.messages[0].text
        if self.required_task in prompt and self.required_rubric_text in prompt:
            payload = {
                "blame": "model",
                "subtype": "fabricated_fact",
                "confidence": 0.88,
                "evidence": "The answer asserts a fact contradicted by retrieval evidence.",
            }
        else:
            payload = {
                "blame": "harness_rubric",
                "subtype": "empty_rubric_or_task",
                "confidence": 0.92,
                "evidence": "The task or rubric was missing.",
            }
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=json.dumps(payload))],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _RecordingAttributor:
    def __init__(self) -> None:
        self.calls = []

    def attribute(self, **kwargs):
        self.calls.append(kwargs)
        raise AssertionError("meta-judge should not be called")


def test_grounding_blame_model_for_refuted_fabricated_fact(tmp_path: Path) -> None:
    report = _faithful_report(
        "s-model",
        metadata={
            "grounding_status": "scored",
            "grounding_score": 0.0,
            "grounding_refuted_fact_claims": [
                {
                    "claim": "The Eiffel Tower is in Berlin.",
                    "evidence": "Sources say the Eiffel Tower is in Paris, France.",
                }
            ],
        },
    )
    trace_root = _write_trace(
        tmp_path,
        "s-model",
        prompt="Where is the Eiffel Tower?",
        final_text="The Eiffel Tower is in Berlin.",
    )
    api_client = _SequenceJudgeApiClient(
        [
            {
                "blame": "model",
                "subtype": "fabricated_fact",
                "confidence": 0.86,
                "evidence": "Retrieval contradicted the answer.",
            }
        ]
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attribute_faithful_grounding_report(
        report,
        attributor=attributor,
        trace_root=trace_root,
    )

    item = result["attributions"][0]
    assert item["session_id"] == "s-model"
    assert item["blame"] == "model"
    assert item["subtype"] == "fabricated_fact"
    assert api_client.calls == 1
    prompt = api_client.requests[0].messages[0].text
    assert "The Eiffel Tower is in Berlin." in prompt
    assert "Sources say the Eiffel Tower is in Paris" in prompt


def test_grounding_blame_model_for_refuted_fact_from_persisted_metadata() -> None:
    task = "How tall is the Eiffel Tower?"
    answer = "The Eiffel Tower is 500m tall."
    rubric_text = GROUNDING_ATTRIBUTION_RUBRIC["grounding"]["text"]
    report = _faithful_report(
        "s-metadata-only",
        metadata={
            "grounding_status": "unverifiable",
            "grounding_score": 0.0,
            "grounding_refuted_fact_claims": [
                {
                    "claim": "The Eiffel Tower is 500m tall.",
                    "evidence": "Search results say the Eiffel Tower is 330m tall.",
                }
            ],
            "grounding_task": task,
            "grounding_answer": answer,
        },
    )
    api_client = _ConditionalJudgeApiClient(
        required_task=task,
        required_rubric_text=rubric_text,
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attribute_faithful_grounding_report(report, attributor=attributor)

    item = result["attributions"][0]
    assert item["session_id"] == "s-metadata-only"
    assert item["blame"] == "model"
    assert item["subtype"] == "fabricated_fact"
    assert api_client.calls == 1
    prompt = api_client.requests[0].messages[0].text
    assert task in prompt
    assert rubric_text in prompt
    assert answer in prompt


def test_grounding_blame_harness_boundary_for_sandbox_blocked_shortcut() -> None:
    report = _faithful_report(
        "s-harness",
        metadata={
            "grounding_status": "sandbox_blocked",
            "grounding_score": 0.0,
            "grounding_refuted_fact_claims": [],
        },
    )
    attributor = _RecordingAttributor()

    result = attribute_faithful_grounding_report(report, attributor=attributor)

    item = result["attributions"][0]
    assert item["blame"] == "harness_boundary"
    assert item["subtype"] == "sandbox_blocked"
    assert item["confidence"] == 1.0
    assert attributor.calls == []


def test_grounding_blame_judge_fault_when_refuted_evidence_supports_answer(
    tmp_path: Path,
) -> None:
    report = _faithful_report(
        "s-judge",
        metadata={
            "grounding_status": "scored",
            "grounding_score": 0.0,
            "grounding_refuted_fact_claims": [
                {
                    "claim": "Paris is the capital of France.",
                    "evidence": "Official sources identify Paris as France's capital.",
                }
            ],
        },
    )
    trace_root = _write_trace(
        tmp_path,
        "s-judge",
        prompt="What is the capital of France?",
        final_text="Paris is the capital of France.",
    )
    api_client = _SequenceJudgeApiClient(
        [
            {
                "blame": "judge_fault",
                "subtype": "supported_claim_marked_refuted",
                "confidence": 0.91,
                "evidence": "The cited evidence supports the answer.",
            }
        ]
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attribute_faithful_grounding_report(
        report,
        attributor=attributor,
        trace_root=trace_root,
    )

    item = result["attributions"][0]
    assert item["blame"] == "judge_fault"
    assert item["subtype"] == "supported_claim_marked_refuted"
    assert api_client.calls == 1


def test_faithful_report_case_metadata_persists_bounded_grounding_detail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = get_eval_store(tmp_path / "workspace")
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="s-grounding",
            user_text="private request",
        )
    )
    runner = FaithfulSessionRunner(
        api_client=object(),
        model="model",
        system_prompt="",
        agent_runner=object(),
    )
    final_answer = "private final " * 200

    async def fake_run_session(**_kwargs):
        return EvalSessionRunResult(
            session_id="s-grounding",
            final_text=final_answer,
            turn_count=1,
            metadata={
                "transcript": (
                    ("user", "private request"),
                    ("assistant", "private final"),
                )
            },
        )

    async def fake_score_faithful_session(*_args, **_kwargs):
        return {
            "passed": False,
            "score": 2 / 3,
            "checks": {
                "intent_met": True,
                "constraints_held": True,
                "grounding_ok": False,
            },
            "observed_capabilities": [],
            "missing_capabilities": [],
            "grounding_task": "derived private request intent",
            "grounding": {
                "score": 0.0,
                "status": "scored",
                "claims": [
                    {
                        "kind": "fact",
                        "claim": f"fabricated fact {index}",
                        "verdict": "refuted",
                        "evidence": "x" * 300,
                    }
                    for index in range(7)
                ]
                + [
                    {
                        "kind": "action",
                        "claim": "made a file",
                        "verdict": "refuted",
                        "evidence": "missing",
                    }
                ],
            },
        }

    runner.run_session = fake_run_session
    monkeypatch.setattr(
        "ohmo.evals.runner.score_faithful_session",
        fake_score_faithful_session,
    )

    case = runner_module._run_session_report_case(
        store=store,
        group=EvalSessionGroup(session_id="s-grounding", episode_ids=("ep-1",)),
        runner=runner,
        gold_capabilities_by_session=None,
        user_simulator_factory=None,
        clarification_allowed_by_session=None,
        judge_votes=1,
        grounding_votes=1,
    )

    assert case.metadata["grounding_status"] == "scored"
    assert case.metadata["grounding_score"] == 0.0
    assert case.metadata["grounding_task"] == "derived private request intent"
    assert str(case.metadata["grounding_answer"]).startswith("private final ")
    assert len(str(case.metadata["grounding_answer"])) <= 1500
    claims = case.metadata["grounding_refuted_fact_claims"]
    assert len(claims) == 6
    assert claims[0]["claim"] == "fabricated fact 0"
    assert len(claims[0]["evidence"]) <= 240
    assert all("made a file" not in claim["claim"] for claim in claims)


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
                    "intent_met": True,
                    "constraints_held": True,
                    "grounding_ok": False,
                },
                metadata=metadata,
            )
        ],
    )


def _write_trace(
    tmp_path: Path,
    session_id: str,
    *,
    prompt: str,
    final_text: str,
) -> Path:
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    (trace_root / f"{session_id}-0.json").write_text(
        json.dumps(
            {
                "intent": prompt,
                "prompt": prompt,
                "final_text": final_text,
            }
        ),
        encoding="utf-8",
    )
    return trace_root
