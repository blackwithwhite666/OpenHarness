from __future__ import annotations

import json
from pathlib import Path

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.evals.constraint_blame import (
    attribute_faithful_constraints_report,
    constraint_attribution_rubric,
)
from openharness.evals.meta_judge import MetaJudgeAttributor
from openharness.evals.models import EvalEpisode, EvalSessionReport, EvalSessionReportCase
from openharness.evals.store import EvalStore


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


def test_constraint_blame_model_for_satisfiable_constraint_ignored(
    tmp_path: Path,
) -> None:
    store = EvalStore(tmp_path / "evals")
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="s-model",
            user_text="Summarize the report and cite the source URL.",
        )
    )
    report = _faithful_report(
        "s-model",
        metadata={
            "grounding_answer": "The report says the migration is complete.",
            "intent_evidence": "The answer did not include a citation.",
        },
    )
    api_client = _SequenceJudgeApiClient(
        [
            {
                "intent": "Summarize the report.",
                "known_info": [],
                "constraints": ["cite the source URL"],
            },
            {
                "blame": "model",
                "subtype": "constraint_ignored",
                "confidence": 0.87,
                "evidence": "The answer omits the required source URL.",
            },
        ]
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attribute_faithful_constraints_report(
        report,
        attributor=attributor,
        api_client=api_client,
        model="judge-model",
        store=store,
        app="ohmo",
    )

    item = result["attributions"][0]
    assert item["blame"] == "model"
    assert item["subtype"] == "constraint_ignored"
    assert item["constraints"] == ["cite the source URL"]
    assert api_client.calls == 2
    prompt = api_client.requests[1].messages[0].text
    assert "MUST cite the source URL" in prompt
    assert '"constraints_held": false' in prompt


def test_constraint_blame_harness_rubric_for_off_task_constraint() -> None:
    report = _faithful_report(
        "s-harness",
        metadata={
            "constraints": ["use the missing secret recording from /missing.wav"],
            "grounding_task": "Summarize the meeting notes.",
            "grounding_answer": "The meeting notes say the launch moved to Friday.",
            "intent_evidence": "The answer summarized the supplied notes.",
        },
    )
    api_client = _SequenceJudgeApiClient(
        [
            {
                "blame": "harness_rubric",
                "subtype": "off_task_constraint",
                "confidence": 0.91,
                "evidence": "The constraint requires unavailable off-task material.",
            }
        ]
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attribute_faithful_constraints_report(report, attributor=attributor)

    item = result["attributions"][0]
    assert item["blame"] == "harness_rubric"
    assert item["subtype"] == "off_task_constraint"
    assert item["constraints"] == [
        "use the missing secret recording from /missing.wav"
    ]
    assert result["summary"]["constraint_harness_debt_pct"] == 100.0
    assert api_client.calls == 1


def test_constraint_blame_uses_persisted_constraints_without_store(monkeypatch) -> None:
    def fail_derive(*_args, **_kwargs):
        raise AssertionError("constraints should come from persisted metadata")

    monkeypatch.setattr(
        "openharness.evals.constraint_blame.derive_ironuser_spec",
        fail_derive,
    )
    report = _faithful_report(
        "s-persisted",
        metadata={
            "constraints": ["be concise"],
            "grounding_task": "Give a concise project status.",
            "grounding_answer": "The project is delayed because reviews are pending.",
            "intent_evidence": "The answer exceeded the requested brevity.",
        },
    )
    api_client = _SequenceJudgeApiClient(
        [
            {
                "blame": "model",
                "subtype": "verbosity_constraint_ignored",
                "confidence": 0.82,
                "evidence": "The persisted constraint was enough to attribute.",
            }
        ]
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attribute_faithful_constraints_report(report, attributor=attributor)

    item = result["attributions"][0]
    assert item["blame"] == "model"
    assert item["constraints"] == ["be concise"]
    assert api_client.calls == 1


def test_constraint_attribution_rubric_frames_each_constraint_as_must() -> None:
    rubric = constraint_attribution_rubric(["cite sources", "stay under 100 words"])

    assert rubric == {
        "task_completion": {
            "text": (
                "Attribute only the failed constraints_held check. A violated "
                "satisfiable constraint is MODEL. An unsatisfiable, off-task, "
                "or hallucinated constraint from the IronUser spec is "
                "HARNESS_RUBRIC. A compliant answer that the original "
                "constraints judge failed is JUDGE_FAULT.\n"
                "MUST cite sources\n"
                "MUST stay under 100 words"
            )
        }
    }


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
                    "constraints_held": False,
                    "grounding_ok": True,
                },
                metadata=metadata,
            )
        ],
    )
