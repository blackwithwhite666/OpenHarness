from __future__ import annotations

import json

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.evals.meta_judge import MetaJudgeAttributor, signal_prefilter


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


def test_attributor_parses_label():
    api_client = _SequenceJudgeApiClient(
        [
            {
                "blame": "harness_boundary",
                "subtype": "missing_tool",
                "confidence": 0.82,
                "evidence": "The trajectory shows the required tool failed.",
            }
        ]
    )
    attributor = MetaJudgeAttributor(api_client=api_client, model="judge-model")

    result = attributor.attribute(
        task="Read the attached report.",
        rubric={"task_completion": [{"id": "tc1", "text": "Summarize report"}]},
        answer="I could not read the file.",
        trajectory=[{"tool": "read_file", "is_error": True, "output": "mount failed"}],
        aspect_scores={"task_completion": 0},
        gate_failures=["task_completion"],
    )

    assert result == {
        "blame": "harness_boundary",
        "subtype": "missing_tool",
        "confidence": 0.82,
        "evidence": "The trajectory shows the required tool failed.",
        "votes": 1,
    }
    assert api_client.requests[0].model == "judge-model"
    assert api_client.requests[0].tools == []


def test_votes_take_majority_label():
    api_client = _SequenceJudgeApiClient(
        [
            _vote("harness_boundary"),
            _vote("model"),
            _vote("harness_boundary"),
        ]
    )
    attributor = MetaJudgeAttributor(
        api_client=api_client, model="judge-model", votes=3
    )

    result = _attribute_minimal(attributor)

    assert api_client.calls == 3
    assert result["blame"] == "harness_boundary"
    assert result["votes"] == 3


def test_tie_prefers_model():
    api_client = _SequenceJudgeApiClient(
        [
            _vote("harness_boundary", confidence=0.9),
            _vote("model", confidence=0.6),
        ]
    )
    attributor = MetaJudgeAttributor(
        api_client=api_client, model="judge-model", votes=2
    )

    result = _attribute_minimal(attributor)

    assert result["blame"] == "model"
    assert result["confidence"] == 0.6


def test_signal_prefilter_flags_only_strong_harness_signals():
    assert signal_prefilter([], "bash: ripgrep: command not found") == "harness_boundary"
    assert signal_prefilter([], "Я не нашёл подходящих результатов.") is None


def _vote(label: str, *, confidence: float = 0.7) -> dict:
    return {
        "blame": label,
        "subtype": None,
        "confidence": confidence,
        "evidence": f"{label} evidence",
    }


def _attribute_minimal(attributor: MetaJudgeAttributor) -> dict:
    return attributor.attribute(
        task="Find a fact.",
        rubric={},
        answer="No result.",
        trajectory=[],
        aspect_scores={"task_completion": 0},
        gate_failures=["task_completion"],
    )
