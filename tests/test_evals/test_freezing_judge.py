from __future__ import annotations

import json
from pathlib import Path

import pytest

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.evals import (
    EvalEpisode,
    EvalExecutionContext,
    EvalExecutorResult,
    EvalObservedCall,
    EvalRunPack,
    EvalRunPackCase,
    EvalStore,
    FreezingJudgeScorer,
)


class _StaticJudgeApiClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=self._text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


@pytest.mark.parametrize(
    ("response", "expected_passed", "expected_verdict"),
    [
        ("PASS - PRIVATE JUDGE REASON", True, "pass"),
        ("FAIL - PRIVATE JUDGE REASON", False, "fail"),
        ("", False, "error"),
    ],
)
def test_trajectory_judge_scores_verdicts_metadata_only(
    tmp_path: Path,
    response: str,
    expected_passed: bool,
    expected_verdict: str,
):
    api_client = _StaticJudgeApiClient(response)
    scorer = FreezingJudgeScorer(api_client=api_client, model="judge-model")

    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(
            final_text="PRIVATE FINAL ANSWER",
            tool_path=("web_fetch",),
            tool_calls=(
                EvalObservedCall(
                    tool_name="web_fetch",
                    arguments={"url": "https://example.invalid/private"},
                    is_error=False,
                ),
            ),
        ),
    )

    assert result.passed is expected_passed
    assert result.score == (1.0 if expected_passed else 0.0)
    assert result.scorer_name == "freezing_judge"
    assert result.metadata["judge_model"] == "judge-model"
    assert result.metadata["verdict"] == expected_verdict
    assert "reason_hash" in result.metadata
    assert result.metadata["observed_capability_count"] == 1
    assert result.metadata["had_tool_error"] is False
    serialized = json.dumps(result.metadata, sort_keys=True)
    assert "PRIVATE USER GOAL" not in serialized
    assert "PRIVATE FINAL ANSWER" not in serialized
    assert "PRIVATE ACCEPTED OUTCOME" not in serialized
    assert "PRIVATE JUDGE REASON" not in serialized
    assert api_client.requests[0].model == "judge-model"
    assert api_client.requests[0].tools == []


class _SequenceJudgeApiClient:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    async def stream_message(self, request):
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


@pytest.mark.parametrize(
    ("texts", "expected_passed", "expected_verdict", "expected_pass_votes"),
    [
        (["PASS a", "PASS b", "FAIL c"], True, "pass", 2),
        (["FAIL a", "FAIL b", "PASS c"], False, "fail", 1),
        (["PASS a", "FAIL b", "FAIL c"], False, "fail", 1),
    ],
)
def test_trajectory_judge_majority_vote(
    tmp_path: Path,
    texts: list[str],
    expected_passed: bool,
    expected_verdict: str,
    expected_pass_votes: int,
):
    api_client = _SequenceJudgeApiClient(texts)
    scorer = FreezingJudgeScorer(api_client=api_client, model="judge-model", votes=3)

    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(
            final_text="answer",
            tool_path=("web_fetch",),
            tool_calls=(),
        ),
    )

    assert api_client.calls == 3
    assert result.passed is expected_passed
    assert result.metadata["verdict"] == expected_verdict
    assert result.metadata["judge_votes"] == 3
    assert result.metadata["judge_pass_votes"] == expected_pass_votes


def test_trajectory_judge_grounding_mode_judges_method_not_facts(tmp_path: Path):
    api_client = _StaticJudgeApiClient("PASS - grounded in sources")
    scorer = FreezingJudgeScorer(
        api_client=api_client, model="judge-model", votes=1, grounding_mode=True
    )

    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(
            final_text="answer", tool_path=("web_fetch",), tool_calls=()
        ),
    )

    assert result.passed is True
    request = api_client.requests[0]
    assert "METHOD and GROUNDING" in request.system_prompt
    user_text = request.messages[0].text
    assert "TIME-SENSITIVE" in user_text
    assert "possibly-stale" in user_text.lower()


def _context(tmp_path: Path) -> EvalExecutionContext:
    store = EvalStore(tmp_path / "evals")
    case = EvalRunPackCase(
        gold_case_id="gold-1",
        case_id="case-1",
        episode_id="ep-1",
        case_kind="unit",
        tool_names=["web_fetch"],
        capability_path=["web_fetch"],
        rubric=["Answer the user's request."],
    )
    pack = EvalRunPack(
        pack_id="pack-1",
        source_records_path="cases/gold_cases.jsonl",
        cases=[case],
    )
    episode = EvalEpisode(
        episode_id="ep-1",
        source="unit",
        app="test",
        user_text="PRIVATE USER GOAL",
    )
    return EvalExecutionContext(
        store=store,
        pack=pack,
        case=case,
        episode=episode,
        events=(),
        input_facets=(),
        expected_facets=(),
        tool_fixtures=(),
        primary_prompt="PRIVATE USER GOAL",
        expected_final_text="PRIVATE ACCEPTED OUTCOME",
        resource_snapshot_status="absent",
    )
