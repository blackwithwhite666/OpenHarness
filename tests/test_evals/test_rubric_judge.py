from __future__ import annotations

import json
from pathlib import Path

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
    RubricJudgeScorer,
    derive_case_rubric,
)
from openharness.evals.judge import _aggregate_v2, _parse_v2_scores


class _StaticJudgeApiClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.requests: list = []

    async def stream_message(self, request):
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant", content=[TextBlock(text=self._text)]
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def _wrap(scores_json: dict) -> str:
    return "Reasoning about the run.\n```json\n" + json.dumps(scores_json) + "\n```"


_ALL_GOOD = _wrap(
    {
        "task_completion": {"items": {"tc1": "pass", "tc2": "pass"}},
        "grounding": {"items": {"g1": "pass"}},
        "tool_use": {"score": 1},
        "answer_quality": {"score": 1},
        "error_recovery": {"score": 1},
        "efficiency": {"score": 0.5},
    }
)


def test_v2_grades_aspects_and_passes(tmp_path: Path):
    api_client = _StaticJudgeApiClient(_ALL_GOOD)
    scorer = RubricJudgeScorer(api_client=api_client, model="judge-model", votes=1)

    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(
            final_text="PRIVATE FINAL ANSWER",
            tool_path=("web_fetch",),
            tool_calls=(
                EvalObservedCall(
                    tool_name="web_fetch",
                    arguments={"url": "https://example.invalid/x"},
                    output="PRIVATE TOOL OUTPUT",
                ),
            ),
        ),
    )

    assert result.passed is True
    assert result.scorer_name == "rubric_judge"
    # Graded, not binary: efficiency 0.5 pulls it below 1.0.
    assert 0.9 < result.score < 1.0
    assert result.graded_score == result.score
    assert result.metadata["aspect.efficiency"] == 0.5
    assert result.metadata["aspect.task_completion"] == 1.0
    assert result.metadata["verdict"] == "pass"
    # Metadata is privacy-safe.
    serialized = json.dumps(result.metadata, sort_keys=True)
    assert "PRIVATE" not in serialized


def test_v2_hard_gate_fails_even_with_high_soft_aspects(tmp_path: Path):
    api_client = _StaticJudgeApiClient(
        _wrap(
            {
                "task_completion": {"items": {"tc1": "fail", "tc2": "fail"}},
                "grounding": {"items": {"g1": "pass"}},
                "tool_use": {"score": 1},
                "answer_quality": {"score": 1},
                "error_recovery": {"score": 1},
                "efficiency": {"score": 1},
            }
        )
    )
    scorer = RubricJudgeScorer(api_client=api_client, model="judge-model", votes=1)
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="a", tool_calls=()),
    )

    assert result.passed is False  # task_completion below the gate floor
    assert result.metadata["aspect.task_completion"] == 0.0
    assert "task_completion" in result.metadata["rubric_gate_failures"]
    # Still graded (not zero): the soft aspects contribute.
    assert 0.6 < result.graded_score < 0.7


def test_v2_uses_derived_checklist_in_prompt(tmp_path: Path):
    rubrics = {
        "case-1": {
            "task_completion": [
                {"id": "tc1", "text": "A reminder was created"},
                {"id": "tc2", "text": "Due time is tomorrow at 09:00"},
            ],
            "grounding": [{"id": "g1", "text": "Confirmation matches the reminder"}],
        }
    }
    api_client = _StaticJudgeApiClient(_ALL_GOOD)
    scorer = RubricJudgeScorer(
        api_client=api_client, model="judge-model", votes=1, rubrics=rubrics
    )
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ok", tool_calls=()),
    )

    assert result.metadata["rubric_used_checklist"] is True
    prompt = api_client.requests[0].messages[0].text
    assert "Due time is tomorrow at 09:00" in prompt
    assert "A reminder was created" in prompt


def test_v2_all_votes_unparseable_is_error(tmp_path: Path):
    scorer = RubricJudgeScorer(
        api_client=_StaticJudgeApiClient("no json here"), model="m", votes=2
    )
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="a", tool_calls=()),
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.metadata["verdict"] == "error"


def test_parse_and_aggregate_pure():
    scores = _parse_v2_scores(_ALL_GOOD)
    assert scores["task_completion"] == 1.0
    assert scores["efficiency"] == 0.5
    aspects, graded, passed, gates = _aggregate_v2([scores])
    assert passed is True
    assert gates == ()
    assert 0.9 < graded < 1.0


def test_derive_case_rubric_parses_checklist():
    api_client = _StaticJudgeApiClient(
        _wrap(
            {
                "task_completion": [
                    {"id": "tc1", "text": "A reminder was created"},
                    {"id": "tc2", "text": "Due tomorrow 09:00"},
                ],
                "grounding": [{"id": "g1", "text": "Confirmation is truthful"}],
            }
        )
    )
    rubric = derive_case_rubric(
        api_client=api_client,
        model="m",
        goal="remind me tomorrow at 9 to check tickets",
        gold_trajectory="[]",
        gold_answer="Reminder set for tomorrow 09:00.",
    )
    assert len(rubric["task_completion"]) == 2
    assert rubric["grounding"][0]["text"] == "Confirmation is truthful"


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
        episode_id="ep-1", source="unit", app="test", user_text="PRIVATE USER GOAL"
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
