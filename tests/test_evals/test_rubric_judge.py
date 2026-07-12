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
    RubricJudgeScorer,
    derive_case_rubric,
)
from openharness.evals.judge import _aggregate_v2, _parse_v2_scores, _verify_grounding


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


# --- verification-grounding (grounding_mode="verify") -------------------------


class _RoutedJudgeApiClient:
    """Route stream_message by system_prompt: rubric-score / extract / verdict."""

    def __init__(self, *, rubric: str, extract: str = "", verdict: str = "") -> None:
        self._rubric, self._extract, self._verdict = rubric, extract, verdict
        self.requests: list = []

    async def stream_message(self, request):
        self.requests.append(request)
        sp = (request.system_prompt or "").lower()
        if "extract the checkable" in sp:
            text = self._extract
        elif "verify claims against evidence" in sp:
            text = self._verdict
        else:
            text = self._rubric
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


async def _search_ok(query, *, max_results=4):
    return f"Search results for: {query}\n1. Source\n   confirms it."


# rubric response with a FAILING grounding (process=0) so the verify override is visible.
_LOW_GROUNDING = _wrap(
    {
        "task_completion": {"items": {"tc1": "pass"}},
        "grounding": {"items": {"g1": "fail"}},
        "tool_use": {"score": 1},
        "answer_quality": {"score": 1},
        "error_recovery": {"score": 1},
        "efficiency": {"score": 1},
    }
)


def _verify_scorer(api_client, **kw):
    return RubricJudgeScorer(
        api_client=api_client, model="m", votes=1, grounding_mode="verify", search=_search_ok, **kw
    )


def test_verify_grounding_overrides_true_answer_to_high(tmp_path: Path):
    extract = _wrap(
        {
            "sandbox_blocked": False,
            "claims": [
                {"id": "c1", "text": "Salmon dish is 950 rub", "kind": "fact", "public": True, "relevant": True, "query": "salmon 950"},
                {"id": "c2", "text": "Open daily 12-23", "kind": "fact", "public": True, "relevant": True, "query": "hours"},
            ],
        }
    )
    verdict = _wrap(
        {
            "verdicts": [
                {"id": "c1", "verdict": "verified", "evidence": "matches menu"},
                {"id": "c2", "verdict": "verified", "evidence": "matches"},
            ]
        }
    )
    scorer = _verify_scorer(_RoutedJudgeApiClient(rubric=_LOW_GROUNDING, extract=extract, verdict=verdict))
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    # process grounding was 0.0 (fail); verify lifts it to 1.0 (2/2 verified)
    assert result.metadata["aspect.grounding"] == 1.0
    assert result.metadata["grounding_mode"] == "verify"
    assert result.metadata["grounding_status"] == "scored"
    assert result.metadata["grounding_verified"] == 2
    assert result.metadata["grounding_refuted"] == 0
    assert "grounding" not in result.metadata["rubric_gate_failures"]
    assert {c["id"] for c in result.metadata["grounding_claims"]} == {"c1", "c2"}


def test_verify_grounding_refuted_lowers_score(tmp_path: Path):
    extract = _wrap(
        {
            "sandbox_blocked": False,
            "claims": [
                {"id": "c1", "text": "A", "kind": "fact", "public": True, "relevant": True, "query": "a"},
                {"id": "c2", "text": "B", "kind": "fact", "public": True, "relevant": True, "query": "b"},
            ],
        }
    )
    verdict = _wrap(
        {
            "verdicts": [
                {"id": "c1", "verdict": "verified", "evidence": "ok"},
                {"id": "c2", "verdict": "refuted", "evidence": "contradicted"},
            ]
        }
    )
    scorer = _verify_scorer(_RoutedJudgeApiClient(rubric=_LOW_GROUNDING, extract=extract, verdict=verdict))
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    assert result.metadata["aspect.grounding"] == 0.5  # 1 verified / (1 verified + 1 refuted)


def test_verify_grounding_sandbox_blocked_scores_zero_kept_in_denom(tmp_path: Path):
    extract = _wrap({"sandbox_blocked": True, "claims": []})
    scorer = _verify_scorer(_RoutedJudgeApiClient(rubric=_LOW_GROUNDING, extract=extract))
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="can't open the attachment", tool_calls=()),
    )
    assert result.metadata["aspect.grounding"] == 0.0
    assert result.metadata["grounding_status"] == "sandbox_blocked"
    assert "grounding" in result.metadata["rubric_gate_failures"]
    assert result.passed is False
    # kept in the denominator: still graded from the other aspects (not dropped)
    assert result.graded_score > 0.0


def test_verify_grounding_private_falls_back_to_process(tmp_path: Path):
    rubric_pass_grounding = _wrap(
        {
            "task_completion": {"items": {"tc1": "pass"}},
            "grounding": {"items": {"g1": "pass"}},
            "tool_use": {"score": 1},
            "answer_quality": {"score": 1},
            "error_recovery": {"score": 1},
            "efficiency": {"score": 1},
        }
    )
    extract = _wrap(
        {
            "sandbox_blocked": False,
            "claims": [{"id": "c1", "text": "PRIVATE my chat said hello", "kind": "fact", "public": False, "relevant": True, "query": ""}],
        }
    )
    scorer = _verify_scorer(_RoutedJudgeApiClient(rubric=rubric_pass_grounding, extract=extract))
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    assert result.metadata["grounding_status"] == "private_fallback"
    assert result.metadata["aspect.grounding"] == 1.0  # unchanged process score
    # private claim is hashed, never leaked into the report
    assert "PRIVATE" not in json.dumps(result.metadata, sort_keys=True)
    assert any(c["verdict"] == "unverifiable_private" for c in result.metadata["grounding_claims"])


def test_verify_grounding_default_process_mode_leaves_no_verify_metadata(tmp_path: Path):
    scorer = RubricJudgeScorer(api_client=_StaticJudgeApiClient(_LOW_GROUNDING), model="m", votes=1)
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    assert result.metadata["aspect.grounding"] == 0.0  # process score, unchanged
    assert "grounding_mode" not in result.metadata
    assert "grounding_status" not in result.metadata


def test_verify_grounding_action_claim_refuted_against_trajectory(tmp_path: Path):
    # answer claims it created a file; the trajectory shows no such success -> refuted
    extract = _wrap(
        {
            "sandbox_blocked": False,
            "claims": [
                {"id": "a1", "text": "Created report.html with the results", "kind": "action",
                 "public": False, "relevant": True, "query": ""},
            ],
        }
    )
    verdict = _wrap({"verdicts": [{"id": "a1", "verdict": "refuted", "evidence": "no write_file success in trajectory"}]})
    scorer = _verify_scorer(_RoutedJudgeApiClient(rubric=_LOW_GROUNDING, extract=extract, verdict=verdict))
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="Готово, сделал report.html", tool_calls=()),
    )
    assert result.metadata["aspect.grounding"] == 0.0  # 0 verified / 1 refuted
    assert result.metadata["grounding_status"] == "scored"
    assert result.metadata["grounding_refuted"] == 1
    assert result.metadata["grounding_claims"][0]["kind"] == "action"


@pytest.mark.asyncio
async def test_verify_grounding_artifact_action_supported_by_trajectory() -> None:
    extract = _wrap(
        {
            "sandbox_blocked": False,
            "claims": [
                {
                    "id": "a1",
                    "text": "Created and published index.html at https://example.test/index.html",
                    "kind": "action",
                    "public": False,
                    "relevant": True,
                    "query": "",
                }
            ],
        }
    )
    verdict = _wrap(
        {
            "verdicts": [
                {
                    "id": "a1",
                    "verdict": "verified",
                    "evidence": "write_file and publish succeeded",
                }
            ]
        }
    )
    client = _RoutedJudgeApiClient(rubric="", extract=extract, verdict=verdict)
    search_queries: list[str] = []

    async def fake_search(query: str, *, max_results: int = 5) -> str:
        del max_results
        search_queries.append(query)
        return "unexpected web search"

    result = await _verify_grounding(
        client,
        "m",
        task="Create and publish an HTML page.",
        answer="Created and published index.html at https://example.test/index.html",
        trajectory=(
            "tool: write_file args={'path': 'index.html'} output=ok\n"
            "tool: publish args={'path': 'index.html'} "
            "output=https://example.test/index.html"
        ),
        checklist_items=[],
        search=fake_search,
    )

    assert result["status"] == "scored"
    assert result["score"] == 1.0
    assert result["verified"] == 1
    assert result["refuted"] == 0
    assert result["claims"][0]["kind"] == "action"
    assert search_queries == []
    verdict_prompt = client.requests[-1].messages[0].text
    assert "write_file" in verdict_prompt
    assert "https://example.test/index.html" in verdict_prompt


@pytest.mark.asyncio
async def test_verify_grounding_artifact_action_fabricated_without_trajectory() -> None:
    extract = _wrap(
        {
            "sandbox_blocked": False,
            "claims": [
                {
                    "id": "a1",
                    "text": "Created and published index.html at https://example.test/index.html",
                    "kind": "action",
                    "public": False,
                    "relevant": True,
                    "query": "",
                }
            ],
        }
    )
    verdict = _wrap(
        {
            "verdicts": [
                {
                    "id": "a1",
                    "verdict": "refuted",
                    "evidence": "trajectory has no create or publish action",
                }
            ]
        }
    )
    client = _RoutedJudgeApiClient(rubric="", extract=extract, verdict=verdict)

    async def fake_search(query: str, *, max_results: int = 5) -> str:
        raise AssertionError(f"action claims should not web-search: {query} {max_results}")

    result = await _verify_grounding(
        client,
        "m",
        task="Create and publish an HTML page.",
        answer="Created and published index.html at https://example.test/index.html",
        trajectory="user: Create and publish an HTML page.\nassistant: I can do that.",
        checklist_items=[],
        search=fake_search,
    )

    assert result["status"] == "scored"
    assert result["score"] == 0.0
    assert result["verified"] == 0
    assert result["refuted"] == 1
    assert result["claims"][0]["verdict"] == "refuted"


@pytest.mark.asyncio
async def test_verify_grounding_empty_answer_still_sandbox_blocked() -> None:
    client = _RoutedJudgeApiClient(rubric="", extract="", verdict="")

    async def fake_search(query: str, *, max_results: int = 5) -> str:
        raise AssertionError(f"empty answers should not web-search: {query} {max_results}")

    result = await _verify_grounding(
        client,
        "m",
        task="Create and publish an HTML page.",
        answer="",
        trajectory="tool: write_file args={'path': 'index.html'} output=ok",
        checklist_items=[],
        search=fake_search,
    )

    assert result == {
        "score": 0.0,
        "status": "sandbox_blocked",
        "verified": 0,
        "refuted": 0,
        "claims": [],
    }
    assert client.requests == []


def test_verify_grounding_irrelevant_padding_claims_excluded(tmp_path: Path):
    # one task-relevant (verified) + one irrelevant padding claim -> only relevant counts
    extract = _wrap(
        {
            "sandbox_blocked": False,
            "claims": [
                {"id": "c1", "text": "The price is 950", "kind": "fact", "public": True, "relevant": True, "query": "price"},
                {"id": "pad", "text": "Paris is the capital of France", "kind": "fact", "public": True, "relevant": False, "query": "paris"},
            ],
        }
    )
    verdict = _wrap({"verdicts": [{"id": "c1", "verdict": "verified", "evidence": "matches"}]})
    scorer = _verify_scorer(_RoutedJudgeApiClient(rubric=_LOW_GROUNDING, extract=extract, verdict=verdict))
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    assert result.metadata["aspect.grounding"] == 1.0
    assert {c["id"] for c in result.metadata["grounding_claims"]} == {"c1"}  # padding excluded
    assert result.metadata["grounding_verified"] == 1


def test_serper_search_serves_from_cache_without_network(tmp_path: Path, monkeypatch):
    import hashlib

    from openharness.evals.executor import _run_eval_coroutine
    from openharness.evals.judge import _SEARCH_MEM, _serper_search

    monkeypatch.setenv("OPENHARNESS_SEARCH_CACHE", str(tmp_path))
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    _SEARCH_MEM.clear()
    key = hashlib.sha256("q|5".encode("utf-8")).hexdigest()[:32]
    (tmp_path / f"{key}.txt").write_text("CACHED RESULT", encoding="utf-8")
    # on-disk hit -> no httpx call; then promoted to the in-process map
    assert _run_eval_coroutine(_serper_search("q", max_results=5)) == "CACHED RESULT"
    assert _SEARCH_MEM[key] == "CACHED RESULT"


# --- verify-grounding vote stabilization (grounding_votes > 1) -----------------


class _SequencedJudgeClient:
    """Like _RoutedJudgeApiClient, but the extract and verdict routes each hand
    back the NEXT canned response per call, so a multi-vote verify run can be
    driven through a flapping verdict (or a shifting claim set) deterministically.
    Lists shorter than the vote count clamp to their last entry."""

    def __init__(self, *, rubric: str, extracts: list[str], verdicts: list[str]) -> None:
        self._rubric, self._extracts, self._verdicts = rubric, extracts, verdicts
        self._ei = self._vi = 0
        self.requests: list = []

    async def stream_message(self, request):
        self.requests.append(request)
        sp = (request.system_prompt or "").lower()
        if "extract the checkable" in sp:
            text = self._extracts[min(self._ei, len(self._extracts) - 1)]
            self._ei += 1
        elif "verify claims against evidence" in sp:
            text = self._verdicts[min(self._vi, len(self._verdicts) - 1)]
            self._vi += 1
        else:
            text = self._rubric
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=text)]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def _public_claim(query: str = "price") -> str:
    return _wrap(
        {
            "sandbox_blocked": False,
            "claims": [
                {"id": "c1", "text": "The price is 950", "kind": "fact",
                 "public": True, "relevant": True, "query": query},
            ],
        }
    )


def _private_claim() -> str:
    return _wrap(
        {
            "sandbox_blocked": False,
            "claims": [
                {"id": "c1", "text": "PRIVATE my chat said hello", "kind": "fact",
                 "public": False, "relevant": True, "query": ""},
            ],
        }
    )


def _verdict(v: str) -> str:
    return _wrap({"verdicts": [{"id": "c1", "verdict": v, "evidence": "e"}]})


def test_median_even_averages_two_middle():
    from openharness.evals.judge import _median

    assert _median([1.0]) == 1.0
    assert _median([0.0, 1.0]) == 0.5
    assert _median([0.0, 1.0, 1.0]) == 1.0
    assert _median([0.2, 0.4, 0.6, 0.8]) == 0.5


def test_verify_votes_median_smooths_verdict_flip(tmp_path: Path):
    # one relevant public claim; the verdict flaps verified/refuted/verified across
    # the 3 votes. A single shot would land 1.0 or 0.0 by luck; the median is 1.0.
    client = _SequencedJudgeClient(
        rubric=_LOW_GROUNDING,
        extracts=[_public_claim()],  # stable claim set every vote
        verdicts=[_verdict("verified"), _verdict("refuted"), _verdict("verified")],
    )
    scorer = _verify_scorer(client, grounding_votes=3)
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    assert result.metadata["aspect.grounding"] == 1.0  # median of 1.0 / 0.0 / 1.0
    assert result.metadata["grounding_status"] == "scored"
    assert result.metadata["grounding_votes"] == 3


def test_verify_votes_refuted_majority_median_fails_gate(tmp_path: Path):
    # the median swings the OTHER way when refuted dominates -> below the gate.
    client = _SequencedJudgeClient(
        rubric=_LOW_GROUNDING,
        extracts=[_public_claim()],
        verdicts=[_verdict("refuted"), _verdict("verified"), _verdict("refuted")],
    )
    scorer = _verify_scorer(client, grounding_votes=3)
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    assert result.metadata["aspect.grounding"] == 0.0  # median of 0.0 / 1.0 / 0.0
    assert "grounding" in result.metadata["rubric_gate_failures"]
    assert result.passed is False


def test_verify_votes_minority_scored_keeps_process_grounding(tmp_path: Path):
    # 2 runs go private_fallback, 1 goes scored -> class-majority keeps process,
    # so a lone noisy scored run cannot flip the override.
    rubric_pass = _wrap(
        {
            "task_completion": {"items": {"tc1": "pass"}},
            "grounding": {"items": {"g1": "pass"}},
            "tool_use": {"score": 1}, "answer_quality": {"score": 1},
            "error_recovery": {"score": 1}, "efficiency": {"score": 1},
        }
    )
    client = _SequencedJudgeClient(
        rubric=rubric_pass,
        extracts=[_private_claim(), _private_claim(), _public_claim()],
        verdicts=[_verdict("verified")],  # only the one scored run reaches a verdict
    )
    scorer = _verify_scorer(client, grounding_votes=3)
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    assert result.metadata["grounding_status"] == "private_fallback"
    assert result.metadata["aspect.grounding"] == 1.0  # untouched process grounding
    assert result.metadata["grounding_votes"] == 3


def test_verify_votes_tie_break_prefers_scored(tmp_path: Path):
    # N=2, one scored + one private_fallback -> the tie resolves to scored.
    client = _SequencedJudgeClient(
        rubric=_LOW_GROUNDING,
        extracts=[_public_claim(), _private_claim()],
        verdicts=[_verdict("verified")],
    )
    scorer = _verify_scorer(client, grounding_votes=2)
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    assert result.metadata["grounding_status"] == "scored"
    assert result.metadata["aspect.grounding"] == 1.0  # process 0.0 overridden
    assert result.metadata["grounding_votes"] == 2


def test_verify_votes_reuse_cached_search_across_votes(tmp_path: Path):
    # 3 votes over the same claim/query -> a cached search fetches only once, which
    # is why extra votes bill extract/verdict tokens but not extra search calls.
    fetches = {"n": 0}
    cache: dict[str, str] = {}

    async def counting_search(query, *, max_results=4):
        if query not in cache:
            fetches["n"] += 1
            cache[query] = f"results for {query}"
        return cache[query]

    client = _RoutedJudgeApiClient(
        rubric=_LOW_GROUNDING, extract=_public_claim("price"), verdict=_verdict("verified")
    )
    scorer = RubricJudgeScorer(
        api_client=client, model="m", votes=1, grounding_mode="verify",
        search=counting_search, grounding_votes=3,
    )
    result = scorer.score(
        context=_context(tmp_path),
        executor_result=EvalExecutorResult(final_text="ans", tool_calls=()),
    )
    assert result.metadata["grounding_votes"] == 3
    assert result.metadata["aspect.grounding"] == 1.0
    assert fetches["n"] == 1  # cached across the 3 votes
