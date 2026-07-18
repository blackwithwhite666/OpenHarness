"""Tests for memory recall agent-use and grounding scores."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ohmo.evals.memory.benchmark import MemoryCase, QueryObservation, Turn
from ohmo.evals.memory.provisioning import provision_backend
from ohmo.evals.memory.recall_judge import answer_turn, judge_recall, score_case


class FakeCompleter:
    def __init__(self, respond: Callable[[str], str]) -> None:
        self._respond = respond
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._respond(prompt)


async def test_recalled_fact_used_by_agent_is_grounded_and_scores_high() -> None:
    fake = FakeCompleter(lambda _prompt: "The luggage tag identifier is ORCHID-482.")

    answer = await answer_turn(
        "What is the luggage tag identifier?",
        "The luggage tag identifier is ORCHID-482.",
        complete=fake,
    )
    verdict = await judge_recall(
        _observation(expected_hit={"ORCHID-482": True}),
        answer,
    )

    assert len(fake.prompts) == 1
    assert "using ONLY facts in the provided memory block" in fake.prompts[0]
    assert "ORCHID-482" in fake.prompts[0]
    assert verdict.recalled_into_context is True
    assert verdict.used_in_answer is True
    assert verdict.grounded is True
    assert verdict.score == 1.0


async def test_recalled_fact_ignored_by_agent_is_not_used() -> None:
    verdict = await judge_recall(
        _observation(expected_hit={"ORCHID-482": True}),
        "Sea turtles migrate long distances.",
    )

    assert verdict.recalled_into_context is True
    assert verdict.used_in_answer is False
    assert verdict.grounded is True
    assert verdict.score == 0.0


@pytest.mark.parametrize(
    ("answer", "expected_grounded"),
    [
        ("The Neptune vacation cabin code is NEPTUNE-9.", False),
        ("I don't know; the supplied memory does not contain a cabin code.", True),
    ],
)
async def test_no_fabrication_answer_grounding(
    answer: str,
    expected_grounded: bool,
) -> None:
    verdict = await judge_recall(
        _observation(expected_hit={}, must_not_leak={"NEPTUNE-9": False}),
        answer,
    )

    assert verdict.recalled_into_context is True
    assert verdict.used_in_answer is True
    assert verdict.grounded is expected_grounded
    assert verdict.score == float(expected_grounded)


async def test_semantic_fallback_detects_paraphrased_use() -> None:
    fake = FakeCompleter(lambda prompt: "YES" if prompt.startswith("SEMANTIC USE") else "NO")

    verdict = await judge_recall(
        _observation(expected_hit={"112 centimeters": True}),
        "The current desk setting is one hundred twelve cm.",
        complete=fake,
    )

    assert verdict.used_in_answer is True
    assert verdict.grounded is True
    assert verdict.score == 1.0
    assert len(fake.prompts) == 1
    assert fake.prompts[0].startswith("SEMANTIC USE CHECK")


async def test_score_case_catalog_end_to_end_with_fake_completer() -> None:
    case = MemoryCase(
        id="workshop-code",
        category="write-early/recall-late",
        seed_entries=[("Workshop Access", "The workshop access code is MARIGOLD-17.")],
        turns=[
            Turn(
                kind="query",
                query="workshop access code",
                expected_recall=["MARIGOLD-17"],
                must_not_recall=[],
            )
        ],
    )
    provisioned = await provision_backend(
        "catalog",
        run="recall-judge",
        case=case.id,
        sample=0,
        seed_entries=case.seed_entries,
    )
    fake = FakeCompleter(lambda _prompt: "The workshop access code is MARIGOLD-17.")

    try:
        result = await score_case(case, provisioned, complete=fake)
    finally:
        await provisioned.teardown()

    assert result.case_id == case.id
    assert result.category == case.category
    assert result.mean_score == 1.0
    assert result.recall_rate == 1.0
    assert result.use_rate == 1.0
    assert result.grounded_rate == 1.0
    assert len(fake.prompts) == 1
    assert "MARIGOLD-17" in fake.prompts[0]


def _observation(
    *,
    expected_hit: dict[str, bool],
    must_not_leak: dict[str, bool] | None = None,
) -> QueryObservation:
    return QueryObservation(
        query="memory query",
        surfaced=["memory.md"],
        surfaced_content=["memory content"],
        expected_hit=expected_hit,
        must_not_leak=must_not_leak or {},
    )
