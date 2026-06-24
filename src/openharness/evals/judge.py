"""LLM judge scorer for outcome-by-trajectory evals."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    SupportsStreamingMessages,
)
from openharness.engine.messages import ConversationMessage
from openharness.evals.executor import (
    EvalExecutionContext,
    EvalExecutorResult,
    _run_eval_coroutine,
)
from openharness.evals.tool_labels import effective_tool_label

if TYPE_CHECKING:
    from openharness.evals.execution import EvalExecutionScorerResult


DEFAULT_TRAJECTORY_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge. Decide whether the agent ACCOMPLISHED THE "
    "USER'S REQUEST, judging from the observed trajectory and the final answer. "
    "The gold answer is ONE acceptable reference, NOT a required template: the "
    "agent's answer need not match its wording, structure, ordering, or level of "
    "detail. PASS when the answer is correct and responsive to what the user "
    "actually asked -- even if it is shorter, organized differently, or omits "
    "extra facts the gold happened to include. Tolerate a different-but-valid "
    "tool path. FAIL only if the answer is wrong, off-topic, or omits something "
    "the USER EXPLICITLY asked for. Reply with the first word PASS or FAIL, then "
    "one short sentence explaining why."
)
_VERDICT_RE = re.compile(r"^\s*([A-Za-z]+)\b(.*)$", re.DOTALL)


class TrajectoryJudgeScorer:
    """Score success by asking an LLM to judge trajectory plus outcome."""

    name = "trajectory_judge_v1"
    requires_exact_tool_sequence = False

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        model: str,
        system_prompt: str = DEFAULT_TRAJECTORY_JUDGE_SYSTEM_PROMPT,
        max_tokens: int = 512,
        min_chars: int = 1,
    ) -> None:
        self._api_client = api_client
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self._min_chars = min_chars

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        from openharness.evals.execution import EvalExecutionScorerResult

        observed = _observed_capabilities(executor_result)
        prompt = _judge_prompt(
            user_goal=context.primary_prompt,
            observed=observed,
            final_answer=executor_result.final_text,
            accepted_outcome=context.expected_final_text,
        )
        response = _run_eval_coroutine(self._complete(prompt))
        passed, verdict, reason = _parse_verdict(response, min_chars=self._min_chars)
        return EvalExecutionScorerResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            scorer_name=self.name,
            metadata={
                "judge_model": self._model,
                "verdict": verdict,
                "reason_hash": _hash_text(reason),
                "reason_length": len(reason),
                "observed_capability_count": len(observed),
                "had_tool_error": any(item["is_error"] for item in observed),
            },
        )

    async def _complete(self, prompt: str) -> str:
        text = ""
        async for event in self._api_client.stream_message(
            ApiMessageRequest(
                model=self._model,
                messages=[ConversationMessage.from_user_text(prompt)],
                system_prompt=self._system_prompt,
                max_tokens=self._max_tokens,
                tools=[],
            )
        ):
            if isinstance(event, ApiMessageCompleteEvent):
                text = event.message.text.strip()
        return text


def _observed_capabilities(
    executor_result: EvalExecutorResult,
) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "capability": effective_tool_label(call.tool_name, call.arguments),
            "is_error": bool(call.is_error),
        }
        for index, call in enumerate(executor_result.tool_calls, 1)
    ]


def _judge_prompt(
    *,
    user_goal: str,
    observed: list[dict[str, object]],
    final_answer: str,
    accepted_outcome: str,
) -> str:
    trajectory = (
        json.dumps(observed, ensure_ascii=True, indent=2)
        if observed
        else "No observed tool calls."
    )
    reference = (
        "\n\nONE acceptable reference answer (NOT a required template -- the "
        "agent's answer need not match its wording, structure, or completeness):\n"
        f"{accepted_outcome.strip()}"
        if accepted_outcome.strip()
        else ""
    )
    return (
        "Did the agent accomplish the user's request? Grade task accomplishment, "
        "NOT similarity to the reference. PASS a correct, responsive answer even "
        "if it is shorter or organized differently than the reference; FAIL only "
        "if it is wrong, off-topic, or misses something the user EXPLICITLY asked "
        "for. Tolerate a different-but-valid tool path. Reply with the FIRST word "
        "PASS or FAIL, then one short sentence why.\n\n"
        f"User goal:\n{user_goal.strip()}\n\n"
        f"Observed trajectory:\n{trajectory}\n\n"
        f"Agent final answer:\n{final_answer.strip()}"
        f"{reference}"
    )


def _parse_verdict(text: str, *, min_chars: int) -> tuple[bool, str, str]:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False, "error", ""
    match = _VERDICT_RE.match(stripped)
    if match is None:
        return False, "error", stripped
    token = match.group(1).lower()
    reason = match.group(2).strip(" \t\r\n:-")
    if token == "pass":
        return True, "pass", reason
    if token == "fail":
        return False, "fail", reason
    return False, "error", stripped


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
