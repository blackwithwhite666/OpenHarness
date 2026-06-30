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
    "the USER EXPLICITLY asked for. "
    "ABSENCE-CLAIM RULE: if the answer says information is unavailable / not "
    "found / inaccessible, treat that as a substantive claim -- PASS it only when "
    "the information is genuinely absent; FAIL it when the reference or the "
    "observed trajectory shows the information was in fact reachable. "
    "Be decisive and consistent: identical answers must get the same verdict. "
    "Reply with the first word PASS or FAIL, then one short sentence explaining why."
)
GROUNDING_TRAJECTORY_JUDGE_SYSTEM_PROMPT = (
    "You are an evaluation judge for a TIME-SENSITIVE / live task whose facts "
    "change over time, so the reference answer may be STALE. Judge METHOD and "
    "GROUNDING, not fact-match. PASS when the agent consulted appropriate "
    "sources/tools for the request and reported concrete findings grounded in "
    "them (or correctly determined none exist after a real search). FAIL when it "
    "used the wrong sources or none, fabricated, gave a vacuous/ungrounded "
    "answer, or claimed unavailable without a genuine search. Do NOT penalize "
    "differences from the reference answer's specific facts/numbers/dates. Reply "
    "with the first word PASS or FAIL, then one short sentence explaining why."
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
        system_prompt: str | None = None,
        max_tokens: int = 512,
        min_chars: int = 1,
        votes: int = 3,
        grounding_mode: bool = False,
    ) -> None:
        self._api_client = api_client
        self._model = model
        self._grounding = grounding_mode
        self._system_prompt = system_prompt or (
            GROUNDING_TRAJECTORY_JUDGE_SYSTEM_PROMPT
            if grounding_mode
            else DEFAULT_TRAJECTORY_JUDGE_SYSTEM_PROMPT
        )
        self._max_tokens = max_tokens
        self._min_chars = min_chars
        self._votes = max(1, int(votes))

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
            grounding=self._grounding,
        )
        # Judge the SAME trajectory N times and take the majority verdict; an
        # LLM judge flips on borderline-equivalent answers, so a single vote is
        # the dominant source of run-to-run flap. Ties resolve to FAIL.
        verdicts: list[tuple[bool, str, str]] = [
            _parse_verdict(
                _run_eval_coroutine(self._complete(prompt)),
                min_chars=self._min_chars,
            )
            for _ in range(self._votes)
        ]
        pass_votes = sum(1 for _, v, _ in verdicts if v == "pass")
        fail_votes = sum(1 for _, v, _ in verdicts if v == "fail")
        if pass_votes + fail_votes == 0:
            # Every vote was unparseable -> preserve the distinct error verdict.
            passed = False
            verdict = "error"
        else:
            passed = pass_votes > fail_votes  # tie among valid votes -> fail
            verdict = "pass" if passed else "fail"
        # Surface a reason from the winning side (fall back to any reason).
        reason = next(
            (r for p, v, r in verdicts if v == verdict and r),
            next((r for _, _, r in verdicts if r), ""),
        )
        return EvalExecutionScorerResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            scorer_name=self.name,
            metadata={
                "judge_model": self._model,
                "verdict": verdict,
                "judge_votes": self._votes,
                "judge_pass_votes": pass_votes,
                "reason_hash": _hash_text(reason),
                "reason_length": len(reason),
                "observed_capability_count": len(observed),
                "had_tool_error": any(item["is_error"] for item in observed),
            },
            raw_reason=reason,
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
    grounding: bool = False,
) -> str:
    trajectory = (
        json.dumps(observed, ensure_ascii=True, indent=2)
        if observed
        else "No observed tool calls."
    )
    if grounding:
        reference = (
            "\n\nPossibly-stale reference (facts may have changed; do NOT "
            f"fact-match against it):\n{accepted_outcome.strip()}"
            if accepted_outcome.strip()
            else ""
        )
        instruction = (
            "This is a TIME-SENSITIVE task: judge METHOD and GROUNDING, not "
            "fact-match. PASS if the agent used appropriate sources/tools and "
            "reported concrete findings grounded in them (or correctly found none "
            "after a real search); FAIL if it used wrong/no sources, fabricated, "
            "answered vacuously, or claimed unavailable without searching. Ignore "
            "differences from the reference's specific facts. Reply with the FIRST "
            "word PASS or FAIL, then one short sentence why.\n\n"
        )
    else:
        reference = (
            "\n\nONE acceptable reference answer (NOT a required template -- the "
            "agent's answer need not match its wording, structure, or completeness):\n"
            f"{accepted_outcome.strip()}"
            if accepted_outcome.strip()
            else ""
        )
        instruction = (
            "Did the agent accomplish the user's request? Grade task accomplishment, "
            "NOT similarity to the reference. PASS a correct, responsive answer even "
            "if it is shorter or organized differently than the reference; FAIL only "
            "if it is wrong, off-topic, or misses something the user EXPLICITLY asked "
            "for. Tolerate a different-but-valid tool path. If the answer claims the "
            "info is unavailable/not found, FAIL when the reference or trajectory shows "
            "it was reachable. Reply with the FIRST word PASS or FAIL, then one short "
            "sentence why.\n\n"
        )
    return (
        f"{instruction}"
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
