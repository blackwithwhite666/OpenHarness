"""LLM judge scorer for outcome-by-trajectory evals."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# trajectory_judge_v2 — multi-aspect, checklist-gated quality judge
# ---------------------------------------------------------------------------
#
# v1 returns a single binary PASS/FAIL. v2 grades the run across quality
# ASPECTS and returns a graded [0,1] score, so a live "faithful" run yields a
# quality signal instead of a bimodal pass/fail. Two aspects (task_completion,
# grounding) can be judged against a per-case checklist DERIVED offline from the
# gold episode (Agent-as-a-Judge / SWE-bench-derives-tests style: the gold is
# used to distill requirements, never to match the path); the rest use a generic
# rubric. See adrs/ohmo-eval-goldens-flywheel.md.


@dataclass(frozen=True)
class _AspectSpec:
    key: str
    weight: float
    gate_floor: float | None  # if set, aspect < floor => the run fails outright
    checklist: bool  # may be scored against a derived per-case checklist
    rubric: str


# Weights sum to 1.0. task_completion + grounding are the hard gates.
TRAJECTORY_JUDGE_V2_ASPECTS: tuple[_AspectSpec, ...] = (
    _AspectSpec(
        "task_completion", 0.35, 0.5, True,
        "Did the agent accomplish EVERYTHING the user asked -- correct, "
        "responsive results with every explicit constraint satisfied?",
    ),
    _AspectSpec(
        "grounding", 0.20, 0.6, True,
        "Is every factual claim and action in the answer supported by an actual "
        "observed tool output or the given context (no fabrication; no 'not "
        "found' unless a real search showed it absent)?",
    ),
    _AspectSpec(
        "tool_use", 0.15, None, False,
        "Were the chosen tools and arguments appropriate (right capability, sane "
        "args, the necessary calls present, no needless or destructive calls)? "
        "A different-but-valid tool path is fine.",
    ),
    _AspectSpec(
        "answer_quality", 0.15, None, False,
        "Is the final answer correct, complete, relevant and clear for the user, "
        "judged semantically (NOT by matching the reference wording)?",
    ),
    _AspectSpec(
        "error_recovery", 0.08, None, False,
        "When a tool errored or returned something unexpected, did the agent "
        "notice and adapt/recover rather than ignore it or proceed confidently "
        "wrong? (Full credit if no error arose.)",
    ),
    _AspectSpec(
        "efficiency", 0.07, None, False,
        "Did the agent reach the goal without wasteful loops or clearly "
        "redundant steps? Do NOT reward verbosity.",
    ),
)

TRAJECTORY_JUDGE_V2_SYSTEM_PROMPT = (
    "You are a rigorous evaluation judge for a live AI agent. Score the agent's "
    "task execution across quality ASPECTS, judging from the user's request, the "
    "observed trajectory (tool calls AND their outputs), and the final answer. "
    "Any reference solution shown is ONE valid way to solve the task -- never "
    "require the agent to match its wording, tools, or order; a different-but-"
    "valid path that meets the requirements earns full marks. Observed tool "
    "OUTPUTS are authoritative: if the final answer contradicts what the tools "
    "returned, trust the tools. Reason briefly, then output ONE fenced ```json "
    "block. Be decisive and consistent: identical executions get identical scores."
)

RUBRIC_DERIVE_SYSTEM_PROMPT = (
    "You extract an evaluation checklist from ONE correct reference solution of "
    "an agent task. Produce atomic, path-independent, verifiable requirements: "
    "what MUST be true for the task to count as solved, regardless of which tools "
    "or order an agent uses. Do NOT reference specific tool names or the exact "
    "wording of the reference. Output ONE fenced ```json block."
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
_VERDICT_SCORE = {
    "pass": 1.0, "yes": 1.0, "true": 1.0, "met": 1.0,
    "partial": 0.5, "partially": 0.5,
    "fail": 0.0, "no": 0.0, "false": 0.0, "unmet": 0.0,
}


class TrajectoryJudgeScorerV2:
    """Multi-aspect, checklist-gated LLM judge returning a graded [0,1] score."""

    name = "trajectory_judge_v2"
    requires_exact_tool_sequence = False

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        model: str,
        rubrics: dict[str, dict[str, object]] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1400,
        votes: int = 3,
        pass_threshold: float = 0.75,
        output_excerpt_chars: int = 400,
    ) -> None:
        self._api_client = api_client
        self._model = model
        self._rubrics = rubrics or {}
        self._system_prompt = system_prompt or TRAJECTORY_JUDGE_V2_SYSTEM_PROMPT
        self._max_tokens = max_tokens
        self._votes = max(1, int(votes))
        self._pass_threshold = pass_threshold
        self._excerpt = output_excerpt_chars

    def _rubric_for(self, context: EvalExecutionContext) -> dict[str, object]:
        case = context.case
        for key in (getattr(case, "case_id", None), getattr(case, "gold_case_id", None)):
            if key and key in self._rubrics:
                return self._rubrics[key]
        return {}

    def score(
        self,
        *,
        context: EvalExecutionContext,
        executor_result: EvalExecutorResult,
    ) -> EvalExecutionScorerResult:
        from openharness.evals.execution import EvalExecutionScorerResult

        rubric = self._rubric_for(context)
        prompt = _v2_judge_prompt(
            user_goal=context.primary_prompt,
            trajectory=_v2_trajectory(executor_result, excerpt=self._excerpt),
            final_answer=executor_result.final_text,
            reference=context.expected_final_text,
            rubric=rubric,
        )
        votes = [_parse_v2_scores(_run_eval_coroutine(self._complete(prompt))) for _ in range(self._votes)]
        aspects, graded, passed, gate_fails = _aggregate_v2(votes)
        verdict = "pass" if passed else ("error" if not aspects else "fail")
        metadata: dict[str, object] = {
            "judge_model": self._model,
            "verdict": verdict,
            "judge_votes": self._votes,
            "judge_parsed_votes": sum(1 for v in votes if v),
            "judge_v2_graded_score": round(graded, 4),
            "judge_v2_gate_failures": list(gate_fails),
            "judge_v2_used_checklist": bool(rubric),
        }
        for key, value in aspects.items():
            metadata[f"aspect.{key}"] = round(value, 4)
        return EvalExecutionScorerResult(
            passed=passed,
            score=graded,
            graded_score=graded,
            scorer_name=self.name,
            metadata=metadata,
            raw_reason="; ".join(f"{k}={round(v, 2)}" for k, v in aspects.items()),
        )

    async def _complete(self, prompt: str) -> str:
        return await _complete_text(
            self._api_client,
            self._model,
            system_prompt=self._system_prompt,
            prompt=prompt,
            max_tokens=self._max_tokens,
        )


async def _complete_text(
    api_client: SupportsStreamingMessages,
    model: str,
    *,
    system_prompt: str,
    prompt: str,
    max_tokens: int,
) -> str:
    text = ""
    async for event in api_client.stream_message(
        ApiMessageRequest(
            model=model,
            messages=[ConversationMessage.from_user_text(prompt)],
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            tools=[],
        )
    ):
        if isinstance(event, ApiMessageCompleteEvent):
            text = event.message.text.strip()
    return text


def _v2_trajectory(executor_result: EvalExecutorResult, *, excerpt: int) -> str:
    rows = []
    for index, call in enumerate(executor_result.tool_calls, 1):
        output = (call.output or "").strip().replace("\n", " ")
        if len(output) > excerpt:
            output = output[:excerpt] + "…"
        rows.append(
            {
                "step": index,
                "tool": effective_tool_label(call.tool_name, call.arguments),
                "is_error": bool(call.is_error),
                "output": output,
            }
        )
    return json.dumps(rows, ensure_ascii=True, indent=2) if rows else "No observed tool calls."


def _checklist_block(rubric: dict[str, object], aspect: str) -> str:
    items = rubric.get(aspect) if isinstance(rubric, dict) else None
    if not isinstance(items, list) or not items:
        return ""
    lines = []
    for item in items:
        if isinstance(item, dict):
            iid = str(item.get("id") or len(lines) + 1)
            text = str(item.get("text") or item)
        else:
            iid, text = str(len(lines) + 1), str(item)
        lines.append(f'    - id "{iid}": {text}')
    return (
        f"\n  For {aspect}, judge EACH checklist item pass/partial/fail and return "
        f'them as "items" {{id: verdict}}:\n' + "\n".join(lines)
    )


def _v2_judge_prompt(
    *,
    user_goal: str,
    trajectory: str,
    final_answer: str,
    reference: str,
    rubric: dict[str, object],
) -> str:
    aspect_lines = []
    for spec in TRAJECTORY_JUDGE_V2_ASPECTS:
        block = _checklist_block(rubric, spec.key) if spec.checklist else ""
        scale = (
            'per-item verdict' if block else 'a "score" of 0, 0.5, or 1'
        )
        aspect_lines.append(f'- "{spec.key}" — {spec.rubric} ({scale}){block}')
    reference_block = (
        "\n\nReference solution (ONE valid way — do NOT require a match):\n"
        f"{reference.strip()}"
        if reference.strip()
        else ""
    )
    schema = (
        '{"task_completion": {"items": {"tc1": "pass"}}, '
        '"grounding": {"items": {"g1": "pass"}}, '
        '"tool_use": {"score": 1}, "answer_quality": {"score": 1}, '
        '"error_recovery": {"score": 1}, "efficiency": {"score": 1}}'
    )
    return (
        "Score the agent's execution on each aspect below. For a checklist "
        "aspect, mark every item pass/partial/fail; otherwise give score 0/0.5/1. "
        "Reason briefly, then output ONE fenced ```json block matching the schema.\n\n"
        "Aspects:\n" + "\n".join(aspect_lines) + "\n\n"
        f"JSON schema (checklist aspects use \"items\", others use \"score\"):\n```json\n{schema}\n```\n\n"
        f"User goal:\n{user_goal.strip()}\n\n"
        f"Observed trajectory (tool calls + outputs):\n{trajectory}\n\n"
        f"Agent final answer:\n{final_answer.strip()}"
        f"{reference_block}"
    )


def _extract_json(text: str) -> dict | None:
    match = _JSON_FENCE_RE.search(text)
    raw = match.group(1) if match else None
    if raw is None:
        start, end = text.find("{"), text.rfind("}")
        raw = text[start : end + 1] if start != -1 and end > start else None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_score(value: object) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        return _VERDICT_SCORE.get(value.strip().lower())
    return None


def _aspect_vote_score(entry: object) -> float | None:
    if isinstance(entry, (int, float, str, bool)):
        return _coerce_score(entry)
    if not isinstance(entry, dict):
        return None
    items = entry.get("items")
    verdicts: list[float] = []
    if isinstance(items, dict):
        verdicts = [s for v in items.values() if (s := _coerce_score(v)) is not None]
    elif isinstance(items, list):
        for it in items:
            v = it.get("verdict") if isinstance(it, dict) else it
            s = _coerce_score(v)
            if s is not None:
                verdicts.append(s)
    if verdicts:
        return sum(verdicts) / len(verdicts)
    if "score" in entry:
        return _coerce_score(entry["score"])
    return None


def _parse_v2_scores(text: str) -> dict[str, float]:
    parsed = _extract_json(text)
    if not parsed:
        return {}
    scores: dict[str, float] = {}
    for spec in TRAJECTORY_JUDGE_V2_ASPECTS:
        if spec.key in parsed:
            value = _aspect_vote_score(parsed[spec.key])
            if value is not None:
                scores[spec.key] = value
    return scores


def _aggregate_v2(
    votes: list[dict[str, float]],
) -> tuple[dict[str, float], float, bool, tuple[str, ...]]:
    """Average aspect scores across votes -> (aspects, graded, passed, gate_failures)."""
    aspects: dict[str, float] = {}
    for spec in TRAJECTORY_JUDGE_V2_ASPECTS:
        vals = [v[spec.key] for v in votes if spec.key in v]
        if vals:
            aspects[spec.key] = sum(vals) / len(vals)
    if not aspects:
        return {}, 0.0, False, ()
    total_weight = sum(spec.weight for spec in TRAJECTORY_JUDGE_V2_ASPECTS if spec.key in aspects)
    graded = (
        sum(spec.weight * aspects[spec.key] for spec in TRAJECTORY_JUDGE_V2_ASPECTS if spec.key in aspects)
        / total_weight
        if total_weight
        else 0.0
    )
    gate_failures = tuple(
        spec.key
        for spec in TRAJECTORY_JUDGE_V2_ASPECTS
        if spec.gate_floor is not None
        and spec.key in aspects
        and aspects[spec.key] < spec.gate_floor
    )
    # A pass needs the weighted quality bar AND every hard gate satisfied.
    return aspects, graded, (not gate_failures and graded >= 0.75), gate_failures


def _rubric_derive_prompt(*, goal: str, gold_trajectory: str, gold_answer: str) -> str:
    schema = (
        '{"task_completion": [{"id": "tc1", "text": "..."}], '
        '"grounding": [{"id": "g1", "text": "..."}]}'
    )
    return (
        "Below is ONE correct reference solution of an agent task. Extract two "
        "checklists of atomic, path-independent requirements:\n"
        '- "task_completion": outcomes/constraints that MUST hold for the task to '
        "be solved (subgoals, required values/effects, format/constraint items). "
        "3-7 items.\n"
        '- "grounding": facts the final answer asserts that MUST be backed by a '
        "tool result or given context. 1-5 items.\n"
        "Each item is a single yes/no-checkable requirement, tool-agnostic. Output "
        f"ONE fenced ```json block:\n```json\n{schema}\n```\n\n"
        f"User goal:\n{goal.strip()}\n\n"
        f"Reference trajectory (tool calls + outputs):\n{gold_trajectory}\n\n"
        f"Reference final answer:\n{gold_answer.strip()}"
    )


def _parse_rubric(text: str) -> dict[str, list[dict[str, str]]]:
    parsed = _extract_json(text) or {}
    out: dict[str, list[dict[str, str]]] = {}
    for aspect in ("task_completion", "grounding"):
        items = parsed.get(aspect)
        if not isinstance(items, list):
            continue
        cleaned: list[dict[str, str]] = []
        for index, item in enumerate(items, 1):
            if isinstance(item, dict) and item.get("text"):
                cleaned.append(
                    {"id": str(item.get("id") or f"{aspect[:2]}{index}"), "text": str(item["text"])}
                )
            elif isinstance(item, str) and item.strip():
                cleaned.append({"id": f"{aspect[:2]}{index}", "text": item.strip()})
        if cleaned:
            out[aspect] = cleaned
    return out


def derive_case_rubric(
    *,
    api_client: SupportsStreamingMessages,
    model: str,
    goal: str,
    gold_trajectory: str,
    gold_answer: str,
    max_tokens: int = 1200,
) -> dict[str, list[dict[str, str]]]:
    """Derive per-case task_completion + grounding checklists from a gold episode."""
    text = _run_eval_coroutine(
        _complete_text(
            api_client,
            model,
            system_prompt=RUBRIC_DERIVE_SYSTEM_PROMPT,
            prompt=_rubric_derive_prompt(
                goal=goal, gold_trajectory=gold_trajectory, gold_answer=gold_answer
            ),
            max_tokens=max_tokens,
        )
    )
    return _parse_rubric(text)
