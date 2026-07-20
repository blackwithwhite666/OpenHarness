"""LLM judge scorer for outcome-by-trajectory evals."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from collections.abc import Sequence
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


FREEZING_JUDGE_SYSTEM_PROMPT = (
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

# A side-effecting tool can only be simulated (or can fail for user-namespace
# reasons) in the faithful fs-sandbox. Keep these authoritative trajectory
# markers explicit and easy to extend; ordinary mentions of "mock" do not count.
SANDBOX_SIDE_EFFECT_CREDIT_SENTINELS: frozenset[str] = frozenset(
    {
        '"mock": true',
        "No user exists for uid",
    }
)
SANDBOX_SIDE_EFFECT_CREDIT_METADATA_TOKENS: frozenset[str] = frozenset({"mock"})
SANDBOX_SIDE_EFFECT_CREDIT_RULE = (
    "When the observed trajectory shows the assistant correctly invoked a "
    "side-effecting tool or skill but the result is a sandbox mock (marked "
    '`"mock": true` or metadata token `mock`) or a sandbox-environment failure '
    "(for example, `No user exists for uid ...`), treat that action as COMPLETED "
    "for both intent_met and constraints_held; judge the assistant's DECISION and "
    "tool-invocation correctness, NOT whether the real external side effect occurred."
)
GROUNDING_FREEZING_JUDGE_SYSTEM_PROMPT = (
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


class FreezingJudgeScorer:
    """Score success by asking an LLM to judge trajectory plus outcome."""

    name = "freezing_judge"
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
            GROUNDING_FREEZING_JUDGE_SYSTEM_PROMPT
            if grounding_mode
            else FREEZING_JUDGE_SYSTEM_PROMPT
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
# rubric_judge — multi-aspect, checklist-gated quality judge
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
RUBRIC_JUDGE_ASPECTS: tuple[_AspectSpec, ...] = (
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

RUBRIC_JUDGE_SYSTEM_PROMPT = (
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

# verification-grounding — score the TRUTH of the answer's factual claims against
# INDEPENDENT web retrieval, not whether each claim was tool-cited. See ADR
# adrs/ohmo-eval-verification-grounding.md. Two orchestrated LLM calls (extract
# claims, then verdict-against-evidence) with a web_search in between.
GROUNDING_EXTRACT_SYSTEM_PROMPT = (
    "You extract the checkable CLAIMS from an AI agent's answer so they can be "
    "verified. A claim is a concrete assertion: a price, time, rating, address, "
    "identity, count, code, opening hours, an explicit 'X is unavailable', OR an "
    "assertion that the agent itself did/produced something. IGNORE hedges, "
    "opinions, offers, recommendations, and meta-talk. For each claim set:\n"
    "- kind: 'action' if it asserts something the AGENT ITSELF did or produced "
    "(created/wrote/edited a file, published a URL or path, sent a message, saved "
    "data). This includes the CONTENT, STRUCTURE, or EXISTENCE of a file, report, "
    "or artifact the agent created or edited with write_file/edit_file. Example: "
    "'The report/file contains X' -> kind='action'. Use 'fact' for claims about "
    "the external world that are checkable on the public web, not for the agent's "
    "own authored artifact.\n"
    "- public: for a 'fact', true if verifiable on the open web, false if it is "
    "about the USER'S OWN private data (chats, calendar, files, config, memory). "
    "('action' claims are checked against the agent's own trajectory, not the "
    "web -- set public=false for them.)\n"
    "- relevant: true ONLY if the claim is central to answering the user's "
    "request (or matches one of the task's key facts); false for incidental or "
    "padding trivia that does not address what the user actually asked.\n"
    "If the answer says the agent created, wrote, saved, attached, or published "
    "an artifact/file/URL, extract that as kind='action' so it can be checked "
    "against the trajectory; do NOT set sandbox_blocked merely because a file, "
    "attachment, or artifact is mentioned.\n"
    "If the answer makes NO substantive claim because a required INPUT was "
    "missing -- it only asks a clarifying question, or says it cannot open/read "
    "an attachment/file -- set sandbox_blocked=true. Also set sandbox_blocked=true "
    "for an empty answer. Output ONE fenced ```json block."
)
GROUNDING_VERDICT_SYSTEM_PROMPT = (
    "You verify claims against evidence. Each claim carries its kind and its "
    "evidence. Decide 'verified', 'refuted', or 'unverifiable' (insufficient):\n"
    "- kind 'fact': evidence is web search results. verified = supported (or the "
    "claim correctly reports something genuinely unavailable); refuted = "
    "contradicted. Time-sensitive values (prices, schedules, ratings) consistent "
    "with a cited source count as verified even if fresh results differ slightly "
    "-- do not punish drift the agent could not foresee.\n"
    "- kind 'action': evidence is the AGENT'S OWN TRAJECTORY, including tool-call "
    "inputs and outputs. For write_file/edit_file, each entry's 'input' field can "
    "contain the actual authored file content. Return verified when the relevant "
    "write_file/edit_file/publish input or output supports the claimed content, "
    "structure, existence, URL, or path. Return refuted only when the trajectory "
    "contradicts the claim; do NOT return refuted for a claimed URL or path that "
    "literally appears anywhere in the trajectory input or output, because its "
    "presence is supporting evidence. Return unverifiable only when the trajectory "
    "genuinely lacks evidence either way.\n"
    "- kind 'private': the claim reports the USER'S OWN private data (from files, "
    "chats, calendar, attachments, memory) that cannot be web-checked. Evidence is "
    "the agent's trajectory. verified = the trajectory shows the agent SUCCESSFULLY "
    "read a source that plausibly holds this information (a "
    "read_file/glob/grep/pdf-extract/db tool with is_error=false on a matching path "
    "or query); a successful read of the relevant source is sufficient support even "
    "when the exact value is truncated in the shown output. unverifiable = the "
    "answer asserts private data but NO supporting read appears in the trajectory "
    "(possible fabrication) -- do NOT reward it. refuted = a read output in the "
    "trajectory contradicts the claim.\n"
    "Judge TRUTH, not phrasing. Output ONE fenced ```json block."
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


class RubricJudgeScorer:
    """Multi-aspect, checklist-gated LLM judge returning a graded [0,1] score."""

    name = "rubric_judge"
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
        grounding_mode: str = "process",
        grounding_votes: int = 1,
        search=None,
        max_claims: int = 8,
        max_search_results: int = 4,
    ) -> None:
        self._api_client = api_client
        self._model = model
        self._rubrics = rubrics or {}
        self._system_prompt = system_prompt or RUBRIC_JUDGE_SYSTEM_PROMPT
        self._max_tokens = max_tokens
        self._votes = max(1, int(votes))
        self._pass_threshold = pass_threshold
        self._excerpt = output_excerpt_chars
        # process = original checklist-citation grounding; verify = truth-vs-web
        # (see ADR ohmo-eval-verification-grounding). Default stays process.
        self._grounding_mode = grounding_mode
        # >1 vote-stabilizes the verify path (median-of-N over the extract+verdict
        # flap); default 1 = single-shot, unchanged. Searches are cached, so votes
        # cost extract+verdict tokens only.
        self._grounding_votes = max(1, int(grounding_votes))
        self._search = search or _default_grounding_search
        self._max_claims = max_claims
        self._max_search_results = max_search_results

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
        grounding_meta: dict[str, object] = {}
        if self._grounding_mode == "verify" and aspects:
            vg = _run_eval_coroutine(
                _verify_grounding_voted(
                    self._api_client,
                    self._model,
                    votes=self._grounding_votes,
                    task=context.primary_prompt,
                    answer=executor_result.final_text,
                    trajectory=_v2_trajectory(executor_result, excerpt=self._excerpt),
                    checklist_items=_checklist_texts(rubric),
                    search=self._search,
                    max_claims=self._max_claims,
                    max_results=self._max_search_results,
                )
            )
            status = vg["status"]
            if status == "sandbox_blocked":
                aspects["grounding"] = 0.0  # kept in denom (don't hide harness debt)
            elif vg["score"] is not None:
                aspects["grounding"] = vg["score"]
            # else private_fallback / no_claims / unverifiable -> keep process grounding
            graded, passed, gate_fails = _grade_aspects(aspects)
            grounding_meta = {
                "grounding_mode": "verify",
                "grounding_status": status,
                "grounding_votes": vg.get("votes", self._grounding_votes),
                "grounding_verified": vg["verified"],
                "grounding_refuted": vg["refuted"],
                "grounding_claims": vg["claims"],
            }
        verdict = "pass" if passed else ("error" if not aspects else "fail")
        metadata: dict[str, object] = {
            "judge_model": self._model,
            "verdict": verdict,
            "judge_votes": self._votes,
            "judge_parsed_votes": sum(1 for v in votes if v),
            "rubric_graded_score": round(graded, 4),
            "rubric_gate_failures": list(gate_fails),
            "rubric_used_checklist": bool(rubric),
            **grounding_meta,
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


def _format_transcript_for_intent_judge(
    transcript: Sequence[tuple[str, str]],
) -> str:
    lines: list[str] = []
    for role, text in transcript:
        r = role.strip() if isinstance(role, str) else ""
        t = text.strip() if isinstance(text, str) else ""
        if t:
            lines.append(f"{r}: {t}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def _format_checklist(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items if item.strip())


def detect_sandbox_side_effect_credit_signals(trajectory: str) -> tuple[str, ...]:
    """Return recognized sandbox-only side-effect markers from a tool trajectory."""

    lowered = trajectory.lower()
    sentinels = SANDBOX_SIDE_EFFECT_CREDIT_SENTINELS | frozenset(
        f'"{token}": true'
        for token in SANDBOX_SIDE_EFFECT_CREDIT_METADATA_TOKENS
    )
    return tuple(
        sentinel
        for sentinel in sorted(sentinels)
        if sentinel.lower() in lowered
    )


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "y", "on"}:
            return True
        if lowered in {"false", "no", "0", "n", "off"}:
            return False
    return False


async def judge_intent_met(
    api_client: SupportsStreamingMessages,
    model: str,
    *,
    intent: str,
    constraints: Sequence[str],
    transcript: Sequence[tuple[str, str]],
    trajectory: str = "",
    votes: int = 1,
    max_tokens: int = 600,
) -> dict[str, object]:
    tool_trajectory = trajectory.strip() or "No observed tool calls."
    credit_signals = detect_sandbox_side_effect_credit_signals(tool_trajectory)
    credit_signal_block = (
        ", ".join(credit_signals) if credit_signals else "(none)"
    )
    prompt = (
        "You are a strict outcome judge for a task execution transcript.\n\n"
        f"Intent: {intent or '(unspecified)'}\n"
        f"Constraints: {_format_checklist(constraints) or '- (none)'}\n\n"
        "Transcript (role: text per line):\n"
        f"{_format_transcript_for_intent_judge(transcript)}\n\n"
        "Observed tool trajectory (authoritative tool calls and results):\n"
        f"{tool_trajectory}\n\n"
        "SANDBOX SIDE-EFFECT CREDIT RULE: "
        f"{SANDBOX_SIDE_EFFECT_CREDIT_RULE}\n"
        "Do NOT use this rule to excuse genuinely skipping the action, producing "
        "only a draft or placeholder without invoking the tool, or inventing or "
        "fabricating a result. When applying this credit, explicitly mention it "
        "in evidence.\n"
        "Detected sandbox-only side-effect signals in the observed tool trajectory: "
        f"{credit_signal_block}\n\n"
        'Return only ONE JSON object with keys:\n'
        '{"intent_met": true|false, "constraints_held": true|false, '
        '"evidence": "brief evidence snippets"}\n'
        "intent_met is true only when the assistant achieved the user's final objective.\n"
        "constraints_held is true only when no stated constraints were violated.\n"
        "Do not ask clarifying questions; decide from the transcript as-is."
    )
    run_count = max(1, int(votes))
    intent_votes: list[bool] = []
    constraints_votes: list[bool] = []
    evidences: list[str] = []
    for _ in range(run_count):
        text = await _complete_text(
            api_client,
            model,
            system_prompt="You are an objective judge. Return strict JSON only.",
            prompt=prompt,
            max_tokens=max_tokens,
        )
        parsed = _extract_json(text) or {}
        i_met = _coerce_bool(parsed.get("intent_met"))
        c_held = _coerce_bool(parsed.get("constraints_held"))
        intent_votes.append(bool(i_met))
        constraints_votes.append(bool(c_held))
        evidence = parsed.get("evidence")
        if isinstance(evidence, str):
            e = evidence.strip()
            if e:
                evidences.append(e)

    intent_met = sum(intent_votes) > run_count / 2
    constraints_held = sum(constraints_votes) > run_count / 2
    evidence = evidences[0] if evidences else ""
    if credit_signals and (intent_met or constraints_held):
        credit_evidence = (
            "Sandbox side-effect credit applied based on observed signal(s): "
            + ", ".join(credit_signals)
        )
        evidence = f"{evidence}; {credit_evidence}" if evidence else credit_evidence

    return {
        "intent_met": intent_met,
        "constraints_held": constraints_held,
        "evidence": evidence,
        "votes": run_count,
    }


def _v2_trajectory(executor_result: EvalExecutorResult, *, excerpt: int) -> str:
    rows = []
    for index, call in enumerate(executor_result.tool_calls, 1):
        output = (call.output or "").strip().replace("\n", " ")
        if len(output) > excerpt:
            output = output[:excerpt] + "…"
        # Tool INPUT arguments carry the ground truth for many action claims that
        # the tool OUTPUT never echoes — a queued meeting's title, an edit_file's
        # new content, a published path. The verdict prompt already tells the
        # judge to check "trajectory input/output", so surface the input here
        # (same excerpt) or those claims get falsely refuted as "not shown".
        # Transient/score-time only — never persisted (report keeps metadata-only
        # tool labels; EvalObservedCall.arguments is explicitly transient).
        arguments = (
            json.dumps(call.arguments, ensure_ascii=True, sort_keys=True)
            if call.arguments
            else ""
        ).replace("\n", " ")
        if len(arguments) > excerpt:
            arguments = arguments[:excerpt] + "…"
        rows.append(
            {
                "step": index,
                "tool": effective_tool_label(call.tool_name, call.arguments),
                "is_error": bool(call.is_error),
                "input": arguments,
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
    for spec in RUBRIC_JUDGE_ASPECTS:
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
    for spec in RUBRIC_JUDGE_ASPECTS:
        if spec.key in parsed:
            value = _aspect_vote_score(parsed[spec.key])
            if value is not None:
                scores[spec.key] = value
    return scores


def _grade_aspects(aspects: dict[str, float]) -> tuple[float, bool, tuple[str, ...]]:
    """Weighted-mean graded score + hard-gate check over the present aspects.

    Split out of ``_aggregate_v2`` so the verify-grounding path can override the
    grounding aspect and re-grade without re-running the judge votes.
    """
    total_weight = sum(spec.weight for spec in RUBRIC_JUDGE_ASPECTS if spec.key in aspects)
    graded = (
        sum(spec.weight * aspects[spec.key] for spec in RUBRIC_JUDGE_ASPECTS if spec.key in aspects)
        / total_weight
        if total_weight
        else 0.0
    )
    gate_failures = tuple(
        spec.key
        for spec in RUBRIC_JUDGE_ASPECTS
        if spec.gate_floor is not None
        and spec.key in aspects
        and aspects[spec.key] < spec.gate_floor
    )
    # A pass needs the weighted quality bar AND every hard gate satisfied.
    return graded, (not gate_failures and graded >= 0.75), gate_failures


def _aggregate_v2(
    votes: list[dict[str, float]],
) -> tuple[dict[str, float], float, bool, tuple[str, ...]]:
    """Average aspect scores across votes -> (aspects, graded, passed, gate_failures)."""
    aspects: dict[str, float] = {}
    for spec in RUBRIC_JUDGE_ASPECTS:
        vals = [v[spec.key] for v in votes if spec.key in v]
        if vals:
            aspects[spec.key] = sum(vals) / len(vals)
    if not aspects:
        return {}, 0.0, False, ()
    graded, passed, gate_failures = _grade_aspects(aspects)
    return aspects, graded, passed, gate_failures


# --- verification-grounding: extract claims -> web_search -> verdict ----------

def _checklist_texts(rubric: dict[str, object], aspect: str = "grounding", limit: int = 12) -> list[str]:
    """The derived grounding checklist as plain text -- the seed list of facts."""
    items = rubric.get(aspect) if isinstance(rubric, dict) else None
    out: list[str] = []
    if isinstance(items, list):
        for item in items:
            text = item.get("text") if isinstance(item, dict) else item
            if isinstance(text, str) and text.strip():
                out.append(text.strip())
    return out[:limit]


def _grounding_extract_prompt(*, task: str, answer: str, checklist_items: list[str]) -> str:
    seed = ""
    if checklist_items:
        seed = (
            "\n\nThe facts that matter for this task (a claim is 'relevant' if it "
            "addresses one of these or the user's core request):\n"
            + "\n".join(f"- {t}" for t in checklist_items)
        )
    return (
        f"User request:\n{task.strip()}\n\n"
        f"Agent final answer:\n{answer.strip()}"
        f"{seed}\n\n"
        "Extract the answer's checkable claims. For a 'fact' claim give a web "
        "search query that would confirm or refute it (empty for private/action "
        "claims). Claims about the content, structure, or existence of an "
        "agent-created/edited file or report are 'action' claims; for example, "
        "'the report/file contains X' has kind='action'. Schema:\n"
        '```json\n{"sandbox_blocked": false, "claims": [{"id": "c1", '
        '"text": "the concrete claim", "kind": "fact", "public": true, '
        '"relevant": true, "query": "search query"}]}\n```'
    )


def _grounding_verdict_prompt(*, claims: list[dict], evidence: dict[str, str]) -> str:
    blocks = []
    for claim in claims:
        ev = evidence.get(claim["id"], "(no evidence)")
        blocks.append(
            f"[{claim['id']}] kind={claim.get('kind', 'fact')} CLAIM: {claim['text']}\nEVIDENCE:\n{ev}"
        )
    joined = "\n\n".join(blocks)
    return (
        f"{joined}\n\n"
        "For each claim id, return a verdict against its evidence, applying the "
        "kind-specific standard (fact=web evidence; action=trajectory input/output, "
        "including authored content in write_file/edit_file 'input', must support "
        "the claim; private=trajectory must show the agent successfully READ a "
        "matching private source, else unverifiable). A URL/path that literally "
        "appears in the trajectory supports the action claim and must not be called "
        "refuted. Schema:\n"
        '```json\n{"verdicts": [{"id": "c1", '
        '"verdict": "verified|refuted|unverifiable", "evidence": "one short phrase"}]}\n```'
    )


_SEARCH_MEM: dict[str, str] = {}


def _search_cache_file(cache_key: str):
    import os
    from pathlib import Path

    root = os.environ.get("OPENHARNESS_SEARCH_CACHE")
    base = Path(root) if root else (Path.home() / ".cache" / "openharness" / "serper")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{cache_key}.txt"


async def _serper_search(query: str, *, max_results: int = 5, api_key: str | None = None) -> str:
    """Google results via Serper.dev (google_search MCP backend), cached by query.

    Identical (query, max_results) is fetched once: served from an in-process map,
    then a persistent on-disk cache (``OPENHARNESS_SEARCH_CACHE``, default
    ``~/.cache/openharness/serper``), so re-runs, repeated claims, and judge votes
    never re-bill the API. Transient errors are not cached.
    """
    import os

    import httpx

    key = api_key or os.environ.get("SERPER_API_KEY", "")
    if not key:
        return "(serper not configured)"
    cache_key = hashlib.sha256(f"{query}|{max_results}".encode("utf-8")).hexdigest()[:32]
    if cache_key in _SEARCH_MEM:
        return _SEARCH_MEM[cache_key]
    cache_file = _search_cache_file(cache_key)
    if cache_file.exists():
        cached = cache_file.read_text(encoding="utf-8")
        _SEARCH_MEM[cache_key] = cached
        return cached
    n = max(1, min(int(max_results), 10))
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": n},
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
            )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # transient -> return but do NOT cache
        return f"(search error: {exc})"
    lines = [f"Google results for: {query}"]
    ab = data.get("answerBox")
    if isinstance(ab, dict) and (ab.get("answer") or ab.get("snippet")):
        lines.append(f"[answer] {ab.get('answer') or ab.get('snippet')}")
    kg = data.get("knowledgeGraph")
    if isinstance(kg, dict) and kg.get("description"):
        lines.append(f"[kg] {kg.get('title', '')}: {kg.get('description')}")
    for i, item in enumerate((data.get("organic") or [])[:n], 1):
        snippet = (item.get("snippet", "") or "").replace("\n", " ").strip()
        lines.append(f"{i}. {item.get('title', '')}\n   URL: {item.get('link', '')}\n   {snippet}")
    result = "\n".join(lines)
    try:
        cache_file.write_text(result, encoding="utf-8")
    except OSError:
        pass
    _SEARCH_MEM[cache_key] = result
    return result


async def _default_grounding_search(query: str, *, max_results: int = 5) -> str:
    """Independent retrieval: Google (Serper) when SERPER_API_KEY is set, else DuckDuckGo."""
    import os

    if os.environ.get("SERPER_API_KEY"):
        return await _serper_search(query, max_results=max_results)
    from pathlib import Path

    from openharness.tools.base import ToolExecutionContext
    from openharness.tools.web_search_tool import WebSearchTool, WebSearchToolInput

    try:
        result = await WebSearchTool().execute(
            WebSearchToolInput(query=query, max_results=min(max_results, 10)),
            ToolExecutionContext(cwd=Path(".")),
        )
    except Exception as exc:  # retrieval is best-effort; a miss -> unverifiable
        return f"(search error: {exc})"
    return result.output


async def _verify_grounding(
    api_client: SupportsStreamingMessages,
    model: str,
    *,
    task: str,
    answer: str,
    trajectory: str,
    checklist_items: list[str],
    search,
    max_claims: int = 8,
    max_results: int = 5,
    max_tokens: int = 1200,
) -> dict:
    """Score grounding as truth-vs-independent-retrieval, over task-relevant claims.

    Only claims marked ``relevant`` count (closes the pad-with-true-trivia hack).
    ``fact`` claims are checked against web retrieval; ``action`` claims (the agent
    asserting it did/produced something) against the ``trajectory`` (catches a
    fabricated "done / created a file"). Returns {score, status, verified, refuted,
    claims}: ``score`` is None (caller keeps process-grounding) when nothing is
    checkable; ``status='sandbox_blocked'`` forces grounding=0 but stays in denom.
    """
    answer = (answer or "").strip()
    if not answer:
        return {"score": 0.0, "status": "sandbox_blocked", "verified": 0, "refuted": 0, "claims": []}
    extract_raw = await _complete_text(
        api_client,
        model,
        system_prompt=GROUNDING_EXTRACT_SYSTEM_PROMPT,
        prompt=_grounding_extract_prompt(task=task, answer=answer, checklist_items=checklist_items),
        max_tokens=max_tokens,
    )
    parsed = _extract_json(extract_raw) or {}
    if parsed.get("sandbox_blocked") is True:
        return {"score": 0.0, "status": "sandbox_blocked", "verified": 0, "refuted": 0, "claims": []}
    raw_claims = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []
    claims = [c for c in raw_claims if isinstance(c, dict) and c.get("id") and c.get("text")][:max_claims]
    relevant = [c for c in claims if c.get("relevant")]
    if not relevant:
        # no task-relevant claim to verify (e.g. pure padding) -> keep process
        return {"score": None, "status": "no_relevant_claims", "verified": 0, "refuted": 0, "claims": []}

    def _priv_hash(claim: dict) -> str:
        return f"sha:{_hash_text(str(claim['text']))[:12]}"

    facts_public = [c for c in relevant if c.get("kind") != "action" and c.get("public")]
    actions = [c for c in relevant if c.get("kind") == "action"]
    facts_private = [c for c in relevant if c.get("kind") != "action" and not c.get("public")]
    # Private facts are process-grounded against the trajectory: a private value the
    # agent demonstrably READ from a real source (read_file/glob/grep/pdf-extract) is
    # grounded, not a hallucination -- even though it can't be web-verified. Route
    # them through the same trajectory verdict as actions (tagged 'private' so the
    # judge applies the read-a-matching-source standard), but keep them HASHED in the
    # metadata-only report. A private fact with NO supporting read stays unverifiable
    # -> out of denom: the honest "can't confirm" floor is preserved; we just stop
    # scoring 0 for private files the agent actually opened.
    for claim in facts_private:
        claim["kind"] = "private"
    checkable = facts_public + actions + facts_private
    if not checkable:
        return {"score": None, "status": "no_relevant_claims", "verified": 0, "refuted": 0, "claims": []}
    trajectory_evidence = f"AGENT TRAJECTORY (tool-call inputs and outputs):\n{trajectory}"
    evidence: dict[str, str] = {}
    for claim in facts_public:
        evidence[claim["id"]] = await search(str(claim.get("query") or claim["text"]), max_results=max_results)
    for claim in actions + facts_private:
        evidence[claim["id"]] = trajectory_evidence
    verdict_raw = await _complete_text(
        api_client,
        model,
        system_prompt=GROUNDING_VERDICT_SYSTEM_PROMPT,
        prompt=_grounding_verdict_prompt(claims=checkable, evidence=evidence),
        max_tokens=max_tokens,
    )
    vparsed = _extract_json(verdict_raw) or {}
    vlist = vparsed.get("verdicts") if isinstance(vparsed.get("verdicts"), list) else []
    vmap = {v.get("id"): v for v in vlist if isinstance(v, dict)}
    per_claim: list[dict] = []
    supported = refuted = 0
    for claim in checkable:
        vote = vmap.get(claim["id"], {})
        verdict = vote.get("verdict")
        if verdict not in ("verified", "refuted", "unverifiable"):
            verdict = "unverifiable"
        if verdict == "verified":
            supported += 1
        elif verdict == "refuted":
            refuted += 1
        if claim.get("kind") == "private":
            # never persist private content: hash the claim text, keep the verdict.
            per_claim.append(
                {
                    "id": claim["id"],
                    "kind": "private",
                    "claim": _priv_hash(claim),
                    "verdict": verdict if verdict in ("verified", "refuted") else "unverifiable_private",
                    "evidence": "",
                }
            )
        else:
            per_claim.append(
                {
                    "id": claim["id"],
                    "kind": claim.get("kind", "fact"),
                    "claim": str(claim["text"])[:160],
                    "verdict": verdict,
                    "evidence": str(vote.get("evidence") or "")[:160],
                }
            )
    denom = supported + refuted
    if denom == 0:
        status = "private_fallback" if facts_private and not facts_public and not actions else "unverifiable"
        return {"score": None, "status": status, "verified": 0, "refuted": 0, "claims": per_claim}
    return {"score": supported / denom, "status": "scored", "verified": supported, "refuted": refuted, "claims": per_claim}


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


async def _verify_grounding_voted(
    api_client: SupportsStreamingMessages,
    model: str,
    *,
    votes: int,
    task: str,
    answer: str,
    trajectory: str,
    checklist_items: list[str],
    search,
    max_claims: int = 8,
    max_results: int = 5,
    max_tokens: int = 1200,
) -> dict:
    """Run :func:`_verify_grounding` ``votes`` times and aggregate, to kill the
    single-shot extract/verdict flap (a lone run flips one claim
    verified<->refuted between runs, so a 1-claim answer swings 0.0<->1.0).

    Runs are SEQUENTIAL on purpose: run 1 warms the query cache and the rest serve
    from it, so extra votes bill LLM tokens for extract+verdict, NOT search-API
    calls. Aggregation is by OVERRIDE CLASS majority, then median of the numeric
    scores: each run either carries a number (``scored``), forces 0
    (``sandbox_blocked``), or declines to override (a None-status run keeps process
    grounding). The class with the most runs wins; a tie prefers the class with the
    most signal (scored > blocked > keep), so a single noisy ``scored`` run cannot
    flip an otherwise keep-process majority. The reported per-claim breakdown comes
    from the surviving run closest to the median. ``votes<=1`` is the original
    single-shot path, unchanged.
    """
    n = max(1, int(votes))
    runs = [
        await _verify_grounding(
            api_client,
            model,
            task=task,
            answer=answer,
            trajectory=trajectory,
            checklist_items=checklist_items,
            search=search,
            max_claims=max_claims,
            max_results=max_results,
            max_tokens=max_tokens,
        )
        for _ in range(n)
    ]
    if n == 1:
        return {**runs[0], "votes": 1}
    scored = [r for r in runs if r["status"] == "scored" and r["score"] is not None]
    blocked = [r for r in runs if r["status"] == "sandbox_blocked"]
    keep = [r for r in runs if r["status"] not in ("scored", "sandbox_blocked")]
    # Highest count wins; tie -> most-signal class (rank scored 3 > blocked 2 > keep 1).
    winner = max(
        (("scored", len(scored), 3), ("sandbox_blocked", len(blocked), 2), ("keep", len(keep), 1)),
        key=lambda t: (t[1], t[2]),
    )[0]
    if winner == "scored":
        med = _median([r["score"] for r in scored])
        rep = min(scored, key=lambda r: abs(r["score"] - med))
        return {"score": med, "status": "scored", "verified": rep["verified"],
                "refuted": rep["refuted"], "claims": rep["claims"], "votes": n}
    if winner == "sandbox_blocked":
        return {"score": 0.0, "status": "sandbox_blocked", "verified": 0, "refuted": 0,
                "claims": [], "votes": n}
    statuses = [r["status"] for r in keep]
    status = max(statuses, key=statuses.count)  # most common None-status (metadata only)
    rep = next(r for r in keep if r["status"] == status)
    return {"score": None, "status": status, "verified": 0, "refuted": 0,
            "claims": rep["claims"], "votes": n}


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
