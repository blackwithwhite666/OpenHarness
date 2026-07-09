"""LLM meta-judge for routing failed eval cases to model or harness backlog."""

from __future__ import annotations

import json
from collections import Counter

from openharness.api.client import SupportsStreamingMessages
from openharness.evals.executor import _run_eval_coroutine
from openharness.evals.judge import _complete_text, _extract_json


BLAME_LABELS = ("model", "harness_boundary", "harness_rubric", "infra", "judge_fault")
HARNESS_LABELS = {"harness_boundary", "harness_rubric", "infra", "judge_fault"}
MAX_TRAJ_CALLS = 24
MAX_CALL_CHARS = 240
MAX_ANSWER_CHARS = 3000
MAX_RUBRIC_CHARS = 1500
_STRONG_HARNESS_SIGNALS = ("command not found", "не смонтир", "не установлен")
_SCHEMA_JSON = (
    '{"blame":"model|harness_boundary|harness_rubric|infra|judge_fault",'
    '"subtype":null,"confidence":0.0,"evidence":"short grounded explanation"}'
)
_TRAJ_HEAD_CALLS = 12
_TRAJ_TAIL_CALLS = 8
_TRUNCATION_MARKER = "…[truncated]…"

META_JUDGE_SYSTEM_PROMPT = (
    "You are an adversarial meta-judge for failed AI-agent evals. Given ONLY the harness this agent ran in "
    "(its available tools, the provided rubric, the trajectory incl. tool errors), could a competent agent "
    "have PASSED this case? If forced by a missing/broken tool, unmountable input, unsatisfiable/off-task "
    "rubric, or a judge that mis-scored a correct answer -- that is HARNESS. If the agent had everything it "
    "needed and still under-delivered or fabricated -- that is MODEL. Return strict JSON only."
)


class MetaJudgeAttributor:
    name = "meta_judge_attributor"

    def __init__(self, *, api_client: SupportsStreamingMessages, model: str, votes: int = 1) -> None:
        self._api_client = api_client
        self._model = model
        self._votes = max(1, int(votes))

    def attribute(
        self,
        *,
        task: str,
        rubric: dict,
        answer: str,
        trajectory: list,
        aspect_scores: dict,
        gate_failures: list,
    ) -> dict:
        bounded = _bounded_prompt_inputs(
            rubric=rubric,
            answer=answer,
            trajectory=trajectory,
        )
        prompt = _attribution_prompt(
            task=task,
            rubric=bounded["rubric"],
            answer=bounded["answer"],
            trajectory=bounded["trajectory"],
            aspect_scores=aspect_scores,
            gate_failures=gate_failures,
            signal_hint=signal_prefilter(trajectory, answer),
        )
        parsed_votes = [
            _parse_vote(_run_eval_coroutine(self._complete(prompt)))
            for _ in range(self._votes)
        ]
        valid_votes = [vote for vote in parsed_votes if vote is not None]
        if not valid_votes:
            return {
                "blame": "model", "subtype": None, "confidence": 0.0,
                "evidence": "No parseable meta-judge votes.", "votes": self._votes,
            }

        winner = _majority_label([vote["blame"] for vote in valid_votes])
        winning_votes = [vote for vote in valid_votes if vote["blame"] == winner]
        representative = winning_votes[0]
        confidence = sum(vote["confidence"] for vote in winning_votes) / len(winning_votes)
        return {
            "blame": winner, "subtype": representative["subtype"],
            "confidence": round(confidence, 4), "evidence": representative["evidence"],
            "votes": self._votes,
        }

    async def _complete(self, prompt: str) -> str:
        return await _complete_text(self._api_client, self._model, system_prompt=META_JUDGE_SYSTEM_PROMPT, prompt=prompt, max_tokens=700)


def signal_prefilter(trajectory: list, answer: str) -> str | None:
    """Return a strong harness hint, never a verdict.
    Generic "не нашёл" / "not found" is ambiguous: it can be the agent's own empty search
    result, so it must NOT auto-label harness; only the LLM decides those.
    """
    haystack = f"{answer}\n{_safe_json(trajectory)}".lower()
    if any(_is_error_step(step) for step in trajectory or []) or any(
        signal in haystack for signal in _STRONG_HARNESS_SIGNALS
    ):
        return "harness_boundary"
    return None


def summarize_attributions(items: list[dict]) -> dict:
    counts = Counter({label: 0 for label in BLAME_LABELS})
    subtypes: Counter[str] = Counter()
    for item in items:
        label = str(item.get("blame") or "judge_fault")
        counts[label] += 1
        subtype = item.get("subtype")
        if subtype:
            subtypes[str(subtype)] += 1

    total = len(items)
    return {
        "counts": dict(counts),
        "harness_debt_pct": _pct(sum(counts[label] for label in HARNESS_LABELS), total),
        "model_signal_pct": _pct(counts["model"], total),
        "subtypes": dict(subtypes),
    }


def _attribution_prompt(
    *,
    task: str,
    rubric: object,
    answer: str,
    trajectory: object,
    aspect_scores: dict,
    gate_failures: list,
    signal_hint: str | None,
) -> str:
    instruction = (
        "Attribute this FAILED eval case. Given ONLY the harness this agent ran "
        "in (its available tools, the provided rubric, the trajectory incl. tool "
        "errors), could a competent agent have PASSED this case? If the failure "
        "was forced by a missing/broken tool, an unmountable input, a rubric that "
        "is unsatisfiable or off-task, or a judge that mis-scored a correct answer "
        "-- that is HARNESS. If the agent had everything it needed and still "
        "under-delivered or fabricated -- that is MODEL.\n\n"
        "Labels: model = agent under-delivered/fabricated despite enough context; "
        "harness_boundary = missing/broken tool, unmountable input, or sandbox "
        "boundary; harness_rubric = impossible/off-task/path-specific rubric; "
        "infra = transient platform/service failure; judge_fault = original judge "
        "failed a materially correct answer. The deterministic signal is only a "
        "hint, not a verdict."
    )
    sections = [
        instruction,
        f"Deterministic signal hint: {signal_hint or 'none'}",
        f"Task:\n{task.strip()}",
        f"Rubric:\n{_safe_json(rubric)}",
        f"Aspect scores:\n{_safe_json(aspect_scores)}",
        f"Gate failures:\n{_safe_json(gate_failures)}",
        f"Trajectory:\n{_safe_json(trajectory)}",
        f"Agent final answer:\n{answer.strip()}",
        f"Return strict JSON only matching this schema:\n{_SCHEMA_JSON}",
    ]
    return "\n\n".join(sections)


def _bounded_prompt_inputs(*, rubric: dict, answer: str, trajectory: list) -> dict[str, object]:
    return {
        "rubric": _bounded_rubric(rubric),
        "answer": _truncate_head_tail(answer.strip(), MAX_ANSWER_CHARS),
        "trajectory": _bounded_trajectory(trajectory),
    }


def _bounded_trajectory(trajectory: list) -> list[object]:
    indexed_steps = list(enumerate(trajectory or [], 1))
    if len(indexed_steps) > MAX_TRAJ_CALLS:
        elided = len(indexed_steps) - _TRAJ_HEAD_CALLS - _TRAJ_TAIL_CALLS
        marker = f"… {elided} calls elided …"
        indexed_steps = (
            indexed_steps[:_TRAJ_HEAD_CALLS]
            + [(None, marker)]
            + indexed_steps[-_TRAJ_TAIL_CALLS:]
        )

    bounded: list[object] = []
    for index, step in indexed_steps:
        if index is None:
            bounded.append(step)
            continue
        bounded.append(
            {
                "index": index,
                "is_error": _is_error_step(step),
                "entry": _truncate_right(_safe_json(step), MAX_CALL_CHARS),
            }
        )
    return bounded


def _bounded_rubric(rubric: dict) -> list[str]:
    texts: list[str] = []
    if isinstance(rubric, dict):
        for aspect in ("task_completion", "grounding"):
            _collect_rubric_texts(rubric.get(aspect), texts)

    bounded: list[str] = []
    remaining = MAX_RUBRIC_CHARS
    for text in texts:
        text = text.strip()
        if not text or remaining <= 0:
            continue
        if len(text) <= remaining:
            bounded.append(text)
            remaining -= len(text)
        else:
            bounded.append(_truncate_head_tail(text, remaining))
            break
    return bounded


def _collect_rubric_texts(value: object, texts: list[str]) -> None:
    if isinstance(value, dict):
        text = value.get("text")
        if text not in (None, ""):
            texts.append(str(text))
        for child in value.values():
            if child is not text:
                _collect_rubric_texts(child, texts)
    elif isinstance(value, list):
        for child in value:
            _collect_rubric_texts(child, texts)


def _truncate_head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= len(_TRUNCATION_MARKER):
        return text[:limit]
    keep = limit - len(_TRUNCATION_MARKER)
    head = keep // 2
    tail = keep - head
    return f"{text[:head]}{_TRUNCATION_MARKER}{text[-tail:]}"


def _truncate_right(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= len(_TRUNCATION_MARKER):
        return text[:limit]
    return f"{text[:limit - len(_TRUNCATION_MARKER)]}{_TRUNCATION_MARKER}"


def _parse_vote(text: str) -> dict | None:
    parsed = _extract_json(text)
    if not parsed:
        return None
    label = str(parsed.get("blame") or "").strip().lower()
    if label not in BLAME_LABELS:
        return None
    subtype = parsed.get("subtype")
    confidence = _coerce_confidence(parsed.get("confidence"))
    return {
        "blame": label,
        "subtype": str(subtype).strip() if subtype not in (None, "") else None,
        "confidence": confidence,
        "evidence": str(parsed.get("evidence") or "").strip(),
    }


def _majority_label(labels: list[str]) -> str:
    counts = Counter(labels)
    tied = {label for label, count in counts.items() if count == max(counts.values())}
    return "model" if "model" in tied else next(label for label in BLAME_LABELS if label in tied)


def _coerce_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, str)):
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _is_error_step(step: object) -> bool:
    value = step.get("is_error") if isinstance(step, dict) else getattr(step, "is_error", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, default=str)


def _pct(part: int, total: int) -> float:
    return 0.0 if total == 0 else part * 100.0 / total
