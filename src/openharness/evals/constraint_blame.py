"""Constraint-failure attribution helpers for faithful session reports."""

from __future__ import annotations

from pathlib import Path

from openharness.api.client import SupportsStreamingMessages
from openharness.evals.executor import _run_eval_coroutine
from openharness.evals.grounding_trace import read_session_trace, trace_text_fields
from openharness.evals.meta_judge import MetaJudgeAttributor, summarize_attributions
from openharness.evals.models import EvalSessionReport
from openharness.evals.session import (
    group_episodes_into_sessions,
    segment_sessions_into_conversations,
)
from openharness.evals.session_user_simulator import derive_ironuser_spec
from openharness.evals.store import EvalStore

CONSTRAINT_ATTRIBUTION_FRAME = (
    "Attribute only the failed constraints_held check. A violated satisfiable "
    "constraint is MODEL. An unsatisfiable, off-task, or hallucinated constraint "
    "from the IronUser spec is HARNESS_RUBRIC. A compliant answer that the "
    "original constraints judge failed is JUDGE_FAULT."
)


def constraint_report_metadata(
    *,
    constraints: object | None = None,
    intent_evidence: object | None = None,
) -> dict[str, object]:
    """Return report-case metadata needed to attribute constraints failures."""
    metadata: dict[str, object] = {}
    if constraints is not None:
        metadata["constraints"] = _coerce_constraints(constraints)
    if intent_evidence is not None:
        metadata["intent_evidence"] = str(intent_evidence)
    return metadata


def attribute_faithful_constraints_report(
    report: EvalSessionReport,
    *,
    attributor: MetaJudgeAttributor | None,
    api_client: SupportsStreamingMessages | None = None,
    model: str = "",
    store: EvalStore | None = None,
    trace_root: Path | None = None,
    app: str | None = None,
) -> dict[str, object]:
    """Attribute failed faithful constraint checks to model or harness causes."""
    attributions: list[dict[str, object]] = []
    for case in report.cases:
        if case.checks.get("constraints_held") is not False:
            continue

        metadata = case.metadata
        constraints = _coerce_constraints(metadata.get("constraints"))
        derived_intent = ""
        if not constraints:
            if api_client is None or store is None:
                raise ValueError(
                    f"constraint detail missing for {case.session_id}; "
                    "provide api_client/model and workspace eval store to re-derive it"
                )
            captured_prompts = _captured_prompts_for_session(
                store,
                case.session_id,
                app=app,
            )
            if not captured_prompts:
                raise ValueError(
                    f"captured prompts not found for session {case.session_id}"
                )
            spec = _run_eval_coroutine(
                derive_ironuser_spec(
                    api_client,
                    model,
                    captured_prompts=captured_prompts,
                )
            )
            constraints = list(spec.constraints)
            derived_intent = spec.intent

        trace = read_session_trace(trace_root, case.session_id) if trace_root else {}
        fields = trace_text_fields(trace, metadata)
        task = fields["task"] or derived_intent
        answer = fields["answer"]
        if not answer:
            raise ValueError(
                f"constraint attribution requires grounding_answer/final_text "
                f"for {case.session_id}"
            )
        if attributor is None:
            raise ValueError("meta-judge attributor is required for constraints failures")

        attribution = attributor.attribute(
            task=task,
            rubric=constraint_attribution_rubric(constraints),
            answer=answer,
            trajectory=[{"evidence": str(metadata.get("intent_evidence") or "")}],
            aspect_scores={"constraints_held": False},
            gate_failures=["constraints_held"],
        )
        item: dict[str, object] = {
            "session_id": case.session_id,
            "constraints": constraints,
        }
        item.update(attribution)
        attributions.append(item)

    summary = summarize_attributions(attributions)
    summary["constraint_harness_debt_pct"] = summary["harness_debt_pct"]
    return {
        "report_id": report.report_id,
        "lane": "faithful",
        "check": "constraints",
        "failed_count": len(attributions),
        "attributions": attributions,
        "summary": summary,
    }


def constraint_attribution_rubric(constraints: list[str]) -> dict[str, object]:
    lines = [CONSTRAINT_ATTRIBUTION_FRAME]
    if constraints:
        lines.extend(f"MUST {constraint}" for constraint in constraints)
    else:
        lines.append(
            "No concrete constraints were persisted or derivable; do not blame "
            "the model unless the answer visibly violates a user-stated constraint."
        )
    return {"task_completion": {"text": "\n".join(lines)}}


def default_constraint_blame_output_path(report_path: str | Path) -> Path:
    path = Path(report_path).expanduser()
    return path.with_name(f"{path.stem}.constraint_blame.json")


def _coerce_constraints(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return []
    return [text for item in values if (text := str(item).strip())]


def _captured_prompts_for_session(
    store: EvalStore,
    session_id: str,
    *,
    app: str | None,
) -> tuple[str, ...]:
    episode_ids = _episode_ids_for_session(store, session_id, app=app)
    prompts: list[str] = []
    for episode_id in episode_ids:
        episode = store.get_episode(episode_id)
        if episode is None:
            prompts.append("")
        else:
            prompts.append(episode.user_goal or episode.user_text)
    return tuple(prompts)


def _episode_ids_for_session(
    store: EvalStore,
    session_id: str,
    *,
    app: str | None,
) -> tuple[str, ...]:
    for group in group_episodes_into_sessions(store, app=app):
        if group.session_id == session_id:
            return group.episode_ids

    split = _split_segment_session_id(session_id)
    if split is None:
        return ()
    base_session_id, segment_index = split
    for conversation in segment_sessions_into_conversations(
        store,
        app=app,
        gap_minutes=30.0,
        min_turns=1,
    ):
        if (
            conversation.session_id == base_session_id
            and conversation.segment_index == segment_index
        ):
            return conversation.episode_ids
    return ()


def _split_segment_session_id(session_id: str) -> tuple[str, int] | None:
    if "#" not in session_id:
        return None
    base, index_text = session_id.rsplit("#", 1)
    try:
        index = int(index_text)
    except ValueError:
        return None
    return base, index
