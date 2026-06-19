from __future__ import annotations

import json
from pathlib import Path

import pytest

from openharness.evals import (
    EvalEpisode,
    EvalEvent,
    EvalStore,
    build_case_candidates,
    build_case_drafts,
    write_candidate_pack,
    write_case_draft_pack,
)


def test_case_candidates_and_drafts_are_metadata_only(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_goal="Secret launch goal",
        user_text="Please investigate private launch risk",
        events=[
            EvalEvent(
                episode_id="ep-1",
                kind="resource_snapshot",
                payload={"path": "states/ep-1/resource_snapshot.json"},
            ),
            EvalEvent(
                episode_id="ep-1",
                kind="tool_started",
                tool_name="web_fetch",
                tool_call_id="tool-1",
                payload={
                    "input_summary": "fetch private launch URL",
                    "input": {"url": "https://private.example/launch"},
                },
            ),
            EvalEvent(
                episode_id="ep-1",
                kind="tool_completed",
                tool_name="web_fetch",
                tool_call_id="tool-1",
                payload={
                    "output_summary": "private risk summary",
                    "output": "full private risk details",
                },
            ),
            EvalEvent(
                episode_id="ep-1",
                kind="gateway_final",
                payload={"text": "final private recommendation"},
            ),
        ],
    )
    _add_episode(
        store,
        episode_id="ep-2",
        user_goal="",
        user_text="Handle secret failure",
        events=[
            EvalEvent(
                episode_id="ep-2",
                kind="gateway_error",
                payload={"text": "private error body"},
                is_error=True,
            ),
        ],
    )

    candidates = build_case_candidates(store)
    drafts = build_case_drafts(store, candidates)

    assert [candidate.episode_id for candidate in candidates] == ["ep-2", "ep-1"]
    assert candidates[0].candidate_kind == "error_recovery"
    assert candidates[0].signals == ["has_error"]
    assert candidates[1].candidate_kind == "tool_workflow"
    assert candidates[1].signals == [
        "uses_tools",
        "has_resource_snapshot",
        "has_final_response",
    ]
    assert candidates[1].tool_path == ["web_fetch"]
    assert candidates[1].metadata["event_count"] == 4

    drafts_by_episode = {draft.episode_id: draft for draft in drafts}
    assert drafts_by_episode["ep-1"].review_status == "draft"
    assert drafts_by_episode["ep-1"].tool_names == ["web_fetch"]
    assert drafts_by_episode["ep-1"].input_facet_ids
    assert drafts_by_episode["ep-1"].expected_facet_ids
    assert "use an equivalent tool strategy" in drafts_by_episode["ep-1"].rubric[-1]
    assert "handle the error path" in drafts_by_episode["ep-2"].rubric[-1]

    candidate_write = write_candidate_pack(store, candidates)
    draft_write = write_case_draft_pack(store, drafts)

    assert candidate_write.manifest.records_path == "candidates/candidates.jsonl"
    assert candidate_write.manifest.record_count == 2
    assert draft_write.manifest.records_path == "cases/case_drafts.jsonl"
    assert draft_write.manifest.record_count == 2

    serialized = "\n".join(
        [
            candidate_write.records_path.read_text(encoding="utf-8"),
            candidate_write.manifest_path.read_text(encoding="utf-8"),
            draft_write.records_path.read_text(encoding="utf-8"),
            draft_write.manifest_path.read_text(encoding="utf-8"),
        ]
    )
    assert json.loads(candidate_write.manifest_path.read_text(encoding="utf-8"))[
        "metadata"
    ] == {"privacy": "metadata_only"}
    for sensitive_fragment in (
        "Secret launch goal",
        "Please investigate private launch risk",
        "fetch private launch URL",
        "https://private.example/launch",
        "private risk summary",
        "full private risk details",
        "final private recommendation",
        "Handle secret failure",
        "private error body",
    ):
        assert sensitive_fragment not in serialized


def test_candidate_and_case_pack_reject_escaped_output_paths(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")

    with pytest.raises(ValueError, match="store.root/candidates"):
        write_candidate_pack(store, records_filename="../escaped.jsonl")

    with pytest.raises(ValueError, match="store.root/cases"):
        write_case_draft_pack(store, records_filename="../escaped.jsonl")


def _add_episode(
    store: EvalStore,
    *,
    episode_id: str,
    user_goal: str,
    user_text: str,
    events: list[EvalEvent],
) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id=f"session-{episode_id}",
            user_goal=user_goal,
            user_text=user_text,
        )
    )
    for event in events:
        store.append_event(event)
