from __future__ import annotations

import json
from pathlib import Path

import pytest

from openharness.evals import (
    EvalEmbeddingRecord,
    EvalEpisode,
    EvalEvent,
    EvalStore,
    build_case_candidates,
    build_case_drafts,
    collect_text_facets,
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
        "has_graph_motif",
    ]
    assert candidates[1].tool_path == ["web_fetch"]
    assert candidates[1].metadata["event_count"] == 4
    assert candidates[1].metadata["graph_motif_key"] == (
        "resource_snapshot|tool_started|tool_completed|gateway_final::web_fetch"
    )

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


def test_case_candidates_include_embedding_signals_from_lookup_index(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_goal="Secret embedding goal",
        user_text="Please inspect private embedding",
        events=[
            EvalEvent(
                episode_id="ep-1",
                kind="gateway_final",
                payload={"text": "private embedding final"},
            ),
        ],
    )
    facet = collect_text_facets(store)[0].facet
    store.replace_embedding_index(
        [
            EvalEmbeddingRecord(
                facet=facet,
                model="BAAI/bge-m3",
                dimensions=2,
                vector=[0.1, 0.2],
            )
        ],
        jsonl_path="embeddings/embedding_records.jsonl",
    )

    candidate = build_case_candidates(store)[0]

    assert "has_embeddings" in candidate.signals
    assert candidate.metadata["embedded_facet_count"] == 1
    assert candidate.metadata["facet_count"] == 3


def test_candidate_and_case_pack_reject_escaped_output_paths(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")

    with pytest.raises(ValueError, match="store.root/candidates"):
        write_candidate_pack(store, records_filename="../escaped.jsonl")

    with pytest.raises(ValueError, match="store.root/cases"):
        write_case_draft_pack(store, records_filename="../escaped.jsonl")


def test_candidate_capability_path_lifts_bash_binary_and_subcommand(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-bash",
        user_goal="weather lookup",
        user_text="погода в спб?",
        events=[
            EvalEvent(
                episode_id="ep-bash",
                kind="tool_started",
                tool_name="bash",
                tool_call_id="c1",
                payload={"input": {"command": "weather-cli forecast 'СПб' --json"}},
            ),
            EvalEvent(
                episode_id="ep-bash",
                kind="tool_completed",
                tool_name="bash",
                tool_call_id="c1",
                payload={"output": "ok"},
            ),
            EvalEvent(episode_id="ep-bash", kind="gateway_final", payload={"text": "+15"}),
        ],
    )

    candidate = build_case_candidates(store)[0]
    # Real tool name is preserved (replay maps fixtures by it)...
    assert candidate.tool_path == ["bash"]
    # ...while the capability is lifted out of the command (binary + subcommand).
    assert candidate.capability_path == ["bash:weather-cli forecast"]

    draft = build_case_drafts(store, [candidate])[0]
    assert draft.tool_names == ["bash"]
    assert draft.capability_path == ["bash:weather-cli forecast"]


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
