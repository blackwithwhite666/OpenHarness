from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals import (
    EvalCaseDraft,
    EvalEpisode,
    EvalEvent,
    EvalResource,
    EvalResourceSnapshot,
    EvalStore,
    build_case_candidates,
    build_case_drafts,
    promote_case_drafts,
    read_case_drafts,
    read_gold_cases,
    write_case_draft_pack,
)


def test_promote_case_drafts_writes_metadata_only_gold_cases(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private request body",
        final_text="private final answer",
    )
    _add_episode(
        store,
        episode_id="ep-2",
        user_text="second private request",
        final_text="second private answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    drafts[0] = drafts[0].model_copy(
        update={"capability_path": ["bash:weather-cli forecast"]}
    )
    write_case_draft_pack(store, drafts)
    selected_case_id = drafts[0].case_id

    result = promote_case_drafts(store, case_ids=[selected_case_id], reviewer="reviewer-1")

    assert result.manifest.records_path == "cases/gold_cases.jsonl"
    assert result.manifest.record_count == 1
    assert result.manifest.metadata == {
        "privacy": "metadata_only",
        "promoted_count": 1,
        "review_status": "approved",
    }
    gold_cases = read_gold_cases(store)
    assert len(gold_cases) == 1
    gold = gold_cases[0]
    assert gold.case_id == selected_case_id
    assert gold.capability_path == ["bash:weather-cli forecast"]
    assert gold.review_status == "approved"
    assert gold.reviewer == "reviewer-1"
    assert gold.metadata == {
        "source_review_status": "draft",
        "candidate_score": drafts[0].metadata["candidate_score"],
        "signals": drafts[0].metadata["signals"],
        "event_count": drafts[0].metadata["event_count"],
    }

    serialized = (
        result.records_path.read_text(encoding="utf-8")
        + result.manifest_path.read_text(encoding="utf-8")
    )
    for sensitive_fragment in (
        "private request body",
        "private final answer",
        "second private request",
        "second private answer",
    ):
        assert sensitive_fragment not in serialized


def test_promote_case_drafts_is_idempotent_for_existing_same_gold(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private request body",
        final_text="private final answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)

    promote_case_drafts(store, case_ids=[drafts[0].case_id], reviewer="first")
    first_gold = read_gold_cases(store)[0]
    promote_case_drafts(store, case_ids=[drafts[0].case_id], reviewer="second")
    second_gold = read_gold_cases(store)[0]

    assert second_gold == first_gold
    assert second_gold.reviewer == "first"


def test_promote_case_drafts_copies_metadata_only_review_metadata(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private request body",
        final_text="private final answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)

    promote_case_drafts(
        store,
        case_ids=[drafts[0].case_id],
        reviewer="reviewer-1",
        review_metadata_by_case={
            drafts[0].case_id: {
                "review_decision": "approved",
                "review_comment_hash": "hash-only",
                "review_comment_length": 12,
            }
        },
    )

    gold = read_gold_cases(store)[0]
    assert gold.metadata["review_decision"] == "approved"
    assert gold.metadata["review_comment_hash"] == "hash-only"
    assert gold.metadata["review_comment_length"] == 12


def test_promote_case_drafts_populates_state_delta_from_snapshots(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private request body",
        final_text="private final answer",
    )
    _write_state_snapshot(
        store,
        episode_id="ep-1",
        phase="world_before",
        reminder_keys=("aaaaaaaaaaaaaaaa",),
    )
    _write_state_snapshot(
        store,
        episode_id="ep-1",
        phase="world_after",
        reminder_keys=("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"),
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)

    promote_case_drafts(store, case_ids=[drafts[0].case_id], reviewer="reviewer-1")

    gold = read_gold_cases(store)[0]
    assert gold.metadata["state_delta"]["changed"] is True
    assert gold.metadata["state_delta"]["reminders"]["added_keys"] == [
        "bbbbbbbbbbbbbbbb"
    ]


def test_promote_case_drafts_omits_state_delta_without_snapshots(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private request body",
        final_text="private final answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)

    promote_case_drafts(store, case_ids=[drafts[0].case_id], reviewer="reviewer-1")

    assert "state_delta" not in read_gold_cases(store)[0].metadata


def test_promote_case_drafts_validates_selection_and_paths(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private request body",
        final_text="private final answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)

    with pytest.raises(ValueError, match="case_ids must not be empty"):
        promote_case_drafts(store, case_ids=[])
    with pytest.raises(ValueError, match="case draft not found: missing"):
        promote_case_drafts(store, case_ids=["missing"])
    with pytest.raises(ValueError, match="store.root/cases"):
        read_case_drafts(store, records_filename="../case_drafts.jsonl")
    with pytest.raises(ValueError, match="store.root/cases"):
        promote_case_drafts(store, records_filename="../gold_cases.jsonl")


def test_promote_case_drafts_rejects_duplicate_or_malformed_rows(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    draft = EvalCaseDraft(
        case_id="case-1",
        candidate_id="candidate-1",
        episode_id="ep-1",
        case_kind="conversation_replay",
    )
    case_drafts_path = store.root / "cases" / "case_drafts.jsonl"
    case_drafts_path.write_text(
        draft.model_dump_json() + "\n" + draft.model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate case draft ids: case-1"):
        promote_case_drafts(store)

    case_drafts_path.write_text(draft.model_dump_json() + "\n", encoding="utf-8")
    (store.root / "cases" / "gold_cases.jsonl").write_text(
        "{not json}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid gold_cases.jsonl row 1"):
        promote_case_drafts(store, case_ids=["case-1"])


def _add_episode(
    store: EvalStore,
    *,
    episode_id: str,
    user_text: str,
    final_text: str,
) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id=f"session-{episode_id}",
            user_text=user_text,
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            payload={"text": final_text},
        )
    )


def _write_state_snapshot(
    store: EvalStore,
    *,
    episode_id: str,
    phase: str,
    reminder_keys: tuple[str, ...],
) -> None:
    path = store.root / "states" / episode_id / f"{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = EvalResourceSnapshot(
        episode_id=episode_id,
        resources=[
            EvalResource(
                resource_id="resource:reminders_json",
                kind="local_file",
                name="reminders_json",
                exists=True,
                metadata={
                    "entry_keys": list(reminder_keys),
                    "record_count": len(reminder_keys),
                    "status_counts": {"pending": len(reminder_keys)},
                },
            )
        ],
    )
    path.write_text(snapshot.model_dump_json(), encoding="utf-8")
