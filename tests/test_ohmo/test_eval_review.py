from __future__ import annotations

import json
from pathlib import Path

import pytest

from openharness.evals import EvalEpisode, EvalEvent, read_gold_cases
from ohmo.evals import (
    get_eval_store,
    promote_ohmo_eval_case_drafts,
    review_ohmo_eval_case_drafts,
    validate_ohmo_eval_review_manifest,
    write_ohmo_eval_mine,
    write_ohmo_eval_review_manifest,
)


def test_review_ohmo_eval_case_drafts_lists_metadata_only_rows(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _add_episode(workspace, episode_id="ep-1", user_text="private request")
    write_ohmo_eval_mine(workspace=workspace)

    result = review_ohmo_eval_case_drafts(workspace=workspace, limit=10)

    assert result.total_count == 1
    assert len(result.shown) == 1
    item = result.shown[0]
    assert item.case_kind == "conversation_replay"
    assert item.episode_id == "ep-1"
    assert item.review_status == "draft"
    assert item.input_facet_count == 1
    assert item.expected_facet_count == 1
    assert item.tool_names == []


def test_promote_ohmo_eval_case_drafts_selected_and_dry_run(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = _add_episode(workspace, episode_id="ep-1", user_text="private request")
    write_ohmo_eval_mine(workspace=workspace)
    review = review_ohmo_eval_case_drafts(workspace=workspace)
    case_id = review.shown[0].case_id

    dry_run = promote_ohmo_eval_case_drafts(
        workspace=workspace,
        case_ids=[case_id],
        dry_run=True,
    )

    assert dry_run.dry_run is True
    assert dry_run.promoted_count == 1
    assert dry_run.selected_case_ids == [case_id]
    assert dry_run.remaining_unpromoted_count == 0
    assert read_gold_cases(store) == []

    result = promote_ohmo_eval_case_drafts(
        workspace=workspace,
        case_ids=[case_id],
        reviewer="reviewer-1",
    )

    assert result.dry_run is False
    assert result.promoted_count == 1
    assert result.remaining_unpromoted_count == 0
    assert result.manifest_path == workspace.resolve() / "evals" / "cases" / (
        "gold_manifest.json"
    )
    gold_cases = read_gold_cases(store)
    assert len(gold_cases) == 1
    assert gold_cases[0].case_id == case_id
    assert gold_cases[0].reviewer == "reviewer-1"
    assert "private request" not in result.records_path.read_text(encoding="utf-8")


def test_write_ohmo_eval_review_manifest_is_metadata_only(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _add_episode(workspace, episode_id="ep-1", user_text="private review manifest request")
    write_ohmo_eval_mine(workspace=workspace)

    result = write_ohmo_eval_review_manifest(
        workspace=workspace,
        filename="batch_review.json",
    )

    assert result.relative_path == "cases/batch_review.json"
    assert result.total_count == 1
    assert result.shown_count == 1
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert payload["manifest_kind"] == "case_draft_review"
    assert payload["metadata"] == {
        "privacy": "metadata_only",
        "case_id": "",
        "limit": 20,
    }
    assert payload["items"][0]["episode_id"] == "ep-1"
    assert payload["items"][0]["decision"] == "pending"
    assert payload["items"][0]["reviewer"] == ""
    assert payload["items"][0]["comment"] == ""
    serialized = result.path.read_text(encoding="utf-8")
    assert "private review manifest request" not in serialized
    assert "private final" not in serialized


def test_promote_ohmo_eval_case_drafts_from_review_manifest(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = _add_episode(workspace, episode_id="ep-1", user_text="private first")
    _add_episode(workspace, episode_id="ep-2", user_text="private second")
    write_ohmo_eval_mine(workspace=workspace)
    manifest = write_ohmo_eval_review_manifest(
        workspace=workspace,
        filename="batch_review.json",
    )
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["items"][0]["decision"] = "approved"
    payload["items"][0]["reviewer"] = "manifest-reviewer"
    payload["items"][0]["comment"] = "keep this case"
    payload["items"][1]["decision"] = "rejected"
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    result = promote_ohmo_eval_case_drafts(
        workspace=workspace,
        manifest_filename="batch_review.json",
        reviewer="reviewer-1",
    )

    assert result.promoted_count == 1
    assert result.selected_case_ids == [payload["items"][0]["case_id"]]
    gold_cases = read_gold_cases(store)
    assert len(gold_cases) == 1
    gold = gold_cases[0]
    assert gold.case_id == payload["items"][0]["case_id"]
    assert gold.reviewer == "reviewer-1"
    assert gold.metadata["review_decision"] == "approved"
    assert gold.metadata["reviewer"] == "manifest-reviewer"
    assert gold.metadata["review_manifest"] == "batch_review.json"
    assert gold.metadata["review_comment_length"] == len("keep this case")
    serialized = result.records_path.read_text(encoding="utf-8")
    assert "keep this case" not in serialized
    assert "private first" not in serialized
    assert "private second" not in serialized


def test_validate_ohmo_eval_review_manifest_summarizes_decisions(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _add_episode(workspace, episode_id="ep-1", user_text="private first")
    _add_episode(workspace, episode_id="ep-2", user_text="private second")
    write_ohmo_eval_mine(workspace=workspace)
    manifest = write_ohmo_eval_review_manifest(
        workspace=workspace,
        filename="batch_review.json",
    )
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["items"][0]["decision"] = "approved"
    payload["items"][1]["decision"] = "rejected"
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_ohmo_eval_review_manifest(
        workspace=workspace,
        filename="batch_review.json",
    )

    assert validation.relative_path == "cases/batch_review.json"
    assert validation.total_count == 2
    assert validation.approved_count == 1
    assert validation.rejected_count == 1
    assert validation.pending_count == 0
    assert validation.missing_case_ids == []
    assert validation.approved_case_ids == [payload["items"][0]["case_id"]]


def test_validate_ohmo_eval_review_manifest_rejects_missing_draft_cases(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    _add_episode(workspace, episode_id="ep-1", user_text="private first")
    write_ohmo_eval_mine(workspace=workspace)
    manifest = write_ohmo_eval_review_manifest(
        workspace=workspace,
        filename="batch_review.json",
    )
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["items"][0]["case_id"] = "missing-case"
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing draft cases: missing-case"):
        validate_ohmo_eval_review_manifest(
            workspace=workspace,
            filename="batch_review.json",
        )


def test_promote_ohmo_eval_case_drafts_from_manifest_validates_selection(
    tmp_path: Path,
):
    workspace = tmp_path / "workspace"
    _add_episode(workspace, episode_id="ep-1", user_text="private first")
    write_ohmo_eval_mine(workspace=workspace)
    manifest = write_ohmo_eval_review_manifest(
        workspace=workspace,
        filename="batch_review.json",
    )

    with pytest.raises(ValueError, match="choose either manifest"):
        promote_ohmo_eval_case_drafts(
            workspace=workspace,
            manifest_filename="batch_review.json",
            promote_all=True,
        )
    with pytest.raises(ValueError, match="no approved cases"):
        promote_ohmo_eval_case_drafts(
            workspace=workspace,
            manifest_filename="batch_review.json",
        )

    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    payload["items"][0]["decision"] = "maybe"
    manifest.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown review decision"):
        promote_ohmo_eval_case_drafts(
            workspace=workspace,
            manifest_filename="batch_review.json",
        )


def test_write_ohmo_eval_review_manifest_validates_paths(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _add_episode(workspace, episode_id="ep-1", user_text="private request")
    write_ohmo_eval_mine(workspace=workspace)

    with pytest.raises(ValueError, match="store.root/cases"):
        write_ohmo_eval_review_manifest(
            workspace=workspace,
            filename="../review_manifest.json",
        )


def _add_episode(workspace: Path, *, episode_id: str, user_text: str):
    store = get_eval_store(workspace)
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
            payload={"text": "private final"},
        )
    )
    return store
