from __future__ import annotations

from pathlib import Path

from openharness.evals import EvalEpisode, EvalEvent
from ohmo.evals import get_eval_store, write_ohmo_eval_mine


def test_write_ohmo_eval_mine_uses_workspace_eval_store(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_text="private ohmo mining request",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="gateway_final",
            payload={"text": "private ohmo final"},
        )
    )

    result = write_ohmo_eval_mine(workspace=workspace)

    assert result.candidates.manifest_path == workspace.resolve() / "evals" / (
        "candidates/candidate_manifest.json"
    )
    assert result.cases.manifest_path == workspace.resolve() / "evals" / (
        "cases/case_manifest.json"
    )
    assert result.candidates.manifest.record_count == 1
    assert result.cases.manifest.record_count == 1
    serialized = (
        result.candidates.records_path.read_text(encoding="utf-8")
        + result.cases.records_path.read_text(encoding="utf-8")
    )
    assert "private ohmo mining request" not in serialized
    assert "private ohmo final" not in serialized
