from __future__ import annotations

from pathlib import Path

from openharness.evals import EvalEpisode, EvalEvent, promote_case_drafts
from ohmo.evals import (
    build_ohmo_eval_pack,
    get_eval_store,
    run_ohmo_eval_smoke,
    write_ohmo_eval_mine,
)


def test_build_ohmo_eval_pack_uses_workspace_gold_cases(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _prepare_gold_case(workspace, episode_id="ep-1", user_text="private ohmo pack")

    result = build_ohmo_eval_pack(workspace=workspace)

    assert result.write.path == workspace.resolve() / "evals" / "packs" / "eval_pack.json"
    assert result.case_count == 1
    assert result.gold_case_count == 1
    assert result.write.pack.cases[0].episode_id == "ep-1"
    assert "private ohmo pack" not in result.write.path.read_text(encoding="utf-8")


def test_build_ohmo_eval_pack_accepts_custom_pack_filename(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _prepare_gold_case(workspace, episode_id="ep-1", user_text="private ohmo pack")

    result = build_ohmo_eval_pack(
        workspace=workspace,
        pack_filename="custom_eval_pack.json",
    )

    assert result.write.path == workspace.resolve() / "evals" / "packs" / (
        "custom_eval_pack.json"
    )
    assert result.case_count == 1
    assert result.write.path.exists()


def test_run_ohmo_eval_smoke_writes_report_only_result(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _prepare_gold_case(workspace, episode_id="ep-1", user_text="private ohmo smoke")
    build_ohmo_eval_pack(workspace=workspace)

    result = run_ohmo_eval_smoke(workspace=workspace, limit=1, report_only=True)

    assert result.report_only is True
    assert result.write.path == workspace.resolve() / "evals" / "reports" / (
        "smoke_report.json"
    )
    assert result.write.report.case_count == 1
    assert result.write.report.passed_count == 1
    assert result.write.report.failed_count == 0
    assert "private ohmo smoke" not in result.write.path.read_text(encoding="utf-8")


def test_run_ohmo_eval_smoke_accepts_custom_report_filename(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _prepare_gold_case(workspace, episode_id="ep-1", user_text="private ohmo smoke")
    build_ohmo_eval_pack(workspace=workspace)

    result = run_ohmo_eval_smoke(
        workspace=workspace,
        report_filename="custom_smoke_report.json",
    )

    assert result.write.path == workspace.resolve() / "evals" / "reports" / (
        "custom_smoke_report.json"
    )
    assert result.write.report.case_count == 1
    assert result.write.path.exists()


def _prepare_gold_case(workspace: Path, *, episode_id: str, user_text: str):
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
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
