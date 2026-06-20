from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals import EvalEpisode, EvalEvent, promote_case_drafts
from ohmo.evals import (
    build_ohmo_eval_pack,
    check_ohmo_eval_run_config,
    get_eval_store,
    run_ohmo_eval_report,
    write_ohmo_eval_mine,
)
from ohmo.evals.runner import _build_agent_runner


def test_run_ohmo_eval_report_writes_metadata_replay_report(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_text="private ohmo eval request",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="gateway_final",
            payload={"text": "private ohmo eval answer"},
        )
    )
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
    build_ohmo_eval_pack(workspace=workspace)

    result = run_ohmo_eval_report(workspace=workspace, limit=1, report_only=True)

    assert result.report_only is True
    assert result.write.path == workspace.resolve() / "evals" / "reports" / "eval_report.json"
    assert result.write.report.metadata["executor_name"] == "replay-tools"
    assert result.write.report.case_count == 1
    assert result.write.report.passed_count == 1
    assert result.write.report.failed_count == 0
    serialized = result.write.path.read_text(encoding="utf-8")
    assert "private ohmo eval request" not in serialized
    assert "private ohmo eval answer" not in serialized


def test_run_ohmo_eval_report_accepts_custom_report_filename(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_text="private ohmo eval request",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="gateway_final",
            payload={"text": "private ohmo eval answer"},
        )
    )
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
    build_ohmo_eval_pack(workspace=workspace)

    result = run_ohmo_eval_report(
        workspace=workspace,
        report_filename="custom_eval_report.json",
    )

    assert result.write.path == workspace.resolve() / "evals" / "reports" / (
        "custom_eval_report.json"
    )
    assert result.write.report.case_count == 1
    assert result.write.path.exists()


def test_run_ohmo_eval_report_rejects_unknown_executor(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown eval executor"):
        run_ohmo_eval_report(
            workspace=tmp_path / "workspace",
            executor_name="live-agent",
        )


def test_run_ohmo_eval_report_rejects_unknown_agent_runner(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown eval agent runner"):
        run_ohmo_eval_report(
            workspace=tmp_path / "workspace",
            agent_runner_name="live-tools",
        )


def test_check_ohmo_eval_run_config_validates_pack_without_running(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_text="private ohmo eval request",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="gateway_final",
            payload={"text": "private ohmo eval answer"},
        )
    )
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
    build_ohmo_eval_pack(workspace=workspace)

    result = check_ohmo_eval_run_config(workspace=workspace, limit=1)

    assert result.pack_case_count == 1
    assert result.selected_case_count == 1
    assert result.executor_name == "replay-tools"
    assert result.agent_runner_name == "scripted"
    assert result.model == ""
    assert result.provider_profile == ""
    assert result.replay_tools_only is True
    assert not (workspace / "evals" / "reports" / "eval_report.json").exists()


def test_check_ohmo_eval_run_config_query_engine_auth_error_is_value_error(
    tmp_path: Path,
    monkeypatch,
):
    def fake_resolve_api_client(settings):
        raise SystemExit(1)

    monkeypatch.setattr(
        "ohmo.evals.runner.resolve_api_client_from_settings",
        fake_resolve_api_client,
    )

    with pytest.raises(ValueError, match="query-engine eval runner requires configured API authentication"):
        check_ohmo_eval_run_config(
            workspace=tmp_path / "workspace",
            agent_runner_name="query-engine",
        )


def test_build_query_engine_runner_uses_real_prompt_by_default(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        "ohmo.evals.runner.resolve_api_client_from_settings",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "ohmo.evals.runner.build_ohmo_system_prompt",
        lambda *args, **kwargs: "REAL_OHMO_PROMPT",
    )

    runner = _build_agent_runner(
        "query-engine",
        workspace=tmp_path,
        model=None,
        provider_profile=None,
        system_prompt=None,
    )

    assert runner._system_prompt == "REAL_OHMO_PROMPT"


def test_build_query_engine_runner_keeps_system_prompt_override(
    tmp_path: Path,
    monkeypatch,
):
    build_prompt_calls = 0

    def fake_build_ohmo_system_prompt(*args, **kwargs):
        nonlocal build_prompt_calls
        build_prompt_calls += 1
        return "REAL_OHMO_PROMPT"

    monkeypatch.setattr(
        "ohmo.evals.runner.resolve_api_client_from_settings",
        lambda settings: object(),
    )
    monkeypatch.setattr(
        "ohmo.evals.runner.build_ohmo_system_prompt",
        fake_build_ohmo_system_prompt,
    )

    runner = _build_agent_runner(
        "query-engine",
        workspace=tmp_path,
        model=None,
        provider_profile=None,
        system_prompt="custom override",
    )

    assert runner._system_prompt == "custom override"
    assert build_prompt_calls == 0


def test_run_ohmo_eval_report_query_engine_auth_error_is_value_error(
    tmp_path: Path,
    monkeypatch,
):
    def fake_resolve_api_client(settings):
        raise SystemExit(1)

    monkeypatch.setattr(
        "ohmo.evals.runner.resolve_api_client_from_settings",
        fake_resolve_api_client,
    )

    with pytest.raises(ValueError, match="query-engine eval runner requires configured API authentication"):
        run_ohmo_eval_report(
            workspace=tmp_path / "workspace",
            agent_runner_name="query-engine",
        )
