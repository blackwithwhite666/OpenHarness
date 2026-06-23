from __future__ import annotations

from pathlib import Path

import pytest

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.evals import (
    EvalEpisode,
    EvalEvent,
    LiveReadAgentRunner,
    SynthContext,
    promote_case_drafts,
)
from openharness.evals import (
    EvalRunPack,
    EvalRunPackCase,
    collect_text_facets,
    write_run_pack,
)
import ohmo.evals.runner as runner_module
from ohmo.evals import (
    build_ohmo_eval_pack,
    check_ohmo_eval_run_config,
    get_eval_store,
    run_ohmo_eval_report,
    run_ohmo_session_eval,
    write_ohmo_eval_mine,
)
from ohmo.evals.runner import _build_agent_runner
from ohmo.workspace import get_reminders_path


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
    assert result.write.report.metadata["fixture_match"] == "order"
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


def test_run_ohmo_eval_report_accepts_argument_fixture_matching(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    _append_session_episode(
        store,
        episode_id="ep-1",
        session_id="session-1",
        user_text="private ohmo eval request",
        tool_call_id="tool-1",
    )
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
    build_ohmo_eval_pack(workspace=workspace)

    result = run_ohmo_eval_report(
        workspace=workspace,
        limit=1,
        fixture_match="arguments",
    )

    assert result.write.report.metadata["fixture_match"] == "arguments"
    assert result.write.report.passed_count == 1
    serialized = result.write.path.read_text(encoding="utf-8")
    assert "SECRET_CITY" not in serialized


def test_run_ohmo_session_eval_writes_metadata_only_report(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    _append_session_episode(
        store,
        episode_id="ep-1",
        session_id="session-1",
        user_text="private ohmo first request",
        tool_call_id="tool-1",
    )
    _append_session_episode(
        store,
        episode_id="ep-2",
        session_id="session-1",
        user_text="private ohmo second request",
        tool_call_id="tool-2",
    )
    api_client = _PerTurnToolModelApiClient()
    monkeypatch.setattr(
        "ohmo.evals.runner.resolve_api_client_from_settings",
        lambda settings: api_client,
    )
    monkeypatch.setattr(
        "ohmo.evals.runner.build_ohmo_system_prompt",
        lambda *args, **kwargs: "REAL_OHMO_PROMPT",
    )

    result = run_ohmo_session_eval(
        workspace=workspace,
        limit=1,
        fixture_match="arguments",
    )

    assert result.write.path == workspace.resolve() / "evals" / "reports" / (
        "session_report.json"
    )
    assert result.write.report.report_kind == "session_report"
    assert result.write.report.session_count == 1
    assert result.write.report.passed_count == 1
    assert result.write.report.failed_count == 0
    assert result.write.report.metadata["privacy"] == "metadata_only"
    assert result.write.report.metadata["fixture_match"] == "arguments"
    assert result.write.report.metadata["gold_source"] == "captured_self_coverage"
    case = result.write.report.cases[0]
    assert case.session_id == "session-1"
    assert case.turn_count == 2
    assert case.checks["capability_coverage"] is True
    assert case.metadata["fixture_match"] == "arguments"

    serialized = result.write.path.read_text(encoding="utf-8")
    assert "private ohmo first request" not in serialized
    assert "private ohmo second request" not in serialized
    assert "private raw tool output" not in serialized
    assert "private model final" not in serialized
    assert "SECRET_CITY" not in serialized


def test_run_ohmo_session_eval_hybrid_user_sim_profile_records_metrics(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    _append_session_episode(
        store,
        episode_id="ep-1",
        session_id="session-1",
        user_text="private ohmo first request",
        tool_call_id="tool-1",
    )
    _append_session_episode(
        store,
        episode_id="ep-2",
        session_id="session-1",
        user_text="private ohmo second request",
        tool_call_id="tool-2",
    )
    agent_client = _ClarifyThenToolModelApiClient()
    user_client = _UserSimApiClient("private synthetic user answer")
    build_calls = []

    def fake_build_agent_runner_config(
        agent_runner_name,
        *,
        workspace,
        model,
        provider_profile,
        system_prompt,
        max_turns: int = 8,
    ):
        build_calls.append(
            {
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
            }
        )
        is_user_sim = provider_profile == "user-profile"
        return runner_module._AgentRunnerConfig(
            agent_runner=object(),
            agent_runner_name="query-engine",
            model=model or ("user-default-model" if is_user_sim else "agent-model"),
            provider_profile=provider_profile or "agent-profile",
            api_client=user_client if is_user_sim else agent_client,
            system_prompt=system_prompt or "AGENT_PROMPT",
            cwd=workspace,
        )

    monkeypatch.setattr(
        "ohmo.evals.runner._build_agent_runner_config",
        fake_build_agent_runner_config,
    )

    result = run_ohmo_session_eval(
        workspace=workspace,
        limit=1,
        provider_profile="agent-profile",
        user_sim_profile="user-profile",
        user_sim_model="user-model",
    )

    assert build_calls[1]["provider_profile"] == "user-profile"
    assert build_calls[1]["model"] == "user-model"
    assert str(build_calls[1]["system_prompt"]).startswith("You are simulating")
    assert result.write.report.metadata["user_simulation"] == "hybrid"
    assert result.write.report.metadata["user_sim_profile"] == "user-profile"
    assert result.write.report.metadata["user_sim_model"] == "user-model"
    assert result.write.report.metadata["mean_replay_hit_rate"] == 0.5
    assert result.write.report.metadata["total_llm_fallback_count"] == 1
    case = result.write.report.cases[0]
    assert case.metadata["user_turn_sources"] == ["replay", "llm_fallback"]
    assert case.metadata["replay_hit_rate"] == 0.5
    assert case.metadata["llm_fallback_count"] == 1
    assert case.metadata["ended_reason"] == "captured_exhausted"

    serialized = result.write.path.read_text(encoding="utf-8")
    assert "private ohmo first request" not in serialized
    assert "private ohmo second request" not in serialized
    assert "private synthetic user answer" not in serialized
    assert "Which city should I use?" not in serialized


def test_run_ohmo_session_eval_rejects_shared_user_sim_profile(
    tmp_path: Path,
    monkeypatch,
):
    def fake_build_agent_runner_config(
        agent_runner_name,
        *,
        workspace,
        model,
        provider_profile,
        system_prompt,
        max_turns: int = 8,
    ):
        return runner_module._AgentRunnerConfig(
            agent_runner=object(),
            agent_runner_name=agent_runner_name,
            model=model or "agent-model",
            provider_profile=provider_profile or "same-profile",
            api_client=object(),
            system_prompt=system_prompt or "AGENT_PROMPT",
            cwd=workspace,
        )

    monkeypatch.setattr(
        "ohmo.evals.runner._build_agent_runner_config",
        fake_build_agent_runner_config,
    )

    with pytest.raises(ValueError, match="user_sim_profile"):
        run_ohmo_session_eval(
            workspace=tmp_path / "workspace",
            provider_profile="same-profile",
            user_sim_profile="same-profile",
        )


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


def test_run_ohmo_eval_report_rejects_unknown_fixture_match(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown fixture match mode"):
        run_ohmo_eval_report(
            workspace=tmp_path / "workspace",
            fixture_match="wrong",
        )


def test_run_ohmo_session_eval_rejects_unknown_fixture_match(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown fixture match mode"):
        run_ohmo_session_eval(
            workspace=tmp_path / "workspace",
            fixture_match="wrong",
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


def test_check_ohmo_eval_run_config_accepts_synth_without_client(tmp_path: Path):
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

    result = check_ohmo_eval_run_config(
        workspace=workspace,
        limit=1,
        fixture_match="synth",
    )

    assert result.executor_name == "replay-tools"
    assert result.agent_runner_name == "scripted"
    assert result.selected_case_count == 1


def test_build_executor_threads_synth_context():
    synth_context = SynthContext(api_client=object(), model="codegen-model")

    executor = runner_module._build_executor(
        "replay-tools",
        agent_runner=runner_module.ReplayScriptAgentRunner(),
        fixture_match="synth",
        synth_context=synth_context,
    )

    assert executor.fixture_match_mode == "synth"
    assert executor._synth_context is synth_context


def test_run_ohmo_eval_report_synth_builds_codegen_context(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    _append_session_episode(
        store,
        episode_id="ep-synth",
        session_id="session-synth",
        user_text="private synth request",
        tool_call_id="tool-synth",
    )
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
    build_ohmo_eval_pack(workspace=workspace)
    api_client = _StaticTextApiClient(
        """
def respond(arguments, captured):
    return captured[0]["output"]
""".strip()
    )
    config_calls: list[dict[str, object]] = []

    def fake_build_agent_runner_config(
        agent_runner_name,
        *,
        workspace,
        model,
        provider_profile,
        system_prompt,
        max_turns: int = 8,
    ):
        config_calls.append(
            {
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
            }
        )
        if agent_runner_name == "query-engine":
            return runner_module._AgentRunnerConfig(
                agent_runner=runner_module.ReplayScriptAgentRunner(),
                agent_runner_name="query-engine",
                model=model or "resolved-synth-model",
                provider_profile=provider_profile or "resolved-synth-profile",
                api_client=api_client,
                system_prompt="",
                cwd=workspace,
            )
        return runner_module._AgentRunnerConfig(
            agent_runner=runner_module.ReplayScriptAgentRunner(),
            agent_runner_name="scripted",
            model="",
            provider_profile="",
        )

    monkeypatch.setattr(
        runner_module,
        "_build_agent_runner_config",
        fake_build_agent_runner_config,
    )

    result = run_ohmo_eval_report(
        workspace=workspace,
        limit=1,
        fixture_match="synth",
        synth_profile="synth-profile",
        synth_model="synth-model",
    )

    assert config_calls[0] == {
        "agent_runner_name": "query-engine",
        "model": "synth-model",
        "provider_profile": "synth-profile",
        "system_prompt": None,
    }
    assert result.write.report.metadata["fixture_match"] == "synth"
    assert result.write.report.metadata["synth_model"] == "synth-model"
    assert result.write.report.metadata["synth_provider_profile"] == "synth-profile"
    assert len(api_client.requests) == 1


def test_run_ohmo_eval_report_history_builds_segment_context(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    _append_session_episode(
        store,
        episode_id="ep-history",
        session_id="session-history",
        user_text="private history request",
        tool_call_id="tool-history",
    )
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
    build_ohmo_eval_pack(workspace=workspace)
    api_client = _StaticTextApiClient('{"start_index": null}')
    config_calls: list[dict[str, object]] = []

    def fake_build_agent_runner_config(
        agent_runner_name,
        *,
        workspace,
        model,
        provider_profile,
        system_prompt,
        max_turns: int = 8,
    ):
        config_calls.append(
            {
                "agent_runner_name": agent_runner_name,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
            }
        )
        if agent_runner_name == "query-engine":
            return runner_module._AgentRunnerConfig(
                agent_runner=runner_module.ReplayScriptAgentRunner(),
                agent_runner_name="query-engine",
                model=model or "resolved-history-model",
                provider_profile=provider_profile or "resolved-history-profile",
                api_client=api_client,
                system_prompt="",
                cwd=workspace,
            )
        return runner_module._AgentRunnerConfig(
            agent_runner=runner_module.ReplayScriptAgentRunner(),
            agent_runner_name="scripted",
            model="",
            provider_profile="",
        )

    monkeypatch.setattr(
        runner_module,
        "_build_agent_runner_config",
        fake_build_agent_runner_config,
    )

    result = run_ohmo_eval_report(
        workspace=workspace,
        limit=1,
        history_profile="history-profile",
        history_model="history-model",
    )

    assert config_calls[0] == {
        "agent_runner_name": "query-engine",
        "model": "history-model",
        "provider_profile": "history-profile",
        "system_prompt": None,
    }
    assert result.write.report.metadata["history_model"] == "history-model"
    assert result.write.report.metadata["history_provider_profile"] == "history-profile"


def test_run_ohmo_eval_report_threads_max_turns_to_agent_runner(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    _append_session_episode(
        store,
        episode_id="ep-max-turns",
        session_id="session-max-turns",
        user_text="max turns request",
        tool_call_id="tool-max-turns",
    )
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
    build_ohmo_eval_pack(workspace=workspace)
    seen_max_turns: list[int] = []

    def fake_build_agent_runner_config(
        agent_runner_name,
        *,
        workspace,
        model,
        provider_profile,
        system_prompt,
        max_turns: int = 8,
    ):
        seen_max_turns.append(max_turns)
        return runner_module._AgentRunnerConfig(
            agent_runner=runner_module.ReplayScriptAgentRunner(),
            agent_runner_name="scripted",
            model="",
            provider_profile="",
        )

    monkeypatch.setattr(
        runner_module,
        "_build_agent_runner_config",
        fake_build_agent_runner_config,
    )

    run_ohmo_eval_report(workspace=workspace, limit=1, max_turns=50)
    assert seen_max_turns == [50]

    with pytest.raises(ValueError):
        run_ohmo_eval_report(workspace=workspace, limit=1, max_turns=0)


def test_run_ohmo_eval_report_without_history_flags_does_not_build_history_client(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    _append_session_episode(
        store,
        episode_id="ep-no-history",
        session_id="session-no-history",
        user_text="private no history request",
        tool_call_id="tool-no-history",
    )
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
    build_ohmo_eval_pack(workspace=workspace)
    config_calls: list[str] = []

    def fake_build_agent_runner_config(
        agent_runner_name,
        *,
        workspace,
        model,
        provider_profile,
        system_prompt,
        max_turns: int = 8,
    ):
        del workspace, model, provider_profile, system_prompt
        config_calls.append(agent_runner_name)
        return runner_module._AgentRunnerConfig(
            agent_runner=runner_module.ReplayScriptAgentRunner(),
            agent_runner_name="scripted",
            model="",
            provider_profile="",
        )

    monkeypatch.setattr(
        runner_module,
        "_build_agent_runner_config",
        fake_build_agent_runner_config,
    )

    run_ohmo_eval_report(workspace=workspace, limit=1)

    assert config_calls == ["scripted"]


def test_run_ohmo_eval_report_sandbox_scores_reminder_state_outcome(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    _write_sandbox_reminder_pack(store)
    api_client = _ReminderCreateApiClient()
    monkeypatch.setattr(
        "ohmo.evals.runner.resolve_api_client_from_settings",
        lambda settings: api_client,
    )
    monkeypatch.setattr(
        "ohmo.evals.runner.build_ohmo_system_prompt",
        lambda *args, **kwargs: "REAL_OHMO_PROMPT",
    )

    result = run_ohmo_eval_report(
        workspace=workspace,
        agent_runner_name="sandbox",
        scorer="state_outcome_oracle_v1",
        limit=1,
    )

    assert result.write.report.case_count == 1
    assert result.write.report.passed_count == 1
    assert result.write.report.failed_count == 0
    assert result.write.report.cases[0].status == "passed"
    assert not get_reminders_path(workspace).exists()
    serialized = result.write.path.read_text(encoding="utf-8")
    assert "SECRET REMINDER TEXT" not in serialized
    assert "private sandbox reminder request" not in serialized
    assert "private sandbox final" not in serialized


def test_run_ohmo_eval_report_trajectory_judge_scores_metadata_only(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    _append_session_episode(
        store,
        episode_id="ep-judge",
        session_id="session-judge",
        user_text="private judge request",
        tool_call_id="tool-judge",
    )
    write_ohmo_eval_mine(workspace=workspace)
    promote_case_drafts(store)
    build_ohmo_eval_pack(workspace=workspace)
    api_client = _StaticTextApiClient("PASS - PRIVATE JUDGE REASON")
    config_calls: list[dict[str, object]] = []

    def fake_build_agent_runner_config(
        agent_runner_name,
        *,
        workspace,
        model,
        provider_profile,
        system_prompt,
        max_turns: int = 8,
    ):
        config_calls.append(
            {
                "agent_runner_name": agent_runner_name,
                "workspace": workspace,
                "model": model,
                "provider_profile": provider_profile,
                "system_prompt": system_prompt,
            }
        )
        if agent_runner_name == "query-engine":
            return runner_module._AgentRunnerConfig(
                agent_runner=runner_module.ReplayScriptAgentRunner(),
                agent_runner_name="query-engine",
                model=model or "resolved-judge-model",
                provider_profile=provider_profile or "resolved-judge-profile",
                api_client=api_client,
                system_prompt="",
                cwd=workspace,
            )
        return runner_module._AgentRunnerConfig(
            agent_runner=runner_module.ReplayScriptAgentRunner(),
            agent_runner_name="scripted",
            model="",
            provider_profile="",
        )

    monkeypatch.setattr(
        runner_module,
        "_build_agent_runner_config",
        fake_build_agent_runner_config,
    )

    result = run_ohmo_eval_report(
        workspace=workspace,
        scorer="trajectory_judge_v1",
        judge_profile="judge-profile",
        judge_model="judge-model",
        limit=1,
    )

    assert config_calls[0] == {
        "agent_runner_name": "query-engine",
        "workspace": workspace.resolve(),
        "model": "judge-model",
        "provider_profile": "judge-profile",
        "system_prompt": None,
    }
    assert result.write.report.metadata["scorer_name"] == "trajectory_judge_v1"
    assert result.write.report.metadata["judge_model"] == "judge-model"
    assert result.write.report.metadata["judge_provider_profile"] == "judge-profile"
    assert result.write.report.passed_count == 1
    case = result.write.report.cases[0]
    assert case.status == "passed"
    assert case.observed_trace is not None
    assert case.observed_trace.metadata["verdict"] == "pass"
    assert case.observed_trace.metadata["judge_model"] == "judge-model"
    serialized = result.write.path.read_text(encoding="utf-8")
    assert "private judge request" not in serialized
    assert "private captured final ep-judge" not in serialized
    assert "PRIVATE JUDGE REASON" not in serialized


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


def test_build_live_read_runner_config_uses_query_engine_settings(
    tmp_path: Path,
    monkeypatch,
):
    api_client = object()
    monkeypatch.setattr(
        "ohmo.evals.runner.resolve_api_client_from_settings",
        lambda settings: api_client,
    )
    monkeypatch.setattr(
        "ohmo.evals.runner.build_ohmo_system_prompt",
        lambda *args, **kwargs: "REAL_OHMO_PROMPT",
    )

    config = runner_module._build_agent_runner_config(
        "query-engine-live-read",
        workspace=tmp_path,
        model="eval-model",
        provider_profile=None,
        system_prompt=None,
    )

    assert isinstance(config.agent_runner, LiveReadAgentRunner)
    assert config.agent_runner_name == "query-engine-live-read"
    assert config.model == "eval-model"
    assert config.api_client is api_client
    assert config.system_prompt == "REAL_OHMO_PROMPT"
    assert config.cwd == tmp_path
    assert config.replay_tools_only is False
    assert config.agent_runner._live_typed_read_tool_names == (
        "read_file",
        "glob",
        "grep",
    )


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


class _PerTurnToolModelApiClient:
    def __init__(self) -> None:
        self.requests = []
        self._final_count = 0

    async def stream_message(self, request):
        self.requests.append(request)
        last_message = request.messages[-1]
        if any(isinstance(block, ToolResultBlock) for block in last_message.content):
            self._final_count += 1
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text=f"private model final {self._final_count}")],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return

        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id=f"toolu-session-{len(self.requests)}",
                        name="bash",
                        input={"command": "weather-cli forecast 'SECRET_CITY'"},
                    )
                ],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _ClarifyThenToolModelApiClient:
    def __init__(self) -> None:
        self.requests = []
        self._clarified = False
        self._final_count = 0

    async def stream_message(self, request):
        self.requests.append(request)
        last_message = request.messages[-1]
        if any(isinstance(block, ToolResultBlock) for block in last_message.content):
            self._final_count += 1
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text=f"private model final {self._final_count}")],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return

        if not self._clarified:
            self._clarified = True
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="Which city should I use?")],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return

        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id=f"toolu-session-{len(self.requests)}",
                        name="bash",
                        input={"command": "weather-cli forecast 'SECRET_CITY'"},
                    )
                ],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _UserSimApiClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=self.text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _StaticTextApiClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=self._text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _ReminderCreateApiClient:
    def __init__(self) -> None:
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        last_message = request.messages[-1]
        if any(isinstance(block, ToolResultBlock) for block in last_message.content):
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text="private sandbox final")],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return

        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="toolu-reminder-sandbox",
                        name="remind_create",
                        input={
                            "summary": "SECRET REMINDER TEXT",
                            "dtstart": "2099-01-01T09:00:00+03:00",
                            "rrule": None,
                            "mode": "static",
                            "tz": "Europe/Moscow",
                        },
                    )
                ],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def _write_sandbox_reminder_pack(store) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id="ep-sandbox-reminder",
            source="gateway",
            app="ohmo",
            session_id="session-sandbox",
            user_text="private sandbox reminder request",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-sandbox-reminder",
            kind="tool_started",
            tool_name="remind_create",
            tool_call_id="tool-reminder-1",
            payload={
                "input_summary": "private reminder tool input",
                "input": {
                    "summary": "SECRET REMINDER TEXT",
                    "dtstart": "2099-01-01T09:00:00+03:00",
                },
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-sandbox-reminder",
            kind="tool_completed",
            tool_name="remind_create",
            tool_call_id="tool-reminder-1",
            payload={"output_summary": "private reminder tool output"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-sandbox-reminder",
            kind="gateway_final",
            payload={"text": "private captured reminder final"},
        )
    )
    facets = collect_text_facets(store)
    facet_ids_by_kind = {item.facet.facet_kind: item.facet.facet_id for item in facets}
    write_run_pack(
        store,
        pack=EvalRunPack(
            pack_id="pack-sandbox-reminder",
            source_records_path="cases/gold_cases.jsonl",
            cases=[
                EvalRunPackCase(
                    gold_case_id="gold-sandbox-reminder",
                    case_id="case-sandbox-reminder",
                    episode_id="ep-sandbox-reminder",
                    case_kind="state",
                    input_facet_ids=[facet_ids_by_kind["user_request"]],
                    expected_facet_ids=[facet_ids_by_kind["assistant_final"]],
                    tool_names=["remind_create"],
                    capability_path=["remind_create"],
                    rubric=["Create one reminder in sandbox state."],
                    metadata={
                        "state_delta": _expected_one_reminder_delta("gold-key")
                    },
                )
            ],
            metadata={"privacy": "metadata_only", "case_count": 1},
        ),
    )


def _expected_one_reminder_delta(key: str) -> dict[str, object]:
    return {
        "reminders": {
            "added_keys": [key],
            "removed_keys": [],
            "count_before": 0,
            "count_after": 1,
            "status_counts_before": {},
            "status_counts_after": {"active": 1},
        },
        "memory": {
            "added_keys": [],
            "removed_keys": [],
            "count_before": 0,
            "count_after": 0,
        },
        "todos": {
            "added_keys": [],
            "removed_keys": [],
            "count_before": 0,
            "count_after": 0,
        },
        "changed": True,
    }


def _append_session_episode(
    store,
    *,
    episode_id: str,
    session_id: str,
    user_text: str,
    tool_call_id: str,
) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id=session_id,
            user_text=user_text,
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_started",
            tool_name="bash",
            tool_call_id=tool_call_id,
            payload={
                "input_summary": "private tool input",
                "input": {"command": "weather-cli forecast 'SECRET_CITY'"},
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_completed",
            tool_name="bash",
            tool_call_id=tool_call_id,
            payload={
                "output_summary": "private tool output",
                "output": {"text": "private raw tool output"},
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            payload={"text": f"private captured final {episode_id}"},
        )
    )
