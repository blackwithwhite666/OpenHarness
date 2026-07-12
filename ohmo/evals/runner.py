"""Ohmo helpers for executor-based eval reports."""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openharness.api.client import SupportsStreamingMessages
from openharness.api.resolver import (
    ApiClientResolutionError,
    resolve_api_client_from_settings,
)
from openharness.config import load_settings
from openharness.config.settings import Settings
from openharness.evals.executor import _run_eval_coroutine
from openharness.evals import (
    CachingApiClient,
    CompletionCache,
    NullApiClient,
    EvalExecutionReportWrite,
    EvalSessionReport,
    EvalSessionReportCase,
    FsSandboxAgentRunner,
    HistoryContext,
    HybridUserSimulator,
    EvalSessionGroup,
    FaithfulSessionRunner,
    LiveReadAgentRunner,
    LlmUserSimulator,
    QueryEngineEvalAgentRunner,
    ReplayUserSimulator,
    ReplayScriptAgentRunner,
    ReplayToolsExecutor,
    SandboxMutatingAgentRunner,
    SessionReplayRunner,
    SynthContext,
    FreezingJudgeScorer,
    RubricJudgeScorer,
    UserSimulator,
    build_gold_reference,
    collect_text_facets,
    derive_case_rubric,
    gold_capabilities_for_session,
    group_episodes_into_sessions,
    score_faithful_session,
    read_run_pack,
    resolve_execution_scorer,
    run_execution_report,
    score_session,
    segment_sessions_into_conversations,
)
from openharness.evals.grounding_blame import grounding_report_metadata
from openharness.evals.constraint_blame import constraint_report_metadata
from openharness.evals.judge import _default_grounding_search
from openharness.evals.runner import _report_output_path, _stable_id
from openharness.evals.state import compute_episode_state_delta, extract_state_keys
from openharness.prompts import build_runtime_system_prompt
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from openharness.tools.skill_tool import SkillTool, SkillToolInput
from openharness.utils.fs import atomic_write_text

from ohmo.evals.adapter import get_eval_store
from ohmo.evals.resources import build_ohmo_resource_snapshot
from ohmo.memory_store import MemoryStore
from ohmo.memory_tool import OhmoMemoryTool
from ohmo.prompts import build_ohmo_system_prompt
from ohmo.reminders.store import ReminderStore
from ohmo.reminders.tool import RemindCancelTool, RemindCreateTool, RemindListTool
from ohmo.todo_store import TodoStore
from ohmo.gateway.send_message_tool import (
    SendTelegramMessageInput,
    SendTelegramMessageTool,
)
from ohmo.todo_write_tool import OhmoTodoWriteTool
from ohmo.workspace import get_attachments_dir, get_plugins_dir, get_skills_dir


@dataclass(frozen=True)
class OhmoEvalRunResult:
    """Summary returned after running an Ohmo eval report."""

    write: EvalExecutionReportWrite
    report_only: bool


@dataclass(frozen=True)
class OhmoSessionEvalReportWrite:
    """Summary returned after writing an Ohmo session eval report."""

    report: EvalSessionReport
    path: Path
    relative_path: str


@dataclass(frozen=True)
class OhmoSessionEvalRunResult:
    """Summary returned after running an Ohmo session eval report."""

    write: OhmoSessionEvalReportWrite


@dataclass(frozen=True)
class OhmoEvalRunConfigCheckResult:
    """Metadata returned after validating an Ohmo eval run configuration."""

    pack_id: str
    pack_case_count: int
    selected_case_count: int
    executor_name: str
    agent_runner_name: str
    model: str
    provider_profile: str
    replay_tools_only: bool


@dataclass(frozen=True)
class _AgentRunnerConfig:
    agent_runner: (
        ReplayScriptAgentRunner
        | QueryEngineEvalAgentRunner
        | LiveReadAgentRunner
        | FsSandboxAgentRunner
        | SandboxMutatingAgentRunner
    )
    agent_runner_name: str
    model: str
    provider_profile: str
    api_client: SupportsStreamingMessages | None = None
    system_prompt: str = ""
    cwd: Path | None = None
    replay_tools_only: bool = True


SUPPORTED_EVAL_EXECUTOR_NAMES = ("replay-tools",)
SUPPORTED_EVAL_AGENT_RUNNER_NAMES = (
    "scripted",
    "query-engine",
    "query-engine-live-read",
    "fs-sandbox",
    "sandbox",
)
SUPPORTED_FIXTURE_MATCH_MODES = (
    "order",
    "arguments",
    "args_then_order",
    "synth",
    "synth_state",
)
_SUPPORTED_EXECUTORS = {
    "replay-tools": ReplayToolsExecutor,
    "replay_tools": ReplayToolsExecutor,
}
_SUPPORTED_AGENT_RUNNERS = set(SUPPORTED_EVAL_AGENT_RUNNER_NAMES)
_USER_SIM_SYSTEM_PROMPT = (
    "You are simulating the human user in an evaluation session. Given the "
    "original user goal and the conversation so far, reply as the user would. "
    "Return only the next user message."
)


def _session_progress_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}.progress.jsonl")


def _append_session_progress(
    progress_path: Path,
    *,
    index: int,
    total: int,
    case: EvalSessionReportCase,
) -> None:
    metadata = case.metadata or {}
    checks = metadata.get("check_rates") or case.checks or {}
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as progress_file:
            progress_file.write(
                json.dumps(
                    {
                        "index": index,
                        "total": total,
                        "session_id": case.session_id,
                        "status": case.status,
                        "checks": dict(checks),
                        "pass_rate": metadata.get(
                            "pass_rate",
                            1.0 if case.status == "passed" else 0.0,
                        ),
                        "flaky": bool(metadata.get("flaky", False)),
                        "ts": time.time(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            progress_file.flush()
    except Exception:
        pass


def run_ohmo_eval_report(
    *,
    workspace: str | Path | None = None,
    pack_filename: str = "eval_pack.json",
    report_filename: str = "eval_report.json",
    limit: int | None = None,
    report_only: bool = False,
    executor_name: str = "replay-tools",
    agent_runner_name: str = "scripted",
    model: str | None = None,
    provider_profile: str | None = None,
    system_prompt: str | None = None,
    scorer: str | None = None,
    judge_profile: str | None = None,
    judge_model: str | None = None,
    judge_votes: int = 3,
    judge_grounding: bool = False,
    synth_profile: str | None = None,
    synth_model: str | None = None,
    history_profile: str | None = None,
    history_model: str | None = None,
    samples: int = 1,
    fixture_match: str = "args_then_order",
    max_turns: int = 100,
    sandbox_net_mode: str = "none",
    sandbox_proxy_url: str | None = None,
    sandbox_browser_socket: str | None = None,
    sandbox_browser_name: str | None = None,
    sandbox_ro_dirs: tuple[str, ...] = (),
    cache_completions: str | Path | None = None,
    cache_strict: bool = False,
    cache_prune_to: str | Path | None = None,
    histories_file: str | Path | None = None,
    rubrics_file: str | Path | None = None,
    live_skill: bool = True,
    grounding_mode: str = "process",
    grounding_votes: int = 1,
) -> OhmoEvalRunResult:
    """Run deterministic replay-tools execution checks over an Ohmo eval pack."""
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    fixture_match = _validate_fixture_match(fixture_match)
    workspace_root = Path(workspace).expanduser().resolve() if workspace else None
    # One cache shared across the agent turns and the judge votes, so the
    # reported hit-rate covers every model call in the suite run.
    completion_cache = (
        CompletionCache(cache_completions, strict_offline=cache_strict)
        if cache_completions
        else None
    )
    selected_scorer = None
    judge_config: _AgentRunnerConfig | None = None
    synth_config: _AgentRunnerConfig | None = None
    history_config: _AgentRunnerConfig | None = None
    synth_context: SynthContext | None = None
    history_context: HistoryContext | None = None
    if scorer in (
        FreezingJudgeScorer.name,
        RubricJudgeScorer.name,
        "trajectory_judge_v1",  # back-compat aliases (pre-rename names)
        "trajectory_judge_v2",
    ):
        judge_config = _build_agent_runner_config(
            "query-engine",
            workspace=workspace_root,
            model=judge_model or model,
            provider_profile=judge_profile or provider_profile,
            system_prompt=None,
            completion_cache=completion_cache,
        )
        if judge_config.api_client is None:
            raise ValueError(f"{scorer} requires configured API authentication")
        if scorer in (RubricJudgeScorer.name, "trajectory_judge_v2"):
            selected_scorer = RubricJudgeScorer(
                api_client=judge_config.api_client,
                model=judge_config.model,
                votes=judge_votes,
                rubrics=_load_case_rubrics(rubrics_file),
                grounding_mode=grounding_mode,
                grounding_votes=grounding_votes,
            )
        else:
            selected_scorer = FreezingJudgeScorer(
                api_client=judge_config.api_client,
                model=judge_config.model,
                votes=judge_votes,
                grounding_mode=judge_grounding,
            )
    elif scorer:
        selected_scorer = resolve_execution_scorer(scorer)
    if fixture_match in ("synth", "synth_state"):
        synth_config = _build_agent_runner_config(
            "query-engine",
            workspace=workspace_root,
            model=synth_model or model,
            provider_profile=synth_profile or provider_profile,
            system_prompt=None,
            completion_cache=completion_cache,
        )
        if synth_config.api_client is None:
            raise ValueError("synth fixture match requires configured API authentication")
        synth_context = SynthContext(
            api_client=synth_config.api_client,
            model=synth_config.model,
        )
    if history_profile is not None or history_model is not None:
        history_config = _build_agent_runner_config(
            "query-engine",
            workspace=workspace_root,
            model=(
                history_model
                or (synth_config.model if synth_config is not None else synth_model)
                or (judge_config.model if judge_config is not None else judge_model)
                or model
            ),
            provider_profile=(
                history_profile
                or (
                    synth_config.provider_profile
                    if synth_config is not None
                    else synth_profile
                )
                or (
                    judge_config.provider_profile
                    if judge_config is not None
                    else judge_profile
                )
                or provider_profile
            ),
            system_prompt=None,
            completion_cache=completion_cache,
        )
        if history_config.api_client is None:
            raise ValueError("history segmentation requires configured API authentication")
        history_context = HistoryContext(
            api_client=history_config.api_client,
            model=history_config.model,
        )
    agent_runner_kwargs: dict[str, object] = {}
    if (
        sandbox_net_mode != "none"
        or sandbox_proxy_url is not None
        or sandbox_browser_socket is not None
        or sandbox_browser_name is not None
        or sandbox_ro_dirs
    ):
        agent_runner_kwargs = {
            "sandbox_net_mode": sandbox_net_mode,
            "sandbox_proxy_url": sandbox_proxy_url,
            "sandbox_browser_socket": sandbox_browser_socket,
            "sandbox_browser_name": sandbox_browser_name,
            "sandbox_ro_dirs": sandbox_ro_dirs,
        }
    agent_runner_config = _build_agent_runner_config(
        agent_runner_name,
        workspace=workspace_root,
        model=model,
        provider_profile=provider_profile,
        system_prompt=system_prompt,
        max_turns=max_turns,
        completion_cache=completion_cache,
        live_skill=live_skill,
        **agent_runner_kwargs,
    )
    executor = _build_executor(
        executor_name,
        agent_runner=agent_runner_config.agent_runner,
        fixture_match=fixture_match,
        synth_context=synth_context,
        schema_overrides=_mock_skill_schema_overrides(live_skill),
    )
    store = get_eval_store(workspace)
    pack = read_run_pack(store, pack_filename=pack_filename)
    conversation_histories = _load_conversation_histories(histories_file)
    try:
        write = run_execution_report(
            store,
            pack=pack,
            report_filename=report_filename,
            limit=limit,
            samples=samples,
            max_turns=max_turns,
            executor=executor,
            scorer=selected_scorer,
            history_context=history_context,
            conversation_histories=conversation_histories,
        )
    finally:
        # Persist even on partial failure so a crashed cold run is resumable.
        if completion_cache is not None:
            completion_cache.save()
    if judge_config is not None:
        write.report.metadata["judge_model"] = judge_config.model
        write.report.metadata["judge_provider_profile"] = judge_config.provider_profile
    if synth_config is not None:
        write.report.metadata["synth_model"] = synth_config.model
        write.report.metadata["synth_provider_profile"] = synth_config.provider_profile
    if history_config is not None:
        write.report.metadata["history_model"] = history_config.model
        write.report.metadata["history_provider_profile"] = history_config.provider_profile
    if completion_cache is not None:
        write.report.metadata["completion_cache"] = completion_cache.stats()
        write.report.metadata["completion_cache_path"] = (
            str(completion_cache.path) if completion_cache.path is not None else ""
        )
        if cache_prune_to is not None:
            pruned = completion_cache.save_pruned(cache_prune_to)
            write.report.metadata["completion_cache_pruned"] = {
                **pruned,
                "path": str(Path(cache_prune_to).expanduser().resolve()),
            }
    if (
        judge_config is not None
        or synth_config is not None
        or history_config is not None
        or completion_cache is not None
    ):
        atomic_write_text(write.path, write.report.model_dump_json(indent=2) + "\n")
    return OhmoEvalRunResult(write=write, report_only=report_only)


def run_ohmo_session_eval(
    *,
    workspace: str | Path | None = None,
    report_filename: str = "session_report.json",
    limit: int | None = None,
    model: str | None = None,
    provider_profile: str | None = None,
    system_prompt: str | None = None,
    samples: int = 1,
    gold_capabilities_by_session: Mapping[str, Sequence[str]] | None = None,
    user_sim_profile: str | None = None,
    user_sim_model: str | None = None,
    clarification_allowed_by_session: Mapping[str, bool] | None = None,
    fixture_match: str = "args_then_order",
    max_session_turns: int | None = None,
    max_turns: int = 100,
    segment: bool = False,
    gap_minutes: float = 30.0,
    min_turns: int = 2,
    user_sim_goal_anchored: bool = True,
    judge_votes: int = 1,
    grounding_votes: int = 1,
    preset: str = "inner",
    sandbox_net_mode: str = "none",
    sandbox_proxy_url: str | None = None,
    sandbox_browser_socket: str | None = None,
    sandbox_browser_name: str | None = None,
    sandbox_ro_dirs: tuple[str, ...] = (),
) -> OhmoSessionEvalRunResult:
    """Run P0 session replay checks over captured Ohmo eval episodes.

    With ``segment=True`` the coarse per-chat threads are cut into bounded
    same-task conversations via a time-gap split before replaying — a
    ``session_id`` groups an entire chat's unrelated tasks (e.g. "restaurant
    reviews" then, days later, "schedule a 1-1"), so replaying it whole makes
    the user simulator improvise across topics and the capability union is
    unsatisfiable. Segmenting is opt-in (default off) to preserve the legacy
    whole-thread behaviour for existing callers.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if samples < 1:
        raise ValueError("samples must be positive")
    if max_session_turns is not None and max_session_turns <= 0:
        raise ValueError("max_session_turns must be positive")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if gap_minutes <= 0:
        raise ValueError("gap_minutes must be positive")
    if min_turns < 1:
        raise ValueError("min_turns must be positive")
    if judge_votes < 1:
        raise ValueError("judge_votes must be positive")
    if grounding_votes < 1:
        raise ValueError("grounding_votes must be positive")
    fixture_match = _validate_fixture_match(fixture_match)
    preset = preset.strip().lower()
    if preset not in {"inner", "faithful"}:
        raise ValueError("session preset must be one of: inner, faithful")
    if user_sim_model is not None and user_sim_profile is None:
        raise ValueError("user_sim_model requires user_sim_profile")

    workspace_root = Path(workspace).expanduser().resolve() if workspace else None
    agent_runner_name = "fs-sandbox" if preset == "faithful" else "query-engine"
    agent_runner_config = _build_agent_runner_config(
        agent_runner_name,
        workspace=workspace_root,
        model=model,
        provider_profile=provider_profile,
        system_prompt=system_prompt,
        sandbox_net_mode=sandbox_net_mode,
        sandbox_proxy_url=sandbox_proxy_url,
        sandbox_browser_socket=sandbox_browser_socket,
        sandbox_browser_name=sandbox_browser_name,
        sandbox_ro_dirs=sandbox_ro_dirs,
    )
    if agent_runner_config.api_client is None:
        raise ValueError("session eval runner requires configured API authentication")
    synth_context = (
        SynthContext(
            api_client=agent_runner_config.api_client,
            model=agent_runner_config.model,
        )
        if fixture_match in ("synth", "synth_state")
        else None
    )

    user_simulator_factory: Callable[[], UserSimulator] | None = None
    user_sim_resolved_profile = ""
    user_sim_resolved_model = ""
    if user_sim_profile is not None:
        if user_sim_profile == agent_runner_config.provider_profile:
            raise ValueError("user_sim_profile must differ from provider_profile")
        user_sim_config = _build_agent_runner_config(
            "query-engine",
            workspace=workspace_root,
            model=user_sim_model,
            provider_profile=user_sim_profile,
            system_prompt=_USER_SIM_SYSTEM_PROMPT,
        )
        if user_sim_config.api_client is None:
            raise ValueError("user simulator requires configured API authentication")
        if user_sim_config.provider_profile == agent_runner_config.provider_profile:
            raise ValueError("user_sim_profile must differ from provider_profile")
        user_sim_resolved_profile = user_sim_config.provider_profile
        user_sim_resolved_model = user_sim_config.model

        def _new_user_simulator() -> UserSimulator:
            return HybridUserSimulator(
                replay=ReplayUserSimulator(),
                llm=LlmUserSimulator(
                    api_client=user_sim_config.api_client,
                    model=user_sim_config.model,
                    system_prompt=user_sim_config.system_prompt,
                    goal_anchored=user_sim_goal_anchored,
                ),
            )

        user_simulator_factory = _new_user_simulator

    store = get_eval_store(workspace)
    if segment:
        # L2: cut each coarse per-chat thread into bounded same-task
        # conversations, so a "session" is one coherent task rather than a
        # whole chat's grab-bag of unrelated requests.
        conversations = segment_sessions_into_conversations(
            store, app="ohmo", gap_minutes=gap_minutes, min_turns=min_turns
        )
        groups = [
            EvalSessionGroup(
                session_id=f"{conversation.session_id}#{conversation.segment_index}",
                episode_ids=conversation.episode_ids,
            )
            for conversation in conversations
        ]
    else:
        groups = group_episodes_into_sessions(store, app="ohmo")
    groups = groups[:limit] if limit is not None else groups
    if not groups:
        raise ValueError("eval store must contain ohmo sessions")

    if preset == "faithful":
        base_faithful_runner = agent_runner_config.agent_runner

        def _faithful_agent_runner_factory(workspace: Path) -> FsSandboxAgentRunner:
            if isinstance(base_faithful_runner, FsSandboxAgentRunner):
                return FsSandboxAgentRunner(
                    api_client=agent_runner_config.api_client,
                    model=agent_runner_config.model,
                    system_prompt=agent_runner_config.system_prompt,
                    cwd=workspace,
                    max_turns=max_turns,
                    max_tokens=base_faithful_runner._max_tokens,
                    timeout=base_faithful_runner._timeout,
                    net_mode=base_faithful_runner._net_mode,
                    proxy_url=base_faithful_runner._proxy_url,
                    browser_socket=base_faithful_runner._browser_socket,
                    browser_cli_name=base_faithful_runner._browser_cli_name,
                    live_mcp_server_names=base_faithful_runner._live_mcp_server_names,
                    mutable_dirs=base_faithful_runner._mutable_dirs,
                    ro_source_dirs=base_faithful_runner._ro_source_dirs,
                    extra_ro_source_dirs=(),
                    sandbox_bin_dirs=base_faithful_runner._sandbox_bin_dirs,
                    persist_cwd=True,
                )
            return base_faithful_runner

        runner = FaithfulSessionRunner(
            api_client=agent_runner_config.api_client,
            model=agent_runner_config.model,
            system_prompt=agent_runner_config.system_prompt,
            cwd=agent_runner_config.cwd,
            fixture_match_mode=fixture_match,
            max_session_turns=max_session_turns,
            max_turns=max_turns,
            synth_context=synth_context,
            agent_runner_factory=_faithful_agent_runner_factory
            if isinstance(base_faithful_runner, FsSandboxAgentRunner)
            else None,
            agent_runner=base_faithful_runner
            if not isinstance(base_faithful_runner, FsSandboxAgentRunner)
            else None,
            sandbox_tool_factory=_ohmo_sandbox_tool_factory,
            sandbox_state_fn=_ohmo_sandbox_state,
        )
    else:
        runner = SessionReplayRunner(
            api_client=agent_runner_config.api_client,
            model=agent_runner_config.model,
            system_prompt=agent_runner_config.system_prompt,
            cwd=agent_runner_config.cwd,
            fixture_match_mode=fixture_match,
            max_session_turns=max_session_turns,
            max_turns=max_turns,
            synth_context=synth_context,
        )
    path = _report_output_path(store, report_filename)
    progress_path = _session_progress_path(path)
    cases = []
    total = len(groups)
    for index, group in enumerate(groups, start=1):
        case = _run_session_report_case_sampled(
            store=store,
            group=group,
            runner=runner,
            samples=samples,
            gold_capabilities_by_session=gold_capabilities_by_session,
            user_simulator_factory=user_simulator_factory,
            clarification_allowed_by_session=clarification_allowed_by_session,
            judge_votes=judge_votes,
            grounding_votes=grounding_votes,
        )
        cases.append(case)
        _append_session_progress(
            progress_path,
            index=index,
            total=total,
            case=case,
        )
    passed_count = sum(1 for case in cases if case.status == "passed")
    failed_count = len(cases) - passed_count
    mean_replay_hit_rate = (
        sum(float(case.metadata.get("replay_hit_rate", 0.0)) for case in cases)
        / len(cases)
        if cases
        else 0.0
    )
    report = EvalSessionReport(
        report_id=_stable_id(
            "session-eval",
            *(group.session_id for group in groups),
        ),
        session_count=len(cases),
        passed_count=passed_count,
        failed_count=failed_count,
        cases=cases,
        metadata={
            "privacy": "metadata_only",
            "mode": "session_replay",
            "runner_name": runner.name,
            "fixture_match": fixture_match,
            "model": agent_runner_config.model,
            "provider_profile": agent_runner_config.provider_profile,
            "user_simulation": "hybrid" if user_simulator_factory else "replay",
            "user_sim_profile": user_sim_resolved_profile,
            "user_sim_model": user_sim_resolved_model,
            "mean_replay_hit_rate": mean_replay_hit_rate,
            "total_llm_fallback_count": sum(
                int(case.metadata.get("llm_fallback_count", 0)) for case in cases
            ),
            "gold_source": (
                "provided"
                if gold_capabilities_by_session is not None
                else "captured_self_coverage"
            ),
            "limit": limit or 0,
            "samples": samples,
            "judge_votes": judge_votes,
            "grounding_votes": grounding_votes,
        },
    )
    atomic_write_text(path, report.model_dump_json(indent=2) + "\n")
    return OhmoSessionEvalRunResult(
        write=OhmoSessionEvalReportWrite(
            report=report,
            path=path,
            relative_path=path.relative_to(store.root).as_posix(),
        )
    )


def check_ohmo_eval_run_config(
    *,
    workspace: str | Path | None = None,
    pack_filename: str = "eval_pack.json",
    limit: int | None = None,
    executor_name: str = "replay-tools",
    agent_runner_name: str = "scripted",
    model: str | None = None,
    provider_profile: str | None = None,
    system_prompt: str | None = None,
    scorer: str | None = None,
    fixture_match: str = "args_then_order",
) -> OhmoEvalRunConfigCheckResult:
    """Validate an eval run configuration without executing eval cases."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    fixture_match = _validate_fixture_match(fixture_match)
    if scorer:
        resolve_execution_scorer(scorer)
    workspace_root = Path(workspace).expanduser().resolve() if workspace else None
    agent_runner_config = _build_agent_runner_config(
        agent_runner_name,
        workspace=workspace_root,
        model=model,
        provider_profile=provider_profile,
        system_prompt=system_prompt,
    )
    executor = _build_executor(
        executor_name,
        agent_runner=agent_runner_config.agent_runner,
        fixture_match=fixture_match,
    )
    store = get_eval_store(workspace)
    pack = read_run_pack(store, pack_filename=pack_filename)
    selected_case_count = len(pack.cases[:limit] if limit is not None else pack.cases)
    if selected_case_count == 0:
        raise ValueError("eval pack must contain cases")
    return OhmoEvalRunConfigCheckResult(
        pack_id=pack.pack_id,
        pack_case_count=len(pack.cases),
        selected_case_count=selected_case_count,
        executor_name=executor.name,
        agent_runner_name=agent_runner_config.agent_runner_name,
        model=agent_runner_config.model,
        provider_profile=agent_runner_config.provider_profile,
        replay_tools_only=agent_runner_config.replay_tools_only,
    )


def _build_executor(
    executor_name: str,
    *,
    agent_runner: (
        ReplayScriptAgentRunner
        | QueryEngineEvalAgentRunner
        | LiveReadAgentRunner
        | FsSandboxAgentRunner
        | SandboxMutatingAgentRunner
    ),
    fixture_match: str = "args_then_order",
    synth_context: SynthContext | None = None,
    schema_overrides: dict[str, tuple[str, type]] | None = None,
) -> ReplayToolsExecutor:
    fixture_match = _validate_fixture_match(fixture_match)
    normalized = executor_name.strip().lower()
    executor_factory = _SUPPORTED_EXECUTORS.get(normalized)
    if executor_factory is None:
        supported = ", ".join(SUPPORTED_EVAL_EXECUTOR_NAMES)
        raise ValueError(
            f"unknown eval executor: {executor_name}. Supported executors: {supported}"
        )
    return executor_factory(
        agent_runner=agent_runner,
        match_mode=fixture_match,
        synth_context=synth_context,
        schema_overrides=schema_overrides,
    )


def _mock_skill_schema_overrides(live_skill: bool) -> dict[str, tuple[str, type]] | None:
    """When skill isn't injected live, make its replay fixture wear the real
    SkillTool schema (description + params) — the "mock" skill: captured SKILL.md
    output, genuine tool interface, no ~/.ohmo/skills dependency. Without this the
    replayed skill shows a generic schema and the agent under-invokes it."""
    if live_skill:
        return None
    return {"skill": (SkillTool.description, SkillToolInput)}


def _validate_fixture_match(fixture_match: str) -> str:
    normalized = fixture_match.strip().lower()
    if normalized not in SUPPORTED_FIXTURE_MATCH_MODES:
        supported = ", ".join(SUPPORTED_FIXTURE_MATCH_MODES)
        raise ValueError(
            f"unknown fixture match mode: {fixture_match}. Supported modes: {supported}"
        )
    return normalized


def _build_agent_runner(
    agent_runner_name: str,
    *,
    workspace: Path | None,
    model: str | None,
    provider_profile: str | None,
    system_prompt: str | None,
) -> (
    ReplayScriptAgentRunner
    | QueryEngineEvalAgentRunner
    | LiveReadAgentRunner
    | FsSandboxAgentRunner
    | SandboxMutatingAgentRunner
):
    return _build_agent_runner_config(
        agent_runner_name,
        workspace=workspace,
        model=model,
        provider_profile=provider_profile,
        system_prompt=system_prompt,
    ).agent_runner


def _resolve_eval_system_prompt(
    settings: Settings,
    *,
    workspace: Path | None,
    system_prompt: str | None,
) -> str:
    """Build the eval agent's system prompt to match the live gateway.

    An explicit ``system_prompt`` override is returned verbatim. Otherwise the
    prompt is assembled exactly like the production gateway does
    (``build_runtime()`` -> ``build_runtime_system_prompt()``): the ohmo persona
    plus the "# Available Skills" catalog (and delegation/reasoning sections).
    The bare ``build_ohmo_system_prompt()`` omits the skills catalog, which left
    the query-engine eval agent blind to its own skills (pdf/maps/browser/...)
    and made it under-investigate relative to production.
    """
    if system_prompt:
        return system_prompt
    prompt_cwd = workspace or Path.cwd()
    persona = build_ohmo_system_prompt(prompt_cwd, workspace=workspace)
    if workspace is not None:
        skill_dirs: tuple[str, ...] = (str(get_skills_dir(workspace)),)
        plugin_roots: tuple[str, ...] = (str(get_plugins_dir(workspace)),)
    else:
        skill_dirs = ()
        plugin_roots = ()
    return build_runtime_system_prompt(
        settings.model_copy(update={"system_prompt": persona}),
        cwd=prompt_cwd,
        extra_skill_dirs=skill_dirs,
        extra_plugin_roots=plugin_roots,
        include_project_memory=False,
    )


_MOCK_STATIC_PUBLISHER_SH = """#!/bin/bash
# Eval-only mock of static_publisher-cli: returns a content-addressed
# https://worfalomey.top/static/<hash>/ URL WITHOUT uploading (no network, no
# side effect). Mirrors _MockSendTelegramMessageTool — the real publisher's
# ACL/upload is tested elsewhere; in fs-sandbox eval we only need the publish
# step to yield a stable, groundable URL.
cmd="${1:-}"
if [ "$cmd" = "publish" ]; then
  shift || true
  dir="."
  if [ $# -ge 1 ]; then case "$1" in -*) ;; *) dir="$1";; esac; fi
  json=0
  for a in "$@"; do [ "$a" = "--json" ] && json=1; done
  slug=$( { find "$dir" -type f -exec cat {} + 2>&-; printf '%s' "$dir"; } | sha256sum | cut -c1-16 )
  url="https://worfalomey.top/static/$slug/"
  if [ "$json" = "1" ]; then
    printf '{"url": "%s", "hash": "%s", "mock": true}\\n' "$url" "$slug"
  else
    printf 'Published: %s\\n' "$url"
  fi
elif [ "$cmd" = "list" ]; then
  printf '%s\\n' "$@" | grep -q -- --json && echo "[]" || echo "(mock static_publisher: no entries)"
else
  echo "mock static_publisher-cli: $*"
fi
"""


_MOCK_DROPBOX_SH = """#!/bin/bash
# Eval-only mock of the dropbox CLI (nautilus-dropbox): returns a deterministic
# https://www.dropbox.com/s/<hash>/<name> share link for `dropbox sharelink PATH`
# WITHOUT a running Dropbox daemon or network. Mirrors the static_publisher mock —
# the real share ACL/sync is tested elsewhere; in eval we only need the share step
# to yield a stable, groundable URL so "give me a Dropbox link" cases can complete.
cmd="${1:-}"
case "$cmd" in
  sharelink|share)
    path="${2:-}"
    if [ -z "$path" ]; then echo "dropbox $cmd: missing path" >&2; exit 1; fi
    name=$(basename "$path")
    hash=$(printf '%s' "$path" | sha256sum | cut -c1-15)
    echo "https://www.dropbox.com/s/$hash/$name?dl=0"
    ;;
  status) echo "Up to date (mock)";;
  running) exit 0;;
  *) echo "mock dropbox: $*";;
esac
"""


def _build_sandbox_skill_bin(
    workspace: Path | None, *, live_skill: bool
) -> tuple[Path, ...]:
    """Expose skill CLIs on the fs-sandbox PATH, with a mocked publisher.

    Skill CLIs live nested (``skills/<name>/<name>-cli``), so a bare invocation
    resolves as "command not found" inside the jail even though the dir is
    ro-bound (PATH is only ``/usr/bin:/bin``). Symlink them flat, and shadow the
    side-effecting ``static_publisher-cli`` with a deterministic mock (a
    content-addressed ``worfalomey.top/static/<hash>/`` URL, no upload) so the
    faithful lane's publish step yields a groundable URL without a real side
    effect. Returns bin dirs to prepend to PATH (mock dir first, so it wins).
    No-op unless ``live_skill`` (the faithful lane).
    """
    if not live_skill or workspace is None:
        return ()
    skills_dir = get_skills_dir(workspace)
    bin_root = Path(tempfile.mkdtemp(prefix="openharness-eval-skillbin-"))
    mock_dir = bin_root / "mock"
    flat_dir = bin_root / "skills"
    mock_dir.mkdir()
    flat_dir.mkdir()
    if skills_dir.is_dir():
        for cli in skills_dir.glob("*/*-cli"):
            link = flat_dir / cli.name
            if not link.exists():
                try:
                    link.symlink_to(cli.resolve())
                except OSError:
                    continue
    mock_publisher = mock_dir / "static_publisher-cli"
    mock_publisher.write_text(_MOCK_STATIC_PUBLISHER_SH, encoding="utf-8")
    mock_publisher.chmod(0o755)
    # Shadow the real ~/bin/dropbox (nautilus daemon controller, needs a running
    # daemon + creds) with a deterministic share-link mock; mock_dir is first on
    # PATH so it wins.
    mock_dropbox = mock_dir / "dropbox"
    mock_dropbox.write_text(_MOCK_DROPBOX_SH, encoding="utf-8")
    mock_dropbox.chmod(0o755)
    dirs = [mock_dir, flat_dir]
    home_bin = Path.home() / "bin"
    if home_bin.is_dir():
        dirs.append(home_bin)
    return tuple(dirs)


def _build_agent_runner_config(
    agent_runner_name: str,
    *,
    workspace: Path | None,
    model: str | None,
    provider_profile: str | None,
    system_prompt: str | None,
    max_turns: int = 100,
    sandbox_net_mode: str = "none",
    sandbox_proxy_url: str | None = None,
    sandbox_browser_socket: str | None = None,
    sandbox_browser_name: str | None = None,
    sandbox_ro_dirs: tuple[str, ...] = (),
    completion_cache: CompletionCache | None = None,
    live_skill: bool = True,
) -> _AgentRunnerConfig:
    normalized = agent_runner_name.strip().lower()
    if normalized not in _SUPPORTED_AGENT_RUNNERS:
        supported = ", ".join(SUPPORTED_EVAL_AGENT_RUNNER_NAMES)
        raise ValueError(
            f"unknown eval agent runner: {agent_runner_name}. Supported runners: {supported}"
        )
    if normalized == "scripted":
        return _AgentRunnerConfig(
            agent_runner=ReplayScriptAgentRunner(),
            agent_runner_name="scripted",
            model="",
            provider_profile="",
        )

    settings = load_settings().merge_cli_overrides(
        model=model,
        active_profile=provider_profile,
    )
    settings = settings.materialize_active_profile()
    if completion_cache is not None and completion_cache.strict_offline:
        # Offline replay (--cache-strict): the model is never called (hits from
        # cache, misses -> stub), so don't require API auth — a CI host with no
        # provider credentials can still run the frozen bundle. Pin --model so the
        # cache key matches the recording.
        api_client: SupportsStreamingMessages = NullApiClient()
    else:
        try:
            api_client = resolve_api_client_from_settings(settings)
        except (ApiClientResolutionError, SystemExit) as exc:
            raise ValueError(
                f"{normalized} eval runner requires configured API authentication"
            ) from exc
    if completion_cache is not None:
        # Inner-loop cache: replay recorded model completions on unchanged
        # prompts; record live completions on a cold run. Same wrapper for the
        # agent turns and the judge votes, so they share one hit/miss counter.
        api_client = CachingApiClient(api_client, completion_cache)
    resolved_prompt = _resolve_eval_system_prompt(
        settings, workspace=workspace, system_prompt=system_prompt
    )
    if normalized == "query-engine-live-read":
        return _AgentRunnerConfig(
            agent_runner=LiveReadAgentRunner(
                api_client=api_client,
                model=settings.model,
                system_prompt=resolved_prompt,
                cwd=workspace,
                max_turns=max_turns,
                live_mcp_server_names=("google_search",),
                live_typed_read_tool_names=("read_file", "glob", "grep"),
                live_local_tool_factory=_make_live_local_tool_factory(workspace, include_skill=live_skill),
                live_read_passthrough_roots=(
                    (get_attachments_dir(workspace),) if workspace is not None else ()
                ),
            ),
            agent_runner_name="query-engine-live-read",
            model=settings.model,
            provider_profile=settings.active_profile,
            api_client=api_client,
            system_prompt=resolved_prompt,
            cwd=workspace,
            replay_tools_only=False,
        )
    if normalized == "fs-sandbox":
        return _AgentRunnerConfig(
            agent_runner=FsSandboxAgentRunner(
                api_client=api_client,
                model=settings.model,
                system_prompt=resolved_prompt,
                cwd=workspace,
                max_turns=max_turns,
                net_mode=sandbox_net_mode,
                proxy_url=sandbox_proxy_url,
                browser_socket=sandbox_browser_socket,
                browser_cli_name=sandbox_browser_name,
                extra_ro_source_dirs=sandbox_ro_dirs,
                sandbox_bin_dirs=_build_sandbox_skill_bin(
                    workspace, live_skill=live_skill
                ),
                live_mcp_server_names=(
                    ("google_search",) if sandbox_net_mode.startswith("netns:") else ()
                ),
            ),
            agent_runner_name="fs-sandbox",
            model=settings.model,
            provider_profile=settings.active_profile,
            api_client=api_client,
            system_prompt=resolved_prompt,
            cwd=workspace,
            replay_tools_only=False,
        )
    if normalized == "sandbox":
        return _AgentRunnerConfig(
            agent_runner=SandboxMutatingAgentRunner(
                api_client=api_client,
                model=settings.model,
                system_prompt=resolved_prompt,
                cwd=workspace,
                sandbox_tool_factory=_ohmo_sandbox_tool_factory,
                sandbox_state_fn=_ohmo_sandbox_state,
            ),
            agent_runner_name="sandbox",
            model=settings.model,
            provider_profile=settings.active_profile,
            api_client=api_client,
            system_prompt=resolved_prompt,
            cwd=workspace,
            replay_tools_only=False,
        )
    return _AgentRunnerConfig(
        agent_runner=QueryEngineEvalAgentRunner(
            api_client=api_client,
            model=settings.model,
            system_prompt=resolved_prompt,
            cwd=workspace,
            max_turns=max_turns,
            live_local_tool_factory=_make_live_local_tool_factory(workspace, include_skill=live_skill),
            local_state_root=_stable_local_state_root(completion_cache),
        ),
        agent_runner_name="query-engine",
        model=settings.model,
        provider_profile=settings.active_profile,
        api_client=api_client,
        system_prompt=resolved_prompt,
        cwd=workspace,
    )


_CACHED_EVAL_LOCAL_STATE_DIR = "openharness-eval-local-state"


def _stable_local_state_root(completion_cache: CompletionCache | None) -> Path | None:
    """A fixed local-state dir for cached runs.

    Live local tools (todo_write) echo their state's absolute path in tool
    output, which becomes part of the completion-cache key. A random per-run
    mkdtemp — OR a path derived from the cache file — makes that output differ
    between record and replay (or on a different machine/CI where the cache lives
    elsewhere), which breaks the cache. So use ONE fixed path: identical every
    run, every machine, regardless of where the cache file sits, which is exactly
    what a portable, committable cache needs. Eval runs are sequential and each
    run wipes this dir at start, so a shared constant is safe. Returns None when
    caching is off, preserving the mkdtemp isolation for normal runs.
    """
    if completion_cache is None:
        return None
    return Path(tempfile.gettempdir()) / _CACHED_EVAL_LOCAL_STATE_DIR


def _load_conversation_histories(
    histories_file: str | Path | None,
) -> dict[str, tuple[tuple[str, str], ...]] | None:
    """Load baked per-case conversation history: {case_id: [[role, text], ...]}.

    Committed in the eval bundle so a slim store replays the same seeded session
    history the full store produced (the recompute reads the whole store's
    episode set/order and isn't portable). See run_execution_report.
    """
    if not histories_file:
        return None
    raw = json.loads(Path(histories_file).expanduser().read_text(encoding="utf-8"))
    return {
        case_id: tuple((str(role), str(text)) for role, text in turns)
        for case_id, turns in raw.items()
    }


def _load_case_rubrics(
    rubrics_file: str | Path | None,
) -> dict[str, dict[str, object]] | None:
    """Load per-case derived checklists: {case_id: {task_completion:[...], grounding:[...]}}.

    Committed in the bundle (evals/rubrics.json) so rubric_judge grades the
    two hard-gate aspects (task_completion, grounding) against requirements
    distilled offline from the gold episode (see ``derive_ohmo_case_rubrics``).
    """
    if not rubrics_file:
        return None
    raw = json.loads(Path(rubrics_file).expanduser().read_text(encoding="utf-8"))
    return {str(case_id): value for case_id, value in raw.items()}


@dataclass(frozen=True)
class OhmoRubricDeriveResult:
    """Summary of an offline rubric-derivation pass."""

    path: Path
    case_count: int
    derived_count: int


def derive_ohmo_case_rubrics(
    *,
    output_path: str | Path,
    workspace: str | Path | None = None,
    pack_filename: str = "eval_pack.json",
    model: str | None = None,
    provider_profile: str | None = None,
    limit: int | None = None,
) -> OhmoRubricDeriveResult:
    """Distil per-case task_completion + grounding checklists from gold episodes.

    Offline live-model step (like recording the completion cache): reads each
    pack case's gold reference (goal + gold trajectory + gold answer), asks the
    model to extract path-independent requirements, and writes
    ``evals/rubrics.json`` for ``rubric_judge`` to gate against. Run once
    on a host with model auth; commit the result to the bundle.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    workspace_root = Path(workspace).expanduser().resolve() if workspace else None
    judge_config = _build_agent_runner_config(
        "query-engine",
        workspace=workspace_root,
        model=model,
        provider_profile=provider_profile,
        system_prompt=None,
    )
    if judge_config.api_client is None:
        raise ValueError("rubric derivation requires configured API authentication")
    store = get_eval_store(workspace)
    pack = read_run_pack(store, pack_filename=pack_filename)
    cases = pack.cases[:limit] if limit is not None else pack.cases
    facet_inputs_by_id = {item.facet.facet_id: item for item in collect_text_facets(store)}
    rubrics: dict[str, dict[str, list[dict[str, str]]]] = {}
    for case in cases:
        reference = build_gold_reference(store, case, facet_inputs_by_id)
        if reference is None:
            continue
        goal, gold_answer, gold_trajectory = reference
        checklist = derive_case_rubric(
            api_client=judge_config.api_client,
            model=judge_config.model,
            goal=goal,
            gold_trajectory=gold_trajectory,
            gold_answer=gold_answer,
        )
        if checklist:
            rubrics[case.case_id] = checklist
    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(rubrics, ensure_ascii=False, indent=2) + "\n")
    return OhmoRubricDeriveResult(
        path=out, case_count=len(cases), derived_count=len(rubrics)
    )


def _ohmo_todo_write_tool_factory(state_root: Path) -> Sequence[BaseTool]:
    return (OhmoTodoWriteTool(TodoStore(state_root), lambda: "eval-sandbox"),)


class _MockSendTelegramMessageTool(BaseTool):
    """Eval-only mock of ``send_telegram_message`` — returns a deterministic
    success WITHOUT sending anything.

    The real tool fail-closes on a non-human sender (e.g. ``__scheduler__``
    background turns), but gold trajectories for scheduler tasks were captured
    from a successful human-origin send; replaying the refusal makes those tasks
    unsatisfiable in eval (the agent can never complete the required send). The
    send tool's real ACL / signing / contact-resolution behaviour is unit-tested
    separately; in eval we only need the send step to proceed so the judge can
    score the rest of the task. No real Telegram side effect ever happens in
    eval — this mock is the faithful stand-in for "the message was sent".
    """

    name = "send_telegram_message"
    description = SendTelegramMessageTool.description
    input_model = SendTelegramMessageInput

    def is_read_only(self, arguments: SendTelegramMessageInput) -> bool:
        del arguments
        return False

    async def execute(
        self, arguments: SendTelegramMessageInput, context: ToolExecutionContext
    ) -> ToolResult:
        del context
        return ToolResult(
            output=f"Message queued for delivery to {arguments.recipient}.",
            metadata={"mock": True, "tool": "send_telegram_message"},
        )


def _make_live_local_tool_factory(
    workspace: Path | None,
    *,
    include_skill: bool = True,
) -> Callable[[Path], Sequence[BaseTool]]:
    """Inject real, deterministic, local tools over the replay registry.

    Mirrors the production tool registry faithfully: ``todo_write`` writes to an
    isolated store, and ``skill`` reads local SKILL.md files (read-only). Both
    are local and deterministic, so they are safe inside frozen replay and make
    the eval harness match prod's skill-invocation mechanism — prod's
    ``create_default_tool_registry`` always exposes a live ``SkillTool``, while
    the replay registry only carried ``skill`` if the gold episode happened to
    call it. Network / non-deterministic tools (web_search/web_fetch/MCP)
    intentionally stay replay fixtures.

    The injected ``SkillTool`` is wrapped so the eval's per-workspace skill /
    plugin directories reach ``SkillTool.execute`` via ``context.metadata`` —
    the same dirs that seed the "# Available Skills" catalog in the prompt.
    """
    if workspace is not None:
        skill_dirs: tuple[str, ...] = (str(get_skills_dir(workspace)),)
        plugin_roots: tuple[str, ...] = (str(get_plugins_dir(workspace)),)
    else:
        skill_dirs = ()
        plugin_roots = ()

    def factory(state_root: Path) -> Sequence[BaseTool]:
        tools: list[BaseTool] = [*_ohmo_todo_write_tool_factory(state_root)]
        if include_skill:
            # Live skill reads the workspace's SKILL.md files. That output is
            # workspace-dependent, so a portable/committed cache must instead
            # replay skill from the captured fixtures (include_skill=False) —
            # otherwise a CI host without the skills dir gets "Skill not found"
            # and the cache misses.
            tools.append(
                _ToolContextMetadataWrapper(
                    SkillTool(),
                    metadata={
                        "extra_skill_dirs": skill_dirs,
                        "extra_plugin_roots": plugin_roots,
                    },
                )
            )
        tools.append(_MockSendTelegramMessageTool())
        return tuple(tools)

    return factory


def _ohmo_sandbox_tool_factory(sandbox_ws: Path) -> Sequence[BaseTool]:
    default_tz, reminder_max_per_chat = _ohmo_reminder_defaults()
    memory_store = MemoryStore(sandbox_ws)
    todo_store = TodoStore(sandbox_ws)
    reminder_store = ReminderStore(workspace=sandbox_ws)
    reminder_lock = asyncio.Lock()
    reminder_metadata = {
        "ohmo_reminder_ctx": {
            "channel": "eval",
            "chat_id": "eval-sandbox",
            "session_key": "eval-sandbox",
            "sender_id": "eval-sandbox",
            "chat_type": "private",
            "is_group": False,
            "tz": default_tz,
        }
    }
    return (
        OhmoMemoryTool(memory_store),
        OhmoTodoWriteTool(todo_store, lambda: "eval-sandbox"),
        _ToolContextMetadataWrapper(
            RemindCreateTool(
                reminder_store,
                reminder_lock,
                default_tz=default_tz,
                max_per_chat=reminder_max_per_chat,
            ),
            metadata=reminder_metadata,
        ),
        _ToolContextMetadataWrapper(
            RemindListTool(
                reminder_store,
                reminder_lock,
                default_tz=default_tz,
            ),
            metadata=reminder_metadata,
        ),
        _ToolContextMetadataWrapper(
            RemindCancelTool(reminder_store, reminder_lock),
            metadata=reminder_metadata,
        ),
    )


def _ohmo_reminder_defaults() -> tuple[str, int]:
    # Avoid importing the gateway runtime during ohmo.evals module initialization.
    from ohmo.gateway.runtime import (  # noqa: PLC0415
        DEFAULT_REMINDER_MAX_PER_CHAT,
        DEFAULT_REMINDER_TZ,
    )

    return DEFAULT_REMINDER_TZ, DEFAULT_REMINDER_MAX_PER_CHAT


def _ohmo_sandbox_state(sandbox_ws: Path) -> dict[str, Any]:
    return extract_state_keys(
        build_ohmo_resource_snapshot(episode_id="sandbox", workspace=sandbox_ws)
    )


class _ToolContextMetadataWrapper(BaseTool):
    """Inject fixed runtime metadata for tools that require channel context."""

    def __init__(self, tool: BaseTool, *, metadata: dict[str, object]) -> None:
        self._tool = tool
        self._metadata = metadata
        self.name = tool.name
        self.description = tool.description
        self.input_model = tool.input_model

    async def execute(self, arguments, context: ToolExecutionContext) -> ToolResult:
        return await self._tool.execute(
            arguments,
            ToolExecutionContext(
                cwd=context.cwd,
                metadata={**context.metadata, **self._metadata},
                hook_executor=context.hook_executor,
            ),
        )

    def is_read_only(self, arguments) -> bool:
        return self._tool.is_read_only(arguments)


def _run_session_report_case_sampled(
    *,
    store,
    group,
    runner: SessionReplayRunner | FaithfulSessionRunner,
    samples: int,
    gold_capabilities_by_session: Mapping[str, Sequence[str]] | None,
    user_simulator_factory: Callable[[], UserSimulator] | None,
    clarification_allowed_by_session: Mapping[str, bool] | None,
    judge_votes: int,
    grounding_votes: int,
) -> EvalSessionReportCase:
    first = _run_session_report_case(
        store=store,
        group=group,
        runner=runner,
        gold_capabilities_by_session=gold_capabilities_by_session,
        user_simulator_factory=user_simulator_factory,
        clarification_allowed_by_session=clarification_allowed_by_session,
        judge_votes=judge_votes,
        grounding_votes=grounding_votes,
    )
    if samples == 1:
        return first

    sample_cases = [first]
    for _ in range(samples - 1):
        sample_cases.append(
            _run_session_report_case(
                store=store,
                group=group,
                runner=runner,
                gold_capabilities_by_session=gold_capabilities_by_session,
                user_simulator_factory=user_simulator_factory,
                clarification_allowed_by_session=clarification_allowed_by_session,
                judge_votes=judge_votes,
                grounding_votes=grounding_votes,
            )
        )
    pass_count = sum(1 for case in sample_cases if case.status == "passed")
    check_keys: set[str] = set()
    for case in sample_cases:
        check_keys.update((case.checks or {}).keys())
    check_rates = {
        key: sum(1 for case in sample_cases if (case.checks or {}).get(key))
        / samples
        for key in sorted(check_keys)
    }
    return first.model_copy(
        update={
            "status": "passed" if pass_count * 2 > samples else "failed",
            "score": sum(case.score for case in sample_cases) / samples,
            "metadata": {
                **first.metadata,
                "sample_count": samples,
                "pass_count": pass_count,
                "pass_rate": pass_count / samples,
                "flaky": 0 < pass_count < samples,
                "check_rates": check_rates,
            },
        }
    )


def _run_session_report_case(
    *,
    store,
    group,
    runner: SessionReplayRunner | FaithfulSessionRunner,
    gold_capabilities_by_session: Mapping[str, Sequence[str]] | None,
    user_simulator_factory: Callable[[], UserSimulator] | None,
    clarification_allowed_by_session: Mapping[str, bool] | None,
    judge_votes: int,
    grounding_votes: int,
) -> EvalSessionReportCase:
    gold_source = (
        "provided"
        if gold_capabilities_by_session
        and group.session_id in gold_capabilities_by_session
        else "captured_self_coverage"
    )
    gold_capabilities = gold_capabilities_for_session(
        store,
        group,
        overrides=gold_capabilities_by_session,
    )
    state_delta = compute_episode_state_delta(store, group.episode_ids[-1])
    state_changed = (
        None if state_delta is None else bool(state_delta.get("changed") is True)
    )
    user_simulator = (
        user_simulator_factory() if user_simulator_factory is not None else None
    )
    result = (
        runner.run(
            group=group,
            store=store,
            user_simulator=user_simulator,
        )
        if not isinstance(runner, FaithfulSessionRunner)
        else _run_eval_coroutine(
            runner.run_session(
                group=group,
                store=store,
                user_simulator=user_simulator,
            )
        )
    )
    captured_prompts_list: list[str] = []
    for episode_id in group.episode_ids:
        episode = store.get_episode(episode_id)
        if episode is None:
            captured_prompts_list.append("")
        else:
            captured_prompts_list.append(episode.user_goal or episode.user_text)
    captured_prompts = tuple(captured_prompts_list)
    clarification_allowed = bool(
        clarification_allowed_by_session
        and clarification_allowed_by_session.get(group.session_id, False)
    )
    score_payload: dict
    if isinstance(runner, FaithfulSessionRunner):
        transcript = result.metadata.get("transcript", ())
        score_payload = _run_eval_coroutine(
            score_faithful_session(
                runner._api_client,
                runner._model,
                captured_prompts=captured_prompts,
                transcript=transcript,
                final_text=result.final_text,
                search=_default_grounding_search,
                judge_votes=judge_votes,
                grounding_votes=grounding_votes,
            )
        )
    else:
        score_payload = score_session(
            result,
            gold_capabilities=gold_capabilities,
            state_changed=state_changed,
            clarification_allowed=clarification_allowed,
        )
    metadata = {
        "episode_count": len(group.episode_ids),
        "gold_source": gold_source,
        "gold_capability_count": len(gold_capabilities),
        "observed_capabilities": score_payload["observed_capabilities"],
        "missing_capabilities": score_payload["missing_capabilities"],
        "state_delta_captured": state_delta is not None,
        "state_changed": state_changed,
        "final_text_length": len(result.final_text),
        "fixture_count": result.metadata.get("fixture_count", 0),
        "fixture_match": result.metadata.get("fixture_match", "order"),
        "source_event_count": result.metadata.get("source_event_count", 0),
        "engine_message_count": result.metadata.get("engine_message_count", 0),
        "user_turn_sources": result.metadata.get("user_turn_sources", []),
        "replay_hit_count": result.metadata.get("replay_hit_count", 0),
        "llm_fallback_count": result.metadata.get("llm_fallback_count", 0),
        "replay_hit_rate": result.metadata.get("replay_hit_rate", 0.0),
        "ended_reason": result.metadata.get("ended_reason", ""),
        "clarification_allowed": clarification_allowed,
        "terminal_clarification": bool(
            score_payload.get("terminal_clarification", False)
        ),
    }
    if "state_delta" in result.metadata:
        metadata["state_delta"] = result.metadata["state_delta"]
    if isinstance(runner, FaithfulSessionRunner):
        grounding_task = score_payload.get("grounding_task")
        if grounding_task is None:
            grounding_task = captured_prompts[-1] if captured_prompts else ""
        metadata.update(
            constraint_report_metadata(
                constraints=score_payload.get("constraints", ()),
                intent_evidence=score_payload.get("intent_evidence", ""),
            )
        )
        metadata.update(
            grounding_report_metadata(
                score_payload.get("grounding"),
                task=grounding_task,
                answer=result.final_text,
            )
        )
    warnings = list(score_payload.get("warnings", []))
    if warnings:
        metadata["warnings"] = warnings
    return EvalSessionReportCase(
        session_id=group.session_id,
        status="passed" if score_payload["passed"] else "failed",
        score=float(score_payload["score"]),
        checks=dict(score_payload["checks"]),
        turn_count=result.turn_count,
        metadata=metadata,
    )
