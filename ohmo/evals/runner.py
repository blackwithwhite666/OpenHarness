"""Ohmo helpers for executor-based eval reports."""

from __future__ import annotations

import asyncio
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
from openharness.evals import (
    EvalExecutionReportWrite,
    EvalSessionReport,
    EvalSessionReportCase,
    FsSandboxAgentRunner,
    HistoryContext,
    HybridUserSimulator,
    LiveReadAgentRunner,
    LlmUserSimulator,
    QueryEngineEvalAgentRunner,
    ReplayUserSimulator,
    ReplayScriptAgentRunner,
    ReplayToolsExecutor,
    SandboxMutatingAgentRunner,
    SessionReplayRunner,
    SynthContext,
    TrajectoryJudgeScorer,
    UserSimulator,
    gold_capabilities_for_session,
    group_episodes_into_sessions,
    read_run_pack,
    resolve_execution_scorer,
    run_execution_report,
    score_session,
)
from openharness.evals.runner import _report_output_path, _stable_id
from openharness.evals.state import compute_episode_state_delta, extract_state_keys
from openharness.prompts import build_runtime_system_prompt
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult
from openharness.tools.skill_tool import SkillTool
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
) -> OhmoEvalRunResult:
    """Run deterministic replay-tools execution checks over an Ohmo eval pack."""
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    fixture_match = _validate_fixture_match(fixture_match)
    workspace_root = Path(workspace).expanduser().resolve() if workspace else None
    selected_scorer = None
    judge_config: _AgentRunnerConfig | None = None
    synth_config: _AgentRunnerConfig | None = None
    history_config: _AgentRunnerConfig | None = None
    synth_context: SynthContext | None = None
    history_context: HistoryContext | None = None
    if scorer == TrajectoryJudgeScorer.name:
        judge_config = _build_agent_runner_config(
            "query-engine",
            workspace=workspace_root,
            model=judge_model or model,
            provider_profile=judge_profile or provider_profile,
            system_prompt=None,
        )
        if judge_config.api_client is None:
            raise ValueError("trajectory_judge_v1 requires configured API authentication")
        selected_scorer = TrajectoryJudgeScorer(
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
    ):
        agent_runner_kwargs = {
            "sandbox_net_mode": sandbox_net_mode,
            "sandbox_proxy_url": sandbox_proxy_url,
            "sandbox_browser_socket": sandbox_browser_socket,
            "sandbox_browser_name": sandbox_browser_name,
        }
    agent_runner_config = _build_agent_runner_config(
        agent_runner_name,
        workspace=workspace_root,
        model=model,
        provider_profile=provider_profile,
        system_prompt=system_prompt,
        max_turns=max_turns,
        **agent_runner_kwargs,
    )
    executor = _build_executor(
        executor_name,
        agent_runner=agent_runner_config.agent_runner,
        fixture_match=fixture_match,
        synth_context=synth_context,
    )
    store = get_eval_store(workspace)
    pack = read_run_pack(store, pack_filename=pack_filename)
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
    )
    if judge_config is not None:
        write.report.metadata["judge_model"] = judge_config.model
        write.report.metadata["judge_provider_profile"] = judge_config.provider_profile
    if synth_config is not None:
        write.report.metadata["synth_model"] = synth_config.model
        write.report.metadata["synth_provider_profile"] = synth_config.provider_profile
    if history_config is not None:
        write.report.metadata["history_model"] = history_config.model
        write.report.metadata["history_provider_profile"] = history_config.provider_profile
    if (
        judge_config is not None
        or synth_config is not None
        or history_config is not None
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
) -> OhmoSessionEvalRunResult:
    """Run P0 session replay checks over captured Ohmo eval episodes."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if samples < 1:
        raise ValueError("samples must be positive")
    if max_session_turns is not None and max_session_turns <= 0:
        raise ValueError("max_session_turns must be positive")
    fixture_match = _validate_fixture_match(fixture_match)
    if user_sim_model is not None and user_sim_profile is None:
        raise ValueError("user_sim_model requires user_sim_profile")

    workspace_root = Path(workspace).expanduser().resolve() if workspace else None
    agent_runner_config = _build_agent_runner_config(
        "query-engine",
        workspace=workspace_root,
        model=model,
        provider_profile=provider_profile,
        system_prompt=system_prompt,
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
                ),
            )

        user_simulator_factory = _new_user_simulator

    store = get_eval_store(workspace)
    groups = group_episodes_into_sessions(store, app="ohmo")
    groups = groups[:limit] if limit is not None else groups
    if not groups:
        raise ValueError("eval store must contain ohmo sessions")

    runner = SessionReplayRunner(
        api_client=agent_runner_config.api_client,
        model=agent_runner_config.model,
        system_prompt=agent_runner_config.system_prompt,
        cwd=agent_runner_config.cwd,
        fixture_match_mode=fixture_match,
        max_session_turns=max_session_turns,
        synth_context=synth_context,
    )
    cases = [
        _run_session_report_case_sampled(
            store=store,
            group=group,
            runner=runner,
            samples=samples,
            gold_capabilities_by_session=gold_capabilities_by_session,
            user_simulator_factory=user_simulator_factory,
            clarification_allowed_by_session=clarification_allowed_by_session,
        )
        for group in groups
    ]
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
            "runner_name": SessionReplayRunner.name,
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
        },
    )
    path = _report_output_path(store, report_filename)
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
    )


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
    try:
        api_client = resolve_api_client_from_settings(settings)
    except (ApiClientResolutionError, SystemExit) as exc:
        raise ValueError(
            f"{normalized} eval runner requires configured API authentication"
        ) from exc
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
                live_local_tool_factory=_make_live_local_tool_factory(workspace),
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
            live_local_tool_factory=_make_live_local_tool_factory(workspace),
        ),
        agent_runner_name="query-engine",
        model=settings.model,
        provider_profile=settings.active_profile,
        api_client=api_client,
        system_prompt=resolved_prompt,
        cwd=workspace,
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
        return (
            *_ohmo_todo_write_tool_factory(state_root),
            _ToolContextMetadataWrapper(
                SkillTool(),
                metadata={
                    "extra_skill_dirs": skill_dirs,
                    "extra_plugin_roots": plugin_roots,
                },
            ),
            _MockSendTelegramMessageTool(),
        )

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
    runner: SessionReplayRunner,
    samples: int,
    gold_capabilities_by_session: Mapping[str, Sequence[str]] | None,
    user_simulator_factory: Callable[[], UserSimulator] | None,
    clarification_allowed_by_session: Mapping[str, bool] | None,
) -> EvalSessionReportCase:
    first = _run_session_report_case(
        store=store,
        group=group,
        runner=runner,
        gold_capabilities_by_session=gold_capabilities_by_session,
        user_simulator_factory=user_simulator_factory,
        clarification_allowed_by_session=clarification_allowed_by_session,
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
            )
        )
    pass_count = sum(1 for case in sample_cases if case.status == "passed")
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
            },
        }
    )


def _run_session_report_case(
    *,
    store,
    group,
    runner: SessionReplayRunner,
    gold_capabilities_by_session: Mapping[str, Sequence[str]] | None,
    user_simulator_factory: Callable[[], UserSimulator] | None,
    clarification_allowed_by_session: Mapping[str, bool] | None,
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
    result = runner.run(
        group=group,
        store=store,
        user_simulator=(
            user_simulator_factory() if user_simulator_factory is not None else None
        ),
    )
    clarification_allowed = bool(
        clarification_allowed_by_session
        and clarification_allowed_by_session.get(group.session_id, False)
    )
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
