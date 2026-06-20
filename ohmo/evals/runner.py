"""Ohmo helpers for executor-based eval reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openharness.api.resolver import (
    ApiClientResolutionError,
    resolve_api_client_from_settings,
)
from openharness.config import load_settings
from openharness.evals import (
    EvalExecutionReportWrite,
    QueryEngineEvalAgentRunner,
    ReplayScriptAgentRunner,
    ReplayToolsExecutor,
    read_run_pack,
    resolve_execution_scorer,
    run_execution_report,
)

from ohmo.evals.adapter import get_eval_store
from ohmo.prompts import build_ohmo_system_prompt


@dataclass(frozen=True)
class OhmoEvalRunResult:
    """Summary returned after running an Ohmo eval report."""

    write: EvalExecutionReportWrite
    report_only: bool


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
    agent_runner: ReplayScriptAgentRunner | QueryEngineEvalAgentRunner
    agent_runner_name: str
    model: str
    provider_profile: str


SUPPORTED_EVAL_EXECUTOR_NAMES = ("replay-tools",)
SUPPORTED_EVAL_AGENT_RUNNER_NAMES = ("scripted", "query-engine")
_SUPPORTED_EXECUTORS = {
    "replay-tools": ReplayToolsExecutor,
    "replay_tools": ReplayToolsExecutor,
}
_SUPPORTED_AGENT_RUNNERS = set(SUPPORTED_EVAL_AGENT_RUNNER_NAMES)


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
) -> OhmoEvalRunResult:
    """Run deterministic replay-tools execution checks over an Ohmo eval pack."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected_scorer = resolve_execution_scorer(scorer) if scorer else None
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
    )
    store = get_eval_store(workspace)
    pack = read_run_pack(store, pack_filename=pack_filename)
    write = run_execution_report(
        store,
        pack=pack,
        report_filename=report_filename,
        limit=limit,
        executor=executor,
        scorer=selected_scorer,
    )
    return OhmoEvalRunResult(write=write, report_only=report_only)


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
) -> OhmoEvalRunConfigCheckResult:
    """Validate an eval run configuration without executing eval cases."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
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
        replay_tools_only=True,
    )


def _build_executor(
    executor_name: str,
    *,
    agent_runner: ReplayScriptAgentRunner | QueryEngineEvalAgentRunner,
) -> ReplayToolsExecutor:
    normalized = executor_name.strip().lower()
    executor_factory = _SUPPORTED_EXECUTORS.get(normalized)
    if executor_factory is None:
        supported = ", ".join(SUPPORTED_EVAL_EXECUTOR_NAMES)
        raise ValueError(
            f"unknown eval executor: {executor_name}. Supported executors: {supported}"
        )
    return executor_factory(agent_runner=agent_runner)


def _build_agent_runner(
    agent_runner_name: str,
    *,
    workspace: Path | None,
    model: str | None,
    provider_profile: str | None,
    system_prompt: str | None,
) -> ReplayScriptAgentRunner | QueryEngineEvalAgentRunner:
    return _build_agent_runner_config(
        agent_runner_name,
        workspace=workspace,
        model=model,
        provider_profile=provider_profile,
        system_prompt=system_prompt,
    ).agent_runner


def _build_agent_runner_config(
    agent_runner_name: str,
    *,
    workspace: Path | None,
    model: str | None,
    provider_profile: str | None,
    system_prompt: str | None,
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
            "query-engine eval runner requires configured API authentication"
        ) from exc
    resolved_prompt = system_prompt or build_ohmo_system_prompt(
        workspace or Path.cwd(),
        workspace=workspace,
    )
    return _AgentRunnerConfig(
        agent_runner=QueryEngineEvalAgentRunner(
            api_client=api_client,
            model=settings.model,
            system_prompt=resolved_prompt,
            cwd=workspace,
        ),
        agent_runner_name="query-engine",
        model=settings.model,
        provider_profile=settings.active_profile,
    )
