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
    run_execution_report,
)

from ohmo.evals.adapter import get_eval_store


@dataclass(frozen=True)
class OhmoEvalRunResult:
    """Summary returned after running an Ohmo eval report."""

    write: EvalExecutionReportWrite
    report_only: bool


_SUPPORTED_EXECUTORS = {
    "replay-tools": ReplayToolsExecutor,
    "replay_tools": ReplayToolsExecutor,
}
_SUPPORTED_AGENT_RUNNERS = {"scripted", "query-engine"}
_DEFAULT_QUERY_ENGINE_SYSTEM_PROMPT = "You are running an Ohmo replay-only eval."


def run_ohmo_eval_report(
    *,
    workspace: str | Path | None = None,
    pack_filename: str = "eval_pack.json",
    limit: int | None = None,
    report_only: bool = False,
    executor_name: str = "replay-tools",
    agent_runner_name: str = "scripted",
    model: str | None = None,
    provider_profile: str | None = None,
    system_prompt: str = _DEFAULT_QUERY_ENGINE_SYSTEM_PROMPT,
) -> OhmoEvalRunResult:
    """Run deterministic replay-tools execution checks over an Ohmo eval pack."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    workspace_root = Path(workspace).expanduser().resolve() if workspace else None
    agent_runner = _build_agent_runner(
        agent_runner_name,
        workspace=workspace_root,
        model=model,
        provider_profile=provider_profile,
        system_prompt=system_prompt,
    )
    executor = _build_executor(executor_name, agent_runner=agent_runner)
    store = get_eval_store(workspace)
    pack = read_run_pack(store, pack_filename=pack_filename)
    write = run_execution_report(store, pack=pack, limit=limit, executor=executor)
    return OhmoEvalRunResult(write=write, report_only=report_only)


def _build_executor(
    executor_name: str,
    *,
    agent_runner: ReplayScriptAgentRunner | QueryEngineEvalAgentRunner,
) -> ReplayToolsExecutor:
    normalized = executor_name.strip().lower()
    executor_factory = _SUPPORTED_EXECUTORS.get(normalized)
    if executor_factory is None:
        supported = ", ".join(sorted({"replay-tools"}))
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
    system_prompt: str,
) -> ReplayScriptAgentRunner | QueryEngineEvalAgentRunner:
    normalized = agent_runner_name.strip().lower()
    if normalized not in _SUPPORTED_AGENT_RUNNERS:
        supported = ", ".join(sorted(_SUPPORTED_AGENT_RUNNERS))
        raise ValueError(
            f"unknown eval agent runner: {agent_runner_name}. Supported runners: {supported}"
        )
    if normalized == "scripted":
        return ReplayScriptAgentRunner()

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
    return QueryEngineEvalAgentRunner(
        api_client=api_client,
        model=settings.model,
        system_prompt=system_prompt,
        cwd=workspace,
    )
