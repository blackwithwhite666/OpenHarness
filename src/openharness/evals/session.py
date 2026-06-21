"""Session-level replay eval primitives."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openharness.api.client import SupportsStreamingMessages
from openharness.config.settings import PermissionSettings
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import (
    AssistantTurnComplete,
    ErrorEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.evals.execution import _INCIDENTAL_CAPABILITIES, _tool_fixtures
from openharness.evals.executor import (
    EvalToolFixture,
    _run_eval_coroutine,
    build_replay_tool_registry,
)
from openharness.evals.models import EvalEpisode
from openharness.evals.session_user_simulator import (
    ReplayUserSimulator,
    UserSimulator,
    _resolve_user_turn,
    _simulator_ended_reason,
)
from openharness.evals.store import EvalStore
from openharness.evals.tool_labels import effective_tool_label
from openharness.permissions.checker import PermissionChecker
from openharness.permissions.modes import PermissionMode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalSessionGroup:
    """Ordered episode ids sharing a session identity."""

    session_id: str
    episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvalSessionTurnResult:
    """Transient result for one replayed session turn."""

    episode_id: str
    tool_path: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    tool_error: bool = False
    final_text: str = ""


@dataclass(frozen=True)
class EvalSessionRunResult:
    """Transient result for one replayed eval session."""

    session_id: str
    turns: tuple[EvalSessionTurnResult, ...] = ()
    union_capabilities: tuple[str, ...] = ()
    final_text: str = ""
    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def group_episodes_into_sessions(
    store: EvalStore,
    *,
    app: str | None = None,
    source: str | None = None,
) -> list[EvalSessionGroup]:
    """Group episodes by session id while preserving deterministic capture order."""
    indexed: list[tuple[int, EvalEpisode]] = []
    for index, episode_id in enumerate(store.list_episode_ids()):
        episode = store.get_episode(episode_id)
        if episode is None:
            continue
        if app is not None and episode.app != app:
            continue
        if source is not None and episode.source != source:
            continue
        indexed.append((index, episode))

    groups: dict[str, list[tuple[int, EvalEpisode]]] = {}
    first_index: dict[str, int] = {}
    for index, episode in indexed:
        session_id = episode.session_id or episode.episode_id
        groups.setdefault(session_id, []).append((index, episode))
        first_index.setdefault(session_id, index)

    out: list[EvalSessionGroup] = []
    for session_id, rows in groups.items():
        ordered = sorted(rows, key=lambda item: (item[1].created_at, item[0]))
        out.append(
            EvalSessionGroup(
                session_id=session_id,
                episode_ids=tuple(episode.episode_id for _, episode in ordered),
            )
        )
    return sorted(out, key=lambda group: (first_index[group.session_id], group.session_id))


class SessionReplayRunner:
    """Run an ordered episode group through one persistent QueryEngine."""

    name = "session-query-engine"

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        model: str,
        system_prompt: str,
        cwd: str | Path | None = None,
        max_turns: int = 8,
        max_tokens: int = 4096,
        max_session_turns: int | None = None,
    ) -> None:
        self._api_client = api_client
        self._model = model
        self._system_prompt = system_prompt
        self._cwd = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._max_session_turns = max_session_turns

    def run(
        self,
        *,
        group: EvalSessionGroup,
        store: EvalStore,
        user_simulator: UserSimulator | None = None,
    ) -> EvalSessionRunResult:
        """Replay one captured session with order-based replay fixtures."""
        return _run_eval_coroutine(
            self._run(group=group, store=store, user_simulator=user_simulator)
        )

    async def _run(
        self,
        *,
        group: EvalSessionGroup,
        store: EvalStore,
        user_simulator: UserSimulator | None,
    ) -> EvalSessionRunResult:
        if not group.episode_ids:
            raise ValueError("session group must contain episodes")

        episodes: list[EvalEpisode] = []
        fixtures: list[EvalToolFixture] = []
        captured_capabilities: list[tuple[str, ...]] = []
        source_event_count = 0
        for episode_id in group.episode_ids:
            episode = store.get_episode(episode_id)
            if episode is None:
                raise ValueError(f"episode not found: {episode_id}")
            events = list(store.iter_events(episode_id))
            source_event_count += len(events)
            episode_fixtures = _tool_fixtures(events)
            episodes.append(episode)
            fixtures.extend(episode_fixtures)
            captured_capabilities.append(
                tuple(
                    effective_tool_label(
                        fixture.tool_name,
                        _fixture_arguments(events, fixture),
                    )
                    for fixture in episode_fixtures
                )
            )
        captured_prompts = tuple(_episode_prompt(episode) for episode in episodes)

        engine = QueryEngine(
            api_client=self._api_client,
            tool_registry=build_replay_tool_registry(tuple(fixtures)),
            permission_checker=PermissionChecker(
                PermissionSettings(mode=PermissionMode.FULL_AUTO)
            ),
            cwd=self._cwd,
            model=self._model,
            system_prompt=self._system_prompt,
            max_turns=self._max_turns,
            max_tokens=self._max_tokens,
        )

        turns: list[EvalSessionTurnResult] = []
        user_turn_sources: list[str] = []
        transcript: list[tuple[str, str]] = []
        simulator = user_simulator or ReplayUserSimulator(captured_prompts)
        max_session_turns = self._max_session_turns or _default_max_session_turns(
            len(captured_prompts)
        )
        ended_reason = "model_done"
        last_turn: EvalSessionTurnResult | None = None
        while len(turns) < max_session_turns:
            user_turn = await _resolve_user_turn(
                simulator.next_turn(
                    transcript=tuple(transcript),
                    captured_prompts=captured_prompts,
                    captured_capabilities=tuple(captured_capabilities),
                    index=len(turns),
                    last_turn=last_turn,
                )
            )
            if user_turn is None:
                ended_reason = _simulator_ended_reason(
                    simulator,
                    default=(
                        "captured_exhausted"
                        if len(turns) >= len(captured_prompts)
                        else "model_done"
                    ),
                )
                break

            episode_id = (
                episodes[len(turns)].episode_id
                if user_turn.source == "replay" and len(turns) < len(episodes)
                else ""
            )
            transcript.append(("user", user_turn.text))
            turn = await consume_engine_turn(
                engine,
                user_turn.text,
                episode_id=episode_id,
            )
            turns.append(turn)
            user_turn_sources.append(user_turn.source)
            transcript.append(("assistant", turn.final_text))
            last_turn = turn
        else:
            ended_reason = "budget"
            log.info(
                "session replay stopped by budget session_id=%s max_session_turns=%d",
                group.session_id,
                max_session_turns,
            )

        union_capabilities = sorted(
            {capability for turn in turns for capability in turn.capabilities}
        )
        final_text = turns[-1].final_text if turns else ""
        replay_hit_count = sum(1 for source in user_turn_sources if source == "replay")
        llm_fallback_count = sum(
            1 for source in user_turn_sources if source == "llm_fallback"
        )
        replay_hit_rate = (
            replay_hit_count / len(user_turn_sources) if user_turn_sources else 0.0
        )
        return EvalSessionRunResult(
            session_id=group.session_id,
            turns=tuple(turns),
            union_capabilities=tuple(union_capabilities),
            final_text=final_text,
            turn_count=len(turns),
            metadata={
                "agent_runner": self.name,
                "episode_count": len(episodes),
                "fixture_count": len(fixtures),
                "source_event_count": source_event_count,
                "engine_message_count": len(engine.messages),
                "user_turn_sources": user_turn_sources,
                "replay_hit_count": replay_hit_count,
                "llm_fallback_count": llm_fallback_count,
                "replay_hit_rate": replay_hit_rate,
                "ended_reason": ended_reason,
            },
        )


async def consume_engine_turn(
    engine: QueryEngine,
    prompt: str,
    *,
    episode_id: str = "",
) -> EvalSessionTurnResult:
    """Consume one QueryEngine turn into the compact session result shape."""
    tool_path: list[str] = []
    capabilities: list[str] = []
    tool_error = False
    final_text = ""
    async for event in engine.submit_message(prompt):
        if isinstance(event, ToolExecutionStarted):
            arguments = dict(event.tool_input or {})
            tool_path.append(event.tool_name)
            capabilities.append(effective_tool_label(event.tool_name, arguments))
        elif isinstance(event, ToolExecutionCompleted):
            tool_error = tool_error or event.is_error
        elif isinstance(event, AssistantTurnComplete):
            final_text = event.message.text
        elif isinstance(event, ErrorEvent):
            tool_error = True
    return EvalSessionTurnResult(
        episode_id=episode_id,
        tool_path=tuple(tool_path),
        capabilities=tuple(capabilities),
        tool_error=tool_error,
        final_text=final_text,
    )


def score_session(
    result: EvalSessionRunResult,
    *,
    gold_capabilities: Sequence[str],
    state_changed: bool | None,
    clarification_allowed: bool = False,
) -> dict[str, Any]:
    """Score one session with capability coverage plus a final-outcome signal."""
    core_gold = {
        capability
        for capability in gold_capabilities
        if capability not in _INCIDENTAL_CAPABILITIES
    }
    observed = set(result.union_capabilities)
    missing = sorted(core_gold - observed)
    terminal_clarification = (
        bool(result.turns)
        and not result.turns[-1].tool_path
        and bool(result.turns[-1].final_text.strip())
    )
    final_outcome_reached = (
        state_changed is True
        if state_changed is not None
        else bool(result.final_text.strip())
    )
    warnings: list[str] = []
    capability_coverage = not missing
    if clarification_allowed and terminal_clarification:
        final_outcome_reached = True
        if missing:
            capability_coverage = True
            warnings.append("capability_coverage_terminal_clarification")
    checks = {
        "capability_coverage": capability_coverage,
        "no_turn_tool_errors": not any(turn.tool_error for turn in result.turns),
        "final_outcome_reached": final_outcome_reached,
    }
    passed = all(checks.values())
    payload: dict[str, Any] = {
        "passed": passed,
        "score": sum(1 for value in checks.values() if value) / len(checks),
        "checks": checks,
        "missing_capabilities": missing,
        "observed_capabilities": sorted(observed),
        "turn_count": result.turn_count,
    }
    if clarification_allowed:
        payload.update(
            {
                "clarification_allowed": True,
                "terminal_clarification": terminal_clarification,
                "warnings": warnings,
            }
        )
    return payload


def gold_capabilities_for_session(
    store: EvalStore,
    group: EvalSessionGroup,
    *,
    overrides: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    """Return supplied session golds or the P0 captured self-coverage golds."""
    if overrides and group.session_id in overrides:
        return tuple(overrides[group.session_id])

    capabilities: list[str] = []
    seen: set[str] = set()
    for episode_id in group.episode_ids:
        events = list(store.iter_events(episode_id))
        for fixture in _tool_fixtures(events):
            arguments = _fixture_arguments(events, fixture)
            label = effective_tool_label(fixture.tool_name, arguments)
            if label not in seen:
                seen.add(label)
                capabilities.append(label)
    return tuple(capabilities)


def _episode_prompt(episode: EvalEpisode) -> str:
    return episode.user_goal or episode.user_text


def _default_max_session_turns(captured_prompt_count: int) -> int:
    return max(captured_prompt_count * 2, captured_prompt_count + 4, 1)


def _fixture_arguments(events: Sequence[Any], fixture: EvalToolFixture) -> dict[str, Any]:
    idx = fixture.start_event_index
    if idx is None or idx < 0 or idx >= len(events):
        return {}
    payload = getattr(events[idx], "payload", None) or {}
    arguments = payload.get("input")
    return dict(arguments) if isinstance(arguments, dict) else {}
