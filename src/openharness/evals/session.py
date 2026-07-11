"""Session-level replay eval primitives."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
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
    EvalExecutorResult,
    SynthContext,
    _run_eval_coroutine,
    build_replay_tool_registry,
)
from openharness.evals.models import EvalEpisode
from openharness.evals.session_user_simulator import (
    ReplayUserSimulator,
    derive_ironuser_spec,
    UserSimulator,
    _resolve_user_turn,
    _simulator_ended_reason,
)
from openharness.evals.judge import _verify_grounding_voted, judge_intent_met
from openharness.evals.store import EvalStore
from openharness.evals.tool_labels import effective_tool_label
from openharness.evals.state import compute_state_delta
from openharness.permissions.checker import PermissionChecker
from openharness.permissions.modes import PermissionMode
from openharness.tools.base import ToolRegistry

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalSessionGroup:
    """Ordered episode ids sharing a session identity."""

    session_id: str
    episode_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvalConversation:
    """Bounded contiguous task conversation inside a coarse session thread."""

    session_id: str
    segment_index: int
    episode_ids: tuple[str, ...]
    n_turns: int
    span_minutes: float


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


async def score_faithful_session(
    api_client: SupportsStreamingMessages,
    model: str,
    *,
    captured_prompts: Sequence[str],
    transcript: Sequence[tuple[str, str]],
    final_text: str,
    search: Callable[[str], Awaitable[str]],
    judge_votes: int = 1,
    grounding_votes: int = 1,
    max_tokens: int = 600,
) -> dict[str, Any]:
    """Score a faithful session by outcome and grounded final answer.

    The scorer uses intent extraction from captured prompts to judge whether the
    final request intent was met, constraints were respected, and whether the
    final answer is sufficiently grounded.
    """
    spec = await derive_ironuser_spec(
        api_client,
        model,
        captured_prompts=captured_prompts,
        max_tokens=max_tokens,
    )
    transcript_tuple = tuple(
        (str(role), str(text)) for role, text in transcript if role is not None
    )
    intent = await judge_intent_met(
        api_client,
        model,
        intent=spec.intent,
        constraints=spec.constraints,
        transcript=transcript_tuple,
        votes=judge_votes,
        max_tokens=max_tokens,
    )
    intent_met = bool(intent.get("intent_met"))
    constraints_held = bool(intent.get("constraints_held"))

    checklist_items = tuple(
        item.strip() for item in (spec.intent, *spec.constraints) if item.strip()
    )
    grounding = await _verify_grounding_voted(
        api_client,
        model,
        votes=grounding_votes,
        task=spec.intent,
        answer=final_text,
        trajectory=final_text,
        checklist_items=checklist_items,
        search=search,
    )
    grounding_score = grounding.get("score")
    grounding_ok = grounding_score is None or grounding_score >= 0.6

    checks = {
        "intent_met": intent_met,
        "constraints_held": constraints_held,
        "grounding_ok": grounding_ok,
    }
    return {
        "passed": all(checks.values()),
        "score": sum(1 for value in checks.values() if value) / len(checks),
        "checks": checks,
        "missing_capabilities": [],
        "observed_capabilities": [],
        "turn_count": len(transcript_tuple) // 2,
        "intent_evidence": str(intent.get("evidence") or ""),
        "grounding": grounding,
    }


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


def segment_sessions_into_conversations(
    store: EvalStore,
    *,
    app: str | None = None,
    source: str | None = None,
    gap_minutes: float = 30.0,
    min_turns: int = 1,
) -> list[EvalConversation]:
    """Split coarse session threads into bounded time-gap conversations."""
    conversations: list[EvalConversation] = []
    for group in group_episodes_into_sessions(store, app=app, source=source):
        episodes: list[EvalEpisode] = []
        for episode_id in group.episode_ids:
            episode = store.get_episode(episode_id)
            if episode is not None:
                episodes.append(episode)

        for segment_index, segment in enumerate(
            _time_gap_conversation_segments(episodes, gap_minutes=gap_minutes)
        ):
            if len(segment) < min_turns:
                continue
            conversations.append(
                EvalConversation(
                    session_id=group.session_id,
                    segment_index=segment_index,
                    episode_ids=tuple(episode.episode_id for episode in segment),
                    n_turns=len(segment),
                    span_minutes=_conversation_span_minutes(segment),
                )
            )
    return conversations


def _time_gap_conversation_segments(
    episodes: Sequence[EvalEpisode],
    *,
    gap_minutes: float,
) -> list[tuple[EvalEpisode, ...]]:
    """Split ordered episodes on capture-time gaps.

    NOTE: This helper is the seam for a future topic-aware LLM refinement that
    can use ``_segment_conversation`` / ``HistoryContext`` from ``execution.py``.
    The current implementation is intentionally deterministic and time-gap only.
    """
    if not episodes:
        return []

    segments: list[tuple[EvalEpisode, ...]] = []
    current: list[EvalEpisode] = [episodes[0]]
    previous = episodes[0]
    for episode in episodes[1:]:
        if _conversation_gap_exceeds(previous, episode, gap_minutes=gap_minutes):
            segments.append(tuple(current))
            current = []
        current.append(episode)
        previous = episode
    segments.append(tuple(current))
    return segments


def _conversation_gap_exceeds(
    previous: EvalEpisode,
    current: EvalEpisode,
    *,
    gap_minutes: float,
) -> bool:
    previous_at = _parse_conversation_timestamp(previous.created_at)
    current_at = _parse_conversation_timestamp(current.created_at)
    if previous_at is None or current_at is None:
        return True
    return (current_at - previous_at).total_seconds() / 60.0 > gap_minutes


def _conversation_span_minutes(episodes: Sequence[EvalEpisode]) -> float:
    if len(episodes) < 2:
        return 0.0
    first_at = _parse_conversation_timestamp(episodes[0].created_at)
    last_at = _parse_conversation_timestamp(episodes[-1].created_at)
    if first_at is None or last_at is None:
        return 0.0
    return max(0.0, (last_at - first_at).total_seconds() / 60.0)


def _parse_conversation_timestamp(value: Any) -> datetime | None:
    try:
        if isinstance(value, datetime):
            timestamp = value
        elif isinstance(value, str):
            text = value.strip()
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            timestamp = datetime.fromisoformat(text)
        else:
            return None
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


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
        fixture_match_mode: str = "order",
        synth_context: SynthContext | None = None,
    ) -> None:
        if fixture_match_mode not in {
            "order",
            "arguments",
            "args_then_order",
            "synth",
            "synth_state",
        }:
            raise ValueError(
                "fixture match mode must be one of: order, arguments, "
                "args_then_order, synth, synth_state"
            )
        if fixture_match_mode in {"synth", "synth_state"} and synth_context is None:
            raise ValueError("synth fixture match requires a SynthContext")
        self._api_client = api_client
        self._model = model
        self._system_prompt = system_prompt
        self._cwd = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._max_session_turns = max_session_turns
        self._fixture_match_mode = fixture_match_mode
        self._synth_context = synth_context

    @property
    def fixture_match_mode(self) -> str:
        return self._fixture_match_mode

    def run(
        self,
        *,
        group: EvalSessionGroup,
        store: EvalStore,
        user_simulator: UserSimulator | None = None,
    ) -> EvalSessionRunResult:
        return _run_eval_coroutine(
            self.run_session(
                group=group,
                store=store,
                user_simulator=user_simulator,
            )
        )

    async def run_session(
        self,
        *,
        group: EvalSessionGroup,
        store: EvalStore,
        user_simulator: UserSimulator | None = None,
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
        fixture_records = tuple(fixtures)

        engine = QueryEngine(
            api_client=self._api_client,
            tool_registry=build_replay_tool_registry(
                fixture_records,
                match_mode=self._fixture_match_mode,
                synth_context=self._synth_context,
            ),
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
                prompt=user_turn.text,
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
                "fixture_match": self._fixture_match_mode,
                "episode_count": len(episodes),
                "fixture_count": len(fixtures),
                "source_event_count": source_event_count,
                "engine_message_count": len(engine.messages),
                "user_turn_sources": user_turn_sources,
                "replay_hit_count": replay_hit_count,
                "llm_fallback_count": llm_fallback_count,
                "replay_hit_rate": replay_hit_rate,
                "ended_reason": ended_reason,
                "transcript": tuple(transcript),
            },
        )


class FaithfulSessionRunner:
    """Run an ordered episode group through one persistent fs-sandbox runner."""

    name = "session-query-engine-faithful"

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
        fixture_match_mode: str = "order",
        synth_context: SynthContext | None = None,
        agent_runner_factory: Callable[[Path], object] | None = None,
        agent_runner: object | None = None,
        sandbox_tool_factory: Callable[[Path], object] | None = None,
        sandbox_state_fn: Callable[[Path], dict[str, Any]] | None = None,
    ) -> None:
        if fixture_match_mode not in {
            "order",
            "arguments",
            "args_then_order",
            "synth",
            "synth_state",
        }:
            raise ValueError(
                "fixture match mode must be one of: order, arguments, "
                "args_then_order, synth, synth_state"
            )
        if fixture_match_mode in {"synth", "synth_state"} and synth_context is None:
            raise ValueError("synth fixture match requires a SynthContext")
        if agent_runner_factory is None and agent_runner is None:
            raise ValueError(
                "faithful session runner requires an agent_runner or agent_runner_factory"
            )
        self._api_client = api_client
        self._model = model
        self._system_prompt = system_prompt
        self._cwd = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._max_session_turns = max_session_turns
        self._fixture_match_mode = fixture_match_mode
        self._synth_context = synth_context
        self._agent_runner_factory = agent_runner_factory
        self._agent_runner = agent_runner
        self._sandbox_tool_factory = sandbox_tool_factory
        self._sandbox_state_fn = sandbox_state_fn or (lambda _path: {})

    @property
    def fixture_match_mode(self) -> str:
        return self._fixture_match_mode

    def run(
        self,
        *,
        group: EvalSessionGroup,
        store: EvalStore,
        user_simulator: UserSimulator | None = None,
    ) -> EvalSessionRunResult:
        return _run_eval_coroutine(
            self.run_session(
                group=group,
                store=store,
                user_simulator=user_simulator,
            )
        )

    def _build_default_agent_runner(self, workspace: Path):
        if self._agent_runner_factory is not None:
            return self._agent_runner_factory(workspace)
        if self._agent_runner is None:
            raise ValueError("faithful session runner missing agent runner")
        runner = self._agent_runner
        if hasattr(runner, "workspaces") and isinstance(runner.workspaces, list):
            runner.workspaces[:] = [workspace]
        if hasattr(runner, "_cwd"):
            setattr(runner, "_cwd", workspace)
        if hasattr(runner, "workspace"):
            setattr(runner, "workspace", workspace)
        return runner

    async def run_session(
        self,
        *,
        group: EvalSessionGroup,
        store: EvalStore,
        user_simulator: UserSimulator | None = None,
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
        fixture_records = tuple(fixtures)
        session_workspace = Path(
            tempfile.mkdtemp(prefix="openharness-eval-faithful-session-")
        ).resolve()
        agent_runner = self._build_default_agent_runner(session_workspace)

        turns: list[EvalSessionTurnResult] = []
        user_turn_sources: list[str] = []
        transcript: list[tuple[str, str]] = []
        simulator = user_simulator or ReplayUserSimulator(captured_prompts)
        max_session_turns = self._max_session_turns or _default_max_session_turns(
            len(captured_prompts)
        )
        ended_reason = "model_done"
        last_turn: EvalSessionTurnResult | None = None
        context = SimpleNamespace(
            events=(),
            conversation_history=(),
            tool_fixtures=fixture_records,
        )
        before_state = self._sandbox_state_fn(session_workspace)
        try:
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
                executor_result = agent_runner.run(
                    prompt=_serialize_transcript_for_turn(
                        transcript=transcript,
                    ),
                    tool_registry=ToolRegistry(),
                    context=context,
                )
                turn = _executor_result_to_turn_result(
                    episode_id=episode_id,
                    result=executor_result,
                )
                turns.append(turn)
                user_turn_sources.append(user_turn.source)
                transcript.append(("assistant", turn.final_text))
                last_turn = turn
            else:
                ended_reason = "budget"
                log.info(
                    "session faithful stopped by budget session_id=%s max_session_turns=%d",
                    group.session_id,
                    max_session_turns,
                )
        finally:
            after_state = self._sandbox_state_fn(session_workspace)
            state_delta = compute_state_delta(before_state, after_state)
            shutil.rmtree(session_workspace, ignore_errors=True)

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
                "fixture_match": self._fixture_match_mode,
                "episode_count": len(episodes),
                "fixture_count": len(fixtures),
                "source_event_count": source_event_count,
                "user_turn_sources": user_turn_sources,
                "replay_hit_count": replay_hit_count,
                "llm_fallback_count": llm_fallback_count,
                "replay_hit_rate": replay_hit_rate,
                "ended_reason": ended_reason,
                "transcript": tuple(transcript),
                "state_delta": state_delta,
            },
        )


def _serialize_transcript_for_turn(
    transcript: Sequence[tuple[str, str]],
) -> str:
    return "\n".join(
        f"{role}: {text.strip()}"
        for role, text in transcript
        if text.strip()
    )


def _executor_result_to_turn_result(
    *,
    episode_id: str,
    result: EvalExecutorResult,
) -> EvalSessionTurnResult:
    tool_calls = result.tool_calls or ()
    if tool_calls:
        tool_path = tuple(call.tool_name for call in tool_calls)
        capabilities = tuple(
            effective_tool_label(call.tool_name, dict(call.arguments))
            for call in tool_calls
        )
        tool_error = any(call.is_error for call in tool_calls)
    else:
        tool_path = tuple(result.tool_path)
        capabilities = tuple(result.tool_path)
        tool_error = False
    return EvalSessionTurnResult(
        episode_id=episode_id,
        tool_path=tool_path,
        capabilities=capabilities,
        tool_error=tool_error,
        final_text=result.final_text,
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
