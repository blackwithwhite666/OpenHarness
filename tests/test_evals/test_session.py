from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

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
    EvalSessionGroup,
    EvalSessionRunResult,
    EvalSessionTurnResult,
    EvalStore,
    HybridUserSimulator,
    IronUserSpec,
    LlmUserSimulator,
    FaithfulSessionRunner,
    ReplayUserSimulator,
    SessionReplayRunner,
    UserTurn,
    derive_ironuser_spec,
    group_episodes_into_sessions,
    replay_matches,
    score_session,
)
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


def test_group_episodes_into_sessions_orders_shared_and_singleton(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _append_episode(
        store,
        episode_id="ep-1",
        session_id="session-shared",
        created_at=base,
    )
    _append_episode(
        store,
        episode_id="ep-2",
        session_id="session-shared",
        created_at=base + timedelta(seconds=1),
    )
    _append_episode(
        store,
        episode_id="ep-3",
        session_id="",
        created_at=base + timedelta(seconds=2),
    )

    groups = group_episodes_into_sessions(store, app="ohmo")

    assert groups == [
        EvalSessionGroup(
            session_id="session-shared",
            episode_ids=("ep-1", "ep-2"),
        ),
        EvalSessionGroup(session_id="ep-3", episode_ids=("ep-3",)),
    ]


def test_session_replay_runner_reuses_one_engine_across_turns(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _append_tool_episode(
        store,
        episode_id="ep-1",
        session_id="session-1",
        user_text="private first turn",
        tool_call_id="tool-1",
    )
    _append_tool_episode(
        store,
        episode_id="ep-2",
        session_id="session-1",
        user_text="private second turn",
        tool_call_id="tool-2",
    )
    api_client = _PerTurnToolModelApiClient()

    result = SessionReplayRunner(
        api_client=api_client,
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
    ).run(
        group=EvalSessionGroup("session-1", ("ep-1", "ep-2")),
        store=store,
    )

    assert result.turn_count == 2
    assert result.final_text == "session final 2"
    assert result.union_capabilities == ("bash:weather-cli forecast",)
    assert result.metadata["engine_message_count"] >= 8
    assert len(api_client.requests) == 4
    second_turn_request = api_client.requests[2]
    user_texts = [
        message.text
        for message in second_turn_request.messages
        if message.role == "user" and message.text.startswith("private")
    ]
    assert user_texts == ["private first turn", "private second turn"]


class _FaithfulTurnInput(BaseModel):
    value: str = ""


class _WriteStateTool(BaseTool):
    name = "write_state"
    description = "Write a marker value into the persistent session workspace."
    input_model = _FaithfulTurnInput

    def __init__(self, workspace: Path, seen: list[str]) -> None:
        self._workspace = workspace
        self._seen = seen

    async def execute(
        self,
        arguments: _FaithfulTurnInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        (self._workspace / "state.txt").write_text(arguments.value, encoding="utf-8")
        self._seen.append("write")
        return ToolResult(output="state written")


class _ReadStateTool(BaseTool):
    name = "read_state"
    description = "Read back the marker value from the session workspace."
    input_model = _FaithfulTurnInput

    def __init__(self, workspace: Path, seen: list[str]) -> None:
        self._workspace = workspace
        self._seen = seen

    async def execute(
        self,
        arguments: _FaithfulTurnInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del arguments, context
        state_path = self._workspace / "state.txt"
        _ = state_path.read_text(encoding="utf-8")
        self._seen.append("read")
        return ToolResult(output="state read")


class _CannedUserSimulator:
    def __init__(self) -> None:
        self._turns = [
            UserTurn("private first turn", "llm_fallback"),
            UserTurn("private second turn", "llm_fallback"),
        ]
        self.ended_reason = ""

    def next_turn(self, **_kwargs):
        if not self._turns:
            self.ended_reason = "model_done"
            return None
        return self._turns.pop(0)


class _WriteThenReadToolModelClient:
    def __init__(self) -> None:
        self.requests = []
        self._tool_calls = 0
        self._final_count = 0

    async def stream_message(self, request):
        self.requests.append(request)
        last_message = request.messages[-1]
        if any(isinstance(block, ToolResultBlock) for block in last_message.content):
            self._final_count += 1
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[TextBlock(text=f"session final {self._final_count}")],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return

        tool_name = "write_state" if self._tool_calls == 0 else "read_state"
        self._tool_calls += 1
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id=f"toolu-faithful-{self._tool_calls}",
                        name=tool_name,
                        input={"value": "persisted"},
                    )
                ],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def test_faithful_session_runner_uses_persistent_workspace_between_turns(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    _append_tool_episode(
        store,
        episode_id="ep-1",
        session_id="session-1",
        user_text="private first turn",
        tool_call_id="tool-1",
    )
    _append_tool_episode(
        store,
        episode_id="ep-2",
        session_id="session-1",
        user_text="private second turn",
        tool_call_id="tool-2",
    )
    observed_tool_calls: list[str] = []
    tool_roots: list[Path] = []

    def fake_tool_factory(root: Path) -> list[BaseTool]:
        tool_roots.append(root)
        return [_WriteStateTool(root, observed_tool_calls), _ReadStateTool(root, observed_tool_calls)]

    def fake_state(_root: Path) -> dict[str, object]:
        return {}

    result = FaithfulSessionRunner(
        api_client=_WriteThenReadToolModelClient(),
        model="eval-model",
        system_prompt="faithful system",
        cwd=tmp_path,
        sandbox_tool_factory=fake_tool_factory,
        sandbox_state_fn=fake_state,
    ).run(
        group=EvalSessionGroup("session-1", ("ep-1", "ep-2")),
        store=store,
        user_simulator=_CannedUserSimulator(),
    )

    assert result.turn_count == 2
    assert result.metadata["fixture_count"] == 2
    assert tool_roots
    assert len(set(tool_roots)) == 1
    assert not tool_roots[0].exists()
    assert observed_tool_calls == ["write", "read"]


def test_session_replay_runner_default_matches_replay_simulator(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _append_tool_episode(
        store,
        episode_id="ep-1",
        session_id="session-1",
        user_text="private first turn",
        tool_call_id="tool-1",
    )
    _append_tool_episode(
        store,
        episode_id="ep-2",
        session_id="session-1",
        user_text="private second turn",
        tool_call_id="tool-2",
    )
    group = EvalSessionGroup("session-1", ("ep-1", "ep-2"))

    default_result = SessionReplayRunner(
        api_client=_PerTurnToolModelApiClient(),
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
    ).run(group=group, store=store)
    replay_result = SessionReplayRunner(
        api_client=_PerTurnToolModelApiClient(),
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
    ).run(group=group, store=store, user_simulator=ReplayUserSimulator())

    assert default_result.turns == replay_result.turns
    assert default_result.metadata["user_turn_sources"] == ["replay", "replay"]
    assert default_result.metadata["replay_hit_rate"] == 1.0
    assert default_result.metadata["llm_fallback_count"] == 0
    assert default_result.metadata["ended_reason"] == "captured_exhausted"


def test_replay_matches_core_capability_sets():
    assert replay_matches(
        ("bash:weather-cli forecast", "todo_write"),
        ("bash:weather-cli forecast",),
    )
    assert not replay_matches((), ("bash:weather-cli forecast",))
    assert not replay_matches(
        ("bash:calendar-cli create",),
        ("bash:weather-cli forecast",),
    )


@pytest.mark.asyncio
async def test_derive_ironuser_spec_extracts_json_fields():
    client = _RecordingTextApiClient(
        '{"intent":"Book a seafood dinner after the 29th.",'
        '"known_info":["the date must be after the 29th","the restaurant is Vinci"],'
        '"constraints":["seafood only","cite the source page"]}'
    )

    spec = await derive_ironuser_spec(
        client,
        "sim-model",
        captured_prompts=(
            "Find a seafood restaurant.",
            "After the 29th, and it is Vinci not Vince.",
        ),
    )

    assert spec.intent == "Book a seafood dinner after the 29th."
    assert spec.known_info == (
        "the date must be after the 29th",
        "the restaurant is Vinci",
    )
    assert spec.constraints == ("seafood only", "cite the source page")


@pytest.mark.asyncio
async def test_derive_ironuser_spec_falls_back_on_unparseable_output():
    client = _RecordingTextApiClient("not json")

    spec = await derive_ironuser_spec(
        client,
        "sim-model",
        captured_prompts=("Find a seafood restaurant.",),
    )

    assert spec == IronUserSpec(
        intent="Find a seafood restaurant.",
        known_info=(),
        constraints=(),
    )


@pytest.mark.asyncio
async def test_goal_anchored_llm_user_simulator_prompts_from_spec_and_ends():
    client = _RecordingTextApiClient(
        '{"intent":"Book a seafood dinner after the 29th.",'
        '"known_info":["the date must be after the 29th"],'
        '"constraints":["seafood only"]}',
        '{"message": null, "ended_reason": "intent_met"}',
    )
    simulator = LlmUserSimulator(
        api_client=client,
        model="sim-model",
        system_prompt="SIM_SYSTEM",
        goal_anchored=True,
    )

    turn = await simulator.next_turn(
        transcript=(
            ("user", "Find a seafood restaurant."),
            ("assistant", "Booked a seafood dinner after the 29th."),
        ),
        captured_prompts=(
            "Find a seafood restaurant.",
            "After the 29th, and seafood only.",
        ),
        captured_capabilities=(),
        index=1,
        last_turn=None,
    )

    assert turn is None
    assert simulator.ended_reason == "intent_met"
    assert len(client.requests) == 2
    prompt = client.requests[1].messages[0].text
    assert "Book a seafood dinner after the 29th." in prompt
    assert "reveal known_info only when asked" in prompt.lower()
    assert "do not volunteer" in prompt.lower()


def test_hybrid_user_simulator_falls_back_after_divergence(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _append_tool_episode(
        store,
        episode_id="ep-1",
        session_id="session-1",
        user_text="private first turn",
        tool_call_id="tool-1",
    )
    _append_tool_episode(
        store,
        episode_id="ep-2",
        session_id="session-1",
        user_text="private second turn",
        tool_call_id="tool-2",
    )

    result = SessionReplayRunner(
        api_client=_ClarifyThenToolModelApiClient(),
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
    ).run(
        group=EvalSessionGroup("session-1", ("ep-1", "ep-2")),
        store=store,
        user_simulator=HybridUserSimulator(llm=_OneTurnUserSimulator()),
    )

    assert result.metadata["user_turn_sources"] == ["replay", "llm_fallback"]
    assert result.metadata["replay_hit_count"] == 1
    assert result.metadata["llm_fallback_count"] == 1
    assert result.metadata["replay_hit_rate"] == 0.5
    assert result.metadata["ended_reason"] == "captured_exhausted"
    assert result.turns[1].episode_id == ""


def test_hybrid_user_simulator_ends_when_fallback_unavailable(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _append_tool_episode(
        store,
        episode_id="ep-1",
        session_id="session-1",
        user_text="private first turn",
        tool_call_id="tool-1",
    )
    _append_tool_episode(
        store,
        episode_id="ep-2",
        session_id="session-1",
        user_text="private second turn",
        tool_call_id="tool-2",
    )

    result = SessionReplayRunner(
        api_client=_ClarifyThenToolModelApiClient(),
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
    ).run(
        group=EvalSessionGroup("session-1", ("ep-1", "ep-2")),
        store=store,
        user_simulator=HybridUserSimulator(),
    )

    assert result.turn_count == 1
    assert result.metadata["user_turn_sources"] == ["replay"]
    assert result.metadata["llm_fallback_count"] == 0
    assert result.metadata["ended_reason"] == "fallback_unavailable"


def test_score_session_covers_capabilities_and_final_outcome():
    passing = EvalSessionRunResult(
        session_id="session-1",
        turns=(
            EvalSessionTurnResult(
                episode_id="ep-1",
                capabilities=("bash:weather-cli forecast",),
            ),
        ),
        union_capabilities=("bash:weather-cli forecast",),
        final_text="done",
        turn_count=1,
    )

    passed = score_session(
        passing,
        gold_capabilities=("bash:weather-cli forecast", "todo_write"),
        state_changed=True,
    )
    assert passed["passed"] is True
    assert passed["missing_capabilities"] == []
    assert passed["checks"]["capability_coverage"] is True
    assert passed["checks"]["final_outcome_reached"] is True

    missing = score_session(
        passing,
        gold_capabilities=("bash:calendar-cli create",),
        state_changed=True,
    )
    assert missing["passed"] is False
    assert missing["missing_capabilities"] == ["bash:calendar-cli create"]

    unchanged = score_session(
        passing,
        gold_capabilities=("todo_write",),
        state_changed=False,
    )
    assert unchanged["checks"]["capability_coverage"] is True
    assert unchanged["checks"]["final_outcome_reached"] is False

    fallback = score_session(
        passing,
        gold_capabilities=(),
        state_changed=None,
    )
    assert fallback["checks"]["final_outcome_reached"] is True


def test_score_session_allows_terminal_clarification():
    clarifying = EvalSessionRunResult(
        session_id="session-1",
        turns=(
            EvalSessionTurnResult(
                episode_id="ep-1",
                final_text="Which city should I use?",
            ),
        ),
        union_capabilities=(),
        final_text="Which city should I use?",
        turn_count=1,
    )

    blocked = score_session(
        clarifying,
        gold_capabilities=("bash:weather-cli forecast",),
        state_changed=False,
    )
    assert blocked["passed"] is False
    assert blocked["checks"]["capability_coverage"] is False
    assert blocked["checks"]["final_outcome_reached"] is False
    assert "terminal_clarification" not in blocked

    allowed = score_session(
        clarifying,
        gold_capabilities=("bash:weather-cli forecast",),
        state_changed=False,
        clarification_allowed=True,
    )
    assert allowed["passed"] is True
    assert allowed["checks"]["capability_coverage"] is True
    assert allowed["checks"]["final_outcome_reached"] is True
    assert allowed["terminal_clarification"] is True
    assert allowed["clarification_allowed"] is True
    assert allowed["warnings"] == ["capability_coverage_terminal_clarification"]


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
                    content=[TextBlock(text=f"session final {self._final_count}")],
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
                    content=[TextBlock(text=f"session final {self._final_count}")],
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


class _OneTurnUserSimulator:
    def __init__(self) -> None:
        self.calls = 0
        self.ended_reason = ""

    def next_turn(self, **kwargs):
        del kwargs
        if self.calls == 0:
            self.calls += 1
            self.ended_reason = ""
            return UserTurn("private synthetic answer", "llm_fallback")
        self.ended_reason = "model_done"
        return None


class _RecordingTextApiClient:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        text = self.responses.pop(0) if self.responses else ""
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def _append_episode(
    store: EvalStore,
    *,
    episode_id: str,
    session_id: str,
    created_at: datetime,
) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id=session_id,
            created_at=created_at,
            user_text=f"private request {episode_id}",
        )
    )


def _append_tool_episode(
    store: EvalStore,
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
