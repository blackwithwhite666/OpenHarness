"""User simulators for session-level eval replay."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Awaitable, Literal, Protocol

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    SupportsStreamingMessages,
)
from openharness.engine.messages import ConversationMessage
from openharness.evals.execution import _INCIDENTAL_CAPABILITIES

log = logging.getLogger(__name__)


class SessionTurnLike(Protocol):
    """Structural shape needed from one model turn result."""

    capabilities: Sequence[str]
    tool_path: Sequence[str]
    final_text: str


@dataclass(frozen=True)
class UserTurn:
    """One simulator-produced user message and its provenance."""

    text: str
    source: Literal["replay", "llm_fallback", "none"]


UserTurnResult = UserTurn | None | Awaitable[UserTurn | None]
Transcript = Sequence[tuple[str, str]]


class UserSimulator(Protocol):
    """Supplies the next user turn for a dynamic session replay run."""

    def next_turn(
        self,
        *,
        transcript: Transcript,
        captured_prompts: Sequence[str],
        captured_capabilities: Sequence[Sequence[str]],
        index: int,
        last_turn: SessionTurnLike | None,
    ) -> UserTurnResult:
        """Return the next user turn or None to end the session."""


class ReplayUserSimulator:
    """Replay captured user prompts in order."""

    def __init__(self, captured_prompts: Sequence[str] = ()) -> None:
        self._captured_prompts = tuple(captured_prompts)
        self.ended_reason = ""

    def next_turn(
        self,
        *,
        transcript: Transcript,
        captured_prompts: Sequence[str],
        captured_capabilities: Sequence[Sequence[str]],
        index: int,
        last_turn: SessionTurnLike | None,
    ) -> UserTurn | None:
        del transcript, captured_capabilities, last_turn
        prompts = self._captured_prompts or tuple(captured_prompts)
        if index < len(prompts):
            self.ended_reason = ""
            return UserTurn(prompts[index], "replay")
        self.ended_reason = "captured_exhausted"
        return None


class LlmUserSimulator:
    """Generate fallback user turns with an identity-separated API client.

    The caller must provide an api_client/profile that is separate from the
    evaluated agent's client/profile; OHMO wiring enforces that boundary.
    """

    def __init__(
        self,
        *,
        api_client: SupportsStreamingMessages,
        model: str,
        system_prompt: str,
        max_tokens: int = 512,
    ) -> None:
        self._api_client = api_client
        self._model = model
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens
        self.ended_reason = ""

    async def next_turn(
        self,
        *,
        transcript: Transcript,
        captured_prompts: Sequence[str],
        captured_capabilities: Sequence[Sequence[str]],
        index: int,
        last_turn: SessionTurnLike | None,
    ) -> UserTurn | None:
        del captured_capabilities, last_turn
        prompt = _user_simulator_prompt(
            captured_prompts=captured_prompts,
            transcript=transcript,
            index=index,
        )
        text = ""
        async for event in self._api_client.stream_message(
            ApiMessageRequest(
                model=self._model,
                messages=[ConversationMessage.from_user_text(prompt)],
                system_prompt=self._system_prompt,
                max_tokens=self._max_tokens,
                tools=[],
            )
        ):
            if isinstance(event, ApiMessageCompleteEvent):
                text = event.message.text.strip()
        if not text:
            self.ended_reason = "model_done"
            return None
        self.ended_reason = ""
        return UserTurn(text, "llm_fallback")


class HybridUserSimulator:
    """Replay while the agent stays on trajectory, then fall back to an LLM."""

    def __init__(
        self,
        *,
        replay: ReplayUserSimulator | None = None,
        llm: UserSimulator | None = None,
    ) -> None:
        self._replay = replay or ReplayUserSimulator()
        self._llm = llm
        self.ended_reason = ""

    async def next_turn(
        self,
        *,
        transcript: Transcript,
        captured_prompts: Sequence[str],
        captured_capabilities: Sequence[Sequence[str]],
        index: int,
        last_turn: SessionTurnLike | None,
    ) -> UserTurn | None:
        self.ended_reason = ""
        if index == 0:
            turn = self._replay.next_turn(
                transcript=transcript,
                captured_prompts=captured_prompts,
                captured_capabilities=captured_capabilities,
                index=index,
                last_turn=last_turn,
            )
            if turn is None:
                self.ended_reason = _simulator_ended_reason(
                    self._replay,
                    default="captured_exhausted",
                )
            log.info(
                "session user simulator decision source=%s reason=first_turn index=%d",
                turn.source if turn else "none",
                index,
            )
            return turn

        matched = False
        if last_turn is not None and 0 <= index - 1 < len(captured_capabilities):
            captured_for_last_turn: Sequence[str] = captured_capabilities[index - 1]
            matched = replay_matches(last_turn.capabilities, captured_for_last_turn)
        if matched:
            if index < len(captured_prompts):
                turn = self._replay.next_turn(
                    transcript=transcript,
                    captured_prompts=captured_prompts,
                    captured_capabilities=captured_capabilities,
                    index=index,
                    last_turn=last_turn,
                )
                log.info(
                    "session user simulator decision source=%s reason=replay_match index=%d",
                    turn.source if turn else "none",
                    index,
                )
                return turn
            self.ended_reason = "captured_exhausted"
            log.info(
                "session user simulator decision source=none reason=captured_exhausted index=%d",
                index,
            )
            return None

        if index >= len(captured_prompts) and not _turn_waiting_for_user(last_turn):
            self.ended_reason = "captured_exhausted"
            log.info(
                "session user simulator decision source=none reason=captured_exhausted index=%d",
                index,
            )
            return None

        if self._llm is None:
            self.ended_reason = "fallback_unavailable"
            log.info(
                "session user simulator decision source=none reason=fallback_unavailable index=%d",
                index,
            )
            return None

        turn = await _resolve_user_turn(
            self._llm.next_turn(
                transcript=transcript,
                captured_prompts=captured_prompts,
                captured_capabilities=captured_capabilities,
                index=index,
                last_turn=last_turn,
            )
        )
        if turn is None:
            self.ended_reason = _simulator_ended_reason(
                self._llm,
                default="model_done",
            )
        log.info(
            "session user simulator decision source=%s reason=llm_fallback index=%d",
            turn.source if turn else "none",
            index,
        )
        return turn


def replay_matches(
    observed_capabilities: Sequence[str],
    captured_capabilities: Sequence[str],
) -> bool:
    """Return whether a turn stayed on the captured capability trajectory.

    This policy is deliberately small and tunable: for P1, a replay hit means the
    model's non-incidental capability set equals the captured non-incidental set.
    """
    return _core_capabilities(observed_capabilities) == _core_capabilities(
        captured_capabilities
    )


async def _resolve_user_turn(value: UserTurnResult) -> UserTurn | None:
    if inspect.isawaitable(value):
        return await value
    return value


def _simulator_ended_reason(simulator: object, *, default: str) -> str:
    ended_reason = getattr(simulator, "ended_reason", "")
    return str(ended_reason or default)


def _core_capabilities(capabilities: Sequence[str]) -> frozenset[str]:
    return frozenset(
        capability
        for capability in capabilities
        if capability not in _INCIDENTAL_CAPABILITIES
    )


def _turn_waiting_for_user(turn: SessionTurnLike | None) -> bool:
    return bool(turn is not None and not turn.tool_path and turn.final_text.strip())


def _user_simulator_prompt(
    *,
    captured_prompts: Sequence[str],
    transcript: Transcript,
    index: int,
) -> str:
    goal = captured_prompts[0] if captured_prompts else ""
    transcript_text = "\n".join(
        f"{role}: {text.strip()}" for role, text in transcript if text.strip()
    )
    if not transcript_text:
        transcript_text = "(no conversation yet)"
    return (
        "Original user goal or first captured user turn:\n"
        f"{goal}\n\n"
        "Conversation so far:\n"
        f"{transcript_text}\n\n"
        f"Write the next user message for turn {index + 1}. "
        "Return only the user message, with no analysis or labels."
    )
