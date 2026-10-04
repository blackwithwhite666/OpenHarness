"""Camera choices adapter over OpenHarness's existing session user simulator."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from openharness.channels.bus.events import InboundMessage
from openharness.evals.session_user_simulator import SessionTurnLike, Transcript, UserSimulator


@dataclass(frozen=True)
class OfferedCameraChoices:
    question: str
    options: tuple[str, ...]
    callback_ids: tuple[str, ...]
    native_message_id: str
    media_source_ids: tuple[str, ...]


@dataclass(frozen=True)
class CameraUserAction:
    text: str
    callback_data: str | None
    reply_to_message_id: str
    media_source_ids: tuple[str, ...]

    def to_inbound_message(
        self,
        *,
        sender_id: str,
        chat_id: str,
        source_message_id: str,
        received_at: datetime,
        channel: str = "telegram",
    ) -> InboundMessage:
        """Create typed replies only; callbacks must pass through TelegramChannel."""
        if self.callback_data is not None:
            raise ValueError("Camera callbacks must use the actual Telegram callback handler")
        if not source_message_id.strip() or received_at.tzinfo is None:
            raise ValueError("typed Camera replies need a source ID and aware receive time")
        return InboundMessage(
            channel=channel,
            sender_id=sender_id,
            chat_id=chat_id,
            content=self.text,
            metadata={
                "message_id": source_message_id,
                "reply_to_message_id": self.reply_to_message_id,
                "_telegram_raw_text": self.text,
                "_camera_virtual_media_source_ids": self.media_source_ids,
                "is_group": False,
                "chat_type": "private",
            },
            timestamp=received_at,
        )


class CameraVirtualUser:
    """Translate one existing simulator turn into an offered callback or typed reply."""

    def __init__(self, simulator: UserSimulator) -> None:
        self._simulator = simulator

    async def next_camera_action(
        self,
        *,
        offered: OfferedCameraChoices,
        transcript: Transcript,
        captured_prompts: Sequence[str],
        captured_capabilities: Sequence[Sequence[str]],
        index: int,
        last_turn: SessionTurnLike | None,
    ) -> CameraUserAction | None:
        _validate_offered(offered)
        result = self._simulator.next_turn(
            transcript=(
                *transcript,
                ("assistant", _render_offered_choices(offered)),
            ),
            captured_prompts=captured_prompts,
            captured_capabilities=captured_capabilities,
            index=index,
            last_turn=last_turn,
        )
        turn = await result if inspect.isawaitable(result) else result
        if turn is None:
            return None
        text = turn.text.strip()
        if not text:
            return None
        choice = _matching_choice(text, offered.options)
        return CameraUserAction(
            text=offered.options[choice] if choice is not None else text,
            callback_data=offered.callback_ids[choice] if choice is not None else None,
            reply_to_message_id=offered.native_message_id,
            media_source_ids=offered.media_source_ids,
        )


def _validate_offered(offered: OfferedCameraChoices) -> None:
    if not offered.question.strip() or not offered.native_message_id.strip():
        raise ValueError("Camera choices require a question and native message id")
    if not 2 <= len(offered.options) <= 8:
        raise ValueError("Camera choices require 2 to 8 Telegram offered options")
    if len(offered.callback_ids) != len(offered.options) or any(
        not value.startswith("ask:") for value in offered.callback_ids
    ):
        raise ValueError("Camera choices require callback IDs from the issued keyboard")
    normalized = [option.strip().casefold() for option in offered.options]
    if any(not option or len(option) > 60 for option in normalized) or len(set(normalized)) != len(
        normalized
    ):
        raise ValueError("Camera offered options must be nonempty and unique")
    try:
        if int(offered.native_message_id) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("Camera choice native message id must be a positive Telegram ID") from None
    if not offered.media_source_ids or any(not item.strip() for item in offered.media_source_ids):
        raise ValueError("Camera choices require media source provenance")


def _matching_choice(text: str, options: Sequence[str]) -> int | None:
    for index, option in enumerate(options):
        if text == option:
            return index
    return None


def _render_offered_choices(offered: OfferedCameraChoices) -> str:
    """Give the simulator the same visible question and labels as Telegram."""
    return (
        f"{offered.question}\n\nВыберите один из предложенных вариантов:\n"
        + "\n".join(f"• {option}" for option in offered.options)
    )
