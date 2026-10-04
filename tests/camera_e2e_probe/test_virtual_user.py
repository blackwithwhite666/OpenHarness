from __future__ import annotations

import pytest
from datetime import datetime, timezone

from camera_virtual_user import CameraVirtualUser, OfferedCameraChoices
from openharness.evals.session_user_simulator import UserTurn


class FixedUser:
    def __init__(self, text: str) -> None:
        self.text = text

    def next_turn(self, **_kwargs):
        return UserTurn(self.text, "llm_fallback")


@pytest.mark.parametrize(
    ("utterance", "expected_callback", "expected_text"),
    [
        ("Да, я это съел(а)", "ask:0", "Да, я это съел(а)"),
        ("Нет, не ел(а)", "ask:1", "Нет, не ел(а)"),
        ("Это не еда", "ask:2", "Это не еда"),
        ("something else", None, "something else"),
        ("Я не съела это", None, "Я не съела это"),
        ("I ate half", None, "I ate half"),
        ("I ate it yesterday, not today", None, "I ate it yesterday, not today"),
        ("Yes, but only one spoon", None, "Yes, but only one spoon"),
        ("I did not eat it", None, "I did not eat it"),
    ],
)
async def test_camera_user_uses_only_offered_choices_and_keeps_media_provenance(
    utterance, expected_callback, expected_text
):
    choices = OfferedCameraChoices(
        question="Did you eat this?",
        options=("Да, я это съел(а)", "Нет, не ел(а)", "Это не еда"),
        callback_ids=("ask:0", "ask:1", "ask:2"),
        native_message_id="77",
        media_source_ids=("candidate-1",),
    )
    user = CameraVirtualUser(FixedUser(utterance))

    action = await user.next_camera_action(
        offered=choices,
        transcript=(),
        captured_prompts=(),
        captured_capabilities=(),
        index=0,
        last_turn=None,
    )

    assert action.callback_data == expected_callback
    assert action.text == expected_text
    assert action.reply_to_message_id == "77"
    assert action.media_source_ids == ("candidate-1",)
    if expected_callback is not None:
        with pytest.raises(ValueError, match="actual Telegram callback handler"):
            action.to_inbound_message(
                sender_id="owner-1",
                chat_id="chat-1",
                source_message_id="82",
                received_at=datetime.now(timezone.utc),
            )
        return
    inbound = action.to_inbound_message(
        sender_id="owner-1",
        chat_id="chat-1",
        source_message_id="82",
        received_at=datetime.now(timezone.utc),
    )
    assert inbound.content == expected_text
    assert inbound.metadata["message_id"] == "82"
    assert inbound.metadata["reply_to_message_id"] == "77"
    assert inbound.metadata["_camera_virtual_media_source_ids"] == ("candidate-1",)


@pytest.mark.parametrize(
    "choices",
    [
        OfferedCameraChoices("Q?", (), (), "77", ("candidate-1",)),
        OfferedCameraChoices("Q?", ("Yes", "Yes"), ("ask:0", "ask:1"), "77", ("candidate-1",)),
    ],
)
async def test_camera_user_rejects_invalid_offered_choices(choices):
    user = CameraVirtualUser(FixedUser("yes"))
    with pytest.raises(ValueError):
        await user.next_camera_action(
            offered=choices,
            transcript=(),
            captured_prompts=(),
            captured_capabilities=(),
            index=0,
            last_turn=None,
        )


@pytest.mark.parametrize(
    "choices",
    [
        OfferedCameraChoices(
            "Q?", ("Yes", "No"), ("ask:0", "ask:1"), "not-an-id", ("candidate-1",)
        ),
        OfferedCameraChoices("Q?", ("x" * 61, "No"), ("ask:0", "ask:1"), "77", ("candidate-1",)),
    ],
)
async def test_camera_user_rejects_unissued_or_truncated_choice_shapes(choices):
    user = CameraVirtualUser(FixedUser("Yes"))
    with pytest.raises(ValueError):
        await user.next_camera_action(
            offered=choices,
            transcript=(),
            captured_prompts=(),
            captured_capabilities=(),
            index=0,
            last_turn=None,
        )
