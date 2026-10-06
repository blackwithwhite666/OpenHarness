from __future__ import annotations

import pytest
from datetime import datetime, timezone

from camera_virtual_user import CameraVirtualUser, OfferedCameraChoices
from camera_runtime_support import camera_runtime_limits
from openharness.evals.session_user_simulator import UserTurn


class FixedUser:
    def __init__(self, text: str) -> None:
        self.text = text

    def next_turn(self, **_kwargs):
        return UserTurn(self.text, "llm_fallback")


class RecordingSimulator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.request = None

    def next_turn(self, **kwargs):
        self.request = kwargs
        return UserTurn(self.text, "llm_fallback")


@pytest.mark.parametrize(
    ("native_mode", "expected"),
    [(True, (8, "medium")), (False, (4, "none"))],
)
def test_camera_runtime_limits_are_mode_specific(native_mode, expected):
    assert camera_runtime_limits(native_mode=native_mode) == expected


async def test_camera_user_request_includes_visible_choices_and_owner_goal_without_losing_history():
    choices = OfferedCameraChoices(
        question="Вы съели это? Какую часть порции учитывать?",
        options=("Всю тарелку", "Яйцо и часть риса", "Не ела"),
        callback_ids=("ask:0", "ask:1", "ask:2"),
        native_message_id="77",
        media_source_ids=("candidate-1",),
    )
    history = (("user", "Я отправила фотографию еды."), ("assistant", "Фото получил."))
    scenario = "Owner ate only part of the meal yesterday. Preserve that amount and date."
    capabilities = (("camera-photo", "telegram-reply"),)
    last_turn = UserTurn("prior user turn", "history")
    simulator = RecordingSimulator("Нет, не ела; если и пробовать, то только завтра")

    action = await CameraVirtualUser(simulator).next_camera_action(
        offered=choices,
        transcript=history,
        captured_prompts=(scenario,),
        captured_capabilities=capabilities,
        index=4,
        last_turn=last_turn,
    )

    assert action is not None
    assert action.callback_data is None
    assert action.text == simulator.text
    assert action.reply_to_message_id == "77"
    assert action.media_source_ids == ("candidate-1",)
    request = simulator.request
    assert request is not None
    assert request["transcript"][: len(history)] == history
    offered_turn = request["transcript"][-1]
    assert offered_turn[0] == "assistant"
    assert choices.question in offered_turn[1]
    assert all(option in offered_turn[1] for option in choices.options)
    assert "ask:0" not in offered_turn[1]
    assert request["captured_prompts"] == (scenario,)
    assert request["captured_capabilities"] == capabilities
    assert request["index"] == 4
    assert request["last_turn"] is last_turn


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
        ("да, я это съел(а)", None, "да, я это съел(а)"),
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


async def test_context_mode_keeps_visible_choices_and_typed_text_without_reply_metadata():
    offered = OfferedCameraChoices(
        question="Вы съели это? Какую часть порции учитывать?",
        options=("Всю порцию", "Половину порции", "Не ела"),
        callback_ids=("ask:0", "ask:1", "ask:2"),
        native_message_id="77",
        media_source_ids=("candidate-1",),
    )
    utterance = "Я съела примерно половину порции"
    reply_simulator = RecordingSimulator(utterance)
    context_simulator = RecordingSimulator(utterance)
    common = dict(
        offered=offered, transcript=(("user", "Вот фотография"),),
        captured_prompts=("Use the actual food amount.",),
        captured_capabilities=(("camera-photo", "telegram-reply"),), index=0,
        last_turn=None,
    )
    reply_action = await CameraVirtualUser(reply_simulator).next_camera_action(**common)
    context_action = await CameraVirtualUser(
        context_simulator, typed_reply_mode="context",
    ).next_camera_action(**common)

    assert reply_simulator.request["transcript"] == context_simulator.request["transcript"]
    assert context_action.text == utterance
    assert context_action.callback_data is None
    assert context_action.typed_reply_mode == "context"
    assert context_action.reply_to_message_id is None
    assert reply_action.reply_to_message_id == "77"
    received_at = datetime.now(timezone.utc)
    inbound = context_action.to_inbound_message(
        sender_id="123", chat_id="123", source_message_id="context-source-1",
        received_at=received_at,
    )
    assert inbound.content == utterance
    assert inbound.timestamp is received_at
    assert inbound.metadata["message_id"] == "context-source-1"
    assert "reply_to_message_id" not in inbound.metadata
    assert "native_message_id" not in inbound.metadata
    assert inbound.metadata["_camera_virtual_media_source_ids"] == ("candidate-1",)


@pytest.mark.parametrize("mode", ["", "reply-to-anything", None])
def test_camera_user_rejects_unknown_typed_reply_mode(mode):
    with pytest.raises(ValueError, match="reply or context"):
        CameraVirtualUser(FixedUser("typed answer"), typed_reply_mode=mode)


async def test_context_mode_keeps_native_callback_selection():
    offered = OfferedCameraChoices(
        question="Did you eat this?",
        options=("Да, я это съел(а)", "Нет, не ел(а)"),
        callback_ids=("ask:0", "ask:1"),
        native_message_id="77",
        media_source_ids=("candidate-1",),
    )
    action = await CameraVirtualUser(
        FixedUser("Да, я это съел(а)"), typed_reply_mode="context",
    ).next_camera_action(
        offered=offered, transcript=(), captured_prompts=(),
        captured_capabilities=(), index=0, last_turn=None,
    )
    assert action.callback_data == "ask:0"
    with pytest.raises(ValueError, match="actual Telegram callback handler"):
        action.to_inbound_message(
            sender_id="123", chat_id="123", source_message_id="context-source",
            received_at=datetime.now(timezone.utc),
        )


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
