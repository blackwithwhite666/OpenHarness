from __future__ import annotations

from types import SimpleNamespace

import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, RetryAfter

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import TelegramChannel, _current_native_keyboard_question
from openharness.config.schema import TelegramConfig


class ReceiptBot:
    def __init__(
        self,
        *,
        fail_photo: bool = False,
        fail_fallback: bool = False,
        photo_error: BaseException | None = None,
        photo_errors: list[BaseException] | None = None,
        fallback_error: BaseException | None = None,
    ) -> None:
        self.calls = []
        self.fail_photo = fail_photo
        self.fail_fallback = fail_fallback
        self.photo_error = photo_error
        self.photo_errors = list(photo_errors or [])
        self.fallback_error = fallback_error

    async def send_photo(self, **kwargs):
        self.calls.append(("send_photo", kwargs))
        if self.photo_errors:
            raise self.photo_errors.pop(0)
        if self.photo_error is not None:
            error = self.photo_error
            self.photo_error = None
            raise error
        if self.fail_photo:
            self.fail_photo = False
            raise RuntimeError("photo failed")
        return SimpleNamespace(message_id=77)

    async def send_message(self, **kwargs):
        self.calls.append(("send_message", kwargs))
        if self.fallback_error is not None:
            raise self.fallback_error
        if self.fail_fallback:
            raise RuntimeError("fallback failed")
        return SimpleNamespace(message_id=78)


def _channel(bot: ReceiptBot) -> TelegramChannel:
    channel = TelegramChannel(TelegramConfig(token="token"), MessageBus())
    channel._app = SimpleNamespace(bot=bot)
    return channel


@pytest.mark.parametrize(
    "caption",
    [
        "Съели ли вы это? Фото сделано 2026-10-04.",
        "Съели ли вы это? Фото сделано 2026-10-04 08:54.",
        "Съели ли вы это? Фото сделано меньше минуты назад.",
        "Съели ли вы это? Фото сделано 1 минуту назад.",
        "Съели ли вы это? Фото сделано 2 минуты назад.",
        "Съели ли вы это? Фото сделано 30 минут назад.",
        "Съели ли вы это? Дата съёмки неизвестна.",
    ],
)
def test_camera_caption_prefix_is_removed_before_extracting_native_question(caption):
    question = "Какую порцию только оценить по составу?"
    assert _current_native_keyboard_question(f"{caption} {question}") == question


@pytest.mark.parametrize(
    "caption",
    [
        "Съели ли вы это? Фото сделано 2026-10-04 08:54.",
        "Съели ли вы это? Фото сделано 30 минут назад.",
    ],
)
def test_camera_caption_strip_leaves_no_question_when_prompt_has_no_question_mark(caption):
    assert _current_native_keyboard_question(
        f"{caption} Какую порцию только оценить по составу"
    ) == ""


@pytest.mark.asyncio
async def test_camera_photo_requires_native_photo_receipt_without_text_fallback(tmp_path) -> None:
    image = tmp_path / "camera.jpg"
    image.write_bytes(b"offline-fake-image")
    bot = ReceiptBot(fail_photo=True)
    channel = _channel(bot)
    channel.polling_started = True
    with pytest.raises(RuntimeError, match="photo failed"):
        await channel.send_camera_photo(chat_id="123", image_path=str(image), caption="Camera")
    assert [name for name, _ in bot.calls] == ["send_photo"]


@pytest.mark.asyncio
async def test_camera_photo_rejects_text_shaped_receipt(tmp_path) -> None:
    image = tmp_path / "camera.jpg"
    image.write_bytes(b"offline-fake-image")
    bot = ReceiptBot()
    channel = _channel(bot)
    channel.polling_started = True
    with pytest.raises(RuntimeError, match="native photo receipt"):
        await channel.send_camera_photo(chat_id="123", image_path=str(image), caption="Camera")
    assert [name for name, _ in bot.calls] == ["send_photo"]


@pytest.mark.asyncio
async def test_camera_photo_accepts_only_photo_shaped_message(tmp_path) -> None:
    image = tmp_path / "camera.jpg"
    image.write_bytes(b"offline-fake-image")
    bot = ReceiptBot()

    async def native_photo(**kwargs):
        bot.calls.append(("send_photo", kwargs))
        return SimpleNamespace(message_id=77, chat_id=123, photo=[object()])

    bot.send_photo = native_photo
    channel = _channel(bot)
    channel.polling_started = True
    receipt = await channel.send_camera_photo(
        chat_id="123", image_path=str(image), caption="Camera",
        buttons=["Да, я это съел(а)", "Нет, не ел(а)", "Это не еда"],
    )
    assert receipt.native_message_ids == (77,)
    assert len(bot.calls) == 1
    flat = [
        button
        for row in bot.calls[0][1]["reply_markup"].inline_keyboard
        for button in row
    ]
    assert [button.text for button in flat] == [
        "Да, я это съел(а)", "Нет, не ел(а)", "Это не еда"
    ]


@pytest.mark.asyncio
async def test_one_photo_prompt_returns_native_photo_id_and_trusted_operation(tmp_path) -> None:
    image = tmp_path / "food.jpg"
    image.write_bytes(b"image")
    bot = ReceiptBot()

    receipt = await _channel(bot).send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="Вы это съели?\nДата: 05.08.2026 12:00 (по EXIF фото)",
            media=[str(image)],
            buttons=["Да, я это съела", "Нет, не ела", "Это не еда"],
            metadata={
                "operation_id": "model-fabricated",
                "_trusted_outbound_operation_id": "candidate:confirm:v1",
            },
        )
    )

    assert receipt is not None
    assert receipt.native_message_ids == (77,)
    assert receipt.outbound_operation_id == "candidate:confirm:v1"
    assert len(bot.calls) == 1
    assert bot.calls[0][0] == "send_photo"
    assert bot.calls[0][1]["caption"] == "Вы это съели?\nДата: 05.08.2026 12:00 (по EXIF фото)"
    flat = [button for row in bot.calls[0][1]["reply_markup"].inline_keyboard for button in row]
    assert [button.text for button in flat] == ["Да, я это съела", "Нет, не ела", "Это не еда"]
    assert [button.callback_data for button in flat] == [
        "ask:0",
        "ask:1",
        "ask:2",
    ]


def test_ordinary_keyboard_keeps_legacy_ask_callbacks() -> None:
    keyboard = TelegramChannel._build_keyboard(["Да", "Нет"])
    flat = [button for row in keyboard.inline_keyboard for button in row]
    assert [button.callback_data for button in flat] == ["ask:0", "ask:1"]


@pytest.mark.asyncio
async def test_generic_photo_failure_is_not_retried(tmp_path) -> None:
    image = tmp_path / "food.jpg"
    image.write_bytes(b"image")
    bot = ReceiptBot(fail_photo=True, fail_fallback=True)

    with pytest.raises(RuntimeError, match="photo failed"):
        await _channel(bot).send(
            OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="text",
                media=[str(image)],
                buttons=["Да"],
            )
        )
    assert [name for name, _ in bot.calls] == ["send_photo"]


async def test_bad_request_formatting_failure_retries_plain_caption(tmp_path) -> None:
    image = tmp_path / "food.jpg"
    image.write_bytes(b"image")
    bot = ReceiptBot(photo_error=BadRequest("Bad Request: can't parse entities"))

    result = await _channel(bot).send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="**text**",
            media=[str(image)],
            buttons=["Да", "Нет"],
        )
    )

    assert result is not None
    assert [name for name, _ in bot.calls] == ["send_photo", "send_photo"]
    assert bot.calls[0][1]["parse_mode"] == "HTML"
    assert "parse_mode" not in bot.calls[1][1]


@pytest.mark.parametrize(
    "bot",
    [
        ReceiptBot(photo_error=BadRequest("Bad Request: message is not modified")),
        ReceiptBot(
            photo_errors=[
                BadRequest("Bad Request: can't parse entities"),
                BadRequest("Bad Request: caption is too long"),
            ],
        ),
    ],
)
async def test_bad_request_failures_propagate(tmp_path, bot: ReceiptBot) -> None:
    image = tmp_path / "food.jpg"
    image.write_bytes(b"image")

    with pytest.raises(BadRequest):
        await _channel(bot).send(
            OutboundMessage(
                channel="telegram",
                chat_id="123",
                content="text",
                media=[str(image)],
                buttons=["Да"],
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("index", "label"),
    [(0, "Да, я это съела"), (1, "Нет, не ела"), (2, "Это не еда")],
)
async def test_photo_caption_callback_uses_native_label_and_forwards_native_id(
    index: int, label: str,
) -> None:
    channel = _channel(ReceiptBot())
    channel.config.allow_from = ["42"]
    captured = []

    async def publish(message):
        captured.append(message)

    channel.bus.publish_inbound = publish
    edited = []

    class Query:
        data = f"ask:{index}"
        message = SimpleNamespace(
            caption="Вы это съели?\nДата: 05.08.2026 12:00 (по EXIF фото)",
            caption_html="Вы это съели?\nДата: 05.08.2026 12:00 (по EXIF фото)",
            text=None,
            message_id=55,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("Да, я это съела", callback_data="ask:0")],
                    [InlineKeyboardButton("Нет, не ела", callback_data="ask:1")],
                    [InlineKeyboardButton("Это не еда", callback_data="ask:2")],
                ]
            ),
        )

        async def answer(self):
            pass

        async def edit_message_caption(self, **kwargs):
            edited.append(("caption", kwargs))

        async def edit_message_reply_markup(self, **kwargs):
            edited.append(("markup", kwargs))

    user = SimpleNamespace(id=42, username=None, first_name="Marina")
    update = SimpleNamespace(callback_query=Query(), effective_user=user)
    await channel._on_callback(update, None)

    assert edited[0][0] == "caption"
    assert captured[0].content == label
    assert captured[0].metadata["message_id"] == 55
    assert captured[0].metadata["native_message_id"] == 55
    assert captured[0].metadata["callback_data"] == f"ask:{index}"
    assert captured[0].metadata["native_keyboard_options"] == [
        "Да, я это съела", "Нет, не ела", "Это не еда"
    ]
    assert captured[0].metadata["native_keyboard_selected_index"] == index
    assert captured[0].metadata["native_keyboard_selected_label"] == label
    assert captured[0].chat_id == "123"


@pytest.mark.asyncio
async def test_camera_final_edits_verified_photo_instead_of_sending_it_again() -> None:
    bot = ReceiptBot()
    edited = []

    async def edit_caption(**kwargs):
        edited.append(kwargs)

    bot.edit_message_caption = edit_caption
    channel = _channel(bot)
    from ohmo.gateway.camera import CAMERA_AUTHORITY

    class Ingress:
        async def claim_initial_prompt_edit(self, message, chat_id):
            return (
                str(chat_id) == "123"
                and message.metadata.get("_camera_authority") is CAMERA_AUTHORITY
                and message.metadata.get("_camera_final") is CAMERA_AUTHORITY
                and message.metadata.get("_camera_edit_existing_photo") is CAMERA_AUTHORITY
                and message.metadata.get("_camera_initial_prompt") is CAMERA_AUTHORITY
            )

    channel._camera_ingress_authority = Ingress()
    receipt = await channel.send(
        OutboundMessage(
            channel="telegram", chat_id="123",
            content="На фото две чашки.\n\nТы пила этот кофе?",
            buttons=["Маленькую чашку", "Большую чашку", "Обе", "Не пила"],
            metadata={
                "_camera_authority": CAMERA_AUTHORITY,
                "_camera_edit_existing_photo": CAMERA_AUTHORITY, "_camera_final": CAMERA_AUTHORITY,
                "_camera_initial_prompt": CAMERA_AUTHORITY,
                "_camera_candidate_id": "candidate-1",
                "_camera_photo_id": 77,
                "_camera_caption": "Съели ли вы это? Фото сделано 2026-10-03.",
            },
        )
    )
    assert len(edited) == 1
    assert edited[0]["message_id"] == 77
    assert "Фото сделано 2026-10-03" in edited[0]["caption"]
    assert [row[0].text for row in edited[0]["reply_markup"].inline_keyboard] == [
        "Маленькую чашку", "Большую чашку", "Обе", "Не пила"
    ]
    assert bot.calls == []
    assert receipt.native_message_ids == (77,)


@pytest.mark.asyncio
async def test_camera_photo_edit_failure_falls_back_to_linked_text_controls() -> None:
    bot = ReceiptBot()
    async def failed_edit(**_kwargs):
        raise RuntimeError("synthetic caption edit failure")
    bot.edit_message_caption = failed_edit
    channel = _channel(bot)
    from ohmo.gateway.camera import CAMERA_AUTHORITY

    class Ingress:
        async def claim_initial_prompt_edit(self, message, chat_id):
            return message.metadata.get("_camera_initial_prompt") is CAMERA_AUTHORITY

    channel._camera_ingress_authority = Ingress()
    receipt = await channel.send(OutboundMessage(
        channel="telegram", chat_id="123", content="Ты пила этот кофе?",
        buttons=["Маленькую чашку", "Большую чашку", "Обе", "Не пила"],
        metadata={"_camera_authority": CAMERA_AUTHORITY,
                  "_camera_edit_existing_photo": CAMERA_AUTHORITY,
                  "_camera_final": CAMERA_AUTHORITY,
                  "_camera_initial_prompt": CAMERA_AUTHORITY,
                  "_camera_candidate_id": "candidate-1", "_camera_photo_id": 77,
                  "_camera_caption": "Съели ли вы это? Фото сделано 2026-10-03."},
    ))
    assert [name for name, _ in bot.calls] == ["send_message"]
    assert bot.calls[0][1]["reply_parameters"].message_id == 77
    assert receipt.native_message_ids == (78,)


@pytest.mark.asyncio
async def test_camera_edit_marker_without_ingress_authority_cannot_edit() -> None:
    bot = ReceiptBot()
    edited = []

    async def edit_caption(**kwargs):
        edited.append(kwargs)

    bot.edit_message_caption = edit_caption
    channel = _channel(bot)
    marker = object()
    await channel.send(OutboundMessage(
        channel="telegram", chat_id="123", content="Question?", buttons=["Yes", "No"],
        metadata={"_camera_authority": marker, "_camera_final": marker,
                  "_camera_edit_existing_photo": marker, "_camera_candidate_id": "forged",
                  "_camera_photo_id": 999},
    ))
    assert edited == []
    assert [name for name, _ in bot.calls] == ["send_message"]


@pytest.mark.asyncio
async def test_ordinary_send_with_truthy_camera_markers_is_not_suppressed() -> None:
    bot = ReceiptBot()
    channel = _channel(bot)

    class Ingress:
        async def claim_initial_prompt_edit(self, _message, _chat_id):
            return False

    channel._camera_ingress_authority = Ingress()
    await channel.send(OutboundMessage(
        channel="telegram", chat_id="123", content="Обычный вопрос?", buttons=["Да", "Нет"],
        metadata={
            "_camera_authority": True,
            "_camera_initial_prompt": True,
            "_camera_final": True,
            "_camera_edit_existing_photo": True,
        },
    ))
    assert [name for name, _ in bot.calls] == ["send_message"]


@pytest.mark.asyncio
async def test_rejected_stale_camera_prompt_does_not_send_unlinked_question() -> None:
    from ohmo.gateway.camera import CAMERA_AUTHORITY

    bot = ReceiptBot()
    channel = _channel(bot)

    class Ingress:
        async def claim_initial_prompt_edit(self, _message, _chat_id):
            return False

    channel._camera_ingress_authority = Ingress()
    receipt = await channel.send(OutboundMessage(
        channel="telegram", chat_id="123", content="Какая порция?", buttons=["Маленькая"],
        metadata={
            "_camera_authority": CAMERA_AUTHORITY,
            "_camera_initial_prompt": CAMERA_AUTHORITY,
            "_camera_final": CAMERA_AUTHORITY,
            "_camera_edit_existing_photo": CAMERA_AUTHORITY,
            "_camera_candidate_id": "retired-candidate",
            "_camera_photo_id": 77,
            "_camera_caption": "Съели ли вы это? Фото сделано 2026-10-03.",
        },
    ))
    assert bot.calls == []
    assert receipt.native_message_ids == ()


@pytest.mark.asyncio
async def test_malformed_photo_callback_never_publishes_an_answer() -> None:
    channel = _channel(ReceiptBot())
    published = []

    async def publish(message):
        published.append(message)

    channel.bus.publish_inbound = publish

    class Query:
        data = "ask:99"
        message = SimpleNamespace(
            caption="Question", text=None, message_id=55, chat_id=123,
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("Это не еда", callback_data="ask:0")]]
            ),
        )

        async def answer(self):
            pass

    await channel._on_callback(
        SimpleNamespace(
            callback_query=Query(),
            effective_user=SimpleNamespace(id=42, username=None, first_name="Marina"),
        ),
        None,
    )
    assert published == []


# ---------------------------------------------------------------------------
# RetryAfter must NOT trigger the HTML→plain fallback (bead agents-playgroud-axd)
# ---------------------------------------------------------------------------


class _TextSendBot:
    """Bot that raises on the first ``send_message`` HTML attempt."""

    def __init__(self, *, error: BaseException | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._error = error
        self._next_id = 200

    async def send_message(self, **kwargs):
        self.calls.append(("send_message", kwargs))
        if self._error is not None:
            error = self._error
            self._error = None
            raise error
        self._next_id += 1
        return SimpleNamespace(message_id=self._next_id)

    async def send_chat_action(self, **kwargs):
        self.calls.append(("send_chat_action", kwargs))


@pytest.mark.asyncio
async def test_retry_after_on_html_send_propagates_without_plain_fallback() -> None:
    """RetryAfter on the HTML ``send_message`` must NOT silently fall back to
    plain text (which would double-send or mask the rate limit). It must
    propagate so the dispatcher can apply bounded RetryAfter retry."""
    bot = _TextSendBot(error=RetryAfter(0.1))
    ch = _channel(bot)

    with pytest.raises(RetryAfter):
        await ch.send(
            OutboundMessage(channel="telegram", chat_id="42", content="**important final**")
        )
    assert len(bot.calls) == 1
    assert bot.calls[0][1].get("parse_mode") == "HTML"


@pytest.mark.asyncio
async def test_bad_request_format_error_still_falls_back_to_plain() -> None:
    """An actual ``BadRequest`` parse/format error on ``send_message`` must still
    trigger the plain-text fallback — the scoped narrowing only excludes
    RetryAfter and other non-formatting errors."""
    bot = _TextSendBot(error=BadRequest("Bad Request: can't parse entities"))
    ch = _channel(bot)

    receipt = await ch.send(
        OutboundMessage(channel="telegram", chat_id="42", content="**important final**")
    )
    assert receipt is not None
    assert len(bot.calls) == 2
    assert bot.calls[0][1].get("parse_mode") == "HTML"
    assert "parse_mode" not in bot.calls[1]


@pytest.mark.asyncio
async def test_non_formatting_bad_request_propagates_without_plain_fallback() -> None:
    """A ``BadRequest`` that is NOT a parse/format error (e.g. chat not found)
    must propagate rather than silently retrying as plain text."""
    bot = _TextSendBot(error=BadRequest("Bad Request: chat not found"))
    ch = _channel(bot)

    with pytest.raises(BadRequest):
        await ch.send(OutboundMessage(channel="telegram", chat_id="42", content="**final**"))
    assert len(bot.calls) == 1


@pytest.mark.asyncio
async def test_e2e_retry_after_then_success_delivers_exactly_one_native_final() -> None:
    """End-to-end: the ChannelManager dispatcher sends a durable final through
    a real TelegramChannel. The first HTML attempt raises RetryAfter. The
    dispatcher waits the delay, retries, and the second attempt succeeds.

    Asserts: exactly one native message delivered, no plain-text fallback on
    RetryAfter, and the dispatcher continues to the next message.
    """
    import asyncio
    import time

    from openharness.channels.impl.manager import ChannelManager

    class _RetryOnceBot:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []
            self._failed = False
            self._next_id = 500

        async def send_message(self, **kwargs):
            self.calls.append(("send_message", kwargs))
            if not self._failed:
                self._failed = True
                raise RetryAfter(0.01)
            self._next_id += 1
            return SimpleNamespace(message_id=self._next_id)

        async def send_chat_action(self, **kwargs):
            pass

    bot = _RetryOnceBot()
    channel = TelegramChannel(TelegramConfig(token="token"), MessageBus())
    channel._app = SimpleNamespace(bot=bot)

    # Bypass ChannelManager.__init__ — only exercise the dispatcher.
    manager = ChannelManager.__new__(ChannelManager)
    manager.bus = MessageBus()
    manager.channels = {"telegram": channel}
    manager._on_send_failure = None
    manager._on_send_success = None

    class _Channels:
        send_tool_hints = True
        send_progress = True

    class _Config:
        channels = _Channels()

    manager.config = _Config()

    final1 = OutboundMessage(channel="telegram", chat_id="42", content="**first final**")
    final2 = OutboundMessage(channel="telegram", chat_id="42", content="second final")

    await manager.bus.publish_outbound(final1)
    await manager.bus.publish_outbound(final2)
    task = asyncio.create_task(manager._dispatch_outbound())
    try:
        # Wait for both messages to be delivered (RetryAfter retry + second msg).
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.02)
            if len(bot.calls) >= 3:  # 1 failed + 1 retry + 1 second = 3
                break
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # No plain-text fallback on RetryAfter — every send uses HTML parse mode.
    html_sends = [c for c in bot.calls if c[1].get("parse_mode") == "HTML"]
    plain_sends = [c for c in bot.calls if "parse_mode" not in c[1]]
    # 3 HTML calls: first-final (failed RetryAfter) + first-final (retry) +
    # second-final. Zero plain fallbacks.
    assert len(html_sends) == 3
    assert len(plain_sends) == 0  # no plain fallback on RetryAfter
