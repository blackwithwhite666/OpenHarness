from __future__ import annotations

from types import SimpleNamespace

import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, RetryAfter

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import TelegramChannel
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
                "_nutrition_callback_prefix": "nutrition:bound-candidate:",
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
        "nutrition:bound-candidate:0",
        "nutrition:bound-candidate:1",
        "nutrition:bound-candidate:2",
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
                channel="telegram", chat_id="123", content="text", media=[str(image)], buttons=["Да"]
            )
        )


@pytest.mark.asyncio
async def test_photo_caption_callback_uses_caption_api_and_forwards_native_id() -> None:
    channel = _channel(ReceiptBot())
    channel.config.allow_from = ["42"]
    captured = []

    async def publish(message):
        captured.append(message)

    channel.bus.publish_inbound = publish
    edited = []

    class Query:
        data = "nutrition:bound-candidate:2"
        message = SimpleNamespace(
            caption="Вы это съели?\nДата: 05.08.2026 12:00 (по EXIF фото)",
            caption_html="Вы это съели?\nДата: 05.08.2026 12:00 (по EXIF фото)",
            text=None,
            message_id=55,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Да, я это съела", callback_data="nutrition:bound-candidate:0"
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Нет, не ела", callback_data="nutrition:bound-candidate:1"
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "Это не еда", callback_data="nutrition:bound-candidate:2"
                        )
                    ],
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
    assert captured[0].content == "Это не еда"
    assert captured[0].metadata["message_id"] == 55
    assert captured[0].metadata["native_message_id"] == 55
    assert captured[0].metadata["callback_data"] == "nutrition:bound-candidate:2"
    assert captured[0].chat_id == "123"


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
            OutboundMessage(
                channel="telegram", chat_id="42", content="**important final**"
            )
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
        OutboundMessage(
            channel="telegram", chat_id="42", content="**important final**"
        )
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
        await ch.send(
            OutboundMessage(
                channel="telegram", chat_id="42", content="**final**"
            )
        )
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
