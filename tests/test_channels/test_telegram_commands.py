from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from telegram import Bot, Chat, Message, MessageEntity, Update, User
from telegram.ext import CommandHandler

from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl import telegram as telegram_module
from openharness.channels.impl.telegram import TelegramChannel
from openharness.config.schema import TelegramConfig


class _FakeBuilder:
    def __init__(self, app):
        self.app = app

    def token(self, _token):
        return self

    def request(self, _request):
        return self

    def get_updates_request(self, _request):
        return self

    def build(self):
        return self.app


class _FakeApplication:
    def __init__(self, channel: TelegramChannel):
        self.handlers = []
        self.bot = SimpleNamespace(
            get_me=AsyncMock(return_value=SimpleNamespace(username="ohmo")),
            set_my_commands=AsyncMock(),
        )
        self.updater = SimpleNamespace(start_polling=AsyncMock())
        self._channel = channel

    def add_error_handler(self, _handler):
        pass

    def add_handler(self, handler):
        self.handlers.append(handler)

    async def initialize(self):
        pass

    async def start(self):
        self._channel._running = False


def _command_update(command: str) -> Update:
    bot = Bot("123:abc")
    bot._bot_user = User(123, "ohmo", True, username="ohmo")
    text = f"/{command}"
    message = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=Chat(7, "private"),
        from_user=User(42, "User", False, username="user"),
        text=text,
        entities=[MessageEntity(MessageEntity.BOT_COMMAND, 0, len(text))],
    )
    message.set_bot(bot)
    return Update(update_id=1, message=message)


@pytest.mark.asyncio
async def test_telegram_control_commands_use_registered_handlers(monkeypatch):
    channel = TelegramChannel(TelegramConfig(token="token", allow_from=["*"]), MessageBus())
    app = _FakeApplication(channel)
    monkeypatch.setattr(
        telegram_module.Application,
        "builder",
        staticmethod(lambda: _FakeBuilder(app)),
    )
    channel._forward_command = AsyncMock()

    await channel.start()

    command_handlers = {
        command: handler
        for handler in app.handlers
        if isinstance(handler, CommandHandler)
        for command in handler.commands
    }
    assert command_handlers["start"].callback == channel._on_start
    assert command_handlers["help"].callback == channel._on_help

    for command in ("new", "clear", "debug", "quiet", "verbose", "stop", "restart"):
        update = _command_update(command)
        matching = [
            handler
            for handler in app.handlers
            if isinstance(handler, CommandHandler) and handler.check_update(update)
        ]
        assert len(matching) == 1
        check_result = matching[0].check_update(update)
        await matching[0].handle_update(update, app, check_result, SimpleNamespace())

    assert channel._forward_command.await_count == 7

    unknown_update = _command_update("unknown")
    assert not any(
        isinstance(handler, CommandHandler) and handler.check_update(unknown_update)
        for handler in app.handlers
    )
    assert channel._forward_command.await_count == 7
