from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.manager import ChannelManager
from openharness.channels.impl.telegram import TelegramChannel, silence_telegram_token_url_loggers
from openharness.config.schema import Config, TelegramConfig


def test_silence_telegram_token_url_loggers_raises_dependency_log_levels():
    for name in ("httpx", "httpcore", "telegram.ext"):
        logging.getLogger(name).setLevel(logging.INFO)

    silence_telegram_token_url_loggers()

    for name in ("httpx", "httpcore", "telegram.ext"):
        assert logging.getLogger(name).level == logging.WARNING


@pytest.mark.asyncio
async def test_telegram_start_and_help_use_configured_bot_name():
    channel = TelegramChannel(TelegramConfig(token="token", bot_name="ohmo", allow_from=["*"]), MessageBus())
    message = SimpleNamespace(chat_id=1, reply_text=AsyncMock())
    user = SimpleNamespace(first_name="Jabin")
    update = SimpleNamespace(message=message, effective_user=user)

    await channel._on_start(update, SimpleNamespace())
    await channel._on_help(update, SimpleNamespace())

    start_text = message.reply_text.await_args_list[0].args[0]
    help_text = message.reply_text.await_args_list[1].args[0]
    assert "I'm ohmo" in start_text
    assert "ohmo commands" in help_text
    assert "nanobot" not in start_text
    assert "nanobot" not in help_text


@pytest.mark.asyncio
async def test_telegram_help_lists_provider_and_model_commands():
    channel = TelegramChannel(TelegramConfig(token="token", bot_name="ohmo", allow_from=["*"]), MessageBus())
    message = SimpleNamespace(chat_id=1, reply_text=AsyncMock())
    update = SimpleNamespace(message=message, effective_user=SimpleNamespace(first_name="Jabin"))

    await channel._on_help(update, SimpleNamespace())

    help_text = message.reply_text.await_args.args[0]
    assert "/provider — Show or switch provider profile" in help_text
    assert "/model — Show or switch model" in help_text


def test_telegram_command_menu_includes_provider_and_model():
    menu = {command.command: command.description for command in TelegramChannel.BOT_COMMANDS}

    assert menu["provider"] == "Show or switch provider profile"
    assert menu["model"] == "Show or switch model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    ["/provider", "/provider list", "/model", "/model openai/gpt-5.6-terra"],
)
async def test_telegram_forwards_provider_and_model_commands_to_bus(text):
    bus = MessageBus()
    channel = TelegramChannel(TelegramConfig(token="token", allow_from=["*"]), bus)
    message = SimpleNamespace(chat_id=42, text=text)
    user = SimpleNamespace(id=7, username=None)
    update = SimpleNamespace(message=message, effective_user=user)

    await channel._forward_command(update, SimpleNamespace())

    inbound = await bus.consume_inbound()
    assert inbound.channel == "telegram"
    assert inbound.chat_id == "42"
    assert inbound.content == text


@pytest.mark.asyncio
async def test_telegram_registers_provider_and_model_command_handlers(monkeypatch):
    import openharness.channels.impl.telegram as telegram_module

    registered: dict[str, object] = {}

    class FakeApp:
        def __init__(self):
            self.bot = SimpleNamespace(
                get_me=AsyncMock(return_value=SimpleNamespace(username="bot")),
                set_my_commands=AsyncMock(),
            )
            self.updater = SimpleNamespace(
                start_polling=AsyncMock(),
                stop=AsyncMock(),
            )

        def add_error_handler(self, handler):
            pass

        def add_handler(self, handler):
            for command in getattr(handler, "commands", set()):
                registered[command] = handler.callback

        async def initialize(self):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

        async def shutdown(self):
            pass

    class FakeBuilder:
        def token(self, _token):
            return self

        def request(self, _request):
            return self

        def get_updates_request(self, _request):
            return self

        def build(self):
            return FakeApp()

    monkeypatch.setattr(telegram_module.Application, "builder", staticmethod(lambda: FakeBuilder()))

    channel = TelegramChannel(TelegramConfig(token="token", allow_from=["*"]), MessageBus())
    start_task = asyncio.create_task(channel.start())
    try:
        for _ in range(200):
            if channel.polling_started:
                break
            await asyncio.sleep(0.01)
        assert channel.polling_started
    finally:
        channel._running = False
        await asyncio.wait_for(start_task, timeout=5)
        await channel.stop()

    assert registered["provider"] == channel._forward_command
    assert registered["model"] == channel._forward_command


@pytest.mark.asyncio
async def test_telegram_error_handler_records_last_error():
    channel = TelegramChannel(TelegramConfig(token="token", allow_from=["*"]), MessageBus())

    await channel._on_error(None, SimpleNamespace(error=RuntimeError("poll failed")))

    assert channel.last_error == "poll failed"


@pytest.mark.asyncio
async def test_channel_manager_records_start_failure_on_channel():
    bus = MessageBus()
    manager = ChannelManager(Config(), bus)

    class BrokenChannel:
        async def start(self):
            raise RuntimeError("boom")

    channel = BrokenChannel()
    await manager._start_channel("telegram", channel)  # type: ignore[arg-type]

    assert getattr(channel, "last_error") == "boom"
