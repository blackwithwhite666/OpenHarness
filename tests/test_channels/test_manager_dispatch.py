"""Outbound-dispatcher tests: a failed channel.send() invokes the failure hook.

The bus only enqueues (``publish_*`` never raises), so a real send failure —
e.g. a Telegram Forbidden/blocked — surfaces ONLY in the dispatcher. The
reminder scheduler relies on this hook to pause a reminder whose target blocked
the bot. These tests drive ``_dispatch_outbound`` directly via ``__new__`` so we
don't have to build a full ``Config``.
"""

from __future__ import annotations

import asyncio

import pytest

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.manager import ChannelManager


class _FakeChannel:
    def __init__(self, *, raise_on_send: bool) -> None:
        self.raise_on_send = raise_on_send
        self.sent: list[OutboundMessage] = []

    async def send(self, msg: OutboundMessage) -> None:
        if self.raise_on_send:
            raise RuntimeError("Forbidden: bot was blocked by the user")
        self.sent.append(msg)


def _manager(channel: _FakeChannel, *, on_send_failure) -> ChannelManager:
    # Bypass __init__ (which needs a full Config) — we only exercise dispatch.
    manager = ChannelManager.__new__(ChannelManager)
    manager.bus = MessageBus()
    manager.channels = {"telegram": channel}
    manager._on_send_failure = on_send_failure

    class _Channels:
        send_tool_hints = True
        send_progress = True

    class _Config:
        channels = _Channels()

    manager.config = _Config()
    return manager


async def _dispatch_one(manager: ChannelManager, msg: OutboundMessage) -> None:
    await manager.bus.publish_outbound(msg)
    task = asyncio.create_task(manager._dispatch_outbound())
    try:
        # Give the loop a couple of ticks to consume + dispatch the one message.
        for _ in range(50):
            await asyncio.sleep(0.005)
            if manager.bus.outbound_size == 0:
                break
        await asyncio.sleep(0.02)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_send_failure_invokes_hook_with_message_and_error() -> None:
    failures: list[tuple[OutboundMessage, BaseException]] = []

    async def hook(msg: OutboundMessage, error: BaseException) -> None:
        failures.append((msg, error))

    channel = _FakeChannel(raise_on_send=True)
    manager = _manager(channel, on_send_failure=hook)
    msg = OutboundMessage(channel="telegram", chat_id="100", content="hi", metadata={"_reminder_id": "r1"})

    await _dispatch_one(manager, msg)

    assert len(failures) == 1
    failed_msg, error = failures[0]
    assert failed_msg.metadata["_reminder_id"] == "r1"
    assert "forbidden" in str(error).lower()


@pytest.mark.asyncio
async def test_successful_send_does_not_invoke_hook() -> None:
    failures: list = []

    async def hook(msg: OutboundMessage, error: BaseException) -> None:
        failures.append((msg, error))

    channel = _FakeChannel(raise_on_send=False)
    manager = _manager(channel, on_send_failure=hook)
    msg = OutboundMessage(channel="telegram", chat_id="100", content="hi")

    await _dispatch_one(manager, msg)

    assert failures == []
    assert len(channel.sent) == 1
