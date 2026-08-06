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

from openharness.channels.bus.events import OutboundDeliveryReceipt, OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.manager import ChannelManager


class _FakeChannel:
    def __init__(self, *, raise_on_send: bool, receipt=None) -> None:
        self.raise_on_send = raise_on_send
        self.receipt = receipt
        self.sent: list[OutboundMessage] = []

    async def send(self, msg: OutboundMessage):
        if self.raise_on_send:
            raise RuntimeError("Forbidden: bot was blocked by the user")
        self.sent.append(msg)
        return self.receipt


def _manager(channel: _FakeChannel, *, on_send_failure) -> ChannelManager:
    # Bypass __init__ (which needs a full Config) — we only exercise dispatch.
    manager = ChannelManager.__new__(ChannelManager)
    manager.bus = MessageBus()
    manager.channels = {"telegram": channel}
    manager._on_send_failure = on_send_failure
    manager._on_send_success = None

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


def _manager_flags(channel: _FakeChannel, *, send_progress: bool, send_tool_hints: bool) -> ChannelManager:
    manager = ChannelManager.__new__(ChannelManager)
    manager.bus = MessageBus()
    manager.channels = {"telegram": channel}
    manager._on_send_failure = None
    manager._on_send_success = None

    class _Channels:
        pass

    _Channels.send_tool_hints = send_tool_hints
    _Channels.send_progress = send_progress

    class _Config:
        channels = _Channels()

    manager.config = _Config()
    return manager


@pytest.mark.asyncio
async def test_collapse_progress_bypasses_progress_and_tool_hint_drop() -> None:
    """With both global switches OFF, a normal progress/tool_hint is dropped, but a
    ``_collapse`` one still reaches the channel so the compact status can animate."""
    channel = _FakeChannel(raise_on_send=False)
    manager = _manager_flags(channel, send_progress=False, send_tool_hints=False)

    dropped = OutboundMessage(channel="telegram", chat_id="1", content="hint",
                              metadata={"_progress": True, "_tool_hint": True})
    kept = OutboundMessage(channel="telegram", chat_id="1", content="status",
                           metadata={"_progress": True, "_tool_hint": True, "_collapse": True})

    await _dispatch_one(manager, dropped)
    await _dispatch_one(manager, kept)

    assert len(channel.sent) == 1
    assert channel.sent[0].metadata.get("_collapse") is True


@pytest.mark.asyncio
async def test_success_hook_receives_optional_receipt() -> None:
    receipts = []

    async def hook(msg, receipt):
        receipts.append((msg, receipt))

    receipt = OutboundDeliveryReceipt(
        channel="telegram", chat_id="100", native_message_ids=(42,), outbound_operation_id="op-1"
    )
    channel = _FakeChannel(raise_on_send=False, receipt=receipt)
    manager = _manager(channel, on_send_failure=None)
    manager._on_send_success = hook

    await _dispatch_one(manager, OutboundMessage(channel="telegram", chat_id="100", content="hi"))

    assert len(receipts) == 1
    assert receipts[0][1] == receipt


@pytest.mark.asyncio
async def test_success_hook_failure_does_not_stop_dispatch() -> None:
    calls = []

    async def hook(msg, receipt):
        calls.append(msg.content)
        raise RuntimeError("hook failure")

    channel = _FakeChannel(raise_on_send=False)
    manager = _manager(channel, on_send_failure=None)
    manager._on_send_success = hook

    await _dispatch_one(manager, OutboundMessage(channel="telegram", chat_id="100", content="first"))
    await _dispatch_one(manager, OutboundMessage(channel="telegram", chat_id="100", content="second"))

    assert calls == ["first", "second"]
    assert [message.content for message in channel.sent] == ["first", "second"]
