"""Outbound-dispatcher tests: a failed channel.send() invokes the failure hook.

The bus only enqueues (``publish_*`` never raises), so a real send failure —
e.g. a Telegram Forbidden/blocked — surfaces ONLY in the dispatcher. The
reminder scheduler relies on this hook to pause a reminder whose target blocked
the bot. These tests drive ``_dispatch_outbound`` directly via ``__new__`` so we
don't have to build a full ``Config``.
"""

from __future__ import annotations

import asyncio
import datetime
import math
import time
from types import SimpleNamespace

import pytest

from openharness.channels.bus.events import OutboundDeliveryReceipt, OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.manager import ChannelManager, _retry_after_seconds


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


async def _dispatch_one(
    manager: ChannelManager, msg: OutboundMessage, *, poll_budget: float = 0.3
) -> None:
    await manager.bus.publish_outbound(msg)
    task = asyncio.create_task(manager._dispatch_outbound())
    try:
        # Give the loop enough ticks to consume + dispatch the one message,
        # including any bounded RetryAfter retry delay. We poll until the
        # budget elapses rather than breaking on outbound_size==0, because the
        # send may still be in a RetryAfter retry sleep after consumption.
        deadline = time.monotonic() + poll_budget
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
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
async def test_telegram_todo_progress_bypasses_generic_progress_switches() -> None:
    channel = _FakeChannel(raise_on_send=False)
    manager = _manager_flags(channel, send_progress=False, send_tool_hints=False)
    todo = OutboundMessage(
        channel="telegram",
        chat_id="1",
        content="",
        metadata={
            "_progress": True,
            "progress_event": {
                "kind": "todo",
                "todos": [{"content": "A", "status": "pending"}],
                "changed": True,
            },
        },
    )

    await _dispatch_one(manager, todo)

    assert channel.sent == [todo]


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


# ---------------------------------------------------------------------------
# RetryAfter handling (bead agents-playgroud-axd)
# ---------------------------------------------------------------------------


class _RetryAfterError(Exception):
    """Duck-typed RetryAfter stand-in: the dispatcher detects ``retry_after``."""

    def __init__(self, retry_after: float):
        super().__init__(f"RetryAfter {retry_after}")
        self.retry_after = retry_after


class _RetryAfterChannel:
    """Channel that raises RetryAfter N times then succeeds for durable sends,
    and always raises for progress sends."""

    def __init__(self, *, fail_times: int = 1, retry_after: float = 0.01) -> None:
        self.fail_times = fail_times
        self.retry_after = retry_after
        self.sent: list[OutboundMessage] = []
        self.send_calls = 0
        self.receipt = OutboundDeliveryReceipt(
            channel="telegram", chat_id="100", native_message_ids=(99,)
        )

    async def send(self, msg: OutboundMessage):
        self.send_calls += 1
        is_progress = msg.metadata.get("_progress", False)
        if not is_progress and self.send_calls <= self.fail_times:
            raise _RetryAfterError(self.retry_after)
        if is_progress:
            raise _RetryAfterError(self.retry_after)
        self.sent.append(msg)
        return self.receipt


@pytest.mark.asyncio
async def test_retry_after_on_durable_message_retries_and_delivers() -> None:
    """A durable (non-progress) message that hits RetryAfter must be retried
    after the server delay and eventually delivered with a receipt."""
    channel = _RetryAfterChannel(fail_times=1, retry_after=0.01)
    manager = _manager(channel, on_send_failure=None)
    msg = OutboundMessage(channel="telegram", chat_id="100", content="final answer")

    await _dispatch_one(manager, msg, poll_budget=2.0)

    assert len(channel.sent) == 1
    assert channel.sent[0].content == "final answer"


@pytest.mark.asyncio
async def test_retry_after_on_progress_message_is_dropped_not_starving_final() -> None:
    """Progress traffic hitting RetryAfter is dropped (not retried) so it can
    never starve a subsequent durable final waiting behind it."""
    channel = _RetryAfterChannel(fail_times=0, retry_after=0.01)
    manager = _manager(channel, on_send_failure=None)
    progress = OutboundMessage(
        channel="telegram", chat_id="100", content="thinking", metadata={"_progress": True}
    )
    final = OutboundMessage(channel="telegram", chat_id="100", content="done")

    await _dispatch_one(manager, progress)
    await _dispatch_one(manager, final)

    assert channel.sent == [final]


@pytest.mark.asyncio
async def test_retry_after_per_chat_serialization_preserves_order() -> None:
    """Two durable messages for the same chat: if the first hits RetryAfter,
    the second must wait (serialized) and be delivered after, in order."""

    class _SequencedChannel:
        def __init__(self):
            self.sent: list[str] = []
            self.first_attempted = False
            self.receipt = OutboundDeliveryReceipt(
                channel="telegram", chat_id="200", native_message_ids=(1,)
            )

        async def send(self, msg: OutboundMessage):
            if not self.first_attempted:
                self.first_attempted = True
                raise _RetryAfterError(0.01)
            self.sent.append(msg.content)
            return self.receipt

    channel = _SequencedChannel()
    manager = _manager(channel, on_send_failure=None)
    msg1 = OutboundMessage(channel="telegram", chat_id="200", content="first-final")
    msg2 = OutboundMessage(channel="telegram", chat_id="200", content="second-final")

    await _dispatch_one(manager, msg1, poll_budget=2.0)
    await _dispatch_one(manager, msg2, poll_budget=2.0)

    assert channel.sent == ["first-final", "second-final"]


@pytest.mark.asyncio
async def test_retry_after_exhausted_invokes_failure_hook() -> None:
    """When all retry attempts are exhausted, the failure hook is invoked."""
    failures: list[tuple[OutboundMessage, BaseException]] = []

    async def hook(msg, error):
        failures.append((msg, error))

    channel = _RetryAfterChannel(fail_times=100, retry_after=0.01)
    manager = _manager(channel, on_send_failure=hook)
    msg = OutboundMessage(channel="telegram", chat_id="100", content="never delivered")

    await _dispatch_one(manager, msg, poll_budget=5.0)

    assert len(failures) == 1
    assert getattr(failures[0][1], "retry_after", None) == 0.01


# ---------------------------------------------------------------------------
# _retry_after_seconds: numeric + timedelta acceptance, strict rejection
# (PTB is migrating RetryAfter.retry_after from float to datetime.timedelta)
# ---------------------------------------------------------------------------


class _RawRetryAfter(Exception):
    def __init__(self, retry_after):
        super().__init__(f"RetryAfter {retry_after!r}")
        self.retry_after = retry_after


def test_retry_after_seconds_accepts_positive_int() -> None:
    assert _retry_after_seconds(_RawRetryAfter(5)) == 5.0


def test_retry_after_seconds_accepts_positive_float() -> None:
    assert _retry_after_seconds(_RawRetryAfter(1.25)) == 1.25


def test_retry_after_seconds_accepts_positive_timedelta() -> None:
    exc = _RawRetryAfter(datetime.timedelta(seconds=2, milliseconds=500))
    assert _retry_after_seconds(exc) == 2.5


def test_retry_after_seconds_rejects_zero() -> None:
    assert _retry_after_seconds(_RawRetryAfter(0)) is None
    assert _retry_after_seconds(_RawRetryAfter(datetime.timedelta(0))) is None


def test_retry_after_seconds_rejects_negative() -> None:
    assert _retry_after_seconds(_RawRetryAfter(-3)) is None
    assert _retry_after_seconds(_RawRetryAfter(-0.5)) is None
    assert _retry_after_seconds(_RawRetryAfter(datetime.timedelta(seconds=-1))) is None


def test_retry_after_seconds_rejects_malformed_values() -> None:
    assert _retry_after_seconds(_RawRetryAfter("5")) is None
    assert _retry_after_seconds(_RawRetryAfter(None)) is None
    assert _retry_after_seconds(_RawRetryAfter(True)) is None  # bool is an int
    assert _retry_after_seconds(_RawRetryAfter(float("nan"))) is None
    assert _retry_after_seconds(_RawRetryAfter(math.inf)) is None
    assert _retry_after_seconds(RuntimeError("no retry_after attr")) is None


# ---------------------------------------------------------------------------
# Per-chat worker concurrency: a RetryAfter sleep in one chat must not block
# delivery to other chats, while FIFO holds within a chat.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_after_sleep_blocks_only_that_chat() -> None:
    """Chat A hits RetryAfter and sleeps; chat B's final must be delivered
    before chat A's retry completes (cross-chat concurrency)."""

    class _CrossChatChannel:
        def __init__(self) -> None:
            self.sent: list[tuple[str, str]] = []
            self._a_failed = False

        async def send(self, msg: OutboundMessage):
            if msg.chat_id == "A" and not self._a_failed:
                self._a_failed = True
                raise _RetryAfterError(0.2)
            self.sent.append((msg.chat_id, msg.content))
            return None

    channel = _CrossChatChannel()
    manager = _manager(channel, on_send_failure=None)
    await manager.bus.publish_outbound(
        OutboundMessage(channel="telegram", chat_id="A", content="a-final")
    )
    await manager.bus.publish_outbound(
        OutboundMessage(channel="telegram", chat_id="B", content="b-final")
    )

    task = asyncio.create_task(manager._dispatch_outbound())
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and ("B", "b-final") not in channel.sent:
            await asyncio.sleep(0.01)
        # B delivered while A is still sleeping on its RetryAfter backoff.
        assert ("B", "b-final") in channel.sent
        assert ("A", "a-final") not in channel.sent

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and ("A", "a-final") not in channel.sent:
            await asyncio.sleep(0.01)
        assert ("A", "a-final") in channel.sent
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_same_chat_fifo_holds_across_retry_within_one_dispatch() -> None:
    """Two durable messages for the same chat published back-to-back: the
    first hits RetryAfter; the second must still be delivered after it."""

    class _SequencedChannel:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.first_attempted = False

        async def send(self, msg: OutboundMessage):
            if not self.first_attempted:
                self.first_attempted = True
                raise _RetryAfterError(0.01)
            self.sent.append(msg.content)
            return None

    channel = _SequencedChannel()
    manager = _manager(channel, on_send_failure=None)
    await manager.bus.publish_outbound(
        OutboundMessage(channel="telegram", chat_id="200", content="first-final")
    )
    await manager.bus.publish_outbound(
        OutboundMessage(channel="telegram", chat_id="200", content="second-final")
    )

    task = asyncio.create_task(manager._dispatch_outbound())
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(channel.sent) < 2:
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert channel.sent == ["first-final", "second-final"]


@pytest.mark.asyncio
async def test_dispatcher_cancellation_cleans_up_delivery_workers() -> None:
    """Cancelling the dispatcher must deterministically cancel per-chat
    delivery workers: no pending tasks, no leaked exceptions."""
    channel = _FakeChannel(raise_on_send=False)
    manager = _manager(channel, on_send_failure=None)
    await manager.bus.publish_outbound(
        OutboundMessage(channel="telegram", chat_id="1", content="hello")
    )

    task = asyncio.create_task(manager._dispatch_outbound())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not channel.sent:
        await asyncio.sleep(0.01)
    assert channel.sent

    workers = list(manager._delivery_workers.values())
    assert workers, "expected a live per-chat delivery worker"

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert all(w.done() for w in workers)
    assert manager._delivery_workers == {}
    assert manager._chat_queues == {}
    for w in workers:
        if not w.cancelled():
            assert w.exception() is None


@pytest.mark.asyncio
async def test_cancelled_notice_is_durable_and_retried_on_retry_after() -> None:
    """The structured cancelled notice carries no ``_progress`` flag, so it is
    durable: a RetryAfter on its standalone send must be retried, not dropped."""
    channel = _RetryAfterChannel(fail_times=1, retry_after=0.01)
    manager = _manager(channel, on_send_failure=None)
    notice = OutboundMessage(
        channel="telegram",
        chat_id="100",
        content="",
        metadata={
            "_collapse": True,
            "progress_event": {"kind": "cancelled", "reason": "stopped by user command"},
        },
    )

    await _dispatch_one(manager, notice, poll_budget=2.0)

    assert channel.sent == [notice]
