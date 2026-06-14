"""Tests for the burst-coalescer in the ohmo gateway bridge.

Mirrors the harness in test_gateway.py: an in-memory MessageBus, a fake runtime
pool, and a bridge.run() task fed via bus.publish_inbound. Determinism comes
from a small coalesce window (so _next_flush_timeout clamps to 0.05 and the loop
self-flushes quickly) plus consuming the outbound queue with a short budget —
no real 0.8s sleeps.
"""

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from openharness.channels.bus.events import InboundMessage
from openharness.channels.bus.queue import MessageBus

from ohmo.gateway.bridge import OhmoGatewayBridge, _coalesce

INTERRUPT_NOTICE = "⏹️ Остановил предыдущую задачу, перехожу к новому сообщению."
RESET_NOTICE = "🧹 Контекст сброшен — начинаю новую сессию."


def _make_bridge(bus, pool, **kw):
    return OhmoGatewayBridge(bus=bus, runtime_pool=pool, **kw)


async def _drain_until_final(bus, *, final_text, budget=2.0):
    """Collect outbound messages until the given final reply arrives.

    Returns the full list of outbounds seen (including the final). Raises on a
    per-poll timeout so a hung loop fails loudly rather than blocking forever.
    """
    collected: list = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise AssertionError(f"final {final_text!r} not seen; got {[m.content for m in collected]}")
        out = await asyncio.wait_for(bus.consume_outbound(), timeout=remaining)
        collected.append(out)
        if out.content == final_text:
            return collected


@pytest.mark.asyncio
async def test_gateway_bridge_coalesces_burst_into_single_turn():
    bus = MessageBus()
    calls: list[tuple[str, str]] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            calls.append((message.content, session_key))
            yield SimpleNamespace(kind="final", text="burst-done", metadata={"_session_key": session_key})

    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=0.05, message_coalesce_max=20)
    task = asyncio.create_task(bridge.run())
    try:
        for i in range(1, 5):
            await bus.publish_inbound(
                InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content=f"m{i}")
            )
        outbounds = await _drain_until_final(bus, final_text="burst-done")
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # Exactly one coalesced turn covering all 4 texts in order.
    assert len(calls) == 1
    assert calls[0][0] == "m1\n\nm2\n\nm3\n\nm4"
    # Exactly zero stop notices here: no prior task was running, so the single
    # coalesced dispatch had nothing to interrupt. (The in-flight interrupt case
    # — exactly ONE notice — is covered by the dedicated test above.)
    stop_notices = [m for m in outbounds if m.content == INTERRUPT_NOTICE]
    assert len(stop_notices) == 0


@pytest.mark.asyncio
async def test_gateway_bridge_coalesced_burst_interrupts_inflight_with_one_notice():
    # The headline UX guarantee: with coalescing ON, a burst that arrives while a
    # turn is already in flight interrupts it with EXACTLY ONE stop notice (not
    # zero, not N-1) — the coalesced burst is a single dispatch, so a single
    # interrupt. This is the positive mirror of the window=0 legacy test below,
    # which proves N messages → N-1 notices.
    bus = MessageBus()
    calls: list[str] = []
    first_running = asyncio.Event()
    release_first = asyncio.Event()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            calls.append(message.content)
            if message.content == "first":
                yield SimpleNamespace(kind="progress", text="🤔", metadata={"_progress": True, "_session_key": session_key})
                first_running.set()
                await release_first.wait()  # stay in flight until interrupted
                yield SimpleNamespace(kind="final", text="first-final", metadata={"_session_key": session_key})
            else:
                yield SimpleNamespace(kind="final", text="burst-final", metadata={"_session_key": session_key})

    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=0.05, message_coalesce_max=20)
    task = asyncio.create_task(bridge.run())
    try:
        # First message → its turn flushes and goes in flight.
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="first")
        )
        await asyncio.wait_for(bus.consume_outbound(), timeout=2.0)  # first progress
        await asyncio.wait_for(first_running.wait(), timeout=2.0)
        # Now a 3-message burst within the window interrupts the in-flight turn.
        for i in range(1, 4):
            await bus.publish_inbound(
                InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content=f"b{i}")
            )
        outbounds = await _drain_until_final(bus, final_text="burst-final")
    finally:
        release_first.set()
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # The burst was coalesced into ONE turn covering all three texts in order.
    assert calls == ["first", "b1\n\nb2\n\nb3"]
    # And interrupting the in-flight first turn emitted EXACTLY ONE stop notice.
    stop_notices = [m for m in outbounds if m.content == INTERRUPT_NOTICE]
    assert len(stop_notices) == 1


def test_coalesce_merges_media_across_burst_in_order():
    # The design mandates _coalesce merge/concatenate media so nothing is dropped
    # (forwarding a media block is a primary trigger for this feature). A
    # regression to media=last.media would silently lose attachments and pass the
    # text-only tests, so assert the merge directly.
    m1 = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="see this", media=["a.jpg", "b.jpg"])
    m2 = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="and this", media=["c.pdf"])
    coalesced = _coalesce([m1, m2])
    assert coalesced.content == "see this\n\nand this"
    assert coalesced.media == ["a.jpg", "b.jpg", "c.pdf"]
    # Threaded under the LAST message; the merged media is a fresh list.
    assert coalesced.media is not m1.media
    assert coalesced.media is not m2.media


def test_coalesce_single_message_passes_through_unchanged():
    only = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="solo", media=["x.png"])
    assert _coalesce([only]) is only


@pytest.mark.asyncio
async def test_gateway_bridge_coalesces_single_message_once():
    bus = MessageBus()
    calls: list[str] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            calls.append(message.content)
            yield SimpleNamespace(kind="final", text="solo-done", metadata={"_session_key": session_key})

    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=0.05)
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="hello")
        )
        await _drain_until_final(bus, final_text="solo-done")
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert calls == ["hello"]
    assert "\n\n" not in calls[0]


@pytest.mark.asyncio
async def test_gateway_bridge_coalesces_per_session_independently():
    bus = MessageBus()
    calls: list[tuple[str, str]] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            calls.append((session_key, message.content))
            yield SimpleNamespace(kind="final", text=f"done:{message.sender_id}", metadata={"_session_key": session_key})

    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=0.05, message_coalesce_max=20)
    task = asyncio.create_task(bridge.run())
    try:
        # Two senders in the same group chat → distinct session keys per router.
        for sender, text in [("uA", "a1"), ("uB", "b1"), ("uA", "a2"), ("uB", "b2")]:
            await bus.publish_inbound(
                InboundMessage(
                    channel="feishu",
                    sender_id=sender,
                    chat_id="c1",
                    content=text,
                    metadata={"chat_type": "group"},
                )
            )
        # Both sessions must produce their final; drain until both seen.
        seen_finals: set[str] = set()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 2.0
        while seen_finals != {"done:uA", "done:uB"}:
            remaining = deadline - loop.time()
            assert remaining > 0, f"only saw {seen_finals}"
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=remaining)
            if out.content.startswith("done:"):
                seen_finals.add(out.content)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert len(calls) == 2
    by_session = dict(calls)
    assert by_session["feishu:c1:uA"] == "a1\n\na2"
    assert by_session["feishu:c1:uB"] == "b1\n\nb2"


@pytest.mark.asyncio
async def test_gateway_bridge_coalesce_max_force_flushes():
    bus = MessageBus()
    calls: list[str] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            calls.append(message.content)
            yield SimpleNamespace(kind="final", text="cap-done", metadata={"_session_key": session_key})

    # Window is huge so the timer would never fire within the test; only the cap
    # can flush. Publishing exactly max=3 messages must force-flush immediately.
    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=10.0, message_coalesce_max=3)
    task = asyncio.create_task(bridge.run())
    try:
        for i in range(1, 4):
            await bus.publish_inbound(
                InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content=f"x{i}")
            )
        final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert final.content == "cap-done"
    assert calls == ["x1\n\nx2\n\nx3"]


@pytest.mark.asyncio
async def test_gateway_bridge_new_command_flushes_pending_then_resets():
    # Two plain messages get buffered+coalesced and dispatched; a /new arriving
    # after the burst's turn finished must reset the session. The /new special
    # path runs _flush_pending first (a no-op once the burst already flushed),
    # then _handle_new → reset_session, so the burst turn precedes the reset.
    bus = MessageBus()
    events: list[str] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            events.append(f"stream:{message.content}")
            yield SimpleNamespace(kind="final", text="flushed-done", metadata={"_session_key": session_key})

        async def reset_session(self, session_key):
            events.append(f"reset:{session_key}")

    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=0.05, message_coalesce_max=20)
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="p1")
        )
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="p2")
        )
        # Let the debounce window flush the coalesced burst and finish its turn.
        await _drain_until_final(bus, final_text="flushed-done")
        # Now reset the session.
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/new")
        )
        await _drain_until_final(bus, final_text=RESET_NOTICE)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # The buffered burst was streamed exactly once (coalesced), then reset once.
    # Private feishu chats keep the bare channel:chat_id session key.
    assert events == ["stream:p1\n\np2", "reset:feishu:c1"]


@pytest.mark.asyncio
async def test_gateway_bridge_synthetic_reminder_flushes_buffer_through_run_loop():
    # Real-loop check of the no-data-loss invariant. Two plain messages are
    # buffered, then a synthetic reminder for the SAME session arrives before the
    # debounce window elapses. The reminder path dispatches its own turn right
    # after flushing, so unless the run loop waits for the just-flushed buffered
    # turn to actually run, that not-yet-started task gets cancelled and the
    # buffered messages are silently dropped. Drive everything through run()/the
    # bus with the production-default window so the test reflects real scheduling
    # (the earlier direct-method test inserted an artificial yield the loop does
    # not have, masking exactly this bug).
    bus = MessageBus()
    streamed: list[str] = []

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            streamed.append(message.content)
            yield SimpleNamespace(
                kind="final", text=f"reply:{message.content}", metadata={"_session_key": session_key}
            )

        async def reset_session(self, session_key):  # pragma: no cover - unused here
            streamed.append(f"reset:{session_key}")

    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=0.8, message_coalesce_max=20)
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="p1")
        )
        await bus.publish_inbound(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="p2")
        )
        # Synthetic reminder, same session, BEFORE the 0.8s window elapses.
        await bus.publish_inbound(
            InboundMessage(
                channel="telegram",
                sender_id="__scheduler__",
                chat_id="c1",
                content="reminder!",
                session_key_override="telegram:c1",
                metadata={"_synthetic": True},
            )
        )
        await _drain_until_final(bus, final_text="reply:reminder!")
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # Both turns ran, coalesced burst FIRST then the reminder, in arrival order.
    assert streamed == ["p1\n\np2", "reminder!"]


@pytest.mark.asyncio
async def test_gateway_bridge_control_command_flushes_buffer_through_run_loop():
    # Companion to the reminder case for a control command (/new). The buffered
    # burst must still be dispatched (no data loss) before the reset runs, driven
    # end-to-end through run()/the bus. Unlike the synthetic path, /new is allowed
    # to cancel the just-flushed turn — but it is dispatched, so its interrupt
    # (and at most one stop notice) is exercised, and the reset follows it.
    bus = MessageBus()
    events: list[str] = []
    burst_streaming = asyncio.Event()
    burst_seen = asyncio.Event()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            events.append(f"stream:{message.content}")
            burst_seen.set()
            yield SimpleNamespace(kind="progress", text="🤔", metadata={"_progress": True})
            await burst_streaming.wait()  # keep the turn in flight until released
            yield SimpleNamespace(kind="final", text="burst-final", metadata={"_session_key": session_key})

        async def reset_session(self, session_key):
            events.append(f"reset:{session_key}")

    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=0.05, message_coalesce_max=20)
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="p1")
        )
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="p2")
        )
        # Wait until the coalesced burst turn is genuinely in flight, then /new.
        await asyncio.wait_for(burst_seen.wait(), timeout=2.0)
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/new")
        )
        burst_streaming.set()  # let the burst turn complete/cancel
        await _drain_until_final(bus, final_text=RESET_NOTICE)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # The buffered burst was streamed exactly once (coalesced) before the reset.
    assert events == ["stream:p1\n\np2", "reset:feishu:c1"]


@pytest.mark.asyncio
async def test_gateway_bridge_coalesce_disabled_matches_legacy_interrupts():
    # Mirror of test_gateway_bridge_new_message_interrupts_same_session, but with
    # N=3 messages and coalescing OFF: the legacy per-message interrupt behavior
    # must be faithfully preserved (N-1 = 2 stop notices).
    bus = MessageBus()
    release = asyncio.Event()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            if message.content == "first":
                try:
                    yield SimpleNamespace(kind="progress", text="🤔", metadata={"_progress": True, "_session_key": session_key})
                    await release.wait()
                except asyncio.CancelledError:
                    raise
            elif message.content == "second":
                try:
                    yield SimpleNamespace(kind="progress", text="🤔", metadata={"_progress": True, "_session_key": session_key})
                    await release.wait()
                except asyncio.CancelledError:
                    raise
            else:
                yield SimpleNamespace(kind="final", text="third-done", metadata={"_session_key": session_key})

    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=0.0)
    task = asyncio.create_task(bridge.run())
    stop_notices = 0
    try:
        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="first")
        )
        # consume the first progress update so the loop processed message 1
        await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)

        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="second")
        )
        # interrupting message 1 emits one stop notice, then message 2 progress
        notice = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert notice.content == INTERRUPT_NOTICE
        stop_notices += 1
        await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)  # second progress

        await bus.publish_inbound(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="third")
        )
        notice = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert notice.content == INTERRUPT_NOTICE
        stop_notices += 1
        final = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert final.content == "third-done"
    finally:
        release.set()
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # N=3 messages → N-1 = 2 stop notices, exactly the legacy behavior.
    assert stop_notices == 2
