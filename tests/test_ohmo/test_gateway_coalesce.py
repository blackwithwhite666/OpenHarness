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

from ohmo.gateway.bridge import OhmoGatewayBridge

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
    # At most one stop notice (here zero, since no prior task was running).
    stop_notices = [m for m in outbounds if m.content == INTERRUPT_NOTICE]
    assert len(stop_notices) <= 1


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
async def test_gateway_bridge_special_dispatches_buffer_before_handling():
    # Direct check of the invariant: when a special message arrives with plain
    # messages still buffered, _flush_pending dispatches the buffer (coalesced
    # into one task, in arrival order) BEFORE the special handler runs. Driven
    # via the public methods so it is deterministic without loop timing.
    bus = MessageBus()
    coalesced_content: list[str] = []
    started = asyncio.Event()

    class FakeRuntimePool:
        async def stream_message(self, message, session_key):
            coalesced_content.append(message.content)
            started.set()
            await asyncio.Event().wait()  # block so the turn is "in flight"
            yield SimpleNamespace(kind="final", text="ok", metadata={"_session_key": session_key})

        async def reset_session(self, session_key):
            coalesced_content.append(f"reset:{session_key}")

    bridge = _make_bridge(bus, FakeRuntimePool(), message_coalesce_window=10.0, message_coalesce_max=20)
    # Seed the pending buffer as the run loop would for two plain messages.
    bridge._pending["feishu:c1"] = [
        InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="q1"),
        InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="q2"),
    ]
    bridge._pending_deadline["feishu:c1"] = 0.0
    try:
        # Special path step 1: flush pending so the buffer dispatches first.
        await bridge._flush_pending("feishu:c1")
        # The coalesced burst became the in-flight task for this session.
        assert "feishu:c1" in bridge._session_tasks
        await asyncio.wait_for(started.wait(), timeout=1.0)
        # Buffer was emptied; the coalesced (q1+q2) turn is what got dispatched.
        assert bridge._pending == {}
        assert coalesced_content == ["q1\n\nq2"]
        # Step 2: now the special handler runs and interrupts that turn + resets.
        await bridge._handle_new(
            InboundMessage(channel="feishu", sender_id="u1", chat_id="c1", content="/new"),
            "feishu:c1",
        )
    finally:
        bridge.stop()
        for _ in range(5):
            await asyncio.sleep(0)

    # Flush-then-handle order is honored: the coalesced burst was dispatched
    # before the reset ran (no data lost, no reordering).
    assert coalesced_content == ["q1\n\nq2", "reset:feishu:c1"]


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
