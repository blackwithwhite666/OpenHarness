"""Compact-progress: bridge tags a compact chat's progress with ``_collapse`` and
the ``/quiet`` / ``/verbose`` commands flip + persist ``compact_progress_chats``."""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from openharness.channels.bus.events import InboundMessage
from openharness.channels.bus.queue import MessageBus

from ohmo.gateway.bridge import OhmoGatewayBridge
from ohmo.gateway.config import load_gateway_config, save_gateway_config
from ohmo.gateway.models import GatewayConfig


class _FakeRuntimePool:
    async def stream_message(self, message, session_key):
        yield SimpleNamespace(kind="progress", text="🤔…", metadata={"_progress": True})
        yield SimpleNamespace(
            kind="tool_hint", text="🛠️ Bash — a1b2", metadata={"_progress": True, "_tool_hint": True}
        )
        yield SimpleNamespace(kind="final", text="Done", metadata={})

    async def reset_session(self, session_key):  # pragma: no cover - unused here
        pass


async def _run_one(bridge: OhmoGatewayBridge, bus: MessageBus, inbound: InboundMessage, n: int):
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(inbound)
        out = [await asyncio.wait_for(bus.consume_outbound(), timeout=1.0) for _ in range(n)]
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return out


@pytest.mark.asyncio
async def test_compact_chat_tags_progress_with_collapse_but_not_final():
    bus = MessageBus()
    bridge = OhmoGatewayBridge(
        bus=bus, runtime_pool=_FakeRuntimePool(), compact_progress_chats=["555"]
    )
    inbound = InboundMessage(
        channel="telegram",
        sender_id="555|user",
        chat_id="555",
        content="hi",
        metadata={"chat_type": "private", "message_id": 7},
    )
    progress, tool_hint, final = await _run_one(bridge, bus, inbound, 3)

    assert progress.metadata.get("_collapse") is True
    assert tool_hint.metadata.get("_collapse") is True
    assert "_collapse" not in final.metadata  # final tears the status down


@pytest.mark.asyncio
async def test_non_compact_chat_leaves_progress_untagged():
    bus = MessageBus()
    bridge = OhmoGatewayBridge(
        bus=bus, runtime_pool=_FakeRuntimePool(), compact_progress_chats=["999"]
    )
    inbound = InboundMessage(
        channel="telegram",
        sender_id="555|user",
        chat_id="555",  # not in the compact set
        content="hi",
        metadata={"chat_type": "private", "message_id": 7},
    )
    progress, tool_hint, _final = await _run_one(bridge, bus, inbound, 3)

    assert "_collapse" not in progress.metadata
    assert "_collapse" not in tool_hint.metadata


@pytest.mark.asyncio
async def test_quiet_and_verbose_flip_and_persist(tmp_path):
    bus = MessageBus()
    save_gateway_config(GatewayConfig(), tmp_path)
    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=_FakeRuntimePool(), workspace=tmp_path)

    def _cmd(text: str) -> InboundMessage:
        return InboundMessage(
            channel="telegram",
            sender_id="555|user",
            chat_id="555",
            content=text,
            metadata={"chat_type": "private", "message_id": 7},
        )

    # /quiet → chat 555 added + persisted
    reply = (await _run_one(bridge, bus, _cmd("/quiet"), 1))[0]
    assert "555" in bridge._compact_chats
    assert "555" in load_gateway_config(tmp_path).compact_progress_chats
    assert "🔇" in reply.content

    # /verbose → removed + persisted
    reply = (await _run_one(bridge, bus, _cmd("/verbose"), 1))[0]
    assert "555" not in bridge._compact_chats
    assert "555" not in load_gateway_config(tmp_path).compact_progress_chats
    assert "🔊" in reply.content
