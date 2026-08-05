"""Compact-progress defaults and per-chat overrides for the Telegram gateway."""

from __future__ import annotations

import asyncio
import contextlib
import json
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
        yield SimpleNamespace(kind="final", text="Done", metadata={"_collapse": True})

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


def test_legacy_gateway_config_keeps_opt_in_defaults(tmp_path):
    (tmp_path / "gateway.json").write_text(
        json.dumps({"compact_progress_chats": ["555"]}) + "\n",
        encoding="utf-8",
    )

    config = load_gateway_config(tmp_path)

    assert config.compact_progress_default is False
    assert config.compact_progress_chats == ["555"]
    assert config.verbose_progress_chats == []


@pytest.mark.asyncio
async def test_legacy_default_false_collapses_only_compact_chat_but_not_final():
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
async def test_default_true_collapses_unlisted_telegram_chat_but_not_final():
    bus = MessageBus()
    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=_FakeRuntimePool(),
        compact_progress_default=True,
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
    assert "_collapse" not in final.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("compact_progress_default", [False, True])
async def test_verbose_exception_stays_uncollapsed(compact_progress_default):
    bus = MessageBus()
    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=_FakeRuntimePool(),
        compact_progress_default=compact_progress_default,
        compact_progress_chats=["555"],
        verbose_progress_chats=["555"],
    )
    inbound = InboundMessage(
        channel="telegram",
        sender_id="555|user",
        chat_id="555",
        content="hi",
        metadata={"chat_type": "private", "message_id": 7},
    )
    progress, tool_hint, final = await _run_one(bridge, bus, inbound, 3)

    assert "_collapse" not in progress.metadata
    assert "_collapse" not in tool_hint.metadata
    assert "_collapse" not in final.metadata


@pytest.mark.asyncio
async def test_default_true_does_not_change_non_telegram_progress():
    bus = MessageBus()
    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=_FakeRuntimePool(),
        compact_progress_default=True,
    )
    inbound = InboundMessage(
        channel="feishu",
        sender_id="555|user",
        chat_id="555",
        content="hi",
        metadata={"chat_type": "private", "message_id": 7},
    )
    progress, tool_hint, final = await _run_one(bridge, bus, inbound, 3)

    assert "_collapse" not in progress.metadata
    assert "_collapse" not in tool_hint.metadata
    assert "_collapse" not in final.metadata


@pytest.mark.asyncio
async def test_quiet_and_verbose_update_and_persist_both_lists_under_default_true(tmp_path):
    bus = MessageBus()
    save_gateway_config(
        GatewayConfig(
            compact_progress_default=True,
            compact_progress_chats=["777", "111"],
            verbose_progress_chats=["999", "555"],
        ),
        tmp_path,
    )
    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=_FakeRuntimePool(),
        workspace=tmp_path,
        compact_progress_default=True,
        compact_progress_chats=["777", "111"],
        verbose_progress_chats=["999", "555"],
    )

    def _cmd(text: str) -> InboundMessage:
        return InboundMessage(
            channel="telegram",
            sender_id="555|user",
            chat_id="555",
            content=text,
            metadata={"chat_type": "private", "message_id": 7},
        )

    # /quiet removes the verbose exception and records an explicit compact choice.
    reply = (await _run_one(bridge, bus, _cmd("/quiet"), 1))[0]
    assert "555" in bridge._compact_chats
    assert "555" not in bridge._verbose_chats
    config = load_gateway_config(tmp_path)
    assert config.compact_progress_default is True
    assert config.compact_progress_chats == ["111", "555", "777"]
    assert config.verbose_progress_chats == ["999"]
    assert "🔇" in reply.content

    # /verbose moves the chat to the verbose exceptions, keeping lists disjoint.
    reply = (await _run_one(bridge, bus, _cmd("/verbose"), 1))[0]
    assert "555" not in bridge._compact_chats
    assert "555" in bridge._verbose_chats
    config = load_gateway_config(tmp_path)
    assert config.compact_progress_default is True
    assert config.compact_progress_chats == ["111", "777"]
    assert config.verbose_progress_chats == ["555", "999"]
    assert "🔊" in reply.content
