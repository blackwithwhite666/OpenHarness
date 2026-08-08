"""Quiet Telegram progress defaults and the per-chat debug allowlist."""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

import pytest

from ohmo.gateway.bridge import OhmoGatewayBridge
from ohmo.gateway.config import load_gateway_config, save_gateway_config
from ohmo.gateway.models import GatewayConfig
from openharness.channels.bus.events import InboundMessage
from openharness.channels.bus.queue import MessageBus


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


def test_legacy_verbose_allowlist_migrates_to_debug_allowlist(tmp_path):
    (tmp_path / "gateway.json").write_text(
        json.dumps(
            {
                "verbose_progress_chats": ["555"],
                "compact_progress_default": True,
                "compact_progress_chats": ["999"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_gateway_config(tmp_path)

    assert config.debug_progress_chats == ["555"]
    assert "verbose_progress_chats" not in config.model_dump()
    assert "compact_progress_chats" not in config.model_dump()


def test_canonical_debug_allowlist_wins_over_legacy_fields(tmp_path):
    (tmp_path / "gateway.json").write_text(
        json.dumps(
            {
                "debug_progress_chats": ["111"],
                "verbose_progress_chats": ["555"],
                "compact_progress_chats": ["999"],
                "compact_progress_default": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_gateway_config(tmp_path).debug_progress_chats == ["111"]


def test_malformed_canonical_allowlist_fails_quiet_even_with_legacy_fields(tmp_path):
    (tmp_path / "gateway.json").write_text(
        json.dumps(
            {
                "debug_progress_chats": ["111", None],
                "verbose_progress_chats": ["555"],
                "compact_progress_chats": ["999"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_gateway_config(tmp_path).debug_progress_chats == []


def test_partial_legacy_contradiction_fails_migration_quiet(tmp_path):
    (tmp_path / "gateway.json").write_text(
        json.dumps(
            {
                "verbose_progress_chats": ["555", "777"],
                "compact_progress_chats": ["777"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_gateway_config(tmp_path).debug_progress_chats == []


@pytest.mark.parametrize(
    "legacy",
    [
        {"verbose_progress_chats": "555"},
        {"verbose_progress_chats": [None]},
        {"compact_progress_default": "true", "verbose_progress_chats": ["555"]},
        {"compact_progress_chats": ["555"], "verbose_progress_chats": ["555"]},
    ],
)
def test_malformed_or_contradictory_legacy_mode_is_quiet(tmp_path, legacy):
    (tmp_path / "gateway.json").write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    assert load_gateway_config(tmp_path).debug_progress_chats == []


def test_default_progress_mode_is_quiet_without_disabling_delivery():
    config = GatewayConfig()

    assert config.send_progress is True
    assert config.debug_progress_chats == []


@pytest.mark.asyncio
async def test_debug_progress_allowlist_is_the_only_uncollapsed_override():
    bus = MessageBus()
    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=_FakeRuntimePool(),
        debug_progress_chats=["424242"],
    )

    listed = InboundMessage(
        channel="telegram",
        sender_id="424242|user",
        chat_id="424242",
        content="hi",
        metadata={"chat_type": "private", "message_id": 7},
    )
    progress, tool_hint, _final = await _run_one(bridge, bus, listed, 3)

    assert "_collapse" not in progress.metadata
    assert "_collapse" not in tool_hint.metadata


@pytest.mark.asyncio
async def test_unlisted_telegram_chat_is_quiet_even_when_sender_is_dmitriy():
    bus = MessageBus()
    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=_FakeRuntimePool())
    inbound = InboundMessage(
        channel="telegram",
        sender_id="999|dmitry",
        chat_id="999",
        content="hi",
        metadata={"chat_type": "private", "message_id": 7},
    )

    progress, tool_hint, final = await _run_one(bridge, bus, inbound, 3)

    assert progress.metadata.get("_collapse") is True
    assert tool_hint.metadata.get("_collapse") is True
    assert "_collapse" not in final.metadata


@pytest.mark.asyncio
async def test_debug_and_quiet_update_and_persist_one_canonical_list(tmp_path):
    bus = MessageBus()
    save_gateway_config(GatewayConfig(debug_progress_chats=["777", "111"]), tmp_path)
    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=_FakeRuntimePool(),
        workspace=tmp_path,
        debug_progress_chats=["777", "111"],
    )

    def _cmd(text: str) -> InboundMessage:
        return InboundMessage(
            channel="telegram",
            sender_id="555|user",
            chat_id="555",
            content=text,
            metadata={"chat_type": "private", "message_id": 7},
        )

    reply = (await _run_one(bridge, bus, _cmd("/debug"), 1))[0]
    assert "555" in bridge._debug_chats
    config = load_gateway_config(tmp_path)
    assert config.debug_progress_chats == ["111", "555", "777"]
    saved = json.loads((tmp_path / "gateway.json").read_text(encoding="utf-8"))
    assert saved["debug_progress_chats"] == ["111", "555", "777"]
    assert "compact_progress_chats" not in saved
    assert "verbose_progress_chats" not in saved
    assert "🔊" in reply.content

    reply = (await _run_one(bridge, bus, _cmd("/quiet"), 1))[0]
    assert "555" not in bridge._debug_chats
    assert load_gateway_config(tmp_path).debug_progress_chats == ["111", "777"]
    assert "🔇" in reply.content


@pytest.mark.asyncio
async def test_verbose_is_debug_compatibility_alias(tmp_path):
    bus = MessageBus()
    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=_FakeRuntimePool(),
        workspace=tmp_path,
    )
    inbound = InboundMessage(
        channel="telegram",
        sender_id="555|user",
        chat_id="555",
        content="/verbose",
        metadata={"chat_type": "private", "message_id": 7},
    )

    reply = (await _run_one(bridge, bus, inbound, 1))[0]

    assert "555" in bridge._debug_chats
    assert load_gateway_config(tmp_path).debug_progress_chats == ["555"]
    assert "🔊" in reply.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("debug_progress_chats", "collapsed"),
    [([], True), (["200"], False)],
)
async def test_unsuppressed_telegram_scheduler_turn_uses_chat_debug_policy(
    debug_progress_chats, collapsed
):
    bus = MessageBus()
    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=_FakeRuntimePool(),
        debug_progress_chats=debug_progress_chats,
    )
    inbound = InboundMessage(
        channel="telegram",
        sender_id="__scheduler__",
        chat_id="200",
        content="reminder",
        session_key_override="telegram:200",
        metadata={"_synthetic": True, "_reminder_id": "r1"},
    )

    progress, tool_hint, _final = await _run_one(bridge, bus, inbound, 3)

    if collapsed:
        assert progress.metadata.get("_collapse") is True
        assert tool_hint.metadata.get("_collapse") is True
    else:
        assert "_collapse" not in progress.metadata
        assert "_collapse" not in tool_hint.metadata


@pytest.mark.asyncio
async def test_quiet_default_does_not_change_non_telegram_progress():
    bus = MessageBus()
    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=_FakeRuntimePool())
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
