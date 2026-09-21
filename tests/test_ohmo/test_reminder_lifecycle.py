"""Focused integration coverage for current-chat reminder delivery."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ohmo.gateway.bridge import OhmoGatewayBridge
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.reminders.scheduler import ReminderScheduler
from ohmo.reminders.tool import RemindCreateInput, RemindCreateTool
from ohmo.workspace import initialize_workspace
from openharness.api.usage import UsageSnapshot
from openharness.channels.bus.queue import MessageBus
from openharness.engine.messages import ConversationMessage
from openharness.engine.stream_events import AssistantTextDelta
from openharness.tools.base import ToolExecutionContext, ToolRegistry

NOW = 1_000_000.0
MSK = timezone(timedelta(hours=3))


class DeterministicReminderEngine:
    def __init__(self, decisions: deque[str], cwd: str) -> None:
        self._decisions = decisions
        self._cwd = Path(cwd)
        self.tool_metadata: dict[str, object] = {}
        self.messages: list[ConversationMessage] = []
        self.total_usage = UsageSnapshot()
        self.system_prompt = ""
        self.model = "deterministic-test-model"
        self.turn_count = 0

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    async def submit_message(self, message: ConversationMessage):
        self.messages.append(message)
        self.turn_count += 1
        yield AssistantTextDelta(text=self._decisions.popleft())


async def _wait_for_bridge_idle(
    bus: MessageBus,
    bridge: OhmoGatewayBridge,
    engines: list[DeterministicReminderEngine],
) -> None:
    async def wait() -> None:
        while not engines or bus.inbound_size or bridge._session_tasks or bridge._inflight:
            await asyncio.sleep(0)
        assert len(engines) == 1
        assert engines[0].turn_count == 1

    await asyncio.wait_for(wait(), timeout=2.0)


def _creator_context() -> dict:
    return {
        "ohmo_reminder_ctx": {
            "channel": "telegram",
            "chat_id": "100",
            "session_key": "telegram:100",
            "sender_id": "100|alice",
            "chat_type": "private",
            "is_group": False,
            "username": "alice",
            "first_name": "Alice",
            "display_name": "",
            "tz": "Europe/Moscow",
        }
    }


async def _build_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    bus = MessageBus()
    engines: list[DeterministicReminderEngine] = []
    decisions = deque(["ordinary reminder reply"])

    async def fake_build_runtime(**kwargs):
        engine = DeterministicReminderEngine(decisions, kwargs["cwd"])
        engines.append(engine)
        return SimpleNamespace(
            engine=engine,
            cwd=kwargs["cwd"],
            session_id="session-1",
            current_settings=lambda: SimpleNamespace(model="deterministic-test-model"),
            commands=SimpleNamespace(lookup=lambda raw: None),
            tool_registry=ToolRegistry(),
            enforce_max_turns=True,
            extra_skill_dirs=(),
            extra_plugin_roots=(),
            autodream_context=None,
        )

    async def fake_start_runtime(bundle) -> None:
        del bundle

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)

    pool = OhmoSessionRuntimePool(
        cwd=tmp_path,
        workspace=workspace,
        provider_profile="codex",
    )
    store = pool._reminder_store
    create = RemindCreateTool(
        store,
        pool._reminder_lock,
        default_tz="Europe/Moscow",
        max_per_chat=50,
    )
    scheduler = ReminderScheduler(
        bus=bus,
        store=store,
        lock=pool._reminder_lock,
        clock=lambda: NOW,
    )
    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=pool)
    return bus, store, create, scheduler, bridge, pool, engines


@pytest.mark.asyncio
async def test_agentic_reminder_has_one_ordinary_current_chat_bridge_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bus, store, create, scheduler, bridge, pool, engines = await _build_harness(
        tmp_path, monkeypatch
    )
    bridge_task = asyncio.create_task(bridge.run())
    try:
        created = await create.execute(
            RemindCreateInput(
                summary="check the condition",
                dtstart=(datetime.now(MSK) + timedelta(hours=1)).isoformat(),
                mode="agentic",
            ),
            ToolExecutionContext(cwd=tmp_path, metadata=_creator_context()),
        )
        assert not created.is_error, created.output
        reminder = store.list_for_chat("telegram", "100")[0]
        reminder.next_fire_at = NOW - 1
        assert store.update(reminder)

        await scheduler.fire_due()
        await _wait_for_bridge_idle(bus, bridge, engines)

        replies = [await bus.consume_outbound() for _ in range(bus.outbound_size)]
        final_replies = [reply for reply in replies if not reply.metadata.get("_progress")]
        assert len(final_replies) == 1, replies
        reply = final_replies[0]
        assert reply.channel == "telegram"
        assert reply.chat_id == "100"
        assert reply.content == "ordinary reminder reply"
        assert reply.metadata == {"_session_key": f"telegram:reminder:{reminder.id}"}
        assert bus.outbound_size == 0
        assert len(engines) == 1
        assert engines[0].turn_count == 1
    finally:
        bridge.stop()
        await bridge_task
        await pool.aclose()
