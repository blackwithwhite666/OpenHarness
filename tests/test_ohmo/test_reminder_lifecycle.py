"""Deterministic integration coverage for conditional reminder delivery."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ohmo.contact_registry import ContactStore
from ohmo.gateway.bridge import OhmoGatewayBridge
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.gateway.send_message_tool import SendTelegramMessageInput
from ohmo.reminders.scheduler import ReminderScheduler
from ohmo.reminders.store import ReminderStore
from ohmo.reminders.tool import (
    RemindCancelInput,
    RemindCancelTool,
    RemindCreateInput,
    RemindCreateTool,
)
from ohmo.workspace import initialize_workspace
from openharness.api.usage import UsageSnapshot
from openharness.channels.bus.queue import MessageBus
from openharness.engine.messages import ConversationMessage
from openharness.engine.stream_events import (
    AssistantTextDelta,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.tools.base import ToolExecutionContext, ToolRegistry

NOW = 1_000_000.0
MSK = timezone(timedelta(hours=3))


class DeterministicReminderEngine:
    """Runtime double that makes only the model decision deterministic.

    The gateway still creates the real runtime pool, registers the production
    tools, runs the real scheduler and bridge, and publishes through the real
    bus. This double replaces only the LLM decision and executes selected tools
    through their production interfaces.
    """

    def __init__(self, decisions: deque[str], cwd: str, registry: ToolRegistry) -> None:
        self._decisions = decisions
        self._cwd = Path(cwd)
        self._registry = registry
        self.tool_metadata: dict[str, object] = {}
        self.messages: list[ConversationMessage] = []
        self.total_usage = UsageSnapshot()
        self.system_prompt = ""
        self.model = "deterministic-test-model"
        self.session_keys: list[str] = []
        self.reminder_contexts: list[dict[str, object]] = []
        self.send_contexts: list[dict[str, object]] = []

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    async def submit_message(self, message: ConversationMessage):
        self.messages.append(message)
        self.session_keys.append(str(self.tool_metadata["ohmo_reminder_ctx"]["session_key"]))
        self.reminder_contexts.append(dict(self.tool_metadata["ohmo_reminder_ctx"]))
        self.send_contexts.append(dict(self.tool_metadata["ohmo_send_ctx"]))
        decision = self._decisions.popleft()

        if decision == "false":
            yield AssistantTextDelta(text="Done")
            return

        if decision == "alternate":
            tool_name = "send_telegram_message"
            tool_input = {"recipient": "Bob @bob", "text": "Condition met"}
        elif decision in {"true", "missing", "stale"}:
            tool_name = "send_telegram_message"
            send_ctx = self.tool_metadata["ohmo_send_ctx"]
            tool_input = {
                "recipient": send_ctx["fixed_recipient_label"],
                "text": "Condition met",
            }
        else:  # pragma: no cover - guards accidental test setup errors
            raise AssertionError(f"unknown deterministic decision: {decision}")

        yield ToolExecutionStarted(tool_name=tool_name, tool_input=tool_input, tool_call_id="send-1")
        send_tool = self._registry.get(tool_name)
        assert send_tool is not None
        send_result = await send_tool.execute(
            SendTelegramMessageInput.model_validate(tool_input),
            ToolExecutionContext(cwd=self._cwd, metadata=dict(self.tool_metadata)),
        )
        yield ToolExecutionCompleted(
            tool_name=tool_name,
            output=send_result.output,
            is_error=send_result.is_error,
            tool_call_id="send-1",
            metadata=send_result.metadata,
        )

        if decision == "true":
            cancel_tool = self._registry.get("remind_cancel")
            assert cancel_tool is not None
            reminder_ctx = self.tool_metadata["ohmo_reminder_ctx"]
            reminder_id = self.tool_metadata["ohmo_send_ctx"]["reminder_id"]
            cancel_result = await cancel_tool.execute(
                RemindCancelInput(id=reminder_id),
                ToolExecutionContext(
                    cwd=self._cwd,
                    metadata={**self.tool_metadata, "ohmo_reminder_ctx": reminder_ctx},
                ),
            )
            assert not cancel_result.is_error, cancel_result.output

        yield AssistantTextDelta(text="Done")


async def _wait_for_bridge_idle(
    bus: MessageBus,
    bridge: OhmoGatewayBridge,
    engines: list[DeterministicReminderEngine],
    expected_turns: int,
) -> None:
    async def wait() -> None:
        while (
            bus.inbound_size
            or bridge._session_tasks
            or bridge._inflight
            or sum(len(engine.session_keys) for engine in engines) < expected_turns
        ):
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2.0)


def _creator_context(*, chat_id: str = "100", sender_id: str = "100|alice") -> dict:
    return {
        "ohmo_reminder_ctx": {
            "channel": "telegram",
            "chat_id": chat_id,
            "session_key": f"telegram:{chat_id}",
            "sender_id": sender_id,
            "chat_type": "private",
            "is_group": False,
            "username": "alice",
            "first_name": "Alice",
            "display_name": "",
            "tz": "Europe/Moscow",
        }
    }


async def _build_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decisions: list[str],
    *,
    creator_contact: bool = True,
    alternate_contact: bool = False,
) -> tuple[
    MessageBus,
    ReminderStore,
    ContactStore,
    RemindCreateTool,
    RemindCancelTool,
    ReminderScheduler,
    OhmoGatewayBridge,
    OhmoSessionRuntimePool,
    list[DeterministicReminderEngine],
]:
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    bus = MessageBus()
    contacts = ContactStore(workspace)
    if creator_contact:
        contacts.record_inbound(
            channel="telegram",
            chat_id="100",
            user_id="100",
            username="alice",
            first_name="Alice",
        )
    if alternate_contact:
        contacts.record_inbound(
            channel="telegram",
            chat_id="200",
            user_id="200",
            username="bob",
            first_name="Bob",
        )

    engines: list[DeterministicReminderEngine] = []
    decision_queue = deque(decisions)

    async def fake_build_runtime(**kwargs):
        registry = ToolRegistry()
        engine = DeterministicReminderEngine(decision_queue, kwargs["cwd"], registry)
        engines.append(engine)
        return SimpleNamespace(
            engine=engine,
            cwd=kwargs["cwd"],
            session_id=f"session-{len(engines)}",
            current_settings=lambda: SimpleNamespace(model="deterministic-test-model"),
            commands=SimpleNamespace(lookup=lambda raw: None),
            tool_registry=registry,
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
        contact_store=contacts,
        send_outbound=bus.publish_outbound,
    )
    store = pool._reminder_store
    lock = pool._reminder_lock
    create = RemindCreateTool(
        store,
        lock,
        default_tz="Europe/Moscow",
        max_per_chat=50,
        contact_store=contacts,
    )
    cancel = RemindCancelTool(store, lock)
    scheduler = ReminderScheduler(
        bus=bus,
        store=store,
        lock=lock,
        clock=lambda: NOW,
    )
    bridge = OhmoGatewayBridge(bus=bus, runtime_pool=pool)
    return bus, store, contacts, create, cancel, scheduler, bridge, pool, engines


@pytest.mark.asyncio
async def test_conditional_reminder_false_then_true_isolated_and_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        bus,
        store,
        contacts,
        create,
        _cancel,
        scheduler,
        bridge,
        pool,
        engines,
    ) = await _build_harness(tmp_path, monkeypatch, ["false", "true"])
    bridge_task = asyncio.create_task(bridge.run())
    try:
        created = await create.execute(
            RemindCreateInput(
                summary="check the condition",
                dtstart=(datetime.now(MSK) + timedelta(hours=1)).isoformat(),
                rrule="FREQ=DAILY",
                mode="agentic",
                delivery="explicit",
            ),
            ToolExecutionContext(cwd=tmp_path, metadata=_creator_context()),
        )
        assert not created.is_error, created.output
        reminder = store.list_for_chat("telegram", "100")[0]
        assert reminder.recipient_chat_id == "100"
        assert reminder.recipient_principal == "100"
        assert reminder.recipient_label == "Alice @alice"
        reminder.next_fire_at = NOW - 1
        assert store.update(reminder)

        await scheduler.fire_due()
        await _wait_for_bridge_idle(bus, bridge, engines, 1)
        assert bus.outbound_size == 0
        false_reminder = store.get(reminder.id)
        assert false_reminder.status == "active"
        assert false_reminder.fire_count == 1
        assert engines[0].session_keys == [f"telegram:reminder:{reminder.id}"]
        assert engines[0].reminder_contexts[0]["sender_id"] == "__scheduler__"
        assert engines[0].reminder_contexts[0]["session_key"] == f"telegram:reminder:{reminder.id}"
        assert engines[0].send_contexts[0]["fixed_recipient_chat_id"] == "100"
        assert engines[0].send_contexts[0]["fixed_recipient_principal"] == "100"
        assert engines[0].send_contexts[0]["fixed_recipient_label"] == "Alice @alice"

        next_reminder = store.get(reminder.id)
        next_reminder.next_fire_at = NOW - 1
        assert store.update(next_reminder)

        await scheduler.fire_due()
        await _wait_for_bridge_idle(bus, bridge, engines, 2)
        assert bus.outbound_size == 1
        outbound = await bus.consume_outbound()
        assert outbound.channel == "telegram"
        assert outbound.chat_id == "100"
        assert outbound.content == "Condition met\n\n— @alice (отправлено через бота)"
        assert outbound.metadata == {
            "_session_key": "telegram:100",
            "_origin": "send_telegram_message",
            "_reminder_id": reminder.id,
        }
        assert store.get(reminder.id).status == "done"
        assert store.get(reminder.id).fire_count == 2
        assert len(engines) == 1
        assert engines[0].session_keys == [
            f"telegram:reminder:{reminder.id}",
            f"telegram:reminder:{reminder.id}",
        ]
        assert all(
            context["session_key"] == f"telegram:reminder:{reminder.id}"
            for context in engines[0].reminder_contexts
        )
        assert "telegram:100" not in {
            session_key for engine in engines for session_key in engine.session_keys
        }
        assert bus.outbound_size == 0
        assert not store.list_for_chat("telegram", "100", status="active")
        assert contacts.get("telegram", "100") is not None
    finally:
        bridge.stop()
        await bridge_task
        await pool.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["missing", "alternate"])
async def test_conditional_reminder_rejects_untrusted_delivery_without_outbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    creator_contact = case != "missing"
    alternate_contact = case == "alternate"
    (
        bus,
        store,
        _contacts,
        create,
        cancel,
        scheduler,
        bridge,
        pool,
        _engines,
    ) = await _build_harness(
        tmp_path,
        monkeypatch,
        [case],
        creator_contact=creator_contact,
        alternate_contact=alternate_contact,
    )
    bridge_task = asyncio.create_task(bridge.run())
    try:
        created = await create.execute(
            RemindCreateInput(
                summary="check the condition",
                dtstart=(datetime.now(MSK) + timedelta(hours=1)).isoformat(),
                rrule="FREQ=DAILY",
                mode="agentic",
                delivery="explicit",
            ),
            ToolExecutionContext(cwd=tmp_path, metadata=_creator_context()),
        )
        assert not created.is_error, created.output
        reminder = store.list_for_chat("telegram", "100")[0]
        reminder.next_fire_at = NOW - 1
        assert store.update(reminder)

        await scheduler.fire_due()
        await _wait_for_bridge_idle(bus, bridge, _engines, 1)
        assert bus.outbound_size == 0
        assert store.get(reminder.id).status == "active"

        cancelled = await cancel.execute(
            RemindCancelInput(id=reminder.id),
            ToolExecutionContext(cwd=tmp_path, metadata=_creator_context()),
        )
        assert not cancelled.is_error, cancelled.output
        assert store.get(reminder.id).status == "done"
        assert not store.list_for_chat("telegram", "100", status="active")
    finally:
        bridge.stop()
        await bridge_task
        await pool.aclose()


@pytest.mark.asyncio
async def test_conditional_reminder_delivers_after_recipient_label_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        bus,
        store,
        contacts,
        create,
        _cancel,
        scheduler,
        bridge,
        pool,
        engines,
    ) = await _build_harness(
        tmp_path,
        monkeypatch,
        ["stale"],
        alternate_contact=True,
    )
    bridge_task = asyncio.create_task(bridge.run())
    try:
        created = await create.execute(
            RemindCreateInput(
                summary="check the condition",
                dtstart=(datetime.now(MSK) + timedelta(hours=1)).isoformat(),
                rrule="FREQ=DAILY",
                mode="agentic",
                delivery="explicit",
            ),
            ToolExecutionContext(cwd=tmp_path, metadata=_creator_context()),
        )
        assert not created.is_error, created.output
        reminder = store.list_for_chat("telegram", "100")[0]
        assert reminder.recipient_chat_id == "100"
        assert reminder.recipient_principal == "100"
        assert reminder.recipient_label == "Alice @alice"

        contacts.record_inbound(
            channel="telegram",
            chat_id="100",
            user_id="100",
            username="alice",
            first_name="Mallory",
        )
        reminder.next_fire_at = NOW - 1
        assert store.update(reminder)

        await scheduler.fire_due()
        await _wait_for_bridge_idle(bus, bridge, engines, 1)

        assert bus.outbound_size == 1
        outbound = await bus.consume_outbound()
        assert outbound.channel == "telegram"
        assert outbound.chat_id == "100"
        assert outbound.content == "Condition met\n\n— @alice (отправлено через бота)"
        assert outbound.metadata == {
            "_session_key": "telegram:100",
            "_origin": "send_telegram_message",
            "_reminder_id": reminder.id,
        }
        assert bus.outbound_size == 0

        delivered_reminder = store.get(reminder.id)
        assert delivered_reminder.status == "active"
        assert delivered_reminder.fire_count == 1
        assert store.list_for_chat("telegram", "100", status="active") == [
            delivered_reminder
        ]
    finally:
        bridge.stop()
        await bridge_task
        await pool.aclose()
