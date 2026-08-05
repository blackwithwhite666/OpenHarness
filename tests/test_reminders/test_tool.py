"""Tool behavior tests: create / list / cancel."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError


from openharness.tools.base import ToolExecutionContext

from ohmo.contact_registry import ContactStore
from ohmo.reminders.store import ReminderStore
from ohmo.reminders.tool import (
    RemindCancelInput,
    RemindCancelTool,
    RemindCreateInput,
    RemindCreateTool,
    RemindListInput,
    RemindListTool,
    WellnessTenantResolver,
)

MSK = timezone(timedelta(hours=3))


def _ctx(metadata: dict | None, tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path, metadata=metadata or {})


def _reminder_ctx(
    *,
    chat_id: str = "100",
    sender_id: str = "42",
    chat_type: str = "private",
    is_group: bool = False,
    username: str = "",
    first_name: str = "",
    display_name: str = "",
) -> dict:
    return {
        "ohmo_reminder_ctx": {
            "channel": "telegram",
            "chat_id": chat_id,
            "session_key": f"telegram:{chat_id}",
            "sender_id": sender_id,
            "chat_type": chat_type,
            "is_group": is_group,
            "username": username,
            "first_name": first_name,
            "display_name": display_name,
            "tz": "Europe/Moscow",
        }
    }


def _future_iso(hours: int = 2) -> str:
    return (datetime.now(MSK) + timedelta(hours=hours)).isoformat()


async def test_create_requires_context(tmp_path: Path) -> None:
    tool = RemindCreateTool(ReminderStore(), asyncio.Lock(), default_tz="Europe/Moscow", max_per_chat=50)
    result = await tool.execute(
        RemindCreateInput(summary="hi", dtstart=_future_iso()),
        _ctx({}, tmp_path),
    )
    assert result.is_error


async def test_create_persists_and_returns_fire_times(tmp_path: Path) -> None:
    store = ReminderStore()
    tool = RemindCreateTool(store, asyncio.Lock(), default_tz="Europe/Moscow", max_per_chat=50)
    result = await tool.execute(
        RemindCreateInput(summary="standup", dtstart=_future_iso(), rrule="FREQ=DAILY"),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert not result.is_error
    assert "Current time:" in result.output
    assert "Next fire times:" in result.output
    assert store.count_active_for_chat("telegram", "100") == 1


async def test_create_past_dtstart_oneshot_errors(tmp_path: Path) -> None:
    tool = RemindCreateTool(ReminderStore(), asyncio.Lock(), default_tz="Europe/Moscow", max_per_chat=50)
    past = (datetime.now(MSK) - timedelta(hours=1)).isoformat()
    result = await tool.execute(
        RemindCreateInput(summary="late", dtstart=past),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error
    assert "past" in result.output.lower()


async def test_create_bad_rrule_errors(tmp_path: Path) -> None:
    tool = RemindCreateTool(ReminderStore(), asyncio.Lock(), default_tz="Europe/Moscow", max_per_chat=50)
    result = await tool.execute(
        RemindCreateInput(summary="x", dtstart=_future_iso(), rrule="NOT-AN-RRULE"),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error


async def test_create_quota_enforced(tmp_path: Path) -> None:
    store = ReminderStore()
    lock = asyncio.Lock()
    tool = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=2)
    for _ in range(2):
        ok = await tool.execute(
            RemindCreateInput(summary="x", dtstart=_future_iso()),
            _ctx(_reminder_ctx(), tmp_path),
        )
        assert not ok.is_error
    overflow = await tool.execute(
        RemindCreateInput(summary="x", dtstart=_future_iso()),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert overflow.is_error
    assert "limit" in overflow.output.lower()


async def test_list_scoped_to_chat(tmp_path: Path) -> None:
    store = ReminderStore()
    lock = asyncio.Lock()
    create = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=50)
    await create.execute(
        RemindCreateInput(summary="chat-100", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="100"), tmp_path),
    )
    await create.execute(
        RemindCreateInput(summary="chat-200", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="200"), tmp_path),
    )
    list_tool = RemindListTool(store, lock, default_tz="Europe/Moscow")
    result = await list_tool.execute(RemindListInput(), _ctx(_reminder_ctx(chat_id="100"), tmp_path))
    assert "chat-100" in result.output
    assert "chat-200" not in result.output


async def test_cancel_creator_only_in_group(tmp_path: Path) -> None:
    # Telegram-realistic context: the channel emits ``is_group`` (bool) and NEVER
    # ``chat_type``. The ACL must key off ``is_group`` — keying off ``chat_type``
    # alone made this dead code on Telegram (any group member could cancel).
    store = ReminderStore()
    lock = asyncio.Lock()
    create = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=50)
    res = await create.execute(
        RemindCreateInput(summary="owned", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="500", sender_id="owner", is_group=True), tmp_path),
    )
    assert not res.is_error
    reminder_id = store.list_for_chat("telegram", "500")[0].id

    cancel = RemindCancelTool(store, lock)
    # A different group member cannot cancel.
    denied = await cancel.execute(
        RemindCancelInput(id=reminder_id),
        _ctx(_reminder_ctx(chat_id="500", sender_id="intruder", is_group=True), tmp_path),
    )
    assert denied.is_error
    assert store.get(reminder_id).status == "active"

    # The creator can.
    allowed = await cancel.execute(
        RemindCancelInput(id=reminder_id),
        _ctx(_reminder_ctx(chat_id="500", sender_id="owner", is_group=True), tmp_path),
    )
    assert not allowed.is_error
    assert store.get(reminder_id).status == "done"


async def test_cancel_creator_only_feishu_chat_type(tmp_path: Path) -> None:
    # Feishu emits ``chat_type='group'`` (no ``is_group``). The ACL must still
    # apply via the chat_type fallback.
    store = ReminderStore()
    lock = asyncio.Lock()
    create = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=50)
    await create.execute(
        RemindCreateInput(summary="owned", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="700", sender_id="owner", chat_type="group"), tmp_path),
    )
    reminder_id = store.list_for_chat("telegram", "700")[0].id

    cancel = RemindCancelTool(store, lock)
    denied = await cancel.execute(
        RemindCancelInput(id=reminder_id),
        _ctx(_reminder_ctx(chat_id="700", sender_id="intruder", chat_type="group"), tmp_path),
    )
    assert denied.is_error
    assert store.get(reminder_id).status == "active"


async def test_cancel_wrong_chat_errors(tmp_path: Path) -> None:
    store = ReminderStore()
    lock = asyncio.Lock()
    create = RemindCreateTool(store, lock, default_tz="Europe/Moscow", max_per_chat=50)
    await create.execute(
        RemindCreateInput(summary="x", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="100"), tmp_path),
    )
    reminder_id = store.list_for_chat("telegram", "100")[0].id
    cancel = RemindCancelTool(store, lock)
    result = await cancel.execute(
        RemindCancelInput(id=reminder_id),
        _ctx(_reminder_ctx(chat_id="999"), tmp_path),
    )
    assert result.is_error


def _contacts(tmp_path: Path) -> ContactStore:
    contacts = ContactStore(tmp_path)
    contacts.record_inbound(
        channel="telegram",
        chat_id="200",
        user_id="200",
        username="marina",
        first_name="Marina",
    )
    return contacts


def _wellness(**overrides) -> WellnessTenantResolver:
    values = {
        "owner_principals": ("100",),
        "family_principals": {"200": "marina"},
        "enabled_tenants": ("owner", "marina"),
    }
    values.update(overrides)
    return WellnessTenantResolver(**values)


def _create_tool(
    tmp_path: Path,
    *,
    contacts: ContactStore | None = None,
    wellness: WellnessTenantResolver | None = None,
) -> tuple[ReminderStore, RemindCreateTool]:
    store = ReminderStore()
    tool = RemindCreateTool(
        store,
        asyncio.Lock(),
        default_tz="Europe/Moscow",
        max_per_chat=50,
        contact_store=contacts,
        wellness_tenants=wellness,
    )
    return store, tool


def test_create_delivery_defaults_to_auto_and_rejects_other_values() -> None:
    reminder = RemindCreateInput(summary="x", dtstart=_future_iso())
    assert reminder.delivery == "auto"
    with pytest.raises(ValidationError):
        RemindCreateInput(summary="x", dtstart=_future_iso(), delivery="silent")


async def test_create_explicit_delivery_binds_current_private_sender(
    tmp_path: Path,
) -> None:
    # A configured wellness resolver is present, but explicit delivery neither
    # requires nor enables it. The fixed recipient comes only from current
    # authenticated gateway context; no contact lookup is needed.
    store, tool = _create_tool(tmp_path, wellness=_wellness())
    result = await tool.execute(
        RemindCreateInput(
            summary="notify only when the condition becomes true",
            dtstart=_future_iso(),
            mode="agentic",
            delivery="explicit",
        ),
        _ctx(
            _reminder_ctx(
                chat_id="100",
                sender_id="100|dmitry",
                username="dmitry",
                first_name="Dmitry",
            ),
            tmp_path,
        ),
    )

    assert not result.is_error
    assert "Recipient: Dmitry @dmitry" in result.output
    reminder = store.list_for_chat("telegram", "100")[0]
    assert reminder.mode == "agentic"
    assert reminder.recipient_chat_id == "100"
    assert reminder.recipient_principal == "100"
    assert reminder.recipient_label == "Dmitry @dmitry"
    assert reminder.wellness_tenant is None


@pytest.mark.parametrize(
    ("input_overrides", "ctx_overrides", "channel"),
    [
        pytest.param({"mode": "static"}, {}, "telegram", id="agentic_required"),
        pytest.param({}, {"is_group": True}, "telegram", id="telegram_group_flag"),
        pytest.param(
            {}, {"is_group": None}, "telegram", id="missing_private_chat_signal"
        ),
        pytest.param(
            {}, {"chat_type": "supergroup"}, "telegram", id="telegram_group_type"
        ),
        pytest.param({}, {}, "feishu", id="telegram_required"),
        pytest.param(
            {}, {"chat_id": "999"}, "telegram", id="sender_chat_mismatch"
        ),
        pytest.param(
            {},
            {"chat_id": "dmitry", "sender_id": "dmitry"},
            "telegram",
            id="numeric_sender_required",
        ),
        pytest.param(
            {"recipient": "Marina"}, {}, "telegram", id="recipient_rejected"
        ),
        pytest.param(
            {"recipient": None}, {}, "telegram", id="explicit_null_recipient_rejected"
        ),
        pytest.param(
            {"read_recipient_wellness": True},
            {},
            "telegram",
            id="wellness_rejected",
        ),
    ],
)
async def test_create_explicit_delivery_validation_boundaries(
    tmp_path: Path,
    input_overrides: dict,
    ctx_overrides: dict,
    channel: str,
) -> None:
    store, tool = _create_tool(
        tmp_path,
        contacts=_contacts(tmp_path),
        wellness=_wellness(),
    )
    values = {
        "summary": "conditional check",
        "dtstart": _future_iso(),
        "mode": "agentic",
        "delivery": "explicit",
        **input_overrides,
    }
    ctx_values = {"chat_id": "100", "sender_id": "100|dmitry", **ctx_overrides}
    ctx = _reminder_ctx(**ctx_values)
    ctx["ohmo_reminder_ctx"]["channel"] = channel

    result = await tool.execute(RemindCreateInput(**values), _ctx(ctx, tmp_path))

    assert result.is_error
    assert store.load() == []


async def test_create_with_exact_recipient_binds_and_persists(tmp_path: Path) -> None:
    store, tool = _create_tool(tmp_path, contacts=_contacts(tmp_path))
    result = await tool.execute(
        RemindCreateInput(
            summary="check on Marina",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="@marina",
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert not result.is_error
    assert "Recipient: Marina @marina" in result.output
    reminder = store.list_for_chat("telegram", "100")[0]
    assert reminder.recipient_chat_id == "200"
    assert reminder.recipient_principal == "200"
    assert reminder.recipient_label == "Marina @marina"
    assert reminder.wellness_tenant is None
    # The record stays scoped to the originating chat for list/cancel.
    assert reminder.channel == "telegram" and reminder.chat_id == "100"


async def test_create_unknown_recipient_refused(tmp_path: Path) -> None:
    store, tool = _create_tool(tmp_path, contacts=_contacts(tmp_path))
    result = await tool.execute(
        RemindCreateInput(
            summary="x",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="nobody",
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error
    assert "Unknown recipient" in result.output
    assert store.load() == []


async def test_create_ambiguous_recipient_refused(tmp_path: Path) -> None:
    contacts = ContactStore(tmp_path)
    contacts.record_inbound(
        channel="telegram", chat_id="200", user_id="200", first_name="Alex"
    )
    contacts.record_inbound(
        channel="telegram", chat_id="300", user_id="300", first_name="Alex"
    )
    store, tool = _create_tool(tmp_path, contacts=contacts)
    result = await tool.execute(
        RemindCreateInput(
            summary="x",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="Alex",
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error
    assert "Ambiguous recipient" in result.output
    assert store.load() == []


async def test_create_recipient_requires_agentic_mode(tmp_path: Path) -> None:
    store, tool = _create_tool(tmp_path, contacts=_contacts(tmp_path))
    result = await tool.execute(
        RemindCreateInput(
            summary="x",
            dtstart=_future_iso(),
            mode="static",
            recipient="Marina",
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error
    assert "agentic" in result.output
    assert store.load() == []


async def test_create_recipient_requires_telegram_channel(tmp_path: Path) -> None:
    store, tool = _create_tool(tmp_path, contacts=_contacts(tmp_path))
    ctx = _reminder_ctx()
    ctx["ohmo_reminder_ctx"]["channel"] = "feishu"
    result = await tool.execute(
        RemindCreateInput(
            summary="x",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="Marina",
        ),
        _ctx(ctx, tmp_path),
    )
    assert result.is_error
    assert "Telegram" in result.output
    assert store.load() == []


async def test_create_recipient_without_contact_store_refused(tmp_path: Path) -> None:
    store, tool = _create_tool(tmp_path, contacts=None)
    result = await tool.execute(
        RemindCreateInput(
            summary="x",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="Marina",
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error
    assert store.load() == []


async def test_wellness_self_recipient_succeeds(tmp_path: Path) -> None:
    # read_recipient_wellness=true with no explicit recipient resolves the
    # current private Telegram sender as the wellness subject. The scheduler,
    # runtime, and MCP tenant path all reuse the normal bound-reminder flow.
    store, tool = _create_tool(
        tmp_path, contacts=_contacts(tmp_path), wellness=_wellness()
    )
    result = await tool.execute(
        RemindCreateInput(
            summary="morning wellness self-check",
            dtstart=_future_iso(),
            mode="agentic",
            read_recipient_wellness=True,
        ),
        _ctx(_reminder_ctx(chat_id="100", sender_id="100|dmitry"), tmp_path),
    )
    assert not result.is_error
    assert "Recipient:" in result.output
    assert "wellness" in result.output.lower()
    reminder = store.list_for_chat("telegram", "100")[0]
    assert reminder.recipient_chat_id == "100"
    assert reminder.recipient_principal == "100"
    assert reminder.wellness_tenant == "owner"
    assert reminder.recipient_label is not None


# Base context for self-recipient cases: private Telegram chat whose numeric
# sender (100) maps to the "owner" wellness tenant. Each refusal parametrizes
# one deviation from this base.
_SELF_OK = {"chat_id": "100", "sender_id": "100|dmitry"}


@pytest.mark.parametrize(
    ("ctx_overrides", "opts"),
    [
        pytest.param({"is_group": True}, {}, id="is_group"),
        pytest.param({"chat_type": "group"}, {}, id="chat_type_group"),
        pytest.param({"chat_id": "999"}, {}, id="sender_chat_mismatch"),
        pytest.param(
            {"chat_id": "dmitry", "sender_id": "dmitry"}, {}, id="nonnumeric_sender"
        ),
        pytest.param(
            {"chat_id": "__scheduler__", "sender_id": "__scheduler__"},
            {},
            id="scheduler_sender",
        ),
        pytest.param(
            {"chat_id": "777", "sender_id": "777|stranger"}, {}, id="unmapped_tenant"
        ),
        pytest.param(
            {"chat_id": "200", "sender_id": "200|marina"},
            {"wellness_overrides": {"enabled_tenants": ("owner",)}},
            id="disabled_tenant",
        ),
        pytest.param({}, {"wellness": None}, id="no_resolver"),
        pytest.param({}, {"channel": "feishu"}, id="non_telegram"),
        pytest.param({}, {"mode": "static"}, id="static_mode"),
    ],
)
async def test_wellness_self_recipient_refused(
    tmp_path: Path, ctx_overrides: dict, opts: dict
) -> None:
    wellness = opts.get("wellness", _wellness(**opts.get("wellness_overrides", {})))
    store, tool = _create_tool(
        tmp_path, contacts=_contacts(tmp_path), wellness=wellness
    )
    ctx = _reminder_ctx(**{**_SELF_OK, **ctx_overrides})
    if "channel" in opts:
        ctx["ohmo_reminder_ctx"]["channel"] = opts["channel"]
    result = await tool.execute(
        RemindCreateInput(
            summary="x",
            dtstart=_future_iso(),
            mode=opts.get("mode", "agentic"),
            read_recipient_wellness=True,
        ),
        _ctx(ctx, tmp_path),
    )
    assert result.is_error
    assert store.load() == []


async def test_plain_reminder_without_opt_in_stays_unbound(tmp_path: Path) -> None:
    # Without read_recipient_wellness, a plain reminder (no recipient) stays
    # unbound — no recipient, no wellness tenant. This legacy path must continue
    # to fail closed for wellness at fire time.
    store, tool = _create_tool(
        tmp_path, contacts=_contacts(tmp_path), wellness=_wellness()
    )
    result = await tool.execute(
        RemindCreateInput(summary="plain legacy reminder", dtstart=_future_iso()),
        _ctx(_reminder_ctx(chat_id="100", sender_id="100|dmitry"), tmp_path),
    )
    assert not result.is_error
    reminder = store.list_for_chat("telegram", "100")[0]
    assert reminder.recipient_chat_id is None
    assert reminder.recipient_principal is None
    assert reminder.wellness_tenant is None


async def test_wellness_unmapped_recipient_refused(tmp_path: Path) -> None:
    contacts = _contacts(tmp_path)
    contacts.record_inbound(
        channel="telegram", chat_id="555", user_id="555", first_name="Stranger"
    )
    store, tool = _create_tool(tmp_path, contacts=contacts, wellness=_wellness())
    result = await tool.execute(
        RemindCreateInput(
            summary="x",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="Stranger",
            read_recipient_wellness=True,
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error
    assert store.load() == []


async def test_wellness_disabled_tenant_refused(tmp_path: Path) -> None:
    store, tool = _create_tool(
        tmp_path,
        contacts=_contacts(tmp_path),
        wellness=_wellness(enabled_tenants=("owner",)),
    )
    result = await tool.execute(
        RemindCreateInput(
            summary="x",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="Marina",
            read_recipient_wellness=True,
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert result.is_error
    assert store.load() == []


async def test_wellness_derives_marina_tenant(tmp_path: Path) -> None:
    store, tool = _create_tool(
        tmp_path, contacts=_contacts(tmp_path), wellness=_wellness()
    )
    result = await tool.execute(
        RemindCreateInput(
            summary="morning wellness check",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="Marina",
            read_recipient_wellness=True,
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert not result.is_error
    assert "wellness" in result.output.lower()
    reminder = store.list_for_chat("telegram", "100")[0]
    assert reminder.wellness_tenant == "marina"
    assert reminder.recipient_principal == "200"


async def test_wellness_owner_principal_maps_to_owner(tmp_path: Path) -> None:
    contacts = _contacts(tmp_path)
    contacts.record_inbound(
        channel="telegram", chat_id="100", user_id="100", first_name="Boss"
    )
    store, tool = _create_tool(tmp_path, contacts=contacts, wellness=_wellness())
    result = await tool.execute(
        RemindCreateInput(
            summary="x",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="Boss",
            read_recipient_wellness=True,
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert not result.is_error
    reminder = store.list_for_chat("telegram", "100")[0]
    assert reminder.wellness_tenant == "owner"


async def test_list_shows_recipient_and_wellness_presence(tmp_path: Path) -> None:
    store, tool = _create_tool(
        tmp_path, contacts=_contacts(tmp_path), wellness=_wellness()
    )
    await tool.execute(
        RemindCreateInput(
            summary="bound",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="Marina",
            read_recipient_wellness=True,
        ),
        _ctx(_reminder_ctx(), tmp_path),
    )
    await tool.execute(
        RemindCreateInput(summary="plain", dtstart=_future_iso()),
        _ctx(_reminder_ctx(), tmp_path),
    )
    list_tool = RemindListTool(store, asyncio.Lock(), default_tz="Europe/Moscow")
    result = await list_tool.execute(RemindListInput(), _ctx(_reminder_ctx(), tmp_path))
    bound_line = next(line for line in result.output.splitlines() if "bound" in line)
    plain_line = next(line for line in result.output.splitlines() if "plain" in line)
    assert "→ Marina @marina" in bound_line
    assert "reads wellness" in bound_line
    assert "→" not in plain_line
    assert "reads wellness" not in plain_line


def test_create_input_has_no_tenant_field() -> None:
    assert not any(
        "tenant" in name for name in RemindCreateInput.model_fields
    )
