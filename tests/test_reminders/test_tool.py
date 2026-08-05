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


async def test_list_does_not_expose_wellness_scope(tmp_path: Path) -> None:
    store, create = _create_tool(
        tmp_path, contacts=_contacts(tmp_path), wellness=_wellness()
    )
    await create.execute(
        RemindCreateInput(summary="private check", dtstart=_future_iso(), mode="agentic"),
        _ctx(_reminder_ctx(chat_id="100", sender_id="100|dmitry"), tmp_path),
    )
    await create.execute(
        RemindCreateInput(
            summary="recipient check",
            dtstart=_future_iso(),
            mode="agentic",
            recipient="Marina",
        ),
        _ctx(_reminder_ctx(chat_id="100", sender_id="100|dmitry"), tmp_path),
    )

    result = await RemindListTool(
        store, asyncio.Lock(), default_tz="Europe/Moscow"
    ).execute(
        RemindListInput(),
        _ctx(_reminder_ctx(chat_id="100", sender_id="100|dmitry"), tmp_path),
    )

    assert "private check" in result.output
    assert "recipient check" in result.output
    assert "→ Marina @marina" in result.output
    assert "wellness" not in result.output
    assert "wellness_tenant" not in result.output
    assert "read_recipient_wellness" not in result.output


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


@pytest.mark.parametrize("recipient", [None, " \t "], ids=["null", "blank"])
async def test_create_explicit_delivery_accepts_null_or_blank_recipient(
    tmp_path: Path,
    recipient: str | None,
) -> None:
    # The fixed recipient comes only from current authenticated gateway context;
    # normalized null/blank model payloads do not override it.
    store, tool = _create_tool(tmp_path, wellness=_wellness())
    result = await tool.execute(
        RemindCreateInput(
            summary="notify only when the condition becomes true",
            dtstart=_future_iso(),
            rrule=None,
            mode="agentic",
            delivery="explicit",
            tz=None,
            recipient=recipient,
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
    assert reminder.wellness_tenant == "owner"


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
    store, tool = _create_tool(
        tmp_path, contacts=_contacts(tmp_path), wellness=_wellness()
    )
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
    assert reminder.wellness_tenant == "marina"
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


async def test_agentic_self_chat_derives_wellness_without_binding_recipient(
    tmp_path: Path,
) -> None:
    store, tool = _create_tool(tmp_path, wellness=_wellness())
    result = await tool.execute(
        RemindCreateInput(summary="self-check", dtstart=_future_iso(), mode="agentic"),
        _ctx(_reminder_ctx(chat_id="100", sender_id="100|dmitry"), tmp_path),
    )
    assert not result.is_error
    reminder = store.list_for_chat("telegram", "100")[0]
    assert reminder.recipient_chat_id is None
    assert reminder.recipient_principal is None
    assert reminder.wellness_tenant == "owner"


@pytest.mark.parametrize(
    "ctx_overrides",
    [
        {"is_group": True},
        {"chat_type": "group"},
        {"chat_id": "999"},
        {"chat_id": "dmitry", "sender_id": "dmitry"},
    ],
    ids=["group", "group_type", "self_chat_mismatch", "non_numeric"],
)
async def test_auto_wellness_is_not_granted_in_unsafe_contexts(
    tmp_path: Path, ctx_overrides: dict
) -> None:
    store, tool = _create_tool(tmp_path, wellness=_wellness())
    ctx = _reminder_ctx(**{"chat_id": "100", "sender_id": "100|dmitry", **ctx_overrides})
    result = await tool.execute(
        RemindCreateInput(summary="check", dtstart=_future_iso(), mode="agentic"),
        _ctx(ctx, tmp_path),
    )
    assert not result.is_error
    assert store.list_for_chat("telegram", ctx["ohmo_reminder_ctx"]["chat_id"])[0].wellness_tenant is None


async def test_static_reminder_does_not_get_wellness(tmp_path: Path) -> None:
    store, tool = _create_tool(tmp_path, wellness=_wellness())
    result = await tool.execute(
        RemindCreateInput(summary="static", dtstart=_future_iso(), mode="static"),
        _ctx(_reminder_ctx(chat_id="100", sender_id="100|dmitry"), tmp_path),
    )
    assert not result.is_error
    assert store.list_for_chat("telegram", "100")[0].wellness_tenant is None


async def test_non_telegram_auto_reminder_stays_valid_without_wellness(tmp_path: Path) -> None:
    store, tool = _create_tool(tmp_path, wellness=_wellness())
    ctx = _reminder_ctx(chat_id="100", sender_id="100|dmitry")
    ctx["ohmo_reminder_ctx"]["channel"] = "feishu"
    result = await tool.execute(
        RemindCreateInput(summary="feishu check", dtstart=_future_iso(), mode="agentic"),
        _ctx(ctx, tmp_path),
    )
    assert not result.is_error
    assert store.list_for_chat("feishu", "100")[0].wellness_tenant is None


async def test_unmapped_or_disabled_wellness_still_creates_reminder(tmp_path: Path) -> None:
    for resolver in (_wellness(family_principals={}), _wellness(enabled_tenants=("owner",))):
        store, tool = _create_tool(tmp_path, wellness=resolver)
        result = await tool.execute(
            RemindCreateInput(summary="check", dtstart=_future_iso(), mode="agentic"),
            _ctx(_reminder_ctx(chat_id="200", sender_id="200|marina"), tmp_path),
        )
        assert not result.is_error
        assert store.list_for_chat("telegram", "200")[-1].wellness_tenant is None


async def test_named_recipient_derives_fixed_recipient_wellness(tmp_path: Path) -> None:
    store, tool = _create_tool(tmp_path, contacts=_contacts(tmp_path), wellness=_wellness())
    result = await tool.execute(
        RemindCreateInput(summary="check Marina", dtstart=_future_iso(), mode="agentic", recipient="Marina"),
        _ctx(_reminder_ctx(), tmp_path),
    )
    assert not result.is_error
    reminder = store.list_for_chat("telegram", "100")[0]
    assert reminder.recipient_principal == "200"
    assert reminder.wellness_tenant == "marina"


async def test_group_named_recipient_has_no_wellness_but_creates(tmp_path: Path) -> None:
    store, tool = _create_tool(tmp_path, contacts=_contacts(tmp_path), wellness=_wellness())
    result = await tool.execute(
        RemindCreateInput(summary="check Marina", dtstart=_future_iso(), mode="agentic", recipient="Marina"),
        _ctx(_reminder_ctx(is_group=True), tmp_path),
    )
    assert not result.is_error
    assert store.list_for_chat("telegram", "100")[0].wellness_tenant is None


def test_create_input_schema_excludes_wellness_api_field() -> None:
    properties = RemindCreateInput.model_json_schema()["properties"]
    assert "read_recipient_wellness" not in properties
    assert "wellness_tenant" not in properties
