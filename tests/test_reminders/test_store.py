"""CRUD + atomicity + corrupt-file tests for ReminderStore."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ohmo.reminders.model import Reminder
from ohmo.reminders.store import ReminderStore


def _reminder(rid: str, *, chat_id: str = "100", next_fire_at: float = 1000.0, **overrides) -> Reminder:
    base = dict(
        id=rid,
        channel="telegram",
        chat_id=chat_id,
        session_key=f"telegram:{chat_id}",
        created_by="42",
        created_at="2026-06-14T00:00:00+00:00",
        summary=f"ping-{rid}",
        mode="static",
        tz="Europe/Moscow",
        dtstart="2026-06-14T09:00:00+03:00",
        rrule=None,
        next_fire_at=next_fire_at,
        status="active",
        fire_count=0,
    )
    base.update(overrides)
    return Reminder(**base)


@pytest.fixture
def store() -> ReminderStore:
    return ReminderStore()


def test_empty_load(store: ReminderStore) -> None:
    assert store.load() == []


def test_add_and_get(store: ReminderStore) -> None:
    store.add(_reminder("r1"))
    assert len(store.load()) == 1
    got = store.get("r1")
    assert got is not None and got.id == "r1"


def test_list_for_chat_filters_and_sorts(store: ReminderStore) -> None:
    store.add(_reminder("a", chat_id="100", next_fire_at=300.0))
    store.add(_reminder("b", chat_id="100", next_fire_at=100.0))
    store.add(_reminder("c", chat_id="200", next_fire_at=50.0))
    store.add(_reminder("d", chat_id="100", next_fire_at=200.0, status="done"))
    listed = store.list_for_chat("telegram", "100", status="active")
    assert [r.id for r in listed] == ["b", "a"]  # sorted ascending, done excluded


def test_count_active_for_chat(store: ReminderStore) -> None:
    store.add(_reminder("a", chat_id="100"))
    store.add(_reminder("b", chat_id="100", status="paused"))
    store.add(_reminder("c", chat_id="200"))
    assert store.count_active_for_chat("telegram", "100") == 1


def test_update_replaces(store: ReminderStore) -> None:
    store.add(_reminder("r1"))
    r = store.get("r1")
    assert r is not None
    r.summary = "changed"
    assert store.update(r) is True
    assert store.get("r1").summary == "changed"


def test_update_unknown_returns_false(store: ReminderStore) -> None:
    assert store.update(_reminder("nope")) is False


def test_cancel_sets_done(store: ReminderStore) -> None:
    store.add(_reminder("r1"))
    assert store.cancel("r1") is True
    assert store.get("r1").status == "done"


def test_cancel_unknown_returns_false(store: ReminderStore) -> None:
    assert store.cancel("nope") is False


def test_mark_fired_recurring(store: ReminderStore) -> None:
    store.add(_reminder("r1", rrule="FREQ=DAILY"))
    now = time.time()
    assert store.mark_fired("r1", next_fire_at=now + 86400, fired_at=now) is True
    r = store.get("r1")
    assert r is not None
    assert r.fire_count == 1
    assert r.last_fired_at == now
    assert r.status == "active"
    assert r.next_fire_at == now + 86400


def test_mark_fired_oneshot_done(store: ReminderStore) -> None:
    store.add(_reminder("r1"))
    now = time.time()
    assert store.mark_fired("r1", next_fire_at=None, fired_at=now) is True
    r = store.get("r1")
    assert r is not None
    assert r.status == "done"
    assert r.fire_count == 1


def test_set_status_paused(store: ReminderStore) -> None:
    store.add(_reminder("r1"))
    assert store.set_status("r1", "paused") is True
    assert store.get("r1").status == "paused"


def test_corrupt_json_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = tmp_path / "reminders.json"
    bad.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr("ohmo.reminders.store.get_reminders_path", lambda workspace=None: bad)
    assert ReminderStore().load() == []


def test_non_list_json_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = tmp_path / "reminders.json"
    bad.write_text('{"not": "a list"}', encoding="utf-8")
    monkeypatch.setattr("ohmo.reminders.store.get_reminders_path", lambda workspace=None: bad)
    assert ReminderStore().load() == []


def test_atomic_write_leaves_no_temp(store: ReminderStore, tmp_path: Path) -> None:
    store.add(_reminder("r1"))
    names = {p.name for p in tmp_path.iterdir()}
    assert "reminders.json" in names
    assert not any(n.endswith(".tmp") for n in names)
    # Only the registry + optional lock file may remain.
    assert names <= {"reminders.json", "reminders.json.lock"}


def test_recipient_fields_persist(store: ReminderStore) -> None:
    store.add(
        _reminder(
            "r1",
            mode="agentic",
            recipient_chat_id="200",
            recipient_principal="200",
            recipient_label="Marina @marina",
            wellness_tenant="marina",
        )
    )
    loaded = store.get("r1")
    assert loaded is not None
    assert loaded.recipient_chat_id == "200"
    assert loaded.recipient_principal == "200"
    assert loaded.recipient_label == "Marina @marina"
    assert loaded.wellness_tenant == "marina"


def test_legacy_record_without_recipient_fields_loads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "reminders.json"
    legacy = _reminder("r1").model_dump()
    for key in (
        "recipient_chat_id",
        "recipient_principal",
        "recipient_label",
        "wellness_tenant",
    ):
        legacy.pop(key)
    path.write_text(json.dumps([legacy]), encoding="utf-8")
    monkeypatch.setattr("ohmo.reminders.store.get_reminders_path", lambda workspace=None: path)
    loaded = ReminderStore().get("r1")
    assert loaded is not None
    assert loaded.recipient_chat_id is None
    assert loaded.recipient_principal is None
    assert loaded.recipient_label is None
    assert loaded.wellness_tenant is None
    assert loaded.summary == "ping-r1"
