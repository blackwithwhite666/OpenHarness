"""Inbound Telegram location / venue / live-location handling.

Contract: ANY inbound location (static pin, venue, live start, live edit) just
overwrites the chat's last-known location SILENTLY — it never becomes an agent
turn. A later real user turn gets that location injected as context (only if one
exists). Live-share expiry is retained and surfaced in the injected text, but
never drops the record.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import (
    _ALLOWED_UPDATES,
    TelegramChannel,
    _format_last_location,
    _format_location,
    _format_venue,
    _humanize_age,
    _live_expires_at,
    _reply_context,
)
from openharness.channels.last_location import LastLocationStore
from openharness.config.schema import TelegramConfig


def _loc(lat=59.93, lon=30.31, live_period=None, heading=None, accuracy=None):
    return SimpleNamespace(
        latitude=lat, longitude=lon, live_period=live_period,
        heading=heading, horizontal_accuracy=accuracy, proximity_alert_radius=None,
    )


# --- pure helpers ----------------------------------------------------------


def test_polling_subscribes_to_edited_message():
    # Live-location movement arrives as edited_message; without it Telegram never
    # delivers live updates.
    assert "edited_message" in _ALLOWED_UPDATES
    assert "message" in _ALLOWED_UPDATES
    assert "callback_query" in _ALLOWED_UPDATES


def test_humanize_age():
    assert _humanize_age(5) == "5s"
    assert _humanize_age(120) == "2m"
    assert _humanize_age(7200) == "2h"
    assert _humanize_age(2 * 86400) == "2d"


def test_live_expires_at_only_for_live_period():
    msg = SimpleNamespace(date=datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc))
    assert _live_expires_at(msg, _loc(live_period=None)) is None
    assert _live_expires_at(msg, _loc(live_period=3600)) == msg.date.timestamp() + 3600


def test_format_last_location_static_shows_age_no_expiry():
    rec = {"latitude": 59.93, "longitude": 30.31, "updated_at": 100.0}
    out = _format_last_location(rec, now=400.0)
    assert "last known location" in out
    assert "59.93000, 30.31000" in out
    assert "shared 5m ago" in out
    assert "live" not in out  # no expiry → no live annotation


def test_format_last_location_live_active_and_ended():
    rec = {"latitude": 1.0, "longitude": 2.0, "updated_at": 100.0, "expires_at": 4000.0}
    active = _format_last_location(rec, now=200.0)
    assert "live, expires in ~" in active
    ended = _format_last_location({**rec, "expires_at": 150.0}, now=4000.0)
    assert "live share ended ~" in ended


def test_format_last_location_with_label():
    rec = {"latitude": 1.0, "longitude": 2.0, "updated_at": 100.0, "label": "Эрмитаж"}
    assert "«Эрмитаж»" in _format_last_location(rec, now=100.0)


def test_reply_to_location_is_labelled_not_no_text():
    # The exact bug the user hit: a reply to a location showed "[no text]".
    user = SimpleNamespace(is_bot=False, first_name="Дмитрий", username="blackwithwhite")
    reply = SimpleNamespace(text=None, caption=None, from_user=user, message_id=7,
                            venue=None, location=_loc())
    prefix, meta = _reply_context(reply)
    assert "[no text]" not in prefix
    assert "location:" in meta["reply_to_text"]


def test_format_location_and_venue_for_reply_labels():
    assert "[location:" in _format_location(_loc(accuracy=12))
    venue = SimpleNamespace(title="Эрмитаж", address="Дворцовая пл., 2", location=_loc())
    assert "«Эрмитаж»" in _format_venue(venue)


# --- store -----------------------------------------------------------------


def test_store_roundtrip_keeps_expiry_and_never_drops(tmp_path):
    store = LastLocationStore(tmp_path / "loc")
    store.update("42", latitude=1.0, longitude=2.0, source="live", expires_at=999.0)
    rec = store.get("42")
    assert rec["latitude"] == 1.0 and rec["expires_at"] == 999.0
    # an expired share is still the last known location (no time-based drop)
    assert store.get("42") is not None
    store.clear("42")
    assert store.get("42") is None


def test_store_sanitizes_chat_id(tmp_path):
    store = LastLocationStore(tmp_path / "loc")
    store.update("../evil id", latitude=1.0, longitude=2.0)
    files = list((tmp_path / "loc").glob("*.json"))
    assert len(files) == 1 and "/" not in files[0].name and ".." not in files[0].name


# --- _on_message integration ----------------------------------------------


def _channel(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_CHANNEL_STATE_DIR", str(tmp_path))
    ch = TelegramChannel(TelegramConfig(token="t", allow_from=["*"]), MessageBus())
    monkeypatch.setattr(ch, "_start_typing", lambda *a, **k: None)
    sent = []

    async def _capture(**kwargs):
        sent.append(kwargs)

    monkeypatch.setattr(ch, "_handle_message", _capture)
    return ch, sent


def _user():
    return SimpleNamespace(id=116870365, username="blackwithwhite", first_name="Dmitriy", is_bot=False)


def _msg(**over):
    base = dict(
        message_id=10, chat_id=116870365, chat=SimpleNamespace(type="private"),
        text=None, caption=None, photo=None, voice=None, audio=None, document=None,
        location=None, venue=None, media_group_id=None, reply_to_message=None,
        date=datetime.now(timezone.utc),
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_static_pin_is_silent_and_stored(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    upd = SimpleNamespace(effective_user=_user(), message=_msg(location=_loc()),
                          edited_message=None, effective_message=_msg(location=_loc()))
    await ch._on_message(upd, None)
    assert sent == []  # no turn from a location
    rec = ch._last_location.get("116870365")
    assert rec and rec["source"] == "pin" and rec["expires_at"] is None


@pytest.mark.asyncio
async def test_venue_is_silent_and_stored_with_label(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    venue = SimpleNamespace(title="Бар", address="ул. Рубинштейна", location=_loc())
    m = _msg(venue=venue)
    await ch._on_message(
        SimpleNamespace(effective_user=_user(), message=m, edited_message=None, effective_message=m), None
    )
    assert sent == []
    rec = ch._last_location.get("116870365")
    assert rec["source"] == "venue" and rec["label"] == "Бар"


@pytest.mark.asyncio
async def test_live_start_and_edit_both_silent_with_expiry(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    start = _msg(location=_loc(live_period=3600))
    await ch._on_message(
        SimpleNamespace(effective_user=_user(), message=start, edited_message=None, effective_message=start), None
    )
    edit = _msg(location=_loc(lat=60.0, lon=30.5, live_period=3600))
    await ch._on_message(
        SimpleNamespace(effective_user=_user(), message=None, edited_message=edit, effective_message=edit), None
    )
    assert sent == []  # neither start nor edit ever becomes a turn
    rec = ch._last_location.get("116870365")
    assert rec["source"] == "live" and rec["latitude"] == 60.0 and rec["expires_at"] is not None


@pytest.mark.asyncio
async def test_text_with_stored_location_injects_it(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    edit = _msg(location=_loc(lat=60.0, lon=30.5, live_period=3600))
    await ch._on_message(
        SimpleNamespace(effective_user=_user(), message=None, edited_message=edit, effective_message=edit), None
    )
    txt = _msg(text="что рядом?")
    await ch._on_message(
        SimpleNamespace(effective_user=_user(), message=txt, edited_message=None, effective_message=txt), None
    )
    assert len(sent) == 1
    assert "что рядом?" in sent[0]["content"]
    assert "last known location" in sent[0]["content"]
    assert "60.00000, 30.50000" in sent[0]["content"]
    assert "live, expires in ~" in sent[0]["content"]


@pytest.mark.asyncio
async def test_text_without_stored_location_has_no_geo_noise(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    txt = _msg(text="привет")
    await ch._on_message(
        SimpleNamespace(effective_user=_user(), message=txt, edited_message=None, effective_message=txt), None
    )
    assert sent[0]["content"] == "привет"
