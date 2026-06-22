"""Inbound Telegram location / venue / live-location handling.

Covers the contract: a static pin or venue becomes inline turn text; a live-share
*start* produces one turn AND seeds the store; live-share *edits* (edited_message)
update the store SILENTLY (no agent turn); and an ordinary later turn surfaces the
current live coordinates so "what's nearby?" can be answered.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import (
    _ALLOWED_UPDATES,
    TelegramChannel,
    _format_live_current,
    _format_live_started,
    _format_location,
    _format_venue,
    _live_expires_at,
    _reply_context,
)
from openharness.channels.live_location import LiveLocationStore
from openharness.config.schema import TelegramConfig


# --- pure formatters -------------------------------------------------------


def _loc(lat=59.93, lon=30.31, live_period=None, heading=None, accuracy=None):
    return SimpleNamespace(
        latitude=lat,
        longitude=lon,
        live_period=live_period,
        heading=heading,
        horizontal_accuracy=accuracy,
        proximity_alert_radius=None,
    )


def test_polling_subscribes_to_edited_message():
    # Live-location movement arrives as edited_message; without it in
    # allowed_updates Telegram never delivers live updates.
    assert "edited_message" in _ALLOWED_UPDATES
    assert "message" in _ALLOWED_UPDATES
    assert "callback_query" in _ALLOWED_UPDATES


def test_format_static_location_includes_coords_and_accuracy():
    out = _format_location(_loc(accuracy=12))
    assert "59.93000, 30.31000" in out
    assert "±12m" in out
    assert out.startswith("[location:")


def test_format_venue_includes_title_address_coords():
    venue = SimpleNamespace(title="Эрмитаж", address="Дворцовая пл., 2", location=_loc())
    out = _format_venue(venue)
    assert "«Эрмитаж»" in out
    assert "Дворцовая пл., 2" in out
    assert "59.93000, 30.31000" in out


def test_live_expires_at_only_for_live_period():
    msg = SimpleNamespace(date=datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc))
    assert _live_expires_at(msg, _loc(live_period=None)) is None
    exp = _live_expires_at(msg, _loc(live_period=3600))
    assert exp == msg.date.timestamp() + 3600


def test_format_live_started_and_current_mention_live():
    started = _format_live_started(_loc(), expires_at=None)
    assert started.startswith("[live location started:")
    rec = {"latitude": 59.93, "longitude": 30.31, "updated_at": 100.0, "expires_at": 3700.0}
    cur = _format_live_current(rec, now=160.0)
    assert "current live location" in cur
    assert "updated 60s ago" in cur


def test_reply_to_location_is_labelled_not_no_text():
    # The exact bug the user hit: a reply to a location showed "[no text]".
    user = SimpleNamespace(is_bot=False, first_name="Дмитрий", username="blackwithwhite")
    reply = SimpleNamespace(
        text=None, caption=None, from_user=user, message_id=7,
        venue=None, location=_loc(),
    )
    prefix, meta = _reply_context(reply)
    assert "[no text]" not in prefix
    assert "location:" in meta["reply_to_text"]


# --- store -----------------------------------------------------------------


def test_live_store_roundtrip_and_expiry(tmp_path):
    store = LiveLocationStore(tmp_path / "live")
    store.update("42", latitude=1.0, longitude=2.0, expires_at=1000.0, message_id=9)
    assert store.get("42", now=500.0)["latitude"] == 1.0
    assert store.get("42", now=2000.0) is None  # expired
    store.clear("42")
    assert store.get("42", now=500.0) is None


def test_live_store_sanitizes_chat_id(tmp_path):
    store = LiveLocationStore(tmp_path / "live")
    store.update("../evil id", latitude=1.0, longitude=2.0, expires_at=1e12)
    files = list((tmp_path / "live").glob("*.json"))
    assert len(files) == 1
    assert "/" not in files[0].name and ".." not in files[0].name


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
        date=datetime.now(timezone.utc),  # so a live_period stays in the future
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_static_location_publishes_turn(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    upd = SimpleNamespace(effective_user=_user(), message=_msg(location=_loc()), edited_message=None)
    await ch._on_message(upd, None)
    assert len(sent) == 1
    assert "[location: 59.93000, 30.31000" in sent[0]["content"]
    # static pin must not be recorded as a live share
    assert ch._live_store.get("116870365") is None


@pytest.mark.asyncio
async def test_live_start_publishes_turn_and_seeds_store(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    upd = SimpleNamespace(
        effective_user=_user(), message=_msg(location=_loc(live_period=3600)), edited_message=None,
    )
    await ch._on_message(upd, None)
    assert len(sent) == 1
    assert "live location started" in sent[0]["content"]
    rec = ch._live_store.get("116870365")
    assert rec and rec["latitude"] == 59.93


@pytest.mark.asyncio
async def test_live_edit_updates_store_silently_no_turn(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    edited = _msg(location=_loc(lat=60.0, lon=30.5, live_period=3600))
    upd = SimpleNamespace(effective_user=_user(), message=None, edited_message=edited)
    await ch._on_message(upd, None)
    assert sent == []  # the load-bearing invariant: edits never become turns
    rec = ch._live_store.get("116870365")
    assert rec and rec["latitude"] == 60.0


@pytest.mark.asyncio
async def test_text_during_active_live_injects_current_coords(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    # seed an active share via an edit, then send a plain text turn
    edited = _msg(location=_loc(lat=60.0, lon=30.5, live_period=3600))
    await ch._on_message(
        SimpleNamespace(effective_user=_user(), message=None, edited_message=edited), None
    )
    await ch._on_message(
        SimpleNamespace(effective_user=_user(), message=_msg(text="что рядом?"), edited_message=None),
        None,
    )
    assert len(sent) == 1
    assert "что рядом?" in sent[0]["content"]
    assert "current live location: 60.00000, 30.50000" in sent[0]["content"]


@pytest.mark.asyncio
async def test_plain_text_without_live_has_no_geo_noise(tmp_path, monkeypatch):
    ch, sent = _channel(tmp_path, monkeypatch)
    await ch._on_message(
        SimpleNamespace(effective_user=_user(), message=_msg(text="привет"), edited_message=None), None
    )
    assert sent[0]["content"] == "привет"
