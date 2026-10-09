"""Ordinary-turn replacements for former Camera answer/replay semantics."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from openharness.channels.bus.events import InboundMessage

from tests.test_ohmo.test_camera_ingress import _ingress
from tests.test_ohmo.test_nutrition_dialogue_stream import (
    _Honcho,
    _pool,
    _script_finalization,
    _turn,
    _consumed_payload,
)


@pytest.mark.asyncio
async def test_ordinary_owner_turn_persists_model_trace_without_camera_answer_marker(
    tmp_path, monkeypatch,
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    _script_finalization(
        pool,
        _consumed_payload(),
        "The structured result is retained despite legacy classifier wording.",
    )
    message = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Only information, please; do not ask a confirmation question.",
        metadata={"message_id": "ordinary-owner-photo-context", "is_group": False, "_synthetic": True},
        timestamp=datetime.now(timezone.utc),
    )
    try:
        result = await _turn(pool, message, ingress)
        assert result.metadata["nutrition_append_event_id"] == "honcho-2"
        assert "_camera_answer" not in message.metadata
        assert honcho.messages[1].metadata["ingest_source"] == "telegram"
    finally:
        await ingress.close()


@pytest.mark.asyncio
async def test_ordinary_nonfood_response_without_annotation_creates_no_meal(
    tmp_path, monkeypatch,
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    _script_finalization(pool, None, "This is not food, so I did not save a meal.")
    message = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="What is pictured?",
        metadata={"message_id": "ordinary-nonfood-observation", "is_group": False, "_synthetic": True},
        timestamp=datetime.now(timezone.utc),
    )
    try:
        result = await _turn(pool, message, ingress)
        assert "nutrition_append_event_id" not in result.metadata
        assert not any("decision_trace" in row.metadata for row in honcho.messages)
        assert len(honcho.messages) == 0
    finally:
        await ingress.close()


@pytest.mark.asyncio
async def test_unbound_ordinary_correction_cannot_become_a_recent_meal_target(
    tmp_path, monkeypatch,
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    correction = {
        "schema_version": 1,
        "annotations": {
            "nutrition": {
                "schema_version": 2,
                "record_type": "meal_correction",
                "changed_fields": ["meal_at"],
                "meal_at": "2026-10-09T08:00:00+00:00",
            }
        },
    }
    _script_finalization(pool, correction, "I could not identify a verified source.")
    message = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Please change the date for breakfast.",
        metadata={"message_id": "ordinary-unbound-date-correction", "is_group": False, "_synthetic": True},
        timestamp=datetime.now(timezone.utc),
    )
    try:
        result = await _turn(pool, message, ingress)
        assert "nutrition_append_event_id" not in result.metadata
        assert len(honcho.messages) == 0
    finally:
        await ingress.close()
