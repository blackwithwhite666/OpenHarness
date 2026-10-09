"""Former Camera dialogue regressions mapped to ordinary runtime properties."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from openharness.channels.bus.events import InboundMessage

from tests.test_ohmo.test_camera_ingress import _ingress
from tests.test_ohmo.test_nutrition_dialogue_stream import (
    _Honcho,
    _consumed_payload,
    _pool,
    _script_finalization,
    _turn,
)


@pytest.mark.asyncio
async def test_valid_annotation_wins_over_old_negative_or_storage_reply_phrasing(
    tmp_path, monkeypatch,
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    _script_finalization(
        pool,
        _consumed_payload(),
        "I could not save a PNG, but the structured meal result is recorded.",
    )
    msg = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Only estimate this and explain how to save a PNG.",
        metadata={"message_id": "old-lexical-negative", "is_group": False, "_synthetic": True},
        timestamp=datetime.now(timezone.utc),
    )
    try:
        result = await _turn(pool, msg, ingress)
        assert result.metadata["nutrition_append_event_id"] == "honcho-2"
        assert honcho.messages[1].metadata["decision_trace"]["annotations"]["nutrition"][
            "consumption_status"
        ] == "consumed"
    finally:
        await ingress.close()


@pytest.mark.asyncio
async def test_no_annotation_is_not_replaced_by_a_phrase_based_meal_or_question(
    tmp_path, monkeypatch,
):
    ingress, _root, _bus, _telegram = _ingress(tmp_path)
    honcho = _Honcho()
    pool = _pool(tmp_path, ingress, honcho, monkeypatch)
    _script_finalization(pool, None, "Here is the requested information; no meal was saved.")
    msg = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Can you give information about this image?",
        metadata={"message_id": "ordinary-info-only", "is_group": False, "_synthetic": True},
        timestamp=datetime.now(timezone.utc),
    )
    try:
        result = await _turn(pool, msg, ingress)
        assert "nutrition_append_event_id" not in result.metadata
        assert all("decision_trace" not in row.metadata for row in honcho.messages)
    finally:
        await ingress.close()
