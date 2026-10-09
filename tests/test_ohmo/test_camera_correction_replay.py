"""Ordinary Camera meal corrections and receipt-backed replay regressions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest

from openharness.engine.messages import AttachmentRefBlock, ConversationMessage, TextBlock
from ohmo.evals.nutrition_persistence import _fold_events
from tests.test_ohmo import test_camera_f84_joint_runtime as joint
from tests.test_ohmo.test_conversation_attachments import PNG_BYTES


def _quantity(label: str):
    return {
        "schema_version": 2,
        "record_type": "meal_correction",
        "changed_fields": ["items"],
        "items": [{"name": "synthetic meal", "quantity_text": label,
                   "energy_kcal_best": 750}],
    }


def _denial():
    return {
        "schema_version": 2,
        "record_type": "meal_correction",
        "changed_fields": ["consumption_status", "energy_kcal_best"],
        "consumption_status": "not_consumed",
        "energy_kcal_best": 0,
    }


def _events(server, source_id):
    return [
        {"event_id": row["id"], "root_source_message_id": source_id,
         "_created_at": row["created_at"],
         "annotation": row["metadata"]["decision_trace"]["annotations"]["nutrition"]}
        for row in server.rows
        if row["metadata"].get("role") == "assistant"
        and isinstance(row["metadata"].get("decision_trace", {}).get("annotations", {}).get("nutrition"), dict)
    ]


async def _ordinary_history(tmp_path, monkeypatch, *, name="ordinary-history"):
    monkeypatch.setattr(joint, "ROOT", tmp_path)
    photo = tmp_path / "ordinary-meal.png"
    photo.write_bytes(PNG_BYTES)
    pool, bundle, server, client = joint.setup(name)
    source_id = "camera-owner-source"
    message, ctx, user = joint.inbound(
        pool, source_id, "I ate the photographed meal", media=[str(photo)],
    )
    ctx = replace(ctx, is_owner=True)
    photo_ref = next(block for block in user.content if isinstance(block, AttachmentRefBlock))
    original, original_recorder = await joint.turn(
        pool, bundle, message, ctx, user,
        joint.observation(meal_at=joint.BASE.isoformat(),
                          items=[{"name": "synthetic meal", "quantity_text": "one serving",
                                  "energy_kcal_best": 750}]),
        loads=[photo_ref.attachment_id], select_source=True,
        answer="Saved the photographed meal.",
    )
    assert original.metadata["nutrition_append_event_id"] == server.rows[-1]["id"]
    return pool, bundle, server, client, photo_ref, (message, ctx, user, original, original_recorder)


async def _correction(pool, bundle, photo_ref, source_id, text, annotation, *, minutes):
    message, ctx, user = joint.inbound(
        pool, source_id, text, when=joint.BASE + timedelta(minutes=minutes),
        metadata_extra={"reply_to_message_id": "camera-owner-source"},
    )
    ctx = replace(ctx, is_owner=True)
    final, recorder = await joint.turn(
        pool, bundle, message, ctx, user, annotation,
        loads=[photo_ref.attachment_id], select_source=True,
        answer="Updated the saved meal.",
    )
    return message, ctx, user, final, recorder


@pytest.mark.asyncio
async def test_quantity_correction_newer_correction_restart_replay_then_denial(tmp_path, monkeypatch):
    pool, bundle, server, client, photo_ref, original = await _ordinary_history(tmp_path, monkeypatch)
    try:
        first = await _correction(
            pool, bundle, photo_ref, "quantity-first", "It was one piece",
            _quantity("one piece"), minutes=1,
        )
        first_rows = deepcopy(server.rows)
        second = await _correction(
            pool, bundle, photo_ref, "quantity-second", "Actually half a serving",
            _quantity("half a serving"), minutes=2,
        )
        assert first[3].metadata["nutrition_append_event_id"] != second[3].metadata["nutrition_append_event_id"]
        assert server.rows[:len(first_rows)] == first_rows
        assert server.rows[-1]["metadata"]["decision_trace"]["annotations"]["nutrition"]["items"][0]["quantity_text"] == "half a serving"
        assert server.rows[-1]["metadata"]["target_meal_id"] == server.rows[-3]["metadata"]["target_meal_id"]

        # Rebuild the engine without its in-memory conversation. The old exact
        # transport operation is recognized from its paired durable receipt.
        restarted_engine = joint.ScriptEngine()
        restarted_engine.pool = pool
        restarted_engine.messages = [
            original[2], first[2], second[2],
            ConversationMessage(role="user", event_id="later-distinct-event",
                                content=[TextBlock(text="later distinct turn")]),
        ]
        restarted_bundle = bundle.__class__(
            engine=restarted_engine, tool_registry=bundle.tool_registry,
            session_id=bundle.session_id, review_backend=bundle.review_backend,
        )
        rows_before = deepcopy(server.rows)
        updates = [update async for update in pool._stream_engine_message(
            bundle=restarted_bundle, message=first[0], session_key="telegram:123",
            user_prompt=first[2].text, user_message=first[2], turn_ctx=first[1],
            memory_scope=joint.SCOPE, recorder=None, todo_lifecycle=False,
        )]
        assert updates == []
        assert restarted_engine.observed_prompts == []
        assert server.rows == rows_before
        assert _fold_events(_events(server, "camera-owner-source"))["latest_event_id"] == second[3].metadata["nutrition_append_event_id"]

        denied = await _correction(
            pool, bundle, photo_ref, "quantity-denial", "I did not eat it",
            _denial(), minutes=3,
        )
        assert denied[3].metadata["nutrition_append_event_id"] not in {
            original[3].metadata["nutrition_append_event_id"],
            first[3].metadata["nutrition_append_event_id"],
            second[3].metadata["nutrition_append_event_id"],
        }
        assert server.rows[:len(rows_before)] == rows_before
        folded = _fold_events(_events(server, "camera-owner-source"))
        assert folded["consumed"] is False
        assert folded["energy_kcal_best"] == 0
        assert folded["latest_event_id"] == denied[3].metadata["nutrition_append_event_id"]
        assert [event["annotation"]["record_type"] for event in _events(server, "camera-owner-source")] == [
            "meal_observation", "meal_correction", "meal_correction", "meal_correction",
        ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "foreign", "wrong_operation"])
async def test_exact_old_replay_rejects_missing_or_mismatched_receipt(tmp_path, monkeypatch, failure):
    pool, bundle, server, client, photo_ref, _original = await _ordinary_history(
        tmp_path, monkeypatch, name=f"receipt-{failure}",
    )
    try:
        first = await _correction(
            pool, bundle, photo_ref, "quantity-first", "It was one piece",
            _quantity("one piece"), minutes=1,
        )
        await _correction(
            pool, bundle, photo_ref, "quantity-second", "Actually half a serving",
            _quantity("half a serving"), minutes=2,
        )
        bundle.engine.messages.append(
            ConversationMessage(role="user", event_id="later-distinct-event",
                                content=[TextBlock(text="later distinct turn")]),
        )
        row = next(row for row in server.rows if row["id"] == first[3].metadata["nutrition_append_event_id"])
        original_metadata = deepcopy(row["metadata"])
        original_rows = deepcopy(server.rows)
        if failure == "missing":
            server.rows.remove(row)
        else:
            field, value = {
                "foreign": ("tenant_id", "foreign-tenant"),
                "wrong_operation": ("client_op_id", "wrong-operation:assistant"),
            }[failure]
            row["metadata"] = {**row["metadata"], field: value}
        assert await pool._confirmed_exact_owner_replay(
            bundle=bundle, message=first[0], user_message=first[2],
            user_text=first[0].content, turn_ctx=first[1], memory_scope=joint.SCOPE,
        ) is False
        assert len(bundle.engine.observed_prompts) == 3
        if failure == "missing":
            assert len(server.rows) == len(original_rows) - 1
        else:
            assert row["metadata"] != original_metadata
        assert original_rows[-1]["id"] == server.rows[-1]["id"]
    finally:
        await client.aclose()
