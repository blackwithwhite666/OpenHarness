"""Current Camera capture may only consume its admitted original image."""

from datetime import datetime
import asyncio

import pytest
from PIL import Image

from ohmo.attachment_store import AttachmentStore
import ohmo.gateway.runtime as runtime_module
from openharness.engine.messages import AttachmentRefBlock
from openharness.tools.base import ToolRegistry
from tests.test_ohmo.test_nutrition_dialogue_review_regressions import (
    _ScriptedEngine,
    _consumed_trace,
    _open_camera,
    _owner_message,
    _trace,
)
from tests.test_ohmo.test_nutrition_dialogue_stream import _Honcho, _pool, _turn


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("loads", "should_save", "replace_snapshot"),
    [
        (("current",), True, False),
        (("other",), False, False),
        (("other", "current"), False, False),
        (("current", "other"), False, False),
        (("missing", "current"), True, False),
        (("current", "missing"), True, False),
        (("missing",), True, False),
        (("current",), False, True),
    ],
)
async def test_loaded_image_cannot_substitute_another_source_for_current_camera(
    tmp_path, monkeypatch, loads, should_save, replace_snapshot,
):
    build = runtime_module._build_inbound_user_message
    capture = datetime.fromisoformat("2026-10-04T08:49:10+03:00")
    ingress, _, _, _, request = await _open_camera(
        tmp_path, index=810, capture_time=capture
    )
    try:
        honcho = _Honcho()
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        pool._attachment_store = AttachmentStore(pool._workspace)
        pool._test_bundle.tool_registry = ToolRegistry()
        del pool._register_conversation_image_tool
        monkeypatch.setattr(runtime_module, "_build_inbound_user_message", build)

        nutrition = _consumed_trace()["annotations"]["nutrition"]
        nutrition.update(
            basis=["image"], energy_kcal_best=215,
            items=[{"name": "selected photo meal", "quantity_text": "1 bowl"}],
        )
        engine = _ScriptedEngine(
            pool, [(None, "Старое фото принято."),
                   (_trace(nutrition), "Записала порцию: 215 ккал.")]
        )
        pool._test_bundle.engine = engine

        old_path = tmp_path / "synthetic-other-photo.jpg"
        Image.new("RGB", (9, 7), color=(201, 31, 92)).save(old_path, format="JPEG")
        old_message = _owner_message("Это другое старое фото", 9001)
        old_message.timestamp = datetime.fromisoformat("2026-10-01T20:00:00+00:00")
        old_message.media = [str(old_path)]
        await _turn(pool, old_message, ingress)
        await asyncio.gather(*pool._shadow_backend_for_scope(None)._pending)
        old_ref = next(
            block for block in engine.messages[-1].content
            if isinstance(block, AttachmentRefBlock)
        )
        assert len(honcho.messages) == 2

        answer = _owner_message("Да, я это съела", 8102)
        answer.timestamp = datetime.fromisoformat("2026-10-05T07:49:54+00:00")
        attempt = ingress._attempts[request["candidate_id"]]
        if replace_snapshot:
            Image.new("RGB", (11, 8), color=(17, 211, 63)).save(
                attempt["snapshot"], format="JPEG"
            )
        ingress.process_real_inbound(answer)
        current = build(answer, pool._attachment_store, session_key=ingress.config.session_key)
        current_ref = next(
            block for block in current.content if isinstance(block, AttachmentRefBlock)
        )
        assert current_ref.attachment_id != old_ref.attachment_id
        load_ids = {
            "current": current_ref.attachment_id,
            "other": old_ref.attachment_id,
            "missing": "0" * 64,
        }
        engine.load_attachment_ids = [load_ids[name] for name in loads]

        if not should_save:
            with pytest.raises(ValueError, match="Camera consumed meal selected an image"):
                await _turn(pool, answer, ingress)
            assert len(honcho.messages) == 2
            assert attempt.get("camera_commit") is None
            assert not any(
                message.metadata.get("camera_candidate_id") == request["candidate_id"]
                for message in honcho.messages
            )
            return

        final = await _turn(pool, answer, ingress)
        assert all(
            result.is_error is (name == "missing")
            for name, result in zip(loads, engine.loaded_results, strict=True)
        )
        assert final.text == "Записала порцию: 215 ккал.\nЗаписано. Баланс обновляется."
        assert len(honcho.messages) == 4
        event_id = final.metadata["nutrition_append_event_id"]
        assert attempt["camera_commit"]["event_id"] == event_id
        assistant = honcho.messages[-1]
        assert assistant.metadata["camera_candidate_id"] == request["candidate_id"]
        assert assistant.metadata["source_message_id"] == "8102"
        saved = assistant.metadata["decision_trace"]["annotations"]["nutrition"]
        assert datetime.fromisoformat(saved["meal_at"]) == capture
        assert saved.get("meal_date") is None
        assert saved["energy_kcal_best"] == 215
        assert saved["items"][0]["quantity_text"] == "1 bowl"
        assert sum(
            row.metadata.get("client_op_id") == attempt["camera_commit"]["client_op_id"]
            for row in honcho.messages
        ) == 1
        photo_source = assistant.metadata.get("photo_occurrence_source")
        if "current" in loads and "other" not in loads:
            assert photo_source is None or photo_source["source_message_id"] == "8102"
    finally:
        await ingress.close()
