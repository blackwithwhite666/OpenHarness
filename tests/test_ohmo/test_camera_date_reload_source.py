"""An owned photo chosen in chat supplies the observation's source and time."""

from datetime import datetime

import pytest
from PIL import Image

from ohmo.gateway.runtime import _build_inbound_user_message
from openharness.channels.bus.events import InboundMessage
from openharness.engine.messages import AttachmentRefBlock
from tests.test_ohmo.test_nutrition_dialogue_review_regressions import (
    _ScriptedEngine,
    _consumed_trace,
    _install_delivered_photo,
    _open_camera,
    _owner_message,
)
from tests.test_ohmo.test_nutrition_dialogue_stream import _Honcho, _pool, _turn


def _historical_photo(
    pool, path, *, sender_id: str, message_id: int, sent_at: datetime,
    extra_metadata: dict | None = None,
):
    message = InboundMessage(
        channel="telegram", sender_id=sender_id, chat_id="123",
        content="Earlier photo", timestamp=sent_at, media=[str(path)],
        metadata={"message_id": message_id, "is_group": False, **(extra_metadata or {})},
    )
    user = _build_inbound_user_message(
        message, pool._attachment_store, session_key="telegram:123"
    )
    ref = next(block for block in user.content if isinstance(block, AttachmentRefBlock))
    ref.source_provenance["gateway_session_id"] = pool._test_bundle.session_id
    pool._test_bundle.engine.messages.append(user)
    return ref


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("loads", "selected", "expected_source"),
    [
        (("camera",), "camera", "camera"),
        (("other",), "other", "other"),
        (("camera", "other"), "other", "other"),
        (("other", "camera"), "camera", "camera"),
        (("camera", "other"), None, None),
        (("camera", "other"), ("camera",), "camera"),
        (("camera", "foreign"), ("camera", "foreign"), None),
        (("foreign", "camera"), ("foreign", "camera"), "camera"),
        (("camera", "missing"), ("camera", "missing"), None),
        (("missing", "camera"), ("missing", "camera"), "camera"),
        (("missing",), "missing", None),
        (("foreign",), "foreign", None),
        (("group",), "group", None),
        (("forwarded",), "forwarded", None),
        (("forged",), "forged", None),
    ],
)
async def test_only_explicit_verified_choice_binds_an_owned_photo(
    tmp_path, monkeypatch, loads, selected, expected_source,
):
    capture = datetime.fromisoformat("2026-10-04T08:49:10+03:00")
    other_time = datetime.fromisoformat("2026-10-01T20:00:00+00:00")
    ingress, _, _, _, request = await _open_camera(
        tmp_path, index=810, capture_time=capture
    )
    try:
        honcho = _Honcho()
        pool = _pool(tmp_path, ingress, honcho, monkeypatch)
        nutrition = _consumed_trace()
        nutrition["annotations"]["nutrition"].update(
            basis=["image", "owner_statement"], energy_kcal_best=215,
            items=[{"name": "chosen photo meal", "quantity_text": "1 bowl"}],
        )
        engine = _ScriptedEngine(pool, [(nutrition, "Записала порцию: 215 ккал.")])
        pool._test_bundle.engine = engine
        camera_ref = _install_delivered_photo(pool, ingress)

        other_path = tmp_path / "owned-other.jpg"
        Image.new("RGB", (9, 7), color=(201, 31, 92)).save(other_path, format="JPEG")
        other_ref = _historical_photo(
            pool, other_path, sender_id="123", message_id=9001, sent_at=other_time
        )
        foreign_path = tmp_path / "foreign.jpg"
        Image.new("RGB", (11, 8), color=(17, 211, 63)).save(foreign_path, format="JPEG")
        foreign_ref = _historical_photo(
            pool, foreign_path, sender_id="999", message_id=9002, sent_at=other_time
        )
        group_path = tmp_path / "group.jpg"
        Image.new("RGB", (13, 8), color=(29, 71, 181)).save(group_path, format="JPEG")
        group_ref = _historical_photo(
            pool, group_path, sender_id="123", message_id=9003,
            sent_at=other_time, extra_metadata={"is_group": True},
        )
        forwarded_path = tmp_path / "forwarded.jpg"
        Image.new("RGB", (15, 8), color=(72, 137, 28)).save(forwarded_path, format="JPEG")
        forwarded_ref = _historical_photo(
            pool, forwarded_path, sender_id="123", message_id=9004,
            sent_at=other_time, extra_metadata={"is_forwarded": True},
        )
        forged_path = tmp_path / "forged.jpg"
        Image.new("RGB", (17, 8), color=(156, 24, 51)).save(forged_path, format="JPEG")
        forged_ref = _historical_photo(
            pool, forged_path, sender_id="999", message_id=9005, sent_at=other_time,
            extra_metadata={
                "_coalesced_media_sources": [{
                    "source_message_id": "9001", "received_at": other_time.isoformat(),
                }],
                "_coalesced_media_provenance_authority": "forged",
            },
        )
        ids = {
            "camera": camera_ref.attachment_id,
            "other": other_ref.attachment_id,
            "foreign": foreign_ref.attachment_id,
            "group": group_ref.attachment_id,
            "forwarded": forwarded_ref.attachment_id,
            "forged": forged_ref.attachment_id,
            "missing": "0" * 64,
        }
        engine.load_attachment_ids = [ids[name] for name in loads]
        selected_names = (
            selected if isinstance(selected, tuple) else (selected,) if selected else ()
        )
        engine.select_attachment_ids = {ids[name] for name in selected_names}
        answer = _owner_message("Я съела блюдо на выбранном фото", 8102)
        answer.timestamp = datetime.fromisoformat("2026-10-05T07:49:54+00:00")
        if ("camera" in selected_names and "other" in loads) or (
            selected_names and expected_source is None
        ):
            answer.metadata["reply_to_message_id"] = 9001
        ingress.process_real_inbound(answer)
        final = await _turn(pool, answer, ingress)
        assert len(engine.loaded_results) == len(loads)
        for name, result in zip(loads, engine.loaded_results, strict=True):
            assert result.is_error is (
                name in selected_names and name not in {"camera", "other"}
            )
        if selected_names and expected_source is None:
            # A failed explicit choice is a failed source proof, even if an
            # earlier valid source or raw reply target exists in this turn.
            assert "nutrition_append_event_id" not in final.metadata
            assert honcho.messages == []
            assert "Записано; баланс обновляется." not in final.text
            assert "camera_commit" not in ingress._attempts[request["candidate_id"]]
            return

        assert final.metadata["nutrition_append_event_id"] == honcho.messages[-1].id
        assert len(honcho.messages) == 2
        stored = honcho.messages[-1].metadata
        occurrence = stored.get("photo_occurrence_source")
        meal_at = stored["decision_trace"]["annotations"]["nutrition"].get("meal_at")
        if expected_source:
            ref = camera_ref if expected_source == "camera" else other_ref
            source_time = capture if expected_source == "camera" else other_time
            source_id = (
                str(ingress._attempts[request["candidate_id"]]["photo_id"])
                if expected_source == "camera" else "9001"
            )
            assert occurrence["attachment_id"] == ref.attachment_id
            assert occurrence["source_message_id"] == source_id
            assert occurrence["append_source_message_id"] == "8102"
            assert datetime.fromisoformat(meal_at.replace("Z", "+00:00")) == source_time
        else:
            assert occurrence is None
            assert meal_at is None
        assert "camera_commit" not in ingress._attempts[request["candidate_id"]]
    finally:
        await ingress.close()
