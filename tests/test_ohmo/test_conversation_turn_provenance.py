"""Trusted source/reply provenance and bounded fingerprints in Honcho metadata."""

from __future__ import annotations

import hashlib
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

from openharness.channels.bus.events import InboundMessage

from ohmo.gateway.attachment_fingerprints import PHASH_ALGORITHM
from ohmo.gateway.memory_gate import MemoryScope
from ohmo.gateway.runtime import _build_conversation_turn_metadata, _trusted_nutrition_request
from ohmo.gateway.turn_context import TurnContext
from ohmo.nutrition_ingest.trust import COORDINATOR_TRUST_TOKEN


def _turn_ctx(*, is_forwarded: bool = False) -> TurnContext:
    return TurnContext(
        principal="100",
        is_owner=True,
        is_private=True,
        channel="telegram",
        chat_id="100",
        session_id="session-100",
        is_forwarded=is_forwarded,
    )


def _scope() -> MemoryScope:
    return MemoryScope("owner", ())


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _make_rgb_png(path: Path, width: int, height: int, pixel: tuple[int, int, int]) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes(pixel) * width for _ in range(height))
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(data)
    return data


def _message(**metadata: object) -> InboundMessage:
    return InboundMessage(
        channel="telegram",
        sender_id="100",
        chat_id="100",
        content="this was breakfast on 1 August",
        metadata=dict(metadata),
        timestamp=datetime(2026, 8, 2, 9, 30, tzinfo=timezone.utc),
    )


def test_source_and_reply_ids_propagate_to_both_paired_messages() -> None:
    _, user_metadata, assistant_metadata = _build_conversation_turn_metadata(
        turn_ctx=_turn_ctx(),
        message=_message(message_id=111, reply_to_message_id=42),
        scope=_scope(),
    )

    for metadata in (user_metadata, assistant_metadata):
        assert metadata["source_message_id"] == "111"
        assert metadata["reply_to_source_message_id"] == "42"


def test_source_and_reply_ids_default_to_none_without_channel_refs() -> None:
    _, user_metadata, assistant_metadata = _build_conversation_turn_metadata(
        turn_ctx=_turn_ctx(),
        message=_message(),
        scope=_scope(),
    )

    for metadata in (user_metadata, assistant_metadata):
        assert metadata["source_message_id"] is None
        assert metadata["reply_to_source_message_id"] is None
        assert metadata["attachment_fingerprints"] == []


def test_attachment_fingerprints_are_bounded_and_path_free(tmp_path: Path) -> None:
    image_path = tmp_path / "download_token_secret.jpg"
    data = _make_rgb_png(image_path, 16, 16, (12, 200, 30))
    message = _message(message_id=7)
    message.media = [str(image_path)]

    _, user_metadata, assistant_metadata = _build_conversation_turn_metadata(
        turn_ctx=_turn_ctx(),
        message=message,
        scope=_scope(),
    )

    for metadata in (user_metadata, assistant_metadata):
        fingerprints = metadata["attachment_fingerprints"]
        assert len(fingerprints) == 1
        descriptor = fingerprints[0]
        assert descriptor["sha256"] == hashlib.sha256(data).hexdigest()
        assert (descriptor["width"], descriptor["height"]) == (16, 16)
        assert descriptor["phash_algorithm"] == PHASH_ALGORITHM
        assert set(descriptor.keys()) <= {
            "sha256",
            "width",
            "height",
            "phash",
            "phash_algorithm",
        }
        serialized = repr(descriptor)
        assert str(tmp_path) not in serialized
        assert "download_token_secret" not in serialized


def test_model_authored_identity_keys_cannot_override_gateway_metadata() -> None:
    """Identity fields come from the trusted gateway; channel payload content
    with lookalike keys never reaches the paired Honcho metadata."""
    message = _message(
        message_id=111,
        reply_to_message_id=42,
        source_message_id="forged",
        reply_to_source_message_id="forged",
        attachment_fingerprints=[{"sha256": "forged"}],
    )

    _, user_metadata, assistant_metadata = _build_conversation_turn_metadata(
        turn_ctx=_turn_ctx(),
        message=message,
        scope=_scope(),
    )

    for metadata in (user_metadata, assistant_metadata):
        assert metadata["source_message_id"] == "111"
        assert metadata["reply_to_source_message_id"] == "42"
        assert metadata["attachment_fingerprints"] == []


def test_receive_and_forward_semantics_are_unchanged() -> None:
    source_message_at = "2026-07-30T20:15:00+03:00"

    _, user_metadata, _ = _build_conversation_turn_metadata(
        turn_ctx=_turn_ctx(is_forwarded=True),
        message=_message(message_id=5, source_message_at=source_message_at),
        scope=_scope(),
    )

    assert user_metadata["is_forwarded"] is True
    assert user_metadata["source_message_at"] == "2026-07-30T17:15:00+00:00"
    assert user_metadata["received_at"] == "2026-08-02T09:30:00+00:00"


def test_ordinary_telegram_metadata_cannot_stamp_trusted_nutrition_provenance() -> None:
    candidate = "dropbox-camera-v1-" + "e" * 64
    message = _message(
        _nutrition_trusted=True,
        _nutrition_trust_token="copied-spelling",
        _nutrition_candidate_id=candidate,
        _nutrition_client_op_id=f"{candidate}:meal-observation:v1",
        _nutrition_phase="estimation",
        _nutrition_principal="100",
        _nutrition_tenant_id="marina",
        _nutrition_chat_id="100",
        _nutrition_session_key="telegram:100",
    )
    assert _trusted_nutrition_request(message) is None
    _, user_metadata, assistant_metadata = _build_conversation_turn_metadata(
        turn_ctx=_turn_ctx(), message=message, scope=_scope()
    )
    assert "_nutrition_trusted" not in user_metadata
    assert assistant_metadata["client_op_id"].endswith(":assistant")


def test_only_process_local_token_can_mark_a_synthetic_request() -> None:
    candidate = "dropbox-camera-v1-" + "f" * 64
    message = _message(
        _nutrition_trusted=True,
        _nutrition_trust_token=COORDINATOR_TRUST_TOKEN,
        _nutrition_candidate_id=candidate,
        _nutrition_client_op_id=f"{candidate}:meal-observation:v1",
        _nutrition_phase="estimation",
        _nutrition_principal="100",
        _nutrition_tenant_id="marina",
        _nutrition_chat_id="100",
        _nutrition_session_key="telegram:100",
    )
    message.sender_id = "__nutrition_ingest__"
    assert _trusted_nutrition_request(message)["client_op_id"].endswith(":meal-observation:v1")
