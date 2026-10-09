"""Resolve a loaded historical attachment to one trusted conversation source."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime

from openharness.channels.bus.events import InboundMessage
from openharness.engine.messages import AttachmentRefBlock, ConversationMessage
from ohmo.camera_protocol.models import validate_candidate_id

from ohmo.gateway.turn_context import TurnContext, canonical_principal, is_private_message


def _event_id_from_provenance(provenance: Mapping[str, object], source_id: str) -> str | None:
    channel = provenance.get("channel")
    chat_id = provenance.get("chat_id")
    principal = provenance.get("principal")
    if not all(isinstance(value, str) and value for value in (channel, chat_id, principal)):
        return None
    prefix = f"{channel.strip().lower()}:"
    if not principal.startswith(prefix):
        return None
    canonical = canonical_principal(channel, principal[len(prefix):])
    seed = "\x00".join((channel.strip().lower(), chat_id, canonical, source_id)).encode("utf-8")
    return f"ohmo-event-{hashlib.sha256(seed).hexdigest()}"


def resolve_selected_photo_source(
    *,
    attachment_id: str,
    history: list[ConversationMessage],
    message: InboundMessage,
    turn_ctx: TurnContext,
    session_key: str,
    gateway_session_id: str,
    tenant_id: str,
    authorized_participant: bool,
) -> dict[str, object] | None:
    """Return gateway-stamped identity only for one verified private source."""
    if (
        not authorized_participant
        or turn_ctx.channel != "telegram"
        or not turn_ctx.is_private
        or turn_ctx.is_forwarded
        or turn_ctx.session_id != gateway_session_id
        or message.channel != "telegram"
        or not is_private_message(message)
        or message.sender_id == "__camera__"
        or str(message.chat_id) != str(turn_ctx.chat_id)
        or message.metadata.get("is_forwarded") is True
        or message.metadata.get("is_group") is True
    ):
        return None
    source_principal = f"telegram:{canonical_principal('telegram', turn_ctx.principal)}"
    matches: list[tuple[str, str, Mapping[str, object], str | None, Mapping[str, object] | None]] = []
    owner_principal = f"telegram:{canonical_principal('telegram', turn_ctx.principal)}"
    for historical in history:
        for block in historical.content:
            if not isinstance(block, AttachmentRefBlock) or block.attachment_id != attachment_id:
                continue
            provenance = block.source_provenance
            if not isinstance(provenance, Mapping):
                continue
            origin = provenance.get("source_origin", "telegram")
            camera_source = origin == "dropbox_camera"
            source_id = provenance.get("source_message_id")
            if (provenance.get("channel") != "telegram"
                    or provenance.get("chat_id") != str(message.chat_id)
                    or provenance.get("session_key") != session_key
                    or not isinstance(provenance.get("gateway_session_id"), str)
                    or not provenance.get("gateway_session_id", "").strip()
                    or provenance.get("is_group") is not False
                    or provenance.get("is_forwarded") is not False
                    or (not camera_source and provenance.get("principal") != source_principal)
                    or (camera_source and (
                        provenance.get("principal") != "telegram:__camera__"
                        or provenance.get("owner_principal") != owner_principal
                        or provenance.get("timestamp_authority") not in {
                            "camera_capture_time", "camera_delivery_receipt"
                        }
                        or not isinstance(provenance.get("camera_candidate_id"), str)
                        or not isinstance(provenance.get("native_photo_message_id"), str)
                        or provenance.get("native_photo_message_id") != source_id
                        or not provenance.get("native_photo_message_id", "").isdigit()
                    ))
                    or (not camera_source and provenance.get("timestamp_authority") != "inbound_event_timestamp")):
                continue
            if camera_source:
                try:
                    validate_candidate_id(str(provenance.get("camera_candidate_id")))
                except ValueError:
                    continue
            received_at = provenance.get("received_at")
            if not isinstance(source_id, str) or not source_id.strip():
                continue
            if received_at is None and camera_source and provenance.get("timestamp_authority") == "camera_delivery_receipt":
                pass
            elif isinstance(received_at, str):
                try:
                    source_time = datetime.fromisoformat(received_at.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if source_time.tzinfo is None or source_time.utcoffset() is None:
                    continue
            else:
                continue
            consumed_occurrences = provenance.get("consumed_occurrences")
            if isinstance(consumed_occurrences, list) and consumed_occurrences:
                for occurrence in consumed_occurrences:
                    if not isinstance(occurrence, Mapping):
                        continue
                    receipt_id = occurrence.get("receipt_event_id")
                    client_op_id = occurrence.get("client_op_id")
                    consumed_append_id = occurrence.get("append_source_message_id")
                    if (
                        isinstance(receipt_id, str) and receipt_id.strip()
                        and isinstance(client_op_id, str) and client_op_id.strip()
                        and isinstance(consumed_append_id, str) and consumed_append_id.strip()
                        and client_op_id.endswith(":assistant")
                        and (not camera_source or provenance.get("owner_principal") == source_principal)
                    ):
                        matches.append((
                            source_id.strip(), consumed_append_id.strip(),
                            provenance, received_at, occurrence,
                        ))
                continue
            append_id = provenance.get("append_source_message_id")
            if camera_source:
                # A delivered Camera photo may be selected for a first
                # observation. It has no meal append receipt yet, so this
                # source identity cannot authorize a correction or borrow a
                # synthetic delivery event as an owner append identity.
                append_id = source_id
            elif not isinstance(append_id, str) or not append_id.strip():
                if historical.event_id != _event_id_from_provenance(provenance, source_id):
                    continue
                append_id = source_id
            matches.append((source_id.strip(), append_id.strip(), provenance, received_at, None))
    if len(matches) != 1:
        return None
    source_id, append_id, provenance, received_at, occurrence = matches[0]
    result = {
        "schema_version": 1,
        "tenant_id": tenant_id,
        "source_principal": source_principal,
        "gateway_session_id": provenance["gateway_session_id"],
        "current_gateway_session_id": gateway_session_id,
        "source_message_id": source_id,
        "append_source_message_id": append_id,
        "is_private": True,
        "is_forwarded": False,
        "is_group": False,
        "attachment_id": attachment_id,
        "chat_id": str(message.chat_id),
        "session_key": session_key,
        "received_at": received_at,
        "photo_source_message_id": provenance.get("photo_source_message_id", source_id),
        "photo_gateway_session_id": provenance["gateway_session_id"],
        "source_origin": provenance.get("source_origin", "telegram"),
        "origin_principal": provenance.get("origin_principal", source_principal),
        "camera_candidate_id": provenance.get("camera_candidate_id"),
        "native_photo_message_id": provenance.get("native_photo_message_id"),
    }
    if occurrence is not None:
        original_source = occurrence.get("append_source_message_id")
        original_session = occurrence.get("gateway_session_id")
        original_event = occurrence.get("receipt_event_id")
        original_operation = occurrence.get("client_op_id")
        if not all(isinstance(value, str) and value.strip() for value in
                   (original_source, original_session, original_event, original_operation)):
            return None
        result.update(
            original_source_message_id=original_source,
            original_append_source_message_id=original_source,
            original_gateway_session_id=original_session,
            original_receipt_event_id=original_event,
            original_receipt_client_op_id=original_operation,
        )
    return result
