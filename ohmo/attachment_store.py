"""Content-addressed durable image storage for Ohmo conversations."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ohmo.workspace import get_attachments_dir
from openharness.engine.messages import (
    AttachmentRefBlock,
    ConversationMessage,
    ImageBlock,
)
from openharness.utils.fs import atomic_write_bytes, atomic_write_text


_ATTACHMENT_ID_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class StoredAttachment:
    """Verified stored bytes plus their durable conversation reference."""

    ref: AttachmentRefBlock
    data: bytes


class AttachmentStore:
    """Atomic SHA-256 object store rooted under the Ohmo workspace."""

    def __init__(self, workspace: str | Path | None = None) -> None:
        self._root = get_attachments_dir(workspace).resolve()
        self._objects = self._root / "objects"
        self._objects.mkdir(parents=True, exist_ok=True)

    def _paths(self, attachment_id: str) -> tuple[Path, Path]:
        if _ATTACHMENT_ID_RE.fullmatch(attachment_id) is None:
            raise ValueError("invalid attachment_id")
        directory = (self._objects / attachment_id[:2]).resolve()
        try:
            directory.relative_to(self._objects)
        except ValueError as exc:  # defensive; strict IDs make this unreachable
            raise ValueError("attachment_id escapes attachment store") from exc
        return directory / f"{attachment_id}.bin", directory / f"{attachment_id}.json"

    @staticmethod
    def _normalize_media_type(media_type: str) -> str:
        normalized = str(media_type).strip().lower()
        if not normalized.startswith("image/"):
            raise ValueError("attachment media_type must be an image MIME type")
        return normalized

    def ingest_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        label: str = "image",
    ) -> AttachmentRefBlock:
        """Atomically ingest bytes and return their content-addressed ref."""
        if not isinstance(data, bytes) or not data:
            raise ValueError("attachment payload must contain bytes")
        normalized_media_type = self._normalize_media_type(media_type)
        attachment_id = hashlib.sha256(data).hexdigest()
        object_path, metadata_path = self._paths(attachment_id)
        if object_path.is_symlink() or metadata_path.is_symlink():
            raise ValueError("attachment store entry may not be a symlink")
        if object_path.exists():
            existing = object_path.read_bytes()
            if hashlib.sha256(existing).hexdigest() != attachment_id:
                raise ValueError("stored attachment failed content verification")
        else:
            atomic_write_bytes(object_path, data, mode=0o600)
        metadata = {
            "attachment_id": attachment_id,
            "media_type": normalized_media_type,
            "byte_size": len(data),
        }
        if not metadata_path.exists():
            atomic_write_text(
                metadata_path,
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
                mode=0o600,
            )
        stored = self.load_image(attachment_id)
        return AttachmentRefBlock(
            attachment_id=attachment_id,
            media_type=stored.ref.media_type,
            byte_size=stored.ref.byte_size,
            label=label,
        )

    def ingest_image_block(self, block: ImageBlock) -> AttachmentRefBlock:
        """Decode and ingest a request-local inline image."""
        try:
            data = base64.b64decode(block.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("invalid inline image base64") from exc
        label = Path(block.source_path).name if block.source_path else "image"
        return self.ingest_bytes(data, media_type=block.media_type, label=label)

    def load_image(self, attachment_id: str) -> StoredAttachment:
        """Load one verified object by strict ID; paths are never model supplied."""
        object_path, metadata_path = self._paths(attachment_id)
        if object_path.is_symlink() or metadata_path.is_symlink():
            raise ValueError("attachment store entry may not be a symlink")
        if not object_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"attachment not found: {attachment_id}")
        data = object_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != attachment_id:
            raise ValueError("stored attachment failed content verification")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError("invalid attachment metadata") from exc
        if metadata.get("attachment_id") != attachment_id:
            raise ValueError("attachment metadata ID mismatch")
        media_type = self._normalize_media_type(metadata.get("media_type", ""))
        byte_size = metadata.get("byte_size")
        if not isinstance(byte_size, int) or byte_size != len(data):
            raise ValueError("attachment metadata size mismatch")
        return StoredAttachment(
            ref=AttachmentRefBlock(
                attachment_id=attachment_id,
                media_type=media_type,
                byte_size=byte_size,
                label="image",
            ),
            data=data,
        )

    def externalize_messages(
        self,
        messages: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        """Replace every inline image with a durable reference."""
        durable: list[ConversationMessage] = []
        for message in messages:
            original_ref_ids = {
                block.attachment_id
                for block in message.content
                if isinstance(block, AttachmentRefBlock)
            }
            content = []
            for block in message.content:
                if isinstance(block, ImageBlock):
                    ref = self.ingest_image_block(block)
                    if ref.attachment_id not in original_ref_ids:
                        content.append(ref)
                    continue
                content.append(block)
            durable.append(message.model_copy(update={"content": content}))
        return durable

    @staticmethod
    def assert_externalized(messages: list[ConversationMessage]) -> None:
        """Fail closed if a caller attempts to persist inline pixels."""
        if any(
            isinstance(block, ImageBlock)
            for message in messages
            for block in message.content
        ):
            raise ValueError("Ohmo durable history contains inline image data")
