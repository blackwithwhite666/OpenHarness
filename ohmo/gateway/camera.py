"""Best-effort, Camera-specific ingress into Marina's ordinary gateway session.

The journal is an attempt tombstone, not a durable outbox.  In particular a
restart never resumes a send whose result may have been lost.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import stat
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ohmo.nutrition_ingest.models import ClassifierOutput, ManifestV2, validate_candidate_id
from openharness.channels.bus.events import InboundMessage, OutboundDeliveryReceipt, OutboundMessage

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_MANIFEST = 64 * 1024
_MAX_SIDECAR = 1024 * 1024
_MAX_IMAGE = 10 * 1024 * 1024
_MAX_HTTP_BODY = 4096
_MAX_ATTEMPTS = 10000
_MAX_JOURNAL = 8 * 1024 * 1024
_YES = frozenset(
    {"да, я это съела", "я это съела", "я съела это", "я съела", "я это ел", "я это съел"}
)
_NO = frozenset({"нет, не ела", "нет, не ел", "это не еда"})
CAMERA_AUTHORITY = object()


class CameraCandidateRequest(BaseModel):
    """The frozen v1 allowlist; no producer-authored destination or prompt."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    source_revision: str = Field(min_length=1, max_length=256)
    manifest_sha256: str
    image_sha256: str
    capture_time: datetime
    capture_time_authority: str

    @field_validator("candidate_id")
    @classmethod
    def _candidate(cls, value: str) -> str:
        return validate_candidate_id(value)

    @field_validator("manifest_sha256", "image_sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("invalid SHA-256")
        return value

    @field_validator("capture_time")
    @classmethod
    def _capture_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("capture_time must be timezone-aware")
        return value

    @field_validator("capture_time", mode="before")
    @classmethod
    def _capture_time_wire(cls, value: object) -> object:
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError("capture_time must be a bounded ISO-8601 string")
        return value

    @field_validator("capture_time_authority")
    @classmethod
    def _authority(cls, value: str) -> str:
        if value not in {"exif", "filename"}:
            raise ValueError("invalid capture time authority")
        return value


def _read_regular(path: str | Path, maximum: int, *, dir_fd: int | None = None) -> bytes:
    """Snapshot a bounded regular file without following its final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, dir_fd=dir_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("candidate evidence is not a bounded regular file")
        data = bytearray()
        while len(data) <= maximum:
            chunk = os.read(fd, min(65536, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            len(data) != before.st_size
            or len(data) > maximum
            or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
        ):
            raise ValueError("candidate evidence changed during snapshot")
        return bytes(data)
    finally:
        os.close(fd)


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _child_directory(parent_fd: int, name: str, *, create: bool = False) -> int:
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    return os.open(name, _directory_flags(), dir_fd=parent_fd)


def _published_positive(sidecar_bytes: bytes, manifest: ManifestV2) -> None:
    """Validate the bounded Telegent producer snapshot independently of the manifest."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate producer evidence key")
            result[key] = value
        return result

    sidecar = json.loads(sidecar_bytes, object_pairs_hook=unique_object)
    if not isinstance(sidecar, dict):
        raise ValueError("producer evidence is not an object")
    release = sidecar.get("release")
    decision = sidecar.get("decision")
    clip_release = sidecar.get("clip_release")
    clip = sidecar.get("clip_decision")
    if not all(isinstance(item, dict) for item in (release, decision, clip_release, clip)):
        raise ValueError("producer decision or release is absent")
    if (
        type(sidecar.get("schema_version")) is not int
        or sidecar["schema_version"] != 1
        or type(sidecar.get("revision")) is not int
        or sidecar["revision"] < 1
        or sidecar.get("candidate_id") != manifest.candidate_id
        or sidecar.get("state") != "published"
        or sidecar.get("clip_runtime_mode") != "enforce"
    ):
        raise ValueError("candidate is not a published producer result")
    history = sidecar.get("state_history")
    if not isinstance(history, list) or not history or history[-1] != "published":
        raise ValueError("published producer history is absent")
    classifier_fields = {
        "model": manifest.classifier_model,
        "prompt_version": manifest.classifier_prompt_version,
        "policy_version": manifest.classifier_policy_version,
        "dataset_version": manifest.classifier_dataset_version,
    }
    if any(release.get(key) != value for key, value in classifier_fields.items()):
        raise ValueError("producer classifier release differs from manifest")
    if (
        decision.get("candidate_id") != manifest.candidate_id
        or decision.get("is_food_like") is not True
        or any(
            decision.get(key) != classifier_fields[key]
            for key in ("model", "prompt_version", "policy_version")
        )
        or ClassifierOutput.model_validate(decision.get("output")) != manifest.classifier_output
    ):
        raise ValueError("producer classifier decision differs from manifest")
    clip_fields = ("model_id", "model_revision", "preprocessing_version", "threshold")
    if (
        clip.get("candidate_id") != manifest.candidate_id
        or clip.get("outcome") != "pass"
        or clip.get("forward_to_qwen") is not True
        or any(clip.get(key) != clip_release.get(key) for key in clip_fields)
        or not isinstance(clip_release.get("model_id"), str)
        or not clip_release["model_id"]
        or not isinstance(clip_release.get("model_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", clip_release["model_revision"]) is None
        or not isinstance(clip_release.get("preprocessing_version"), str)
        or not clip_release["preprocessing_version"]
    ):
        raise ValueError("producer CLIP release differs from decision")
    threshold, score = clip_release.get("threshold"), clip.get("score")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or score < threshold
    ):
        raise ValueError("producer CLIP pass is invalid")


class CameraIngress:
    """One-owner in-process admission with restart-safe attempt tombstones."""

    def __init__(self, config, *, workspace: Path, bus, telegram) -> None:
        self.config = config
        self._bus = bus
        self._telegram = telegram
        self._workspace = workspace
        self._state_dir = workspace / "camera_ingress"
        self._state_path = self._state_dir / "attempts.json"
        self._lock = asyncio.Lock()
        self._attempts: dict[str, dict] = self._load_attempts()
        self._tasks: set[asyncio.Task] = set()

    def _load_attempts(self) -> dict[str, dict]:
        try:
            state_fd = self._open_state_dir(create=False)
        except FileNotFoundError:
            return {}
        try:
            try:
                payload = json.loads(_read_regular("attempts.json", _MAX_JOURNAL, dir_fd=state_fd))
            except FileNotFoundError:
                return {}
        finally:
            os.close(state_fd)
        if not isinstance(payload, dict) or len(payload) > _MAX_ATTEMPTS:
            raise ValueError("camera attempt journal is invalid")
        for key, value in payload.items():
            validate_candidate_id(key)
            if not isinstance(value, dict) or value.get("state") not in {
                "admitted",
                "photo_sent",
                "answering",
                "final_queued",
                "completed",
                "delivery_unknown",
            }:
                raise ValueError("camera attempt journal is invalid")
        return payload

    def _open_state_dir(self, *, create: bool) -> int:
        workspace_fd = os.open(self._workspace, _directory_flags())
        try:
            return _child_directory(workspace_fd, "camera_ingress", create=create)
        finally:
            os.close(workspace_fd)

    def _save_attempts(self) -> None:
        state_fd = self._open_state_dir(create=True)
        temporary = f".{uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=state_fd
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self._attempts, stream, separators=(",", ":"), sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, "attempts.json", src_dir_fd=state_fd, dst_dir_fd=state_fd)
            os.fsync(state_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=state_fd)
            except FileNotFoundError:
                pass
            os.close(state_fd)

    def mark_restart_unknown(self) -> None:
        """Never resume an in-flight photo or answer after process restart."""
        changed = False
        for attempt in self._attempts.values():
            if attempt["state"] not in {"completed", "delivery_unknown"}:
                attempt["state"] = "delivery_unknown"
                changed = True
        if changed:
            self._save_attempts()

    def _token(self) -> bytes:
        path = self.config.bearer_token_file
        if path is None or path.is_symlink():
            raise ValueError("camera bearer credential is unavailable")
        metadata = path.stat()
        if (
            metadata.st_mode & 0o077
            or metadata.st_uid != getattr(os, "getuid", lambda: metadata.st_uid)()
        ):
            raise ValueError("camera bearer credential must be owner-only")
        token = _read_regular(path, 4096).strip()
        if len(token) < 32 or b"\n" in token or b"\r" in token:
            raise ValueError("camera bearer credential is invalid")
        return token

    def _evidence(self, request: CameraCandidateRequest) -> tuple[bytes, str]:
        root = self.config.synchronized_root
        if root is None or not root.is_dir():
            raise ValueError("camera root is unavailable")
        root_fd = os.open(root, _directory_flags())
        try:
            candidate_fd = _child_directory(root_fd, request.candidate_id)
            try:
                producer_fd = _child_directory(root_fd, "_producer")
                try:
                    sidecar_bytes = _read_regular(
                        f"{request.candidate_id}.json", _MAX_SIDECAR, dir_fd=producer_fd
                    )
                finally:
                    os.close(producer_fd)
                return self._evidence_from_directory(candidate_fd, request, sidecar_bytes)
            finally:
                os.close(candidate_fd)
        finally:
            os.close(root_fd)

    def _evidence_from_directory(
        self, candidate_fd: int, request: CameraCandidateRequest, sidecar_bytes: bytes
    ) -> tuple[bytes, str]:
        try:
            os.stat("result.json", dir_fd=candidate_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("candidate is already owned by the legacy coordinator")
        manifest_bytes = _read_regular("manifest.json", _MAX_MANIFEST, dir_fd=candidate_fd)
        if hashlib.sha256(manifest_bytes).hexdigest() != request.manifest_sha256:
            raise ValueError("candidate manifest digest differs")
        manifest = ManifestV2.model_validate_json(manifest_bytes)
        if (
            manifest.candidate_id != request.candidate_id
            or manifest.event_id != f"{request.candidate_id}:manifest:v1"
            or manifest.rev != request.source_revision
            or manifest.original_sha256 != request.image_sha256
            or manifest.normalized_capture_time != request.capture_time
            or manifest.capture_time_authority != request.capture_time_authority
        ):
            raise ValueError("candidate manifest differs from request")
        _published_positive(sidecar_bytes, manifest)
        image_name = Path(manifest.original_filename)
        if image_name.name != manifest.original_filename or image_name.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            raise ValueError("candidate image name is invalid")
        mime_for_suffix = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        if manifest.mime_type != mime_for_suffix[image_name.suffix.lower()]:
            raise ValueError("candidate image MIME type differs from filename")
        image_bytes = _read_regular(f"original{image_name.suffix}", _MAX_IMAGE, dir_fd=candidate_fd)
        if (
            len(image_bytes) != manifest.original_size_bytes
            or hashlib.sha256(image_bytes).hexdigest() != request.image_sha256
        ):
            raise ValueError("candidate image digest differs")
        return image_bytes, image_name.suffix.lower()

    async def admit(self, authorization: str | None, payload: object) -> tuple[int, dict]:
        if not self.config.enabled:
            return self._error(403, "camera_disabled")
        try:
            expected = self._token()
        except (OSError, ValueError):
            return self._error(503, "pre_admission_unavailable")
        prefix = "Bearer "
        if (
            not isinstance(authorization, str)
            or not authorization.startswith(prefix)
            or not hmac.compare_digest(authorization[len(prefix) :].encode(), expected)
        ):
            return self._error(401, "unauthorized")
        try:
            request = CameraCandidateRequest.model_validate(payload)
        except ValidationError:
            return self._error(400, "invalid_request")
        async with self._lock:
            if request.candidate_id in self._attempts:
                return self._error(409, "candidate_already_attempted")
            if len(self._attempts) >= _MAX_ATTEMPTS:
                return self._error(503, "pre_admission_unavailable")
            if any(item["state"] != "completed" for item in self._attempts.values()):
                return self._error(409, "unresolved_candidate")
            if not self._telegram or not getattr(self._telegram, "polling_started", False):
                return self._error(503, "pre_admission_unavailable")
            try:
                image_bytes, suffix = self._evidence(request)
            except (OSError, ValueError, TypeError, ValidationError):
                return self._error(422, "candidate_evidence_mismatch")
            admission_id = f"cam1-{uuid4().hex}"
            try:
                state_fd = self._open_state_dir(create=True)
                try:
                    snapshot_fd = _child_directory(state_fd, "snapshots", create=True)
                    try:
                        snapshot = self._state_dir / "snapshots" / f"{admission_id}{suffix}"
                        descriptor = os.open(
                            snapshot.name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600,
                            dir_fd=snapshot_fd,
                        )
                    finally:
                        os.close(snapshot_fd)
                finally:
                    os.close(state_fd)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(image_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                self._attempts[request.candidate_id] = {
                    "state": "admitted",
                    "admission_id": admission_id,
                    "snapshot": str(snapshot),
                    "photo_id": None,
                    "reply_ids": [],
                    "final_turn_id": None,
                }
                self._save_attempts()
            except OSError:
                # A journal fsync can fail after replace. Retain any in-memory
                # tombstone rather than risking a second send in this process.
                return self._error(503, "pre_admission_unavailable")
            task = asyncio.create_task(self._deliver(request.candidate_id), name=admission_id)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return 202, {
                "status": "admitted",
                "candidate_id": request.candidate_id,
                "admission_id": admission_id,
                "delivery_semantics": "in_process_only",
            }

    @staticmethod
    def _error(status: int, code: str) -> tuple[int, dict]:
        return status, {"error": {"code": code}}

    async def _deliver(self, candidate_id: str) -> None:
        attempt = self._attempts[candidate_id]
        try:
            receipt: OutboundDeliveryReceipt = await self._telegram.send_camera_photo(
                chat_id=self.config.chat_id,
                image_path=attempt["snapshot"],
                caption="Фото из Camera. Ответьте на это фото: «Я это съела» или «Нет, не ела».",
            )
            if (
                receipt.channel != "telegram"
                or str(receipt.chat_id) != self.config.chat_id
                or len(receipt.native_message_ids) != 1
                or not isinstance(receipt.native_message_ids[0], int)
                or isinstance(receipt.native_message_ids[0], bool)
                or receipt.native_message_ids[0] <= 0
            ):
                raise ValueError("native Camera photo receipt is invalid")
            attempt["photo_id"] = receipt.native_message_ids[0]
            attempt["state"] = "photo_sent"
            self._save_attempts()
            await self._bus.publish_inbound(
                InboundMessage(
                    channel="telegram",
                    sender_id="__camera__",
                    chat_id=self.config.chat_id,
                    content=(
                        "Проанализируй снимок Camera и спроси Марину, ела ли она это. "
                        "Это только анализ: не записывай приём пищи без её явного ответа."
                    ),
                    media=[attempt["snapshot"]],
                    metadata={
                        "_synthetic": True,
                        "_camera_authority": CAMERA_AUTHORITY,
                        "_camera_candidate_id": candidate_id,
                        "_camera_photo_id": attempt["photo_id"],
                        "is_group": False,
                        "chat_type": "private",
                    },
                )
            )
        except Exception:
            # A lost response could still mean Telegram accepted the photo.
            # Keep the attempt unresolved; never send it again automatically.
            attempt["state"] = "delivery_unknown"
            self._save_attempts()

    def process_real_inbound(self, message: InboundMessage) -> None:
        if not self.config.enabled or message.channel != "telegram":
            return
        if (
            str(message.chat_id) != self.config.chat_id
            or message.sender_id.split("|", 1)[0] != self.config.principal
        ):
            return
        metadata = message.metadata
        target = (
            metadata.get("native_message_id")
            if metadata.get("callback_query")
            else metadata.get("reply_to_message_id")
        )
        raw_text = metadata.get("_telegram_raw_text", message.content)
        if (
            self._attempts
            and not message.media
            and target is None
            and not any(item["state"] != "completed" for item in self._attempts.values())
            and isinstance(raw_text, str)
            and raw_text.strip().casefold() in {*_YES, *_NO, "да", "yes", "ага"}
        ):
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        for attempt in self._attempts.values():
            if (
                target is not None
                and attempt["state"] != "photo_sent"
                and str(target)
                in {
                    str(attempt["photo_id"]),
                    *map(str, attempt["reply_ids"]),
                }
            ):
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
        pending = [
            (key, value) for key, value in self._attempts.items() if value["state"] != "completed"
        ]
        if len(pending) != 1:
            return
        candidate_id, attempt = pending[0]
        if message.media:
            # A new, unrelated manual photo keeps its ordinary path. A photo
            # replied to this Camera operation must not bypass its answer gate.
            if str(target) in {str(attempt["photo_id"]), *map(str, attempt["reply_ids"])}:
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        if attempt["state"] != "photo_sent":
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        if metadata.get("callback_query"):
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return  # Camera has no callback buttons; stale/foreign callbacks confer nothing.
        if str(target) not in {str(attempt["photo_id"]), *map(str, attempt["reply_ids"])}:
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        raw_answer = metadata.get("_telegram_raw_text")
        answer = raw_answer.strip().casefold() if isinstance(raw_answer, str) else ""
        if answer in _YES or answer in _NO:
            attempt["state"] = "answering"
            try:
                self._save_attempts()
            except OSError:
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
            metadata["_camera_authority"] = CAMERA_AUTHORITY
            metadata["_camera_candidate_id"] = candidate_id
            metadata["_camera_answer"] = "yes" if answer in _YES else "no"
            metadata["_camera_turn_id"] = uuid4().hex
            if answer in _YES:
                message.media.append(attempt["snapshot"])
            return
        metadata["_camera_unbound"] = CAMERA_AUTHORITY

    def note_assistant_receipt(
        self, message: OutboundMessage, receipt: OutboundDeliveryReceipt | None
    ) -> None:
        if message.metadata.get("_camera_authority") is not CAMERA_AUTHORITY:
            return
        candidate_id = message.metadata.get("_camera_candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in self._attempts:
            return
        attempt = self._attempts[candidate_id]
        turn_id = message.metadata.get("_camera_turn_id")
        expected_turn_id = attempt.get("final_turn_id")
        final = (
            message.metadata.get("_camera_final") is CAMERA_AUTHORITY
            and isinstance(turn_id, str)
            and turn_id
            and turn_id == expected_turn_id
        )
        valid_receipt = (
            receipt is not None
            and receipt.channel == "telegram"
            and str(receipt.chat_id) == self.config.chat_id
            and bool(receipt.native_message_ids)
            and all(
                isinstance(native_id, int) and not isinstance(native_id, bool) and native_id > 0
                for native_id in receipt.native_message_ids
            )
        )
        if not valid_receipt:
            if final and attempt["state"] == "final_queued":
                attempt["state"] = "delivery_unknown"
                self._save_attempts()
            return
        for native_id in receipt.native_message_ids:
            if native_id not in attempt["reply_ids"]:
                attempt["reply_ids"].append(native_id)
        if final and attempt["state"] == "final_queued":
            attempt["state"] = "completed"
        self._save_attempts()

    def note_assistant_failure(self, message: OutboundMessage) -> None:
        if message.metadata.get("_camera_final") is not CAMERA_AUTHORITY:
            return
        if message.metadata.get("_camera_authority") is not CAMERA_AUTHORITY:
            return
        candidate_id = message.metadata.get("_camera_candidate_id")
        attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
        if (
            attempt is not None
            and attempt["state"] == "final_queued"
            and isinstance(message.metadata.get("_camera_turn_id"), str)
            and message.metadata["_camera_turn_id"] == attempt.get("final_turn_id")
        ):
            attempt["state"] = "delivery_unknown"
            self._save_attempts()

    def complete(self, message: InboundMessage, *, recorded: bool) -> None:
        """Arm completion; only a native final-send receipt releases the candidate."""
        if message.metadata.get("_camera_authority") is not CAMERA_AUTHORITY:
            return
        if message.metadata.get("_camera_answer") == "no" or recorded:
            candidate_id = message.metadata.get("_camera_candidate_id")
            turn_id = message.metadata.get("_camera_turn_id")
            attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
            if (
                attempt is not None
                and attempt["state"] == "answering"
                and isinstance(turn_id, str)
                and turn_id
            ):
                attempt["final_turn_id"] = turn_id
                attempt["state"] = "final_queued"
                self._save_attempts()

    async def close(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


async def serve_camera_http(
    ingress: CameraIngress, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """One bounded HTTP/1.1 request per private connection; never log credentials."""
    status, response = 400, {"error": {"code": "invalid_request"}}
    try:
        header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        if len(header) > 8192:
            raise ValueError("large headers")
        lines = header.decode("ascii").split("\r\n")
        method, path, version = lines[0].split(" ")
        fields: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            key, value = line.split(":", 1)
            key = key.lower().strip()
            if key in fields:
                raise ValueError("duplicate header")
            fields[key] = value.strip()
        if method == "POST" and path == "/internal/v1/camera/candidates" and version == "HTTP/1.1":
            if not ingress.config.enabled:
                status, response = ingress._error(403, "camera_disabled")
            elif not fields.get("authorization", "").startswith("Bearer "):
                status, response = ingress._error(401, "unauthorized")
            elif fields.get("transfer-encoding") or not fields.get("content-length", "").isdigit():
                status, response = ingress._error(400, "invalid_request")
            else:
                length = int(fields["content-length"])
                if not 0 < length <= _MAX_HTTP_BODY:
                    status, response = ingress._error(400, "invalid_request")
                else:
                    body = await asyncio.wait_for(reader.readexactly(length), timeout=5)

                    def unique_object(pairs: list[tuple[str, object]]) -> dict:
                        result = {}
                        for key, value in pairs:
                            if key in result:
                                raise ValueError("duplicate JSON key")
                            result[key] = value
                        return result

                    try:
                        payload = json.loads(body, object_pairs_hook=unique_object)
                    except (UnicodeDecodeError, ValueError):
                        payload = None
                    status, response = await ingress.admit(fields.get("authorization"), payload)
    except (
        asyncio.IncompleteReadError,
        asyncio.LimitOverrunError,
        asyncio.TimeoutError,
        UnicodeError,
        ValueError,
    ):
        pass
    body = json.dumps(response, separators=(",", ":")).encode()
    writer.write(
        f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        + body
    )
    try:
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()
