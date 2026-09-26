"""Best-effort Camera API ingress into the configured gateway session.

The journal is an attempt tombstone, not a durable outbox.  In particular a
restart never resumes a send whose result may have been lost.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ohmo.camera_protocol.models import ClassifierOutput, ManifestV2, validate_candidate_id
from openharness.channels.bus.events import InboundMessage, OutboundDeliveryReceipt, OutboundMessage

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_MANIFEST = 64 * 1024
_MAX_SIDECAR = 1024 * 1024
_MAX_IMAGE = 10 * 1024 * 1024
_MAX_REQUEST = 4096
_MAX_HTTP_BODY = 12 * 1024 * 1024
_MAX_ATTEMPTS = 10000
_MAX_JOURNAL = 8 * 1024 * 1024
_SESSION_TTL_SECONDS = 24 * 60 * 60
_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
_YES = frozenset(
    {"да, я это съела", "я это съела", "я съела это", "я съела", "я это ел", "я это съел"}
)
_NO = frozenset({"нет, не ела", "нет, не ел", "это не еда"})
_PENDING_TTL_SECONDS = 6 * 60 * 60
_ANSWER_EXPLICIT_NO_RE = re.compile(
    r"\b(?:не\s+(?:ел|ела|ели|пил|пила|выпил|выпила|употреблял|употребляла)\b"
    r"|ничего\s+не\s+(?:ел|ела|пил|пила)\b"
    r"|это\s+не\s+еда\b)",
    re.IGNORECASE,
)
_ANSWER_ANCHORED_NO_RE = re.compile(r"^(?:нет|no)\b", re.IGNORECASE)
_ANSWER_CONSUMPTION_RE = re.compile(
    r"\b(?:я\s+)?(?:съел(?:а|и)?|ел(?:а|и)?|поел(?:а|и)?|выпил(?:а|и)?|употребил(?:а|и)?)\b",
    re.IGNORECASE,
)
_ANSWER_ANCHORED_YES_RE = re.compile(
    r"^(?:да|ага|угу|конечно|естественно|yes|yeah)\b", re.IGNORECASE
)
_ANSWER_ANCHORED_SCOPE_RE = re.compile(r"^(?:только|лишь)\b", re.IGNORECASE)
_DEEPSEEK_MODEL = "deepseek/deepseek-v4.1-flash"
_DEEPSEEK_ENDPOINT = "deepinfra/fp8"
_DEEPSEEK_RELEASE = "deepseek-camera-production-v1"
CAMERA_AUTHORITY = object()


def _classify_answer(text: object, *, anchored: bool) -> str | None:
    """Classify an explicit user answer to a Camera question.

    Returns "yes", "no", or None (ambiguous, must stay unbound). Bare,
    non-anchored text binds only through explicit consumption or negation
    language; bare affirmations and scope-only answers stay ambiguous
    outside a native reply or a first-party ask button.
    """
    answer = text.strip().casefold() if isinstance(text, str) else ""
    if not answer:
        return None
    if answer in _YES:
        return "yes"
    if answer in _NO:
        return "no"
    if _ANSWER_EXPLICIT_NO_RE.search(answer):
        return "no"
    if _ANSWER_CONSUMPTION_RE.search(answer):
        return "yes"
    if anchored:
        if _ANSWER_ANCHORED_NO_RE.search(answer):
            return "no"
        if _ANSWER_ANCHORED_YES_RE.search(answer) or _ANSWER_ANCHORED_SCOPE_RE.search(answer):
            return "yes"
    return None


def _attempt_age_seconds(attempt: dict, now: datetime) -> float:
    """Age of an attempt since admission; unparsable stamps never expire."""
    admitted_at = attempt.get("admitted_at")
    if not isinstance(admitted_at, str):
        return 0.0
    try:
        stamp = datetime.fromisoformat(admitted_at)
    except ValueError:
        return 0.0
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        return 0.0
    return (now - stamp).total_seconds()


class CameraCandidateRequest(BaseModel):
    """The frozen Camera request allowlist; no producer-authored destination or prompt."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    source_revision: str = Field(min_length=1, max_length=256)
    manifest_sha256: str
    image_sha256: str
    capture_time: datetime
    capture_time_authority: str
    session_id: str
    epoch: str
    seq: int = Field(gt=0, le=2**63 - 1)

    @field_validator("candidate_id")
    @classmethod
    def _candidate(cls, value: str) -> str:
        return validate_candidate_id(value)

    @field_validator("session_id", "epoch")
    @classmethod
    def _session_identity(cls, value: str) -> str:
        if _SESSION_ID.fullmatch(value) is None:
            raise ValueError("invalid Camera session identity")
        return value

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


@dataclass(frozen=True)
class CameraCandidateUpload:
    """Candidate metadata, producer evidence, and image carried by one HTTP request."""

    request: object
    manifest_bytes: bytes
    producer_sidecar_bytes: bytes
    image_bytes: bytes
    image_filename: str
    image_content_type: str


def _unique_json(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _parse_camera_upload(content_type: str, body: bytes) -> CameraCandidateUpload:
    """Parse one bounded multipart upload with exactly the four Camera fields."""
    envelope = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    if (
        envelope.defects
        or envelope.get_content_type() != "multipart/form-data"
        or not envelope.get_boundary()
        or envelope.preamble
        or envelope.epilogue
        or not envelope.is_multipart()
    ):
        raise ValueError("invalid Camera multipart envelope")

    parts: dict[str, object] = {}
    for part in envelope.iter_parts():
        if part.defects or part.is_multipart() or part.get_content_disposition() != "form-data":
            raise ValueError("invalid Camera multipart part")
        name = part.get_param("name", header="content-disposition")
        if name not in {"request", "manifest", "producer", "image"} or name in parts:
            raise ValueError("unexpected Camera multipart field")
        transfer_encoding = part.get("Content-Transfer-Encoding")
        if transfer_encoding not in {None, "7bit", "8bit", "binary"}:
            raise ValueError("encoded Camera multipart fields are not accepted")
        filename = part.get_filename()
        content_type_for_part = part.get_content_type()
        if name == "image":
            if not isinstance(filename, str) or not filename or len(filename) > 255:
                raise ValueError("Camera image filename is invalid")
        elif filename is not None or content_type_for_part != "application/json":
            raise ValueError("Camera metadata fields must be JSON without filenames")
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            raise ValueError("Camera multipart field has no byte payload")
        parts[name] = (payload, filename, content_type_for_part)
    if set(parts) != {"request", "manifest", "producer", "image"}:
        raise ValueError("Camera multipart fields are incomplete")

    request_bytes, _, _ = parts["request"]
    manifest_bytes, _, _ = parts["manifest"]
    producer_bytes, _, _ = parts["producer"]
    image_bytes, image_filename, image_content_type = parts["image"]
    if not 0 < len(request_bytes) <= _MAX_REQUEST:
        raise ValueError("Camera request metadata size is invalid")
    if not 0 < len(manifest_bytes) <= _MAX_MANIFEST:
        raise ValueError("Camera manifest size is invalid")
    if not 0 < len(producer_bytes) <= _MAX_SIDECAR:
        raise ValueError("Camera producer evidence size is invalid")
    if not 0 < len(image_bytes) <= _MAX_IMAGE:
        raise ValueError("Camera image size is invalid")
    try:
        request_payload = json.loads(request_bytes, object_pairs_hook=_unique_json)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Camera request metadata is invalid") from exc
    return CameraCandidateUpload(
        request=request_payload,
        manifest_bytes=manifest_bytes,
        producer_sidecar_bytes=producer_bytes,
        image_bytes=image_bytes,
        image_filename=image_filename,
        image_content_type=image_content_type,
    )


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
        or decision.get("classifier_route_attestation_json")
        != manifest.classifier_route_attestation_json
        or any(
            decision.get(key) != classifier_fields[key]
            for key in ("model", "prompt_version", "policy_version")
        )
        or ClassifierOutput.model_validate(decision.get("output")) != manifest.classifier_output
    ):
        raise ValueError("producer classifier decision differs from manifest")
    _validate_deepseek_route_attestation(manifest)
    clip_fields = ("model_id", "model_revision", "preprocessing_version", "threshold")
    if (
        clip.get("candidate_id") != manifest.candidate_id
        or clip.get("outcome") != "pass"
        or clip.get("forward_to_classifier") is not True
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


def _validate_deepseek_route_attestation(manifest: ManifestV2) -> None:
    """Require the selected DeepSeek model, endpoint, and privacy policy."""
    if (
        manifest.classifier_model != _DEEPSEEK_MODEL
        or manifest.classifier_policy_version != _DEEPSEEK_RELEASE
    ):
        raise ValueError("camera classifier is not the active DeepSeek release")
    raw = manifest.classifier_route_attestation_json
    if not isinstance(raw, str) or not raw:
        raise ValueError("DeepSeek route attestation is absent")

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate DeepSeek route attestation key")
            result[key] = value
        return result

    route = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(route, dict):
        raise ValueError("DeepSeek route attestation is malformed")
    expected = {
        "requested_model": _DEEPSEEK_MODEL,
        "requested_provider_only": [_DEEPSEEK_ENDPOINT],
        "requested_zdr": True,
        "requested_data_collection": "deny",
        "requested_allow_fallbacks": False,
        "requested_response_cache_header": "false",
        "prepared_provider_only": [_DEEPSEEK_ENDPOINT],
        "prepared_zdr": True,
        "prepared_data_collection": "deny",
        "prepared_allow_fallbacks": False,
        "prepared_provider_policy_source": "prepared_outbound_request_body",
        "prepared_response_cache_header": "false",
        "prepared_response_cache_header_source": "prepared_outbound_request_headers",
        "catalog_endpoint": _DEEPSEEK_ENDPOINT,
        "catalog_zdr": True,
        "response_model": _DEEPSEEK_MODEL,
        "response_provider": "deepinfra",
        "response_endpoint": None,
        "response_endpoint_source": "unverified",
    }
    if any(route.get(key) != value for key, value in expected.items()):
        raise ValueError("DeepSeek route or privacy policy differs")
    cache_status = route.get("response_cache_status")
    if cache_status == "HIT" or cache_status not in {None, "MISS"}:
        raise ValueError("DeepSeek response cache was not a fresh model call")
    cache_source = "absent" if cache_status is None else "response_header"
    if route.get("response_cache_status_source") != cache_source:
        raise ValueError("DeepSeek cache evidence is malformed")
    for key in (
        "candidate_identity_sha256",
        "catalog_sha256",
        "catalog_terms_sha256",
        "request_identity_sha256",
        "response_id_sha256",
        "response_identity_sha256",
    ):
        value = route.get(key)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("DeepSeek route evidence digest is malformed")
    for key in ("catalog_observed_at", "requested_at", "response_received_at"):
        value = route.get(key)
        if not isinstance(value, str):
            raise ValueError("DeepSeek route evidence timestamp is absent")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("DeepSeek route evidence timestamp is malformed") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("DeepSeek route evidence timestamp has no timezone")


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
        self._attempts, previous_session = self._load_attempts()
        self._session = self._new_session(
            session_id=previous_session.get("session_id") if previous_session else None
        )
        self._tasks: set[asyncio.Task] = set()
        self._sweep_expired_attempts()
        self._remove_completed_snapshots()

    @staticmethod
    def _new_session(*, session_id: str | None = None) -> dict:
        now = datetime.now(UTC)
        return {
            "session_id": session_id or uuid4().hex,
            "epoch": uuid4().hex,
            "committed_seq": 0,
            "expires_at": (now + timedelta(seconds=_SESSION_TTL_SECONDS)).isoformat(),
            "last_ack": None,
        }

    def _load_attempts(self) -> tuple[dict[str, dict], dict | None]:
        try:
            state_fd = self._open_state_dir(create=False)
        except FileNotFoundError:
            return {}, None
        try:
            try:
                payload = json.loads(_read_regular("attempts.json", _MAX_JOURNAL, dir_fd=state_fd))
            except FileNotFoundError:
                return {}, None
        finally:
            os.close(state_fd)
        session = None
        if isinstance(payload, dict) and payload.get("schema_version") == 2:
            if set(payload) != {"schema_version", "attempts", "session"}:
                raise ValueError("camera attempt journal is invalid")
            attempts = payload["attempts"]
            session = payload["session"]
            if (
                not isinstance(session, dict)
                or _SESSION_ID.fullmatch(session.get("session_id", "")) is None
                or _SESSION_ID.fullmatch(session.get("epoch", "")) is None
                or not isinstance(session.get("committed_seq"), int)
                or session["committed_seq"] < 0
            ):
                raise ValueError("camera session journal is invalid")
        else:
            # Read the previous candidate-keyed journal during the protocol upgrade.
            attempts = payload
        if not isinstance(attempts, dict) or len(attempts) > _MAX_ATTEMPTS:
            raise ValueError("camera attempt journal is invalid")
        for key, value in attempts.items():
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
        now_iso = datetime.now(UTC).isoformat()
        for value in attempts.values():
            if not isinstance(value.get("admitted_at"), str):
                # Legacy in-flight entries adopt this process's start as their
                # admission time, so the pending TTL has a bounded runway.
                value["admitted_at"] = now_iso
        return attempts, session

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
                json.dump(
                    {
                        "schema_version": 2,
                        "attempts": self._attempts,
                        "session": self._session,
                    },
                    stream,
                    separators=(",", ":"),
                    sort_keys=True,
                )
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

    _TTL_SWEEPABLE_STATES = frozenset({"photo_sent", "answering", "final_queued"})

    def _sweep_expired_attempts(self) -> None:
        """Drop answer-gate attempts older than the TTL with their snapshots.

        One ignored photo must not gate the whole chat forever: after the TTL
        the camera flow degrades to ordinary text turns instead of marking
        every inbound message unbound. ``admitted`` and ``delivery_unknown``
        tombstones are never swept: they are the at-most-once admission
        markers for sends whose outcome is unknown.
        """
        now = datetime.now(UTC)
        expired = [
            key
            for key, value in self._attempts.items()
            if value.get("state") in self._TTL_SWEEPABLE_STATES
            and _attempt_age_seconds(value, now) >= _PENDING_TTL_SECONDS
        ]
        if not expired:
            return
        for key in expired:
            self._remove_snapshot_files(self._attempts.pop(key))
        try:
            self._save_attempts()
        except OSError:
            logging.getLogger(__name__).warning(
                "failed to persist Camera attempt expiry", exc_info=True
            )

    def _remove_snapshot_files(self, attempt: dict) -> None:
        snapshot = attempt.get("snapshot")
        admission_id = attempt.get("admission_id")
        if not isinstance(snapshot, str) or not isinstance(admission_id, str):
            return
        name = Path(snapshot)
        if name.parent != self._state_dir / "snapshots" or name.name not in {
            f"{admission_id}.jpg",
            f"{admission_id}.jpeg",
            f"{admission_id}.png",
            f"{admission_id}.webp",
        }:
            return
        try:
            state_fd = self._open_state_dir(create=False)
            try:
                snapshots_fd = _child_directory(state_fd, "snapshots", create=False)
                try:
                    try:
                        os.unlink(name.name, dir_fd=snapshots_fd)
                        os.fsync(snapshots_fd)
                    except FileNotFoundError:
                        pass
                finally:
                    os.close(snapshots_fd)
            finally:
                os.close(state_fd)
        except FileNotFoundError:
            return
        except OSError:
            logging.getLogger(__name__).warning("failed to remove Camera snapshot", exc_info=True)

    def _remove_completed_snapshots(self) -> None:
        """Drop local image copies once the user flow has a confirmed outcome."""
        for attempt in list(self._attempts.values()):
            if attempt.get("state") != "completed":
                continue
            self._remove_snapshot_files(attempt)

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

    def _refresh_expired_session(self) -> bool:
        expires_at = datetime.fromisoformat(self._session["expires_at"])
        if datetime.now(UTC) < expires_at:
            return False
        self._session = self._new_session(session_id=self._session["session_id"])
        return True

    async def lease(self, authorization: str | None) -> tuple[int, dict]:
        """Return the current producer lease, rotating its epoch on expiration."""
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
        async with self._lock:
            try:
                self._refresh_expired_session()
                self._save_attempts()
            except (OSError, ValueError):
                return self._error(503, "pre_admission_unavailable")
            return 200, {
                "session_id": self._session["session_id"],
                "epoch": self._session["epoch"],
                "committed_seq": self._session["committed_seq"],
                "expires_at": self._session["expires_at"],
            }

    def _commit_sequence(
        self, request: CameraCandidateRequest, status: int, response: dict
    ) -> tuple[int, dict]:
        acknowledged = {
            **response,
            "session_id": request.session_id,
            "epoch": request.epoch,
            "ack_seq": request.seq,
        }
        self._session["committed_seq"] = request.seq
        self._session["last_ack"] = {
            "seq": request.seq,
            "status": status,
            "body": acknowledged,
        }
        self._save_attempts()
        return status, acknowledged

    @staticmethod
    def _validate_upload(
        request: CameraCandidateRequest, upload: CameraCandidateUpload
    ) -> tuple[bytes, str]:
        manifest_bytes = upload.manifest_bytes
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
        _published_positive(upload.producer_sidecar_bytes, manifest)
        image_name = upload.image_filename
        suffix = Path(image_name).suffix.lower()
        if (
            Path(image_name).name != image_name
            or "/" in image_name
            or "\\" in image_name
            or image_name != manifest.original_filename
            or suffix
            not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }
        ):
            raise ValueError("candidate image name is invalid")
        mime_for_suffix = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        if (
            manifest.mime_type != mime_for_suffix[suffix]
            or upload.image_content_type != manifest.mime_type
        ):
            raise ValueError("candidate image MIME type differs from filename")
        image_bytes = upload.image_bytes
        if (
            not 0 < len(image_bytes) <= _MAX_IMAGE
            or len(image_bytes) != manifest.original_size_bytes
            or hashlib.sha256(image_bytes).hexdigest() != request.image_sha256
        ):
            raise ValueError("candidate image digest differs")
        return image_bytes, suffix

    async def admit(self, authorization: str | None, upload: object) -> tuple[int, dict]:
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
        if not isinstance(upload, CameraCandidateUpload):
            return self._error(400, "invalid_request")
        try:
            request = CameraCandidateRequest.model_validate(upload.request)
        except ValidationError:
            return self._error(400, "invalid_request")
        async with self._lock:
            if self._refresh_expired_session():
                try:
                    self._save_attempts()
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
                return self._error(409, "session_expired")
            if (
                request.session_id != self._session["session_id"]
                or request.epoch != self._session["epoch"]
            ):
                return self._error(409, "session_expired")
            committed_seq = self._session["committed_seq"]
            if request.seq <= committed_seq:
                last_ack = self._session["last_ack"]
                if isinstance(last_ack, dict) and last_ack.get("seq") == request.seq:
                    return last_ack["status"], last_ack["body"]
                return 200, {
                    "status": "duplicate",
                    "session_id": self._session["session_id"],
                    "epoch": self._session["epoch"],
                    "ack_seq": committed_seq,
                }
            if request.seq > committed_seq + 1:
                return 409, {
                    "error": {"code": "expected_seq"},
                    "expected_seq": committed_seq + 1,
                }
            previous_attempt = self._attempts.get(request.candidate_id)
            if previous_attempt is not None:
                admission_id = previous_attempt.get("admission_id")
                if isinstance(admission_id, str) and admission_id:
                    response = {
                        "status": "admitted",
                        "candidate_id": request.candidate_id,
                        "admission_id": admission_id,
                        "delivery_semantics": "in_process_only",
                    }
                    try:
                        return self._commit_sequence(request, 202, response)
                    except OSError:
                        return self._error(503, "pre_admission_unavailable")
                try:
                    return self._commit_sequence(
                        request, 409, self._error(409, "candidate_already_attempted")[1]
                    )
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
            if len(self._attempts) >= _MAX_ATTEMPTS:
                return self._error(503, "pre_admission_unavailable")
            if any(item["state"] != "completed" for item in self._attempts.values()):
                return self._error(409, "unresolved_candidate")
            if not self._telegram or not getattr(self._telegram, "polling_started", False):
                return self._error(503, "pre_admission_unavailable")
            try:
                if not isinstance(upload, CameraCandidateUpload):
                    raise ValueError("Camera image and producer evidence were not uploaded")
                image_bytes, suffix = self._validate_upload(request, upload)
            except (OSError, ValueError, TypeError, ValidationError):
                status, response = self._error(422, "candidate_evidence_mismatch")
                try:
                    return self._commit_sequence(request, status, response)
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
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
                    "admitted_at": datetime.now(UTC).isoformat(),
                }
                status, response = self._commit_sequence(
                    request,
                    202,
                    {
                        "status": "admitted",
                        "candidate_id": request.candidate_id,
                        "admission_id": admission_id,
                        "delivery_semantics": "in_process_only",
                    },
                )
            except OSError:
                # A journal fsync can fail after replace. Retain any in-memory
                # tombstone rather than risking a second send in this process.
                return self._error(503, "pre_admission_unavailable")
            task = asyncio.create_task(self._deliver(request.candidate_id), name=admission_id)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return status, response

    @staticmethod
    def _error(status: int, code: str) -> tuple[int, dict]:
        return status, {"error": {"code": code}}

    async def _deliver(self, candidate_id: str) -> None:
        attempt = self._attempts[candidate_id]
        try:
            receipt: OutboundDeliveryReceipt = await self._telegram.send_camera_photo(
                chat_id=self.config.chat_id,
                image_path=attempt["snapshot"],
                caption="Фото из Camera. Ответьте на это фото: «Я это съел(а)» или «Нет, не ел(а)».",
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
                        "Проанализируй снимок Camera и спроси пользователя, ел(а) ли он(а) это. "
                        "Это только анализ: не записывай приём пищи без явного ответа пользователя."
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
        self._sweep_expired_attempts()
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
        anchored_ids = {str(attempt["photo_id"]), *map(str, attempt["reply_ids"])}
        if target is not None and str(target) not in anchored_ids:
            # A reply to some other message answers nothing about this photo.
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        if metadata.get("callback_query"):
            # The agent's own [[ask:]] keyboard hangs on this operation's
            # photo or question message; a tap there is a first-party answer.
            # Any other callback (stale, foreign keyboard) confers nothing.
            callback_data = metadata.get("callback_data")
            if not (isinstance(callback_data, str) and callback_data.startswith("ask:")):
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
            classified = _classify_answer(raw_text, anchored=True)
        elif target is not None:
            classified = _classify_answer(raw_text, anchored=True)
        else:
            # Bare text without a reply binds only through explicit
            # consumption or negation language, never a bare affirmation.
            classified = _classify_answer(raw_text, anchored=False)
        if classified is None:
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        attempt["state"] = "answering"
        try:
            self._save_attempts()
        except OSError:
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        metadata["_camera_authority"] = CAMERA_AUTHORITY
        metadata["_camera_candidate_id"] = candidate_id
        metadata["_camera_answer"] = classified
        metadata["_camera_turn_id"] = uuid4().hex
        if classified == "yes":
            message.media.append(attempt["snapshot"])
        return

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
        if final and attempt["state"] == "completed":
            self._remove_completed_snapshots()

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
        if method == "POST" and path == "/internal/v1/camera/session" and version == "HTTP/1.1":
            if fields.get("transfer-encoding") or fields.get("content-length") != "0":
                status, response = ingress._error(400, "invalid_request")
            else:
                status, response = await ingress.lease(fields.get("authorization"))
        elif (
            method == "POST" and path == "/internal/v1/camera/candidates" and version == "HTTP/1.1"
        ):
            if not ingress.config.enabled:
                status, response = ingress._error(403, "camera_disabled")
            elif not fields.get("authorization", "").startswith("Bearer "):
                status, response = ingress._error(401, "unauthorized")
            elif fields.get("transfer-encoding") or not fields.get("content-length", "").isdigit():
                status, response = ingress._error(400, "invalid_request")
            else:
                length = int(fields["content-length"])
                content_type = fields.get("content-type", "")
                if (
                    not 0 < length <= _MAX_HTTP_BODY
                    or not content_type
                    or not content_type.isascii()
                ):
                    status, response = ingress._error(400, "invalid_request")
                else:
                    body = await asyncio.wait_for(reader.readexactly(length), timeout=30)
                    try:
                        upload = _parse_camera_upload(content_type, body)
                    except (UnicodeDecodeError, ValueError, TypeError):
                        status, response = ingress._error(400, "invalid_request")
                    else:
                        status, response = await ingress.admit(fields.get("authorization"), upload)
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
