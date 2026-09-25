"""Strict versioned models for the Camera API wire contract."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


_CANDIDATE_ID_PATTERN = re.compile(r"^dropbox-camera-v1-[0-9a-f]{64}$")
CaptureTimeAuthority = Literal["exif", "filename"]


def validate_candidate_id(value: str) -> str:
    """Validate one protocol candidate id without accepting path syntax."""
    if not isinstance(value, str) or not _CANDIDATE_ID_PATTERN.fullmatch(value):
        raise ValueError("invalid candidate_id")
    return value


def candidate_id_for(file_id: str, rev: str) -> str:
    """Return the canonical identity for one Dropbox file revision."""
    if not isinstance(file_id, str) or not file_id:
        raise ValueError("file_id must be a non-empty string")
    if not isinstance(rev, str) or not rev:
        raise ValueError("rev must be a non-empty string")
    digest = hashlib.sha256(f"{file_id}\0{rev}".encode()).hexdigest()
    return f"dropbox-camera-v1-{digest}"


class ExifMetadata(_StrictModel):
    timezone_status: str
    raw_datetime_original: str | None = Field(default=None, max_length=64)
    offset_time_original: str | None = Field(default=None, max_length=32)
    raw_datetime_digitized: str | None = Field(default=None, max_length=64)
    offset_time_digitized: str | None = Field(default=None, max_length=32)
    raw_capture_time: str | None = Field(default=None, max_length=64)
    capture_timezone_offset: str | None = Field(default=None, max_length=32)
    normalized_capture_time: str | None = Field(default=None, max_length=64)
    orientation: int | None = Field(default=None, ge=1, le=8)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)

    @field_validator("timezone_status")
    @classmethod
    def _timezone_status(cls, value: str) -> str:
        if value not in {"known", "missing", "ambiguous"}:
            raise ValueError("timezone_status must be known, missing, or ambiguous")
        return value


class ClassifierOutput(_StrictModel):
    schema_version: int
    decision: str
    score: float | None = Field(default=None, ge=0, le=1)
    explanation: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _schema_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("unsupported classifier output version")
        return value

    @field_validator("decision")
    @classmethod
    def _decision(cls, value: str) -> str:
        if value not in {"food", "ambiguous"}:
            raise ValueError("classifier decision must be food or ambiguous")
        return value


class ManifestV2(_StrictModel):
    """Strict one-image manifest with explicit capture-time provenance."""

    schema_version: Literal[2]
    candidate_id: str
    event_id: str = Field(min_length=1)
    producer_source: Literal["dropbox_camera"]
    file_id: str = Field(min_length=1)
    rev: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    server_modified: datetime
    client_modified: datetime | None = None
    discovery_time: datetime
    original_filename: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    original_size_bytes: int = Field(gt=0)
    original_sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    exif: ExifMetadata
    normalized_capture_time: datetime
    capture_time_authority: CaptureTimeAuthority
    classifier_model: str = Field(min_length=1)
    classifier_prompt_version: str = Field(min_length=1)
    classifier_policy_version: str = Field(min_length=1)
    classifier_dataset_version: str = Field(min_length=1)
    classifier_route_attestation_json: str | None = Field(default=None, max_length=10000)
    classifier_output: ClassifierOutput
    food_candidate: Literal[True]
    ingest_source: Literal["dropbox_camera"]
    confirmation_required: Literal[True]
    consumption_status: Literal["unknown"]

    @field_validator("candidate_id")
    @classmethod
    def _candidate_id_shape(cls, value: str) -> str:
        return validate_candidate_id(value)

    @field_validator("original_sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("original_sha256 must be lowercase SHA-256")
        return value

    @field_validator(
        "server_modified", "client_modified", "discovery_time", "normalized_capture_time"
    )
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("manifest datetime must include a timezone offset")
        return value

    @model_validator(mode="after")
    def _identity_and_capture(self) -> ManifestV2:
        if self.candidate_id != candidate_id_for(self.file_id, self.rev):
            raise ValueError("candidate_id does not match file_id and rev")
        if self.classifier_output.schema_version != 1:
            raise ValueError("unsupported classifier output version")
        if self.capture_time_authority == "exif":
            if self.exif.timezone_status == "ambiguous" or not self.exif.normalized_capture_time:
                raise ValueError("EXIF capture authority is unavailable")
            if (
                datetime.fromisoformat(self.exif.normalized_capture_time)
                != self.normalized_capture_time
            ):
                raise ValueError("manifest capture time disagrees with EXIF")
        elif self.exif.timezone_status != "ambiguous" and self.exif.normalized_capture_time:
            try:
                exif_capture = datetime.fromisoformat(self.exif.normalized_capture_time)
            except ValueError:
                pass
            else:
                if exif_capture.tzinfo is not None and exif_capture.utcoffset() is not None:
                    raise ValueError("usable EXIF capture time must take priority over filename")
        return self

    @classmethod
    def from_path(cls, manifest_path: str | Path, configured_root: str | Path) -> ManifestV2:
        return parse_manifest(manifest_path, configured_root)


def parse_manifest(manifest_path: str | Path, configured_root: str | Path) -> ManifestV2:
    """Validate a v2 manifest and its named image below the configured root."""
    path = Path(manifest_path).expanduser().resolve()
    root = Path(configured_root).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("manifest path escapes configured camera root") from exc
    if path.name != "manifest.json" or path.parent == root or path.parent.name.startswith("_"):
        raise ValueError("manifest is not a candidate protocol object")
    try:
        manifest = ManifestV2.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is unreadable") from exc
    if path.parent.name != manifest.candidate_id:
        raise ValueError("candidate directory does not match manifest identity")
    image_name = Path(manifest.original_filename)
    if image_name.name != manifest.original_filename or image_name.is_absolute():
        raise ValueError("manifest image filename escapes candidate directory")
    canonical_image = (path.parent / f"original{image_name.suffix}").resolve()
    image_path = (
        canonical_image if canonical_image.is_file() else (path.parent / image_name).resolve()
    )
    if image_path.parent != path.parent or not image_path.is_file():
        raise ValueError("candidate image is missing or escapes candidate directory")
    if image_path.stat().st_size != manifest.original_size_bytes:
        raise ValueError("candidate image size does not match manifest")
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != manifest.original_sha256:
        raise ValueError("candidate image hash does not match manifest")
    return manifest


__all__ = [
    "CaptureTimeAuthority",
    "ClassifierOutput",
    "ExifMetadata",
    "ManifestV2",
    "candidate_id_for",
    "parse_manifest",
    "validate_candidate_id",
]
