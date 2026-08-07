"""Versioned, strict models for the Dropbox camera consumer protocol."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


_CANDIDATE_ID_PATTERN = re.compile(r"^dropbox-camera-v1-[0-9a-f]{64}$")


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


class ManifestV1(_StrictModel):
    """Canonical manifest v1, with trusted provenance validated locally."""

    schema_version: int
    candidate_id: str
    event_id: str = Field(min_length=1)
    producer_source: str
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
    classifier_model: str = Field(min_length=1)
    classifier_prompt_version: str = Field(min_length=1)
    classifier_policy_version: str = Field(min_length=1)
    classifier_dataset_version: str = Field(min_length=1)
    classifier_output: ClassifierOutput
    food_candidate: bool
    ingest_source: str
    confirmation_required: bool
    consumption_status: str

    @field_validator("schema_version")
    @classmethod
    def _version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("unsupported manifest schema version")
        return value

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

    @field_validator("producer_source", "ingest_source")
    @classmethod
    def _source(cls, value: str) -> str:
        if value != "dropbox_camera":
            raise ValueError("unsupported manifest source")
        return value

    @field_validator("server_modified", "client_modified", "discovery_time")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("manifest datetime must include a timezone offset")
        return value

    @field_validator("food_candidate")
    @classmethod
    def _food_candidate(cls, value: bool) -> bool:
        if not value:
            raise ValueError("consumer only accepts positive candidates")
        return value

    @field_validator("confirmation_required")
    @classmethod
    def _confirmation_required(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Dropbox candidates require confirmation")
        return value

    @field_validator("consumption_status")
    @classmethod
    def _consumption_status(cls, value: str) -> str:
        if value != "unknown":
            raise ValueError("published candidate must have unknown consumption status")
        return value

    @model_validator(mode="after")
    def _identity(self) -> ManifestV1:
        if self.candidate_id != candidate_id_for(self.file_id, self.rev):
            raise ValueError("candidate_id does not match file_id and rev")
        if self.classifier_output.schema_version != 1:
            raise ValueError("unsupported classifier output version")
        return self

    @classmethod
    def from_path(cls, manifest_path: str | Path, configured_root: str | Path) -> ManifestV1:
        """Parse and verify a manifest and its named image under ``configured_root``."""
        return parse_manifest(manifest_path, configured_root)


NutritionManifest = ManifestV1
CandidateManifest = ManifestV1


def parse_manifest(manifest_path: str | Path, configured_root: str | Path) -> ManifestV1:
    """Independently validate the complete on-disk v1 candidate contract."""
    path = Path(manifest_path).expanduser().resolve()
    root = Path(configured_root).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("manifest path escapes configured nutrition root") from exc
    if path.name != "manifest.json" or path.parent == root or path.parent.name.startswith("_"):
        raise ValueError("manifest is not a candidate protocol object")
    try:
        manifest = ManifestV1.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("manifest is unreadable") from exc
    if path.parent.name != manifest.candidate_id:
        raise ValueError("candidate directory does not match manifest identity")
    image_name = Path(manifest.original_filename)
    if image_name.name != manifest.original_filename or image_name.is_absolute():
        raise ValueError("manifest image filename escapes candidate directory")
    # The producer stores immutable images as ``original<suffix>`` while the
    # manifest keeps the original Dropbox filename for provenance.  Keep the
    # legacy named-image fallback for artifacts written by early consumers.
    canonical_image_path = (path.parent / f"original{image_name.suffix}").resolve()
    image_path = (
        canonical_image_path
        if canonical_image_path.is_file()
        else (path.parent / image_name).resolve()
    )
    if image_path.parent != path.parent or not image_path.is_file():
        raise ValueError("candidate image is missing or escapes candidate directory")
    if image_path.stat().st_size != manifest.original_size_bytes:
        raise ValueError("candidate image size does not match manifest")
    if hashlib.sha256(image_path.read_bytes()).hexdigest() != manifest.original_sha256:
        raise ValueError("candidate image hash does not match manifest")
    return manifest


class ResultState(StrEnum):
    discovered = "discovered"
    classified = "classified"
    published = "published"
    prompt_sending = "prompt_sending"
    delivery_unknown = "delivery_unknown"
    pending_confirmation = "pending_confirmation"
    confirmed = "confirmed"
    declined = "declined"
    non_food = "non_food"
    estimated = "estimated"
    completed = "completed"
    retryable_error = "retryable_error"
    dead_letter = "dead_letter"
    skipped = "skipped"
    seen = "seen"


SkipReason = Literal["exif_missing", "exif_ambiguous", "exif_invalid", "exif_stale"]
SeenReason = Literal["duplicate_honcho"]
SeenFingerprintKind = Literal["sha256", "phash"]
TombstoneReason = Literal["duplicate_honcho", "expired"]
NonFoodReason = Literal["explicit_feedback", "no_visible_consumable_portion"]


def _validate_optional_audit_id(value: str | None) -> str | None:
    if value is not None and (
        not value or len(value) > 256 or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("audit message id must be a bounded printable string")
    return value


class RecipientBinding(_StrictModel):
    channel: str
    principal: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    session_key: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    honcho_binding: str | None = None


class StageAttempt(_StrictModel):
    stage: str = Field(min_length=1)
    attempt: int = Field(ge=1)
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = Field(default=None, max_length=1024)


class StateHistoryEntry(_StrictModel):
    revision: int = Field(ge=1)
    state: ResultState
    at: datetime
    error: str | None = Field(default=None, max_length=1024)


class NutritionResultSidecar(_StrictModel):
    """Consumer-owned durable result and replay ledger."""

    schema_version: int = 1
    candidate_id: str
    revision: int = Field(ge=1)
    state: ResultState
    state_history: list[StateHistoryEntry] = Field(min_length=1, max_length=32)
    recipient: RecipientBinding
    confirmation_operation_id: str = Field(min_length=1)
    meal_observation_operation_id: str = Field(min_length=1)
    prompt_message_id: int | str | None = None
    reply_message_id: int | str | None = None
    consumption_status: str = "unknown"
    attempts: list[StageAttempt] = Field(default_factory=list, max_length=32)
    emitted_honcho_message_id: str | None = None
    skip_reason: SkipReason | None = None
    seen_reason: SeenReason | None = None
    matched_honcho_message_id: str | None = Field(default=None, max_length=256)
    seen_fingerprint_kind: SeenFingerprintKind | None = None
    seen_phash_algorithm: str | None = Field(default=None, max_length=64)
    non_food_reason: NonFoodReason | None = None

    @field_validator("schema_version")
    @classmethod
    def _sidecar_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("unsupported result sidecar version")
        return value

    @field_validator("matched_honcho_message_id")
    @classmethod
    def _sidecar_matched_message_id(cls, value: str | None) -> str | None:
        return _validate_optional_audit_id(value)

    @model_validator(mode="after")
    def _sidecar_invariants(self) -> NutritionResultSidecar:
        expected_history_length = min(self.revision, 32)
        if len(self.state_history) != expected_history_length:
            raise ValueError("state history must contain the bounded revision window")
        first_revision = self.revision - expected_history_length + 1
        if [entry.revision for entry in self.state_history] != list(
            range(first_revision, self.revision + 1)
        ):
            raise ValueError("state history revisions must be monotonic and contiguous")
        if self.state_history[-1].revision != self.revision:
            raise ValueError("state history must end at current revision")
        if self.state_history[-1].state != self.state:
            raise ValueError("state history must end at current state")
        if self.confirmation_operation_id != f"{self.candidate_id}:confirm:v1":
            raise ValueError("confirmation operation id is not bound to candidate")
        if self.meal_observation_operation_id != f"{self.candidate_id}:meal-observation:v1":
            raise ValueError("meal operation id is not bound to candidate")
        if self.consumption_status not in {
            "unknown",
            "consumed",
            "planned",
            "not_consumed",
            "not_food",
        }:
            raise ValueError("invalid consumption status")
        if (
            self.state in {ResultState.confirmed, ResultState.estimated}
            and self.consumption_status != "consumed"
        ):
            raise ValueError("confirmed and estimated results require consumed status")
        if self.state == ResultState.skipped:
            if self.skip_reason is None:
                raise ValueError("skipped result requires a bounded EXIF skip reason")
            if self.consumption_status != "unknown":
                raise ValueError("skipped result requires unknown consumption status")
            if self.prompt_message_id is not None or self.reply_message_id is not None:
                raise ValueError("skipped result must not have prompt or reply ids")
            if self.emitted_honcho_message_id is not None:
                raise ValueError("skipped result must not have a Honcho message id")
        elif self.skip_reason is not None:
            raise ValueError("skip reason is only valid for skipped results")
        if self.state == ResultState.non_food:
            if self.non_food_reason is None:
                raise ValueError("non-food result requires an explicit reason")
            if self.consumption_status != "not_food":
                raise ValueError("non-food result requires not_food consumption status")
            if self.emitted_honcho_message_id is not None:
                raise ValueError("non-food result must not have a Honcho message id")
        elif self.non_food_reason is not None:
            raise ValueError("non-food reason is only valid for non-food results")
        if self.state == ResultState.seen:
            if self.seen_reason != "duplicate_honcho":
                raise ValueError("seen result requires duplicate_honcho provenance")
            if self.seen_fingerprint_kind not in {"sha256", "phash"}:
                raise ValueError("seen result requires an exact fingerprint kind")
            if self.seen_fingerprint_kind == "phash" and not self.seen_phash_algorithm:
                raise ValueError("pHash duplicate requires its exact algorithm")
            if self.seen_fingerprint_kind == "sha256" and self.seen_phash_algorithm is not None:
                raise ValueError("SHA-256 duplicate must not carry a pHash algorithm")
            if self.consumption_status != "unknown":
                raise ValueError("seen result requires unknown consumption status")
            if self.prompt_message_id is not None or self.reply_message_id is not None:
                raise ValueError("seen result must not have prompt or reply ids")
            if self.emitted_honcho_message_id is not None:
                raise ValueError("seen result must not have an emitted meal id")
        elif any(
            value is not None
            for value in (
                self.seen_reason,
                self.matched_honcho_message_id,
                self.seen_fingerprint_kind,
                self.seen_phash_algorithm,
            )
        ):
            raise ValueError("seen provenance is only valid for seen results")
        if self.state == ResultState.completed:
            if self.consumption_status == "consumed":
                if (
                    not isinstance(self.emitted_honcho_message_id, str)
                    or not self.emitted_honcho_message_id.strip()
                ):
                    raise ValueError("consumed completion requires a durable Honcho message id")
            elif self.consumption_status == "not_consumed":
                if self.emitted_honcho_message_id is not None:
                    raise ValueError("declined completion must not have a meal id")
            elif self.consumption_status == "not_food":
                raise ValueError("non-food results must use the non_food terminal state")
            else:
                raise ValueError("completion requires consumed or not_consumed status")
        return self

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(*args, **kwargs)


NutritionResult = NutritionResultSidecar


class SeenTombstoneV1(_StrictModel):
    """Compact producer-facing suppression marker under ``_seen``."""

    schema_version: Literal[1] = 1
    candidate_id: str
    original_sha256: str
    phash: str | None = Field(default=None, max_length=256)
    phash_algorithm: str | None = Field(default=None, max_length=64)
    terminal_reason: TombstoneReason
    capture_time: datetime
    archived_at: datetime
    state_summary: str = Field(min_length=1, max_length=64)
    matched_honcho_message_id: str | None = Field(default=None, max_length=256)
    matching_fingerprint_kind: SeenFingerprintKind | None = None

    @field_validator("candidate_id")
    @classmethod
    def _tombstone_candidate_id(cls, value: str) -> str:
        return validate_candidate_id(value)

    @field_validator("original_sha256")
    @classmethod
    def _tombstone_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("tombstone SHA-256 must be lowercase hexadecimal")
        return value

    @field_validator("matched_honcho_message_id")
    @classmethod
    def _tombstone_matched_message_id(cls, value: str | None) -> str | None:
        return _validate_optional_audit_id(value)

    @field_validator("phash")
    @classmethod
    def _tombstone_phash(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{1,256}", value):
            raise ValueError("tombstone pHash must be bounded lowercase hexadecimal")
        return value

    @field_validator("capture_time", "archived_at")
    @classmethod
    def _tombstone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("tombstone timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _tombstone_invariants(self) -> SeenTombstoneV1:
        if (self.phash is None) != (self.phash_algorithm is None):
            raise ValueError("tombstone pHash and algorithm must be paired")
        if self.terminal_reason == "duplicate_honcho" and self.matching_fingerprint_kind is None:
            raise ValueError("duplicate tombstone requires a matching fingerprint")
        if self.matched_honcho_message_id is not None and self.matching_fingerprint_kind is None:
            raise ValueError("matched Honcho id requires a matching fingerprint")
        if self.matching_fingerprint_kind == "phash" and self.phash is None:
            raise ValueError("pHash duplicate tombstone requires a pHash")
        return self
