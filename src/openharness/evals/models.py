"""Pydantic models for replayable eval/flywheel data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _validate_json_mapping(value: dict[str, Any]) -> dict[str, Any]:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be JSON-serializable") from exc
    return value


class EvalEpisode(BaseModel):
    """Replayable episode metadata for eval and data-flywheel capture."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    source: str = ""
    app: str = ""
    session_id: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    user_goal: str = ""
    user_text: str = ""
    tags: list[str] = Field(default_factory=list)
    privacy: str = "private"
    status: str = "open"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalEvent(BaseModel):
    """Append-only event belonging to a replayable episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    kind: str
    timestamp: datetime = Field(default_factory=_utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)
    tool_name: str | None = None
    tool_call_id: str | None = None
    is_error: bool = False

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("payload")
    @classmethod
    def _payload_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalResource(BaseModel):
    """Metadata-only description of a resource available to an eval episode."""

    model_config = ConfigDict(extra="forbid")

    resource_id: str
    kind: str
    name: str
    path: str | None = None
    exists: bool
    size_bytes: int | None = None
    mtime_ns: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalResourceSnapshot(BaseModel):
    """Manifest of resources captured at the start of an eval episode."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    resources: list[EvalResource] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)


class EvalUsageGraphNode(BaseModel):
    """Metadata-only node in an offline eval usage graph."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: str
    count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalUsageGraphEdge(BaseModel):
    """Counted metadata-only edge in an offline eval usage graph."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    kind: str
    count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalUsageGraphEpisodeMotif(BaseModel):
    """Per-episode event and tool paths used for offline pattern mining."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    event_kind_path: list[str] = Field(default_factory=list)
    tool_path: list[str] = Field(default_factory=list)


class EvalUsageGraphMotif(BaseModel):
    """Aggregate count for duplicate episode motifs."""

    model_config = ConfigDict(extra="forbid")

    event_kind_path: list[str] = Field(default_factory=list)
    tool_path: list[str] = Field(default_factory=list)
    count: int = 0
    episode_ids: list[str] = Field(default_factory=list)


class EvalUsageGraph(BaseModel):
    """Offline metadata-only graph built from captured eval episodes/events."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    nodes: list[EvalUsageGraphNode] = Field(default_factory=list)
    edges: list[EvalUsageGraphEdge] = Field(default_factory=list)
    episode_motifs: list[EvalUsageGraphEpisodeMotif] = Field(default_factory=list)
    motifs: list[EvalUsageGraphMotif] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalTextFacet(BaseModel):
    """Metadata-only text facet selected from an episode for mining/embedding."""

    model_config = ConfigDict(extra="forbid")

    facet_id: str
    episode_id: str
    facet_kind: str
    source_path: str
    text_hash: str
    text_length: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalEmbeddingRecord(BaseModel):
    """Dense vector for a text facet without persisting the original text."""

    model_config = ConfigDict(extra="forbid")

    facet: EvalTextFacet
    model: str
    dimensions: int
    vector: list[float]
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("vector")
    @classmethod
    def _vector_is_finite(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("vector must not be empty")
        for item in value:
            if not isinstance(item, (int, float)):
                raise ValueError("vector values must be numeric")
            if item != item or item in (float("inf"), float("-inf")):
                raise ValueError("vector values must be finite")
        return [float(item) for item in value]

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalEmbeddingManifest(BaseModel):
    """Manifest for a generated dense embedding index."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    model: str
    dimensions: int
    records_path: str
    facet_count: int
    embedding_count: int
    skipped_count: int = 0
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalCaseCandidate(BaseModel):
    """Metadata-only candidate mined from episodes for future eval cases."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    episode_id: str
    candidate_kind: str
    score: float
    signals: list[str] = Field(default_factory=list)
    facet_ids: list[str] = Field(default_factory=list)
    event_kind_path: list[str] = Field(default_factory=list)
    tool_path: list[str] = Field(default_factory=list)
    capability_path: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalCaseDraft(BaseModel):
    """Reviewable draft eval case assembled from a mined candidate."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    candidate_id: str
    episode_id: str
    case_kind: str
    input_facet_ids: list[str] = Field(default_factory=list)
    expected_facet_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    capability_path: list[str] = Field(default_factory=list)
    rubric: list[str] = Field(default_factory=list)
    scorer: str | None = None
    review_status: str = "draft"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalGoldCase(BaseModel):
    """Reviewed eval case promoted from a draft without raw private text."""

    model_config = ConfigDict(extra="forbid")

    gold_case_id: str
    case_id: str
    candidate_id: str
    episode_id: str
    case_kind: str
    input_facet_ids: list[str] = Field(default_factory=list)
    expected_facet_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    rubric: list[str] = Field(default_factory=list)
    scorer: str | None = None
    review_status: str = "approved"
    reviewer: str = ""
    promoted_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("promoted_at")
    @classmethod
    def _promoted_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalPackManifest(BaseModel):
    """Manifest for generated candidate or case draft packs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    pack_kind: str
    records_path: str
    record_count: int
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalRunPackCase(BaseModel):
    """One reviewed gold case reference inside a runnable eval pack."""

    model_config = ConfigDict(extra="forbid")

    gold_case_id: str
    case_id: str
    episode_id: str
    case_kind: str
    input_facet_ids: list[str] = Field(default_factory=list)
    expected_facet_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    rubric: list[str] = Field(default_factory=list)
    scorer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalRunPack(BaseModel):
    """Runnable metadata-only eval pack assembled from reviewed gold cases."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    pack_id: str
    source_records_path: str
    created_at: datetime = Field(default_factory=_utc_now)
    cases: list[EvalRunPackCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalSmokeReportCase(BaseModel):
    """Smoke/report-only validation result for one eval pack case."""

    model_config = ConfigDict(extra="forbid")

    gold_case_id: str
    case_id: str
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalSmokeReport(BaseModel):
    """Report-only smoke result for a runnable eval pack."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    report_kind: str = "smoke_report"
    report_id: str
    pack_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    case_count: int
    passed_count: int
    failed_count: int
    cases: list[EvalSmokeReportCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalReplayContext(BaseModel):
    """Metadata-only replay context reconstructed for one eval pack case."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    source: str = ""
    app: str = ""
    status: str = ""
    privacy: str = ""
    event_kind_path: list[str] = Field(default_factory=list)
    tool_path: list[str] = Field(default_factory=list)
    event_count: int = 0
    error_count: int = 0
    input_facet_kinds: list[str] = Field(default_factory=list)
    expected_facet_kinds: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalReplayReportCase(BaseModel):
    """Replay-runner result for one eval pack case."""

    model_config = ConfigDict(extra="forbid")

    gold_case_id: str
    case_id: str
    status: str
    checks: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    context: EvalReplayContext | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalReplayReport(BaseModel):
    """Metadata-only replay-runner report for a runnable eval pack."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    report_kind: str = "metadata_replay_report"
    report_id: str
    pack_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    case_count: int
    passed_count: int
    failed_count: int
    cases: list[EvalReplayReportCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalObservedToolCall(BaseModel):
    """Metadata-only tool call observed by an eval executor."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    call_key_hash: str
    started: bool = False
    completed: bool = False
    is_error: bool = False
    start_event_index: int | None = None
    complete_event_index: int | None = None
    input_summary_length: int = 0
    output_summary_length: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalObservedTrace(BaseModel):
    """Metadata-only observed trace produced by an eval executor."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = ""
    executor_name: str = ""
    episode_id: str = ""
    event_kind_path: list[str] = Field(default_factory=list)
    tool_path: list[str] = Field(default_factory=list)
    event_count: int = 0
    error_count: int = 0
    tool_calls: list[EvalObservedToolCall] = Field(default_factory=list)
    final_text_hash: str = ""
    final_text_length: int = 0
    error_type: str = ""
    error_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalExecutionReportCase(BaseModel):
    """Execution-runner result for one eval pack case."""

    model_config = ConfigDict(extra="forbid")

    gold_case_id: str
    case_id: str
    status: str
    score: float = 0.0
    max_score: float = 1.0
    checks: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    context: EvalReplayContext | None = None
    observed_trace: EvalObservedTrace | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalExecutionReport(BaseModel):
    """Metadata-only execution-runner report for a runnable eval pack."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    report_kind: str = "execution_report"
    report_id: str
    pack_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    case_count: int
    passed_count: int
    failed_count: int
    blocked_count: int = 0
    error_count: int = 0
    cases: list[EvalExecutionReportCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalReportComparisonCase(BaseModel):
    """Metadata-only comparison for one eval case across two execution reports."""

    model_config = ConfigDict(extra="forbid")

    gold_case_id: str
    case_id: str
    status: str
    baseline_status: str = ""
    candidate_status: str = ""
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    score_delta: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)


class EvalReportComparison(BaseModel):
    """Metadata-only regression comparison between two execution reports."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    report_kind: str = "execution_comparison_report"
    report_id: str
    baseline_report_id: str
    candidate_report_id: str
    baseline_pack_id: str
    candidate_pack_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    case_count: int
    compared_count: int
    unchanged_count: int
    improvement_count: int
    regression_count: int
    added_count: int
    removed_count: int
    baseline_passed_count: int
    candidate_passed_count: int
    baseline_non_passed_count: int
    candidate_non_passed_count: int
    score_delta: float = 0.0
    cases: list[EvalReportComparisonCase] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _created_at_is_utc(cls, value: datetime) -> datetime:
        return _normalize_timestamp(value)

    @field_validator("metadata")
    @classmethod
    def _metadata_is_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_mapping(value)
