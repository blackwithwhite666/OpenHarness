from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ohmo.nutrition_ingest.models import (
    ManifestV1,
    NutritionResultSidecar,
    RecipientBinding,
    ResultState,
    StageAttempt,
    StateHistoryEntry,
    candidate_id_for,
)

FIXTURE = Path(__file__).parents[2] / "ohmo/nutrition_ingest/manifest_v1_fixture.json"


def test_canonical_fixture_is_strict_and_identity_stable() -> None:
    payload = json.loads(FIXTURE.read_text())
    manifest = ManifestV1.model_validate(payload)
    assert manifest.candidate_id == candidate_id_for(manifest.file_id, manifest.rev)


@pytest.mark.parametrize(
    "field_path",
    [
        ("schema_version",),
        ("candidate_id",),
        ("event_id",),
        ("producer_source",),
        ("file_id",),
        ("rev",),
        ("source_path",),
        ("server_modified",),
        ("discovery_time",),
        ("original_filename",),
        ("mime_type",),
        ("original_size_bytes",),
        ("original_sha256",),
        ("width",),
        ("height",),
        ("exif",),
        ("classifier_model",),
        ("classifier_prompt_version",),
        ("classifier_policy_version",),
        ("classifier_dataset_version",),
        ("classifier_output",),
        ("food_candidate",),
        ("ingest_source",),
        ("confirmation_required",),
        ("consumption_status",),
        ("classifier_output", "schema_version"),
    ],
)
def test_manifest_rejects_every_schema_required_field_when_absent(field_path) -> None:
    payload = json.loads(FIXTURE.read_text())
    value = payload
    for key in field_path[:-1]:
        value = value[key]
    value.pop(field_path[-1])

    with pytest.raises(ValidationError):
        ManifestV1.model_validate(payload)


@pytest.mark.parametrize("field", ["server_modified", "client_modified", "discovery_time"])
def test_manifest_rejects_naive_datetime(field: str) -> None:
    payload = json.loads(FIXTURE.read_text())
    payload[field] = "2026-08-05T10:00:00"

    with pytest.raises(ValidationError):
        ManifestV1.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(unknown_field=True),
        lambda value: value.update(schema_version=2),
        lambda value: value.update(candidate_id="dropbox-camera-v1-" + "0" * 64),
        lambda value: value.update(consumption_status="consumed"),
    ],
)
def test_manifest_rejects_untrusted_or_unsupported_mutations(mutation) -> None:
    payload = json.loads(FIXTURE.read_text())
    mutation(payload)
    with pytest.raises(ValidationError):
        ManifestV1.model_validate(payload)


def _result(state: ResultState, revision: int = 1) -> NutritionResultSidecar:
    candidate = "dropbox-camera-v1-" + "1" * 64
    return NutritionResultSidecar(
        candidate_id=candidate,
        revision=revision,
        state=state,
        state_history=[
            StateHistoryEntry(revision=revision, state=state, at="2026-08-05T10:00:00Z")
        ],
        recipient=RecipientBinding(
            channel="telegram",
            principal="123",
            chat_id="123",
            session_key="telegram:123",
            tenant_id="marina",
        ),
        confirmation_operation_id=f"{candidate}:confirm:v1",
        meal_observation_operation_id=f"{candidate}:meal-observation:v1",
    )


def test_result_sidecar_has_stable_operation_ids_and_bounded_history() -> None:
    result = _result(ResultState.published)
    assert result.confirmation_operation_id.endswith(":confirm:v1")
    assert result.meal_observation_operation_id.endswith(":meal-observation:v1")


def test_skipped_sidecar_has_terminal_invariants() -> None:
    payload = _result(ResultState.published).model_dump(mode="json")
    payload.update(
        state=ResultState.skipped,
        revision=2,
        consumption_status="unknown",
        prompt_message_id=None,
        reply_message_id=None,
        emitted_honcho_message_id=None,
        attempts=[
            StageAttempt(
                stage="prompt",
                attempt=1,
                started_at="2026-08-05T10:00:00Z",
                finished_at="2026-08-05T10:00:01Z",
                error="temporary failure",
            ).model_dump(mode="json")
        ],
        skip_reason="exif_stale",
        state_history=[
            *payload["state_history"],
            StateHistoryEntry(
                revision=2, state=ResultState.skipped, at="2026-08-05T10:01:00Z"
            ).model_dump(mode="json"),
        ],
    )
    result = NutritionResultSidecar.model_validate(payload)
    assert result.state == ResultState.skipped
    assert result.skip_reason == "exif_stale"


@pytest.mark.parametrize("skip_reason", ["bad", "", None])
def test_skipped_sidecar_rejects_unbounded_or_missing_reason(skip_reason) -> None:
    payload = _result(ResultState.published).model_dump(mode="json")
    payload.update(
        state=ResultState.skipped,
        revision=2,
        skip_reason=skip_reason,
        state_history=[
            *payload["state_history"],
            StateHistoryEntry(
                revision=2, state=ResultState.skipped, at="2026-08-05T10:01:00Z"
            ).model_dump(mode="json"),
        ],
    )
    with pytest.raises(ValidationError):
        NutritionResultSidecar.model_validate(payload)


def test_seen_sidecar_enforces_terminal_unknown_consumption_invariants() -> None:
    payload = _result(ResultState.published).model_dump(mode="json")
    payload.update(
        state=ResultState.seen,
        revision=2,
        seen_reason="duplicate_honcho",
        seen_fingerprint_kind="sha256",
        matched_honcho_message_id="honcho-message",
        state_history=[
            *payload["state_history"],
            StateHistoryEntry(
                revision=2,
                state=ResultState.seen,
                at="2026-08-05T10:01:00Z",
            ).model_dump(mode="json"),
        ],
    )
    result = NutritionResultSidecar.model_validate(payload)
    assert result.consumption_status == "unknown"
    assert result.prompt_message_id is None
    assert result.reply_message_id is None
    assert result.emitted_honcho_message_id is None

    for mutation in (
        {"consumption_status": "consumed"},
        {"prompt_message_id": 42},
        {"emitted_honcho_message_id": "meal"},
        {"seen_fingerprint_kind": "phash", "seen_phash_algorithm": None},
    ):
        with pytest.raises(ValidationError):
            NutritionResultSidecar.model_validate({**payload, **mutation})


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (ResultState.prompt_sending, ResultState.delivery_unknown),
        (ResultState.delivery_unknown, ResultState.prompt_sending),
        (ResultState.delivery_unknown, ResultState.dead_letter),
        (ResultState.published, ResultState.skipped),
        (ResultState.prompt_sending, ResultState.skipped),
        (ResultState.delivery_unknown, ResultState.skipped),
        (ResultState.pending_confirmation, ResultState.skipped),
        (ResultState.retryable_error, ResultState.skipped),
    ],
)
def test_delivery_unknown_has_explicit_reconciliation_transitions(previous, current) -> None:
    NutritionResultSidecar.model_validate(_result(previous).model_dump(mode="json"))
    from ohmo.nutrition_ingest.sidecars import NutritionResultStore

    NutritionResultStore._validate_transition(previous, current)


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (ResultState.retryable_error, ResultState.delivery_unknown),
        (ResultState.delivery_unknown, ResultState.pending_confirmation),
        (ResultState.skipped, ResultState.published),
    ],
)
def test_delivery_unknown_cannot_be_automatically_retried(previous, current) -> None:
    from ohmo.nutrition_ingest.sidecars import NutritionResultStore

    with pytest.raises(ValueError, match="illegal nutrition state transition"):
        NutritionResultStore._validate_transition(previous, current)


@pytest.mark.parametrize(
    "state,consumption_status,emitted_honcho_message_id",
    [
        (ResultState.confirmed, "unknown", None),
        (ResultState.estimated, "unknown", None),
        (ResultState.completed, "consumed", None),
        (ResultState.completed, "not_consumed", "honcho-message"),
        (ResultState.completed, "planned", None),
    ],
)
def test_result_sidecar_enforces_durable_completion_invariants(
    state, consumption_status, emitted_honcho_message_id
) -> None:
    payload = _result(ResultState.published).model_dump(mode="json")
    payload.update(
        state=state,
        consumption_status=consumption_status,
        emitted_honcho_message_id=emitted_honcho_message_id,
        state_history=[
            payload["state_history"][0],
            StateHistoryEntry(
                revision=2,
                state=state,
                at="2026-08-05T10:01:00Z",
            ).model_dump(mode="json"),
        ],
        revision=2,
    )
    with pytest.raises(ValidationError):
        NutritionResultSidecar.model_validate(payload)
