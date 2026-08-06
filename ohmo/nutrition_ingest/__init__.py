"""Strict, filesystem-based Dropbox nutrition artifact consumer."""

from .models import (
    CandidateManifest,
    ManifestV1,
    NutritionManifest,
    NutritionResult,
    NutritionResultSidecar,
    RecipientBinding,
    ResultState,
    SeenTombstoneV1,
    candidate_id_for,
    parse_manifest,
    validate_candidate_id,
)
from .sidecars import NutritionResultStore
from .tombstones import SeenTombstoneStore
from .watcher import NutritionArtifactScanner, ReadyNutritionArtifact
from .coordinator import NutritionCoordinatorError, NutritionIngestCoordinator
from .metrics import NutritionMetrics
from .config import NutritionIngestConfig
from .watcher import NutritionIngestWatcher

__all__ = [
    "CandidateManifest",
    "ManifestV1",
    "NutritionArtifactScanner",
    "NutritionCoordinatorError",
    "NutritionIngestConfig",
    "NutritionIngestCoordinator",
    "NutritionMetrics",
    "NutritionIngestWatcher",
    "NutritionManifest",
    "NutritionResult",
    "NutritionResultSidecar",
    "NutritionResultStore",
    "ReadyNutritionArtifact",
    "RecipientBinding",
    "ResultState",
    "SeenTombstoneStore",
    "SeenTombstoneV1",
    "candidate_id_for",
    "parse_manifest",
    "validate_candidate_id",
]
