"""Strict, filesystem-based Dropbox nutrition artifact consumer."""

from .models import (
    CandidateManifest,
    ManifestV1,
    NutritionManifest,
    NutritionResult,
    NutritionResultSidecar,
    RecipientBinding,
    ResultState,
    candidate_id_for,
    parse_manifest,
)
from .sidecars import NutritionResultStore
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
    "candidate_id_for",
    "parse_manifest",
]
