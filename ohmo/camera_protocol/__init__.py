"""Wire contract shared by Camera API producers and the Ohmo gateway."""

from .models import (
    ClassifierOutput,
    ExifMetadata,
    ManifestV2,
    candidate_id_for,
    parse_manifest,
    validate_candidate_id,
)

__all__ = [
    "ClassifierOutput",
    "ExifMetadata",
    "ManifestV2",
    "candidate_id_for",
    "parse_manifest",
    "validate_candidate_id",
]
