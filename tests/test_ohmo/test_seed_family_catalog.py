"""Tests for the owner-approved family catalog seed manifest."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ohmo.memory_backend import CatalogMemoryBackend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.tools.seed_family_catalog import seed_family_catalog

_OWNER_ENTRIES = (
    ("home-location", "Home location", "The family home is in Moscow."),
    (
        "elevenlabs_asr_and_tts_preference",
        "ElevenLabs ASR and TTS preference",
        "Use ElevenLabs for family voice messages.",
    ),
    (
        "marina_context_and_privacy",
        "Marina context and privacy",
        "Marina's context is private to her tenant.",
    ),
    ("marina-health", "Marina health", "Marina's health context."),
    (
        "marina_bp_baseline_and_tinnitus_context",
        "Marina BP baseline and tinnitus context",
        "Marina's blood-pressure and tinnitus baseline.",
    ),
)


def _catalog_with_owner_entries(tmp_path: Path) -> MemoryCatalog:
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    for slug, title, content in _OWNER_ENTRIES:
        assert catalog.import_entry("owner", slug, title, content).ok
    return catalog


async def test_seed_copies_manifest_with_provenance_and_is_idempotent(
    tmp_path: Path,
) -> None:
    catalog = _catalog_with_owner_entries(tmp_path)
    owner_before = catalog.list("owner", include_archived=True)

    first = seed_family_catalog(catalog)

    assert first.copied == [
        "family-shared/home-location.md",
        "family-shared/elevenlabs_asr_and_tts_preference.md",
        "marina/marina_context_and_privacy.md",
        "marina/marina-health.md",
        "marina/marina_bp_baseline_and_tinnitus_context.md",
    ]
    assert first.created == ["family-shared/falai_image_video_defaults.md"]
    assert first.skipped == []
    assert {record.slug for record in catalog.list("family-shared")} == {
        "home-location",
        "elevenlabs_asr_and_tts_preference",
        "falai_image_video_defaults",
    }
    assert {record.slug for record in catalog.list("marina")} == {
        "marina_context_and_privacy",
        "marina-health",
        "marina_bp_baseline_and_tinnitus_context",
    }

    # Seeding copies; it never moves, archives, or changes an owner row.
    assert catalog.list("owner", include_archived=True) == owner_before
    for target_tenant, source_slug in (
        ("family-shared", "home-location"),
        ("family-shared", "elevenlabs_asr_and_tts_preference"),
        ("marina", "marina_context_and_privacy"),
        ("marina", "marina-health"),
        ("marina", "marina_bp_baseline_and_tinnitus_context"),
    ):
        copied = catalog.get(target_tenant, source_slug)
        assert copied is not None
        assert copied.source_ref == f"owner/{source_slug}.md"
        if target_tenant == "family-shared":
            assert copied.provenance_kind == "explicit_shared"
            assert copied.shared_from_tenant_id == "owner"
            assert copied.shared_by_principal == "owner"
            assert copied.shared_at is not None
        else:
            assert copied.provenance_kind == "reported_about_other"
            assert copied.subject_tenant_id == "marina"

    falai = catalog.get("family-shared", "falai_image_video_defaults")
    assert falai is not None
    assert "fal-ai/flux/dev" in falai.content
    assert "Seedance Lite" in falai.content
    assert "params <model>" in falai.content
    with sqlite3.connect(catalog.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM share_ledger").fetchone()[0] == 2

    assert catalog.update("marina", "marina-health", "Marina-tenant-only copy.").ok
    owner_backend = CatalogMemoryBackend(catalog, tmp_path, tenant_id="owner")
    owner_health = await owner_backend.get("marina-health")
    assert owner_health is not None
    assert owner_health.content == "Marina's health context."
    assert all(entry.content != "Marina-tenant-only copy." for entry in await owner_backend.list())

    second = seed_family_catalog(catalog)
    assert second.copied == []
    assert second.created == []
    assert second.skipped == [
        "family-shared/home-location.md",
        "family-shared/elevenlabs_asr_and_tts_preference.md",
        "marina/marina_context_and_privacy.md",
        "marina/marina-health.md",
        "marina/marina_bp_baseline_and_tinnitus_context.md",
        "family-shared/falai_image_video_defaults.md",
    ]
    assert catalog.list("owner", include_archived=True) == owner_before
    with sqlite3.connect(catalog.db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM share_ledger").fetchone()[0] == 2


def test_missing_owner_source_is_reported_without_stopping_seed(tmp_path: Path) -> None:
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    assert catalog.import_entry(*("owner", *_OWNER_ENTRIES[0])).ok

    report = seed_family_catalog(catalog)

    assert report.copied == ["family-shared/home-location.md"]
    assert report.created == ["family-shared/falai_image_video_defaults.md"]
    assert "missing source owner/elevenlabs_asr_and_tts_preference.md" in report.skipped
    assert "missing source owner/marina-health.md" in report.skipped
    assert catalog.get("family-shared", "home-location") is not None
    assert catalog.get("family-shared", "falai_image_video_defaults") is not None
