"""Seed the owner-approved family catalog entries.

This maintenance operation only inserts copies into target tenants. It never
updates, archives, or removes the owner's source rows.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ohmo.memory_catalog import MemoryCatalog

_SHARED_TENANT = "family-shared"
_MARINA_TENANT = "marina"
_SHARED_SLUGS = (
    "home-location.md",
    "elevenlabs_asr_and_tts_preference.md",
)
_MARINA_SLUGS = (
    "marina_context_and_privacy.md",
    "marina-health.md",
    "marina_bp_baseline_and_tinnitus_context.md",
)
_FALAI_SLUG = "falai_image_video_defaults.md"
_FALAI_TITLE = "fal.ai image/video defaults"
_FALAI_CONTENT = (
    "Use the `falai` skill / `falai-cli` for fal.ai media generation through the "
    "SOCKS5 cloud proxy. The default text-to-image model is `fal-ai/flux/dev`; the "
    "default text-to-video and image-to-video model is Seedance Lite. `models` lists "
    "the baked models, and `params <model>` prints the live OpenAPI parameters."
)


@dataclass
class SeedReport:
    """Inserted and skipped entries from one family-catalog seed attempt."""

    copied: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render a compact maintenance report."""

        def entries(values: list[str]) -> str:
            return ", ".join(values) if values else "none"

        return "\n".join(
            (
                "Family catalog seed report",
                f"  Copied: {entries(self.copied)}",
                f"  Created: {entries(self.created)}",
                f"  Skipped: {entries(self.skipped)}",
            )
        )


def seed_family_catalog(
    catalog: MemoryCatalog,
    *,
    owner_tenant: str = "owner",
) -> SeedReport:
    """Copy the curated D8a manifest into its target tenants idempotently."""
    catalog.ensure_tenant(_SHARED_TENANT, "shared")
    catalog.ensure_tenant(_MARINA_TENANT, "private")
    report = SeedReport()

    for source_slug in _SHARED_SLUGS:
        _copy_manifest_entry(
            catalog,
            report,
            owner_tenant=owner_tenant,
            source_slug=source_slug,
            target_tenant=_SHARED_TENANT,
            provenance_kind="explicit_shared",
        )

    for source_slug in _MARINA_SLUGS:
        _copy_manifest_entry(
            catalog,
            report,
            owner_tenant=owner_tenant,
            source_slug=source_slug,
            target_tenant=_MARINA_TENANT,
            provenance_kind="reported_about_other",
        )

    falai_target = _qualified(_SHARED_TENANT, _FALAI_SLUG)
    if catalog.get(_SHARED_TENANT, _FALAI_SLUG) is not None:
        report.skipped.append(falai_target)
    else:
        result = catalog.import_entry(
            _SHARED_TENANT,
            _FALAI_SLUG,
            _FALAI_TITLE,
            _FALAI_CONTENT,
            provenance_kind="maintenance_import",
            source_principal=owner_tenant,
        )
        if result.message.startswith("Already imported "):
            report.skipped.append(falai_target)
        elif not result.ok:
            raise RuntimeError(result.message)
        else:
            report.created.append(falai_target)

    return report


def _copy_manifest_entry(
    catalog: MemoryCatalog,
    report: SeedReport,
    *,
    owner_tenant: str,
    source_slug: str,
    target_tenant: str,
    provenance_kind: str,
) -> None:
    target = _qualified(target_tenant, source_slug)
    if catalog.get(target_tenant, source_slug) is not None:
        report.skipped.append(target)
        return

    source = catalog.get(owner_tenant, source_slug)
    if source is None:
        report.skipped.append(f"missing source {_qualified(owner_tenant, source_slug)}")
        return

    timestamp = _utc_timestamp()
    shared = provenance_kind == "explicit_shared"
    result = catalog.import_entry(
        target_tenant,
        source.slug,
        source.title,
        source.content,
        source=source.source,
        provenance_kind=provenance_kind,
        source_principal=owner_tenant,
        subject_tenant_id=target_tenant if not shared else None,
        source_ref=_qualified(owner_tenant, source_slug),
        shared_from_tenant_id=owner_tenant if shared else None,
        shared_by_principal=owner_tenant if shared else None,
        shared_at=timestamp if shared else None,
    )
    if result.message.startswith("Already imported "):
        report.skipped.append(target)
        return
    if not result.ok:
        raise RuntimeError(result.message)

    if shared:
        digest = hashlib.sha256(source.content.encode("utf-8")).hexdigest()
        catalog.record_share(
            op_id=f"seed-family-catalog-v1:{target_tenant}:{source.slug}",
            source_tenant_id=owner_tenant,
            source_slug=source.slug,
            source_digest=digest,
            actor_principal=owner_tenant,
            target_slug=source.slug,
            created_at=timestamp,
        )
    report.copied.append(target)


def _qualified(tenant_id: str, slug: str) -> str:
    name = slug if slug.lower().endswith(".md") else f"{slug}.md"
    return f"{tenant_id}/{name}"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed the owner-approved entries into the family memory catalog.",
    )
    parser.add_argument("--workspace", required=True, help="Path to the ohmo workspace.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    report = seed_family_catalog(MemoryCatalog(Path(args.workspace)))
    print(report.render())


if __name__ == "__main__":
    main()


__all__ = ["SeedReport", "main", "seed_family_catalog"]
