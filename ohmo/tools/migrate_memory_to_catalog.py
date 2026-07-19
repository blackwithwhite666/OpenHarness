"""Migrate the legacy Markdown memory store to the SQLite catalog.

The importer is intentionally local and non-destructive: migration reads the
legacy ``.md`` files without changing them, while rollback exports catalog rows
to a separate, explicit destination directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ohmo.memory_catalog import CatalogRecord, MemoryCatalog
from ohmo.memory_store import MemoryStore
from ohmo.threat_patterns import first_threat_message
from ohmo.workspace import get_memory_dir

ActiveInventoryEntry = tuple[str, str, str]
ArchivedInventoryEntry = tuple[str, str, str, bool]


@dataclass
class MigrationReport:
    """Reconciliation counters and warnings from one migration attempt."""

    active_imported: int = 0
    archived_imported: int = 0
    skipped_duplicate: int = 0
    skipped_threat: int = 0
    conflicts: list[str] = field(default_factory=list)
    archived_title_fallbacks: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render a compact, human-readable migration summary."""
        conflicts = ", ".join(self.conflicts) if self.conflicts else "none"
        fallbacks = (
            ", ".join(self.archived_title_fallbacks)
            if self.archived_title_fallbacks
            else "none"
        )
        return "\n".join(
            (
                "Memory migration report",
                f"  Active imported/planned: {self.active_imported}",
                f"  Archived imported/planned: {self.archived_imported}",
                f"  Skipped duplicates: {self.skipped_duplicate}",
                f"  Skipped threats: {self.skipped_threat}",
                f"  Conflicts: {conflicts}",
                f"  Archived title fallbacks: {fallbacks}",
            )
        )


def inventory(
    workspace: str | Path | None,
) -> tuple[list[ActiveInventoryEntry], list[ArchivedInventoryEntry]]:
    """Read active and archived entries from the legacy Markdown store."""
    active = [
        (entry.slug, entry.title, entry.content)
        for entry in MemoryStore(workspace).list()
    ]

    archived: list[ArchivedInventoryEntry] = []
    archive_dir = get_memory_dir(workspace) / "archive"
    if archive_dir.exists():
        for path in sorted(archive_dir.glob("*.md")):
            if path.is_symlink() or not path.is_file():
                continue
            archived.append(
                (
                    path.stem,
                    path.stem,
                    path.read_text(encoding="utf-8", errors="replace").strip(),
                    True,
                )
            )
    return active, archived


def migrate(
    workspace: str | Path | None,
    *,
    dry_run: bool = True,
    db_path: str | Path | None = None,
) -> MigrationReport:
    """Plan or apply a non-destructive Markdown-to-catalog migration."""
    active, archived = inventory(workspace)
    report = MigrationReport()
    catalog = None if dry_run else MemoryCatalog(workspace, db_path=db_path)

    for slug, title, content in active:
        _migrate_entry(
            catalog,
            report,
            slug=slug,
            title=title,
            content=content,
            archive_status="active",
            dry_run=dry_run,
        )

    for slug, title, content, title_is_fallback in archived:
        if title_is_fallback:
            report.archived_title_fallbacks.append(slug)
        _migrate_entry(
            catalog,
            report,
            slug=slug,
            title=title,
            content=content,
            archive_status="archived",
            dry_run=dry_run,
        )

    return report


def _migrate_entry(
    catalog: MemoryCatalog | None,
    report: MigrationReport,
    *,
    slug: str,
    title: str,
    content: str,
    archive_status: str,
    dry_run: bool,
) -> None:
    threat = first_threat_message(f"{title}\n{content}", scope="all")
    if threat:
        report.skipped_threat += 1
        return

    if dry_run:
        if archive_status == "active":
            report.active_imported += 1
        else:
            report.archived_imported += 1
        return

    assert catalog is not None
    result = catalog.import_entry(
        "owner",
        slug,
        title,
        content,
        archive_status=archive_status,
    )
    if not result.ok:
        report.conflicts.append(slug)
    elif result.message.startswith("Already imported "):
        report.skipped_duplicate += 1
    elif archive_status == "active":
        report.active_imported += 1
    else:
        report.archived_imported += 1


def export_catalog_to_files(catalog: MemoryCatalog, dest_dir: str | Path) -> None:
    """Reconstruct a legacy file-store layout under an explicit destination."""
    dest = Path(dest_dir).expanduser().resolve()
    live_store = catalog._memory_dir.expanduser().resolve()
    if dest == live_store or live_store in dest.parents:
        raise ValueError("Export destination must be separate from the live memory store.")
    if dest == catalog.db_path.expanduser().resolve().parent:
        raise ValueError("Export destination must be separate from the catalog directory.")

    dest.mkdir(parents=True, exist_ok=True)
    archive_dir = dest / "archive"
    if archive_dir.is_symlink():
        raise ValueError("Export archive directory must not be a symlink.")
    archive_dir.mkdir(parents=True, exist_ok=True)

    records = catalog.list("owner", include_archived=True)
    active_records = [record for record in records if record.archive_status == "active"]
    archived_records = [record for record in records if record.archive_status == "archived"]

    for record in active_records:
        _write_record(record, dest)
    for record in archived_records:
        _write_record(record, archive_dir)

    index_lines = ["# Memory Index", ""]
    index_lines.extend(f"- [{record.title}]({record.slug}.md)" for record in active_records)
    _safe_output_path(dest, "MEMORY.md").write_text(
        "\n".join(index_lines) + "\n",
        encoding="utf-8",
    )


def _write_record(record: CatalogRecord, directory: Path) -> None:
    _safe_output_path(directory, f"{record.slug}.md").write_text(
        record.content,
        encoding="utf-8",
    )


def _safe_output_path(directory: Path, filename: str) -> Path:
    resolved_directory = directory.resolve()
    target = (resolved_directory / filename).resolve()
    if target.parent != resolved_directory:
        raise ValueError(f"Unsafe catalog slug cannot be exported: {filename!r}.")
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate an ohmo Markdown memory store to its local catalog.",
    )
    parser.add_argument("--workspace", required=True, help="Path to the ohmo workspace.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration (the default only prints a dry-run plan).",
    )
    parser.add_argument(
        "--export",
        metavar="DEST",
        help="Also export the resulting/current catalog to a separate directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    report = migrate(args.workspace, dry_run=not args.apply)
    print(report.render())

    if args.export:
        live_store = get_memory_dir(args.workspace).resolve()
        export_dest = Path(args.export).expanduser().resolve()
        if export_dest == live_store or live_store in export_dest.parents:
            raise SystemExit("Refusing to export into the live Markdown memory store.")
        export_catalog_to_files(MemoryCatalog(args.workspace), export_dest)
        print(f"Exported catalog to {export_dest}")


if __name__ == "__main__":
    main()


__all__ = [
    "MigrationReport",
    "export_catalog_to_files",
    "inventory",
    "main",
    "migrate",
]
