"""Per-tenant local DR export and owner-protected deletion.

Deleting a catalog tenant does not delete its Honcho workspace: the runtime has
no admin credential. After local deletion, an operator must separately delete
the printed workspace with Honcho administration credentials.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ohmo.memory_catalog import (
    MemoryCatalog,
    delete_tenant as _delete_catalog_tenant,
    export_tenant as _export_catalog_tenant,
)


def export_tenant(catalog: MemoryCatalog, tenant_id: str) -> dict[str, object]:
    """Return only the requested tenant's deterministic local DR payload."""
    return _export_catalog_tenant(catalog, tenant_id)


def delete_tenant(catalog: MemoryCatalog, tenant_id: str) -> bool:
    """Delete a non-owner tenant locally and leave Honcho deletion to an operator."""
    return _delete_catalog_tenant(catalog, tenant_id)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-workspace",
        required=True,
        help="Local ohmo workspace containing the catalog.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    export_parser = subparsers.add_parser("export", help="Print a tenant DR export as JSON.")
    export_parser.add_argument("--tenant-id", required=True)
    export_parser.add_argument("--output", help="Optional JSON output path.")

    delete_parser = subparsers.add_parser("delete", help="Delete a local non-owner tenant.")
    delete_parser.add_argument("--tenant-id", required=True)
    delete_parser.add_argument(
        "--honcho-workspace",
        required=True,
        help="Workspace to print for the separate manual Honcho deletion step.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    catalog = MemoryCatalog(Path(args.catalog_workspace))
    if args.action == "export":
        payload = export_tenant(catalog, args.tenant_id)
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return

    deleted = delete_tenant(catalog, args.tenant_id)
    print(
        json.dumps(
            {
                "deleted": deleted,
                "tenant_id": args.tenant_id,
                "manual_honcho_workspace_delete": args.honcho_workspace,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = ["delete_tenant", "export_tenant", "main"]
