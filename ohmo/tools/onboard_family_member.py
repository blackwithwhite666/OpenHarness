"""Consent-gated provisioning for one private family memory tenant."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import re
from collections.abc import Sequence
from pathlib import Path

from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service.bootstrap import TenantOnboarding, onboard_tenant


_TENANT_ID_RE = re.compile(r"[a-z0-9_-]+")
_NUMERIC_PRINCIPAL_RE = re.compile(r"[0-9]+")
_DEFAULT_RUNTIME_KEY_TTL = dt.timedelta(days=365)
_CONSENT_NOTE = "Operator confirmed consent with --i-have-consent."


async def onboard_family_member(
    catalog: MemoryCatalog,
    *,
    tenant_id: str,
    person_peer: str,
    principal_ids: Sequence[str],
    workspace: str,
    base_url: str,
    admin_jwt: str,
    i_have_consent: bool,
    consent_note: str = _CONSENT_NOTE,
    ttl: dt.timedelta | int | float = _DEFAULT_RUNTIME_KEY_TTL,
) -> dict[str, object]:
    """Record consent, provision Honcho, and return ready-to-paste config."""
    clean_tenant_id = _validate_tenant_id(tenant_id)
    clean_principals = _validate_principal_ids(principal_ids)
    if not i_have_consent:
        raise PermissionError("--i-have-consent is required for family onboarding")

    catalog.ensure_tenant(clean_tenant_id, "private")
    for principal in clean_principals:
        catalog.record_consent(clean_tenant_id, principal, consent_note)
    if not catalog.has_consent(clean_tenant_id):
        raise PermissionError(
            f"tenant {clean_tenant_id!r} cannot be enabled without recorded consent"
        )

    onboarding = await onboard_tenant(
        base_url=base_url,
        admin_jwt=admin_jwt,
        workspace=workspace,
        person_peer=person_peer,
        ttl=ttl,
    )
    return build_gateway_config_snippet(
        catalog,
        tenant_id=clean_tenant_id,
        principal_ids=clean_principals,
        onboarding=onboarding,
    )


def build_gateway_config_snippet(
    catalog: MemoryCatalog,
    *,
    tenant_id: str,
    principal_ids: Sequence[str],
    onboarding: TenantOnboarding,
) -> dict[str, object]:
    """Build only consented runtime config; never mutate ``gateway.json``."""
    clean_tenant_id = _validate_tenant_id(tenant_id)
    clean_principals = _validate_principal_ids(principal_ids)
    if not catalog.has_consent(clean_tenant_id):
        raise PermissionError(
            f"tenant {clean_tenant_id!r} cannot be enabled without recorded consent"
        )
    return {
        "family_principals": {
            principal: clean_tenant_id for principal in clean_principals
        },
        "enabled_memory_tenants": [clean_tenant_id],
        "tenant_honcho": {
            clean_tenant_id: {
                "workspace": onboarding.workspace,
                "api_key": onboarding.jwt,
                "observed_peer": onboarding.observed_peer,
            }
        },
    }


def _validate_tenant_id(tenant_id: str) -> str:
    clean = (tenant_id or "").strip()
    if clean == "owner":
        raise ValueError("the owner tenant is reserved")
    if _TENANT_ID_RE.fullmatch(clean) is None:
        raise ValueError("tenant_id must match [a-z0-9_-]+")
    return clean


def _validate_principal_ids(principal_ids: Sequence[str]) -> tuple[str, ...]:
    clean = tuple(dict.fromkeys(str(value).strip() for value in principal_ids))
    if not clean or any(_NUMERIC_PRINCIPAL_RE.fullmatch(value) is None for value in clean):
        raise ValueError("principal ids must be canonical numeric ids")
    return clean


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consent-gated family onboarding. Prints JSON to paste into gateway.json; "
            "it never edits the file."
        ),
    )
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--person-peer", required=True)
    parser.add_argument(
        "--principal-id",
        "--principal-ids",
        dest="principal_ids",
        required=True,
        action="extend",
        nargs="+",
        help="One or more canonical numeric principal ids.",
    )
    parser.add_argument("--workspace", required=True, help="This tenant's Honcho workspace.")
    parser.add_argument("--base-url", required=True, help="Honcho service base URL.")
    parser.add_argument("--admin-jwt", required=True, help="Explicit Honcho admin JWT.")
    parser.add_argument(
        "--catalog-workspace",
        help="Local ohmo workspace containing the catalog (defaults to ~/.ohmo).",
    )
    parser.add_argument(
        "--i-have-consent",
        action="store_true",
        help="Confirm the person consented; required for onboarding.",
    )
    return parser


async def _run_cli(args: argparse.Namespace) -> dict[str, object]:
    catalog_workspace = (
        Path(args.catalog_workspace) if args.catalog_workspace is not None else None
    )
    return await onboard_family_member(
        MemoryCatalog(catalog_workspace),
        tenant_id=args.tenant_id,
        person_peer=args.person_peer,
        principal_ids=args.principal_ids,
        workspace=args.workspace,
        base_url=args.base_url,
        admin_jwt=args.admin_jwt,
        i_have_consent=args.i_have_consent,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.i_have_consent:
        parser.error("--i-have-consent is required for family onboarding")
    snippet = asyncio.run(_run_cli(args))
    print(json.dumps(snippet, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "build_gateway_config_snippet",
    "main",
    "onboard_family_member",
]
