"""Per-case memory backend provisioning for focused recall evaluations."""

from __future__ import annotations

import asyncio
import datetime as dt
import re
import shutil
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ohmo.memory_backend import (
    CatalogMemoryBackend,
    FileMemoryBackend,
    MemoryBackend,
    ShadowMemoryBackend,
)
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service.bootstrap import provision_eval_workspace
from ohmo.memory_service.honcho_client import HonchoClient
from ohmo.memory_service.outbox import drain_once
from ohmo.memory_store import MemoryStore

BackendKind = Literal["file", "catalog", "shadow"]

_BACKEND_KINDS = ("file", "catalog", "shadow")
_HONCHO_WORKSPACE_TTL = dt.timedelta(hours=1)


@dataclass
class ProvisionedBackend:
    """One isolated eval backend and the resources needed to tear it down."""

    backend: MemoryBackend
    kind: str
    workspace: Path
    honcho_workspace: str | None
    _honcho_client: HonchoClient | None = field(default=None, repr=False)
    _torn_down: bool = field(default=False, init=False, repr=False)

    async def teardown(self) -> None:
        """Best-effort clear remote data, close clients, and remove local state."""
        if self._torn_down:
            return
        self._torn_down = True

        try:
            if isinstance(self.backend, ShadowMemoryBackend):
                with suppress(Exception):
                    await self.backend.await_pending()
            await _clear_honcho_conclusions(self._honcho_client)
        finally:
            if self._honcho_client is not None:
                with suppress(Exception):
                    await self._honcho_client.aclose()
            await asyncio.to_thread(shutil.rmtree, self.workspace, True)


async def provision_backend(
    kind: BackendKind,
    *,
    run: str,
    case: str,
    sample: int,
    seed_entries: list[tuple[str, str]],
    honcho_base_url: str | None = None,
    honcho_admin_jwt: str | None = None,
) -> ProvisionedBackend:
    """Build and seed an isolated backend for one eval case/sample.

    ``honcho_admin_jwt`` is accepted only as this explicit eval-harness
    argument. It is never obtained from gateway configuration or environment.
    """
    if kind not in _BACKEND_KINDS:
        raise ValueError(f"unsupported eval memory backend: {kind!r}")
    if kind == "shadow":
        if not honcho_admin_jwt:
            raise ValueError("shadow eval backend requires explicit honcho_admin_jwt")
        if not honcho_base_url or not honcho_base_url.strip():
            raise ValueError("shadow eval backend requires explicit honcho_base_url")

    workspace = Path(
        tempfile.mkdtemp(prefix=_workspace_prefix(run=run, case=case, sample=sample))
    ).resolve()
    catalog: MemoryCatalog | None = None
    honcho_client: HonchoClient | None = None
    honcho_workspace: str | None = None

    try:
        if kind == "file":
            backend: MemoryBackend = FileMemoryBackend(MemoryStore(workspace))
        else:
            catalog = MemoryCatalog(workspace)
            catalog_backend = CatalogMemoryBackend(catalog, workspace)
            if kind == "catalog":
                backend = catalog_backend
            else:
                assert honcho_base_url is not None
                assert honcho_admin_jwt is not None
                honcho_workspace, scoped_jwt = await provision_eval_workspace(
                    base_url=honcho_base_url,
                    admin_jwt=honcho_admin_jwt,
                    run=run,
                    case=case,
                    sample=sample,
                    ttl=_HONCHO_WORKSPACE_TTL,
                )
                honcho_client = HonchoClient(
                    honcho_base_url,
                    scoped_jwt,
                    honcho_workspace,
                )
                backend = ShadowMemoryBackend(
                    catalog_backend,
                    honcho_client=honcho_client,
                )

        provisioned = ProvisionedBackend(
            backend=backend,
            kind=kind,
            workspace=workspace,
            honcho_workspace=honcho_workspace,
            _honcho_client=honcho_client,
        )
        for title, content in seed_entries:
            result = await backend.add(title, content)
            if not result.ok:
                raise ValueError(f"failed to seed memory {title!r}: {result.message}")

        if kind == "shadow":
            assert catalog is not None
            assert honcho_client is not None
            report = await drain_once(catalog, honcho_client)
            if report.retried or report.failed:
                raise RuntimeError(
                    "failed to mirror shadow eval seed into Honcho "
                    f"(retried={report.retried}, failed={report.failed})"
                )
        return provisioned
    except BaseException:
        if "provisioned" in locals():
            await provisioned.teardown()
        else:
            if honcho_client is not None:
                with suppress(Exception):
                    await honcho_client.aclose()
            await asyncio.to_thread(shutil.rmtree, workspace, True)
        raise


async def _clear_honcho_conclusions(client: HonchoClient | None) -> None:
    if client is None:
        return
    try:
        conclusions = await client.list_conclusions()
    except Exception:
        return
    for conclusion in conclusions:
        with suppress(Exception):
            await client.delete_conclusion(conclusion.id)


def _workspace_prefix(*, run: str, case: str, sample: int) -> str:
    components = (_safe_tmp_component(run), _safe_tmp_component(case), str(sample))
    return f"ohmo-eval-{'-'.join(components)}-"


def _safe_tmp_component(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value)).strip("-_")
    return (clean or "unnamed")[:32]


__all__ = ["BackendKind", "ProvisionedBackend", "provision_backend"]
