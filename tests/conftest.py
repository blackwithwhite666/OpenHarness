"""Shared test fixtures."""

from __future__ import annotations

import pytest
import pytest_asyncio

from openharness.tasks.manager import shutdown_task_manager


@pytest.fixture(autouse=True)
def _disable_memory_autoindex(monkeypatch):
    """Never let the memory store's auto-reindex hook touch the real search index.

    ``MemoryStore.add/update/remove`` fire a fire-and-forget ``document_search
    index`` (gated by ``OHMO_MEMORY_AUTOINDEX``). Under pytest that would spawn the
    real CLI wherever it exists — notably the self-hosted CI runner — and pollute
    the SHARED ``~/.document_search`` index with ``/tmp/pytest-*`` temp-store paths.
    Force it off for every test; the few tests that exercise the hook itself opt
    back in with their own ``monkeypatch.setenv``.
    """
    monkeypatch.setenv("OHMO_MEMORY_AUTOINDEX", "0")


@pytest_asyncio.fixture(autouse=True)
async def _reset_background_task_manager():
    yield
    await shutdown_task_manager()
