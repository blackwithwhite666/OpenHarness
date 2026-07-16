"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest_asyncio

from openharness.tasks.manager import shutdown_task_manager

# The memory store's auto-reindex hook (``MemoryStore.add/update/remove`` ->
# fire-and-forget ``document_search index``, gated by ``OHMO_MEMORY_AUTOINDEX``) must
# never run in tests: on a host that HAS document_search-cli — notably the
# self-hosted CI runner — it would spawn the real CLI and pollute the SHARED
# ``~/.document_search`` index with ``/tmp/pytest-*`` temp-store paths. Force it off
# for the whole session at import time. This is a plain module-level default, NOT a
# per-test autouse fixture, on purpose: a function-scoped autouse fixture perturbs
# pytest-asyncio's per-test event-loop teardown ordering (observed as a teardown
# ``generator raised StopIteration`` / ``Event loop is closed`` on some async tests).
# The few tests that exercise the hook opt back in with their own ``monkeypatch.setenv``.
os.environ["OHMO_MEMORY_AUTOINDEX"] = "0"


@pytest_asyncio.fixture(autouse=True)
async def _reset_background_task_manager():
    yield
    await shutdown_task_manager()
