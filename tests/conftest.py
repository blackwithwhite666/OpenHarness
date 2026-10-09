"""Shared test fixtures."""

from __future__ import annotations

import os
import socket

import pytest
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


def _probe_asyncio_self_pipe() -> None:
    """Fail early when sandbox policy prevents asyncio's local wakeup writes."""
    try:
        receiver, sender = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        with receiver, sender:
            receiver.settimeout(0.5)
            sender.settimeout(0.5)
            sender.sendall(b"x")
            if receiver.recv(1) != b"x":
                raise OSError("socketpair probe did not deliver its byte")
    except OSError as exc:
        raise pytest.UsageError(
            "pytest cannot run here: asyncio self-pipe AF_UNIX socketpair send/receive "
            "is unavailable; use an operator-authorized test execution environment"
        ) from exc


def pytest_sessionstart(session: pytest.Session) -> None:
    if not getattr(session.config.option, "collectonly", False):
        _probe_asyncio_self_pipe()


@pytest.fixture(autouse=True)
def _restore_camera_synthetic_clock(request: pytest.FixtureRequest):
    """Keep the fixed-date Camera test clock local to its pytest test."""
    if request.path.name not in {"test_camera_ingress.py", "test_camera_replay_eval_capture.py"}:
        yield
        return
    import ohmo.gateway.camera as camera_module

    original_datetime = camera_module.datetime
    yield
    camera_module.datetime = original_datetime


@pytest_asyncio.fixture(autouse=True)
async def _reset_background_task_manager():
    yield
    await shutdown_task_manager()
