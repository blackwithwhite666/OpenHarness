"""Fail-fast coverage for pytest's local asyncio wakeup preflight."""

from __future__ import annotations

import errno
import importlib.util
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

_CONFTEST_SPEC = importlib.util.spec_from_file_location(
    "openharness_test_conftest", Path(__file__).parent / "conftest.py"
)
assert _CONFTEST_SPEC is not None and _CONFTEST_SPEC.loader is not None
conftest = importlib.util.module_from_spec(_CONFTEST_SPEC)
_CONFTEST_SPEC.loader.exec_module(conftest)


class _SyntheticSocket:
    def __init__(self, *, failure_at: str | None = None, error: OSError | None = None):
        self.failure_at = failure_at
        self.error = error
        self.closed = False
        self.inbox = bytearray()
        self.peer: _SyntheticSocket | None = None
        self.timeout: float | None = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()

    def close(self):
        self.closed = True

    def settimeout(self, timeout: float):
        self.timeout = timeout

    def sendall(self, data: bytes):
        if self.failure_at == "send":
            raise self.error
        assert self.peer is not None
        self.peer.inbox.extend(data)

    def recv(self, _size: int) -> bytes:
        if self.failure_at == "recv":
            raise self.error
        result = bytes(self.inbox[:_size])
        del self.inbox[:_size]
        return result


@pytest.mark.parametrize(
    ("failure_at", "error"),
    (
        (None, None),
        ("send", PermissionError(errno.EPERM, "synthetic seccomp denial")),
        ("recv", socket.timeout("synthetic finite timeout")),
    ),
)
def test_asyncio_self_pipe_probe_is_bounded_and_closes_sockets(
    monkeypatch: pytest.MonkeyPatch,
    failure_at: str | None,
    error: OSError | None,
) -> None:
    receiver = _SyntheticSocket(failure_at=failure_at, error=error)
    sender = _SyntheticSocket(failure_at=failure_at, error=error)
    receiver.peer, sender.peer = sender, receiver
    monkeypatch.setattr(conftest.socket, "socketpair", lambda *_args: (receiver, sender))

    if error is None:
        conftest._probe_asyncio_self_pipe()
    else:
        with pytest.raises(pytest.UsageError, match="asyncio self-pipe AF_UNIX socketpair"):
            conftest._probe_asyncio_self_pipe()

    assert receiver.timeout == sender.timeout == 0.5
    assert receiver.closed and sender.closed


def test_pytest_sessionstart_skips_probe_for_collect_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_probe() -> None:
        raise AssertionError("collect-only must not run the socket probe")

    monkeypatch.setattr(conftest, "_probe_asyncio_self_pipe", unexpected_probe)
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(collectonly=True)))
    conftest.pytest_sessionstart(session)


def test_pytest_sessionstart_probes_real_test_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(conftest, "_probe_asyncio_self_pipe", lambda: calls.append(True))
    session = SimpleNamespace(config=SimpleNamespace(option=SimpleNamespace(collectonly=False)))
    conftest.pytest_sessionstart(session)
    assert calls == [True]
