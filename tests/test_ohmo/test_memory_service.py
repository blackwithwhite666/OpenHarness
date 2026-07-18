"""Tests for the capability-authorized out-of-process memory service."""

from __future__ import annotations

import asyncio
import stat
import tempfile
from pathlib import Path
from typing import Mapping

import pytest

from ohmo.gateway.models import GatewayConfig
from ohmo.memory_backend import CatalogMemoryBackend, MemoryHit, make_memory_backend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service import MemoryServiceClient, MemoryServiceServer
from ohmo.memory_service.protocol import mint_capability, read_frame, write_frame
from ohmo.memory_store import MemoryEntry, MemoryOpResult


def _write_secret(path: Path, secret: bytes = b"test-memory-service-secret") -> bytes:
    path.write_bytes(secret)
    path.chmod(0o600)
    return secret


def _entry_values(entry: MemoryEntry) -> tuple[str, str, str, str]:
    return entry.name, entry.slug, entry.title, entry.content


def _entries_values(entries: list[MemoryEntry]) -> list[tuple[str, str, str, str]]:
    return [_entry_values(entry) for entry in entries]


def _assert_op_result_semantics(actual: MemoryOpResult, expected: MemoryOpResult) -> None:
    assert (actual.ok, actual.message) == (expected.ok, expected.message)
    if expected.entries is None:
        assert actual.entries is None
    else:
        assert actual.entries is not None
        assert _entries_values(list(actual.entries)) == _entries_values(list(expected.entries))


async def _send_raw(socket_path: Path, request: Mapping[str, object]) -> dict[str, object]:
    reader, writer = await asyncio.open_unix_connection(path=str(socket_path))
    try:
        await write_frame(writer, request)
        return await read_frame(reader)
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.fixture
async def memory_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OHMO_MEMORY_ENTRY_CHARS", "100")
    monkeypatch.setenv("OHMO_MEMORY_STORE_CHARS", "12")
    secret_file = tmp_path / "memory.secret"
    secret = _write_secret(secret_file)
    with tempfile.TemporaryDirectory(prefix="ohmo-ms-", dir="/tmp") as run_dir:
        socket_path = Path(run_dir) / "memory.sock"
        service = MemoryServiceServer(socket_path, tmp_path / "service-workspace", secret_file)
        await service.start()
        try:
            yield service, MemoryServiceClient(socket_path, secret_file), socket_path, secret
        finally:
            await service.close()


async def test_round_trip_every_operation_matches_catalog_semantics(
    memory_service,
    tmp_path: Path,
):
    service, client, socket_path, _ = memory_service
    reference = CatalogMemoryBackend(
        MemoryCatalog(tmp_path / "reference", entry_char_limit=100, store_char_budget=12),
        tmp_path / "reference",
    )

    assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
    assert await client.list() == await service.backend.list() == []

    _assert_op_result_semantics(
        await client.add("First note", "stable"),
        await reference.add("First note", "stable"),
    )
    first = await client.get("first_note.md")
    direct_first = await service.backend.get("first_note.md")
    reference_first = await reference.get("first_note.md")
    assert first == direct_first
    assert first is not None and reference_first is not None
    assert _entry_values(first) == _entry_values(reference_first)
    assert isinstance(first.path, Path)

    _assert_op_result_semantics(
        await client.add("Duplicate title", "stable"),
        await reference.add("Duplicate title", "stable"),
    )

    _assert_op_result_semantics(
        await client.add("Overflow", "1234567"),
        await reference.add("Overflow", "1234567"),
    )

    client_entries = await client.list()
    assert client_entries == await service.backend.list()
    assert _entries_values(client_entries) == _entries_values(await reference.list())

    expected_hits = await service.backend.search("stable", 3)
    reference_hits = await reference.search("stable", 3)
    assert await client.search("stable", 3) == expected_hits == reference_hits
    assert all(isinstance(hit, MemoryHit) for hit in expected_hits)

    _assert_op_result_semantics(
        await client.update("first_note", "short"),
        await reference.update("first_note", "short"),
    )
    assert await client.get("first_note") == await service.backend.get("first_note")

    service_before_turn = MemoryCatalog(tmp_path / "service-workspace").list(include_archived=True)
    assert await client.append_turn("user", "Do not persist this transient turn.") is None
    assert (
        MemoryCatalog(tmp_path / "service-workspace").list(include_archived=True)
        == service_before_turn
    )

    _assert_op_result_semantics(
        await client.remove("first_note.md"),
        await reference.remove("first_note.md"),
    )
    assert await client.list() == await service.backend.list() == []
    archived = MemoryCatalog(tmp_path / "service-workspace").list(include_archived=True)
    assert len(archived) == 1
    assert archived[0].archive_status == "archived"


async def test_render_prompt_over_socket_equals_in_process_backend(memory_service):
    service, client, _, _ = memory_service
    assert (await client.add("Alpha", "alpha body")).ok
    assert (await client.add("Bravo", "b")).ok

    expected = await service.backend.render_prompt(budget=len("alpha body"))
    actual = await client.render_prompt(budget=len("alpha body"))

    assert actual == expected
    assert "# ohmo Memory" in actual
    assert "- [Alpha](alpha.md)" in actual


@pytest.mark.parametrize("token_kind", ["missing", "invalid", "expired", "wrong-op"])
async def test_unauthorized_requests_are_refused_without_dispatch(
    memory_service,
    token_kind: str,
):
    service, _, socket_path, secret = memory_service
    request: dict[str, object] = {
        "op": "add",
        "args": {"title": "Must not exist", "content": "never dispatched"},
    }
    if token_kind == "invalid":
        request["token"] = mint_capability(b"wrong secret", "add", 30)
    elif token_kind == "expired":
        request["token"] = mint_capability(secret, "add", -60)
    elif token_kind == "wrong-op":
        request["token"] = mint_capability(secret, "list", 30)

    response = await _send_raw(socket_path, request)

    assert response == {"ok": False, "error": "unauthorized"}
    assert await service.backend.list() == []


def test_service_backend_factory_is_opt_in_and_default_remains_file(
    tmp_path: Path,
):
    secret_file = tmp_path / "memory.secret"
    _write_secret(secret_file)
    cfg = GatewayConfig(
        memory_backend="service",
        memory_service_socket=str(tmp_path / "memory.sock"),
        memory_service_secret_file=str(secret_file),
    )

    backend = make_memory_backend(cfg, tmp_path / "unused")

    assert GatewayConfig().memory_backend == "file"
    assert isinstance(backend, MemoryServiceClient)


def test_service_rejects_secret_file_without_owner_only_mode(tmp_path: Path):
    secret_file = tmp_path / "memory.secret"
    secret_file.write_bytes(b"unsafe permissions")
    secret_file.chmod(0o644)

    with pytest.raises(ValueError, match="mode 0600"):
        MemoryServiceClient(tmp_path / "memory.sock", secret_file)
