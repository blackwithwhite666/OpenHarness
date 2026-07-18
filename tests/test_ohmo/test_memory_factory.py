"""Tests for configured memory-backend construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from ohmo.gateway.config import load_gateway_config, save_gateway_config
from ohmo.gateway.models import GatewayConfig
from ohmo.memory_backend import FileMemoryBackend, make_memory_backend


def test_make_memory_backend_returns_file_backend(tmp_path: Path):
    backend = make_memory_backend(GatewayConfig(memory_backend="file"), tmp_path)

    assert isinstance(backend, FileMemoryBackend)


def test_make_memory_backend_rejects_unbuilt_honcho_backend(tmp_path: Path):
    with pytest.raises(
        NotImplementedError,
        match="honcho memory backend not built in Phase 0",
    ):
        make_memory_backend(GatewayConfig(memory_backend="honcho"), tmp_path)


def test_make_memory_backend_rejects_unknown_backend(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported memory backend"):
        make_memory_backend(GatewayConfig(memory_backend="junk"), tmp_path)


def test_gateway_config_memory_backend_defaults_and_round_trips(tmp_path: Path):
    assert load_gateway_config(tmp_path).memory_backend == "file"

    expected = GatewayConfig(
        memory_backend="honcho",
        honcho_base_url="https://honcho.invalid",
        honcho_api_key="secret",
        honcho_workspace="workspace",
    )
    save_gateway_config(expected, tmp_path)

    actual = load_gateway_config(tmp_path)
    assert actual.memory_backend == "honcho"
    assert actual.honcho_base_url == "https://honcho.invalid"
    assert actual.honcho_api_key == "secret"
    assert actual.honcho_workspace == "workspace"
