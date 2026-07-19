"""Tests for configured memory-backend construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals.inference import DEFAULT_INFERENCE_URL

from ohmo.gateway.config import load_gateway_config, save_gateway_config
from ohmo.gateway.models import GatewayConfig
from ohmo.memory_backend import (
    CatalogMemoryBackend,
    FileMemoryBackend,
    ShadowMemoryBackend,
    make_memory_backend,
)
from ohmo.memory_catalog import MemoryCatalog


class FakeInferenceClient:
    constructed_urls: list[str] = []
    from_env_calls = 0

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.close_calls = 0
        type(self).constructed_urls.append(base_url)

    @classmethod
    def from_env(cls) -> "FakeInferenceClient":
        cls.from_env_calls += 1
        return cls(DEFAULT_INFERENCE_URL)

    async def aclose(self) -> None:
        self.close_calls += 1


class ClosableEmbeddingClient:
    def __init__(self, *, raises: bool = False) -> None:
        self.close_calls = 0
        self.raises = raises

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.raises:
            raise RuntimeError("close failed")


def test_make_memory_backend_returns_file_backend(tmp_path: Path):
    backend = make_memory_backend(GatewayConfig(memory_backend="file"), tmp_path)

    assert isinstance(backend, FileMemoryBackend)


@pytest.mark.parametrize("backend_kind", ["catalog", "shadow"])
def test_make_memory_backend_semantic_search_defaults_off_without_embedder(
    backend_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from openharness.evals import inference

    FakeInferenceClient.constructed_urls = []
    FakeInferenceClient.from_env_calls = 0
    monkeypatch.setattr(inference, "InferenceClient", FakeInferenceClient)
    workspace = tmp_path / backend_kind

    backend = make_memory_backend(GatewayConfig(memory_backend=backend_kind), workspace)
    actual = backend._base if isinstance(backend, ShadowMemoryBackend) else backend

    assert isinstance(actual, CatalogMemoryBackend)
    expected = CatalogMemoryBackend(actual._catalog, workspace)
    assert vars(actual) == vars(expected)
    assert actual._embedder is None
    assert FakeInferenceClient.constructed_urls == []
    assert FakeInferenceClient.from_env_calls == 0


@pytest.mark.parametrize("backend_kind", ["catalog", "shadow"])
@pytest.mark.parametrize(
    (
        "inference_url",
        "embedding_model",
        "expected_url",
        "expected_model",
        "expected_from_env_calls",
    ),
    [
        (
            "https://inference.example.test",
            "embedding-override",
            "https://inference.example.test",
            "embedding-override",
            0,
        ),
        (None, None, DEFAULT_INFERENCE_URL, "BAAI/bge-m3", 1),
    ],
)
async def test_make_memory_backend_wires_configured_inference_client(
    backend_kind: str,
    inference_url: str | None,
    embedding_model: str | None,
    expected_url: str,
    expected_model: str,
    expected_from_env_calls: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from openharness.evals import inference

    FakeInferenceClient.constructed_urls = []
    FakeInferenceClient.from_env_calls = 0
    monkeypatch.setattr(inference, "InferenceClient", FakeInferenceClient)

    backend = make_memory_backend(
        GatewayConfig(
            memory_backend=backend_kind,
            semantic_search=True,
            inference_url=inference_url,
            embedding_model=embedding_model,
        ),
        tmp_path / backend_kind,
    )
    actual = backend._base if isinstance(backend, ShadowMemoryBackend) else backend

    assert isinstance(actual, CatalogMemoryBackend)
    assert isinstance(actual._embedder, FakeInferenceClient)
    assert actual._embedder.base_url == expected_url
    assert actual._embedding_model == expected_model
    assert actual._owns_embedder is True
    assert FakeInferenceClient.constructed_urls == [expected_url]
    assert FakeInferenceClient.from_env_calls == expected_from_env_calls

    await backend.aclose()
    await backend.aclose()

    assert actual._embedder.close_calls == 1


async def test_catalog_aclose_closes_only_owned_embedder_once(tmp_path: Path):
    owned = ClosableEmbeddingClient()
    injected = ClosableEmbeddingClient()
    owned_backend = CatalogMemoryBackend(
        MemoryCatalog(tmp_path / "owned"),
        tmp_path / "owned",
        embedder=owned,
        owns_embedder=True,
    )
    injected_backend = CatalogMemoryBackend(
        MemoryCatalog(tmp_path / "injected"),
        tmp_path / "injected",
        embedder=injected,
    )

    await owned_backend.aclose()
    await owned_backend.aclose()
    await injected_backend.aclose()

    assert owned.close_calls == 1
    assert injected.close_calls == 0


async def test_shadow_aclose_closes_owned_embedder_once(tmp_path: Path):
    embedder = ClosableEmbeddingClient()
    backend = ShadowMemoryBackend(
        CatalogMemoryBackend(
            MemoryCatalog(tmp_path),
            tmp_path,
            embedder=embedder,
            owns_embedder=True,
        )
    )

    await backend.aclose()
    await backend.aclose()

    assert embedder.close_calls == 1


async def test_catalog_aclose_swallows_owned_embedder_close_errors(tmp_path: Path):
    embedder = ClosableEmbeddingClient(raises=True)
    backend = CatalogMemoryBackend(
        MemoryCatalog(tmp_path),
        tmp_path,
        embedder=embedder,
        owns_embedder=True,
    )

    await backend.aclose()
    await backend.aclose()

    assert embedder.close_calls == 1


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
    default = load_gateway_config(tmp_path)
    assert default.memory_backend == "file"
    assert default.semantic_search is False
    assert default.inference_url is None
    assert default.embedding_model is None

    expected = GatewayConfig(
        memory_backend="honcho",
        semantic_search=True,
        inference_url="https://inference.example.test",
        embedding_model="embedding-override",
        honcho_base_url="https://honcho.invalid",
        honcho_api_key="secret",
        honcho_workspace="workspace",
    )
    save_gateway_config(expected, tmp_path)

    actual = load_gateway_config(tmp_path)
    assert actual.memory_backend == "honcho"
    assert actual.semantic_search is True
    assert actual.inference_url == "https://inference.example.test"
    assert actual.embedding_model == "embedding-override"
    assert actual.honcho_base_url == "https://honcho.invalid"
    assert actual.honcho_api_key == "secret"
    assert actual.honcho_workspace == "workspace"
