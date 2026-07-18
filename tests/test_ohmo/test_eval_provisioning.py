"""Tests for isolated memory-recall eval backend provisioning."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from ohmo.evals.memory import provisioning
from ohmo.evals.memory.provisioning import provision_backend
from ohmo.memory_backend import CatalogMemoryBackend, ShadowMemoryBackend
from ohmo.memory_catalog import MemoryCatalog


@dataclass(frozen=True)
class FakeConclusion:
    id: str
    content: str


class FakeHonchoClient:
    instances: list["FakeHonchoClient"] = []

    def __init__(self, base_url: str, jwt: str, workspace: str) -> None:
        self.base_url = base_url
        self.jwt = jwt
        self.workspace = workspace
        self.created: list[list[dict[str, object]]] = []
        self.conclusions: list[FakeConclusion] = []
        self.deleted: list[str] = []
        self.queries: list[tuple[str, str, str, int]] = []
        self.closed = False
        self.instances.append(self)

    async def create_conclusions(
        self,
        conclusions: Sequence[Mapping[str, object] | object],
    ) -> list[FakeConclusion]:
        batch = [dict(item) for item in conclusions if isinstance(item, Mapping)]
        self.created.append(batch)
        acknowledged = [
            FakeConclusion(
                id=f"conclusion-{len(self.conclusions) + index + 1}",
                content=str(item["content"]),
            )
            for index, item in enumerate(batch)
        ]
        self.conclusions.extend(acknowledged)
        return acknowledged

    async def list_conclusions(self) -> list[FakeConclusion]:
        return list(self.conclusions)

    async def query_conclusions(
        self,
        query: str,
        *,
        observer: str,
        observed: str,
        top_k: int,
    ) -> list[FakeConclusion]:
        self.queries.append((query, observer, observed, top_k))
        return self.conclusions[:top_k]

    async def delete_conclusion(self, conclusion_id: str) -> None:
        self.deleted.append(conclusion_id)
        self.conclusions = [item for item in self.conclusions if item.id != conclusion_id]

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("kind", ["file", "catalog"])
async def test_local_backends_are_seeded_isolated_and_removed(kind: str) -> None:
    first = await provision_backend(
        kind,  # type: ignore[arg-type]
        run="run1",
        case="case-a",
        sample=0,
        seed_entries=[("Timezone", "User lives in Europe/Moscow.")],
    )
    second = await provision_backend(
        kind,  # type: ignore[arg-type]
        run="run1",
        case="case-b",
        sample=0,
        seed_entries=[("Editor", "User prefers Neovim.")],
    )
    first_path = first.workspace
    second_path = second.workspace

    try:
        assert first.kind == kind
        assert second.kind == kind
        assert first.honcho_workspace is None
        assert second.honcho_workspace is None
        assert first_path != second_path
        assert first_path.is_dir()
        assert second_path.is_dir()
        assert _entry_contents(await first.backend.list()) == {
            "Timezone": "User lives in Europe/Moscow."
        }
        assert _entry_contents(await second.backend.list()) == {"Editor": "User prefers Neovim."}

        assert (await first.backend.add("Shell", "User prefers zsh.")).ok
        assert "Shell" in _entry_contents(await first.backend.list())
        assert "Shell" not in _entry_contents(await second.backend.list())

        await first.teardown()
        assert not first_path.exists()
        assert second_path.is_dir()
        await second.teardown()
        assert not second_path.exists()
    finally:
        await first.teardown()
        await second.teardown()


async def test_shadow_provisions_seeds_mirrors_and_tears_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provisioning_calls: list[dict[str, object]] = []

    async def fake_provision_eval_workspace(**kwargs: object) -> tuple[str, str]:
        provisioning_calls.append(dict(kwargs))
        return "ohmo-eval-run7-case2-3", "scoped-jwt"

    FakeHonchoClient.instances.clear()
    monkeypatch.setattr(
        provisioning,
        "provision_eval_workspace",
        fake_provision_eval_workspace,
    )
    monkeypatch.setattr(provisioning, "HonchoClient", FakeHonchoClient)

    provisioned = await provision_backend(
        "shadow",
        run="run7",
        case="case2",
        sample=3,
        seed_entries=[
            ("Timezone", "User lives in Europe/Moscow."),
            ("Editor", "User prefers Neovim."),
        ],
        honcho_base_url="https://honcho.test",
        honcho_admin_jwt="explicit-admin-jwt",
    )
    workspace = provisioned.workspace
    [fake_honcho] = FakeHonchoClient.instances

    try:
        assert isinstance(provisioned.backend, ShadowMemoryBackend)
        assert provisioned.honcho_workspace == "ohmo-eval-run7-case2-3"
        assert provisioning_calls == [
            {
                "base_url": "https://honcho.test",
                "admin_jwt": "explicit-admin-jwt",
                "run": "run7",
                "case": "case2",
                "sample": 3,
                "ttl": dt.timedelta(hours=1),
            }
        ]
        assert (
            fake_honcho.base_url,
            fake_honcho.jwt,
            fake_honcho.workspace,
        ) == (
            "https://honcho.test",
            "scoped-jwt",
            "ohmo-eval-run7-case2-3",
        )
        assert fake_honcho.created == [
            [
                {
                    "content": "User lives in Europe/Moscow.",
                    "observer_id": "ohmo-curated",
                    "observed_id": "owner",
                }
            ],
            [
                {
                    "content": "User prefers Neovim.",
                    "observer_id": "ohmo-curated",
                    "observed_id": "owner",
                }
            ],
        ]

        catalog = CatalogMemoryBackend(MemoryCatalog(workspace), workspace)
        assert await provisioned.backend.list() == await catalog.list()
        assert await provisioned.backend.render_prompt() == await catalog.render_prompt()
        assert await provisioned.backend.search("Neovim", 5) == await catalog.search("Neovim", 5)
        await provisioned.backend.await_pending()
        assert fake_honcho.queries == [("Neovim", "ohmo-curated", "owner", 5)]

        remote_ids = [item.id for item in fake_honcho.conclusions]
        await provisioned.teardown()
        assert fake_honcho.deleted == remote_ids
        assert fake_honcho.conclusions == []
        assert fake_honcho.closed
        assert not workspace.exists()
    finally:
        await provisioned.teardown()


async def test_shadow_requires_explicit_admin_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_provision(**kwargs: object) -> tuple[str, str]:
        del kwargs
        raise AssertionError("provisioning must not run without the explicit admin credential")

    monkeypatch.setattr(provisioning, "provision_eval_workspace", unexpected_provision)

    with pytest.raises(ValueError, match="requires explicit honcho_admin_jwt"):
        await provision_backend(
            "shadow",
            run="run",
            case="case",
            sample=1,
            seed_entries=[],
            honcho_base_url="https://honcho.test",
        )
    assert "GatewayConfig" not in provisioning.__dict__


def _entry_contents(entries: Sequence[object]) -> dict[str, str]:
    return {str(getattr(entry, "title")): str(getattr(entry, "content")) for entry in entries}
