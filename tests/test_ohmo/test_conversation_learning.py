"""Tests for default-off, owner-scoped Honcho conversation learning."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openharness.api.usage import UsageSnapshot
from openharness.channels.bus.events import InboundMessage
from openharness.engine.stream_events import AssistantTextDelta

from ohmo.gateway.config import save_gateway_config
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.gateway.turn_context import TurnContext
from ohmo.memory_backend import CatalogMemoryBackend, ShadowMemoryBackend, make_memory_backend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service.outbox import DrainReport, drain_once
from ohmo.workspace import initialize_workspace


class FakeHoncho:
    def __init__(self, *, wait: bool = False, fail: bool = False) -> None:
        self.wait = wait
        self.fail = fail
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.messages: list[tuple[str, list[dict[str, object]]]] = []
        self.conclusions: list[list[dict[str, object]]] = []

    async def create_messages(
        self,
        session: str,
        messages: list[dict[str, object]],
    ) -> list[Any]:
        self.messages.append((session, messages))
        self.started.set()
        if self.wait:
            await self.release.wait()
        if self.fail:
            raise OSError("fake Honcho is unavailable")
        return []

    async def create_conclusions(
        self,
        conclusions: list[dict[str, object]],
    ) -> list[Any]:
        self.conclusions.append(conclusions)
        return [SimpleNamespace(id=f"conclusion-{len(self.conclusions)}")]

    async def delete_conclusion(self, conclusion_id: str) -> None:
        del conclusion_id


def _shadow_backend(
    workspace: Path,
    honcho: FakeHoncho,
    *,
    conversation_learning: bool,
) -> ShadowMemoryBackend:
    return ShadowMemoryBackend(
        CatalogMemoryBackend(MemoryCatalog(workspace), workspace),
        honcho_client=honcho,  # type: ignore[arg-type]
        conversation_learning=conversation_learning,
        comparison_log_path=workspace / "comparison.jsonl",
    )


def _owner_context(*, is_owner: bool = True, is_private: bool = True) -> TurnContext:
    return TurnContext(
        principal="owner",
        is_owner=is_owner,
        is_private=is_private,
        channel="feishu",
        chat_id="owner-chat",
        session_id="owner-session",
    )


def _pool_for_gate(
    backend: ShadowMemoryBackend,
    turn_ctx: TurnContext,
    *,
    conversation_learning: bool,
    session_owner_principal: str | None = "owner",
) -> OhmoSessionRuntimePool:
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._gateway_config = GatewayConfig(conversation_learning=conversation_learning)
    pool._prompt_memory_backend = backend
    pool._session_owner_principals = {
        turn_ctx.session_id: session_owner_principal,
    }
    return pool


def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_build_runtime(**kwargs: object) -> SimpleNamespace:
        del kwargs

        class FakeEngine:
            messages: list[object] = []
            total_usage = UsageSnapshot()
            tool_metadata: dict[str, object] = {}

            def set_system_prompt(self, prompt: str) -> None:
                del prompt

            async def submit_message(self, content: object):
                del content
                yield AssistantTextDelta(text="Assistant response.")

        return SimpleNamespace(
            engine=FakeEngine(),
            cwd=str(tmp_path),
            session_id="owner-session",
            current_settings=lambda: SimpleNamespace(model="fake-model"),
            commands=SimpleNamespace(lookup=lambda raw: None),
        )

    async def fake_start_runtime(bundle: object) -> None:
        del bundle

    monkeypatch.setattr("ohmo.gateway.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("ohmo.gateway.runtime.start_runtime", fake_start_runtime)


async def test_owner_private_isolated_turn_ingests_without_delaying_reply(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_gateway_config(
        GatewayConfig(
            evals_capture=False,
            memory_backend="shadow",
            conversation_learning=True,
            owner_principals=("owner",),
        ),
        workspace,
    )
    _install_fake_runtime(monkeypatch, tmp_path)
    honcho = FakeHoncho(wait=True)
    backend = _shadow_backend(workspace, honcho, conversation_learning=True)
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    pool._prompt_memory_backend = backend
    message = InboundMessage(
        channel="feishu",
        sender_id="owner",
        chat_id="owner-chat",
        content="User request.",
        metadata={"chat_type": "p2p"},
    )

    async def collect_updates() -> list[object]:
        return [update async for update in pool.stream_message(message, "feishu:owner-chat")]

    updates = await asyncio.wait_for(collect_updates(), timeout=0.2)

    assert updates[-1].kind == "final"
    assert updates[-1].text == "Assistant response."
    await asyncio.wait_for(honcho.started.wait(), timeout=0.2)
    honcho.release.set()
    await backend.await_pending()
    assert honcho.messages == [
        (
            "ohmo",
            [
                {
                    "content": "User request.",
                    "peer_id": "owner",
                    "metadata": {"role": "user"},
                }
            ],
        ),
        (
            "ohmo",
            [
                {
                    "content": "Assistant response.",
                    "peer_id": "ohmo",
                    "metadata": {"role": "assistant"},
                }
            ],
        ),
    ]


async def test_learning_off_never_calls_create_messages(tmp_path: Path) -> None:
    honcho = FakeHoncho()
    backend = _shadow_backend(tmp_path, honcho, conversation_learning=True)
    turn_ctx = _owner_context()
    pool = _pool_for_gate(backend, turn_ctx, conversation_learning=False)

    await pool._append_conversation_turn(
        turn_ctx=turn_ctx,
        user_text="User request.",
        assistant_text="Assistant response.",
    )
    await backend.await_pending()

    assert GatewayConfig().conversation_learning is False
    assert honcho.messages == []


@pytest.mark.parametrize(
    ("turn_ctx", "session_owner_principal"),
    (
        (_owner_context(is_owner=False), "owner"),
        (_owner_context(is_private=False), "owner"),
        (_owner_context(), "different-principal"),
    ),
    ids=("non-owner", "non-private", "non-isolated"),
)
async def test_runtime_gate_rejects_ineligible_turns(
    turn_ctx: TurnContext,
    session_owner_principal: str | None,
    tmp_path: Path,
) -> None:
    honcho = FakeHoncho()
    backend = _shadow_backend(tmp_path, honcho, conversation_learning=True)
    pool = _pool_for_gate(
        backend,
        turn_ctx,
        conversation_learning=True,
        session_owner_principal=session_owner_principal,
    )

    await pool._append_conversation_turn(
        turn_ctx=turn_ctx,
        user_text="User request.",
        assistant_text="Assistant response.",
    )
    await backend.await_pending()

    assert honcho.messages == []


async def test_honcho_ingestion_error_is_swallowed_and_turn_remains_complete(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    save_gateway_config(
        GatewayConfig(
            evals_capture=False,
            memory_backend="shadow",
            conversation_learning=True,
            owner_principals=("owner",),
        ),
        workspace,
    )
    _install_fake_runtime(monkeypatch, tmp_path)
    honcho = FakeHoncho(fail=True)
    backend = _shadow_backend(workspace, honcho, conversation_learning=True)
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    pool._prompt_memory_backend = backend
    message = InboundMessage(
        channel="feishu",
        sender_id="owner",
        chat_id="owner-chat",
        content="User request.",
        metadata={"chat_type": "p2p"},
    )

    updates = [update async for update in pool.stream_message(message, "feishu:owner-chat")]
    await backend.await_pending()

    assert updates[-1].kind == "final"
    assert updates[-1].text == "Assistant response."
    assert len(honcho.messages) == 2
    assert "conversation learning ingestion failed" in caplog.text


def test_shadow_factory_passes_learning_flag_and_preserves_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from ohmo.memory_service import honcho_client as honcho_client_module

    honcho = FakeHoncho()
    monkeypatch.setattr(honcho_client_module, "HonchoClient", lambda *args: honcho)

    backend = make_memory_backend(
        GatewayConfig(
            memory_backend="shadow",
            conversation_learning=True,
            owner_principals=("owner",),
            honcho_base_url="https://honcho.test",
            honcho_api_key="secret",
            honcho_workspace="workspace-one",
        ),
        tmp_path,
    )

    assert isinstance(backend, ShadowMemoryBackend)
    assert backend._conversation_learning is True
    assert GatewayConfig().memory_backend == "file"
    with pytest.raises(NotImplementedError, match="honcho memory backend not built"):
        make_memory_backend(GatewayConfig(memory_backend="honcho"), tmp_path / "honcho")


async def test_curated_mirror_uses_direct_conclusions_not_messages(tmp_path: Path) -> None:
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    honcho = FakeHoncho()
    assert catalog.add("Timezone", "User lives in Moscow.").ok

    report = await drain_once(catalog, honcho)  # type: ignore[arg-type]

    assert report == DrainReport(mirrored=1)
    assert honcho.conclusions == [
        [
            {
                "content": "User lives in Moscow.",
                "observer_id": "ohmo-curated",
                "observed_id": "owner",
            }
        ]
    ]
    assert honcho.messages == []
