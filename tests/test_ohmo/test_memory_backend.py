from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

import pytest

from ohmo.memory_backend import ConversationAppendReceipt, NutritionReconciliationError, ShadowMemoryBackend
from ohmo.memory_service.honcho_client import Message


class _Base:
    _memory_dir = Path("/tmp")

    async def append_turn(self, role, text):
        del role, text


class _Honcho:
    def __init__(self):
        self.messages = []

    async def find_messages_by_client_op_id(self, session, operation):
        return [item for item in self.messages if item.metadata.get("client_op_id") == operation]

    async def create_messages(self, session, values):
        created = []
        for value in values:
            item = Message(
                id=f"m{len(self.messages) + 1}",
                content=value["content"],
                peer_id=value["peer_id"],
                session_id=session,
                metadata=value["metadata"],
                created_at=dt.datetime.now(dt.timezone.utc),
                workspace_id="workspace",
                token_count=1,
            )
            self.messages.append(item)
            created.append(item)
        return created


def _metadata(candidate: str) -> tuple[dict, dict]:
    common = {
        "_nutrition_trusted": True,
        "tenant_id": "marina",
        "source_principal": "telegram:123",
        "ingest_source": "dropbox_camera",
        "confirmation_required": True,
        "candidate_id": candidate,
        "nutrition_phase": "estimation",
    }
    return (
        {**common, "client_op_id": f"{candidate}:meal-user:v1"},
        {**common, "client_op_id": f"{candidate}:meal-observation:v1"},
    )


@pytest.mark.asyncio
async def test_durable_exchange_returns_receipt_and_reconciles_retry() -> None:
    candidate = "dropbox-camera-v1-" + "a" * 64
    honcho = _Honcho()
    backend = ShadowMemoryBackend(_Base(), honcho, conversation_learning=True)
    user_metadata, assistant_metadata = _metadata(candidate)

    first = await backend.append_exchange(
        "photo",
        "estimate",
        user_metadata=user_metadata,
        assistant_metadata=assistant_metadata,
        durable=True,
        trusted_nutrition=True,
    )
    second = await backend.append_exchange(
        "photo",
        "estimate",
        user_metadata=user_metadata,
        assistant_metadata=assistant_metadata,
        durable=True,
        trusted_nutrition=True,
    )

    assert isinstance(first, ConversationAppendReceipt)
    assert second == first
    assert len(honcho.messages) == 2


@pytest.mark.asyncio
async def test_background_exchange_keeps_non_durable_compatibility() -> None:
    honcho = _Honcho()
    backend = ShadowMemoryBackend(_Base(), honcho, conversation_learning=True)
    user_metadata, assistant_metadata = _metadata("dropbox-camera-v1-" + "b" * 64)
    user_metadata = {"client_op_id": "background-user"}
    assistant_metadata = {"client_op_id": "background-assistant"}

    result = await backend.append_exchange(
        "photo",
        "estimate",
        user_metadata=user_metadata,
        assistant_metadata=assistant_metadata,
    )
    await backend.await_pending()
    assert result is None
    assert len(honcho.messages) == 2


@pytest.mark.asyncio
async def test_durable_exchange_rejects_missing_malformed_and_duplicate_operations() -> None:
    candidate = "dropbox-camera-v1-" + "c" * 64
    user_metadata, assistant_metadata = _metadata(candidate)
    honcho = _Honcho()
    backend = ShadowMemoryBackend(_Base(), honcho, conversation_learning=True)

    with pytest.raises(NutritionReconciliationError, match="operation"):
        await backend.append_exchange(
            "photo",
            "estimate",
            user_metadata=user_metadata,
            assistant_metadata={**assistant_metadata, "client_op_id": ""},
            durable=True,
            trusted_nutrition=True,
        )

    honcho.messages.extend(
        [
            Message(
                id="bad",
                content="estimate",
                peer_id="ohmo",
                session_id="ohmo",
                metadata={"client_op_id": assistant_metadata["client_op_id"], "role": "user"},
                created_at=dt.datetime.now(dt.timezone.utc),
                workspace_id="workspace",
                token_count=1,
            )
        ]
    )
    with pytest.raises(NutritionReconciliationError, match="malformed"):
        await backend.append_exchange(
            "photo",
            "estimate",
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata,
            durable=True,
            trusted_nutrition=True,
        )

    duplicate_honcho = _Honcho()
    duplicate_message = Message(
            id="duplicate",
            content="estimate",
            peer_id="ohmo",
            session_id="ohmo",
            metadata={"client_op_id": assistant_metadata["client_op_id"], "role": "assistant"},
            created_at=dt.datetime.now(dt.timezone.utc),
            workspace_id="workspace",
            token_count=1,
        )
    duplicate_honcho.messages.extend([duplicate_message, duplicate_message])
    duplicate_backend = ShadowMemoryBackend(_Base(), duplicate_honcho, conversation_learning=True)
    with pytest.raises(NutritionReconciliationError, match="ambiguous"):
        await duplicate_backend.append_exchange(
            "photo",
            "estimate",
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata,
            durable=True,
            trusted_nutrition=True,
        )


class _TimeoutHoncho(_Honcho):
    def __init__(self, *, commit: bool) -> None:
        super().__init__()
        self.commit = commit

    async def create_messages(self, session, values):
        if self.commit:
            await super().create_messages(session, values)
        raise asyncio.TimeoutError("timed out")


@pytest.mark.asyncio
async def test_durable_exchange_reconciles_timeout_after_commit_and_rejects_without_commit() -> None:
    candidate = "dropbox-camera-v1-" + "d" * 64
    user_metadata, assistant_metadata = _metadata(candidate)
    committed = _TimeoutHoncho(commit=True)
    backend = ShadowMemoryBackend(_Base(), committed, conversation_learning=True)
    receipt = await backend.append_exchange(
        "photo",
        "estimate",
        user_metadata=user_metadata,
        assistant_metadata=assistant_metadata,
        durable=True,
        trusted_nutrition=True,
    )
    assert isinstance(receipt, ConversationAppendReceipt)
    assert len(committed.messages) == 2

    no_commit = _TimeoutHoncho(commit=False)
    backend = ShadowMemoryBackend(_Base(), no_commit, conversation_learning=True)
    with pytest.raises(asyncio.TimeoutError):
        await backend.append_exchange(
            "photo",
            "estimate",
            user_metadata=user_metadata,
            assistant_metadata=assistant_metadata,
            durable=True,
            trusted_nutrition=True,
        )
