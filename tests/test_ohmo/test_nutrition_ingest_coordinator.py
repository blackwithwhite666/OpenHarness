from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import stat
import struct
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ohmo.gateway.attachment_fingerprints import (
    PHASH_ALGORITHM,
    PHASH_HAMMING_THRESHOLD,
    fingerprint_image_bytes,
)
from ohmo.gateway.bridge import OhmoGatewayBridge
from ohmo.gateway.models import GatewayConfig, NutritionIngestConfig
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.memory_backend import CatalogMemoryBackend, ShadowMemoryBackend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service.honcho_client import Message, RecentMessageMetadata
from ohmo.nutrition_ingest.coordinator import NutritionIngestCoordinator
from ohmo.nutrition_ingest.freshness import exif_freshness_reason
from ohmo.nutrition_ingest.metrics import NutritionMetrics
from ohmo.nutrition_ingest.models import (
    ExifMetadata,
    ResultState,
    SeenTombstoneV1,
    candidate_id_for,
)
from ohmo.nutrition_ingest.sidecars import NutritionResultStore
from ohmo.workspace import initialize_workspace
from openharness.api.usage import UsageSnapshot
from openharness.channels.bus.events import InboundMessage, OutboundDeliveryReceipt
from openharness.channels.bus.queue import MessageBus
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.engine.stream_events import AssistantTextDelta, AssistantTurnComplete
from openharness.evals import TRACE_FINALIZATION
from openharness.tools.base import ToolRegistry


def _candidate(
    root: Path,
    *,
    file_id: str = "id:test",
    rev: str = "rev:test",
    discovery_time: str = "2026-08-05T10:01:00+00:00",
    capture_time: str | None = None,
    exif_updates: dict[str, object] | None = None,
    data: bytes = b"native-photo",
    filename: str = "photo.jpg",
) -> str:
    fixture = json.loads(
        (Path(__file__).parents[2] / "ohmo/nutrition_ingest/manifest_v1_fixture.json").read_text()
    )
    candidate = candidate_id_for(file_id, rev)
    directory = root / candidate
    directory.mkdir()
    image = directory / filename
    image.write_bytes(data)
    fixture.update(
        candidate_id=candidate,
        file_id=file_id,
        rev=rev,
        discovery_time=discovery_time,
        original_filename=filename,
        original_size_bytes=len(data),
        original_sha256=hashlib.sha256(data).hexdigest(),
    )
    fixture["exif"]["normalized_capture_time"] = capture_time or "2099-01-01T00:00:00+00:00"
    if exif_updates:
        fixture["exif"].update(exif_updates)
    (directory / "manifest.json").write_text(json.dumps(fixture))
    return candidate


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _png(pixel: tuple[int, int, int]) -> bytes:
    width = height = 16
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + bytes(pixel) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class _HonchoStore:
    """Deterministic remote transport with the real client-op lookup contract."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    async def find_messages_by_client_op_id(self, session: str, operation: str) -> list[Message]:
        return [
            message
            for message in self.messages
            if message.session_id == session and message.metadata.get("client_op_id") == operation
        ]

    async def create_messages(self, session: str, values: list[dict[str, object]]) -> list[Message]:
        created = []
        for value in values:
            message = Message(
                id=f"honcho-{len(self.messages) + 1}",
                content=str(value["content"]),
                peer_id=str(value["peer_id"]),
                session_id=session,
                metadata=dict(value["metadata"]),
                created_at=datetime.now(UTC),
                workspace_id="family-marina",
                token_count=1,
            )
            self.messages.append(message)
            created.append(message)
        return created


class _RecentSource:
    def __init__(
        self,
        messages: list[RecentMessageMetadata] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.messages = messages or []
        self.error = error
        self.calls = 0
        self.requests: list[dict[str, object]] = []

    async def list_recent_message_metadata(self, session: str, **kwargs):
        self.calls += 1
        self.requests.append({"session": session, **kwargs})
        if self.error is not None:
            raise self.error
        return list(self.messages)


def _recent_message(
    fingerprints: object,
    *,
    identifier: str = "honcho-user-1",
    role: str = "user",
    tenant: str = "marina",
    principal: str = "telegram:123",
    ingest_source: str | None = None,
    peer_id: str = "marina",
    session_id: str = "ohmo",
    created_at: datetime | None = None,
) -> RecentMessageMetadata:
    metadata: dict[str, object] = {
        "role": role,
        "tenant_id": tenant,
        "source_principal": principal,
        "attachment_fingerprints": fingerprints,
    }
    if ingest_source is not None:
        metadata["ingest_source"] = ingest_source
    return RecentMessageMetadata(
        id=identifier,
        peer_id=peer_id,
        session_id=session_id,
        metadata=metadata,
        created_at=created_at or datetime(2026, 8, 5, 9, tzinfo=UTC),
    )


class _NutritionModelStream:
    """A model-only fake: emits one validated nutrition trace and final text."""

    def __init__(self, *, protein_g: object = 30, answer: str | None = None) -> None:
        self.api_client = object()
        self.decision_trace_recorder = None
        self.engine = self
        self.model = "nutrition-test-model"
        self.max_turns = 1
        self.messages: list[ConversationMessage] = []
        self.tool_metadata: dict[str, object] = {}
        self.total_usage = UsageSnapshot()
        self.system_prompt = ""
        self.protein_g = protein_g
        self.answer = answer or "Записала съеденный приём пищи: 999 ккал."

    def set_decision_trace_recorder(self, recorder) -> None:
        self.decision_trace_recorder = recorder

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    async def submit_message(self, message: ConversationMessage):
        self.messages.append(message)
        recorder = self.decision_trace_recorder
        assert recorder is not None
        recorder.record(
            TRACE_FINALIZATION,
            {
                "schema_version": 1,
                "trace_event_id": "nutrition-e2e-trace",
                "annotations": {
                    "nutrition": {
                        "schema_version": 2,
                        "record_type": "meal_observation",
                        "basis": ["image"],
                        "consumption_status": "consumed",
                        "energy_kcal_best": 550,
                        "protein_g": self.protein_g,
                        "fat_g": 20,
                        "carbohydrate_g": 45,
                    }
                },
            },
        )
        answer = self.answer
        yield AssistantTextDelta(answer)
        assistant = ConversationMessage(role="assistant", content=[TextBlock(text=answer)])
        self.messages.append(assistant)
        yield AssistantTurnComplete(message=assistant, usage=UsageSnapshot())


def _nutrition_runtime_pool(
    tmp_path: Path,
    workspace: Path,
    honcho: _HonchoStore,
    *,
    model_stream: _NutritionModelStream | None = None,
) -> OhmoSessionRuntimePool:
    config = GatewayConfig(
        honcho_base_url="https://honcho.test",
        evals_capture=True,
        memory_backend="shadow",
        conversation_learning=True,
        family_principals={"123": "marina"},
        enabled_memory_tenants=("marina",),
        tenant_honcho={
            "marina": {
                "workspace": "family-marina",
                "api_key": "test-key",
                "observed_peer": "marina-peer",
            }
        },
        nutrition_ingest=NutritionIngestConfig(
            enabled=True,
            synchronized_root=tmp_path,
            principal="123",
            chat_id="123",
            session_key="telegram:123",
            retry_backoff_seconds=0,
            require_owner_only_filesystem=False,
        ),
    )
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")
    catalog = MemoryCatalog(workspace)
    catalog.ensure_tenant("marina", "private")
    shadow = ShadowMemoryBackend(
        CatalogMemoryBackend(catalog, workspace, tenant_id="marina"),
        honcho_client=honcho,  # type: ignore[arg-type]
        observed="marina-peer",
        conversation_learning=True,
        session="telegram:123",
        assistant_peer="ohmo",
    )
    pool._gateway_config = config
    pool._prompt_memory_backend = shadow
    pool._tenant_shadow_backends = {"marina": shadow}
    session_key = "telegram:123"
    message_cwd = pool._cwd_for_message(
        InboundMessage(
            channel="telegram",
            sender_id="__nutrition_ingest__",
            chat_id="123",
            content="",
            session_key_override=session_key,
        ),
        session_key,
    )
    bundle = SimpleNamespace(
        engine=model_stream or _NutritionModelStream(),
        session_id="nutrition-runtime-session",
        cwd=message_cwd,
        tool_registry=ToolRegistry(),
        commands=SimpleNamespace(lookup=lambda _prompt: None),
        current_settings=lambda: SimpleNamespace(model="nutrition-test-model"),
        extra_skill_dirs=(),
        extra_plugin_roots=(),
    )
    pool._bundles[session_key] = bundle

    async def deterministic_prompt(*_args, **_kwargs) -> str:
        return "nutrition test system prompt"

    pool._runtime_system_prompt = deterministic_prompt  # type: ignore[method-assign]
    return pool


class _CrashAfterDurableCommit:
    def __init__(self, pool: OhmoSessionRuntimePool) -> None:
        self.pool = pool

    async def stream_message(self, message, session_key):
        async for update in self.pool.stream_message(message, session_key):
            yield update
            if update.kind == "final":
                raise RuntimeError("simulated coordinator crash after Honcho commit")


@pytest.mark.asyncio
async def test_marina_queue_has_one_native_prompt_and_decline_has_no_estimation(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    estimates = []
    metrics = NutritionMetrics()
    coordinator = NutritionIngestCoordinator(
        config,
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        estimate=estimates.append,
        metrics=metrics,
    )

    await coordinator.poll_once()
    assert len(outbound) == 1
    prompt = outbound[0]
    assert prompt.content == "Вы это съели?\nДата: 01.01.2099 00:00 (по EXIF фото)"
    assert prompt.buttons == ["Да", "Нет"]
    assert len(prompt.media) == 1
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt("telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]),
    )
    handled = await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Нет",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    )
    assert handled is True
    assert estimates == []
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.completed
    assert sidecar.consumption_status == "not_consumed"
    assert sidecar.emitted_honcho_message_id is None
    assert len(outbound) == 2
    assert "КБЖУ" not in outbound[-1].content
    events = metrics.snapshot()["events"]
    assert events["verified_candidate|scan|"] == 1
    assert events["confirmation|prompt|declined"] == 1
    assert metrics.snapshot()["pending_latency"]["count"] == 1
    assert metrics.snapshot()["end_to_end_latency"]["count"] == 1


@pytest.mark.asyncio
async def test_non_marina_principal_cannot_confirm(tmp_path: Path) -> None:
    _candidate(tmp_path)
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append)
    await coordinator.poll_once()
    assert await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="999",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
        )
    ) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sender_id", "999"),
        ("chat_id", "999"),
        ("session_key_override", "telegram:999"),
        ("native_message_id", 41),
    ],
)
async def test_callback_requires_exact_marina_binding_and_photo_id(
    tmp_path: Path, field: str, value: object
) -> None:
    candidate = _candidate(tmp_path, file_id="id:callback", rev="rev:callback")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    estimates = []
    coordinator = NutritionIngestCoordinator(
        config,
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        estimate=lambda message: estimates.append(message) or "honcho-1",
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt("telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]),
    )

    message = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="Да",
        session_key_override="telegram:123",
        metadata={"callback_query": True, "native_message_id": 42},
    )
    if field == "native_message_id":
        message.metadata[field] = value
    else:
        setattr(message, field, value)
    assert await coordinator.handle_inbound(message) is (field == "native_message_id")
    assert estimates == []
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.pending_confirmation


@pytest.mark.asyncio
async def test_exact_callback_confirmation_reaches_estimator_once(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:callback-ok", rev="rev:callback-ok")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    estimates = []
    coordinator = NutritionIngestCoordinator(
        config,
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        estimate=lambda message: estimates.append(message) or "honcho-callback",
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt("telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]),
    )
    assert await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    ) is True
    assert len(estimates) == 1
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.completed


@pytest.mark.asyncio
async def test_unknown_typed_reply_is_clarified_and_yes_creates_one_durable_completion(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path, file_id="id:typed", rev="rev:typed")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    estimates = []

    def estimate(message):
        estimates.append(message)
        assert message.content.startswith("[Trusted Dropbox meal estimation turn]")
        assert message.media
        return {"assistant_message_id": "honcho-assistant-1"}

    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append, estimate=estimate)
    await coordinator.poll_once()
    prompt = outbound[0]
    assert prompt.media and len(prompt.media) == 1
    assert prompt.content == "Вы это съели?\nДата: 01.01.2099 00:00 (по EXIF фото)"
    assert prompt.buttons == ["Да", "Нет"]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt("telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]),
    )

    assert await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="может быть",
            session_key_override="telegram:123",
        )
    ) is True
    assert outbound[-1].content.startswith("Пожалуйста")
    assert estimates == []

    assert await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
        )
    ) is True
    assert len(estimates) == 0
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.pending_confirmation

    assert await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    ) is True
    assert len(estimates) == 1
    assert await coordinator.poll_once() == [candidate]
    assert len(estimates) == 1
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.completed
    assert sidecar.consumption_status == "consumed"
    assert sidecar.emitted_honcho_message_id == "honcho-assistant-1"


@pytest.mark.asyncio
async def test_completed_candidate_leaves_later_correction_on_normal_ohmo_path(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path, file_id="id:correction", rev="rev:original")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    coordinator_outbound = []
    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=coordinator_outbound.append)
    await coordinator.poll_once()
    prompt = coordinator_outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt("telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]),
    )
    await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Нет",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    )
    before = NutritionResultStore(tmp_path / candidate / "result.json").load().model_dump(mode="json")
    seen = []
    bus = MessageBus()

    class Runtime:
        async def stream_message(self, message, session_key):
            seen.append((message, session_key))
            yield type("Update", (), {"kind": "final", "text": "Исправление принято", "metadata": {}})()

    bridge = OhmoGatewayBridge(
        bus=bus,
        runtime_pool=Runtime(),
        workspace=tmp_path,
        nutrition_coordinator=coordinator,
    )
    task = asyncio.create_task(bridge.run())
    try:
        await bus.publish_inbound(
            InboundMessage(
                channel="telegram",
                sender_id="123",
                chat_id="123",
                content="Исправление: это был другой ужин",
                session_key_override="telegram:123",
            )
        )
        reply = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
    finally:
        bridge.stop()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert reply.content == "Исправление принято"
    assert seen and seen[0][0].content == "Исправление: это был другой ужин"
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().model_dump(mode="json") == before
    assert len([item for item in coordinator_outbound if item.metadata.get("_nutrition_confirmation")]) == 1


@pytest.mark.asyncio
async def test_ambiguous_failure_is_not_republished(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append)
    await coordinator.poll_once()
    prompt = outbound[0]

    assert await coordinator.on_send_failure(prompt, RuntimeError("telegram timeout")) is True
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.delivery_unknown
    await coordinator.poll_once()
    assert len(outbound) == 1


@pytest.mark.asyncio
async def test_delivery_unknown_does_not_block_later_candidate(tmp_path: Path) -> None:
    _candidate(tmp_path, file_id="id:first-unknown", rev="rev:first-unknown")
    _candidate(tmp_path, file_id="id:later", rev="rev:later")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config, honcho_client=_RecentSource(), publish_outbound=outbound.append
    )
    ordered = [item.candidate_id for item in coordinator._scanner.scan_ready()]
    head, later = ordered

    await coordinator.poll_once()
    assert outbound[0].metadata["_nutrition_candidate_id"] == head
    assert await coordinator.on_send_failure(outbound[0], RuntimeError("telegram timeout"))
    assert NutritionResultStore(tmp_path / head / "result.json").load().state == ResultState.delivery_unknown

    await coordinator.poll_once()
    assert len(outbound) == 2
    assert outbound[1].metadata["_nutrition_candidate_id"] == later
    assert NutritionResultStore(tmp_path / later / "result.json").load().state == ResultState.prompt_sending


@pytest.mark.asyncio
async def test_delivery_unknown_still_allows_only_one_later_pending_prompt(tmp_path: Path) -> None:
    for index in range(3):
        _candidate(tmp_path, file_id=f"id:candidate-{index}", rev=f"rev:candidate-{index}")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config, honcho_client=_RecentSource(), publish_outbound=outbound.append
    )
    ordered = [item.candidate_id for item in coordinator._scanner.scan_ready()]
    head, later, tail = ordered

    await coordinator.poll_once()
    await coordinator.on_send_failure(outbound[0], RuntimeError("telegram timeout"))
    await coordinator.poll_once()
    await coordinator.on_send_success(
        outbound[1],
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), outbound[1].metadata["_trusted_outbound_operation_id"]
        ),
    )
    await coordinator.poll_once()

    assert NutritionResultStore(tmp_path / head / "result.json").load().state == ResultState.delivery_unknown
    assert NutritionResultStore(tmp_path / later / "result.json").load().state == ResultState.pending_confirmation
    assert NutritionResultStore(tmp_path / tail / "result.json").load().state == ResultState.published
    assert [item.metadata["_nutrition_candidate_id"] for item in outbound] == [head, later]


@pytest.mark.asyncio
async def test_plain_text_and_old_callback_do_not_mutate_quarantined_or_current_candidate(
    tmp_path: Path,
) -> None:
    _candidate(tmp_path, file_id="id:quarantined", rev="rev:quarantined")
    _candidate(tmp_path, file_id="id:current", rev="rev:current")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config, honcho_client=_RecentSource(), publish_outbound=outbound.append
    )
    ordered = [item.candidate_id for item in coordinator._scanner.scan_ready()]
    quarantined, current = ordered

    await coordinator.poll_once()
    await coordinator.on_send_failure(outbound[0], RuntimeError("telegram timeout"))
    await coordinator.poll_once()
    await coordinator.on_send_success(
        outbound[1],
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), outbound[1].metadata["_trusted_outbound_operation_id"]
        ),
    )
    before = {
        candidate_id: NutritionResultStore(tmp_path / candidate_id / "result.json").load().model_dump(
            mode="json"
        )
        for candidate_id in (quarantined, current)
    }

    for message in (
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
        ),
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Нет",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 41},
        ),
    ):
        assert await coordinator.handle_inbound(message) is True

    after = {
        candidate_id: NutritionResultStore(tmp_path / candidate_id / "result.json").load().model_dump(
            mode="json"
        )
        for candidate_id in (quarantined, current)
    }
    assert after == before
    assert len([item for item in outbound if item.metadata.get("_nutrition_confirmation")]) == 2


@pytest.mark.asyncio
async def test_restart_reconciles_persisted_prompt_sending_without_resend(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:restart", rev="rev:restart")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    first_outbound = []
    first = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=first_outbound.append)
    await first.poll_once()
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.prompt_sending

    second_outbound = []
    restarted = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=second_outbound.append)
    await restarted.poll_once()
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.delivery_unknown
    assert second_outbound == []


@pytest.mark.asyncio
async def test_restart_publishes_candidate_persisted_before_prompt_publication(
    tmp_path: Path, monkeypatch
) -> None:
    candidate = _candidate(tmp_path, file_id="id:before-prompt", rev="rev:before-prompt")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )

    async def crash_before_publication(_artifact, _sidecar) -> None:
        raise RuntimeError("simulated crash before bus publication")

    first = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=lambda _message: None)
    monkeypatch.setattr(first, "_publish_prompt", crash_before_publication)
    with pytest.raises(RuntimeError, match="before bus publication"):
        await first.poll_once()
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.published

    outbound = []
    restarted = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append)
    await restarted.poll_once()
    assert len(outbound) == 1
    assert outbound[0].metadata["_nutrition_candidate_id"] == candidate
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.prompt_sending


@pytest.mark.asyncio
async def test_runtime_pool_nutrition_chain_reconciles_crash_after_honcho_commit(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path, file_id="id:runtime-e2e", rev="rev:runtime-e2e")
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    honcho = _HonchoStore()
    first_pool = _nutrition_runtime_pool(tmp_path, workspace, honcho)
    outbound = []
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        retry_backoff_seconds=0,
    )
    first = NutritionIngestCoordinator(
        config,
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        runtime_pool=_CrashAfterDurableCommit(first_pool),
    )
    assert first._estimate is None

    await first.poll_once()
    prompt = outbound[0]
    await first.on_send_success(
        prompt,
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]
        ),
    )
    await first.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    )
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.retryable_error
    assert sidecar.consumption_status == "consumed"
    assert len(honcho.messages) == 2

    second_pool = _nutrition_runtime_pool(tmp_path, workspace, honcho)
    restarted = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append, runtime_pool=second_pool)
    await restarted.poll_once()

    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.completed
    assert sidecar.emitted_honcho_message_id == "honcho-2"
    assert len(outbound) == 2
    summary = outbound[1]
    assert summary.metadata["_trusted_outbound_operation_id"] == f"{candidate}:summary:v1"
    assert summary.metadata["_nutrition_display_summary"] == {
        "schema_version": 1,
        "calories_kcal": 550.0,
        "protein_g": 30.0,
        "fat_g": 20.0,
        "carbohydrate_g": 45.0,
    }
    assert summary.reply_to == "42"
    assert "КБЖУ" in summary.content
    assert "550" in summary.content
    assert "999" not in summary.content
    assert "Б 30" in summary.content
    assert "Ж 20" in summary.content
    assert "У 45" in summary.content
    await restarted.poll_once()
    assert len(outbound) == 2
    assert len(honcho.messages) == 2

    by_role = {message.metadata["role"]: message for message in honcho.messages}
    user = by_role["user"]
    assistant = by_role["assistant"]
    expected_common = {
        "_nutrition_trusted": True,
        "ingest_source": "dropbox_camera",
        "tenant_id": "marina",
        "confirmation_required": True,
        "candidate_id": candidate,
        "nutrition_phase": "estimation",
    }
    assert {key: user.metadata[key] for key in expected_common} == expected_common
    assert {key: assistant.metadata[key] for key in expected_common} == expected_common
    assert user.metadata["client_op_id"] == f"{candidate}:meal-user:v1"
    assert assistant.metadata["client_op_id"] == f"{candidate}:meal-observation:v1"
    annotation = assistant.metadata["decision_trace"]["annotations"]["nutrition"]
    assert annotation["schema_version"] == 2
    assert annotation["record_type"] == "meal_observation"
    assert annotation["consumption_status"] == "consumed"
    assert annotation["energy_kcal_best"] == 550
    assert annotation["meal_at"] == "2099-01-01T00:00:00+00:00"
    assert "EXIF is evidence only" not in json.dumps(annotation)


@pytest.mark.asyncio
async def test_runtime_pool_missing_macro_fails_closed_without_result(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:missing-macro", rev="rev:missing-macro")
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    honcho = _HonchoStore()
    pool = _nutrition_runtime_pool(
        tmp_path,
        workspace,
        honcho,
        model_stream=_NutritionModelStream(protein_g=None),
    )
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        retry_backoff_seconds=0,
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config,
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        runtime_pool=pool,
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]
        ),
    )
    await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    )

    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.retryable_error
    assert len(honcho.messages) == 0
    assert len(outbound) == 1


@pytest.mark.asyncio
async def test_invalid_success_receipt_is_ambiguous_and_replay_is_durable(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append)
    await coordinator.poll_once()
    prompt = outbound[0]

    assert await coordinator.on_send_success(prompt, None) is False
    store = NutritionResultStore(tmp_path / candidate / "result.json")
    assert store.load().state == ResultState.delivery_unknown
    assert coordinator.request_replay(candidate) is True
    assert store.load().state == ResultState.published
    assert len(outbound) == 1
    await coordinator.poll_once()
    assert len(outbound) == 2


@pytest.mark.asyncio
async def test_missing_estimator_receipt_is_retryable_not_completed(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config,
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        estimate=lambda _message: None,
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt("telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]),
    )
    assert await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    ) is True
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.retryable_error
    assert sidecar.consumption_status == "consumed"


@pytest.mark.asyncio
async def test_estimation_retries_are_bounded_without_reprompting(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path)
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        max_estimation_attempts=2,
        retry_backoff_seconds=0,
    )
    outbound = []
    estimates = []

    def fail_estimate(_message):
        estimates.append(True)
        raise RuntimeError("estimator unavailable")

    coordinator = NutritionIngestCoordinator(
        config,
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        estimate=fail_estimate,
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt("telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]),
    )
    await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    )
    await coordinator.poll_once()
    await coordinator.poll_once()

    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.dead_letter
    assert sidecar.consumption_status == "consumed"
    assert len([item for item in sidecar.attempts if item.stage == "estimation"]) == 2
    assert len(outbound) == 1
    assert len(estimates) == 2


@pytest.mark.asyncio
async def test_retry_backoff_keeps_head_candidate_in_front_of_later_prompt(tmp_path: Path) -> None:
    first = _candidate(tmp_path, file_id="id:first", rev="rev:first")
    second = _candidate(tmp_path, file_id="id:second", rev="rev:second")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        retry_backoff_seconds=10,
    )
    clock = _Clock()
    outbound = []

    async def publish(message):
        outbound.append(message)
        if len(outbound) == 1:
            raise RuntimeError("temporary prompt publisher failure")

    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=publish, now=clock)
    ordered = [item.candidate_id for item in coordinator._scanner.scan_ready()]
    head = ordered[0]
    tail = ordered[1]

    await coordinator.poll_once()
    assert outbound[0].metadata["_nutrition_candidate_id"] == head
    assert NutritionResultStore(tmp_path / head / "result.json").load().state == ResultState.retryable_error

    clock.advance(9)
    await coordinator.poll_once()
    assert len(outbound) == 1

    clock.advance(1)
    await coordinator.poll_once()
    assert outbound[1].metadata["_nutrition_candidate_id"] == head
    await coordinator.on_send_success(
        outbound[1],
        OutboundDeliveryReceipt("telegram", "123", (42,), outbound[1].metadata["_trusted_outbound_operation_id"]),
    )
    await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Нет",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    )
    await coordinator.poll_once()
    prompts = [item for item in outbound if item.metadata.get("_nutrition_confirmation")]
    assert prompts[-1].metadata["_nutrition_candidate_id"] == tail
    assert {first, second} == {head, tail}


@pytest.mark.asyncio
async def test_prompt_retry_backoff_reaches_dead_letter_at_prompt_limit(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:prompt-limit", rev="rev:prompt-limit")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        retry_backoff_seconds=10,
        max_prompt_attempts=2,
    )
    clock = _Clock()
    attempts = []

    async def publish(_message):
        attempts.append(clock())
        raise RuntimeError("temporary prompt publisher failure")

    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=publish, now=clock)
    await coordinator.poll_once()
    clock.advance(9)
    await coordinator.poll_once()
    assert len(attempts) == 1
    clock.advance(1)
    await coordinator.poll_once()
    assert len(attempts) == 2
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.dead_letter
    clock.advance(100)
    await coordinator.poll_once()
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_operator_replay_resumes_consumed_estimation_without_reprompt(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:replay", rev="rev:replay")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        max_estimation_attempts=1,
    )
    outbound = []
    estimates = []

    def estimate(_message):
        estimates.append(True)
        if len(estimates) == 1:
            raise RuntimeError("estimator unavailable")
        return "honcho-replay"

    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append, estimate=estimate)
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt("telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]),
    )
    await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    )
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.dead_letter
    assert coordinator.request_replay(candidate) is True
    await coordinator.poll_once()
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.completed
    assert sidecar.emitted_honcho_message_id == "honcho-replay"
    assert len(outbound) == 1
    assert len(estimates) == 2


@pytest.mark.asyncio
async def test_estimation_backoff_doubles_caps_and_does_not_reconfirm(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:backoff", rev="rev:backoff")
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        retry_backoff_seconds=2000,
        max_estimation_attempts=4,
    )
    clock = _Clock()
    outbound = []
    estimates = []

    def fail_estimate(message):
        estimates.append(message)
        raise RuntimeError("estimator unavailable")

    coordinator = NutritionIngestCoordinator(
        config,
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        estimate=fail_estimate,
        now=clock,
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt("telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]),
    )
    await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42},
        )
    )
    assert len(estimates) == 1

    clock.advance(1999)
    await coordinator.poll_once()
    assert len(estimates) == 1
    clock.advance(1)
    await coordinator.poll_once()
    assert len(estimates) == 2
    clock.advance(3599)
    await coordinator.poll_once()
    assert len(estimates) == 2
    clock.advance(1)
    await coordinator.poll_once()
    assert len(estimates) == 3
    clock.advance(3599)
    await coordinator.poll_once()
    assert len(estimates) == 3
    clock.advance(1)
    await coordinator.poll_once()
    assert len(estimates) == 4

    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.dead_letter
    assert sidecar.consumption_status == "consumed"
    assert sidecar.prompt_message_id == 42
    assert len(outbound) == 1


@pytest.mark.asyncio
async def test_exif_boundary_is_inclusive_and_stale_candidate_does_not_block_fresh(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    clock.value = datetime(2026, 8, 12, 9, 0, tzinfo=UTC)
    stale = _candidate(tmp_path, file_id="id:stale", rev="rev:stale",
                       capture_time="2026-08-05T11:59:59.999999+03:00")
    fresh = _candidate(tmp_path, file_id="id:fresh", rev="rev:fresh",
                       capture_time="2026-08-05T12:00:00+03:00")
    config = NutritionIngestConfig(enabled=True, synchronized_root=tmp_path,
                                   principal="123", chat_id="123", session_key="telegram:123")
    outbound = []
    metrics = NutritionMetrics()
    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append,
                                              metrics=metrics, now=clock)

    await coordinator.poll_once()

    assert not (tmp_path / stale).exists()
    assert json.loads((tmp_path / "_seen" / f"{stale}.json").read_text())["terminal_reason"] == "expired"
    assert len(outbound) == 1 and outbound[0].metadata["_nutrition_candidate_id"] == fresh
    assert metrics.snapshot()["events"] == {"verified_candidate|scan|": 1}
    await coordinator.poll_once()
    assert len(outbound) == 1
    restarted = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append, now=clock)
    await restarted.poll_once()
    assert len(outbound) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exif_updates", "reason"),
    [({"normalized_capture_time": None}, "exif_missing"),
     ({"timezone_status": "ambiguous"}, "exif_ambiguous"),
     ({"normalized_capture_time": "not-an-iso"}, "exif_invalid")],
)
async def test_unusable_exif_is_terminally_skipped(tmp_path: Path, exif_updates, reason: str) -> None:
    candidate = _candidate(tmp_path, file_id=f"id:{reason}", rev=f"rev:{reason}",
                           exif_updates=exif_updates)
    config = NutritionIngestConfig(enabled=True, synchronized_root=tmp_path,
                                   principal="123", chat_id="123", session_key="telegram:123")
    outbound = []
    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append, now=_Clock())

    await coordinator.poll_once()

    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.skipped
    assert sidecar.skip_reason == reason
    assert sidecar.consumption_status == "unknown"
    assert sidecar.prompt_message_id is None and sidecar.reply_message_id is None
    assert sidecar.emitted_honcho_message_id is None and outbound == []


@pytest.mark.asyncio
async def test_offsetless_moscow_capture_is_checked_again_before_send(tmp_path: Path) -> None:
    clock = _Clock()
    clock.value = datetime(2026, 8, 12, 9, 0, tzinfo=UTC)
    candidate = _candidate(tmp_path, file_id="id:offsetless", rev="rev:offsetless",
                           capture_time="2026-08-05T12:00:00",
                           exif_updates={"timezone_status": "missing", "capture_timezone_offset": None})
    config = NutritionIngestConfig(enabled=True, synchronized_root=tmp_path,
                                   principal="123", chat_id="123", session_key="telegram:123")
    outbound = []
    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append, now=clock)
    original = coordinator._publish_prompt

    async def advance_before_send(artifact, sidecar):
        clock.advance(0.000001)
        await original(artifact, sidecar)

    coordinator._publish_prompt = advance_before_send
    await coordinator.poll_once()
    assert outbound == [] and not (tmp_path / candidate).exists()
    assert json.loads((tmp_path / "_seen" / f"{candidate}.json").read_text())["terminal_reason"] == "expired"


@pytest.mark.asyncio
async def test_old_pending_confirmation_cannot_consume_reply(tmp_path: Path) -> None:
    clock = _Clock()
    candidate = _candidate(tmp_path, file_id="id:legacy", rev="rev:legacy",
                           capture_time="2026-08-05T12:00:00+03:00")
    config = NutritionIngestConfig(enabled=True, synchronized_root=tmp_path,
                                   principal="123", chat_id="123", session_key="telegram:123")
    outbound, estimates = [], []
    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append,
                                              estimate=estimates.append, now=clock)
    await coordinator.poll_once()
    await coordinator.on_send_success(
        outbound[0], OutboundDeliveryReceipt("telegram", "123", (42,),
                                              outbound[0].metadata["_trusted_outbound_operation_id"])
    )
    clock.advance(7 * 24 * 60 * 60 + 0.000001)
    handled = await coordinator.handle_inbound(InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="Да",
        session_key_override="telegram:123"))
    assert handled is False and estimates == []
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.pending_confirmation


@pytest.mark.asyncio
async def test_consumed_estimation_retry_expires_with_bounded_state_summary(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    candidate = _candidate(tmp_path, file_id="id:aging", rev="rev:aging",
                           capture_time="2026-08-05T12:00:00+03:00")
    config = NutritionIngestConfig(enabled=True, synchronized_root=tmp_path,
                                   principal="123", chat_id="123", session_key="telegram:123",
                                   retry_backoff_seconds=0)
    outbound, calls = [], []

    def estimate(_message):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("estimator unavailable")
        return "honcho-aged"

    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append,
                                              estimate=estimate, now=clock)
    await coordinator.poll_once()
    await coordinator.on_send_success(
        outbound[0], OutboundDeliveryReceipt("telegram", "123", (42,),
                                              outbound[0].metadata["_trusted_outbound_operation_id"])
    )
    await coordinator.handle_inbound(InboundMessage(
        channel="telegram", sender_id="123", chat_id="123", content="Да",
        session_key_override="telegram:123",
        metadata={"callback_query": True, "native_message_id": 42}))
    clock.advance(7 * 24 * 60 * 60 + 1)
    await coordinator.poll_once()
    assert len(calls) == 1
    assert not (tmp_path / candidate).exists()
    tombstone = json.loads((tmp_path / "_seen" / f"{candidate}.json").read_text())
    assert tombstone["terminal_reason"] == "expired"
    assert tombstone["state_summary"] == "retryable_error:consumed_incomplete"


def test_freshness_helper_requires_aware_clock() -> None:
    exif = ExifMetadata(timezone_status="known",
                        normalized_capture_time="2026-08-05T12:00:00+03:00")
    with pytest.raises(ValueError, match="timezone-aware"):
        exif_freshness_reason(exif, datetime.fromisoformat("2026-08-05T10:00:00"))


@pytest.mark.asyncio
@pytest.mark.parametrize("exif_updates", [
    {"timezone_status": "known", "capture_timezone_offset": None},
    {"timezone_status": "missing", "capture_timezone_offset": "+03:00"},
])
async def test_naive_exif_with_untruthful_provenance_is_invalid(tmp_path: Path, exif_updates) -> None:
    candidate = _candidate(tmp_path, file_id="id:invalid", rev=str(exif_updates),
                           capture_time="2026-08-05T12:00:00", exif_updates=exif_updates)
    config = NutritionIngestConfig(enabled=True, synchronized_root=tmp_path,
                                   principal="123", chat_id="123", session_key="telegram:123")
    metrics = NutritionMetrics()
    await NutritionIngestCoordinator(config, honcho_client=_RecentSource(), metrics=metrics, now=_Clock()).poll_once()
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.skipped and sidecar.skip_reason == "exif_invalid"
    assert metrics.snapshot()["events"] == {}


@pytest.mark.asyncio
async def test_aware_capture_remains_valid_with_missing_offset_provenance(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:aware", rev="rev:aware",
                           capture_time="2026-08-05T12:00:00+03:00",
                           exif_updates={"timezone_status": "missing", "capture_timezone_offset": None})
    config = NutritionIngestConfig(enabled=True, synchronized_root=tmp_path,
                                   principal="123", chat_id="123", session_key="telegram:123")
    outbound = []
    await NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=outbound.append,
                                     now=_Clock()).poll_once()
    assert len(outbound) == 1 and outbound[0].metadata["_nutrition_candidate_id"] == candidate


@pytest.mark.asyncio
async def test_skipping_candidate_preserves_historical_attempts(tmp_path: Path) -> None:
    clock = _Clock()
    candidate = _candidate(tmp_path, file_id="id:attempts", rev="rev:attempts",
                           capture_time="2026-08-05T12:00:00+03:00")
    config = NutritionIngestConfig(enabled=True, synchronized_root=tmp_path,
                                   principal="123", chat_id="123", session_key="telegram:123",
                                   retry_backoff_seconds=0)

    async def fail(_message):
        raise RuntimeError("temporary prompt failure")

    coordinator = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), publish_outbound=fail, now=clock)
    await coordinator.poll_once()
    before = NutritionResultStore(tmp_path / candidate / "result.json").load()
    clock.advance(7 * 24 * 60 * 60 + 1)
    await coordinator.poll_once()
    assert before.state == ResultState.retryable_error
    assert not (tmp_path / candidate).exists()
    archived = json.loads((tmp_path / "_seen" / f"{candidate}.json").read_text())
    assert archived["terminal_reason"] == "expired"
    assert archived["state_summary"] == "retryable_error"
@pytest.mark.asyncio
async def test_exact_sha_duplicate_becomes_seen_and_writes_owner_only_tombstone(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path, file_id="id:seen", rev="rev:seen")
    sha256 = hashlib.sha256(b"native-photo").hexdigest()
    source = _RecentSource(
        [
            _recent_message(
                [{"sha256": sha256}],
                peer_id="marina-peer",
                session_id="marina-session",
            )
        ]
    )
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config,
        publish_outbound=outbound.append,
        honcho_client=source,
        honcho_session="marina-session",
        observed_peer="marina-peer",
        now=_Clock(),
    )
    tombstone_path = tmp_path / "_seen" / f"{candidate}.json"
    ordering = []
    captured_sidecar = []
    assert coordinator._tombstones is not None
    original_replace = coordinator._tombstones.replace
    original_delete = coordinator._delete_candidate_directory

    def replace(value):
        assert (tmp_path / candidate).is_dir()
        result = original_replace(value)
        ordering.append("tombstone")
        return result

    def delete(artifact):
        assert tombstone_path.is_file()
        captured_sidecar.append(NutritionResultStore(artifact.directory / "result.json").load())
        ordering.append("delete")
        return original_delete(artifact)

    coordinator._tombstones.replace = replace
    coordinator._delete_candidate_directory = delete

    await coordinator.poll_once()

    sidecar = captured_sidecar[0]
    assert sidecar.state == ResultState.seen
    assert sidecar.seen_reason == "duplicate_honcho"
    assert sidecar.seen_fingerprint_kind == "sha256"
    assert sidecar.matched_honcho_message_id == "honcho-user-1"
    assert sidecar.consumption_status == "unknown"
    assert sidecar.prompt_message_id is None
    assert sidecar.reply_message_id is None
    assert sidecar.emitted_honcho_message_id is None
    assert outbound == []
    assert ordering == ["tombstone", "delete"]
    assert not (tmp_path / candidate).exists()
    tombstone = json.loads(tombstone_path.read_text())
    assert tombstone["terminal_reason"] == "duplicate_honcho"
    assert tombstone["original_sha256"] == sha256
    assert tombstone["matched_honcho_message_id"] == "honcho-user-1"
    assert tombstone["matching_fingerprint_kind"] == "sha256"
    assert "source_path" not in tombstone and "filename" not in tombstone
    assert stat.S_IMODE(tombstone_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(tombstone_path.parent.stat().st_mode) == 0o700
    assert source.requests == [
        {
            "session": "marina-session",
            "expected_peer_id": "marina-peer",
            "since": datetime(2026, 7, 29, 10, tzinfo=UTC),
            "until": datetime(2026, 8, 5, 10, tzinfo=UTC),
        }
    ]


@pytest.mark.parametrize(
    "bit_flips, algorithm, expected_duplicate",
    [
        (PHASH_HAMMING_THRESHOLD, PHASH_ALGORITHM, True),
        (PHASH_HAMMING_THRESHOLD + 1, PHASH_ALGORITHM, False),
        (PHASH_HAMMING_THRESHOLD, "ahash-16x16-gray-v1", False),
    ],
)
@pytest.mark.asyncio
async def test_phash_metadata_matches_at_threshold_and_rejects_above_it(
    tmp_path: Path,
    bit_flips: int,
    algorithm: str,
    expected_duplicate: bool,
) -> None:
    data = _png((50, 100, 150))
    descriptor = fingerprint_image_bytes(data)
    assert descriptor is not None
    first = _candidate(
        tmp_path,
        file_id="id:phash-match",
        rev="rev:phash-match",
        capture_time="2026-08-05T09:00:00+00:00",
        data=data,
        filename="photo.png",
    )
    source = _RecentSource(
        [
            _recent_message(
                [
                    {
                        "phash": f"{int(descriptor['phash'], 16) ^ ((1 << bit_flips) - 1):064x}",
                        "phash_algorithm": algorithm,
                    }
                ]
            )
        ]
    )
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config,
        publish_outbound=outbound.append,
        honcho_client=source,
        now=_Clock(),
    )
    await coordinator.poll_once()
    assert (tmp_path / first).is_dir() is not expected_duplicate
    assert len(outbound) == (0 if expected_duplicate else 1)
    if not expected_duplicate:
        assert outbound[0].metadata["_nutrition_candidate_id"] == first


@pytest.mark.asyncio
async def test_untrusted_non_user_and_trusted_dropbox_sha_records_are_handled(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path, file_id="id:exclude", rev="rev:exclude")
    sha256 = hashlib.sha256(b"native-photo").hexdigest()
    fingerprints = [{"sha256": sha256}]
    source = _RecentSource(
        [
            _recent_message(fingerprints, identifier="assistant", role="assistant"),
            _recent_message(fingerprints, identifier="other-tenant", tenant="owner"),
            _recent_message(
                fingerprints,
                identifier="dropbox",
                ingest_source="dropbox_camera",
            ),
            _recent_message([], identifier="missing-fields"),
        ]
    )
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    await NutritionIngestCoordinator(
        config,
        publish_outbound=outbound.append,
        honcho_client=source,
        now=_Clock(),
    ).poll_once()
    assert outbound == []
    assert not (tmp_path / candidate).exists()


@pytest.mark.asyncio
async def test_dropbox_phash_only_record_is_not_a_duplicate(tmp_path: Path) -> None:
    data = _png((50, 100, 150))
    descriptor = fingerprint_image_bytes(data)
    assert descriptor is not None
    candidate = _candidate(
        tmp_path,
        file_id="id:dropbox-phash-only",
        rev="rev:dropbox-phash-only",
        capture_time="2026-08-05T09:00:00+00:00",
        data=data,
        filename="photo.png",
    )
    source = _RecentSource(
        [
            _recent_message(
                [{"phash": descriptor["phash"], "phash_algorithm": PHASH_ALGORITHM}],
                ingest_source="dropbox_camera",
            )
        ]
    )
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    await NutritionIngestCoordinator(
        config, publish_outbound=outbound.append, honcho_client=source, now=_Clock()
    ).poll_once()
    assert len(outbound) == 1
    assert outbound[0].metadata["_nutrition_candidate_id"] == candidate


@pytest.mark.asyncio
async def test_ordinary_telegram_phash_outside_capture_time_gate_is_not_duplicate(
    tmp_path: Path,
) -> None:
    data = _png((50, 100, 150))
    descriptor = fingerprint_image_bytes(data)
    assert descriptor is not None
    candidate = _candidate(
        tmp_path,
        file_id="id:phash-time-gate",
        rev="rev:phash-time-gate",
        capture_time="2026-08-05T09:00:00+00:00",
        data=data,
        filename="photo.png",
    )
    source = _RecentSource(
        [
            _recent_message(
                [{"phash": descriptor["phash"], "phash_algorithm": PHASH_ALGORITHM}],
                created_at=datetime(2026, 8, 5, 6, tzinfo=UTC),
            )
        ]
    )
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    await NutritionIngestCoordinator(
        config, publish_outbound=outbound.append, honcho_client=source, now=_Clock()
    ).poll_once()
    assert len(outbound) == 1
    assert outbound[0].metadata["_nutrition_candidate_id"] == candidate


@pytest.mark.asyncio
async def test_wrong_peer_honcho_response_fails_closed_before_prompt(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:wrong-peer", rev="rev:wrong-peer")
    source = _RecentSource([_recent_message([], peer_id="unexpected-peer")])
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        retry_backoff_seconds=0,
    )
    outbound = []

    await NutritionIngestCoordinator(
        config,
        publish_outbound=outbound.append,
        honcho_client=source,
        now=_Clock(),
    ).poll_once()

    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.retryable_error
    assert sidecar.attempts[-1].stage == "dedup"
    assert outbound == []


@pytest.mark.asyncio
async def test_honcho_failure_is_retryable_fail_closed_and_never_sends(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path, file_id="id:honcho-fail", rev="rev:honcho-fail")
    source = _RecentSource(error=RuntimeError("Honcho unavailable"))
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        retry_backoff_seconds=0,
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config,
        publish_outbound=outbound.append,
        honcho_client=source,
        now=_Clock(),
    )
    await coordinator.poll_once()
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.retryable_error
    assert sidecar.attempts[-1].stage == "dedup"
    assert outbound == []

    source.error = None
    await coordinator.poll_once()
    assert len(outbound) == 1


@pytest.mark.asyncio
async def test_seen_candidate_does_not_block_next_serialized_prompt(tmp_path: Path) -> None:
    first = _candidate(tmp_path, file_id="id:first-seen", rev="rev:first-seen")
    second = _candidate(
        tmp_path,
        file_id="id:second-ready",
        rev="rev:second-ready",
        data=b"later-photo",
    )
    first_sha = json.loads((tmp_path / first / "manifest.json").read_text())["original_sha256"]
    source = _RecentSource([_recent_message([{"sha256": first_sha}])])
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config,
        publish_outbound=outbound.append,
        honcho_client=source,
        now=_Clock(),
    )
    await coordinator.poll_once()
    assert not (tmp_path / first).exists()
    assert (tmp_path / "_seen" / f"{first}.json").is_file()
    source.messages = []
    await coordinator.poll_once()
    assert len(outbound) == 1
    assert outbound[0].metadata["_nutrition_candidate_id"] == second


@pytest.mark.asyncio
async def test_recovery_of_seen_candidate_replaces_tombstone_before_deletion(
    tmp_path: Path,
) -> None:
    candidate = _candidate(tmp_path, file_id="id:seen-recovery", rev="rev:seen-recovery")
    sha256 = json.loads((tmp_path / candidate / "manifest.json").read_text())["original_sha256"]
    source = _RecentSource([_recent_message([{"sha256": sha256}])])
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    first = NutritionIngestCoordinator(config, honcho_client=source, now=_Clock())
    first._delete_candidate_directory = lambda artifact: None
    await first.poll_once()
    assert (tmp_path / candidate).is_dir()
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.seen

    restarted = NutritionIngestCoordinator(config, honcho_client=_RecentSource(), now=_Clock())
    tombstone_path = tmp_path / "_seen" / f"{candidate}.json"
    ordering = []
    assert restarted._tombstones is not None
    original_replace = restarted._tombstones.replace
    original_delete = restarted._delete_candidate_directory

    def replace(value):
        assert (tmp_path / candidate).is_dir()
        result = original_replace(value)
        ordering.append("tombstone")
        return result

    def delete(artifact):
        assert tombstone_path.is_file()
        ordering.append("delete")
        return original_delete(artifact)

    restarted._tombstones.replace = replace
    restarted._delete_candidate_directory = delete

    assert await restarted.poll_once() == []
    assert ordering == ["tombstone", "delete"]
    assert not (tmp_path / candidate).exists()
    tombstone = json.loads(tombstone_path.read_text())
    assert tombstone["terminal_reason"] == "duplicate_honcho"
    assert tombstone["original_sha256"] == sha256
    assert tombstone["matched_honcho_message_id"] == "honcho-user-1"
    assert tombstone["matching_fingerprint_kind"] == "sha256"


@pytest.mark.asyncio
async def test_expiry_writes_tombstone_before_direct_child_deletion_and_keeps_boundary(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    clock.value = datetime(2026, 8, 12, 9, tzinfo=UTC)
    expired = _candidate(
        tmp_path,
        file_id="id:expired-delete",
        rev="rev:expired-delete",
        capture_time="2026-08-05T08:59:59.999999+00:00",
    )
    boundary = _candidate(
        tmp_path,
        file_id="id:boundary-keep",
        rev="rev:boundary-keep",
        capture_time="2026-08-05T09:00:00+00:00",
    )
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config,
        publish_outbound=outbound.append,
        honcho_client=_RecentSource(),
        now=clock,
    )
    ordering: list[str] = []
    assert coordinator._tombstones is not None
    original_replace = coordinator._tombstones.replace
    original_delete = coordinator._delete_candidate_directory

    def replace_first(value):
        assert (tmp_path / value.candidate_id).is_dir()
        ordering.append(f"tombstone:{value.candidate_id}")
        return original_replace(value)

    def delete_after(artifact):
        assert (tmp_path / "_seen" / f"{artifact.candidate_id}.json").is_file()
        ordering.append(f"delete:{artifact.candidate_id}")
        return original_delete(artifact)

    coordinator._tombstones.replace = replace_first
    coordinator._delete_candidate_directory = delete_after
    await coordinator.poll_once()
    assert not (tmp_path / expired).exists()
    expired_tombstone = tmp_path / "_seen" / f"{expired}.json"
    assert json.loads(expired_tombstone.read_text())["terminal_reason"] == "expired"
    assert (tmp_path / boundary).is_dir()
    assert len(outbound) == 1
    assert ordering == [f"tombstone:{expired}", f"delete:{expired}"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("delivery_state", "expected_summary"),
    [
        ("pending", "pending_confirmation"),
        ("unknown", "delivery_unknown"),
    ],
)
async def test_expiry_removes_pending_and_delivery_unknown_but_preserves_reserved_dirs(
    tmp_path: Path,
    delivery_state: str,
    expected_summary: str,
) -> None:
    clock = _Clock()
    candidate = _candidate(
        tmp_path,
        file_id="id:pending-expiry",
        rev="rev:pending-expiry",
        capture_time="2026-08-05T10:00:00+00:00",
    )
    for name in ("_producer", "_errors", ".hidden", "candidate.tmp"):
        (tmp_path / name).mkdir()
    config = NutritionIngestConfig(
        enabled=True,
        synchronized_root=tmp_path,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        config,
        publish_outbound=outbound.append,
        honcho_client=_RecentSource(),
        estimate=lambda _message: (_ for _ in ()).throw(AssertionError("late estimate")),
        now=clock,
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    receipt = (
        OutboundDeliveryReceipt(
            "telegram",
            "123",
            (42,),
            prompt.metadata["_trusted_outbound_operation_id"],
        )
        if delivery_state == "pending"
        else None
    )
    await coordinator.on_send_success(prompt, receipt)
    clock.advance(7 * 24 * 60 * 60 + 1)
    await coordinator.poll_once()
    assert not (tmp_path / candidate).exists()
    tombstone = json.loads((tmp_path / "_seen" / f"{candidate}.json").read_text())
    assert tombstone["terminal_reason"] == "expired"
    assert tombstone["state_summary"] == expected_summary
    assert tombstone["archived_at"] == "2026-08-12T10:00:01Z"
    assert all(
        (tmp_path / name).is_dir() for name in ("_producer", "_errors", ".hidden", "candidate.tmp")
    )
    if delivery_state == "pending":
        late_callback = InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да",
            session_key_override="telegram:123",
            metadata={"callback_query": True, "native_message_id": 42, "message_id": 43},
        )
        assert await coordinator.handle_inbound(late_callback) is False


def test_tombstone_path_rejects_invalid_candidate_and_seen_symlink(tmp_path: Path) -> None:
    from ohmo.nutrition_ingest.tombstones import SeenTombstoneStore

    store = SeenTombstoneStore(tmp_path)
    with pytest.raises(ValueError):
        store.path_for("../escape")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / "_seen")
    candidate = candidate_id_for("id:symlink", "rev:symlink")
    with pytest.raises(ValueError, match="real directory"):
        store.replace(
            SeenTombstoneV1(
                candidate_id=candidate,
                original_sha256="0" * 64,
                terminal_reason="expired",
                capture_time=datetime(2026, 8, 1, tzinfo=UTC),
                archived_at=datetime(2026, 8, 8, tzinfo=UTC),
                state_summary="ready",
            )
        )
