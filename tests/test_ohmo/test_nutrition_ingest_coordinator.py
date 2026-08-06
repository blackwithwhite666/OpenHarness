from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.api.usage import UsageSnapshot
from openharness.channels.bus.events import InboundMessage, OutboundDeliveryReceipt
from openharness.channels.bus.queue import MessageBus
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.engine.stream_events import AssistantTextDelta, AssistantTurnComplete
from openharness.evals import TRACE_FINALIZATION
from openharness.tools.base import ToolRegistry
from ohmo.gateway.bridge import OhmoGatewayBridge
from ohmo.gateway.models import GatewayConfig, NutritionIngestConfig
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.memory_backend import CatalogMemoryBackend, ShadowMemoryBackend
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service.honcho_client import Message
from ohmo.nutrition_ingest.coordinator import NutritionIngestCoordinator
from ohmo.nutrition_ingest.metrics import NutritionMetrics
from ohmo.nutrition_ingest.models import ResultState, candidate_id_for
from ohmo.nutrition_ingest.sidecars import NutritionResultStore
from ohmo.workspace import initialize_workspace


def _candidate(
    root: Path,
    *,
    file_id: str = "id:test",
    rev: str = "rev:test",
    discovery_time: str = "2026-08-05T10:01:00+00:00",
) -> str:
    fixture = json.loads(
        (Path(__file__).parents[2] / "ohmo/nutrition_ingest/manifest_v1_fixture.json").read_text()
    )
    data = b"native-photo"
    candidate = candidate_id_for(file_id, rev)
    directory = root / candidate
    directory.mkdir()
    image = directory / "photo.jpg"
    image.write_bytes(data)
    fixture.update(
        candidate_id=candidate,
        file_id=file_id,
        rev=rev,
        discovery_time=discovery_time,
        original_filename="photo.jpg",
        original_size_bytes=len(data),
        original_sha256=hashlib.sha256(data).hexdigest(),
    )
    (directory / "manifest.json").write_text(json.dumps(fixture))
    return candidate


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 5, 10, 0, tzinfo=timezone.utc)

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
                created_at=datetime.now(timezone.utc),
                workspace_id="family-marina",
                token_count=1,
            )
            self.messages.append(message)
            created.append(message)
        return created


class _NutritionModelStream:
    """A model-only fake: emits one validated nutrition trace and final text."""

    def __init__(self) -> None:
        self.api_client = object()
        self.decision_trace_recorder = None
        self.engine = self
        self.model = "nutrition-test-model"
        self.max_turns = 1
        self.messages: list[ConversationMessage] = []
        self.tool_metadata: dict[str, object] = {}
        self.total_usage = UsageSnapshot()
        self.system_prompt = ""

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
                    }
                },
            },
        )
        answer = "Записала съеденный приём пищи: 550 ккал."
        yield AssistantTextDelta(answer)
        assistant = ConversationMessage(role="assistant", content=[TextBlock(text=answer)])
        self.messages.append(assistant)
        yield AssistantTurnComplete(message=assistant, usage=UsageSnapshot())


def _nutrition_runtime_pool(
    tmp_path: Path,
    workspace: Path,
    honcho: _HonchoStore,
) -> OhmoSessionRuntimePool:
    config = GatewayConfig(
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
        engine=_NutritionModelStream(),
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
        publish_outbound=outbound.append,
        estimate=estimates.append,
        metrics=metrics,
    )

    await coordinator.poll_once()
    assert len(outbound) == 1
    prompt = outbound[0]
    assert prompt.content == "Вы это съели?"
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
        )
    )
    assert handled is True
    assert estimates == []
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.completed
    assert sidecar.consumption_status == "not_consumed"
    assert sidecar.emitted_honcho_message_id is None
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
    coordinator = NutritionIngestCoordinator(config, publish_outbound=outbound.append)
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

    coordinator = NutritionIngestCoordinator(config, publish_outbound=outbound.append, estimate=estimate)
    await coordinator.poll_once()
    prompt = outbound[0]
    assert prompt.media and len(prompt.media) == 1
    assert prompt.content == "Вы это съели?"
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
    coordinator = NutritionIngestCoordinator(config, publish_outbound=coordinator_outbound.append)
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
    coordinator = NutritionIngestCoordinator(config, publish_outbound=outbound.append)
    await coordinator.poll_once()
    prompt = outbound[0]

    assert await coordinator.on_send_failure(prompt, RuntimeError("telegram timeout")) is True
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.delivery_unknown
    await coordinator.poll_once()
    assert len(outbound) == 1


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
    first = NutritionIngestCoordinator(config, publish_outbound=first_outbound.append)
    await first.poll_once()
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.prompt_sending

    second_outbound = []
    restarted = NutritionIngestCoordinator(config, publish_outbound=second_outbound.append)
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

    first = NutritionIngestCoordinator(config, publish_outbound=lambda _message: None)
    monkeypatch.setattr(first, "_publish_prompt", crash_before_publication)
    with pytest.raises(RuntimeError, match="before bus publication"):
        await first.poll_once()
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().state == ResultState.published

    outbound = []
    restarted = NutritionIngestCoordinator(config, publish_outbound=outbound.append)
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
        )
    )
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.retryable_error
    assert sidecar.consumption_status == "consumed"
    assert len(honcho.messages) == 2

    second_pool = _nutrition_runtime_pool(tmp_path, workspace, honcho)
    restarted = NutritionIngestCoordinator(config, publish_outbound=outbound.append, runtime_pool=second_pool)
    await restarted.poll_once()

    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.completed
    assert sidecar.emitted_honcho_message_id == "honcho-2"
    assert len(outbound) == 1
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
    coordinator = NutritionIngestCoordinator(config, publish_outbound=outbound.append)
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

    coordinator = NutritionIngestCoordinator(config, publish_outbound=publish, now=clock)
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

    coordinator = NutritionIngestCoordinator(config, publish_outbound=publish, now=clock)
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

    coordinator = NutritionIngestCoordinator(config, publish_outbound=outbound.append, estimate=estimate)
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
