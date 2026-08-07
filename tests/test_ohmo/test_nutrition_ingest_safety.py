from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ohmo.gateway.bridge import OhmoGatewayBridge
from ohmo.gateway.models import NutritionIngestConfig
from ohmo.nutrition_ingest.coordinator import NutritionIngestCoordinator
from ohmo.nutrition_ingest.models import ResultState
from ohmo.nutrition_ingest.prompts import (
    NO_VISIBLE_CONSUMABLE_PORTION_REJECTION,
    build_post_confirmation_prompt,
)
from ohmo.nutrition_ingest.sidecars import NutritionResultStore
from ohmo.workspace import initialize_workspace
from openharness.channels.bus.events import InboundMessage, OutboundDeliveryReceipt
from openharness.channels.bus.queue import MessageBus
from tests.test_ohmo.test_nutrition_ingest_coordinator import (
    _candidate,
    _Clock,
    _HonchoStore,
    _nutrition_runtime_pool,
    _NutritionModelStream,
    _RecentSource,
)


def _config(root: Path) -> NutritionIngestConfig:
    return NutritionIngestConfig(
        enabled=True,
        synchronized_root=root,
        principal="123",
        chat_id="123",
        session_key="telegram:123",
        retry_backoff_seconds=0,
    )


@pytest.mark.asyncio
async def test_confirmation_keyboard_and_non_food_callback_are_terminal(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:safety-buttons", rev="rev:safety-buttons")
    outbound = []
    estimates = []
    coordinator = NutritionIngestCoordinator(
        _config(tmp_path),
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        estimate=estimates.append,
    )

    await coordinator.poll_once()
    prompt = outbound[0]
    assert prompt.buttons == ["Да, я это съела", "Нет, не ела", "Это не еда"]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]
        ),
    )
    assert await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Это не еда",
            session_key_override="telegram:123",
            metadata={
                "callback_query": True,
                "native_message_id": 42,
                "callback_data": prompt.metadata["_nutrition_callback_prefix"] + "2",
                "message_id": 43,
            },
        )
    )
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.non_food
    assert sidecar.non_food_reason == "explicit_feedback"
    assert sidecar.consumption_status == "not_food"
    assert sidecar.prompt_message_id == 42
    assert sidecar.reply_message_id == 43
    assert sidecar.emitted_honcho_message_id is None
    assert estimates == []
    assert "ничего не записываю" in outbound[-1].content


@pytest.mark.asyncio
async def test_plain_text_and_wrong_callback_pass_through_without_mutation(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:safety-binding", rev="rev:safety-binding")
    outbound = []
    coordinator = NutritionIngestCoordinator(
        _config(tmp_path), honcho_client=_RecentSource(), publish_outbound=outbound.append
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]
        ),
    )
    before = NutritionResultStore(tmp_path / candidate / "result.json").load().model_dump()
    for metadata, expected_handled in (
        ({}, False),
        (
            {
                "callback_query": True,
                "native_message_id": 41,
                "callback_data": prompt.metadata["_nutrition_callback_prefix"] + "0",
            },
            True,
        ),
        (
            {
                "callback_query": True,
                "native_message_id": 42,
                "callback_data": "nutrition:old-candidate:0",
            },
            True,
        ),
        (
            {"callback_query": True, "native_message_id": 42, "callback_data": "ask:0"},
            True,
        ),
        ({"callback_query": True, "native_message_id": 42}, False),
        (
            {"callback_query": True, "native_message_id": 42, "callback_data": "other:0"},
            False,
        ),
        (
            {"callback_query": True, "native_message_id": 42, "callback_data": "nutrition:"},
            True,
        ),
        (
            {"callback_query": True, "native_message_id": 41, "callback_data": "ask:0"},
            False,
        ),
    ):
        assert (
            await coordinator.handle_inbound(
                InboundMessage(
                    channel="telegram",
                    sender_id="123",
                    chat_id="123",
                    content="Да, я это съела",
                    session_key_override="telegram:123",
                    metadata=metadata,
                )
            )
            is expected_handled
        )
    assert NutritionResultStore(tmp_path / candidate / "result.json").load().model_dump() == before

    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]
        ),
    )
    assert (
        await coordinator.handle_inbound(
            InboundMessage(
                channel="telegram",
                sender_id="123",
                chat_id="123",
                content="Да, я это съела",
                session_key_override="telegram:123",
                metadata={
                    "callback_query": True,
                    "native_message_id": 42,
                    "callback_data": prompt.metadata["_nutrition_callback_prefix"] + "0",
                },
            )
        )
        is True
    )


@pytest.mark.asyncio
async def test_current_nutrition_callback_terminally_handles_stale_candidate(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        tmp_path,
        file_id="id:safety-stale-callback",
        rev="rev:safety-stale-callback",
        capture_time="2026-08-05T09:00:00+00:00",
    )
    clock = _Clock()
    outbound = []
    coordinator = NutritionIngestCoordinator(
        _config(tmp_path),
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        now=clock,
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]
        ),
    )
    clock.advance(7 * 24 * 60 * 60 + 1)

    handled = await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да, я это съела",
            session_key_override="telegram:123",
            metadata={
                "callback_query": True,
                "native_message_id": 42,
                "callback_data": prompt.metadata["_nutrition_callback_prefix"] + "0",
            },
        )
    )

    assert handled is True
    assert not (tmp_path / candidate).exists()


@pytest.mark.asyncio
async def test_current_nutrition_callback_terminally_handles_invalid_candidate(
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        tmp_path,
        file_id="id:safety-invalid-callback",
        rev="rev:safety-invalid-callback",
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        _config(tmp_path), honcho_client=_RecentSource(), publish_outbound=outbound.append
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]
        ),
    )
    manifest_path = tmp_path / candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["exif"]["normalized_capture_time"] = "not-an-iso"
    manifest_path.write_text(json.dumps(manifest))

    handled = await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да, я это съела",
            session_key_override="telegram:123",
            metadata={
                "callback_query": True,
                "native_message_id": 42,
                "callback_data": prompt.metadata["_nutrition_callback_prefix"] + "0",
            },
        )
    )

    assert handled is True
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.skipped
    assert sidecar.skip_reason == "exif_invalid"
    assert (
        await coordinator.handle_inbound(
            InboundMessage(
                channel="telegram",
                sender_id="123",
                chat_id="123",
                content="Да, я это съела",
                session_key_override="telegram:123",
                metadata={
                    "callback_query": True,
                    "native_message_id": 42,
                    "callback_data": prompt.metadata["_nutrition_callback_prefix"] + "0",
                },
            )
        )
        is True
    )


@pytest.mark.asyncio
async def test_ordinary_turn_lifecycle_pauses_and_releases_publication(tmp_path: Path) -> None:
    _candidate(tmp_path, file_id="id:safety-pause", rev="rev:safety-pause")
    coordinator = NutritionIngestCoordinator(
        _config(tmp_path), honcho_client=_RecentSource(), publish_outbound=lambda _message: None
    )
    ordinary = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="обычный разговор",
        session_key_override="telegram:123",
    )
    coordinator.on_ordinary_turn_start(ordinary)
    assert coordinator.ordinary_turn_in_flight
    assert await coordinator.poll_once() == []
    coordinator.on_ordinary_turn_finish(ordinary)
    assert not coordinator.ordinary_turn_in_flight
    assert await coordinator.poll_once()


@pytest.mark.asyncio
async def test_publication_pause_is_rechecked_after_blocked_dedup(tmp_path: Path) -> None:
    _candidate(tmp_path, file_id="id:safety-race", rev="rev:safety-race")

    class BlockingRecentSource(_RecentSource):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def list_recent_message_metadata(self, session: str, **kwargs):
            self.started.set()
            await self.release.wait()
            return await super().list_recent_message_metadata(session, **kwargs)

    source = BlockingRecentSource()
    outbound = []
    coordinator = NutritionIngestCoordinator(
        _config(tmp_path), honcho_client=source, publish_outbound=outbound.append
    )
    ordinary = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="обычный разговор",
        session_key_override="telegram:123",
    )

    poll_task = asyncio.create_task(coordinator.poll_once())
    await asyncio.wait_for(source.started.wait(), timeout=1)
    coordinator.on_ordinary_turn_start(ordinary)
    source.release.set()
    await poll_task

    assert outbound == []
    assert coordinator.ordinary_turn_in_flight
    coordinator.on_ordinary_turn_finish(ordinary)
    await coordinator.poll_once()
    assert len(outbound) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_bridge_lifecycle_releases_pause_on_completion_and_exception(
    tmp_path: Path, raises: bool
) -> None:
    coordinator = NutritionIngestCoordinator(_config(tmp_path))
    started = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="обычный разговор",
        session_key_override="telegram:123",
    )

    class Runtime:
        async def stream_message(self, message, session_key):
            if raises:
                raise RuntimeError("runtime failure")
            yield SimpleNamespace(kind="final", text="готово", metadata={})

    bridge = OhmoGatewayBridge(
        bus=MessageBus(), runtime_pool=Runtime(), nutrition_coordinator=coordinator
    )
    await bridge._dispatch(started, started.session_key)
    assert coordinator.ordinary_turn_in_flight
    turn = bridge._session_tasks[started.session_key]
    await turn
    await asyncio.sleep(0)
    assert not coordinator.ordinary_turn_in_flight
    bridge.stop()


@pytest.mark.asyncio
async def test_bridge_lifecycle_releases_pause_on_cancellation_and_replacement(
    tmp_path: Path,
) -> None:
    coordinator = NutritionIngestCoordinator(_config(tmp_path))
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    first = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="первая задача",
        session_key_override="telegram:123",
    )
    second = InboundMessage(
        channel="telegram",
        sender_id="123",
        chat_id="123",
        content="вторая задача",
        session_key_override="telegram:123",
    )

    class Runtime:
        async def stream_message(self, message, session_key):
            if message.content == first.content:
                first_started.set()
                await asyncio.Event().wait()
            second_started.set()
            await release_second.wait()
            yield SimpleNamespace(kind="final", text="готово", metadata={})

    bridge = OhmoGatewayBridge(
        bus=MessageBus(), runtime_pool=Runtime(), nutrition_coordinator=coordinator
    )
    await bridge._dispatch(first, first.session_key)
    await asyncio.wait_for(first_started.wait(), timeout=1)
    assert coordinator.ordinary_turn_in_flight
    await bridge._dispatch(second, second.session_key)
    await asyncio.wait_for(second_started.wait(), timeout=1)
    assert coordinator.ordinary_turn_in_flight
    release_second.set()
    await bridge._session_tasks[second.session_key]
    await asyncio.sleep(0)
    assert not coordinator.ordinary_turn_in_flight
    bridge.stop()


def test_trusted_prompt_has_exact_visible_portion_rejection_contract() -> None:
    prompt = build_post_confirmation_prompt(candidate_id="candidate")
    assert "visible pixels" in prompt
    assert NO_VISIBLE_CONSUMABLE_PORTION_REJECTION in prompt
    assert "and no other text" in prompt


@pytest.mark.asyncio
async def test_exact_final_guard_skips_estimation_result_and_kbju(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:safety-guard", rev="rev:safety-guard")
    outbound = []
    honcho_calls = []

    def estimate(_message):
        honcho_calls.append(True)
        return {
            "_trusted_nutrition_terminal_outcome": "non_food",
            "_trusted_nutrition_rejection": "no_visible_consumable_portion",
            "_trusted_nutrition_rejection_payload": NO_VISIBLE_CONSUMABLE_PORTION_REJECTION,
        }

    coordinator = NutritionIngestCoordinator(
        _config(tmp_path),
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        estimate=estimate,
    )
    await coordinator.poll_once()
    prompt = outbound[0]
    await coordinator.on_send_success(
        prompt,
        OutboundDeliveryReceipt(
            "telegram", "123", (42,), prompt.metadata["_trusted_outbound_operation_id"]
        ),
    )
    assert await coordinator.handle_inbound(
        InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Да, я это съела",
            session_key_override="telegram:123",
            metadata={
                "callback_query": True,
                "native_message_id": 42,
                "callback_data": prompt.metadata["_nutrition_callback_prefix"] + "0",
            },
        )
    )
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.non_food
    assert sidecar.non_food_reason == "no_visible_consumable_portion"
    assert sidecar.emitted_honcho_message_id is None
    assert honcho_calls == [True]
    assert "КБЖУ" not in outbound[-1].content


@pytest.mark.asyncio
async def test_malformed_final_guard_metadata_is_retryable(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path, file_id="id:safety-malformed", rev="rev:safety-malformed")
    outbound = []
    coordinator = NutritionIngestCoordinator(
        _config(tmp_path),
        honcho_client=_RecentSource(),
        publish_outbound=outbound.append,
        estimate=lambda _message: {
            "_trusted_nutrition_terminal_outcome": "non_food",
            "_trusted_nutrition_rejection": "no_visible_consumable_portion",
        },
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
            content="Да, я это съела",
            session_key_override="telegram:123",
            metadata={
                "callback_query": True,
                "native_message_id": 42,
                "callback_data": prompt.metadata["_nutrition_callback_prefix"] + "0",
            },
        )
    )
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.retryable_error
    assert sidecar.consumption_status == "consumed"
    assert all("КБЖУ" not in message.content for message in outbound)


@pytest.mark.asyncio
async def test_runtime_exact_final_guard_skips_honcho_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _candidate(
        tmp_path, file_id="id:safety-runtime-guard", rev="rev:safety-runtime-guard"
    )
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    honcho = _HonchoStore()
    pool = _nutrition_runtime_pool(
        tmp_path,
        workspace,
        honcho,
        model_stream=_NutritionModelStream(answer=NO_VISIBLE_CONSUMABLE_PORTION_REJECTION),
    )
    scheduled_judges = []
    monkeypatch.setattr(
        pool,
        "_maybe_schedule_memory_judge",
        lambda *args, **kwargs: scheduled_judges.append((args, kwargs)),
    )
    outbound = []
    coordinator = NutritionIngestCoordinator(
        _config(tmp_path),
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
            content="Да, я это съела",
            session_key_override="telegram:123",
            metadata={
                "callback_query": True,
                "native_message_id": 42,
                "callback_data": prompt.metadata["_nutrition_callback_prefix"] + "0",
            },
        )
    )
    sidecar = NutritionResultStore(tmp_path / candidate / "result.json").load()
    assert sidecar.state == ResultState.non_food
    assert honcho.messages == []
    assert scheduled_judges == []
