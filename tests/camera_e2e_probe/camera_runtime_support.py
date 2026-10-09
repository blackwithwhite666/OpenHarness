"""Existing OpenHarness client seams for the isolated Camera prototype."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    SupportsStreamingMessages,
)
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import (
    AttachmentRefBlock,
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


SYNTHETIC_KCAL = 125
SYNTHETIC_BUTTON_LABEL = "Да, я это съел(а)"
SYNTHETIC_OPTIONS = (
    SYNTHETIC_BUTTON_LABEL,
    "Нет, не ел(а)",
    "Это не еда",
)
SYNTHETIC_WHOLE_PORTION = "Всю порцию"


def e5_raw_honcho_row_fingerprint(row: Mapping[str, Any]) -> str:
    """Fingerprint content and metadata from list_messages_in_window raw rows."""
    row_id = row.get("id")
    content = row.get("content")
    metadata = row.get("metadata")
    created_at = row.get("created_at")
    if (
        not isinstance(row_id, str) or not row_id
        or not isinstance(content, str)
        or not isinstance(metadata, Mapping)
        or not isinstance(created_at, str) or not created_at
    ):
        raise AssertionError("E5 immutability check requires a full raw Honcho row")
    snapshot = {
        "id": row_id,
        "content": content,
        "metadata": metadata,
        "created_at": created_at,
    }
    try:
        encoded = json.dumps(
            snapshot, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as error:
        raise AssertionError("E5 raw Honcho row is not stable bounded JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def assert_e5_raw_honcho_row_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    """Compare actual persisted raw content, metadata, and creation time."""
    if e5_raw_honcho_row_fingerprint(before) != e5_raw_honcho_row_fingerprint(after):
        raise AssertionError("E5 immutable original raw Honcho row changed")


def validate_e5_date_source_link(
    *, date_metadata: Mapping[str, Any], original_metadata: Mapping[str, Any],
    expected_date_source_id: str,
) -> None:
    """Validate the ordinary date route's trusted owner/session/reply source link."""
    source_id = original_metadata.get("source_message_id")
    date_source_id = date_metadata.get("source_message_id")
    provenance_keys = ("tenant_id", "source_principal", "gateway_session_id")
    if (
        not isinstance(source_id, str) or not source_id
        or not isinstance(date_source_id, str) or not date_source_id
        or date_source_id != expected_date_source_id
        or date_source_id == source_id
        or date_metadata.get("reply_to_source_message_id") != source_id
        or original_metadata.get("is_group") is not False
        or original_metadata.get("is_forwarded") is not False
        or date_metadata.get("is_group") is not False
        or date_metadata.get("is_forwarded") is not False
        or any(
            not isinstance(original_metadata.get(key), str)
            or not original_metadata.get(key)
            or date_metadata.get(key) != original_metadata.get(key)
            for key in provenance_keys
        )
    ):
        raise AssertionError("E5 ordinary date event is not linked to the trusted owner/source/session")


def validate_e5_unique_original_event_ids(
    *, observed_event_ids: Sequence[str], expected_original_event_id: str
) -> None:
    """Require the scoped Honcho read to contain exactly the retained original meal."""
    if list(observed_event_ids) != [expected_original_event_id]:
        raise AssertionError("E5 Honcho read does not contain the one original runtime observation")


def validate_e5_denial_receipt(
    *,
    correction_commit: Mapping[str, Any],
    outbound_event_id: str | None,
    candidate_id: str,
    original_event_id: str,
    original_metadata: Mapping[str, Any],
    honcho_row: Any,
) -> str:
    """Resolve a denial append from its Camera receipt and matching Honcho row."""
    metadata = honcho_row.get("metadata") if isinstance(honcho_row, Mapping) else None
    row_id = honcho_row.get("id") if isinstance(honcho_row, Mapping) else None
    event_id = correction_commit.get("event_id")
    source_id = correction_commit.get("source_message_id")
    client_op_id = correction_commit.get("client_op_id")
    original_source_id = original_metadata.get("source_message_id")
    trace = metadata.get("decision_trace") if isinstance(metadata, Mapping) else None
    annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
    nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
    changed_fields = nutrition.get("changed_fields") if isinstance(nutrition, Mapping) else None
    if (
        correction_commit.get("kind") != "denial"
        or not isinstance(event_id, str) or not event_id
        or not isinstance(source_id, str) or not source_id
        or not isinstance(client_op_id, str) or not client_op_id
        or correction_commit.get("target_event_id") != original_event_id
        or correction_commit.get("target_source_message_id") != original_source_id
        or (outbound_event_id is not None and outbound_event_id != event_id)
        or row_id != event_id
        or not isinstance(original_source_id, str) or not original_source_id
        or not isinstance(metadata, Mapping)
        or metadata.get("source_message_id") != source_id
        or metadata.get("reply_to_source_message_id") != original_source_id
        or metadata.get("client_op_id") != client_op_id
        or metadata.get("camera_candidate_id") != candidate_id
        or metadata.get("camera_operation_id") != candidate_id
        or metadata.get("camera_original_event_id") != original_event_id
        or metadata.get("camera_answer_bound") != "no"
        or metadata.get("camera_correction_bound") is not True
        or any(
            not isinstance(original_metadata.get(key), str)
            or not original_metadata.get(key)
            or metadata.get(key) != original_metadata.get(key)
            for key in ("tenant_id", "source_principal", "gateway_session_id")
        )
        or metadata.get("is_group") is not False
        or metadata.get("is_forwarded") is not False
        or not isinstance(nutrition, Mapping)
        or nutrition.get("record_type") != "meal_correction"
        or nutrition.get("consumption_status") != "not_consumed"
        or nutrition.get("energy_kcal_best") != 0
        or not isinstance(changed_fields, list)
        or "consumption_status" not in changed_fields
    ):
        raise AssertionError("E5 denial Camera receipt does not match its durable Honcho correction")
    return event_id


def validate_e5_post_correction_replay(
    *, status: str | None, delivery_receipt: Any, event_id: str | None,
    expected_event_id: str, original_commit: Mapping[str, Any],
    current_commit: Mapping[str, Any] | None,
    required_phrase: str | None = "исправление уже записано",
) -> str:
    """Require replay to report the latest correction without replacing the meal."""
    if (
        not isinstance(status, str)
        or (required_phrase is not None and required_phrase.casefold() not in status.casefold())
        or delivery_receipt is None
        or not isinstance(expected_event_id, str) or not expected_event_id
        or event_id != expected_event_id
        or current_commit != original_commit
    ):
        raise AssertionError("post-denial replay did not confirm its durable correction")
    return status


def _validate_completed_photo_replay(
    *,
    replay,
    candidate_id: str,
    turn_id: str,
    delivered,
    delivery_receipt,
    existing_event_id: str,
    original_commit: dict[str, Any],
    current_commit: dict[str, Any] | None,
    typed_replay: bool = False,
) -> str:
    """Require a delivered saved-state response tied to the immutable event."""
    if (
        replay.metadata.get(
            "_camera_typed_replay" if typed_replay else "_camera_existing_meal_replay"
        ) is not True
        or replay.metadata.get("_camera_unbound") is not None
        or replay.metadata.get("_camera_candidate_id") != candidate_id
        or replay.metadata.get("_camera_turn_id") != turn_id
    ):
        raise AssertionError("completed-photo replay was not bound to the existing saved meal")
    replay_final = next(
        (
            outbound for outbound, _ in reversed(delivered)
            if outbound.metadata.get("nutrition_append_event_id") is not None
        ),
        None,
    )
    if (
        replay_final is None
        or "уже записана" not in replay_final.content.casefold()
        or replay_final.metadata.get("nutrition_append_event_id") != existing_event_id
        or delivery_receipt is None
    ):
        raise AssertionError("completed-photo replay did not report its verified existing meal")
    if current_commit != original_commit or current_commit.get("event_id") != existing_event_id:
        raise AssertionError("completed-photo replay changed the immutable nutrition observation")
    return replay_final.content


class OfflineCameraBotApi:
    """Script ordinary tool sequencing; never infer intent from owner wording."""

    synthetic = True

    def __init__(self, *, candidate_id: str | None = None, context_items_date: bool = False,
                 omit_nutrition_annotation: bool = False) -> None:
        self.calls = 0
        self.finalization_proposals = 0
        self.candidate_id = candidate_id
        self.context_items_date = context_items_date
        self.omit_nutrition_annotation = omit_nutrition_annotation
        self._queued_correction: dict[str, Any] | None = None

    def queue_correction(self, nutrition: dict[str, Any]) -> None:
        if self._queued_correction is not None:
            raise AssertionError("offline correction queue is already occupied")
        self._queued_correction = nutrition

    @staticmethod
    def _emit(message: ConversationMessage, stop_reason: str):
        return ApiMessageCompleteEvent(
            message=message,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason=stop_reason,
        )

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiMessageCompleteEvent]:
        self.calls += 1
        attachments = [
            block for message in request.messages
            for block in message.content if isinstance(block, AttachmentRefBlock)
        ]
        current_user_index = next(
            (i for i in range(len(request.messages) - 1, -1, -1)
             if request.messages[i].role == "user"
             and not any(isinstance(block, ToolResultBlock)
                         for block in request.messages[i].content)),
            -1,
        )
        current_user = (
            request.messages[current_user_index] if current_user_index >= 0 else None
        )
        if current_user is None or any(
            isinstance(block, AttachmentRefBlock) for block in current_user.content
        ):
            yield self._emit(ConversationMessage(role="assistant", content=[
                TextBlock(text=(
                    "Photo context received; no owner consumption has been recorded. "
                    "[[ask: Did you eat the pictured portion? | Yes, all of it | No, none of it]]"
                ))
            ]), "end_turn")
            return
        if not attachments:
            yield self._emit(ConversationMessage(role="assistant", content=[
                TextBlock(text="No owned image source is available for a nutrition record.")
            ]), "end_turn")
            return
        selected = [
            block for block in attachments
            if self.candidate_id is not None
            and isinstance(block.source_provenance, Mapping)
            and block.source_provenance.get("camera_candidate_id") == self.candidate_id
        ]
        if len(selected) != 1:
            raise AssertionError("fixture must identify exactly one delivered owned source")
        current_results = [
            block for message in request.messages[current_user_index + 1:]
            for block in message.content if isinstance(block, ToolResultBlock)
        ]
        if not current_results:
            yield self._emit(ConversationMessage(role="assistant", content=[ToolUseBlock(
                name="load_conversation_image",
                input={"attachment_id": selected[0].attachment_id,
                       "select_as_nutrition_source": True},
            )]), "tool_use")
            return
        if len(current_results) == 1:
            if self.omit_nutrition_annotation:
                yield self._emit(ConversationMessage(role="assistant", content=[
                    TextBlock(text="I reviewed the selected image and your portion report.")
                ]), "end_turn")
                return
            nutrition = self._queued_correction
            self._queued_correction = None
            if nutrition is None:
                nutrition = {
                    "schema_version": 2, "record_type": "meal_observation",
                    "basis": ["image", "user_statement"], "consumption_status": "consumed",
                    "is_estimate": True,
                    "energy_kcal_best": 225 if self.context_items_date else SYNTHETIC_KCAL,
                    "items": ([
                        {"name": "cottage cheese", "quantity_text": "125 g package", "energy_kcal_best": 210},
                        {"name": "berries", "quantity_text": "15 g", "energy_kcal_best": 15},
                    ] if self.context_items_date else [
                        {"name": "synthetic apple", "quantity_text": "one fixture serving",
                         "energy_kcal_best": SYNTHETIC_KCAL},
                    ]),
                    "assumptions": ["offline structural fixture; not a model estimate"],
                }
            self.finalization_proposals += 1
            yield self._emit(ConversationMessage(role="assistant", content=[ToolUseBlock(
                name="trace",
                input={"kind": "trace_finalization", "payload": {
                    "schema_version": 1,
                    "trace_event_id": f"offline-ordinary-finalization-{self.calls}",
                    "annotations": {"nutrition": nutrition},
                }},
            )]), "tool_use")
            return
        yield self._emit(ConversationMessage(role="assistant", content=[
            TextBlock(text="Recorded the ordinary owner report after verifying its selected image source.")
        ]), "end_turn")


class OfflineCameraUserApi:
    """Deterministic LlmUserSimulator transport; never a real model call."""

    synthetic = True

    def __init__(self, selected_label: str = SYNTHETIC_BUTTON_LABEL) -> None:
        self.selected_label = selected_label
        self.calls = 0

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiMessageCompleteEvent]:
        del request
        self.calls += 1
        yield ApiMessageCompleteEvent(
            message=ConversationMessage.from_user_text(self.selected_label),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )


class OfflinePersonSourceApi:
    """Synthetic person-source fixture: one tool call, then a final response."""

    synthetic = True

    def __init__(self, *, outcome: str = "meal", basis: str = "image") -> None:
        if outcome not in {"meal", "nonfood"}:
            raise ValueError("offline person-source outcome must be meal or nonfood")
        self.outcome = outcome
        if basis not in {"image", "text"}:
            raise ValueError("offline person-source basis must be image or text")
        self.basis = basis
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiMessageCompleteEvent]:
        del request
        self.calls += 1
        if self.outcome == "nonfood":
            message = ConversationMessage(
                role="assistant",
                content=[TextBlock(text="This does not appear to be food, so I did not save a meal.")],
            )
            stop_reason = "end_turn"
        elif self.calls > 1:
            message = ConversationMessage(
                role="assistant",
                content=[TextBlock(text="I recorded the synthetic person-source meal.")],
            )
            stop_reason = "end_turn"
        else:
            message = ConversationMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        name="trace",
                        input={
                            "kind": "trace_finalization",
                            "payload": {
                                "schema_version": 1,
                                "trace_event_id": f"offline-person-source-{self.calls}",
                                "annotations": {
                                    "nutrition": {
                                        "schema_version": 2,
                                        "record_type": "meal_observation",
                                        "basis": [self.basis],
                                        "consumption_status": "consumed",
                                        "is_estimate": True,
                                        "energy_kcal_best": 125,
                                        "items": [{
                                            "name": "synthetic fixture food",
                                            "quantity_text": "one fixture serving",
                                            "energy_kcal_best": 125,
                                        }],
                                        "assumptions": ["synthetic fixture; not an estimate"],
                                    }
                                },
                            },
                        },
                    )
                ],
            )
            stop_reason = "tool_use"
        yield ApiMessageCompleteEvent(
            message=message,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason=stop_reason,
        )


class OfflineTelegramBot:
    """Telegram bot-shaped transport recorder; no polling or network methods."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.next_message_id = 77
        self.photo_bodies: list[bytes] = []

    def _receipt(self, chat_id: Any, *, photo: bool = False):
        from types import SimpleNamespace

        result = SimpleNamespace(
            message_id=self.next_message_id,
            chat_id=int(chat_id),
            photo=[object()] if photo else None,
        )
        self.next_message_id += 1
        return result

    async def send_photo(self, **kwargs):
        photo = kwargs.get("photo")
        if hasattr(photo, "read"):
            self.photo_bodies.append(photo.read())
        self.calls.append(("send_photo", kwargs))
        return self._receipt(kwargs["chat_id"], photo=True)

    async def send_message(self, **kwargs):
        self.calls.append(("send_message", kwargs))
        return self._receipt(kwargs["chat_id"])

    async def edit_message_caption(self, **kwargs):
        self.calls.append(("edit_message_caption", kwargs))
        return self._receipt(kwargs["chat_id"])

    async def edit_message_text(self, **kwargs):
        self.calls.append(("edit_message_text", kwargs))
        return self._receipt(kwargs["chat_id"])

    async def edit_message_reply_markup(self, **kwargs):
        self.calls.append(("edit_message_reply_markup", kwargs))
        return self._receipt(kwargs["chat_id"])

    async def delete_message(self, **kwargs):
        self.calls.append(("delete_message", kwargs))
        return True

    async def send_chat_action(self, **kwargs):
        self.calls.append(("send_chat_action", kwargs))


def distinct_offline_clients(*, context_items_date: bool = False,
                             omit_nutrition_annotation: bool = False):
    """Create distinct bot/user fake clients for the explicit offline mode."""
    return (
        OfflineCameraBotApi(context_items_date=context_items_date,
                            omit_nutrition_annotation=omit_nutrition_annotation),
        OfflineCameraUserApi(
            SYNTHETIC_WHOLE_PORTION if context_items_date else SYNTHETIC_BUTTON_LABEL
        ),
    )


def camera_runtime_limits(*, native_mode: bool) -> tuple[int, str]:
    """Keep model-backed prototype runs bounded without constraining offline fixtures."""
    return (8, "medium") if native_mode else (4, "none")


def isolated_runtime_loaders(runtime_module):
    """Fail closed for builder, prompt-skill and ambient-catalog loaders."""
    from contextlib import contextmanager
    from openharness.commands import registry as command_registry
    from openharness.hooks.loader import HookRegistry
    from openharness.plugins import loader as plugin_loader
    from openharness.skills import loader as skill_loader

    @contextmanager
    def guard():
        patches = (
            (runtime_module, "load_plugins", lambda *_args, **_kwargs: []),
            (runtime_module, "load_mcp_server_configs", lambda *_args, **_kwargs: {}),
            (runtime_module, "load_hook_registry", lambda *_args, **_kwargs: HookRegistry()),
            (plugin_loader, "load_plugins", lambda *_args, **_kwargs: []),
            (command_registry, "load_plugins", lambda *_args, **_kwargs: []),
            (skill_loader, "load_user_skills", lambda: []),
        )
        originals = tuple((module, name, getattr(module, name)) for module, name, _ in patches)
        try:
            for module, name, replacement in patches:
                setattr(module, name, replacement)
            yield
        finally:
            for module, name, original in reversed(originals):
                setattr(module, name, original)

    return guard()


def disable_external_runtime_surfaces(root, *, settings_path) -> None:
    """Write empty task-local hooks/plugins/MCP settings for offline runtime."""
    import json
    from openharness.config.settings import Settings

    defaults = Settings().model_dump(mode="json")
    defaults.update(
        active_profile="claude-api",
        hooks={},
        mcp_servers={},
        enabled_plugins={},
        allow_project_plugins=False,
        allow_project_skills=False,
        project_skill_dirs=[],
        memory={**defaults["memory"], "enabled": False, "auto_extract_enabled": False},
    )
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(defaults), encoding="utf-8")


async def run_camera_runtime_trajectory(
    *,
    root,
    bus,
    ingress,
    initial_message,
    channel,
    fake_bot,
    candidate_id: str,
    bot_client: SupportsStreamingMessages,
    user_client: SupportsStreamingMessages,
    honcho_url: str,
    workspace: str,
    session: str,
    before_answer=None,
    config_dir=None,
    user_scenario: str = "synthetic offline owner selects the exact offered confirmation",
    before_replay=None,
    advance_before_owner_action: bool = False,
    restart_before_replay: bool = False,
    expect_append_failure: bool = False,
    expect_missing_nutrition: bool = False,
    after_save=None,
):
    """Run an initial Camera delivery and a generic ordinary owner turn."""
    import os
    from datetime import datetime, timedelta, timezone

    from openharness.api.codex_client import CodexApiClient
    from openharness.config.paths import get_config_file_path
    from ohmo.gateway.bridge import OhmoGatewayBridge
    import ohmo.gateway.camera as camera_module
    from ohmo.gateway.camera import CameraIngress
    import ohmo.gateway.runtime as gateway_runtime
    from ohmo.gateway.config import save_gateway_config
    from ohmo.gateway.models import GatewayConfig
    from ohmo.gateway.runtime import OhmoSessionRuntimePool
    import openharness.ui.runtime as openharness_runtime
    from probe_support import NativeClientPreconditionError

    if bot_client is user_client:
        raise AssertionError("Camera bot and virtual-user clients must be separate")
    native_mode = os.environ.get("CAMERA_RUN_MODE") == "native"
    if native_mode and not (
        isinstance(bot_client, CodexApiClient) and isinstance(user_client, CodexApiClient)
    ):
        raise NativeClientPreconditionError(
            "native Camera requires separate CodexApiClient instances"
        )
    if not native_mode and not (
        getattr(bot_client, "synthetic", False) and getattr(user_client, "synthetic", False)
    ):
        raise AssertionError("offline Camera mode accepts only explicit synthetic transports")
    max_turns, effort = camera_runtime_limits(native_mode=native_mode)

    os.environ["OPENHARNESS_CONFIG_DIR"] = str(config_dir or root / "openharness-config")
    os.environ["OPENHARNESS_DATA_DIR"] = str(root / "openharness-data")
    os.environ["OPENHARNESS_LOGS_DIR"] = str(root / "openharness-logs")
    os.environ["OPENHARNESS_PROFILE"] = "codex" if native_mode else "claude-api"
    os.environ["OHMO_MEMORY_AUTOINDEX"] = "0"
    os.environ["OHMO_MEMORY_JUDGE"] = "0"
    settings_path = get_config_file_path()
    if not native_mode:
        disable_external_runtime_surfaces(root, settings_path=settings_path)

    save_gateway_config(
        GatewayConfig(
            enabled_channels=["telegram"],
            family_principals={"123": "synthetic_owner"},
            enabled_memory_tenants=("synthetic_owner",),
            conversation_learning=True,
            evals_capture=True,
            memory_backend="shadow",
            honcho_base_url=honcho_url,
            tenant_honcho={
                "synthetic_owner": {
                    "workspace": workspace,
                    "api_key": "local-auth-disabled",
                    "observed_peer": "owner",
                    "session": session,
                }
            },
            camera_ingress=ingress.config,
        ),
        root,
    )

    from openharness.ui.runtime import build_runtime as original_build_runtime

    async def injected_build_runtime(*args, **kwargs):
        kwargs.pop("api_client", None)
        kwargs["api_client"] = bot_client
        return await original_build_runtime(*args, **kwargs)

    from contextlib import ExitStack

    isolation_stack = ExitStack()
    if native_mode:
        isolation_stack.enter_context(isolated_runtime_loaders(openharness_runtime))
    saved_builder = gateway_runtime.build_runtime
    gateway_runtime.build_runtime = injected_build_runtime
    pool = None
    try:
        def new_pool(current_ingress):
            current_pool = OhmoSessionRuntimePool(
                cwd=root,
                workspace=root,
                provider_profile="codex" if native_mode else "claude-api",
                model="gpt-6-luna" if native_mode else "claude-sonnet-4-6",
                max_turns=max_turns,
                effort=effort,
            )
            current_pool._camera_ingress = current_ingress
            return current_pool

        def bind_ingress(current_ingress):
            channel._camera_ingress_authority = current_ingress
            channel._start_typing = lambda _chat_id: None
            channel._stop_typing = lambda _chat_id: None
            current_ingress._telegram = channel

        async def restart_runtime_and_camera(*, advance_minutes: int = 0):
            nonlocal pool, ingress, bridge
            previous_ingress = ingress
            if pool is not None:
                await pool.aclose()
            await previous_ingress.close()
            if advance_minutes:
                controlled_now[0] += timedelta(minutes=advance_minutes)
            ingress = CameraIngress(
                previous_ingress.config,
                workspace=root,
                bus=bus,
                telegram=channel,
            )
            bind_ingress(ingress)
            pool = new_pool(ingress)
            bridge = OhmoGatewayBridge(bus=bus, runtime_pool=pool, camera_ingress=ingress)
            if ingress._state_path != previous_ingress._state_path:
                raise AssertionError("Camera restart changed the journal path")
            return ingress, pool

        class ControlledCameraDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                now = controlled_now[0]
                return now.astimezone(tz) if tz is not None else now.replace(tzinfo=None)

        controlled_now = [datetime.now(timezone.utc)]
        controlled_clock_start = controlled_now[0]
        original_camera_datetime = camera_module.datetime
        camera_module.datetime = ControlledCameraDatetime
        pool = new_pool(ingress)
        bind_ingress(ingress)
        bridge = OhmoGatewayBridge(bus=bus, runtime_pool=pool, camera_ingress=ingress)
        async def process_and_deliver(message, session_key):
            await bridge._process_message(message, session_key)
            delivered = []
            final_receipt = None
            nutrition_event_id = None
            while bus.outbound_size:
                outbound = await bus.consume_outbound()
                receipt = await channel.send(outbound)
                ingress.note_assistant_receipt(outbound, receipt)
                delivered.append((outbound, receipt))
                if (
                    "nutrition_append_event_id" in outbound.metadata
                    and outbound.metadata["nutrition_append_event_id"] is not None
                ):
                    nutrition_event_id = outbound.metadata["nutrition_append_event_id"]
                    final_receipt = receipt
            return delivered, final_receipt, nutrition_event_id

        first_started = datetime.now(timezone.utc)
        first_delivered, _, _ = await process_and_deliver(initial_message, "telegram:123")
        if not any(name == "send_photo" for name, _ in fake_bot.calls):
            raise AssertionError("Camera photo delivery has no native receipt")
        native_photo = ingress._attempts[candidate_id].get("photo_id")
        capture_time = ingress._attempt_capture_time(ingress._attempts[candidate_id])
        if type(native_photo) is not int or native_photo <= 0 or capture_time is None:
            raise AssertionError("Camera source lacks a confirmed native photo/capture receipt")
        if (
            ingress._attempts[candidate_id].get("state") != "photo_sent"
            or ingress._attempts[candidate_id].get("initial_prompt_delivery_confirmed") is not True
        ):
            raise AssertionError("Camera initial question lacks a confirmed native prompt receipt")
        if before_answer is not None:
            await before_answer(first_started)
        if advance_before_owner_action:
            controlled_now[0] += timedelta(minutes=31)
            ingress._sweep_expired_attempts()

        from openharness.channels.bus.events import InboundMessage

        owner_text = "I ate about half of the pictured portion."
        if native_mode:
            from openharness.evals.session_user_simulator import LlmUserSimulator

            visible_prompt = next(
                (item.content for item, _ in reversed(first_delivered) if item.content), ""
            )
            if not visible_prompt:
                raise AssertionError("native Camera photo produced no visible prompt for the virtual owner")
            virtual_user = LlmUserSimulator(
                api_client=user_client, model="gpt-6-luna",
                system_prompt=(
                    "You are the owner in this Camera test. Use the supplied scenario as your facts. "
                    "Reply in your own words with what portion you actually ate and when. "
                    "Do not claim a meal you did not eat."
                ),
                max_tokens=256,
            )
            virtual_turn = await virtual_user.next_turn(
                transcript=(("assistant", visible_prompt),),
                captured_prompts=(user_scenario,), captured_capabilities=(),
                index=0, last_turn=None,
            )
            if virtual_turn is None or not virtual_turn.text.strip():
                raise AssertionError("native virtual owner did not produce an ordinary text turn")
            owner_text = virtual_turn.text.strip()
        answer = InboundMessage(
            channel="telegram", sender_id="123", chat_id="123",
            content=owner_text,
            timestamp=controlled_now[0],
            metadata={"message_id": "offline-ordinary-owner-turn-1", "is_group": False,
                      "chat_type": "private", "_telegram_raw_text":
                      owner_text},
        )
        if any(key in answer.metadata for key in (
            "reply_to_message_id", "callback_query", "_camera_authority", "_camera_answer",
        )):
            raise AssertionError("ordinary owner report carries synthetic Camera answer state")
        delivered, delivery_receipt, event_id = await process_and_deliver(
            answer, answer.session_key or "telegram:123"
        )
        final_text = next((item.content for item, _ in reversed(delivered) if item.content), "")
        if expect_append_failure:
            if event_id is not None or delivery_receipt is not None:
                raise AssertionError("failed Honcho append returned a saved event or receipt")
            if any(word in final_text.casefold() for word in ("saved", "recorded", "записано")):
                raise AssertionError("failed append returned a public Saved claim")
            return {
                "started": first_started, "answer": answer, "capture_time": capture_time,
                "event_id": None, "receipt": None, "owner_failure_text": final_text,
                "final_status_text": final_text, "native_photo_id": native_photo,
                "source_candidate_id": candidate_id, "ingress": ingress,
                "route": "ordinary", "camera_clock_advanced": controlled_now[0] - controlled_clock_start,
            }
        if expect_missing_nutrition:
            if event_id is not None or delivery_receipt is not None or not final_text:
                raise AssertionError("annotation-free owner turn unexpectedly saved nutrition")
            return {
                "started": first_started, "answer": answer, "capture_time": capture_time,
                "event_id": None, "receipt": None, "final_status_text": final_text,
                "native_photo_id": native_photo, "source_candidate_id": candidate_id,
                "ingress": ingress, "route": "ordinary",
                "camera_clock_advanced": controlled_now[0] - controlled_clock_start,
            }
        if not isinstance(event_id, str) or not event_id or delivery_receipt is None:
            raise AssertionError("ordinary owner turn did not return a durable append and delivery receipt")

        after_save_result = None
        if after_save is not None:
            after_save_result = await after_save(
                process_and_deliver, ingress, pool, candidate_id, answer,
                {"event_id": event_id}, restart_runtime_and_camera,
            )
        if before_replay is not None:
            await before_replay(process_and_deliver, ingress, pool)

        if restart_before_replay:
            await restart_runtime_and_camera()
        replay_delivered, replay_receipt, replay_event_id = await process_and_deliver(
            answer, answer.session_key or "telegram:123"
        )
        if replay_delivered or replay_receipt is not None or replay_event_id is not None:
            raise AssertionError("exact owner transport replay was not silent")
        if isinstance(bot_client, OfflineCameraBotApi):
            expected_proposals = 1 + (
                len(after_save_result["correction_event_ids"]) if after_save_result else 0
            )
            if bot_client.finalization_proposals != expected_proposals:
                raise AssertionError("replay reran a proposal or a new correction lacked one")
        print(
            f"{'NATIVE OPT-IN' if native_mode else 'OFFLINE SYNTHETIC'} ordinary owner turn "
            f"source={answer.metadata['message_id']} event={event_id} "
            f"capture={capture_time.isoformat()} replay_event={replay_event_id}",
            flush=True,
        )
        return {
            "started": first_started, "answer": answer, "capture_time": capture_time,
            "event_id": event_id, "receipt": {"event_id": event_id},
            "final_status_text": final_text, "native_photo_id": native_photo,
            "source_candidate_id": candidate_id, "ingress": ingress,
            "owner_replay_event_id": replay_event_id,
            "replay_text": next((item.content for item, _ in reversed(replay_delivered) if item.content), ""),
            "camera_clock_advanced": controlled_now[0] - controlled_clock_start,
            "after_save_result": after_save_result,
        }
    finally:
        try:
            if pool is not None:
                await pool.aclose()
        finally:
            try:
                camera_module.datetime = original_camera_datetime
                gateway_runtime.build_runtime = saved_builder
            finally:
                isolation_stack.close()
