"""Existing OpenHarness client seams for the isolated Camera prototype."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    SupportsStreamingMessages,
)
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import (
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
) -> str:
    """Require a delivered saved-state response tied to the immutable event."""
    if (
        replay.metadata.get("_camera_existing_meal_replay") is not True
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
    """Synthetic bot transport: asks, then uses TraceTool for a fake meal."""

    synthetic = True

    def __init__(self) -> None:
        self.calls = 0
        self.finalization_proposals = 0

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiMessageCompleteEvent]:
        self.calls += 1
        owner_turn_index = next(
            (
                index
                for index in range(len(request.messages) - 1, -1, -1)
                if request.messages[index].role == "user"
                and not any(
                    isinstance(block, ToolResultBlock) for block in request.messages[index].content
                )
            ),
            len(request.messages),
        )
        owner_turn = (
            request.messages[owner_turn_index] if owner_turn_index < len(request.messages) else None
        )
        # Query appends tool results as role=user. Results after this owner turn
        # finish its turn; results before a fresh replay do not suppress it.
        tool_result_seen = any(
            isinstance(block, ToolResultBlock)
            for message in request.messages[owner_turn_index + 1 :]
            for block in message.content
        )
        owner_confirmed = owner_turn is not None and SYNTHETIC_BUTTON_LABEL in owner_turn.text
        if tool_result_seen:
            message = ConversationMessage(
                role="assistant",
                content=[
                    TextBlock(
                        text=(
                            "I recorded the synthetic Camera meal from the delivered photo "
                            "and your explicit confirmation; the stored entry uses the "
                            "trusted capture date."
                        )
                    )
                ],
            )
        elif owner_confirmed:
            self.finalization_proposals += 1
            message = ConversationMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        name="trace",
                        input={
                            "kind": "trace_finalization",
                            "payload": {
                                "schema_version": 1,
                                "trace_event_id": f"offline-camera-finalization-{self.calls}",
                                "annotations": {
                                    "nutrition": {
                                        "schema_version": 2,
                                        "record_type": "meal_observation",
                                        "basis": ["image", "owner_confirmation"],
                                        "consumption_status": "consumed",
                                        "is_estimate": True,
                                        "energy_kcal_best": SYNTHETIC_KCAL,
                                        "items": [
                                            {
                                                "name": "synthetic apple",
                                                "quantity_text": "1 medium apple (fixture)",
                                                "energy_kcal_best": SYNTHETIC_KCAL,
                                            }
                                        ],
                                        "assumptions": [
                                            "offline transport fixture; not a model estimate"
                                        ],
                                    }
                                },
                            },
                        },
                    )
                ],
            )
        else:
            message = ConversationMessage(
                role="assistant",
                content=[
                    TextBlock(
                        text=(
                            "Synthetic offline Camera photo received. "
                            "[[ask: Did you eat the pictured synthetic item? | "
                            + " | ".join(SYNTHETIC_OPTIONS)
                            + "]]"
                        )
                    )
                ],
            )
        yield ApiMessageCompleteEvent(
            message=message,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="tool_use" if owner_confirmed and not tool_result_seen else "end_turn",
        )


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


def distinct_offline_clients():
    """Create distinct bot/user fake clients for the explicit offline mode."""
    return OfflineCameraBotApi(), OfflineCameraUserApi()


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
):
    """Run two actual Ohmo turns and the delivered Telegram callback offline."""
    import asyncio
    import json
    import os
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from openharness.api.codex_client import CodexApiClient
    from openharness.config.paths import get_config_file_path
    from openharness.evals.session_user_simulator import LlmUserSimulator
    from ohmo.gateway.bridge import OhmoGatewayBridge
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
        pool = OhmoSessionRuntimePool(
            cwd=root,
            workspace=root,
            provider_profile="codex" if native_mode else "claude-api",
            model="gpt-6-luna" if native_mode else "claude-sonnet-4-6",
            max_turns=max_turns,
            effort=effort,
        )
        pool._camera_ingress = ingress
        channel._camera_ingress_authority = ingress
        channel._start_typing = lambda _chat_id: None
        channel._stop_typing = lambda _chat_id: None
        ingress._telegram = channel
        bridge = OhmoGatewayBridge(bus=bus, runtime_pool=pool, camera_ingress=ingress)
        simulator = LlmUserSimulator(
            api_client=user_client,
            model="gpt-6-luna" if native_mode else "offline-user-synthetic",
            system_prompt=(
                "You are the Camera owner described here. Answer as that person, using the "
                "actual question and choices in the conversation. Preserve denial, amount, "
                "and time qualifiers. Do not invent details.\n\n" + user_scenario
                if native_mode
                else "You are a synthetic offline owner; select the exact offered answer."
            ),
        )
        from camera_virtual_user import CameraVirtualUser, OfferedCameraChoices
        from probe_support import require_bound_answer

        virtual_user = CameraVirtualUser(simulator)

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
                if outbound.metadata.get("nutrition_append_event_id"):
                    nutrition_event_id = outbound.metadata["nutrition_append_event_id"]
                    final_receipt = receipt
            return delivered, final_receipt, nutrition_event_id

        first_started = datetime.now(timezone.utc)
        first_delivered, _, _ = await process_and_deliver(initial_message, "telegram:123")
        issued = next((item for item, _ in reversed(first_delivered) if item.buttons), None)
        if issued is None:
            raise AssertionError("runtime/bridge did not issue a Camera choice keyboard")
        edit = next(
            (
                kwargs
                for name, kwargs in fake_bot.calls
                if name == "edit_message_caption" and kwargs.get("reply_markup") is not None
            ),
            None,
        )
        if edit is None:
            raise AssertionError("TelegramChannel did not deliver the Camera keyboard")
        markup = edit["reply_markup"]
        buttons = [button for row in markup.inline_keyboard for button in row]
        native_photo = ingress._attempts[candidate_id].get("photo_id")
        if not isinstance(native_photo, int) or native_photo <= 0:
            raise AssertionError("Camera native photo receipt is missing")
        if not any(
            name == "send_photo" and not kwargs.get("reply_markup")
            for name, kwargs in fake_bot.calls
        ):
            raise AssertionError("initial Camera photo must have no buttons")
        question = issued.content.rsplit("\n\n", 1)[-1].strip()
        offered = OfferedCameraChoices(
            question=question,
            options=tuple(button.text for button in buttons),
            callback_ids=tuple(button.callback_data for button in buttons),
            native_message_id=str(native_photo),
            media_source_ids=(candidate_id,),
        )
        if before_answer is not None:
            await before_answer(first_started)
        action = await virtual_user.next_camera_action(
            offered=offered,
            transcript=(("assistant", issued.content),),
            captured_prompts=(user_scenario,),
            captured_capabilities=(),
            index=0,
            last_turn=None,
        )
        if action is None:
            raise AssertionError("Camera virtual user produced no owner action")
        action_observation = root / "camera-virtual-action.json"
        with action_observation.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "question": offered.question,
                    "offered_labels": list(offered.options),
                    "action_text": action.text,
                    "callback_id": action.callback_data,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
        os.chmod(action_observation, 0o600)
        button_ids = {button.callback_data for button in buttons}
        if action.callback_data is not None and action.callback_data not in button_ids:
            raise AssertionError("virtual user selected a callback absent from delivered markup")

        clicked_message = SimpleNamespace(
            message_id=native_photo,
            chat_id=123,
            chat=SimpleNamespace(type="private"),
            caption=edit.get("caption"),
            caption_html=None,
            text=None,
            text_html=None,
            reply_markup=markup,
        )

        callback_number = 0

        async def invoke_issued_callback():
            nonlocal callback_number
            callback_number += 1

            class Query:
                data = action.callback_data
                id = f"offline-camera-callback-{callback_number}"
                message = clicked_message

                async def answer(self):
                    return None

                async def edit_message_caption(self, **kwargs):
                    fake_bot.calls.append(("callback_edit_caption", kwargs))

                async def edit_message_text(self, **kwargs):
                    fake_bot.calls.append(("callback_edit_text", kwargs))

                async def edit_message_reply_markup(self, **kwargs):
                    fake_bot.calls.append(("callback_edit_markup", kwargs))

            await channel._on_callback(
                SimpleNamespace(
                    callback_query=Query(),
                    effective_user=SimpleNamespace(
                        id=123, username=None, first_name="offline owner"
                    ),
                ),
                None,
            )
            return await asyncio.wait_for(bus.consume_inbound(), timeout=2)

        if action.callback_data is not None:
            answer = await invoke_issued_callback()
        else:
            answer = action.to_inbound_message(
                sender_id="123",
                chat_id="123",
                source_message_id="offline-camera-typed-1",
                received_at=datetime.now(timezone.utc),
            )
        ingress.process_real_inbound(answer)
        require_bound_answer(answer, candidate_id, native_photo)
        if (
            (
                action.callback_data is not None
                and answer.metadata.get("native_message_id") != native_photo
            )
            or (
                action.callback_data is None
                and str(answer.metadata.get("reply_to_message_id")) != str(native_photo)
            )
            or (
                action.callback_data is not None
                and answer.metadata.get("native_keyboard_options")
                != [button.text for button in buttons]
            )
            or (
                action.callback_data is not None
                and answer.metadata.get("native_keyboard_selected_label") != action.text
            )
            or (
                action.callback_data is not None
                and answer.metadata.get("message_id") != native_photo
            )
            or (
                action.callback_data is not None
                and answer.metadata.get("callback_query_id") != "offline-camera-callback-1"
            )
            or (
                action.callback_data is None
                and answer.metadata.get("message_id") != "offline-camera-typed-1"
            )
            or not answer.media
        ):
            raise AssertionError("actual Telegram callback did not bind the offered Camera source")
        capture_time = ingress.trusted_capture_time_for_answer(answer)
        if capture_time is None:
            raise AssertionError("Camera callback has no trusted capture time")
        _, final_receipt, nutrition_event_id = await process_and_deliver(answer, answer.session_key)
        if not isinstance(nutrition_event_id, str) or not nutrition_event_id:
            raise AssertionError("runtime did not return its durable nutrition event ID")
        commit = ingress._attempts[candidate_id].get("camera_commit")
        if not isinstance(commit, dict) or commit.get("event_id") != nutrition_event_id:
            raise AssertionError("runtime nutrition event ID differs from Camera durable receipt")
        if final_receipt is None:
            raise AssertionError("owner turn did not deliver a native final receipt")
        action_label = "Telegram callback" if action.callback_data is not None else "typed reply"
        mode_label = "NATIVE OPT-IN" if native_mode else "OFFLINE SYNTHETIC"
        print(
            f"{mode_label} virtual-user {action_label} + real Ohmo finalizer completed; "
            f"event={nutrition_event_id} capture_date={capture_time.date().isoformat()} "
            f"bot_calls={getattr(bot_client, 'calls', 'native')} "
            f"user_calls={getattr(user_client, 'calls', 'native')}",
            flush=True,
        )

        async def replay_callback():
            if action.callback_data is not None:
                replay = await invoke_issued_callback()
            else:
                replay = action.to_inbound_message(
                    sender_id="123",
                    chat_id="123",
                    source_message_id="offline-camera-typed-replay-1",
                    received_at=datetime.now(timezone.utc),
                )
            ingress.process_real_inbound(replay)
            return replay

        replay = await replay_callback()
        replay_delivered, replay_delivery_receipt, replay_event_id = await process_and_deliver(
            replay, replay.session_key
        )
        current_commit = ingress._attempts[candidate_id].get("camera_commit")
        replay_status = _validate_completed_photo_replay(
            replay=replay,
            candidate_id=candidate_id,
            turn_id=answer.metadata["_camera_turn_id"],
            delivered=replay_delivered,
            delivery_receipt=replay_delivery_receipt,
            existing_event_id=nutrition_event_id,
            original_commit=commit,
            current_commit=current_commit,
        )
        if replay_event_id != nutrition_event_id:
            raise AssertionError("completed-photo replay event identity differs from existing meal")
        if isinstance(bot_client, OfflineCameraBotApi) and bot_client.finalization_proposals != 1:
            raise AssertionError("completed-photo replay proposed another nutrition observation")

        return {
            "started": first_started,
            "answer": answer,
            "capture_time": capture_time,
            "event_id": nutrition_event_id,
            "receipt": commit,
            "native_photo_id": native_photo,
            "keyboard_message_id": native_photo,
            "source_candidate_id": candidate_id,
            "bot": channel,
            "bot_transport": fake_bot,
            "action": action,
            "markup": markup,
            "caption": edit.get("caption"),
            "replay_callback": replay_callback,
            "owner_replay_saved_status": replay_status,
            "owner_replay_delivery_confirmed": replay_delivery_receipt is not None,
            "owner_replay_event_id": replay_event_id,
        }
    finally:
        try:
            if pool is not None:
                await pool.aclose()
        finally:
            try:
                gateway_runtime.build_runtime = saved_builder
            finally:
                isolation_stack.close()
