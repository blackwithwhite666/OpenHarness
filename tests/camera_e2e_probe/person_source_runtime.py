"""One ordinary inbound turn through the real Ohmo bridge/runtime."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.telegram import TelegramChannel
from openharness.config.schema import TelegramConfig

from camera_runtime_support import (
    OfflineTelegramBot,
    camera_runtime_limits,
    disable_external_runtime_surfaces,
    isolated_runtime_loaders,
)
from person_source_input import source_run_status


async def honcho_history_snapshot(client, *, session: str, since: datetime, until: datetime) -> list[dict]:
    """Use Honcho's bounded content-free full scoped message query."""
    messages = await client.list_recent_message_metadata(
        session,
        expected_peer_id="ohmo",
        since=since,
        until=until,
        page_size=100,
        max_pages=100,
    )
    return [
        {
            "id": item.id,
            "peer_id": item.peer_id,
            "session_id": item.session_id,
            "created_at": item.created_at.isoformat(),
            "metadata": dict(item.metadata),
        }
        for item in messages
    ]


def runtime_episode_evidence(eval_store, episode_ids: list[str]) -> list[dict]:
    """Export actual recorder episodes and their full recorded events."""
    result = []
    for episode_id in episode_ids:
        episode = eval_store.get_episode(episode_id)
        events = list(eval_store.iter_events(episode_id))
        result.append({
            "episode_id": episode_id,
            "episode": episode.model_dump(mode="json") if episode is not None else None,
            "events": [event.model_dump(mode="json") for event in events],
            "missing": episode is None,
        })
    return result


def new_nutrition_event_ids(before: list[dict], after: list[dict]) -> list[str]:
    """Identify newly observed nutrition rows without dropping other message rows."""
    prior = {str(row.get("id")) for row in before}
    result = []
    for row in after:
        if str(row.get("id")) in prior:
            continue
        trace = row.get("metadata", {}).get("decision_trace")
        annotations = trace.get("annotations") if isinstance(trace, dict) else None
        nutrition = annotations.get("nutrition") if isinstance(annotations, dict) else None
        if isinstance(nutrition, dict):
            result.append(str(row.get("id")))
    return result


def projected_intake_kcal(records: list[dict]) -> int | float:
    """Sum returned nutrition intake rows; this does not subtract expenditure."""
    return sum(record.get("energy_kcal_best") or 0 for record in records)


async def run_person_source_turn(
    *, root: Path, message, owner_id: str, honcho_url: str, workspace: str,
    session: str, bot_client, native_mode: bool, config_dir: Path | None = None,
    before_turn=None,
) -> dict:
    """Run one person message, deliver real bridge output, and capture eval episode IDs."""
    from openharness.api.codex_client import CodexApiClient
    from openharness.config.paths import get_config_file_path
    from ohmo.gateway.bridge import OhmoGatewayBridge
    from ohmo.gateway.config import save_gateway_config
    from ohmo.gateway.models import GatewayConfig
    from ohmo.gateway.runtime import OhmoSessionRuntimePool
    from ohmo.evals.adapter import get_eval_store
    import ohmo.gateway.runtime as gateway_runtime
    import openharness.ui.runtime as openharness_runtime

    if native_mode and not isinstance(bot_client, CodexApiClient):
        raise ValueError("native person-source run requires the Codex subscription client")
    if not native_mode and not getattr(bot_client, "synthetic", False):
        raise ValueError("offline person-source run accepts only its deterministic fixture client")
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
            family_principals={str(message.sender_id): owner_id},
            enabled_memory_tenants=(owner_id,),
            conversation_learning=True,
            evals_capture=True,
            memory_backend="shadow",
            honcho_base_url=honcho_url,
            tenant_honcho={owner_id: {
                "workspace": workspace,
                "api_key": "local-auth-disabled",
                "observed_peer": "owner",
                "session": session,
            }},
        ),
        root,
    )
    bus = MessageBus()
    channel = TelegramChannel(TelegramConfig(token="offline-no-network", allow_from=[str(message.sender_id)]), bus)
    fake_bot = OfflineTelegramBot()
    from types import SimpleNamespace

    channel._app = SimpleNamespace(bot=fake_bot)
    channel.polling_started = True
    channel._start_typing = lambda _chat: None
    channel._stop_typing = lambda _chat: None

    from openharness.ui.runtime import build_runtime as original_build_runtime

    async def injected_build_runtime(*args, **kwargs):
        kwargs.pop("api_client", None)
        kwargs["api_client"] = bot_client
        return await original_build_runtime(*args, **kwargs)

    from contextlib import ExitStack

    isolation = ExitStack()
    if native_mode:
        isolation.enter_context(isolated_runtime_loaders(openharness_runtime))
    saved_builder = gateway_runtime.build_runtime
    gateway_runtime.build_runtime = injected_build_runtime
    pool = None
    started = datetime.now(timezone.utc)
    try:
        pool = OhmoSessionRuntimePool(
            cwd=root,
            workspace=root,
            provider_profile="codex" if native_mode else "claude-api",
            model="gpt-6-luna" if native_mode else "claude-sonnet-4-6",
            max_turns=max_turns,
            effort=effort,
        )
        bridge = OhmoGatewayBridge(bus=bus, runtime_pool=pool)
        eval_store = get_eval_store(root)
        episodes_before = set(eval_store.list_episode_ids())
        await pool.get_bundle(message.session_key, latest_user_prompt=message.content)
        before_until = datetime.now(timezone.utc)
        before_evidence = await before_turn(before_until) if before_turn is not None else None
        await bridge._process_message(message, message.session_key)
        delivered: list[dict[str, object]] = []
        while bus.outbound_size:
            outbound = await bus.consume_outbound()
            receipt = await channel.send(outbound)
            event_id = outbound.metadata.get("nutrition_append_event_id")
            delivered.append({
                "content": outbound.content,
                "metadata": dict(outbound.metadata),
                "delivery_message_ids": list(getattr(receipt, "native_message_ids", ())),
                "event_id": event_id if isinstance(event_id, str) and event_id else None,
            })
        episode_ids = sorted(set(eval_store.list_episode_ids()) - episodes_before)
        return {
            "started_at": started.isoformat(),
            "before_until": before_until.isoformat(),
            "before_evidence": before_evidence,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "session_key": message.session_key,
            "source_message_id": message.metadata.get("message_id"),
            "deliveries": delivered,
            "save": source_run_status(delivered),
            "episode_ids": episode_ids,
            "runtime_episodes": runtime_episode_evidence(eval_store, episode_ids),
            "runtime_mode": "native" if native_mode else "synthetic_fixture",
        }
    finally:
        try:
            if pool is not None:
                await pool.aclose()
        finally:
            gateway_runtime.build_runtime = saved_builder
            isolation.close()
