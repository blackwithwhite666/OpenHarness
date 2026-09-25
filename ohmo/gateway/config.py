"""Config IO for ohmo gateway."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ohmo.gateway.models import GatewayConfig
from ohmo.workspace import get_gateway_config_path
from openharness.config.schema import Config


def load_gateway_config(workspace: str | Path | None = None) -> GatewayConfig:
    """Load ``.ohmo/gateway.json``."""
    path = get_gateway_config_path(workspace)
    if path.exists():
        raw = _normalize_progress_config(path.read_text(encoding="utf-8"))
        return GatewayConfig.model_validate(raw)
    return GatewayConfig()


def _normalize_progress_config(raw_text: str) -> dict[str, Any]:
    """Read progress settings with a fail-closed legacy migration.

    ``debug_progress_chats`` is the only authority.  The old compact fields
    are accepted only long enough to migrate ``verbose_progress_chats``; they
    never make an unlisted chat verbose.  Invalid or contradictory legacy
    values discard the legacy debug entries rather than risking a noisy
    default.
    """
    import json

    raw = json.loads(raw_text)
    if not isinstance(raw, dict):
        return raw

    def _chat_list(value: object) -> list[str] | None:
        if not isinstance(value, list):
            return None
        result: list[str] = []
        for item in value:
            if not isinstance(item, (str, int)) or isinstance(item, bool):
                return None
            chat_id = str(item).strip()
            if not chat_id:
                return None
            result.append(chat_id)
        return sorted(set(result))

    if "debug_progress_chats" in raw:
        canonical_ids = _chat_list(raw["debug_progress_chats"])
        raw["debug_progress_chats"] = canonical_ids or []
    else:
        verbose_ids = _chat_list(raw.get("verbose_progress_chats", []))
        compact_ids = _chat_list(raw.get("compact_progress_chats", []))
        legacy_default = raw.get("compact_progress_default", False)
        legacy_valid = (
            verbose_ids is not None
            and compact_ids is not None
            and isinstance(legacy_default, bool)
            and not set(verbose_ids) & set(compact_ids)
        )
        # Any malformed or contradictory legacy setting invalidates the whole
        # migration, rather than partially reviving verbose chat IDs.
        raw["debug_progress_chats"] = (
            verbose_ids if legacy_valid and verbose_ids is not None else []
        )

    for key in (
        "compact_progress_default",
        "compact_progress_chats",
        "verbose_progress_chats",
    ):
        raw.pop(key, None)
    return raw


def save_gateway_config(config: GatewayConfig, workspace: str | Path | None = None) -> Path:
    """Persist ``.ohmo/gateway.json``."""
    path = get_gateway_config_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def build_channel_manager_config(config: GatewayConfig) -> Config:
    """Project gateway settings into the channel compatibility models."""
    root = Config()
    root.channels.send_progress = config.send_progress
    root.channels.send_tool_hints = config.send_tool_hints
    for name in config.enabled_channels:
        if not hasattr(root.channels, name):
            continue
        channel_config = getattr(root.channels, name).model_copy(
            update={"enabled": True, **config.channel_configs.get(name, {})}
        )
        setattr(root.channels, name, channel_config)
    return root
