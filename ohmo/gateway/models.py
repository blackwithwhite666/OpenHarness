"""Gateway models for ohmo."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GatewayConfig(BaseModel):
    """Persistent gateway configuration."""

    provider_profile: str = "codex"
    enabled_channels: list[str] = Field(default_factory=list)
    session_routing: str = "chat-thread"
    send_progress: bool = True
    send_tool_hints: bool = True
    permission_mode: str = "default"
    sandbox_enabled: bool = False
    allow_remote_admin_commands: bool = False
    allowed_remote_admin_commands: list[str] = Field(default_factory=list)
    log_level: str = "INFO"
    channel_configs: dict[str, dict] = Field(default_factory=dict)
    default_tz: str = "Europe/Moscow"
    reminder_max_per_chat: int = 50
    reminder_catchup: Literal["once", "none"] = "once"
    message_coalesce_window: float = 0.8
    message_coalesce_media_window: float = 3.0
    message_coalesce_max: int = 20
    evals_capture: bool = True


class GatewayState(BaseModel):
    """Runtime gateway status snapshot."""

    running: bool = False
    pid: int | None = None
    active_sessions: int = 0
    provider_profile: str = "codex"
    enabled_channels: list[str] = Field(default_factory=list)
    last_error: str | None = None
