"""Gateway models for ohmo."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


_TENANT_ID_RE = re.compile(r"[a-z0-9_-]+")
_NUMERIC_PRINCIPAL_RE = re.compile(r"[0-9]+")


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
    memory_backend: str = "file"
    semantic_search: bool = False
    inference_url: str | None = None
    embedding_model: str | None = None
    visible_recall: bool = False
    conversation_learning: bool = False
    memory_service_socket: str | None = None
    memory_service_secret_file: str | None = None
    owner_principals: tuple[str, ...] = ()
    family_principals: dict[str, str] = Field(default_factory=dict)
    shared_tenants: tuple[str, ...] = ()
    enabled_memory_tenants: tuple[str, ...] = ()
    honcho_base_url: str | None = None
    honcho_api_key: str | None = None
    honcho_workspace: str | None = None
    tenant_honcho: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_memory_tenant_config(self) -> GatewayConfig:
        """Validate canonical principals and memory tenant identifiers."""
        for principal, tenant_id in self.family_principals.items():
            if _NUMERIC_PRINCIPAL_RE.fullmatch(principal) is None:
                raise ValueError("family_principals keys must be canonical numeric ids")
            if _TENANT_ID_RE.fullmatch(tenant_id) is None:
                raise ValueError("memory tenant ids must match [a-z0-9_-]+")
            if tenant_id == "owner":
                raise ValueError("the owner tenant is reserved for owner_principals")

        for tenant_id in (*self.shared_tenants, *self.enabled_memory_tenants):
            if _TENANT_ID_RE.fullmatch(tenant_id) is None:
                raise ValueError("memory tenant ids must match [a-z0-9_-]+")

        for tenant_id, binding in self.tenant_honcho.items():
            if _TENANT_ID_RE.fullmatch(tenant_id) is None:
                raise ValueError("tenant_honcho keys must match [a-z0-9_-]+")
            for field_name in ("workspace", "api_key"):
                value = binding.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"tenant_honcho[{tenant_id!r}].{field_name} is required"
                    )
            observed_peer = binding.get("observed_peer", tenant_id)
            if not isinstance(observed_peer, str) or not observed_peer.strip():
                raise ValueError(
                    f"tenant_honcho[{tenant_id!r}].observed_peer must not be empty"
                )

        if "owner" in self.shared_tenants:
            raise ValueError("the owner tenant cannot be a shared tenant")

        owner_numeric_ids = {
            canonical
            for owner in self.owner_principals
            if (canonical := str(owner).strip().split("|", 1)[0].strip())
            and _NUMERIC_PRINCIPAL_RE.fullmatch(canonical) is not None
        }
        overlap = owner_numeric_ids.intersection(self.family_principals)
        if overlap:
            raise ValueError("numeric principals cannot map to both owner and family tenants")

        return self


class GatewayState(BaseModel):
    """Runtime gateway status snapshot."""

    running: bool = False
    pid: int | None = None
    active_sessions: int = 0
    provider_profile: str = "codex"
    enabled_channels: list[str] = Field(default_factory=list)
    last_error: str | None = None
