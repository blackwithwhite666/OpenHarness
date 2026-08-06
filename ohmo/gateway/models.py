"""Gateway models for ohmo."""

from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_TENANT_ID_RE = re.compile(r"[a-z0-9_-]+")
_NUMERIC_PRINCIPAL_RE = re.compile(r"[1-9][0-9]*")
_HONCHO_SESSION_RE = re.compile(r"[A-Za-z0-9_-]{1,512}")


class GatewayConfig(BaseModel):
    """Persistent gateway configuration."""

    provider_profile: str = "codex"
    enabled_channels: list[str] = Field(default_factory=list)
    session_routing: str = "chat-thread"
    send_progress: bool = True
    send_tool_hints: bool = True
    compact_progress_default: bool = False
    compact_progress_chats: list[str] = Field(default_factory=list)
    verbose_progress_chats: list[str] = Field(default_factory=list)
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
    nutrition_ingest: "NutritionIngestConfig" = Field(default_factory=lambda: NutritionIngestConfig())

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
            session = binding.get("session", "ohmo")
            if not isinstance(session, str) or _HONCHO_SESSION_RE.fullmatch(session) is None:
                raise ValueError(
                    f"tenant_honcho[{tenant_id!r}].session must be a bounded Honcho identifier"
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

        self.nutrition_ingest.validate_runtime(self)
        return self


class NutritionIngestConfig(BaseModel):
    """Fail-closed, Marina-only Dropbox confirmation configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = False
    synchronized_root: Path | None = None
    principal: str = ""
    chat_id: str = ""
    session_key: str = ""
    tenant_id: Literal["marina"] = "marina"
    locale: Literal["ru"] = "ru"
    poll_interval_seconds: float = Field(default=10.0, gt=0, le=3600)
    max_prompt_attempts: int = Field(default=3, ge=1, le=20)
    max_estimation_attempts: int = Field(default=5, ge=1, le=20)
    retry_backoff_seconds: float = Field(default=5.0, ge=0, le=3600)
    require_owner_only_filesystem: bool = True

    @property
    def canonical_principal(self) -> str:
        return self.principal

    @property
    def private_chat_id(self) -> str:
        return self.chat_id

    @property
    def exact_session_key(self) -> str:
        return self.session_key

    @property
    def root(self) -> Path | None:
        return self.synchronized_root

    @model_validator(mode="after")
    def validate_enabled_binding(self) -> "NutritionIngestConfig":
        if not self.enabled:
            return self
        if not re.fullmatch(r"[1-9][0-9]*", self.principal):
            raise ValueError("nutrition_ingest principal must be canonical numeric")
        if not re.fullmatch(r"[1-9][0-9]*", self.chat_id):
            raise ValueError("nutrition_ingest private chat id must be a positive numeric id")
        if self.chat_id != self.principal:
            raise ValueError("nutrition_ingest chat_id must equal the principal")
        if self.session_key != f"telegram:{self.principal}":
            raise ValueError("nutrition_ingest session_key must match the private Telegram chat")
        if self.synchronized_root is None:
            raise ValueError("nutrition_ingest synchronized_root is required")
        return self

    def validate_runtime(self, gateway: GatewayConfig) -> None:
        """Validate cross-config and filesystem invariants at service startup."""
        if not self.enabled:
            return
        if gateway.conversation_learning is not True:
            raise ValueError("nutrition ingest requires conversation learning")
        if gateway.family_principals.get(self.principal) != "marina":
            raise ValueError("nutrition ingest principal is not bound to marina")
        if self.tenant_id not in gateway.enabled_memory_tenants:
            raise ValueError("nutrition ingest Marina tenant is not enabled")
        if not isinstance(gateway.honcho_base_url, str) or not gateway.honcho_base_url.strip():
            raise ValueError("nutrition ingest requires the Honcho base URL")
        binding = gateway.tenant_honcho.get("marina")
        if not binding or not all(
            isinstance(binding.get(key), str) and binding.get(key, "").strip()
            for key in ("workspace", "api_key", "observed_peer")
        ):
            raise ValueError("tenant_honcho marina binding is incomplete")
        root = self.synchronized_root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError("nutrition ingest synchronized_root must be a directory")
        if self.require_owner_only_filesystem:
            paths = [root, *root.rglob("*")]
            for path in paths:
                try:
                    mode = path.stat().st_mode
                except OSError as exc:
                    raise ValueError("nutrition ingest filesystem is not readable") from exc
                if stat.S_IMODE(mode) & 0o077:
                    raise ValueError("nutrition ingest filesystem must be owner-only")


class GatewayState(BaseModel):
    """Runtime gateway status snapshot."""

    running: bool = False
    pid: int | None = None
    active_sessions: int = 0
    provider_profile: str = "codex"
    enabled_channels: list[str] = Field(default_factory=list)
    last_error: str | None = None
