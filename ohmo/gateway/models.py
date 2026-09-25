"""Gateway models for ohmo."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_TENANT_ID_RE = re.compile(r"[a-z0-9_-]+")
_NUMERIC_PRINCIPAL_RE = re.compile(r"[1-9][0-9]*")
_PROFILE_PRINCIPAL_RE = re.compile(r"[1-9][0-9]{0,19}")
_PROJECT_SLUG_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MAX_PROFILE_PROJECTS_PER_PRINCIPAL = 8
_MAX_PROFILE_PRINCIPALS = 100
_MAX_PROJECT_SLUG_LENGTH = 64
_HONCHO_SESSION_RE = re.compile(r"[A-Za-z0-9_-]{1,512}")


class GatewayConfig(BaseModel):
    """Persistent gateway configuration."""

    provider_profile: str = "codex"
    enabled_channels: list[str] = Field(default_factory=list)
    session_routing: str = "chat-thread"
    send_progress: bool = True
    send_tool_hints: bool = True
    debug_progress_chats: list[str] = Field(default_factory=list)
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
    knowledge_base_root: Path | None = None
    principal_knowledge_projects: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    shared_tenants: tuple[str, ...] = ()
    enabled_memory_tenants: tuple[str, ...] = ()
    honcho_base_url: str | None = None
    honcho_api_key: str | None = None
    honcho_workspace: str | None = None
    tenant_honcho: dict[str, dict[str, str]] = Field(default_factory=dict)
    camera_ingress: "CameraIngressConfig" = Field(default_factory=lambda: CameraIngressConfig())

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

        if len(self.principal_knowledge_projects) > _MAX_PROFILE_PRINCIPALS:
            raise ValueError("principal_knowledge_projects has too many principals")
        if self.principal_knowledge_projects and self.knowledge_base_root is None:
            raise ValueError("knowledge_base_root is required for principal knowledge projects")
        if self.knowledge_base_root is not None and not self.knowledge_base_root.is_absolute():
            raise ValueError("knowledge_base_root must be an absolute path")
        for principal, project_slugs in self.principal_knowledge_projects.items():
            if _PROFILE_PRINCIPAL_RE.fullmatch(principal) is None:
                raise ValueError(
                    "principal_knowledge_projects keys must be bounded canonical numeric ids"
                )
            if not project_slugs or len(project_slugs) > _MAX_PROFILE_PROJECTS_PER_PRINCIPAL:
                raise ValueError("each principal must have a bounded non-empty project list")
            if len(set(project_slugs)) != len(project_slugs):
                raise ValueError("principal knowledge project slugs must not repeat")
            for slug in project_slugs:
                if len(slug) > _MAX_PROJECT_SLUG_LENGTH or _PROJECT_SLUG_RE.fullmatch(slug) is None:
                    raise ValueError("knowledge project slugs must be lowercase kebab-case")

        for tenant_id in (*self.shared_tenants, *self.enabled_memory_tenants):
            if _TENANT_ID_RE.fullmatch(tenant_id) is None:
                raise ValueError("memory tenant ids must match [a-z0-9_-]+")

        for tenant_id, binding in self.tenant_honcho.items():
            if _TENANT_ID_RE.fullmatch(tenant_id) is None:
                raise ValueError("tenant_honcho keys must match [a-z0-9_-]+")
            for field_name in ("workspace", "api_key"):
                value = binding.get(field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"tenant_honcho[{tenant_id!r}].{field_name} is required")
            observed_peer = binding.get("observed_peer", tenant_id)
            if not isinstance(observed_peer, str) or not observed_peer.strip():
                raise ValueError(f"tenant_honcho[{tenant_id!r}].observed_peer must not be empty")
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

        self.camera_ingress.validate_runtime(self)
        return self


class CameraIngressConfig(BaseModel):
    """Server-owned Camera ingress binding."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    enabled: bool = False
    listen_host: Literal["127.0.0.1", "::1", "10.8.0.8"] = "127.0.0.1"
    listen_port: int = Field(default=0, ge=0, le=65535)
    bearer_token_file: Path | None = None
    synchronized_root: Path | None = None
    principal: str = ""
    tenant_id: str = ""
    chat_id: str = ""
    session_key: str = ""

    def validate_runtime(self, gateway: GatewayConfig) -> None:
        if not self.enabled:
            return
        if self.listen_port == 0:
            raise ValueError("camera ingress requires a private listener port")
        if self.bearer_token_file is None or not self.bearer_token_file.is_absolute():
            raise ValueError("camera ingress requires an absolute bearer token file")
        if self.synchronized_root is None or not self.synchronized_root.is_absolute():
            raise ValueError("camera ingress requires an absolute synchronized root")
        if not _NUMERIC_PRINCIPAL_RE.fullmatch(self.principal):
            raise ValueError("camera ingress principal must be canonical numeric")
        if (
            self.chat_id != self.principal
            or self.session_key != f"telegram:{self.principal}"
            or not self.tenant_id
        ):
            raise ValueError("camera ingress must use its configured private Telegram binding")
        if gateway.family_principals.get(self.principal) != self.tenant_id:
            raise ValueError("camera ingress principal and tenant binding disagree")
        if (
            self.tenant_id not in gateway.enabled_memory_tenants
            or not gateway.conversation_learning
        ):
            raise ValueError("camera ingress requires conversation learning for its tenant")
        if not gateway.evals_capture:
            raise ValueError("camera ingress requires nutrition finalization validation")
        if gateway.memory_backend != "shadow" or not gateway.honcho_base_url:
            raise ValueError("camera ingress requires Honcho shadow memory")
        binding = gateway.tenant_honcho.get(self.tenant_id)
        if not binding or not all(
            isinstance(binding.get(key), str) and binding.get(key, "").strip()
            for key in ("workspace", "api_key", "observed_peer")
        ):
            raise ValueError("camera ingress requires a complete tenant Honcho binding")
        if "telegram" not in gateway.enabled_channels:
            raise ValueError("camera ingress requires Telegram")


class GatewayState(BaseModel):
    """Runtime gateway status snapshot."""

    running: bool = False
    pid: int | None = None
    active_sessions: int = 0
    provider_profile: str = "codex"
    enabled_channels: list[str] = Field(default_factory=list)
    last_error: str | None = None
