"""Compatibility channel config models.

These models keep the synced channel adapters importable while the main
OpenHarness settings system evolves independently.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _CompatModel(BaseModel):
    """Base model that tolerates adapter-specific extra fields."""

    model_config = ConfigDict(extra="allow")


class ProviderApiKeyConfig(_CompatModel):
    api_key: str = ""


class ProviderConfigs(_CompatModel):
    groq: ProviderApiKeyConfig = Field(default_factory=ProviderApiKeyConfig)


class BaseChannelConfig(_CompatModel):
    enabled: bool = False
    # Secure default: enabling a channel does not automatically trust every
    # remote sender. Operators must explicitly allow specific identities, or
    # intentionally set ["*"] when they want open access.
    allow_from: list[str] = Field(default_factory=list)


class TelegramConfig(BaseChannelConfig):
    token: str = ""
    chat_id: str | None = None
    proxy: str | None = None
    reply_to_message: bool = True
    bot_name: str = "ohmo"
    # Local ASR subprocess wiring for voice/audio messages (fail-closed by
    # default: disabled, no command configured). When enabled, the argv list is
    # the full command template (program + flags, no shell); the downloaded
    # audio path is appended as the final, separate argv element. The command
    # must print a JSON object with a ``text`` field on stdout. Non-zero exits
    # must write the ElevenLabs schema_version=1 JSON error object to stderr;
    # prose stderr is intentionally treated as a non-retryable failure.
    voice_transcription_enabled: bool = False
    voice_transcription_argv: list[str] = Field(default_factory=list)
    voice_transcription_timeout_seconds: float = 30.0
    voice_transcription_max_attempts: int = 2
    voice_transcription_base_backoff_seconds: float = 0.5
    voice_transcription_total_budget_seconds: float = 90.0
    # Quiet progress: how many of the most recent tool activities stay visible.
    compact_tool_rows: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def _validate_voice_transcription(self) -> TelegramConfig:
        if not 0 < float(self.voice_transcription_timeout_seconds) <= 600:
            raise ValueError(
                "voice_transcription_timeout_seconds must be in (0, 600]"
            )
        if self.voice_transcription_enabled and (
            not self.voice_transcription_argv
            or any(
                not isinstance(part, str) or not part.strip()
                for part in self.voice_transcription_argv
            )
        ):
            raise ValueError(
                "voice_transcription_argv must be a non-empty list of non-empty "
                "strings when voice transcription is enabled"
            )
        if not 1 <= int(self.voice_transcription_max_attempts) <= 3:
            raise ValueError("voice_transcription_max_attempts must be in [1, 3]")
        if not 0 <= float(self.voice_transcription_base_backoff_seconds) <= 30:
            raise ValueError(
                "voice_transcription_base_backoff_seconds must be in [0, 30]"
            )
        if not 0 < float(self.voice_transcription_total_budget_seconds) <= 900:
            raise ValueError(
                "voice_transcription_total_budget_seconds must be in (0, 900]"
            )
        return self


class SlackConfig(BaseChannelConfig):
    bot_token: str = ""
    app_token: str = ""
    signing_secret: str = ""


class DiscordConfig(BaseChannelConfig):
    token: str = ""


class FeishuConfig(BaseChannelConfig):
    app_id: str = ""
    app_secret: str = ""
    encrypt_key: str = ""
    verification_token: str = ""
    # Group reply policy is enforced by ohmo gateway because managed-group
    # metadata lives outside the generic Feishu channel adapter.
    group_policy: str = "managed_or_mention"
    bot_open_id: str = ""
    bot_names: list[str] = Field(default_factory=lambda: ["ohmo", "openclaw", "openharness"])
    domain: str = "https://open.feishu.cn"  # use https://open.larksuite.com for Lark international


class DingTalkConfig(BaseChannelConfig):
    client_id: str = ""
    client_secret: str = ""
    robot_code: str = ""


class EmailConfig(BaseChannelConfig):
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    from_address: str = ""


class QQConfig(BaseChannelConfig):
    token: str = ""
    app_id: str = ""
    app_secret: str = ""


class MatrixConfig(BaseChannelConfig):
    homeserver: str = ""
    access_token: str = ""
    user_id: str = ""


class WhatsAppConfig(BaseChannelConfig):
    access_token: str = ""
    phone_number_id: str = ""
    verify_token: str = ""


class MochatConfig(BaseChannelConfig):
    endpoint: str = ""
    token: str = ""


class ChannelConfigs(_CompatModel):
    send_progress: bool = True
    send_tool_hints: bool = True
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    dingtalk: DingTalkConfig = Field(default_factory=DingTalkConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    qq: QQConfig = Field(default_factory=QQConfig)
    matrix: MatrixConfig = Field(default_factory=MatrixConfig)
    whatsapp: WhatsAppConfig = Field(default_factory=WhatsAppConfig)
    mochat: MochatConfig = Field(default_factory=MochatConfig)


class Config(_CompatModel):
    channels: ChannelConfigs = Field(default_factory=ChannelConfigs)
    providers: ProviderConfigs = Field(default_factory=ProviderConfigs)
