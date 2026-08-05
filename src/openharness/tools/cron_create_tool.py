"""Tool for creating local cron-style jobs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from openharness.services.cron import upsert_cron_job, validate_cron_expression, validate_timezone
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class CronCreateToolInput(BaseModel):
    """Arguments for cron job creation."""

    name: str = Field(description="Unique cron job name")
    schedule: str = Field(
        description=(
            "Cron schedule expression (e.g. '*/5 * * * *' for every 5 minutes, "
            "'0 9 * * 1-5' for weekdays at 9am)"
        ),
    )
    command: str | None = Field(
        default=None,
        description="Machine/background shell command to run when triggered.",
    )
    message: str | None = Field(
        default=None,
        description=(
            "Instruction for a headless agent_turn cron job. This runs in the "
            "background and is not a user-facing chat reply; in an ohmo gateway "
            "chat use remind_create instead."
        ),
    )
    timezone: str | None = Field(default=None, description="IANA timezone for interpreting cron schedule")
    cwd: str | None = Field(default=None, description="Optional working directory override")
    enabled: bool = Field(default=True, description="Whether the job is active")
    payload: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional headless nanobot-style payload for background work; it is "
            "not a user-facing chat delivery. In an ohmo gateway chat use "
            "remind_create instead. Example: "
            "{'kind': 'agent_turn', 'message': 'check GitHub', 'deliver': True, 'channel': 'feishu', 'to': 'ou_xxx'}."
        ),
    )
    notify: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional notification target. Example: "
            "{'type': 'feishu_dm', 'user_open_id': 'ou_xxx'} to send job output to a Feishu private chat."
        ),
    )


class CronCreateTool(BaseTool):
    """Create or replace a local cron job."""

    name = "cron_create"
    description = (
        "Create or replace a background cron job with a standard cron expression. "
        "Use 'command' for machine work or 'message'/'payload' for a headless "
        "agent_turn; neither produces a user-facing chat reply. In an ohmo gateway "
        "chat use remind_create for reminders. Use 'oh cron start' to run the "
        "scheduler daemon."
    )
    input_model = CronCreateToolInput

    async def execute(
        self,
        arguments: CronCreateToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not validate_cron_expression(arguments.schedule):
            return ToolResult(
                output=(
                    f"Invalid cron expression: {arguments.schedule!r}\n"
                    "Use standard 5-field format: minute hour day month weekday\n"
                    "Examples: '*/5 * * * *' (every 5 min), '0 9 * * 1-5' (weekdays 9am)"
                ),
                is_error=True,
            )
        if not validate_timezone(arguments.timezone):
            return ToolResult(output=f"Invalid timezone: {arguments.timezone!r}", is_error=True)

        # The scheduler gives an explicit command precedence over any payload.
        # Classify the effective job using that same precedence, while accepting
        # harmless case/whitespace variations in gateway input.
        requests_agent_turn = False
        if not arguments.command:
            if arguments.message is not None:
                requests_agent_turn = True
            elif arguments.payload is not None:
                requested_payload_kind = str(
                    arguments.payload.get("kind", "agent_turn")
                ).strip().casefold()
                requests_agent_turn = requested_payload_kind == "agent_turn"
        if "ohmo_reminder_ctx" in context.metadata and requests_agent_turn:
            return ToolResult(
                output=(
                    "Agent-turn cron jobs are not available in an ohmo gateway "
                    "chat because cron output is not delivered as a chat reply. "
                    "Use remind_create instead (delivery='explicit' for a "
                    "conditionally silent current-chat reminder). Machine command "
                    "cron jobs remain available through cron_create."
                ),
                is_error=True,
            )

        payload = dict(arguments.payload or {})
        if arguments.message:
            payload.setdefault("kind", "agent_turn")
            payload.setdefault("message", arguments.message)
        if arguments.notify is not None:
            payload.setdefault("deliver", True)
            if str(arguments.notify.get("type") or "").strip().lower() == "feishu_dm":
                payload.setdefault("channel", "feishu")
                payload.setdefault("to", arguments.notify.get("user_open_id") or arguments.notify.get("open_id"))

        if payload and not payload.get("message") and not arguments.command:
            return ToolResult(output="Cron job requires payload.message, message, or command.", is_error=True)
        if not payload and not arguments.command:
            return ToolResult(output="Cron job requires command or message.", is_error=True)

        job = {
            "name": arguments.name,
            "schedule": arguments.schedule,
            "cwd": arguments.cwd or str(context.cwd),
            "enabled": arguments.enabled,
        }
        if arguments.timezone:
            job["timezone"] = arguments.timezone
        if arguments.command is not None:
            job["command"] = arguments.command
        if payload:
            payload.setdefault("kind", "agent_turn")
            job["payload"] = payload
        if arguments.notify is not None:
            job["notify"] = arguments.notify
        upsert_cron_job(job)
        status = "enabled" if arguments.enabled else "disabled"
        return ToolResult(
            output=f"Created cron job '{arguments.name}' [{arguments.schedule}] ({status})"
        )
