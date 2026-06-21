"""Ohmo gateway episode recorder built on the generic eval store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from openharness.channels.bus.events import InboundMessage
from openharness.evals import EvalEpisode, EvalEvent, EvalStore
from openharness.evals.tool_labels import effective_tool_label, tool_call_binaries
from openharness.engine.stream_events import (
    ErrorEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)

from ohmo.evals.adapter import get_eval_store
from ohmo.evals.resources import ResourceSnapshotWrite, write_ohmo_resource_snapshot


@dataclass
class GatewayEvalRecorder:
    """Append Ohmo gateway episodes and events to the local eval store."""

    store: EvalStore
    episode_id: str
    _finished: bool = False

    @classmethod
    def start(
        cls,
        *,
        workspace: str | Path,
        bundle: Any,
        message: InboundMessage,
        session_key: str,
        user_text: str,
        user_goal: str,
    ) -> "GatewayEvalRecorder":
        recorder = cls(
            store=get_eval_store(workspace),
            episode_id=f"ohmo-gateway-{uuid4().hex}",
        )
        metadata = {
            "workspace": str(Path(workspace).expanduser().resolve()),
            "session_key": session_key,
            "cwd": _bundle_cwd(bundle),
            "model": _bundle_model(bundle),
            "media_count": len(message.media or []),
            "inbound": _inbound_metadata(message),
        }
        recorder.store.append_episode(
            EvalEpisode(
                episode_id=recorder.episode_id,
                source="gateway",
                app="ohmo",
                session_id=str(getattr(bundle, "session_id", "") or ""),
                user_goal=user_goal,
                user_text=user_text,
                tags=["gateway", str(message.channel)],
                privacy=str((message.metadata or {}).get("privacy") or "private"),
                status=str((message.metadata or {}).get("status") or "open"),
                metadata=_json_safe_mapping(metadata),
            )
        )
        recorder.record_inbound_message(message, user_text=user_text, user_goal=user_goal)
        recorder.record_resource_snapshot(workspace=workspace, bundle=bundle)
        return recorder

    def record_inbound_message(
        self, message: InboundMessage, *, user_text: str, user_goal: str
    ) -> None:
        self.record_event(
            "inbound_message",
            payload={
                **_inbound_metadata(message),
                "user_text": user_text,
                "user_goal": user_goal,
            },
        )

    def record_resource_snapshot(
        self,
        *,
        workspace: str | Path | None,
        bundle: Any | None,
        phase: str = "world_before",
    ) -> ResourceSnapshotWrite:
        snapshot = write_ohmo_resource_snapshot(
            store=self.store,
            episode_id=self.episode_id,
            workspace=workspace,
            bundle=bundle,
            phase=phase,
        )
        self.record_event(
            "resource_snapshot",
            payload={
                "path": snapshot.relative_path,
                "phase": phase,
                "resource_count": snapshot.resource_count,
                "local_resource_count": snapshot.local_resource_count,
                "tool_count": snapshot.tool_count,
            },
        )
        return snapshot

    def record_tool_started(self, event: ToolExecutionStarted) -> None:
        payload: dict[str, Any] = {
            "input_summary": _summary(event.tool_input),
            "input": event.tool_input,
        }
        # Lift the real capability out of a shell tool's command so the
        # trajectory is not collapsed to "bash": record which binaries were
        # invoked and the effective label (e.g. "bash:maps-cli reviews").
        binaries = tool_call_binaries(event.tool_name, event.tool_input)
        if binaries:
            payload["binaries"] = binaries
            payload["capability"] = effective_tool_label(event.tool_name, event.tool_input)
        self.record_event(
            "tool_started",
            payload=payload,
            tool_name=event.tool_name,
            tool_call_id=event.tool_call_id,
        )

    def record_tool_completed(self, event: ToolExecutionCompleted) -> None:
        self.record_event(
            "tool_completed",
            payload={
                "output_summary": _summary(event.output),
                "output": event.output,
            },
            tool_name=event.tool_name,
            tool_call_id=event.tool_call_id,
            is_error=event.is_error,
        )

    def record_engine_error(self, event: ErrorEvent) -> None:
        self.record_event(
            "engine_error",
            payload={"message": event.message, "recoverable": event.recoverable},
            is_error=True,
        )

    def record_gateway_final(self, *, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self.record_event(
            "gateway_final",
            payload={"text": text, "metadata": metadata or {}},
        )

    def record_gateway_error(
        self, *, text: str, metadata: Mapping[str, Any] | None = None
    ) -> None:
        self.record_event(
            "gateway_error",
            payload={"text": text, "metadata": metadata or {}},
            is_error=True,
        )

    def record_exception(self, exc: Exception) -> None:
        self.record_event(
            "exception",
            payload={"type": type(exc).__name__, "message": str(exc)},
            is_error=True,
        )

    def finish(self, *, status: str) -> None:
        if self._finished:
            return
        self.record_event(
            "episode_finished",
            payload={"status": status},
            is_error=status not in {"completed", "ok"},
        )
        self._finished = True

    def record_event(
        self,
        kind: str,
        *,
        payload: Mapping[str, Any] | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> None:
        self.store.append_event(
            EvalEvent(
                episode_id=self.episode_id,
                kind=kind,
                payload=_json_safe_mapping(payload or {}),
                tool_name=tool_name or None,
                tool_call_id=tool_call_id or None,
                is_error=is_error,
            )
        )


def _inbound_metadata(message: InboundMessage) -> dict[str, Any]:
    return {
        "channel": message.channel,
        "chat_id": str(message.chat_id),
        "sender_id": str(message.sender_id),
        "timestamp": message.timestamp,
        "media_count": len(message.media or []),
        "metadata": message.metadata or {},
    }


def _bundle_cwd(bundle: Any) -> str:
    cwd = getattr(bundle, "cwd", "")
    return str(cwd) if cwd else ""


def _bundle_model(bundle: Any) -> str:
    settings = getattr(bundle, "current_settings", None)
    if callable(settings):
        current = settings()
        model = getattr(current, "model", None)
        if model:
            return str(model)
    engine = getattr(bundle, "engine", None)
    model = getattr(engine, "model", None)
    return str(model or "")


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    safe = _json_safe(value)
    if not isinstance(safe, dict):
        return {"value": safe}
    return safe


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _summary(value: Any, *, limit: int = 160) -> str:
    safe = _json_safe(value)
    if isinstance(safe, str):
        text = safe
    else:
        text = repr(safe)
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."
