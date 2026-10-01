#!/usr/bin/env python3
"""Synthetic-only subprocess fixture for the real Ohmo Camera HTTP ingress."""

from __future__ import annotations

import argparse
import hashlib
import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

MAX_CONTROL_LINE = 2048


def _private_regular(path: Path, *, owner: int) -> os.stat_result:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != owner
        or info.st_mode & 0o077
        or path.is_symlink()
    ):
        raise ValueError("fixture token must be an owner-private regular file")
    return info


def _load_runtime(ohmo_root: Path):
    ohmo_root = ohmo_root.absolute()
    if not ohmo_root.is_dir() or ohmo_root.is_symlink():
        raise RuntimeError("the read-only Ohmo source checkout is unavailable")
    sys.path.insert(0, str(ohmo_root))
    sys.path.insert(0, str(ohmo_root / "tests" / "test_ohmo"))
    from ohmo.gateway import camera as camera_module
    from ohmo.gateway.camera import CameraIngress, serve_camera_http
    from ohmo.gateway.models import CameraIngressConfig
    from openharness.channels.bus.queue import MessageBus
    from openharness.channels.bus.events import InboundMessage
    from test_camera_ingress import FakeTelegram

    return CameraIngress, serve_camera_http, CameraIngressConfig, MessageBus, FakeTelegram, camera_module, InboundMessage


class FixtureClock(datetime):
    current = datetime.now(timezone.utc)

    @classmethod
    def now(cls, tz=None):
        value = cls.current
        return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)


class FixtureServer:
    def __init__(self, workspace: Path, token_file: Path, ohmo_root: Path, *, hold_photo: bool):
        self.workspace = workspace
        self.token_file = token_file
        (
            self.CameraIngress,
            self.serve_camera_http,
            self.CameraIngressConfig,
            self.MessageBus,
            self.FakeTelegram,
            self.camera_module,
            self.InboundMessage,
        ) = _load_runtime(ohmo_root)
        FixtureClock.current = datetime.now(timezone.utc)
        self.camera_module.datetime = FixtureClock
        self.bus = self.MessageBus()
        self.photo_gate = asyncio.Event()
        if not hold_photo:
            self.photo_gate.set()

        class FixtureTelegram(self.FakeTelegram):
            async def send_camera_photo(inner_self, **kwargs):
                await self.photo_gate.wait()
                return await super(FixtureTelegram, inner_self).send_camera_photo(**kwargs)

        self.telegram = FixtureTelegram()
        self.config = None
        self.ingress = None
        self.server = None
        self.port = 0
        self.active: set[asyncio.Task] = set()
        self.http_requests = 0
        self.http_statuses: dict[str, int] = {}

    async def start(self) -> None:
        self.config = self.CameraIngressConfig(
            enabled=True,
            listen_host="127.0.0.1",
            listen_port=0,
            bearer_token_file=self.token_file,
            principal="123",
            tenant_id="synthetic-fixture",
            chat_id="123",
            session_key="telegram:123",
        )
        self.ingress = self.CameraIngress(
            self.config,
            workspace=self.workspace,
            bus=self.bus,
            telegram=self.telegram,
        )
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = int(self.server.sockets[0].getsockname()[1])

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        task = asyncio.current_task()
        if task is not None:
            self.active.add(task)
        self.http_requests += 1

        class CountedWriter:
            def __init__(self, wrapped, record):
                self.wrapped = wrapped
                self.record = record

            def write(self, data: bytes) -> None:
                if data.startswith(b"HTTP/1.1 "):
                    code = data.split(b" ", 2)[1].decode("ascii", "replace")
                    self.record(code)
                self.wrapped.write(data)

            async def drain(self):
                await self.wrapped.drain()

            def close(self):
                self.wrapped.close()

            async def wait_closed(self):
                await self.wrapped.wait_closed()

        try:
            await self.serve_camera_http(
                self.ingress,
                reader,
                CountedWriter(writer, self._record_status),
            )
        finally:
            if task is not None:
                self.active.discard(task)

    def _record_status(self, code: str) -> None:
        self.http_statuses[code] = self.http_statuses.get(code, 0) + 1

    def counters(self) -> dict[str, Any]:
        attempts = self.ingress._attempts
        journal_info = self.ingress._state_path.stat()
        journal_sha = hashlib.sha256(self.ingress._state_path.read_bytes()).hexdigest()
        states: dict[str, int] = {}
        for attempt in attempts.values():
            state = str(attempt.get("state", "unknown"))
            states[state] = states.get(state, 0) + 1
        return {
            "http_requests": self.http_requests,
            "http_statuses": dict(sorted(self.http_statuses.items())),
            "journal_entries": len(attempts),
            "attempt_states": dict(sorted(states.items())),
            "committed_seq": self.ingress._session["committed_seq"],
            "photo_calls": len(self.telegram.calls),
            "queued_messages": self.bus.inbound_size,
            "confirmed_photo_attempts": sum(
                attempt.get("photo_delivery_confirmed") is True
                for attempt in attempts.values()
            ),
            "reference_source_records": sum(
                isinstance(attempt.get("reference_source"), dict)
                for attempt in attempts.values()
            ),
            "attempt_history": {
                candidate_id: {
                    "state": attempt.get("state"),
                    "attention_active": attempt.get("attention_active"),
                    "answer_kind": attempt.get("answer_kind"),
                    "request_identity": attempt.get("request_identity"),
                    "request_ack": attempt.get("request_ack"),
                    "admission_id": attempt.get("admission_id"),
                    "image_sha256": attempt.get("image_sha256"),
                    "capture_time": attempt.get("capture_time"),
                    "capture_time_authority": attempt.get("capture_time_authority"),
                    "photo_id": attempt.get("photo_id"),
                    "photo_delivery_confirmed": attempt.get("photo_delivery_confirmed"),
                }
                for candidate_id, attempt in attempts.items()
            },
            "camera_meal_commits": sum(
                isinstance(attempt.get("camera_commit"), dict)
                for attempt in attempts.values()
            ),
            "journal_sha256": journal_sha,
            "journal_mtime_ns": journal_info.st_mtime_ns,
            "camera_now": FixtureClock.current.isoformat(),
        }

    async def await_idle(self, timeout_ms: int) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_ms / 1000
        while self.ingress._tasks:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("camera delivery tasks did not become idle")
            await asyncio.sleep(0.01)

    async def restart(self) -> None:
        await self.await_idle(5000)
        await self._close_listener()
        await self.ingress.close()
        self.ingress = self.CameraIngress(
            self.config,
            workspace=self.workspace,
            bus=self.bus,
            telegram=self.telegram,
        )
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = int(self.server.sockets[0].getsockname()[1])

    def make_legacy_source_conflict(self) -> dict[str, str]:
        """Persist a legacy receipt whose retained image digest contradicts its source."""
        candidates = [
            (candidate_id, attempt)
            for candidate_id, attempt in self.ingress._attempts.items()
            if attempt.get("photo_delivery_confirmed") is True
        ]
        if len(candidates) != 1:
            raise ValueError("fixture requires exactly one confirmed Camera receipt")
        candidate_id, attempt = candidates[0]
        request_identity = attempt.get("request_identity")
        original_sha256 = (
            request_identity.get("image_sha256")
            if isinstance(request_identity, dict)
            else None
        )
        snapshot = attempt.get("snapshot")
        admission_id = attempt.get("admission_id")
        if (
            not isinstance(original_sha256, str)
            or attempt.get("image_sha256") != original_sha256
            or not isinstance(snapshot, str)
            or Path(snapshot).parent != self.ingress._state_dir / "snapshots"
            or not isinstance(admission_id, str)
            or Path(snapshot).name not in {
                f"{admission_id}.jpg", f"{admission_id}.jpeg",
                f"{admission_id}.png", f"{admission_id}.webp",
            }
            or hashlib.sha256(Path(snapshot).read_bytes()).hexdigest() != original_sha256
        ):
            raise ValueError("fixture receipt does not retain its exact original image evidence")
        conflicting_sha256 = "0" * 64 if original_sha256 != "0" * 64 else "1" * 64
        attempt.pop("reference_source", None)
        attempt.pop("capture_time", None)
        attempt.pop("capture_time_authority", None)
        attempt["image_sha256"] = conflicting_sha256
        self.ingress._save_attempts()
        return {
            "candidate_id": candidate_id,
            "original_image_sha256": original_sha256,
            "conflicting_image_sha256": conflicting_sha256,
        }

    def make_restorable_legacy_source_gap(self) -> str:
        """Model a legacy receipt missing capture fields and its managed snapshot."""
        candidates = [
            (candidate_id, attempt)
            for candidate_id, attempt in self.ingress._attempts.items()
            if attempt.get("photo_delivery_confirmed") is True
        ]
        if len(candidates) != 1:
            raise ValueError("fixture requires exactly one confirmed Camera receipt")
        candidate_id, attempt = candidates[0]
        snapshot = attempt.get("snapshot")
        admission_id = attempt.get("admission_id")
        if (
            not isinstance(snapshot, str)
            or Path(snapshot).parent != self.ingress._state_dir / "snapshots"
            or not isinstance(admission_id, str)
            or Path(snapshot).name not in {
                f"{admission_id}.jpg", f"{admission_id}.jpeg",
                f"{admission_id}.png", f"{admission_id}.webp",
            }
        ):
            raise ValueError("fixture receipt does not name its exact managed snapshot")
        Path(snapshot).unlink()
        attempt.pop("reference_source", None)
        attempt.pop("capture_time", None)
        attempt.pop("capture_time_authority", None)
        self.ingress._save_attempts()
        return candidate_id

    def complete_negative_answer(self) -> str:
        candidates = [
            (candidate_id, attempt)
            for candidate_id, attempt in self.ingress._attempts.items()
            if attempt.get("photo_delivery_confirmed") is True
            and attempt.get("attention_active") is True
        ]
        if len(candidates) != 1:
            raise ValueError("fixture requires exactly one active delivered photo")
        candidate_id, attempt = candidates[0]
        answer = self.InboundMessage(
            channel="telegram",
            sender_id=self.config.principal,
            chat_id=self.config.chat_id,
            content="Нет, не ел(а)",
            metadata={
                "reply_to_message_id": attempt["photo_id"],
                "message_id": 9000 + len(self.telegram.calls),
                "_telegram_raw_text": "Нет, не ел(а)",
            },
        )
        self.ingress.process_real_inbound(answer)
        if answer.metadata.get("_camera_answer") != "no":
            raise ValueError("synthetic owner denial did not bind to its photo")
        self.ingress.complete(answer, recorded=False)
        return candidate_id

    def release_photos(self) -> None:
        self.photo_gate.set()

    def hold_photos(self) -> None:
        self.photo_gate.clear()

    def advance_camera_days(self, days: int) -> None:
        FixtureClock.current += timedelta(days=days)

    async def _close_listener(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        if self.active:
            await asyncio.wait_for(
                asyncio.gather(*tuple(self.active), return_exceptions=True), 5
            )

    async def close(self) -> None:
        await self._close_listener()
        if self.ingress is not None:
            await self.ingress.close()


def _control(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) - {"op", "timeout_ms", "days"}:
        raise ValueError("control must be one bounded operation object")
    op = value.get("op")
    if op not in {
        "counters", "await_idle", "restart", "release_photos", "hold_photos",
        "advance_camera_days", "make_legacy_source_conflict",
        "make_restorable_legacy_source_gap", "complete_negative_answer", "quit",
    }:
        raise ValueError("unsupported fixture control operation")
    timeout = value.get("timeout_ms", 5000)
    if type(timeout) is not int or not 1 <= timeout <= 30000:
        raise ValueError("timeout_ms must be between 1 and 30000")
    days = value.get("days", 0)
    if type(days) is not int or not 0 <= days <= 14:
        raise ValueError("days must be between 0 and 14")
    return {"op": op, "timeout_ms": timeout, "days": days}


async def _run(workspace: Path, token_file: Path, ohmo_root: Path, *, hold_photo: bool) -> None:
    server = FixtureServer(workspace, token_file, ohmo_root, hold_photo=hold_photo)
    await server.start()
    print(json.dumps({"ready": True, "host": "127.0.0.1", "port": server.port}), flush=True)
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.buffer.readline, MAX_CONTROL_LINE + 1)
            if not line:
                break
            try:
                if len(line) > MAX_CONTROL_LINE or not line.endswith(b"\n"):
                    raise ValueError("control line exceeds bound")
                command = _control(json.loads(line))
                op = command["op"]
                if op == "counters":
                    result = {"ok": True, "counters": server.counters()}
                elif op == "await_idle":
                    await server.await_idle(command["timeout_ms"])
                    result = {"ok": True, "counters": server.counters()}
                elif op == "restart":
                    before = len(server.ingress._attempts)
                    await server.restart()
                    result = {
                        "ok": True,
                        "port": server.port,
                        "journal_entries": len(server.ingress._attempts),
                        "journal_preserved": len(server.ingress._attempts) == before,
                    }
                elif op == "release_photos":
                    server.release_photos()
                    result = {"ok": True}
                elif op == "hold_photos":
                    server.hold_photos()
                    result = {"ok": True}
                elif op == "advance_camera_days":
                    server.advance_camera_days(command["days"])
                    result = {"ok": True, "camera_now": FixtureClock.current.isoformat()}
                elif op == "make_legacy_source_conflict":
                    result = {"ok": True, **server.make_legacy_source_conflict()}
                elif op == "make_restorable_legacy_source_gap":
                    result = {
                        "ok": True,
                        "candidate_id": server.make_restorable_legacy_source_gap(),
                    }
                elif op == "complete_negative_answer":
                    result = {"ok": True, "candidate_id": server.complete_negative_answer()}
                else:
                    result = {"ok": True, "stopping": True}
                    print(json.dumps(result, separators=(",", ":")), flush=True)
                    return
                print(json.dumps(result, separators=(",", ":")), flush=True)
            except Exception as exc:
                print(
                    json.dumps(
                    {"ok": False, "error": type(exc).__name__, "detail": str(exc)[:160]},
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    finally:
        await server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--ohmo-checkout", required=True, type=Path)
    parser.add_argument("--hold-photo", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.absolute()
    token_file = args.token_file.absolute()
    ohmo_root = args.ohmo_checkout.absolute()
    owner = os.getuid()
    workspace_stat = workspace.lstat()
    if (
        not stat.S_ISDIR(workspace_stat.st_mode)
        or workspace_stat.st_uid != owner
        or workspace_stat.st_mode & 0o077
        or workspace.is_symlink()
        or any(workspace.iterdir())
    ):
        raise SystemExit("workspace must be a fresh owner-private directory")
    _private_regular(token_file, owner=owner)
    token = token_file.read_bytes()
    if len(token.rstrip(b"\n")) < 32 or len(token) > 256 or b"\x00" in token:
        raise SystemExit("synthetic token file is malformed")
    try:
        asyncio.run(_run(workspace, token_file, ohmo_root, hold_photo=args.hold_photo))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
