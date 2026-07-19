"""Import one consenting tenant's conversation history into Honcho."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import json
import math
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from ohmo.memory_service.honcho_client import HonchoClient

_SCHEDULER_SENDER_MARKER = "Sender id: __scheduler__"


@dataclasses.dataclass(frozen=True, slots=True)
class BackfillReport:
    """Counts from a tenant-history backfill plan or execution."""

    prepared: int
    sent: int
    skipped_scheduler: int
    skipped_empty: int
    batches: int

    def render(self) -> str:
        """Render a compact operator-facing report."""
        return "\n".join(
            (
                "Tenant history backfill report",
                f"  Prepared: {self.prepared}",
                f"  Sent: {self.sent}",
                f"  Skipped scheduler: {self.skipped_scheduler}",
                f"  Skipped empty: {self.skipped_empty}",
                f"  Batches: {self.batches}",
            )
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _SelectedSession:
    created_at: float
    path: Path
    session_id: str
    messages: tuple[Mapping[str, object], ...]


async def backfill_tenant_history(
    *,
    sessions_dir: str | Path,
    session_keys: Collection[str],
    honcho_client: HonchoClient,
    observed_peer: str,
    assistant_peer: str = "ohmo",
    session: str = "ohmo",
    import_run_id: str,
    batch_size: int = 40,
    dry_run: bool = False,
) -> BackfillReport:
    """Backfill only the selected person's turns into one tenant client.

    ``honcho_client`` is the sole send destination and ``session_keys`` is the
    sole source selector. Callers therefore cannot supply per-message clients
    or destinations that could mix tenant histories.
    """
    if not observed_peer:
        raise ValueError("observed_peer must not be empty")
    if not assistant_peer:
        raise ValueError("assistant_peer must not be empty")
    if not session:
        raise ValueError("session must not be empty")
    if not import_run_id:
        raise ValueError("import_run_id must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    selected_sessions = _load_selected_sessions(Path(sessions_dir), frozenset(session_keys))
    prepared: list[dict[str, object]] = []
    skipped_scheduler = 0
    skipped_empty = 0

    for selected in selected_sessions:
        for index, message in enumerate(selected.messages):
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue

            text = _message_text(message.get("content"))
            if not text.strip():
                skipped_empty += 1
                continue
            if role == "user" and _SCHEDULER_SENDER_MARKER in text:
                skipped_scheduler += 1
                continue

            peer_id = observed_peer if role == "user" else assistant_peer
            prepared.append(
                {
                    "peer_id": peer_id,
                    "content": text,
                    "metadata": {
                        "role": role,
                        "backfill": True,
                        "source_session_id": selected.session_id,
                        "import_run_id": import_run_id,
                        "client_op_id": f"{selected.session_id}:{index}",
                    },
                }
            )

    allowed_peers = {observed_peer, assistant_peer}
    assert all(message.get("peer_id") in allowed_peers for message in prepared), (
        "tenant-history backfill prepared a message for a foreign peer"
    )

    batches = math.ceil(len(prepared) / batch_size)
    sent = 0
    if not dry_run:
        for offset in range(0, len(prepared), batch_size):
            batch = prepared[offset : offset + batch_size]
            await honcho_client.create_messages(session, batch)
            sent += len(batch)

    return BackfillReport(
        prepared=len(prepared),
        sent=sent,
        skipped_scheduler=skipped_scheduler,
        skipped_empty=skipped_empty,
        batches=batches,
    )


def _load_selected_sessions(
    directory: Path, session_keys: frozenset[str]
) -> list[_SelectedSession]:
    selected: list[_SelectedSession] = []
    seen_session_ids: set[str] = set()

    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Session file must contain an object: {path}")

        # Do not inspect another person's messages after identifying its key.
        if payload.get("session_key") not in session_keys:
            continue

        raw_session_id = payload.get("session_id", path.stem)
        if not isinstance(raw_session_id, str) or not raw_session_id:
            raise ValueError(f"Session file has an invalid session_id: {path}")
        if raw_session_id in seen_session_ids:
            continue

        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError(f"Session file has an invalid messages list: {path}")
        messages: list[Mapping[str, object]] = []
        for index, message in enumerate(raw_messages):
            if not isinstance(message, Mapping):
                raise ValueError(f"Session message {index} must be an object: {path}")
            messages.append(message)

        selected.append(
            _SelectedSession(
                created_at=_created_at_timestamp(payload.get("created_at"), path),
                path=path,
                session_id=raw_session_id,
                messages=tuple(messages),
            )
        )
        seen_session_ids.add(raw_session_id)

    selected.sort(key=lambda item: (item.created_at, item.path.name))
    return selected


def _created_at_timestamp(value: object, path: Path) -> float:
    timestamp: float
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = float(value)
    elif isinstance(value, str):
        try:
            timestamp = float(value)
        except ValueError:
            try:
                timestamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError as exc:
                raise ValueError(f"Session file has an invalid created_at: {path}") from exc
    else:
        raise ValueError(f"Session file has an invalid created_at: {path}")
    if not math.isfinite(timestamp):
        raise ValueError(f"Session file has an invalid created_at: {path}")
    return timestamp


def _message_text(content: object) -> str:
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill one consenting tenant's conversation history into Honcho.",
    )
    parser.add_argument("--sessions-dir", required=True, help="Directory containing session JSON.")
    parser.add_argument(
        "--session-keys",
        required=True,
        nargs="+",
        help="The consenting person's session keys (for example telegram:110006972).",
    )
    parser.add_argument("--observed-peer", required=True, help="This tenant's person peer.")
    parser.add_argument("--workspace", required=True, help="This tenant's Honcho workspace.")
    parser.add_argument("--base-url", required=True, help="Honcho service base URL.")
    parser.add_argument("--jwt", required=True, help="JWT scoped to this tenant's workspace.")
    parser.add_argument(
        "--import-run-id", required=True, help="Stable identifier for this import run."
    )
    parser.add_argument("--batch-size", type=int, default=40, help="Messages per Honcho request.")
    parser.add_argument("--dry-run", action="store_true", help="Plan without sending messages.")
    parser.add_argument(
        "--i-have-consent",
        action="store_true",
        help="Confirm the person consented; required unless --dry-run is used.",
    )
    return parser


async def _run_cli(args: argparse.Namespace) -> BackfillReport:
    async with HonchoClient(
        base_url=args.base_url,
        jwt=args.jwt,
        workspace=args.workspace,
    ) as client:
        return await backfill_tenant_history(
            sessions_dir=args.sessions_dir,
            session_keys=args.session_keys,
            honcho_client=client,
            observed_peer=args.observed_peer,
            import_run_id=args.import_run_id,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not args.i_have_consent:
        parser.error("--i-have-consent is required for a non-dry backfill")
    print(asyncio.run(_run_cli(args)).render())


if __name__ == "__main__":
    main()


__all__ = ["BackfillReport", "backfill_tenant_history", "main"]
