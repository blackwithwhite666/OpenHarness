import json
from pathlib import Path

import pytest

from ohmo.tools.backfill_tenant_history import backfill_tenant_history, main


class FakeHoncho:
    def __init__(self, workspace: str = "tenant-marina") -> None:
        self.workspace = workspace
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    async def create_messages(
        self,
        session: str,
        messages: list[dict[str, object]],
    ) -> list[object]:
        self.calls.append((session, [dict(message) for message in messages]))
        return []


def _write_session(
    directory: Path,
    filename: str,
    *,
    session_id: str,
    session_key: str,
    created_at: float | str,
    messages: list[dict[str, object]],
) -> None:
    (directory / filename).write_text(
        json.dumps(
            {
                "session_id": session_id,
                "session_key": session_key,
                "created_at": created_at,
                "messages": messages,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def family_sessions(tmp_path: Path) -> Path:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    _write_session(
        sessions_dir,
        "z-marina-early.json",
        session_id="marina-early",
        session_key="telegram:110006972",
        created_at="2026-07-18T08:00:00Z",
        messages=[
            {
                "role": "user",
                "peer_id": "foreign-peer",
                "content": [{"type": "text", "text": "Marina first"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Assistant first"}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Sender id: __scheduler__\nTake a tablet",
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "image", "data": "ignored"}]},
            {
                "role": "foreign-peer",
                "content": [{"type": "text", "text": "must never be imported"}],
            },
        ],
    )
    _write_session(
        sessions_dir,
        "a-marina-late.json",
        session_id="marina-late",
        session_key="telegram:110006972",
        created_at=1_784_448_000,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "   "}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Marina "},
                    {"type": "text", "text": "second"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Assistant second"}],
            },
        ],
    )
    _write_session(
        sessions_dir,
        "boris.json",
        session_id="boris-only",
        session_key="telegram:220001337",
        created_at=1,
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Boris private turn"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Boris private answer"}],
            },
        ],
    )
    return sessions_dir


@pytest.mark.asyncio
async def test_backfill_sends_only_selected_person_in_chronological_batches(
    family_sessions: Path,
) -> None:
    honcho = FakeHoncho()

    report = await backfill_tenant_history(
        sessions_dir=family_sessions,
        session_keys={"telegram:110006972"},
        honcho_client=honcho,  # type: ignore[arg-type]
        observed_peer="marina",
        import_run_id="family-rollout-1",
        batch_size=2,
    )

    assert report.prepared == 4
    assert report.sent == 4
    assert report.skipped_scheduler == 1
    assert report.skipped_empty == 2
    assert report.batches == 2
    assert [session for session, _ in honcho.calls] == ["ohmo", "ohmo"]

    sent = [message for _, batch in honcho.calls for message in batch]
    assert [message["content"] for message in sent] == [
        "Marina first",
        "Assistant first",
        "Marina second",
        "Assistant second",
    ]
    assert [message["peer_id"] for message in sent] == ["marina", "ohmo", "marina", "ohmo"]
    assert all(message["peer_id"] in {"marina", "ohmo"} for message in sent)
    assert all("Boris" not in str(message["content"]) for message in sent)
    assert all("must never" not in str(message["content"]) for message in sent)


@pytest.mark.asyncio
async def test_backfill_dry_run_reports_plan_without_sending(family_sessions: Path) -> None:
    honcho = FakeHoncho()

    report = await backfill_tenant_history(
        sessions_dir=family_sessions,
        session_keys={"telegram:110006972"},
        honcho_client=honcho,  # type: ignore[arg-type]
        observed_peer="marina",
        import_run_id="family-rollout-1",
        batch_size=3,
        dry_run=True,
    )

    assert report.prepared == 4
    assert report.sent == 0
    assert report.skipped_scheduler == 1
    assert report.skipped_empty == 2
    assert report.batches == 2
    assert honcho.calls == []


@pytest.mark.asyncio
async def test_backfill_metadata_has_stable_source_order(family_sessions: Path) -> None:
    honcho = FakeHoncho()

    await backfill_tenant_history(
        sessions_dir=family_sessions,
        session_keys={"telegram:110006972"},
        honcho_client=honcho,  # type: ignore[arg-type]
        observed_peer="marina",
        import_run_id="family-rollout-1",
    )

    sent = [message for _, batch in honcho.calls for message in batch]
    assert [message["metadata"] for message in sent] == [
        {
            "role": "user",
            "backfill": True,
            "source_session_id": "marina-early",
            "import_run_id": "family-rollout-1",
            "client_op_id": "marina-early:0",
        },
        {
            "role": "assistant",
            "backfill": True,
            "source_session_id": "marina-early",
            "import_run_id": "family-rollout-1",
            "client_op_id": "marina-early:1",
        },
        {
            "role": "user",
            "backfill": True,
            "source_session_id": "marina-late",
            "import_run_id": "family-rollout-1",
            "client_op_id": "marina-late:1",
        },
        {
            "role": "assistant",
            "backfill": True,
            "source_session_id": "marina-late",
            "import_run_id": "family-rollout-1",
            "client_op_id": "marina-late:2",
        },
    ]


def test_cli_requires_explicit_consent_for_a_send(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--sessions-dir",
                str(tmp_path),
                "--session-keys",
                "telegram:110006972",
                "--observed-peer",
                "marina",
                "--workspace",
                "tenant-marina",
                "--base-url",
                "https://honcho.invalid",
                "--jwt",
                "tenant-jwt",
                "--import-run-id",
                "family-rollout-1",
            ]
        )

    assert exc_info.value.code == 2


@pytest.mark.asyncio
async def test_backfill_rejects_invalid_batch_size(family_sessions: Path) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        await backfill_tenant_history(
            sessions_dir=family_sessions,
            session_keys={"telegram:110006972"},
            honcho_client=FakeHoncho(),  # type: ignore[arg-type]
            observed_peer="marina",
            import_run_id="family-rollout-1",
            batch_size=0,
        )
