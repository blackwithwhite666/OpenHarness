"""Synthetic contract checks for ordinary person-origin gateway inputs."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from openharness.channels.bus.events import InboundMessage
from openharness.engine.messages import AttachmentRefBlock, ImageBlock, TextBlock, ToolUseBlock
from ohmo.gateway.runtime import _build_inbound_user_message
from ohmo.attachment_store import AttachmentStore
from ohmo.workspace import initialize_workspace
from probe_support import NativeClientPreconditionError, native_person_source_clients
from camera_runtime_support import OfflinePersonSourceApi
from person_source_input import source_run_status
from person_source_runtime import (
    honcho_history_snapshot,
    new_nutrition_event_ids,
    projected_intake_kcal,
    runtime_episode_evidence,
    run_person_source_turn,
)

from person_source_input import person_source_message


def test_person_text_uses_production_inbound_and_keeps_unthreaded_identity():
    sent_at = datetime(2026, 10, 5, 11, 12, tzinfo=timezone.utc)
    message = person_source_message(
        sender_id="42",
        chat_id="42",
        source_message_id="8",
        sent_at=sent_at,
        text="I had tea and a biscuit this morning.",
    )

    assert isinstance(message, InboundMessage)
    assert message.timestamp is sent_at
    assert message.sender_id == "42" and message.chat_id == "42"
    assert message.session_key == "telegram:42"
    assert message.metadata["is_group"] is False
    assert not any(
        key in message.metadata
        for key in (
            "reply_to_message_id",
            "native_message_id",
            "callback_query_id",
            "_camera_candidate_id",
        )
    )
    runtime_message = _build_inbound_user_message(message)
    user_text = next(
        block.text
        for block in runtime_message.content
        if isinstance(block, TextBlock) and block.text == message.content
    )
    assert runtime_message.event_id
    assert user_text == "I had tea and a biscuit this morning."
    assert message.metadata["message_id"] == "8"


def test_person_photo_is_passed_as_original_attachment_path(tmp_path: Path):
    photo = tmp_path / "person-photo.jpg"
    image_bytes = BytesIO()
    exif = Image.Exif()
    exif[0x010F] = "Synthetic Source Camera"
    Image.new("RGB", (2, 2), (20, 30, 40)).save(image_bytes, format="JPEG", exif=exif)
    photo.write_bytes(image_bytes.getvalue())
    sent_at = datetime(2026, 10, 5, 12, tzinfo=timezone.utc)
    message = person_source_message(
        sender_id="42",
        chat_id="42",
        source_message_id="9",
        sent_at=sent_at,
        text="Lunch",
        photo_path=photo,
    )

    assert message.media == [str(photo)]
    assert message.timestamp is sent_at
    assert message.metadata == {
        "message_id": "9",
        "chat_type": "private",
        "is_group": False,
    }
    attachment_store = AttachmentStore(initialize_workspace(tmp_path / ".ohmo-home"))
    runtime_message = _build_inbound_user_message(message, attachment_store)
    refs = [block for block in runtime_message.content if isinstance(block, AttachmentRefBlock)]
    images = [block for block in runtime_message.content if isinstance(block, ImageBlock)]
    assert len(refs) == len(images) == 1
    assert refs[0].source_provenance == {
        "schema_version": 1,
        "channel": "telegram",
        "principal": "telegram:42",
        "chat_id": "42",
        "session_key": "telegram:42",
        "received_at": sent_at.isoformat(),
        "timestamp_authority": "inbound_event_timestamp",
        "is_group": False,
        "is_forwarded": False,
        "source_message_id": "9",
        "append_source_message_id": "9",
    }
    assert images[0].data == base64.b64encode(photo.read_bytes()).decode("ascii")
    assert photo.read_bytes() == image_bytes.getvalue()
    assert attachment_store.load_image(refs[0].attachment_id).data == photo.read_bytes()
    with Image.open(BytesIO(base64.b64decode(images[0].data))) as delivered_image:
        assert delivered_image.getexif()[0x010F] == "Synthetic Source Camera"


@pytest.mark.parametrize(
    "overrides",
    [
        {"sender_id": ""},
        {"chat_id": ""},
        {"source_message_id": ""},
        {"source_message_id": "source-10"},
        {"sent_at": datetime(2026, 10, 5)},
        {"text": "", "photo_path": None},
    ],
)
def test_person_input_rejects_missing_identity_time_or_payload(overrides):
    values = {
        "sender_id": "42",
        "chat_id": "42",
        "source_message_id": "10",
        "sent_at": datetime(2026, 10, 5, tzinfo=timezone.utc),
        "text": "Tea",
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        person_source_message(**values)


def test_failed_or_missing_append_never_reports_saved():
    assert source_run_status([{"content": "error", "event_id": None}]) == {
        "saved": False,
        "event_id": None,
        "event_ids": [],
        "ambiguous": False,
    }
    ambiguous = source_run_status([
        {"event_id": "event-a"}, {"event_id": "event-b"}
    ])
    assert ambiguous["saved"] is False and ambiguous["event_id"] is None
    assert ambiguous["ambiguous"] is True


def test_native_person_source_requires_two_distinct_codex_clients_without_fallback():
    client_type = type("CodexFixture", (), {})
    profile = SimpleNamespace(
        provider="openai_codex", auth_source="codex_subscription",
        last_model="gpt-6-luna", default_model="gpt-6-luna",
    )
    settings = SimpleNamespace(
        hooks={}, mcp_servers={}, enabled_plugins={}, allow_project_plugins=False,
        allow_project_skills=False, project_skill_dirs=[],
        resolve_profile=lambda: ("codex", profile),
    )
    resolved = []

    def resolver(_settings):
        client = client_type()
        resolved.append(client)
        return client

    pair = native_person_source_clients(
        settings, scenario="public synthetic owner scenario", resolver=resolver,
        codex_client_type=client_type,
    )
    assert pair[0] is resolved[0] and pair[1] is resolved[1]
    assert pair[0] is not pair[1] and len(resolved) == 2

    with pytest.raises(NativeClientPreconditionError, match="public scenario"):
        native_person_source_clients(settings, scenario=" ", resolver=resolver)

    same = client_type()
    with pytest.raises(NativeClientPreconditionError, match="no provider fallback"):
        native_person_source_clients(
            settings, scenario="scenario", resolver=lambda _settings: same,
            codex_client_type=client_type,
        )
    with pytest.raises(NativeClientPreconditionError, match="no provider fallback"):
        native_person_source_clients(
            settings, scenario="scenario", resolver=lambda _settings: object(),
            codex_client_type=client_type,
        )


@pytest.mark.asyncio
async def test_honcho_export_keeps_every_scoped_row_without_content_filtering():
    created = datetime(2026, 10, 5, 11, tzinfo=timezone.utc)

    class Client:
        async def list_recent_message_metadata(
            self, session, *, expected_peer_id, since, until, page_size, max_pages
        ):
            assert session == "synthetic-session"
            assert expected_peer_id == "ohmo"
            assert since < until
            assert page_size == max_pages == 100
            return [
                SimpleNamespace(
                    id="meal-event", peer_id="ohmo", session_id=session,
                    created_at=created, metadata={"decision_trace": {"annotations": {"nutrition": {}}}},
                ),
                SimpleNamespace(
                    id="ordinary-event", peer_id="ohmo", session_id=session,
                    created_at=created, metadata={"role": "assistant", "kind": "text"},
                ),
            ]

    rows = await honcho_history_snapshot(
        Client(), session="synthetic-session", since=created.replace(hour=10), until=created
    )
    assert [row["id"] for row in rows] == ["meal-event", "ordinary-event"]
    assert rows[1]["metadata"] == {"role": "assistant", "kind": "text"}
    assert all("content" not in row for row in rows)


def test_clear_negative_requires_no_new_nutrition_rows():
    before = [{"id": "old", "metadata": {"decision_trace": {"annotations": {"nutrition": {}}}}}]
    after = before + [{"id": "ordinary", "metadata": {"kind": "message"}}]
    assert new_nutrition_event_ids(before, after) == []
    after.append({
        "id": "new-meal",
        "metadata": {"decision_trace": {"annotations": {"nutrition": {"record_type": "meal_observation"}}}},
    })
    assert new_nutrition_event_ids(before, after) == ["new-meal"]


def test_projected_kcal_is_intake_only():
    assert projected_intake_kcal([
        {"energy_kcal_best": 240},
        {"energy_kcal_best": 85},
        {"energy_kcal_best": None},
    ]) == 325


def test_runtime_episode_export_marks_missing_ids_explicitly():
    class Store:
        def get_episode(self, episode_id):
            return None

        def iter_events(self, episode_id):
            return iter(())

    assert runtime_episode_evidence(Store(), ["episode-missing"]) == [{
        "episode_id": "episode-missing",
        "episode": None,
        "events": [],
        "missing": True,
    }]


@pytest.mark.asyncio
async def test_offline_person_source_negative_is_complete_no_tool_finalization():
    api = OfflinePersonSourceApi(outcome="nonfood", basis="text")
    events = [event async for event in api.stream_message(SimpleNamespace())]
    assert len(events) == 1
    assert events[0].stop_reason == "end_turn"
    assert all(block.type != "tool_use" for block in events[0].message.content)


@pytest.mark.asyncio
async def test_offline_person_source_food_proposes_finalizer_and_receipt_is_required():
    api = OfflinePersonSourceApi(outcome="meal", basis="text")
    event = [item async for item in api.stream_message(SimpleNamespace())][0]
    proposal = next(block for block in event.message.content if isinstance(block, ToolUseBlock))
    assert event.stop_reason == "tool_use"
    assert proposal.name == "trace"
    assert proposal.input["kind"] == "trace_finalization"
    assert proposal.input["payload"]["annotations"]["nutrition"]["basis"] == ["text"]
    assert source_run_status([{"event_id": "honcho-event-1"}])["saved"] is True
    assert source_run_status([{"event_id": None}])["saved"] is False


@pytest.mark.asyncio
async def test_offline_person_source_meal_finalizes_with_text_after_one_trace():
    api = OfflinePersonSourceApi(outcome="meal", basis="image")
    request = SimpleNamespace()

    proposal = [event async for event in api.stream_message(request)][0]
    trace_call = next(
        block for block in proposal.message.content if isinstance(block, ToolUseBlock)
    )
    nutrition = trace_call.input["payload"]["annotations"]["nutrition"]
    assert proposal.stop_reason == "tool_use"
    assert nutrition["energy_kcal_best"] == 125
    assert nutrition["basis"] == ["image"]

    final = [event async for event in api.stream_message(request)][0]
    assert final.stop_reason == "end_turn"
    assert "recorded" in final.message.text
    assert all(block.type != "tool_use" for block in final.message.content)
    assert api.calls == 2


@pytest.mark.asyncio
async def test_person_source_turn_uses_real_bridge_and_preserves_unthreaded_source(
    tmp_path, monkeypatch
):
    import ohmo.evals.adapter as eval_adapter
    import ohmo.gateway.config as gateway_config
    import ohmo.gateway.runtime as gateway_runtime
    from ohmo.gateway.runtime import GatewayStreamUpdate

    message = person_source_message(
        sender_id="42",
        chat_id="42",
        source_message_id="1001",
        sent_at=datetime(2026, 10, 5, 11, tzinfo=timezone.utc),
        text="Tea with lunch",
    )

    class Episode:
        def model_dump(self, mode):
            return {"episode_id": "episode-1"}

    class Store:
        def __init__(self):
            self.ids = []

        def list_episode_ids(self):
            return list(self.ids)

        def get_episode(self, episode_id):
            return Episode() if episode_id in self.ids else None

        def iter_events(self, episode_id):
            return iter(())

    store = Store()
    monkeypatch.setattr(eval_adapter, "get_eval_store", lambda _root: store)
    monkeypatch.setattr(gateway_config, "save_gateway_config", lambda *_args: None)
    monkeypatch.setattr(
        "openharness.config.paths.get_config_file_path", lambda: tmp_path / "settings.json"
    )
    monkeypatch.setattr(
        "camera_runtime_support.disable_external_runtime_surfaces", lambda *_args, **_kw: None
    )

    class RuntimePool:
        def __init__(self, **_kwargs):
            self.ordered = []

        async def get_bundle(self, session_key, latest_user_prompt=None):
            self.ordered.append("initialized")
            assert session_key == "telegram:42"
            assert latest_user_prompt == message.content

        async def stream_message(self, inbound, session_key):
            assert inbound is message
            assert session_key == "telegram:42"
            assert "reply_to_message_id" not in inbound.metadata
            assert inbound.timestamp == message.timestamp
            yield GatewayStreamUpdate(
                kind="final",
                text="Synthetic finalizer fixture completed.",
                metadata={"nutrition_append_event_id": "synthetic-event-id"},
            )
            store.ids.append("episode-1")

        async def aclose(self):
            return None

    monkeypatch.setattr(gateway_runtime, "OhmoSessionRuntimePool", RuntimePool)
    before_calls = []

    async def before_turn(cutoff):
        before_calls.append(cutoff)
        return [{"id": "non-nutrition-before", "metadata": {"kind": "ordinary"}}]

    result = await run_person_source_turn(
        root=tmp_path,
        message=message,
        owner_id="synthetic_owner",
        honcho_url="http://unused.invalid",
        workspace="synthetic-workspace",
        session="synthetic-session",
        bot_client=SimpleNamespace(synthetic=True),
        native_mode=False,
        before_turn=before_turn,
    )
    assert result["save"] == {
        "saved": True,
        "event_id": "synthetic-event-id",
        "event_ids": ["synthetic-event-id"],
        "ambiguous": False,
    }
    assert result["source_message_id"] == "1001"
    assert len(before_calls) == 1
    assert result["runtime_episodes"][0]["episode_id"] == "episode-1"
    assert result["deliveries"][0]["delivery_message_ids"]
