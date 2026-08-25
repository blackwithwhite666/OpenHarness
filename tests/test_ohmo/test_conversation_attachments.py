from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ohmo.attachment_store import AttachmentStore
from ohmo.conversation_image_tool import LoadConversationImageInput, LoadConversationImageTool
from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.session_storage import OhmoSessionBackend, get_session_dir
from ohmo.workspace import initialize_workspace
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import AttachmentRefBlock, ConversationMessage, ImageBlock
from openharness.tools.base import ToolExecutionContext
from openharness.tools.base import ToolRegistry


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _inline_image() -> ImageBlock:
    return ImageBlock(
        media_type="image/png",
        data=base64.b64encode(PNG_BYTES).decode("ascii"),
        source_path="/untrusted/original/name.png",
    )


def _object_files(workspace: Path) -> list[Path]:
    return sorted((workspace / "attachments" / "objects").rglob("*.bin"))


def test_ohmo_snapshot_externalizes_inline_images_and_deduplicates(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    backend = OhmoSessionBackend(workspace)
    message = ConversationMessage(
        role="user",
        event_id="evt-1",
        content=[_inline_image(), _inline_image()],
    )

    backend.save_snapshot(
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="system",
        messages=[message],
        usage=UsageSnapshot(),
        session_id="sid",
        session_key="telegram:42",
    )

    raw = json.loads((get_session_dir(workspace) / "latest.json").read_text())
    blocks = raw["messages"][0]["content"]
    assert [block["type"] for block in blocks] == ["attachment_ref", "attachment_ref"]
    assert blocks[0]["attachment_id"] == blocks[1]["attachment_id"]
    assert '"data":' not in json.dumps(raw)
    assert base64.b64encode(PNG_BYTES).decode("ascii") not in json.dumps(raw)
    assert len(_object_files(workspace)) == 1


def test_legacy_inline_snapshot_load_migrates_equal_payloads_to_one_object(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    backend = OhmoSessionBackend(workspace)
    encoded = base64.b64encode(PNG_BYTES).decode("ascii")
    legacy = {
        "app": "ohmo",
        "session_id": "legacy",
        "session_key": "telegram:42",
        "cwd": str(tmp_path),
        "model": "gpt-5.5",
        "system_prompt": "system",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "media_type": "image/png", "data": encoded},
                    {"type": "image", "media_type": "image/png", "data": encoded},
                ],
            }
        ],
        "usage": {},
        "tool_metadata": {},
        "created_at": 1,
    }
    latest = get_session_dir(workspace) / "latest.json"
    latest.write_text(json.dumps(legacy), encoding="utf-8")

    loaded = backend.load_latest(tmp_path)

    assert loaded is not None
    blocks = loaded["messages"][0]["content"]
    assert [block["type"] for block in blocks] == ["attachment_ref", "attachment_ref"]
    assert blocks[0]["attachment_id"] == blocks[1]["attachment_id"]
    assert all("data" not in block for block in blocks)
    assert len(_object_files(workspace)) == 1

    restored = [ConversationMessage.model_validate(item) for item in loaded["messages"]]
    backend.save_snapshot(
        cwd=tmp_path,
        model="gpt-5.5",
        system_prompt="system",
        messages=restored,
        usage=UsageSnapshot(),
        session_id="legacy",
    )
    assert latest.stat().st_size < 4096


@pytest.mark.parametrize(
    "attachment_id",
    ["../escape", "/tmp/escape", "a" * 63, "A" * 64, "0" * 64],
)
def test_attachment_store_invalid_or_missing_ids_fail_closed(
    tmp_path: Path,
    attachment_id: str,
) -> None:
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    store = AttachmentStore(workspace)

    with pytest.raises((FileNotFoundError, ValueError)):
        store.load_image(attachment_id)

    assert not (tmp_path / "escape").exists()


@pytest.mark.asyncio
async def test_load_conversation_image_returns_trusted_transient_content_only(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    store = AttachmentStore(workspace)
    ref = store.ingest_bytes(PNG_BYTES, media_type="image/png", label="meal.png")
    tool = LoadConversationImageTool(
        store,
        is_attachment_allowed=lambda attachment_id: attachment_id == ref.attachment_id,
    )

    result = await tool.execute(
        LoadConversationImageInput(attachment_id=ref.attachment_id),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is False
    assert ref.attachment_id in result.output
    transient = result.metadata["_openharness_transient_image"]
    assert isinstance(transient, ImageBlock)
    assert transient.data == base64.b64encode(PNG_BYTES).decode("ascii")
    assert "data" not in {key for key in result.metadata if not key.startswith("_")}


@pytest.mark.asyncio
async def test_load_conversation_image_rejects_nonexistent_id(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    tool = LoadConversationImageTool(
        AttachmentStore(workspace),
        is_attachment_allowed=lambda _attachment_id: True,
    )

    result = await tool.execute(
        LoadConversationImageInput(attachment_id="0" * 64),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert result.is_error is True
    assert result.output == "Conversation image unavailable."
    assert "_openharness_transient_image" not in result.metadata


@pytest.mark.asyncio
async def test_existing_unreferenced_attachment_is_indistinguishable_from_missing(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    store = AttachmentStore(workspace)
    ref = store.ingest_bytes(PNG_BYTES, media_type="image/png", label="private.png")
    tool = LoadConversationImageTool(
        store,
        is_attachment_allowed=lambda _attachment_id: False,
    )

    unauthorized = await tool.execute(
        LoadConversationImageInput(attachment_id=ref.attachment_id),
        ToolExecutionContext(cwd=tmp_path),
    )
    missing = await tool.execute(
        LoadConversationImageInput(attachment_id="0" * 64),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert unauthorized.is_error is True
    assert unauthorized.output == missing.output == "Conversation image unavailable."
    assert ref.attachment_id not in unauthorized.output


@pytest.mark.parametrize(
    "messages",
    [[], [ConversationMessage.from_user_text("restored text-only turn")]],
    ids=["new", "restored"],
)
def test_conversation_image_tool_is_absent_for_ref_free_bundles(
    tmp_path: Path,
    messages: list[ConversationMessage],
) -> None:
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._attachment_store = AttachmentStore(workspace)
    bundle = SimpleNamespace(
        engine=SimpleNamespace(messages=messages),
        tool_registry=ToolRegistry(),
    )

    pool._register_conversation_image_tool(bundle)

    assert bundle.tool_registry.get("load_conversation_image") is None


@pytest.mark.asyncio
async def test_conversation_image_tool_active_turn_allowance_is_torn_down(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    store = AttachmentStore(workspace)
    ref = store.ingest_bytes(PNG_BYTES, media_type="image/png", label="new.png")
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._attachment_store = store
    bundle = SimpleNamespace(
        engine=SimpleNamespace(messages=[]),
        tool_registry=ToolRegistry(),
    )
    current_message = ConversationMessage(role="user", content=[ref, _inline_image()])

    pool._register_conversation_image_tool(bundle, current_message=current_message)
    active_tool = bundle.tool_registry.get("load_conversation_image")
    assert isinstance(active_tool, LoadConversationImageTool)
    active_result = await active_tool.execute(
        LoadConversationImageInput(attachment_id=ref.attachment_id),
        ToolExecutionContext(cwd=tmp_path),
    )
    assert active_result.is_error is False

    pool._register_conversation_image_tool(bundle)

    assert bundle.tool_registry.get("load_conversation_image") is None
    stale_result = await active_tool.execute(
        LoadConversationImageInput(attachment_id=ref.attachment_id),
        ToolExecutionContext(cwd=tmp_path),
    )
    assert stale_result.is_error is True
    assert stale_result.output == "Conversation image unavailable."


@pytest.mark.asyncio
async def test_two_bundles_sharing_store_cannot_load_each_others_attachment(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / ".ohmo-home")
    store = AttachmentStore(workspace)
    ref_a = store.ingest_bytes(PNG_BYTES + b"a", media_type="image/png", label="a.png")
    ref_b = store.ingest_bytes(PNG_BYTES + b"b", media_type="image/png", label="b.png")
    pool = object.__new__(OhmoSessionRuntimePool)
    pool._attachment_store = store
    bundle_a = SimpleNamespace(
        engine=SimpleNamespace(
            messages=[ConversationMessage(role="user", content=[ref_a])]
        ),
        tool_registry=ToolRegistry(),
    )
    bundle_b = SimpleNamespace(
        engine=SimpleNamespace(
            messages=[ConversationMessage(role="user", content=[ref_b])]
        ),
        tool_registry=ToolRegistry(),
    )
    pool._register_conversation_image_tool(bundle_a)
    pool._register_conversation_image_tool(bundle_b)
    tool_a = bundle_a.tool_registry.get("load_conversation_image")
    tool_b = bundle_b.tool_registry.get("load_conversation_image")
    assert isinstance(tool_a, LoadConversationImageTool)
    assert isinstance(tool_b, LoadConversationImageTool)

    own = await tool_a.execute(
        LoadConversationImageInput(attachment_id=ref_a.attachment_id),
        ToolExecutionContext(cwd=tmp_path),
    )
    cross_a = await tool_a.execute(
        LoadConversationImageInput(attachment_id=ref_b.attachment_id),
        ToolExecutionContext(cwd=tmp_path),
    )
    cross_b = await tool_b.execute(
        LoadConversationImageInput(attachment_id=ref_a.attachment_id),
        ToolExecutionContext(cwd=tmp_path),
    )

    assert own.is_error is False
    assert cross_a.is_error is True
    assert cross_b.is_error is True
    assert cross_a.output == cross_b.output == "Conversation image unavailable."


def test_attachment_ref_label_is_bounded_and_contains_no_inline_payload() -> None:
    ref = AttachmentRefBlock(
        attachment_id="a" * 64,
        media_type="image/png",
        byte_size=123,
        label="x" * 10_000,
    )

    dumped = ref.model_dump(mode="json")
    assert len(ref.label) <= 160
    assert set(dumped) == {"type", "attachment_id", "media_type", "byte_size", "label"}
    assert "data" not in dumped


def test_anthropic_wire_renders_attachment_ref_as_text_placeholder() -> None:
    message = ConversationMessage(
        role="user",
        content=[
            AttachmentRefBlock(
                attachment_id="c" * 64,
                media_type="image/png",
                byte_size=123,
                label="meal.png",
            )
        ],
    )

    wire = message.to_api_param()

    assert wire["content"] == [
        {
            "type": "text",
            "text": "[conversation image attachment_id=" + "c" * 64 + " label=meal.png]",
        }
    ]
