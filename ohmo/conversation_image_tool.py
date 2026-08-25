"""Read-only model tool for reopening one durable conversation image."""

from __future__ import annotations

import base64
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from ohmo.attachment_store import AttachmentStore
from openharness.engine.messages import ImageBlock
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


_UNAVAILABLE_MESSAGE = "Conversation image unavailable."


class LoadConversationImageInput(BaseModel):
    """Strict content-addressed identifier; never accepts a filesystem path."""

    model_config = ConfigDict(extra="forbid")

    attachment_id: str = Field(pattern=r"^[0-9a-f]{64}$")


class LoadConversationImageTool(BaseTool):
    """Load one historical image into only the active provider query."""

    name = "load_conversation_image"
    description = (
        "Reopen one image from conversation history by its exact attachment_id. "
        "This is read-only and accepts no path."
    )
    input_model = LoadConversationImageInput

    def __init__(
        self,
        store: AttachmentStore,
        *,
        is_attachment_allowed: Callable[[str], bool],
    ) -> None:
        self._store = store
        self._is_attachment_allowed = is_attachment_allowed

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True

    async def execute(
        self,
        arguments: LoadConversationImageInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        try:
            allowed = self._is_attachment_allowed(arguments.attachment_id)
        except Exception:
            allowed = False
        if not allowed:
            return ToolResult(output=_UNAVAILABLE_MESSAGE, is_error=True)
        try:
            stored = self._store.load_image(arguments.attachment_id)
        except (FileNotFoundError, ValueError):
            return ToolResult(output=_UNAVAILABLE_MESSAGE, is_error=True)
        ref = stored.ref
        return ToolResult(
            output=(
                "Loaded conversation image "
                f"{ref.attachment_id} ({ref.media_type}, {ref.byte_size} bytes)."
            ),
            metadata={
                "attachment_id": ref.attachment_id,
                "media_type": ref.media_type,
                "byte_size": ref.byte_size,
                "_openharness_transient_image": ImageBlock(
                    media_type=ref.media_type,
                    data=base64.b64encode(stored.data).decode("ascii"),
                ),
            },
        )
