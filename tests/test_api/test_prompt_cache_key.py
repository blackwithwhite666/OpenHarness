from __future__ import annotations

from openharness.api.client import ApiMessageRequest
from openharness.api.codex_client import _build_codex_body
from openharness.api.openai_client import _build_openai_body


def test_api_message_request_cache_key_defaults_to_none() -> None:
    request = ApiMessageRequest(model="m", messages=[])

    assert request.cache_key is None


def test_codex_body_includes_prompt_cache_key() -> None:
    request = ApiMessageRequest(model="m", messages=[], cache_key="sess-123")

    assert _build_codex_body(request)["prompt_cache_key"] == "sess-123"


def test_codex_body_omits_prompt_cache_key_when_unset() -> None:
    request = ApiMessageRequest(model="m", messages=[])

    assert "prompt_cache_key" not in _build_codex_body(request)


def test_openai_body_includes_prompt_cache_key() -> None:
    request = ApiMessageRequest(model="m", messages=[], cache_key="sess-123")

    assert _build_openai_body(request)["prompt_cache_key"] == "sess-123"


def test_openai_body_omits_prompt_cache_key_when_unset() -> None:
    request = ApiMessageRequest(model="m", messages=[])

    assert "prompt_cache_key" not in _build_openai_body(request)
