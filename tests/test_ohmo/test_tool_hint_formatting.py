from __future__ import annotations

from ohmo.gateway.runtime import (
    _CHANNEL_THINKING_PHRASES_EN,
    _format_channel_progress,
    _format_tool_args_block,
    _format_tool_done,
    _format_tool_result_block,
    _pretty_tool_name,
    _short_call_id,
)
from openharness.channels.impl.telegram import _markdown_to_telegram_html


def test_short_call_id_pairs_start_and_result():
    # Same tool_use id -> same short tag on the start hint and the result hint.
    cid = _short_call_id("toolu_01AbCdEfGh")
    assert cid == "EfGh"
    assert _short_call_id("toolu_01AbCdEfGh") == cid  # stable
    assert _short_call_id("") == ""
    assert _short_call_id(None) == ""  # type: ignore[arg-type]


def test_format_tool_done_includes_call_id_to_match_start():
    done = _format_tool_done("bash", "ok", is_error=False, call_id="toolu_zzzzWXYZ")
    assert done.startswith("Bash — WXYZ ✅")  # name — id mark, matches '🛠️ Bash — WXYZ'
    no_id = _format_tool_done("bash", "ok", is_error=False)
    assert no_id.startswith("Bash ✅")  # unchanged when no id


def test_format_tool_result_block_truncates_like_params():
    assert _format_tool_result_block("") == ""
    assert _format_tool_result_block("  ") == ""
    assert _format_tool_result_block("ok") == "```\nok\n```"
    long = "x" * 700
    block = _format_tool_result_block(long)
    assert block.endswith("\n…\n```")
    assert "x" * 600 in block and "x" * 601 not in block  # clipped at 600


def test_format_tool_done_success_and_error():
    ok = _format_tool_done("bash", "hello\nworld", is_error=False)
    assert ok.startswith("Bash ✅")
    assert "```\nhello\nworld\n```" in ok

    bad = _format_tool_done("mcp__worfalomey__get_time", "boom", is_error=True)
    assert bad.startswith("Get time ❌")
    assert "boom" in bad


def test_format_tool_done_no_output_is_just_name_and_mark():
    assert _format_tool_done("bash", "", is_error=False) == "Bash ✅"


def test_pretty_tool_name_strips_mcp_prefix_and_humanizes():
    assert _pretty_tool_name("mcp__worfalomey__read_calendar_event") == "Read calendar event"
    assert _pretty_tool_name("mcp__google_search__google_search") == "Google search"
    assert _pretty_tool_name("bash") == "Bash"
    assert _pretty_tool_name("list_merged_events") == "List merged events"


def test_format_tool_args_single_value_is_plain_code_block():
    assert _format_tool_args_block({"link": "https://x/event/1"}) == "```\nhttps://x/event/1\n```"
    assert _format_tool_args_block({"command": "ls -la"}) == "```bash\nls -la\n```"


def test_format_tool_args_multi_is_pretty_json_block():
    block = _format_tool_args_block({"b": 2, "a": 1})
    assert block.startswith("```json\n")
    assert '"a": 1' in block and '"b": 2' in block  # sorted + indented


def test_format_tool_args_empty():
    assert _format_tool_args_block({}) == ""
    assert _format_tool_args_block(None) == ""


def test_rendered_hint_does_not_mangle_underscores():
    """Regression: mcp__server__tool used to render as bold-mangled
    'mcpserveTool' because the markdown converter reads __x__ as bold."""
    name = _pretty_tool_name("mcp__worfalomey__read_calendar_event")
    args = _format_tool_args_block({"link": "https://x/event/1?a__b__c=1"})
    html = _markdown_to_telegram_html(f"\U0001f6e0️ {name}\n{args}")

    assert "Read calendar event" in html
    assert "<pre><code>" in html  # args rendered as a code block
    assert "a__b__c=1" in html  # underscores preserved inside the code
    assert "<b>" not in html  # nothing got bolded by the __ regex


def _progress(kind, text, content="can you help"):
    return _format_channel_progress(
        channel="telegram", kind=kind, text=text, session_key="s", content=content
    )


def test_format_channel_progress_tool_hint_adds_wrench():
    assert _progress("tool_hint", "Read file") == "🛠️ Read file"


def test_format_channel_progress_status_adds_bubble():
    assert _progress("status", "syncing") == "🫧 syncing"


def test_format_channel_progress_thinking_picks_english_phrase():
    assert _progress("thinking", "") in _CHANNEL_THINKING_PHRASES_EN


def test_format_channel_progress_image_fallback_has_icon():
    assert "🖼️" in _progress("image_fallback", "")
