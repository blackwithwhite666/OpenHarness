from openharness.channels.impl.telegram import _markdown_to_telegram_html


def test_markdown_table_renders_aligned_monospace_block():
    md = (
        "Вот календарь:\n"
        "| Приоритет | Встреча | Окно |\n"
        "|---|---|---|\n"
        "| 4 | 1-1 Вася : Дима | 2026-06-08 → 2026-06-12 |\n"
        "| 2 | 1-1 Кирилл : Дима | 2026-06-08 → 2026-06-12 |\n"
    )
    html = _markdown_to_telegram_html(md)
    assert "<pre><code>" in html            # aligned monospace block, not raw pipes
    for token in ("Приоритет", "Встреча", "Окно", "1-1 Кирилл : Дима"):
        assert token in html
    assert "|---|" not in html               # raw md separator gone
    assert "Вот календарь:" in html          # surrounding prose preserved


def test_non_table_pipe_text_is_left_alone():
    html = _markdown_to_telegram_html("run a | b in the shell")
    assert "<pre>" not in html
    assert "a | b" in html


def test_table_cell_underscores_not_bold_mangled():
    html = _markdown_to_telegram_html("| a | b |\n|---|---|\n| foo__bar__baz | x |\n")
    assert "foo__bar__baz" in html
    assert "<b>" not in html
