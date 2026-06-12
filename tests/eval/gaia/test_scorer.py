"""Offline tests for the GAIA scorer + answer extraction.

Zero network, zero model. Runs in the default pytest gate (NOT marked
``eval``) — only the live-web runner is ``@pytest.mark.eval``.

Covers:
  * the canonical (model_answer, ground_truth) -> expected pairs from the
    reference brief, asserted against the STRICT (official) scorer;
  * adversarial cases the ADR §4 calls out (empty/None, reasoning-prefix, the
    EU-comma trio, comma-inside-a-list-element, trailing units, commentary-
    wrapped, case/whitespace);
  * the answer-extraction layer (sentinel present/absent, fallback heuristics);
  * the deliberate ``strict=False`` extensions, tested separately so we never
    conflate them with GAIA semantics;
  * the offline runner helpers (wilson_ci, JSONL/REPORT writers) from
    run_subset.py.
"""

from __future__ import annotations

import json
import math

import pytest

from tests.eval.gaia import run_subset
from tests.eval.gaia.scorer import (
    extract_answer,
    normalize_number_str,
    normalize_str,
    question_scorer,
    split_string,
)

# --------------------------------------------------------------------------- #
# Canonical pairs from the reference brief (STRICT == official GAIA scorer).
# (model_answer, ground_truth, expected, label)
# --------------------------------------------------------------------------- #
CANONICAL_PAIRS = [
    ("42", "42.0", True, "number 42.0==42.0"),
    ("$1,234.50", "1234.5", True, "number strips $ and ,"),
    ("17%", "17", True, "number strips %"),
    ("100,5", "100.5", False, "EU comma wart: comma stripped -> 1005 != 100.5"),
    ("Sea Gull", "seagull", True, "string ws-removed + lowercased"),
    ("the answer is 42", "42", False, "number branch; commentary -> inf"),
    ("apple, banana, cherry", "apple,banana,cherry", True, "list 3==3 string-eq"),
    ("cherry, banana, apple", "apple,banana,cherry", False, "list order-sensitive"),
    ("a, b, 1234", "a, b, 1,234", False, "GT 1,234 splits to 4 vs MA 3"),
    (None, "42", False, "None->'None'->number inf != 42"),
    ("42 km", "42", False, "units NOT stripped in strict -> inf"),
    (
        "St. Petersburg, Russia",
        "St. Petersburg, Russia",
        True,
        "list per-elem string-eq, remove_punct=False keeps the '.'",
    ),
]


@pytest.mark.parametrize(
    "model_answer,ground_truth,expected,label",
    CANONICAL_PAIRS,
    ids=[p[3] for p in CANONICAL_PAIRS],
)
def test_canonical_pairs_strict(model_answer, ground_truth, expected, label):
    # The comma-mismatch + EU rows legitimately emit a UserWarning; allow it.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert question_scorer(model_answer, ground_truth, strict=True) is expected, label


# --------------------------------------------------------------------------- #
# normalize_number_str (strict / official)
# --------------------------------------------------------------------------- #
def test_normalize_number_strict_strips_currency_percent_comma():
    assert normalize_number_str("$1,234.50") == 1234.5
    assert normalize_number_str("17%") == 17.0
    assert normalize_number_str("1,000,000") == 1000000.0


def test_normalize_number_strict_eu_comma_wart():
    # The known wart: 100,5 -> 1005.0, NOT 100.5.
    assert normalize_number_str("100,5") == 1005.0


def test_normalize_number_strict_unparseable_is_inf():
    assert normalize_number_str("42 km") == float("inf")
    assert normalize_number_str("the answer is 42") == float("inf")


def test_number_exact_no_tolerance():
    assert question_scorer("42.000001", "42") is False
    assert question_scorer("42.0", "42") is True


# --------------------------------------------------------------------------- #
# normalize_str
# --------------------------------------------------------------------------- #
def test_normalize_str_removes_all_whitespace_and_lowercases():
    assert normalize_str("Sea Gull") == "seagull"
    assert normalize_str("  HeLLo\tWorld\n") == "helloworld"


def test_normalize_str_remove_punct_default_true():
    assert normalize_str("a.b,c!") == "abc"


def test_normalize_str_keeps_punct_when_false():
    assert normalize_str("St. Petersburg", remove_punct=False) == "st.petersburg"


def test_normalize_str_keeps_articles():
    # No article removal — "the" survives.
    assert normalize_str("the answer") == "theanswer"


# --------------------------------------------------------------------------- #
# split_string (strict)
# --------------------------------------------------------------------------- #
def test_split_string_strict_splits_every_delimiter():
    assert split_string("a,b;c") == ["a", "b", "c"]
    # Comma-inside-element trap: 1,234 -> two elements.
    assert split_string("a, 1,234") == ["a", " 1", "234"]


# --------------------------------------------------------------------------- #
# String branch / list branch behaviour
# --------------------------------------------------------------------------- #
def test_string_case_and_whitespace_insensitive():
    assert question_scorer("  HELLO  world ", "hello world") is True


def test_list_length_mismatch_warns_and_fails():
    with pytest.warns(UserWarning):
        assert question_scorer("a, b", "a, b, c") is False


def test_list_element_keeps_punctuation():
    # remove_punct=False inside lists keeps the period.
    assert question_scorer("U.S.A, NATO", "U.S.A, NATO") is True


def test_list_mixed_number_and_string_elements():
    assert question_scorer("apple, 42", "apple, 42.0") is True


# --------------------------------------------------------------------------- #
# Adversarial: empty / None / commentary-wrapped
# --------------------------------------------------------------------------- #
def test_none_and_empty_are_wrong():
    assert question_scorer(None, "42") is False
    assert question_scorer("", "42") is False
    assert question_scorer("", "hello") is False


def test_reasoning_prefix_is_wrong_in_number_branch():
    assert question_scorer("After analysis, the answer is 42", "42") is False


def test_commentary_wrapped_string_is_wrong():
    # String GT; commentary tokens survive whitespace removal -> mismatch.
    assert (
        question_scorer("I believe the answer is Paris", "Paris") is False
    )


# --------------------------------------------------------------------------- #
# Adversarial: the EU-comma trio (strict semantics)
# --------------------------------------------------------------------------- #
def test_eu_comma_trio_strict():
    # "100,000" -> 100000 ; "100000" GT parses to 100000 -> PASS
    assert question_scorer("100,000", "100000", strict=True) is True
    # "100.000" -> 100.0 (dot is decimal) vs GT 100000 -> FAIL
    assert question_scorer("100.000", "100000", strict=True) is False
    # EU decimal comma silently mishandled (the wart).
    assert question_scorer("100,5", "100.5", strict=True) is False


# --------------------------------------------------------------------------- #
# strict=False extensions (OUR layer — NOT GAIA). Tested separately.
# --------------------------------------------------------------------------- #
def test_lenient_eu_decimal_comma():
    assert question_scorer("100,5", "100.5", strict=False) is True
    assert normalize_number_str("100,5", strict=False) == 100.5
    # EU thousands+decimal: 1.234,56 -> 1234.56
    assert normalize_number_str("1.234,56", strict=False) == 1234.56
    # US thousands+decimal still works.
    assert normalize_number_str("1,234.56", strict=False) == 1234.56
    # Bare thousands comma stays thousands.
    assert normalize_number_str("100,000", strict=False) == 100000.0


def test_lenient_trailing_units_stripped():
    assert question_scorer("42 km", "42", strict=False) is True
    assert question_scorer("3.5kg", "3.5", strict=False) is True
    assert normalize_number_str("$42 USD", strict=False) == 42.0


def test_lenient_list_does_not_missplit_thousands():
    # The strict scorer fails this (length mismatch); lenient protects 1,234.
    assert question_scorer("apple, 1,234", "banana, 1,234", strict=False) is False
    assert question_scorer("apple, 1,234", "apple, 1,234", strict=False) is True
    assert split_string("a, 1,234, b", strict=False) == ["a", "1,234", "b"]


def test_lenient_does_not_break_strict_passes():
    # Everything that passes strict and isn't a wart should still pass lenient.
    assert question_scorer("$1,234.50", "1234.5", strict=False) is True
    assert question_scorer("Sea Gull", "seagull", strict=False) is True
    assert question_scorer("apple, banana", "apple, banana", strict=False) is True


# --------------------------------------------------------------------------- #
# OWL-derived fixtures (equivalents derived from the official spec).
# OWL's question_scorer is logic-identical to the HF scorer; these mirror the
# example shapes used in camel-ai/owl's GAIA utils.
# --------------------------------------------------------------------------- #
OWL_FIXTURES = [
    ("3", "3", True),
    ("3.0", "3", True),
    ("Hello World", "hello world", True),
    ("a, b, c", "a, b, c", True),
    ("a, c, b", "a, b, c", False),
    ("1, 2, 3", "1, 2, 3", True),
    ("wrong", "right", False),
]


@pytest.mark.parametrize("ma,gt,expected", OWL_FIXTURES)
def test_owl_derived_fixtures(ma, gt, expected):
    assert question_scorer(ma, gt, strict=True) is expected


# --------------------------------------------------------------------------- #
# extract_answer
# --------------------------------------------------------------------------- #
def test_extract_sentinel_primary():
    text = "blah blah\n<final_answer>42</final_answer>\ntrailing"
    assert extract_answer(text) == "42"


def test_extract_sentinel_dotall_multiline():
    text = "<final_answer>line one\nline two</final_answer>"
    assert extract_answer(text) == "line one\nline two"


def test_extract_sentinel_last_wins():
    text = "<final_answer>old</final_answer> ... <final_answer>new</final_answer>"
    assert extract_answer(text) == "new"


def test_extract_label_fallback():
    text = "Reasoning...\nFINAL ANSWER: Paris"
    assert extract_answer(text) == "Paris"


def test_extract_label_strips_markdown():
    text = "stuff\n**FINAL ANSWER:** `Paris`"
    assert extract_answer(text) == "Paris"


def test_extract_label_requires_standalone_token():
    # "semifinal answer:" must NOT be read as the label (no word boundary bug);
    # with no real label/sentinel it falls through to the last-line heuristic.
    assert extract_answer("blah\nsemifinal answer: nonsense") == "semifinal answer: nonsense"
    # "finalanswer:" (no space) is not the label either.
    assert extract_answer("x\nfinalanswer: nope") == "finalanswer: nope"
    # a genuine label on a later line still wins over earlier noise.
    assert extract_answer("semifinal answer: trap\nFINAL ANSWER: 42") == "42"


def test_extract_last_line_fallback():
    # No sentinel, no label -> the whole last non-empty line is returned
    # verbatim (the fallback does not try to parse prose). Trailing blank
    # lines / markdown wrappers are stripped.
    assert extract_answer("reasoning...\nBerlin\n\n") == "Berlin"
    assert extract_answer("reasoning...\n`Berlin`") == "Berlin"
    # A prose last line comes back whole — deliberately not magically parsed.
    assert extract_answer("I reasoned.\nThe city is Berlin") == "The city is Berlin"


def test_extract_empty_sentinel_yields_empty_not_none():
    # Agent emitted a (blank) answer; scorer then marks it wrong.
    assert extract_answer("<final_answer></final_answer>") == ""
    assert question_scorer(extract_answer("<final_answer></final_answer>"), "42") is False


def test_extract_none_on_empty_input():
    assert extract_answer("") is None
    assert extract_answer("   \n  \n") is None
    assert extract_answer(None) is None


def test_extract_then_score_end_to_end():
    transcript = "Let me think.\n<final_answer>$1,234.50</final_answer>"
    assert question_scorer(extract_answer(transcript), "1234.5") is True


# --------------------------------------------------------------------------- #
# Offline runner helpers (wilson_ci + JSONL/REPORT writers).
# --------------------------------------------------------------------------- #
def test_wilson_ci_known_values():
    lo, hi = run_subset.wilson_ci(0, 0)
    assert (lo, hi) == (0.0, 0.0)

    lo, hi = run_subset.wilson_ci(10, 10)
    assert 0.0 <= lo <= 1.0 and hi == pytest.approx(1.0, abs=1e-9)
    assert lo > 0.6  # 10/10 lower bound is well above half

    lo, hi = run_subset.wilson_ci(5, 10)
    # Symmetric around 0.5, classic Wilson 95% CI for 5/10 ~ (0.237, 0.763).
    assert lo == pytest.approx(0.2366, abs=2e-3)
    assert hi == pytest.approx(0.7634, abs=2e-3)
    assert lo < 0.5 < hi


def test_wilson_ci_bounds_are_within_unit_interval():
    for k, n in [(0, 5), (1, 3), (3, 7), (30, 30)]:
        lo, hi = run_subset.wilson_ci(k, n)
        assert 0.0 <= lo <= hi <= 1.0
        assert not math.isnan(lo) and not math.isnan(hi)


def test_wilson_ci_rejects_bad_input():
    with pytest.raises(ValueError):
        run_subset.wilson_ci(5, 3)
    with pytest.raises(ValueError):
        run_subset.wilson_ci(-1, 3)


def test_write_jsonl_roundtrip(tmp_path):
    rows = [
        {"task_id": "t1", "level": 1, "score": 1.0, "k": 3},
        {"task_id": "t2", "level": 2, "score": 0.0, "k": 3},
    ]
    path = tmp_path / "results" / "abc123.jsonl"
    run_subset.write_jsonl(path, rows)
    assert path.exists()
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded == rows


def test_write_report_contains_cis_and_levels(tmp_path):
    rows = [
        {"task_id": "t1", "level": 1, "score": 1.0},
        {"task_id": "t2", "level": 1, "score": 0.0},
        {"task_id": "t3", "level": 2, "score": 1.0},
    ]
    path = tmp_path / "REPORT.md"
    run_subset.write_report(path, rows, git_sha="deadbeef")
    text = path.read_text()
    assert "deadbeef" in text
    assert "Level 1" in text and "Level 2" in text
    assert "Overall" in text
    # Wilson CI columns rendered as bracketed bounds.
    assert "[" in text and "]" in text
    # Overall accuracy = 2/3.
    assert "0.67" in text or "0.667" in text


def test_aggregate_by_level():
    rows = [
        {"level": 1, "score": 1.0},
        {"level": 1, "score": 0.0},
        {"level": 2, "score": 1.0},
        {"level": 3, "score": 0.0},
    ]
    agg = run_subset.aggregate(rows)
    assert agg[1]["n"] == 2 and agg[1]["k"] == 1
    assert agg[2]["n"] == 1 and agg[2]["k"] == 1
    assert agg[3]["n"] == 1 and agg[3]["k"] == 0
    assert agg["overall"]["n"] == 4 and agg["overall"]["k"] == 2
