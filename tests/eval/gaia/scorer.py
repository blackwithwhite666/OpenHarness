"""GAIA quasi-exact-match scorer + tolerant answer extraction.

Layer 3 of the deep-research architecture (ADR §2½): this is a *test driver*
that scores the answer OpenHarness produces. It is pure — zero network, zero
model, stdlib-only — and the dependency direction is one-way: the eval harness
imports/drives OpenHarness, never the reverse.

Two surfaces:

* ``question_scorer(model_answer, ground_truth) -> bool`` — the GAIA
  quasi-exact-match comparison, type-dispatched (number / list / string).
* ``extract_answer(text) -> str | None`` — pulls the final answer string out of
  an agent transcript before it ever reaches the scorer.

Provenance & faithfulness
--------------------------
The comparison logic is ported from the canonical HF leaderboard scorer
(``huggingface.co/spaces/gaia-benchmark/leaderboard/raw/main/scorer.py`` —
``question_scorer`` / ``normalize_number_str`` / ``split_string`` /
``normalize_str``), cross-checked against the OWL mirror
(``raw.githubusercontent.com/camel-ai/owl/main/owl/utils/gaia.py``). The two
agree except OWL lacks the ``model_answer is None`` guard and uses ``logger``
instead of ``print``; we keep the official guard.

We expose **two** behaviours, gated by a flag, because the official scorer has
known warts that the ADR §4 asks us to optionally fix:

* ``strict=True`` (default) — bit-for-bit the official GAIA scorer. It strips
  ``$ % ,`` before ``float()`` (so EU decimal comma ``"100,5"`` wrongly becomes
  ``1005.0``), does NOT strip trailing units (``"42 km"`` -> ``inf`` -> wrong),
  and splits list answers on every ``,``/``;`` (so ``"1,234"`` inside a list is
  mis-split). This is the function the leaderboard runs; it is the contract.
* ``strict=False`` — *our deliberate extension*, NOT GAIA. It additionally
  understands EU decimal comma, strips trailing units, and splits lists on a
  delimiter chosen so numbers-with-thousands-separators are not mis-split. Every
  divergence from the official scorer lives behind this flag and is covered by
  its own tests so we never silently claim GAIA semantics we don't have.

The official reference implementation is reproduced verbatim in
``_strict`` helpers below so the port is auditable.
"""

from __future__ import annotations

import re
import string
import warnings

__all__ = [
    "question_scorer",
    "extract_answer",
    "normalize_number_str",
    "normalize_str",
    "split_string",
]

# ASCII punctuation, exactly as the official scorer's str.maketrans table.
_PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)

# Trailing-unit stripping (extension only). A run of letters / a few unit
# symbols after the numeric body, e.g. "42 km", "3.5kg", "10°". Deliberately
# conservative: we only strip a *trailing* alpha/symbol tail, never interior
# characters, so "1e3" style scientific notation is left to float() to judge.
_TRAILING_UNIT_RE = re.compile(r"[a-zA-Z°µ%\s]+$")


def is_float(element: object) -> bool:
    """Return True iff ``float(element)`` succeeds (official helper)."""
    try:
        float(element)  # type: ignore[arg-type]
        return True
    except (ValueError, TypeError):
        return False


# --------------------------------------------------------------------------- #
# Number normalization
# --------------------------------------------------------------------------- #
def _normalize_number_strict(number_str: str) -> float:
    """Verbatim port of the official ``normalize_number_str``.

    Strips ``$ % ,`` (in that order) then ``float()``; ``inf`` on failure so a
    non-parseable answer can never equal a finite ground truth. Commas are
    stripped *unconditionally* — this is the EU-decimal-comma wart.
    """
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def _normalize_number_lenient(number_str: str) -> float:
    """Our extension: handle EU decimal comma + trailing units + currency.

    NOT GAIA. Behind ``strict=False`` only.

    * Strip currency / percent symbols and whitespace.
    * Strip a trailing unit tail ("42 km" -> "42").
    * Disambiguate ``,`` vs ``.`` as decimal/thousands separator:
        - both present -> the *last* one is the decimal separator
          ("1.234,56" EU -> 1234.56 ; "1,234.56" US -> 1234.56).
        - only comma present -> a single comma with 1-2 trailing digits is a
          decimal comma ("100,5" -> 100.5); otherwise it is a thousands sep
          ("100,000" -> 100000).
    """
    s = number_str.strip()
    for sym in ("$", "€", "£", "¥", "%"):
        s = s.replace(sym, "")
    s = s.strip()
    s = _TRAILING_UNIT_RE.sub("", s).strip()

    has_comma = "," in s
    has_dot = "." in s
    if has_comma and has_dot:
        if s.rfind(",") > s.rfind("."):  # comma is decimal sep (EU)
            s = s.replace(".", "").replace(",", ".")
        else:  # dot is decimal sep (US)
            s = s.replace(",", "")
    elif has_comma:
        frac = s.rsplit(",", 1)[-1]
        if s.count(",") == 1 and 1 <= len(frac) <= 2 and frac.isdigit():
            s = s.replace(",", ".")  # decimal comma
        else:
            s = s.replace(",", "")  # thousands sep
    try:
        return float(s)
    except ValueError:
        return float("inf")


def normalize_number_str(number_str: str, *, strict: bool = True) -> float:
    """Normalize a numeric answer string to float (``inf`` on failure)."""
    if strict:
        return _normalize_number_strict(number_str)
    return _normalize_number_lenient(number_str)


# --------------------------------------------------------------------------- #
# String normalization
# --------------------------------------------------------------------------- #
def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """Verbatim port of the official ``normalize_str``.

    Removes ALL whitespace (not collapse), lowercases, and — if
    ``remove_punct`` — strips all ASCII punctuation. No article removal, no
    Unicode-punctuation handling. Identical in strict and lenient modes; the
    official scorer never diverges here.
    """
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        return no_spaces.lower().translate(_PUNCT_TRANSLATOR)
    return no_spaces.lower()


# --------------------------------------------------------------------------- #
# List splitting
# --------------------------------------------------------------------------- #
def _split_string_strict(s: str, char_list: list[str] | None = None) -> list[str]:
    """Verbatim port of the official ``split_string``.

    Splits on EVERY ``,`` or ``;``. Order-preserving, elements NOT trimmed.
    Commas inside an element ("1,234") are NOT protected -> mis-split.
    """
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


# A number-with-thousands-separator: 1+ digits, then groups of exactly 3 digits
# preceded by a comma, optionally a decimal tail. Used to mask such numbers
# before lenient list splitting so they are not mis-split.
_THOUSANDS_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_MASK = "\x00"


def _split_string_lenient(s: str) -> list[str]:
    """Our extension: split a list answer without mis-splitting 1,234.

    NOT GAIA. We temporarily mask ``,``-grouped numbers, split on the remaining
    ``,``/``;`` delimiters, trim each element, then restore the masked commas.
    """
    masked = _THOUSANDS_NUM_RE.sub(lambda m: m.group(0).replace(",", _MASK), s)
    parts = re.split(r"[,;]", masked)
    return [p.replace(_MASK, ",").strip() for p in parts]


def split_string(
    s: str, char_list: list[str] | None = None, *, strict: bool = True
) -> list[str]:
    """Split a delimited list answer into elements."""
    if strict:
        return _split_string_strict(s, char_list)
    return _split_string_lenient(s)


# --------------------------------------------------------------------------- #
# Top-level scorer
# --------------------------------------------------------------------------- #
def _has_list_delim(s: str) -> bool:
    return any(c in s for c in (",", ";"))


def question_scorer(
    model_answer: str | None, ground_truth: str, *, strict: bool = True
) -> bool:
    """GAIA quasi-exact-match: is ``model_answer`` correct for ``ground_truth``?

    Type dispatch keyed on the *ground truth* (official behaviour):

    * ``ground_truth`` parses as float -> NUMBER branch (float equality, no
      tolerance).
    * else ``ground_truth`` contains ``,`` / ``;`` -> LIST branch
      (order-sensitive, positional zip, length-mismatch -> False; per-element
      dispatch — list-element strings keep punctuation, ``remove_punct=False``).
    * else -> STRING branch (``normalize_str`` with ``remove_punct=True``).

    ``model_answer is None`` -> coerced to ``"None"`` (official guard) -> always
    wrong against a real ground truth. Empty / commentary-wrapped answers fall
    through to the same branches and score wrong, matching GAIA.

    ``strict=True`` (default) is bit-for-bit the official scorer.
    ``strict=False`` applies our EU-comma / units / safe-list extensions.
    """
    if model_answer is None:
        model_answer = "None"

    # NUMBER branch
    if is_float(ground_truth):
        return normalize_number_str(model_answer, strict=strict) == float(ground_truth)

    # LIST branch
    if _has_list_delim(ground_truth):
        gt_elems = split_string(ground_truth, strict=strict)
        ma_elems = split_string(model_answer, strict=strict)
        if len(gt_elems) != len(ma_elems):
            warnings.warn(
                "Answer lists have different lengths, returning False.",
                UserWarning,
                stacklevel=2,
            )
            return False
        comparisons: list[bool] = []
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if is_float(gt_elem):
                comparisons.append(
                    normalize_number_str(ma_elem, strict=strict) == float(gt_elem)
                )
            else:
                # Asymmetry vs the standalone-string branch: list-element
                # strings keep punctuation (remove_punct=False).
                comparisons.append(
                    normalize_str(ma_elem, remove_punct=False)
                    == normalize_str(gt_elem, remove_punct=False)
                )
        return all(comparisons)

    # STRING branch
    return normalize_str(model_answer) == normalize_str(ground_truth)


# --------------------------------------------------------------------------- #
# Answer extraction (OUR layer — upstream of question_scorer)
# --------------------------------------------------------------------------- #
# Primary sentinel — OUR convention, not GAIA's. The agent is prompted to wrap
# the final answer in <final_answer>…</final_answer>.
_SENTINEL_RE = re.compile(
    r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE
)
# Fallback heuristic: a "FINAL ANSWER:" label (GAIA's own prompt convention),
# possibly bold/markdown-decorated, capturing the rest of that line.
_LABEL_RE = re.compile(
    # ``(?<![A-Za-z])`` so "semifinal answer:" doesn't match, and ``\s+`` (not
    # ``\s*``) between the words so "finalanswer:" doesn't either — the label must
    # be a standalone "final answer" token, optionally markdown-decorated.
    r"(?<![A-Za-z])final\s+answer\s*[:\-]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_STRIP_WRAP = " \t\r\n`*_\"'"


def extract_answer(text: str | None) -> str | None:
    """Pull the final-answer string out of an agent transcript.

    Tolerant, by design (ADR §4): the ``<final_answer>`` sentinel is *our*
    invention, so a strict-only extractor would score correct answers as wrong
    and depress the baseline. Resolution order:

    1. last ``<final_answer>…</final_answer>`` block (``re.DOTALL``);
    2. last ``FINAL ANSWER: …`` labelled line;
    3. last non-empty line of the transcript;
    4. ``None`` if nothing usable (recorded as extraction-failure upstream).

    Surrounding whitespace and markdown wrappers (`` ` ``, ``*``, ``_``, quotes)
    are stripped. An empty match (e.g. ``<final_answer></final_answer>``) yields
    ``""``, which the scorer then treats as wrong — not ``None``, because the
    agent *did* emit a (blank) answer rather than failing to format one.
    """
    if text is None:
        return None

    sentinel = _SENTINEL_RE.findall(text)
    if sentinel:
        return sentinel[-1].strip().strip(_STRIP_WRAP)

    labelled = _LABEL_RE.findall(text)
    if labelled:
        return labelled[-1].strip().strip(_STRIP_WRAP)

    for line in reversed(text.splitlines()):
        stripped = line.strip().strip(_STRIP_WRAP)
        if stripped:
            return stripped

    return None
