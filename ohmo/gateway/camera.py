"""Best-effort Camera API ingress into the configured gateway session.

The journal is an attempt tombstone, not a durable outbox.  In particular a
restart never resumes a send whose result may have been lost.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ohmo.camera_protocol.models import (
    ClassifierOutput,
    ManifestV2,
    candidate_id_for,
    validate_candidate_id,
)
from ohmo.gateway.attachment_fingerprints import (
    PHASH_ALGORITHM,
    PHASH_HAMMING_THRESHOLD,
    fingerprint_image_bytes,
    phash_hamming_distance,
)
from openharness.channels.bus.events import InboundMessage, OutboundDeliveryReceipt, OutboundMessage

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_MANIFEST = 64 * 1024
_MAX_SIDECAR = 1024 * 1024
_MAX_IMAGE = 10 * 1024 * 1024
_MAX_REQUEST = 4096
_MAX_HTTP_BODY = 12 * 1024 * 1024
_MAX_ATTEMPTS = 10000
_MAX_JOURNAL = 8 * 1024 * 1024
_MAX_HISTORY_FINGERPRINTS = 100_000
_DEDUP_WINDOW = timedelta(days=7)
_REFERENCE_WINDOW = timedelta(minutes=5)
_SESSION_TTL_SECONDS = 24 * 60 * 60
_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
_YES = frozenset(
    {"да, я это съела", "я это съела", "я съела это", "я съела", "я это ел", "я это съел"}
)
_NO = frozenset({"нет, не ела", "нет, не ел"})
_NOT_FOOD = frozenset({"это не еда", "не еда", "это не пища"})
_PENDING_TTL_SECONDS = 30 * 60
_ANSWER_EXPLICIT_NO_RE = re.compile(
    r"\b(?:не\s+(?:ел|ела|ели|пил|пила|выпил|выпила|употреблял|употребляла)\b"
    r"|ничего\s+не\s+(?:ел|ела|пил|пила)\b"
    r"|это\s+не\s+еда\b)",
    re.IGNORECASE,
)
_ANSWER_ANCHORED_NO_RE = re.compile(r"^(?:нет|no)\b", re.IGNORECASE)
_ANSWER_CONSUMPTION_RE = re.compile(
    r"\b(?:я\s+)?(?:съел(?:а|и)?|ел(?:а|и)?|поел(?:а|и)?|выпил(?:а|и)?|употребил(?:а|и)?)\b",
    re.IGNORECASE,
)
_ANSWER_NEGATED_CONSUMPTION_RE = re.compile(
    r"\b(?:я\s+)?не\s+(?:съел[аи]?|ел[аи]?|поел[аи]?|выпил[аи]?|употребил[аи]?)\b"
    r"|\bничего\s+не\s+(?:съел[аи]?|ел[аи]?|пил[аи]?)\b"
    r"|\b(?:i\s+)?(?:did\s+not|didn't|never)\s+(?:eat|have|drink)\b",
    re.IGNORECASE,
)
_ANSWER_ANCHORED_YES_RE = re.compile(
    r"^(?:да|ага|угу|конечно|естественно|yes|yeah)\b", re.IGNORECASE
)
_CAMERA_CONSUMPTION_QUESTION_RE = re.compile(
    r"\b(?:съел(?:а|и)?|ел(?:а|и)?|пил(?:а|и)?|выпил(?:а|и)?|"
    r"did\s+you\s+(?:eat|drink)|have\s+you\s+(?:eaten|drunk))\b",
    re.IGNORECASE,
)
_CAMERA_PORTION_QUESTION_RE = re.compile(
    r"\b(?:which\s+(?:portion|part|amount|quantity)|how\s+much|"
    r"portion\s+to\s+(?:count|log)|(?:какую|какая|какой)\s+(?:часть|порци\w*|количеств\w*)|"
    r"сколько\s+(?:съел\w*|порци\w*|учитывать))\b",
    re.IGNORECASE,
)
_CAMERA_ANALYSIS_ONLY_QUESTION_RE = re.compile(
    r"\b(?:только|лишь)\s+(?:оцен\w*|разобрат\w*|посмотр\w*|узнать|"
    r"состав\w*|рецепт\w*|информац\w*)\b|"
    r"\b(?:оцен\w*|разобрат\w*|посмотр\w*|узнать)\b.{0,32}\b"
    r"(?:состав\w*|рецепт\w*|информац\w*)\b|"
    r"\bonly\s+(?:estimate|assess|analy[sz]e|review|inspect|view)\b.{0,48}\b"
    r"(?:composition|ingredients?|recipe|nutrition\s+facts|information)\b",
    re.IGNORECASE,
)
_PORTION_BUTTON_RE = re.compile(
    r"^(?:(?:только|лишь)\s+)?(?:маленьк\w*|больш\w*|small|large|big)\s+"
    r"(?:чашк\w*|кружк\w*|cup)s?[.! ]*$|^(?:обе|оба|обе\s+чашки|both(?:\s+cups?)?)[.! ]*$",
    re.IGNORECASE,
)
_NON_CONSUMPTION_SCOPE_BUTTON_RE = re.compile(
    r"\b(?:состав\w*|разобрат\w*|оцен\w*|узнать|калори\w*|ккал|"
    r"не\s+(?:уверен\w*|знаю|помню)|неизвестн\w*|сомнева\w*|затрудня\w*|"
    r"рецепт\w*|посмотр\w*|информац\w*|"
    r"unsure|uncertain|unknown|don't\s+know|composition|nutrition\s+facts|"
    r"recipe|view|look|watch|see|information)\b",
    re.IGNORECASE,
)
_CLARIFICATION_NEW_MEAL_RE = re.compile(
    r"\b(?:нов(?:ый|ая|ое|ые)|друг(?:ой|ая|ое|ие)|не\s+тот|не\s+это|tomorrow|another|different)\b",
    re.IGNORECASE,
)
_CLARIFICATION_EXPLICIT_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
_CAMERA_UNRELATED_CONTEXT_RE = re.compile(
    r"\b(?:weather|погод\w*|спасибо|благодар\w*|thanks?|payment|оплат\w*|перевод\w*|"
    r"деньг\w*|сч[её]т\w*|карт\w*|рубл\w*|валют\w*|invoice|transfer|bank)\b",
    re.IGNORECASE,
)
_CAMERA_MEDIA_STATUS_RE = re.compile(
    r"\b(?:\d+\s+)?(?:фотограф\w*|фото|снимк\w*|изображен\w*|сообщен\w*|"
    r"photos?|pictures?|images?|messages?)\b.{0,32}\b(?:пропал\w*|исчез\w*|"
    r"приш\w*|получ\w*|отправ\w*|загруз\w*|удал\w*|не\s+дошл\w*|"
    r"disappeared|arrived|received|sent|uploaded|deleted|missing)\b|"
    r"\b(?:пропал\w*|исчез\w*|приш\w*|получ\w*|отправ\w*|загруз\w*|"
    r"удал\w*|disappeared|arrived|received|sent|uploaded|deleted|missing)\b"
    r".{0,32}\b(?:\d+\s+)?(?:фотограф\w*|фото|снимк\w*|изображен\w*|сообщен\w*|"
    r"photos?|pictures?|images?|messages?)\b",
    re.IGNORECASE,
)
_CAMERA_TEMPORAL_CONTEXT_RE = re.compile(
    r"\b(?:позже|потом|позднее|раньше|затем|later|afterwards|after\s+that)\b",
    re.IGNORECASE,
)
_CAMERA_UNCERTAIN_CONSUMPTION_RE = re.compile(
    r"\b(?:не\s+(?:уверен\w*|знаю|помню)|неизвестн\w*|сомнева\w*|затрудня\w*|"
    r"unsure|uncertain|unknown|don't\s+know)\b",
    re.IGNORECASE,
)
_CAMERA_FOOD_CONTEXT_RE = re.compile(
    r"^(?:only|just|только|лишь)\s+[\w-]+(?:\s+[\w-]+){0,2}[.! ]*$",
    re.IGNORECASE,
)
_CLARIFICATION_ONLY_PART_RE = re.compile(
    r"\b(?:only\s+part|часть\s+порци\w*|только\s+часть|половин\w*\s+порци\w*)\b",
    re.IGNORECASE,
)
_CLARIFICATION_PLATE_RE = re.compile(
    r"^(?:whole\s+plate|всю\s+тарелк\w*|цел\w*\s+тарелк\w*)[.! ]*$",
    re.IGNORECASE,
)
_NATIVE_WHOLE_PLATE_RE = re.compile(
    r"^(?:everything(?:\s+on\s+(?:the\s+)?plate)?|all(?:\s+of\s+it)?|"
    r"whole\s+plate|the\s+whole\s+plate|"
    r"всё\s+на\s+тарелк\w*|все\s+на\s+тарелк\w*|"
    r"всё\s+с\s+тарелк\w*|все\s+с\s+тарелк\w*|всю\s+тарелк\w*|"
    r"цел\w*\s+тарелк\w*)[.! ]*$",
    re.IGNORECASE,
)
_NATIVE_WHOLE_PORTION_RE = re.compile(
    r"^(?:всё|все)\s*:\s*\S.{2,}$",
    re.IGNORECASE,
)
_NATIVE_PARTIAL_PORTION_RE = re.compile(
    r"\b(?:part(?:\s+of)?|some\s+of|half(?:\s+of)?|portion|кусоч\w*|"
    r"част\w*|половин\w*|немного)\b",
    re.IGNORECASE,
)
_CONTEXTUAL_PARTIAL_PORTION_RE = re.compile(
    r"^(?:"
    r"(?:part\s+of|some\s+of|half(?:\s+of)?)\s+(?:it|the\s+plate|the\s+dish|[\w-]+)"
    r"|(?:some|a\s+little)\s+[\w-]+"
    r"|(?:часть|половин\w*|кусоч\w*)\s+[\w-]+"
    r"|[\w-]+(?:\s+[\w-]+){0,2}\s+и\s+(?:часть|половин\w*|кусоч\w*)\s+[\w-]+"
    r")\s*[.!?…]*$",
    re.IGNORECASE,
)
_CAMERA_UNITLESS_QUANTITY_PREFIX = (
    r"(?:\d+(?:[.,]\d+)?|полтора|полторы|half|a\s+little|немного)"
)
_CAMERA_COUNTED_QUANTITY_PREFIX = (
    r"(?:one|two|three|four|five|несколько|один|одна|два|две|три|четыре|пять)"
)
_CAMERA_SUPPORTED_QUANTITY_PREFIX = (
    rf"(?:{_CAMERA_UNITLESS_QUANTITY_PREFIX}|{_CAMERA_COUNTED_QUANTITY_PREFIX})"
)
_CAMERA_QUANTITY_MEASURE = (
    r"(?:шт\.?|кусоч\w*|порци\w*|горст\w*|handfuls?|грамм\w*|"
    r"г(?=\s|$|[.!])|кг|мл|л|pieces?|portions?|grams?|kg|ml|l|cups?|"
    r"чашк\w*|стакан\w*|яблок\w*|груш\w*|банан\w*)"
)
_CAMERA_WHOLE_PORTION_PREFIX_RE = re.compile(
    r"^\s*(?:всё|все)\s*:\s*(?P<payload>.+?)\s*$", re.IGNORECASE
)
_CAMERA_WHOLE_PORTION_SOURCE_SUFFIX_RE = re.compile(
    r"\s+(?:на\s+фото|с\s+этого\s+фото)\s*[.!?…]*$", re.IGNORECASE
)
_CAMERA_WHOLE_PORTION_NONFOOD_SUBJECT_RE = re.compile(
    r"\b(?:фотограф\w*|фото|снимк\w*|изображен\w*|видео|ролик\w*|"
    r"аудио|файл\w*|медиа|документ\w*|сообщен\w*|письм\w*|чат\w*|"
    r"photos?|pictures?|images?|videos?|audio|files?|media|documents?|messages?|emails?|chats?)\b",
    re.IGNORECASE,
)
_CAMERA_WHOLE_PORTION_ATTACHED_MEASURE_RE = re.compile(
    rf"^(?:{_CAMERA_QUANTITY_MEASURE})(?=$|[\s.!?,;:…])", re.IGNORECASE
)
_CAMERA_WHOLE_PORTION_LEADING_NUMBER_RE = re.compile(
    r"^\s*\d+(?:[.,]\d+)?(?P<tail>.*)$", re.DOTALL
)
_CAMERA_QUANTITY_QUALIFIER = r"(?:(?:about|around|примерно|около)\s*)?"
_CAMERA_COUNTED_BERRY_DESCRIPTOR = (
    r"(?:виноград\w*|ягод\w*|клубник\w*|grapes?|berries|strawberr(?:y|ies))"
)
_CLARIFICATION_QUANTITY_RE = re.compile(
    r"^\s*"
    + _CAMERA_QUANTITY_QUALIFIER
    + r"(?:"
    + rf"{_CAMERA_UNITLESS_QUANTITY_PREFIX}\s*(?:{_CAMERA_QUANTITY_MEASURE})?"
    + rf"|{_CAMERA_COUNTED_QUANTITY_PREFIX}\s*{_CAMERA_QUANTITY_MEASURE}"
    + rf"|{_CAMERA_SUPPORTED_QUANTITY_PREFIX}\s+(?:горст\w*|handfuls?)\s+(?:of\s+)?{_CAMERA_COUNTED_BERRY_DESCRIPTOR}"
    + rf"|{_CAMERA_SUPPORTED_QUANTITY_PREFIX}\s+кусоч\w*\s+хлеб\w*"
    + r"|немного\s+каш\w*"
    + r"|(?:whole\s+plate|half\s+portion|всю\s+тарелк\w*|цел\w*\s+тарелк\w*|половин\w*\s+порци\w*|часть\s+порци\w*)"
    + r"|(?:маленьк\w*|больш\w*|средн\w*)\s+(?:чашк\w*|стакан\w*|порци\w*)"
    + r")\s*[.!]?\s*$",
    re.IGNORECASE,
)
_CAMERA_QUANTITY_LEAD_RE = re.compile(
    r"^\s*"
    + _CAMERA_QUANTITY_QUALIFIER
    + r"(?:\d+(?:[.,]\d+)?|"
    + _CAMERA_SUPPORTED_QUANTITY_PREFIX
    + rf"(?=\s|$|[.!]|{_CAMERA_QUANTITY_MEASURE}))",
    re.IGNORECASE,
)

_CAMERA_COMPOSITION_WORD = r"[^\W\d_][\w-]*"
_CAMERA_COMPOSITION_NOUN_PHRASE = (
    rf"{_CAMERA_COMPOSITION_WORD}(?:\s+{_CAMERA_COMPOSITION_WORD}){{0,2}}"
)
_CAMERA_COMPOSITION_QUANTIFIED_PHRASE = (
    rf"{_CAMERA_SUPPORTED_QUANTITY_PREFIX}"
    + rf"(?:\s*{_CAMERA_QUANTITY_MEASURE}\s+|\s+)"
    + rf"{_CAMERA_COMPOSITION_NOUN_PHRASE}"
)
_CAMERA_COMPOSITION_COMPONENT = (
    rf"(?:{_CAMERA_COMPOSITION_QUANTIFIED_PHRASE}|{_CAMERA_COMPOSITION_NOUN_PHRASE})"
)
_CAMERA_COMPOSITION_DESCRIPTION_RE = re.compile(
    rf"^{_CAMERA_COMPOSITION_COMPONENT}\s+(?:и|and)\s+"
    + rf"{_CAMERA_COMPOSITION_COMPONENT}[.!?…]*$",
    re.IGNORECASE,
)
_CAMERA_COMPOSITION_PARTS_RE = re.compile(
    r"^\s*(?P<left>.+?)\s+(?:и|and)\s+(?P<right>.+?)[.!?…]*\s*$",
    re.IGNORECASE,
)
_CAMERA_COMPOSITION_TOKEN_RE = re.compile(_CAMERA_COMPOSITION_WORD, re.IGNORECASE)
_CAMERA_COMPOSITION_OPERATOR_RE = re.compile(
    r"\b(?:шт\.?|кусоч\w*|порци\w*|горст\w*|handfuls?|грамм\w*|г|кг|мл|л|"
    r"pieces?|portions?|grams?|kg|ml|cups?|чашк\w*|стакан\w*|част\w*|half|of|"
    r"маленьк\w*|больш\w*|средн\w*|small|large|big|medium)\b",
    re.IGNORECASE,
)


def _clarification_related(text: object) -> bool:
    return bool(
        _classify_answer(text, anchored=False)
        or (isinstance(text, str) and _CLARIFICATION_QUANTITY_RE.search(text))
    )


def _camera_food_context_hint(text: object) -> bool:
    return bool(
        isinstance(text, str)
        and _CAMERA_FOOD_CONTEXT_RE.fullmatch(text.strip())
        and not _CAMERA_UNRELATED_CONTEXT_RE.search(text)
    )
_DEEPSEEK_MODEL = "deepseek/deepseek-v4.1-flash"
_DEEPSEEK_ENDPOINT = "deepinfra/fp8"
_DEEPSEEK_RELEASE = "deepseek-camera-production-v1"
CAMERA_AUTHORITY = object()
CAMERA_CONTEXT_QUESTION_AUTHORITY = object()
COALESCED_ATTACHMENT_PROVENANCE_AUTHORITY = object()
logger = logging.getLogger(__name__)


def _classify_answer(text: object, *, anchored: bool) -> str | None:
    """Classify an explicit user answer to a Camera question.

    Returns "yes", "no", "not_food", or None (ambiguous, must stay unbound). Bare,
    non-anchored text binds only through explicit consumption or negation
    language; bare affirmations and scope-only answers stay ambiguous
    outside a native reply or a first-party ask button.
    """
    answer = text.strip().casefold() if isinstance(text, str) else ""
    if not answer:
        return None
    if answer in _YES:
        return "yes"
    if answer in _NOT_FOOD:
        return "not_food"
    if answer in _NO:
        return "no"
    if _ANSWER_NEGATED_CONSUMPTION_RE.search(answer) or _ANSWER_EXPLICIT_NO_RE.search(answer):
        return "no"
    if _ANSWER_CONSUMPTION_RE.search(answer):
        return "yes"
    if anchored:
        if _ANSWER_ANCHORED_NO_RE.search(answer):
            return "no"
        if _ANSWER_ANCHORED_YES_RE.search(answer):
            return "yes"
    return None


def _camera_explicit_consumption(text: object) -> bool:
    answer = text if isinstance(text, str) else ""
    return bool(
        _ANSWER_CONSUMPTION_RE.search(answer)
        and not _ANSWER_NEGATED_CONSUMPTION_RE.search(answer)
    )


def _camera_whole_portion_payload_excluded(text: str) -> bool:
    match = _CAMERA_WHOLE_PORTION_PREFIX_RE.match(text)
    if match is None:
        return False
    payload = _CAMERA_WHOLE_PORTION_SOURCE_SUFFIX_RE.sub("", match["payload"]).strip()
    leading_number = _CAMERA_WHOLE_PORTION_LEADING_NUMBER_RE.match(payload)
    if leading_number:
        tail = leading_number["tail"]
        if tail and not tail[0].isspace() and tail[0] not in ".,!?:;…":
            if not _CAMERA_WHOLE_PORTION_ATTACHED_MEASURE_RE.match(tail):
                return True
    return bool(_CAMERA_WHOLE_PORTION_NONFOOD_SUBJECT_RE.search(payload))


def _camera_composition_description(
    text: object, *, source_context: object = None, require_source_context: bool = False
) -> bool:
    if (
        not isinstance(text, str)
        or "?" in text
        or not _CAMERA_COMPOSITION_DESCRIPTION_RE.fullmatch(text.strip())
    ):
        return False
    if (
        _CAMERA_WHOLE_PORTION_NONFOOD_SUBJECT_RE.search(text)
        or _CAMERA_UNRELATED_CONTEXT_RE.search(text)
        or _CLARIFICATION_NEW_MEAL_RE.search(text)
        or _ANSWER_NEGATED_CONSUMPTION_RE.search(text)
    ):
        return False
    if not require_source_context:
        return True
    if not isinstance(source_context, str):
        return False
    parts = _CAMERA_COMPOSITION_PARTS_RE.fullmatch(text.strip())
    if parts is None:
        return False
    context_words = {
        token.casefold() for token in _CAMERA_COMPOSITION_TOKEN_RE.findall(source_context)
    }
    for component in (parts["left"], parts["right"]):
        if not re.fullmatch(_CAMERA_COMPOSITION_COMPONENT, component.strip(), re.IGNORECASE):
            return False
        descriptor = re.sub(
            rf"^\s*{_CAMERA_QUANTITY_QUALIFIER}{_CAMERA_SUPPORTED_QUANTITY_PREFIX}",
            "",
            component.strip(),
            flags=re.IGNORECASE,
        )
        descriptor = _CAMERA_COMPOSITION_OPERATOR_RE.sub(" ", descriptor)
        words = [
            token.casefold()
            for token in _CAMERA_COMPOSITION_TOKEN_RE.findall(descriptor)
        ]
        if not words or not all(word in context_words for word in words):
            return False
    return True


def _camera_affirmation_excluded(
    text: object, *, source_context: object = None, require_composition_source: bool = False
) -> bool:
    """Reject uncertainty and deferral; scope words alone lose to explicit eating."""
    if not isinstance(text, str):
        return True
    composition_description = _camera_composition_description(
        text, source_context=source_context,
        require_source_context=require_composition_source,
    )
    if _camera_whole_portion_payload_excluded(text):
        return True
    if _CAMERA_UNCERTAIN_CONSUMPTION_RE.search(text):
        return True
    explicit_consumption = _camera_explicit_consumption(text)
    if (
        _CAMERA_QUANTITY_LEAD_RE.search(text)
        and not _CLARIFICATION_QUANTITY_RE.fullmatch(text.strip())
        and not composition_description
        and not explicit_consumption
    ):
        return True
    if _CAMERA_MEDIA_STATUS_RE.search(text) and not explicit_consumption:
        return True
    if _CAMERA_TEMPORAL_CONTEXT_RE.search(text) and not explicit_consumption:
        return True
    return bool(
        _NON_CONSUMPTION_SCOPE_BUTTON_RE.search(text)
        and not explicit_consumption
    )


def _native_button_answer_kind(metadata: Mapping[str, object], raw_text: object) -> str | None:
    """Interpret only the selected label from its recorded native keyboard."""
    if metadata.get("callback_query") is not True:
        return None
    options = metadata.get("native_keyboard_options")
    index = metadata.get("native_keyboard_selected_index")
    label = metadata.get("native_keyboard_selected_label")
    data = metadata.get("callback_data")
    question = metadata.get("native_keyboard_question")
    prompt = metadata.get("native_keyboard_prompt")
    reflection = metadata.get("native_keyboard_reflection_confirmed")
    reflected_text = metadata.get("native_keyboard_reflection")
    if (
        not isinstance(options, list) or not 2 <= len(options) <= 8
        or type(index) is not int or not 0 <= index < len(options)
        or not isinstance(label, str) or options[index] != label
        or not isinstance(data, str) or data != f"ask:{index}"
        or not isinstance(raw_text, str) or raw_text.strip() != label
        or reflection is not True
        or not isinstance(reflected_text, str)
        or f"✅ {label.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')}" not in reflected_text
    ):
        return None
    # Native option text is evidence only when that exact option expresses a
    # decision. Anchored scope prefixes such as “only estimate composition”
    # are not evidence of consumption.
    direct = _classify_answer(label, anchored=False)
    if direct in {"no", "not_food"}:
        return direct
    if _camera_affirmation_excluded(label):
        return None
    if direct == "yes":
        return direct
    eating_prompt = " ".join(
        value for value in (question, prompt) if isinstance(value, str)
    )
    current_question = question if isinstance(question, str) else ""
    if _CAMERA_ANALYSIS_ONLY_QUESTION_RE.search(current_question):
        return None
    if (
        isinstance(question, str)
        and _CAMERA_CONSUMPTION_QUESTION_RE.search(question)
        and _ANSWER_ANCHORED_YES_RE.match(label.strip())
        and any(_classify_answer(option, anchored=True) == "no" for option in options)
    ):
        return "yes"
    if (
        _CAMERA_CONSUMPTION_QUESTION_RE.search(eating_prompt)
        and (
            _CAMERA_CONSUMPTION_QUESTION_RE.search(current_question)
            or _CAMERA_PORTION_QUESTION_RE.search(current_question)
        )
        and (
            _NATIVE_WHOLE_PLATE_RE.fullmatch(label.strip())
            or _NATIVE_WHOLE_PORTION_RE.fullmatch(label.strip())
            or _NATIVE_PARTIAL_PORTION_RE.search(label)
            or _camera_composition_description(label)
        )
    ):
        return "yes"
    if (
        isinstance(question, str)
        and _CAMERA_CONSUMPTION_QUESTION_RE.search(question)
        and re.match(r"^(?:только|лишь|only|just)\b", label.strip(), re.IGNORECASE)
        and _camera_food_context_hint(label)
    ):
        return "yes"
    if (
        (
            metadata.get("_camera_known_consumption_clarification") is CAMERA_AUTHORITY
            or (
                isinstance(question, str)
                and _CAMERA_CONSUMPTION_QUESTION_RE.search(question)
            )
        )
        and _PORTION_BUTTON_RE.fullmatch(label.strip())
    ):
        # Only explicit portion/composition choices under a consumption
        # question are affirmative. Other useful or uncertainty labels stay
        # unbound. The selected label is still preserved for portion parsing.
        return "yes"
    if (
        (metadata.get("_camera_known_consumption_clarification") is CAMERA_AUTHORITY
         or (
             _CAMERA_CONSUMPTION_QUESTION_RE.search(eating_prompt)
             and (
                 _CAMERA_CONSUMPTION_QUESTION_RE.search(current_question)
                 or _CAMERA_PORTION_QUESTION_RE.search(current_question)
             )
         ))
        and _CLARIFICATION_QUANTITY_RE.fullmatch(label.strip())
    ):
        # A selected quantity is meaningful only under a verified eating
        # keyboard (validated above) or a retained consumed-meal clarification.
        return "yes"
    return None


def _camera_caption(captured: datetime | None, delivered_at: datetime) -> str:
    if captured is None:
        return "Съели ли вы это? Дата съёмки неизвестна."
    age = (delivered_at.astimezone(timezone.utc) - captured.astimezone(timezone.utc)).total_seconds()
    if 0 <= age < 3600:
        minutes = int(age // 60)
        if minutes == 0:
            elapsed = "меньше минуты назад"
        elif minutes % 10 == 1 and minutes % 100 != 11:
            elapsed = f"{minutes} минуту назад"
        elif minutes % 10 in {2, 3, 4} and minutes % 100 not in {12, 13, 14}:
            elapsed = f"{minutes} минуты назад"
        else:
            elapsed = f"{minutes} минут назад"
        return f"Съели ли вы это? Фото сделано {elapsed}."
    return f"Съели ли вы это? Фото сделано {captured.strftime('%Y-%m-%d %H:%M')}."


def _attempt_caption(attempt: dict) -> str:
    frozen = attempt.get("_camera_caption")
    if isinstance(frozen, str):
        return frozen
    captured = CameraIngress._attempt_capture_time(attempt)
    # Existing journal entries retain the exact date-only caption sent before
    # caption freezing was introduced. Do not alter their edit receipt.
    return (
        f"Съели ли вы это? Фото сделано {captured.date().isoformat()}."
        if captured is not None
        else "Съели ли вы это? Дата съёмки неизвестна."
    )


def _camera_context_answer_kind(
    text: object, *, source_context: object = None
) -> str | None:
    """Classify concise text answers only while an unambiguous Camera source is active."""
    if (
        not isinstance(text, str)
        or _CLARIFICATION_NEW_MEAL_RE.search(text)
        or _CLARIFICATION_EXPLICIT_DATE_RE.search(text)
    ):
        return None
    direct = _classify_answer(text, anchored=True)
    if direct in {"no", "not_food"}:
        return direct
    if _camera_affirmation_excluded(
        text, source_context=source_context, require_composition_source=True
    ):
        return None
    if direct == "yes":
        return direct
    if (
        _NATIVE_WHOLE_PLATE_RE.fullmatch(text.strip())
        or _NATIVE_WHOLE_PORTION_RE.fullmatch(text.strip())
        or _CONTEXTUAL_PARTIAL_PORTION_RE.fullmatch(text.strip())
        or _CLARIFICATION_QUANTITY_RE.fullmatch(text)
        or _camera_composition_description(
            text, source_context=source_context, require_source_context=True
        )
    ):
        return "yes"
    return None


def _bare_context_answer(text: object, *, source_context: object = None) -> bool:
    contextual = _camera_context_answer_kind(text, source_context=source_context)
    explicit = _classify_answer(text, anchored=False)
    return contextual in {"yes", "no"} and explicit != contextual


def _source_message_id(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    rendered = str(value).strip()
    return rendered or None


def _attempt_age_seconds(attempt: dict, now: datetime) -> float:
    """Age of an attempt since admission; unparsable stamps never expire."""
    admitted_at = attempt.get("admitted_at")
    if not isinstance(admitted_at, str):
        return 0.0
    try:
        stamp = datetime.fromisoformat(admitted_at)
    except ValueError:
        return 0.0
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        return 0.0
    return (now - stamp).total_seconds()


def _validate_camera_correction_annotation(value: Mapping[str, object]):
    """Validate a correction while ignoring only unmasked schema defaults."""
    from ohmo.evals.nutrition_trace import NutritionAnnotationV2

    changed = value.get("changed_fields")
    if not isinstance(changed, list) or not all(isinstance(field, str) for field in changed):
        raise ValueError("Camera correction is missing its validated field mask")
    defaults = NutritionAnnotationV2.model_construct(record_type="meal_correction").model_dump(
        mode="json"
    )
    correctable = {
        "basis", "consumption_status", "meal_at", "meal_date", "is_estimate",
        "energy_kcal_min", "energy_kcal_max", "energy_kcal_best", "protein_g",
        "fat_g", "carbohydrate_g", "items", "confidence", "assumptions", "warnings",
    }
    normalized = dict(value)
    for field in correctable - set(changed):
        if field in normalized:
            if normalized[field] != defaults[field]:
                raise ValueError("Camera correction contains an unmasked replacement value")
            normalized.pop(field)
    return NutritionAnnotationV2.model_validate(normalized)


class CameraCandidateRequest(BaseModel):
    """The frozen Camera request allowlist; no producer-authored destination or prompt."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    source_revision: str = Field(min_length=1, max_length=256)
    manifest_sha256: str
    image_sha256: str
    capture_time: datetime
    capture_time_authority: str
    session_id: str
    epoch: str
    seq: int = Field(gt=0, le=2**63 - 1)

    @field_validator("candidate_id")
    @classmethod
    def _candidate(cls, value: str) -> str:
        return validate_candidate_id(value)

    @field_validator("session_id", "epoch")
    @classmethod
    def _session_identity(cls, value: str) -> str:
        if _SESSION_ID.fullmatch(value) is None:
            raise ValueError("invalid Camera session identity")
        return value

    @field_validator("manifest_sha256", "image_sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("invalid SHA-256")
        return value

    @field_validator("capture_time")
    @classmethod
    def _capture_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("capture_time must be timezone-aware")
        return value

    @field_validator("capture_time", mode="before")
    @classmethod
    def _capture_time_wire(cls, value: object) -> object:
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError("capture_time must be a bounded ISO-8601 string")
        return value

    @field_validator("capture_time_authority")
    @classmethod
    def _authority(cls, value: str) -> str:
        if value not in {"exif", "filename"}:
            raise ValueError("invalid capture time authority")
        return value


class CameraReferencesRequest(BaseModel):
    """Strict query for the configured owner's recent confirmed Camera photos."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    capture_time: datetime

    @field_validator("candidate_id")
    @classmethod
    def _candidate(cls, value: str) -> str:
        return validate_candidate_id(value)

    @field_validator("capture_time", mode="before")
    @classmethod
    def _capture_time_wire(cls, value: object) -> object:
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError("capture_time must be a bounded ISO-8601 string")
        return value

    @field_validator("capture_time")
    @classmethod
    def _capture_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("capture_time must be timezone-aware")
        return value


class CameraReferencesRequestV2(CameraReferencesRequest):
    schema_version: Literal[2]


class CameraReferenceSourceRequest(BaseModel):
    """Current immutable source evidence for an existing native Camera photo."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    candidate_id: str
    source_revision: str = Field(min_length=1, max_length=256)
    manifest_sha256: str
    image_sha256: str
    capture_time: datetime
    capture_time_authority: Literal["exif", "filename"]

    @field_validator("candidate_id")
    @classmethod
    def _candidate(cls, value: str) -> str:
        return validate_candidate_id(value)

    @field_validator("manifest_sha256", "image_sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("invalid SHA-256")
        return value

    @field_validator("capture_time", mode="before")
    @classmethod
    def _capture_time_wire(cls, value: object) -> object:
        if not isinstance(value, str) or len(value) > 64:
            raise ValueError("capture_time must be a bounded ISO-8601 string")
        return value

    @field_validator("capture_time")
    @classmethod
    def _capture_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("capture_time must be timezone-aware")
        return value


class CameraReferenceScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal: str
    chat_id: str
    tenant_id: str
    session_key: str


class CameraReferenceSourceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    source_evidence_authority: Literal["current_immutable_original_revision"]
    reestablished_at: datetime
    scope: CameraReferenceScope
    candidate_id: str
    file_id: str = Field(min_length=1, max_length=256)
    source_revision: str = Field(min_length=1, max_length=256)
    image_sha256: str
    capture_time: datetime
    capture_time_authority: Literal["exif", "filename"]
    snapshot: str | None

    @field_validator("candidate_id")
    @classmethod
    def _candidate(cls, value: str) -> str:
        return validate_candidate_id(value)

    @field_validator("image_sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if _SHA256.fullmatch(value) is None:
            raise ValueError("invalid SHA-256")
        return value

    @field_validator("capture_time", "reestablished_at")
    @classmethod
    def _aware_times(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reference source timestamp must be timezone-aware")
        return value


@dataclass(frozen=True)
class CameraCandidateUpload:
    """Candidate metadata, producer evidence, and image carried by one HTTP request."""

    request: object
    manifest_bytes: bytes
    producer_sidecar_bytes: bytes
    image_bytes: bytes
    image_filename: str
    image_content_type: str


@dataclass(frozen=True)
class RetainedAttachmentEvidence:
    """One bounded attachment observation from a trusted retained session."""

    id: str
    peer_id: str
    session_id: str
    metadata: Mapping[str, object]
    created_at: datetime


class IncompleteAttachmentHistory(ValueError):
    """A bounded source exposed a known partial album that another source may complete."""


def _fingerprints_match(
    fingerprints: object, candidate: Mapping[str, object]
) -> bool:
    """Validate bounded descriptors and compare exact SHA or supported pHash."""
    if not isinstance(fingerprints, list) or len(fingerprints) > _MAX_HISTORY_FINGERPRINTS:
        raise ValueError("attachment history fingerprints are malformed")
    matched = False
    for fingerprint in fingerprints:
        if (
            not isinstance(fingerprint, Mapping)
            or not isinstance(fingerprint.get("sha256"), str)
            or _SHA256.fullmatch(fingerprint["sha256"]) is None
        ):
            raise ValueError("attachment history fingerprint is malformed")
        if "phash" in fingerprint and (
            fingerprint.get("phash_algorithm") != PHASH_ALGORITHM
            or not isinstance(fingerprint.get("phash"), str)
            or re.fullmatch(r"[0-9a-f]{64}", fingerprint["phash"]) is None
        ):
            raise ValueError("attachment history pHash version is malformed")
        if "phash_algorithm" in fingerprint and "phash" not in fingerprint:
            raise ValueError("attachment history pHash is incomplete")
        if fingerprint["sha256"] == candidate["sha256"]:
            matched = True
        old_phash = fingerprint.get("phash")
        candidate_phash = candidate.get("phash")
        distance = (
            phash_hamming_distance(old_phash, candidate_phash)
            if isinstance(old_phash, str) and isinstance(candidate_phash, str)
            else None
        )
        if (
            fingerprint.get("phash_algorithm") == PHASH_ALGORITHM
            and distance is not None
            and distance <= PHASH_HAMMING_THRESHOLD
        ):
            matched = True
    return matched


def _find_recent_attachment_duplicate(
    history: object,
    candidate: Mapping[str, object],
    *,
    since: datetime,
    until: datetime,
    principal: str,
    expected_session: str | None,
    expected_peer: str | None,
    session_key: str,
    chat_id: str,
    expected_tenant: str | None = None,
    local_snapshot: bool = False,
) -> str | None:
    """Validate one complete bounded owner history and return its stable target."""
    if not isinstance(history, list):
        raise ValueError("attachment history is incomplete")
    first_match: str | None = None
    history_incomplete = False
    for item in history:
        metadata = getattr(item, "metadata", None)
        created = getattr(item, "created_at", None)
        session_id = getattr(item, "session_id", None)
        peer_id = getattr(item, "peer_id", None)
        target_id = getattr(item, "id", None)
        if (
            not isinstance(metadata, Mapping)
            or not isinstance(created, datetime)
            or created.tzinfo is None
            or created.utcoffset() is None
            or not isinstance(target_id, str)
            or not target_id
            or len(target_id) > 256
        ):
            raise ValueError("attachment history item is invalid")
        if not since <= created.astimezone(timezone.utc) <= until:
            raise ValueError("attachment history escaped its time bounds")
        if local_snapshot:
            if (
                not isinstance(session_id, str)
                or peer_id != principal
                or metadata.get("source_snapshot_session_id") != session_id
                or metadata.get("source_session_key") != session_key
                or metadata.get("source_chat_id") != chat_id
                or metadata.get("timestamp_authority") not in {
                    "inbound_event_timestamp",
                    "owner_local_object_mtime_observed_retention",
                }
                or metadata.get("target_authority") not in {
                    "native_event_id",
                    "owner_local_observed_retention_ref_group",
                }
                or (target_id.startswith("retained:")
                    and metadata.get("target_authority")
                    != "owner_local_observed_retention_ref_group")
                or metadata.get("source_channel") != "telegram"
                or metadata.get("received_at")
                != created.astimezone(timezone.utc).isoformat()
            ):
                raise ValueError("retained attachment snapshot provenance is invalid")
        elif session_id != expected_session or peer_id != expected_peer:
            raise ValueError("attachment history escaped its owner/session bounds")
        if metadata.get("role") != "user":
            continue
        if metadata.get("source_principal") != principal:
            continue
        fingerprints = metadata.get("attachment_fingerprints", [])
        image_count = metadata.get("source_image_attachment_count")
        if image_count is not None and (
            not isinstance(image_count, int)
            or isinstance(image_count, bool)
            or not 0 <= image_count <= _MAX_HISTORY_FINGERPRINTS
            or not isinstance(fingerprints, list)
            or image_count < len(fingerprints)
        ):
            raise ValueError("attachment history image count is malformed")
        if image_count is not None and image_count > len(fingerprints):
            history_incomplete = True
        if fingerprints == []:
            continue
        legacy_candidate = metadata.get("candidate_id")
        legacy_turn = metadata.get("logical_turn_id")
        if (
            metadata.get("ingest_source") == "dropbox_camera"
            and metadata.get("confirmation_required") is True
            and metadata.get("_nutrition_trusted") is True
            and metadata.get("nutrition_phase") == "estimation"
            and isinstance(legacy_candidate, str)
            and re.fullmatch(r"dropbox-camera-v1-[0-9a-f]{64}", legacy_candidate)
            and isinstance(legacy_turn, str)
            and bool(legacy_turn)
            and metadata.get("client_op_id") == f"{legacy_candidate}:meal-user:v1"
            and expected_tenant is not None
            and metadata.get("tenant_id") == expected_tenant
            and metadata.get("is_forwarded") is False
            and "is_group" not in metadata
            and "source_image_attachment_count" not in metadata
            and "camera_candidate_id" not in metadata
        ):
            # This exact pre-migration estimation tuple was emitted only by
            # the coordinator-authorized synthetic nutrition writer. It is not
            # evidence that a native Camera delivery occurred. Validate all
            # descriptors before excluding it from human-photo matching.
            _fingerprints_match(fingerprints, candidate)
            continue
        if metadata.get("is_forwarded") is True or metadata.get("is_group") is True:
            continue
        if metadata.get("is_forwarded") is not False or metadata.get("is_group") is not False:
            raise ValueError("attachment history is missing private-source provenance")
        matched = _fingerprints_match(fingerprints, candidate)
        if first_match is None and matched:
            first_match = target_id
    if first_match is None and history_incomplete:
        raise IncompleteAttachmentHistory("attachment history is missing image fingerprints")
    return first_match


def _unique_json(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _parse_camera_upload(content_type: str, body: bytes) -> CameraCandidateUpload:
    """Parse one bounded multipart upload with exactly the four Camera fields."""
    envelope = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("ascii") + body
    )
    if (
        envelope.defects
        or envelope.get_content_type() != "multipart/form-data"
        or not envelope.get_boundary()
        or envelope.preamble
        or envelope.epilogue
        or not envelope.is_multipart()
    ):
        raise ValueError("invalid Camera multipart envelope")

    parts: dict[str, object] = {}
    for part in envelope.iter_parts():
        if part.defects or part.is_multipart() or part.get_content_disposition() != "form-data":
            raise ValueError("invalid Camera multipart part")
        name = part.get_param("name", header="content-disposition")
        if name not in {"request", "manifest", "producer", "image"} or name in parts:
            raise ValueError("unexpected Camera multipart field")
        transfer_encoding = part.get("Content-Transfer-Encoding")
        if transfer_encoding not in {None, "7bit", "8bit", "binary"}:
            raise ValueError("encoded Camera multipart fields are not accepted")
        filename = part.get_filename()
        content_type_for_part = part.get_content_type()
        if name == "image":
            if not isinstance(filename, str) or not filename or len(filename) > 255:
                raise ValueError("Camera image filename is invalid")
        elif filename is not None or content_type_for_part != "application/json":
            raise ValueError("Camera metadata fields must be JSON without filenames")
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            raise ValueError("Camera multipart field has no byte payload")
        parts[name] = (payload, filename, content_type_for_part)
    if set(parts) != {"request", "manifest", "producer", "image"}:
        raise ValueError("Camera multipart fields are incomplete")

    request_bytes, _, _ = parts["request"]
    manifest_bytes, _, _ = parts["manifest"]
    producer_bytes, _, _ = parts["producer"]
    image_bytes, image_filename, image_content_type = parts["image"]
    if not 0 < len(request_bytes) <= _MAX_REQUEST:
        raise ValueError("Camera request metadata size is invalid")
    if not 0 < len(manifest_bytes) <= _MAX_MANIFEST:
        raise ValueError("Camera manifest size is invalid")
    if not 0 < len(producer_bytes) <= _MAX_SIDECAR:
        raise ValueError("Camera producer evidence size is invalid")
    if not 0 < len(image_bytes) <= _MAX_IMAGE:
        raise ValueError("Camera image size is invalid")
    try:
        request_payload = json.loads(request_bytes, object_pairs_hook=_unique_json)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("Camera request metadata is invalid") from exc
    return CameraCandidateUpload(
        request=request_payload,
        manifest_bytes=manifest_bytes,
        producer_sidecar_bytes=producer_bytes,
        image_bytes=image_bytes,
        image_filename=image_filename,
        image_content_type=image_content_type,
    )


def _read_regular(path: str | Path, maximum: int, *, dir_fd: int | None = None) -> bytes:
    """Snapshot a bounded regular file without following its final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, dir_fd=dir_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("candidate evidence is not a bounded regular file")
        data = bytearray()
        while len(data) <= maximum:
            chunk = os.read(fd, min(65536, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            len(data) != before.st_size
            or len(data) > maximum
            or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
            != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)
        ):
            raise ValueError("candidate evidence changed during snapshot")
        return bytes(data)
    finally:
        os.close(fd)


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _child_directory(parent_fd: int, name: str, *, create: bool = False) -> int:
    if create:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
    return os.open(name, _directory_flags(), dir_fd=parent_fd)


def _published_positive(sidecar_bytes: bytes, manifest: ManifestV2) -> None:
    """Validate the bounded Telegent producer snapshot independently of the manifest."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate producer evidence key")
            result[key] = value
        return result

    sidecar = json.loads(sidecar_bytes, object_pairs_hook=unique_object)
    if not isinstance(sidecar, dict):
        raise ValueError("producer evidence is not an object")
    release = sidecar.get("release")
    decision = sidecar.get("decision")
    clip_release = sidecar.get("clip_release")
    clip = sidecar.get("clip_decision")
    if not all(isinstance(item, dict) for item in (release, decision, clip_release, clip)):
        raise ValueError("producer decision or release is absent")
    if (
        type(sidecar.get("schema_version")) is not int
        or sidecar["schema_version"] != 1
        or type(sidecar.get("revision")) is not int
        or sidecar["revision"] < 1
        or sidecar.get("candidate_id") != manifest.candidate_id
        or sidecar.get("state") != "published"
        or sidecar.get("clip_runtime_mode") != "enforce"
    ):
        raise ValueError("candidate is not a published producer result")
    history = sidecar.get("state_history")
    if not isinstance(history, list) or not history or history[-1] != "published":
        raise ValueError("published producer history is absent")
    classifier_fields = {
        "model": manifest.classifier_model,
        "prompt_version": manifest.classifier_prompt_version,
        "policy_version": manifest.classifier_policy_version,
        "dataset_version": manifest.classifier_dataset_version,
    }
    if any(release.get(key) != value for key, value in classifier_fields.items()):
        raise ValueError("producer classifier release differs from manifest")
    if (
        decision.get("candidate_id") != manifest.candidate_id
        or decision.get("is_food_like") is not True
        or decision.get("classifier_route_attestation_json")
        != manifest.classifier_route_attestation_json
        or any(
            decision.get(key) != classifier_fields[key]
            for key in ("model", "prompt_version", "policy_version")
        )
        or ClassifierOutput.model_validate(decision.get("output")) != manifest.classifier_output
    ):
        raise ValueError("producer classifier decision differs from manifest")
    _validate_deepseek_route_attestation(manifest)
    clip_fields = ("model_id", "model_revision", "preprocessing_version", "threshold")
    if (
        clip.get("candidate_id") != manifest.candidate_id
        or clip.get("outcome") != "pass"
        or clip.get("forward_to_classifier") is not True
        or any(clip.get(key) != clip_release.get(key) for key in clip_fields)
        or not isinstance(clip_release.get("model_id"), str)
        or not clip_release["model_id"]
        or not isinstance(clip_release.get("model_revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", clip_release["model_revision"]) is None
        or not isinstance(clip_release.get("preprocessing_version"), str)
        or not clip_release["preprocessing_version"]
    ):
        raise ValueError("producer CLIP release differs from decision")
    threshold, score = clip_release.get("threshold"), clip.get("score")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or score < threshold
    ):
        raise ValueError("producer CLIP pass is invalid")


def _validate_deepseek_route_attestation(manifest: ManifestV2) -> None:
    """Require the selected DeepSeek model, endpoint, and privacy policy."""
    if (
        manifest.classifier_model != _DEEPSEEK_MODEL
        or manifest.classifier_policy_version != _DEEPSEEK_RELEASE
    ):
        raise ValueError("camera classifier is not the active DeepSeek release")
    raw = manifest.classifier_route_attestation_json
    if not isinstance(raw, str) or not raw:
        raise ValueError("DeepSeek route attestation is absent")

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate DeepSeek route attestation key")
            result[key] = value
        return result

    route = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(route, dict):
        raise ValueError("DeepSeek route attestation is malformed")
    expected = {
        "requested_model": _DEEPSEEK_MODEL,
        "requested_provider_only": [_DEEPSEEK_ENDPOINT],
        "requested_zdr": True,
        "requested_data_collection": "deny",
        "requested_allow_fallbacks": False,
        "requested_response_cache_header": "false",
        "prepared_provider_only": [_DEEPSEEK_ENDPOINT],
        "prepared_zdr": True,
        "prepared_data_collection": "deny",
        "prepared_allow_fallbacks": False,
        "prepared_provider_policy_source": "prepared_outbound_request_body",
        "prepared_response_cache_header": "false",
        "prepared_response_cache_header_source": "prepared_outbound_request_headers",
        "catalog_endpoint": _DEEPSEEK_ENDPOINT,
        "catalog_zdr": True,
        "response_model": _DEEPSEEK_MODEL,
        "response_provider": "deepinfra",
        "response_endpoint": None,
        "response_endpoint_source": "unverified",
    }
    if any(route.get(key) != value for key, value in expected.items()):
        raise ValueError("DeepSeek route or privacy policy differs")
    cache_status = route.get("response_cache_status")
    if cache_status == "HIT" or cache_status not in {None, "MISS"}:
        raise ValueError("DeepSeek response cache was not a fresh model call")
    cache_source = "absent" if cache_status is None else "response_header"
    if route.get("response_cache_status_source") != cache_source:
        raise ValueError("DeepSeek cache evidence is malformed")
    for key in (
        "candidate_identity_sha256",
        "catalog_sha256",
        "catalog_terms_sha256",
        "request_identity_sha256",
        "response_id_sha256",
        "response_identity_sha256",
    ):
        value = route.get(key)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError("DeepSeek route evidence digest is malformed")
    for key in ("catalog_observed_at", "requested_at", "response_received_at"):
        value = route.get(key)
        if not isinstance(value, str):
            raise ValueError("DeepSeek route evidence timestamp is absent")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("DeepSeek route evidence timestamp is malformed") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("DeepSeek route evidence timestamp has no timezone")


class CameraIngress:
    """One-owner in-process admission with restart-safe attempt tombstones."""

    def __init__(
        self, config, *, workspace: Path, bus, telegram, recent_attachments=None,
        recent_session: str | None = None, recent_peer: str | None = None,
        retained_attachments=None, history_required: bool = False,
        retained_history_required: bool = False,
    ) -> None:
        self.config = config
        self._bus = bus
        self._telegram = telegram
        if telegram is not None:
            # The Telegram sender asks the ingress to revalidate an initial
            # prompt edit against the live attempt and native photo receipt.
            try:
                setattr(telegram, "_camera_ingress_authority", self)
            except (AttributeError, TypeError):
                logger.debug("Telegram sender cannot bind Camera prompt edit authority")
        self._recent_attachments = recent_attachments
        self._recent_session = recent_session
        self._recent_peer = recent_peer
        self._retained_attachments = retained_attachments
        self._retained_history_required = retained_history_required
        self._history_required = history_required
        self._workspace = workspace
        self._state_dir = workspace / "camera_ingress"
        self._state_path = self._state_dir / "attempts.json"
        self._lock = asyncio.Lock()
        self._attempts, previous_session = self._load_attempts()
        self._session = self._new_session(
            session_id=previous_session.get("session_id") if previous_session else None
        )
        self._reference_source_pending_commit: set[str] = set()
        if getattr(self, "_journal_migrated", False):
            self._save_attempts()
        self._tasks: set[asyncio.Task] = set()
        self._sweep_expired_attempts()
        self._remove_completed_snapshots()

    @staticmethod
    def _new_session(*, session_id: str | None = None) -> dict:
        now = datetime.now(timezone.utc)
        return {
            "session_id": session_id or uuid4().hex,
            "epoch": uuid4().hex,
            "committed_seq": 0,
            "expires_at": (now + timedelta(seconds=_SESSION_TTL_SECONDS)).isoformat(),
            "last_ack": None,
        }

    @staticmethod
    def _attempt_capture_time(attempt: dict) -> datetime | None:
        raw = attempt.get("capture_time")
        if (
            not isinstance(raw, str)
            or len(raw) > 64
            or attempt.get("capture_time_authority") not in {"exif", "filename"}
        ):
            return None
        try:
            instant = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return instant if instant.tzinfo is not None and instant.utcoffset() is not None else None

    async def camera_references(
        self, authorization: str | None, request: CameraReferencesRequest
    ) -> tuple[int, dict]:
        """Read confirmed, snapshot-verified Camera references without journal writes."""
        if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
            return self._error(401, "unauthorized")
        try:
            expected = self._token()
        except (OSError, ValueError):
            return self._error(503, "pre_admission_unavailable")
        if not hmac.compare_digest(authorization[7:].encode(), expected):
            return self._error(401, "unauthorized")
        if not self.config.enabled:
            return self._error(403, "camera_disabled")
        async with self._lock:
            return self._camera_references_locked(request)

    def _camera_references_locked(self, request: CameraReferencesRequest) -> tuple[int, dict]:
        if isinstance(request, CameraReferencesRequestV2):
            return self._camera_references_v2_locked(request)
        now = datetime.now(timezone.utc)
        query_time = request.capture_time.astimezone(timezone.utc)
        if query_time < now - _DEDUP_WINDOW or query_time > now:
            return self._error(422, "capture_out_of_window")
        lower = max(query_time - _REFERENCE_WINDOW, now - _DEDUP_WINDOW)
        references: list[dict] = []
        pending: list[str] = []
        # Reads under the admission lock so the returned projection is one stable
        # view relative to concurrent Camera state transitions. Nothing is saved.
        for candidate_id, attempt in self._attempts.items():
            if candidate_id == request.candidate_id:
                continue
            if attempt.get("state") in {"duplicate", "retired"}:
                continue
            captured = self._attempt_capture_time(attempt)
            if captured is None:
                if attempt.get("state") in {
                    "admitted", "photo_sent", "answering", "final_queued", "completed",
                    "delivery_unknown",
                }:
                    return self._error(503, "reference_evidence_unavailable")
                continue
            captured = captured.astimezone(timezone.utc)
            if not lower <= captured <= query_time:
                continue
            if attempt.get("photo_delivery_confirmed") is not True:
                pending.append(candidate_id)
                continue
            photo_id = attempt.get("photo_id")
            image_sha = attempt.get("image_sha256")
            snapshot = attempt.get("snapshot")
            admission_id = attempt.get("admission_id")
            if (
                type(photo_id) is not int or photo_id <= 0
                or not isinstance(image_sha, str) or _SHA256.fullmatch(image_sha) is None
                or not isinstance(snapshot, str) or not isinstance(admission_id, str)
            ):
                return self._error(503, "reference_evidence_unavailable")
            path = Path(snapshot)
            if (
                path.parent != self._state_dir / "snapshots"
                or path.name not in {
                    f"{admission_id}.jpg", f"{admission_id}.jpeg",
                    f"{admission_id}.png", f"{admission_id}.webp",
                }
            ):
                return self._error(503, "reference_evidence_unavailable")
            try:
                image = _read_regular(path, _MAX_IMAGE)
            except (OSError, ValueError):
                return self._error(503, "reference_evidence_unavailable")
            if not hmac.compare_digest(hashlib.sha256(image).hexdigest(), image_sha):
                return self._error(503, "reference_evidence_unavailable")
            references.append({
                "candidate_id": candidate_id,
                "image_sha256": image_sha,
                "capture_time": captured.isoformat(),
                "capture_time_authority": attempt["capture_time_authority"],
                "native_photo_message_id": photo_id,
            })
        references.sort(key=lambda item: (item["capture_time"], item["candidate_id"]), reverse=True)
        pending.sort()
        return 200, {
            "schema_version": 1,
            "scope": {
                "principal": self.config.principal,
                "chat_id": self.config.chat_id,
                "tenant_id": self.config.tenant_id,
                "session_key": self.config.session_key,
            },
            "selection_policy": "confirmed-camera-five-minute-v1",
            "references": references,
            "pending_candidate_ids": pending,
        }

    def _reference_scope(self) -> CameraReferenceScope:
        return CameraReferenceScope(
            principal=self.config.principal,
            chat_id=self.config.chat_id,
            tenant_id=self.config.tenant_id,
            session_key=self.config.session_key,
        )

    def _reference_source_record(
        self, candidate_id: str, attempt: dict
    ) -> CameraReferenceSourceObservation | None:
        raw = attempt.get("reference_source")
        if raw is None:
            return None
        try:
            source = CameraReferenceSourceObservation.model_validate(raw)
        except ValidationError as error:
            raise ValueError("Camera current reference source is invalid") from error
        if (
            source.candidate_id != candidate_id
            or source.scope != self._reference_scope()
            or source.image_sha256 != attempt.get("image_sha256", source.image_sha256)
            or (
                self._attempt_capture_time(attempt) is not None
                and source.capture_time.astimezone(timezone.utc)
                != self._attempt_capture_time(attempt).astimezone(timezone.utc)
            )
            or (
                attempt.get("capture_time_authority") is not None
                and source.capture_time_authority != attempt.get("capture_time_authority")
            )
        ):
            raise ValueError("Camera current reference source contradicts its journal")
        return source

    @staticmethod
    def _safe_snapshot_name(attempt: dict, suffix: str) -> str:
        admission_id = attempt.get("admission_id")
        if (
            not isinstance(admission_id, str)
            or re.fullmatch(r"cam1-[0-9a-f]{32}", admission_id) is None
            or suffix not in {".jpg", ".jpeg", ".png", ".webp"}
        ):
            raise ValueError("Camera managed snapshot identity is invalid")
        return f"{admission_id}{suffix}"

    def _camera_references_v2_locked(
        self, request: CameraReferencesRequestV2
    ) -> tuple[int, dict]:
        now = datetime.now(timezone.utc)
        query_time = request.capture_time.astimezone(timezone.utc)
        if query_time < now - _DEDUP_WINDOW or query_time > now:
            return self._error(422, "capture_out_of_window")
        lower = max(query_time - _REFERENCE_WINDOW, now - _DEDUP_WINDOW)
        references: list[dict] = []
        reestablished: list[dict] = []
        pending: list[str] = []
        unresolved: list[str] = []
        for candidate_id, attempt in self._attempts.items():
            if candidate_id == request.candidate_id or attempt.get("state") in {"duplicate", "retired"}:
                continue
            try:
                source = self._reference_source_record(candidate_id, attempt)
            except ValueError:
                unresolved.append(candidate_id)
                continue
            captured = self._attempt_capture_time(attempt)
            if captured is not None:
                captured = captured.astimezone(timezone.utc)
                if source is not None and captured != source.capture_time.astimezone(timezone.utc):
                    unresolved.append(candidate_id)
                    continue
            elif source is not None:
                captured = source.capture_time.astimezone(timezone.utc)
            if captured is None:
                unresolved.append(candidate_id)
                continue
            if not lower <= captured <= query_time:
                continue
            if candidate_id in self._reference_source_pending_commit:
                unresolved.append(candidate_id)
                continue
            if attempt.get("photo_delivery_confirmed") is not True:
                pending.append(candidate_id)
                continue
            photo_id = attempt.get("photo_id")
            if type(photo_id) is not int or photo_id <= 0:
                unresolved.append(candidate_id)
                continue
            if source is not None:
                snapshot = source.snapshot
                image_sha = source.image_sha256
                authority = source.capture_time_authority
            else:
                snapshot = attempt.get("snapshot")
                image_sha = attempt.get("image_sha256")
                authority = attempt.get("capture_time_authority")
            if (
                not isinstance(snapshot, str)
                or not isinstance(image_sha, str)
                or _SHA256.fullmatch(image_sha) is None
                or authority not in {"exif", "filename"}
            ):
                unresolved.append(candidate_id)
                continue
            path = Path(snapshot)
            try:
                suffix = path.suffix.lower()
                expected_name = self._safe_snapshot_name(attempt, suffix)
                if path.parent != self._state_dir / "snapshots" or path.name != expected_name:
                    raise ValueError("managed snapshot path differs")
                image = _read_regular(path, _MAX_IMAGE)
            except (OSError, ValueError):
                unresolved.append(candidate_id)
                continue
            if not hmac.compare_digest(hashlib.sha256(image).hexdigest(), image_sha):
                unresolved.append(candidate_id)
                continue
            item = {
                "candidate_id": candidate_id,
                "image_sha256": image_sha,
                "capture_time": captured.isoformat(),
                "capture_time_authority": authority,
                "native_photo_message_id": photo_id,
            }
            if source is None:
                references.append(item)
            else:
                item["source_evidence_authority"] = source.source_evidence_authority
                item["reestablished_at"] = source.reestablished_at.astimezone(timezone.utc).isoformat()
                reestablished.append(item)
        references.sort(
            key=lambda item: (item["capture_time"], item["candidate_id"]), reverse=True
        )
        reestablished.sort(
            key=lambda item: (item["capture_time"], item["candidate_id"]), reverse=True
        )
        pending.sort()
        unresolved.sort()
        return 200, {
            "schema_version": 2,
            "scope": self._reference_scope().model_dump(mode="json"),
            "selection_policy": "confirmed-camera-five-minute-v1",
            "references": references,
            "reestablished_references": reestablished,
            "pending_candidate_ids": pending,
            "unresolved_candidate_ids": unresolved,
            "coverage": "incomplete" if unresolved else "complete",
        }

    async def reference_source(
        self, authorization: str | None, upload: object
    ) -> tuple[int, dict]:
        if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
            return self._error(401, "unauthorized")
        try:
            expected = self._token()
        except (OSError, ValueError):
            return self._error(503, "pre_admission_unavailable")
        if not hmac.compare_digest(authorization[7:].encode(), expected):
            return self._error(401, "unauthorized")
        if not self.config.enabled:
            return self._error(403, "camera_disabled")
        if not isinstance(upload, CameraCandidateUpload):
            return self._error(400, "invalid_request")
        try:
            request = CameraReferenceSourceRequest.model_validate(upload.request)
            image_bytes, suffix, manifest = self._validate_source_artifacts(request, upload)
        except (ValidationError, OSError, ValueError, TypeError):
            return self._error(422, "reference_source_mismatch")
        source_time = request.capture_time.astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        if source_time > now:
            return self._error(422, "reference_source_time_invalid")
        retain_image = source_time >= now - _DEDUP_WINDOW
        async with self._lock:
            attempt = self._attempts.get(request.candidate_id)
            if attempt is None:
                return self._error(503, "reference_source_unresolved")
            if (
                attempt.get("photo_delivery_confirmed") is not True
                or type(attempt.get("photo_id")) is not int
                or attempt["photo_id"] <= 0
            ):
                return self._error(503, "reference_source_unresolved")
            if attempt.get("state") in {"duplicate", "retired"}:
                return self._error(503, "reference_source_unresolved")
            conflicts = self._reference_source_conflicts(attempt, request, manifest)
            if conflicts:
                return self._error(409, "reference_source_conflict")
            admission_id = attempt.get("admission_id")
            if not isinstance(admission_id, str) or re.fullmatch(r"cam1-[0-9a-f]{32}", admission_id) is None:
                return self._error(503, "reference_source_unresolved")
            snapshot = (
                str(self._state_dir / "snapshots" / self._safe_snapshot_name(attempt, suffix))
                if retain_image else None
            )
            try:
                previous = self._reference_source_record(request.candidate_id, attempt)
            except ValueError:
                return self._error(503, "reference_source_unresolved")
            if previous is not None and (
                previous.file_id != manifest.file_id
                or previous.source_revision != manifest.rev
                or previous.image_sha256 != request.image_sha256
                or previous.capture_time.astimezone(timezone.utc) != source_time
                or previous.capture_time_authority != request.capture_time_authority
            ):
                return self._error(409, "reference_source_conflict")
            if previous is not None and previous.snapshot != snapshot:
                # Retention is a projection of current policy, not part of the
                # immutable source identity. Preserve the exact prior observation
                # when an idempotent durability retry crosses the retention edge.
                if retain_image or previous.snapshot is None:
                    return self._error(409, "reference_source_conflict")
                snapshot = previous.snapshot
            temp_name = None
            if retain_image:
                try:
                    temp_name = self._stage_reference_source_image(
                        snapshot, image_bytes, request.image_sha256
                    )
                except (OSError, ValueError):
                    return self._error(503, "reference_source_persistence_failed")
            if previous is None:
                observation = CameraReferenceSourceObservation(
                    schema_version=1,
                    source_evidence_authority="current_immutable_original_revision",
                    reestablished_at=now,
                    scope=self._reference_scope(),
                    candidate_id=request.candidate_id,
                    file_id=manifest.file_id,
                    source_revision=manifest.rev,
                    image_sha256=request.image_sha256,
                    capture_time=request.capture_time,
                    capture_time_authority=request.capture_time_authority,
                    snapshot=snapshot,
                )
                attempt["reference_source"] = observation.model_dump(mode="json")
                try:
                    self._save_attempts()
                except OSError:
                    # Atomic replacement can have reached disk before a later
                    # fsync failed. Preserve the exact current observation only
                    # when a bounded reread proves it was written.
                    try:
                        durable = json.loads(_read_regular(self._state_path, _MAX_JOURNAL))
                        stored = durable.get("attempts", {}).get(request.candidate_id, {}).get(
                            "reference_source"
                        )
                    except (OSError, ValueError, TypeError, AttributeError):
                        stored = None
                    if stored != observation.model_dump(mode="json"):
                        attempt.pop("reference_source", None)
                    else:
                        self._reference_source_pending_commit.add(request.candidate_id)
                    return self._error(503, "reference_source_persistence_failed")
            else:
                observation = previous
                if request.candidate_id in self._reference_source_pending_commit:
                    try:
                        self._save_attempts()
                    except OSError:
                        return self._error(503, "reference_source_persistence_failed")
                    self._reference_source_pending_commit.discard(request.candidate_id)
            if retain_image:
                try:
                    self._publish_reference_source_image(
                        snapshot, request.image_sha256, temp_name
                    )
                except (OSError, ValueError):
                    return self._error(503, "reference_source_persistence_failed")
            return 200, {
                "status": "source_reestablished",
                "candidate_id": request.candidate_id,
                "source_evidence_authority": observation.source_evidence_authority,
                "reestablished_at": observation.reestablished_at.astimezone(timezone.utc).isoformat(),
            }

    def _reference_source_conflicts(
        self, attempt: dict, request: CameraReferenceSourceRequest, manifest: ManifestV2
    ) -> bool:
        if attempt.get("image_sha256") not in {None, request.image_sha256}:
            return True
        capture = self._attempt_capture_time(attempt)
        if capture is not None and (
            capture.astimezone(timezone.utc) != request.capture_time.astimezone(timezone.utc)
            or attempt.get("capture_time_authority") != request.capture_time_authority
        ):
            return True
        request_identity = attempt.get("request_identity")
        if isinstance(request_identity, dict):
            try:
                original = CameraCandidateRequest.model_validate(request_identity)
            except ValidationError:
                return True
            if (
                original.candidate_id != request.candidate_id
                or original.source_revision != request.source_revision
                or original.manifest_sha256 != request.manifest_sha256
                or original.image_sha256 != request.image_sha256
                or original.capture_time.astimezone(timezone.utc)
                != request.capture_time.astimezone(timezone.utc)
                or original.capture_time_authority != request.capture_time_authority
            ):
                return True
        recovery = attempt.get("legacy_base_recovery")
        if isinstance(recovery, dict):
            observed = recovery.get("snapshot_sha256_observed")
            if isinstance(observed, str) and observed != request.image_sha256:
                return True
        retained_revision = attempt.get("source_revision")
        if isinstance(retained_revision, str) and retained_revision != manifest.rev:
            return True
        retained_manifest = attempt.get("manifest_sha256")
        if isinstance(retained_manifest, str) and retained_manifest != request.manifest_sha256:
            return True
        return False

    def _stage_reference_source_image(
        self, snapshot: str, image: bytes, image_sha256: str
    ) -> str | None:
        target = Path(snapshot)
        if target.parent != self._state_dir / "snapshots":
            raise ValueError("reference snapshot path is outside managed storage")
        temp_name = f".{target.name}.{image_sha256}.source-tmp"
        state_fd = self._open_state_dir(create=True)
        try:
            snapshots_fd = _child_directory(state_fd, "snapshots", create=True)
            try:
                target_ok = False
                try:
                    retained = _read_regular(target.name, _MAX_IMAGE, dir_fd=snapshots_fd)
                    target_ok = hmac.compare_digest(hashlib.sha256(retained).hexdigest(), image_sha256)
                except FileNotFoundError:
                    pass
                except (OSError, ValueError):
                    pass
                if target_ok:
                    return None
                staged = None
                try:
                    staged = _read_regular(temp_name, _MAX_IMAGE, dir_fd=snapshots_fd)
                except FileNotFoundError:
                    pass
                except (OSError, ValueError):
                    try:
                        os.unlink(temp_name, dir_fd=snapshots_fd)
                    except FileNotFoundError:
                        pass
                if staged is None or not hmac.compare_digest(
                    hashlib.sha256(staged).hexdigest(), image_sha256
                ):
                    try:
                        os.unlink(temp_name, dir_fd=snapshots_fd)
                    except FileNotFoundError:
                        pass
                    descriptor = os.open(
                        temp_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=snapshots_fd,
                    )
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(image)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.fsync(snapshots_fd)
                return temp_name
            finally:
                os.close(snapshots_fd)
        finally:
            os.close(state_fd)

    def _publish_reference_source_image(
        self, snapshot: str, image_sha256: str, temp_name: str | None
    ) -> None:
        target = Path(snapshot)
        state_fd = self._open_state_dir(create=True)
        try:
            snapshots_fd = _child_directory(state_fd, "snapshots", create=True)
            try:
                if temp_name is not None:
                    os.replace(temp_name, target.name, src_dir_fd=snapshots_fd, dst_dir_fd=snapshots_fd)
                    os.fsync(snapshots_fd)
                verified = _read_regular(target.name, _MAX_IMAGE, dir_fd=snapshots_fd)
                if not hmac.compare_digest(hashlib.sha256(verified).hexdigest(), image_sha256):
                    raise ValueError("published reference original failed verification")
            finally:
                os.close(snapshots_fd)
        finally:
            os.close(state_fd)

    def trusted_capture_time_for_answer(self, message: InboundMessage) -> datetime | None:
        """Resolve capture evidence only for this ingress-bound owner answer turn."""
        metadata = message.metadata
        if (
            not self.config.enabled
            or message.channel != "telegram"
            or str(message.chat_id) != self.config.chat_id
            or message.sender_id.split("|", 1)[0] != self.config.principal
            or metadata.get("_camera_authority") is not CAMERA_AUTHORITY
            or metadata.get("_camera_answer") != "yes"
        ):
            return None
        candidate_id = metadata.get("_camera_candidate_id")
        attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
        if (
            attempt is None
            or attempt.get("state") not in {"answering", "final_queued", "delivery_unknown"}
            or (
                attempt.get("state") == "delivery_unknown"
                and attempt.get("photo_delivery_confirmed") is not True
            )
            or not isinstance(metadata.get("_camera_turn_id"), str)
            or not metadata["_camera_turn_id"]
            or metadata["_camera_turn_id"] != attempt.get("answer_turn_id")
        ):
            return None
        return self._attempt_capture_time(attempt)

    def record_committed_meal(
        self,
        message: InboundMessage,
        receipt: object,
        expected_nutrition: Mapping[str, object] | None,
        *,
        legacy_reconciliation: bool = False,
    ) -> dict:
        """Persist target identities only from a validated durable append receipt."""
        metadata = getattr(receipt, "assistant_metadata", None)
        candidate_id = message.metadata.get("_camera_candidate_id")
        turn_id = message.metadata.get("_camera_turn_id")
        attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
        assistant_id = getattr(receipt, "assistant_message_id", None)
        assistant_op = getattr(receipt, "assistant_client_op_id", None)
        if (
            not isinstance(metadata, Mapping)
            or attempt is None
            or (
                attempt.get("state") not in {"answering", "final_queued", "delivery_unknown"}
                and not (legacy_reconciliation and attempt.get("state") == "completed")
            )
            or (
                attempt.get("state") == "delivery_unknown"
                and attempt.get("photo_delivery_confirmed") is not True
            )
            or turn_id not in {attempt.get("answer_turn_id"), attempt.get("final_turn_id")}
            or not isinstance(assistant_id, str) or not assistant_id
            or not isinstance(assistant_op, str) or not assistant_op
            or getattr(receipt, "user_client_op_id", None) != f"{turn_id}:user"
            or assistant_op != f"{turn_id}:assistant"
            or metadata.get("role") != "assistant"
            or metadata.get("client_op_id") != assistant_op
            or metadata.get("logical_turn_id") != turn_id
            or metadata.get("tenant_id") != self.config.tenant_id
            or metadata.get("source_principal") != f"telegram:{self.config.principal}"
            or metadata.get("camera_candidate_id") != candidate_id
            or (not legacy_reconciliation and metadata.get("camera_operation_id") != candidate_id)
            or metadata.get("camera_answer_bound") != "yes"
            or (not legacy_reconciliation and metadata.get("ingest_source") != "dropbox_camera")
            or (not legacy_reconciliation and metadata.get("confirmation_required") is not True)
            or not isinstance(metadata.get("source_message_id"), str)
            or not metadata["source_message_id"]
        ):
            raise ValueError("Camera commit receipt does not prove the bound owner operation")
        trace = metadata.get("decision_trace")
        annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
        nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
        if (
            not isinstance(nutrition, Mapping)
            or nutrition.get("record_type") != "meal_observation"
            or nutrition.get("consumption_status") != "consumed"
            or "image" not in nutrition.get("basis", [])
        ):
            raise ValueError("Camera commit receipt does not contain a consumed image event")
        from ohmo.evals.nutrition_trace import NutritionAnnotationV2

        observed_annotation = NutritionAnnotationV2.model_validate(nutrition)
        if expected_nutrition is not None:
            expected_annotation = NutritionAnnotationV2.model_validate(expected_nutrition)
        else:
            expected_annotation = observed_annotation
        if observed_annotation.model_dump(mode="json") != expected_annotation.model_dump(mode="json"):
            raise ValueError("Camera commit receipt nutrition differs from the validated finalizer")
        if (
            observed_annotation.meal_at != self._attempt_capture_time(attempt)
            or observed_annotation.meal_date is not None
            or observed_annotation.explicit_new_consumption
        ):
            raise ValueError("Camera commit receipt lacks trusted replay and capture semantics")
        commit = {
            "event_id": assistant_id,
            "source_message_id": metadata["source_message_id"],
            "client_op_id": assistant_op,
            "candidate_id": candidate_id,
            "tenant_id": self.config.tenant_id,
            "principal": self.config.principal,
            "meal_at": nutrition.get("meal_at"),
            "record_type": nutrition.get("record_type"),
            "consumption_status": nutrition.get("consumption_status"),
        }
        previous_commit = attempt.get("camera_commit")
        if isinstance(previous_commit, dict) and previous_commit != commit:
            raise ValueError("Camera operation has conflicting durable commit evidence")
        attempt["camera_commit"] = commit
        attempt["finalizer_status"] = "committed"
        self._save_attempts()
        return commit

    def mark_finalizer_unknown(self, candidate_id: str, turn_id: str) -> None:
        attempt = self._attempts.get(candidate_id)
        if (
            attempt is not None
            and turn_id in {attempt.get("answer_turn_id"), attempt.get("final_turn_id")}
            and attempt.get("finalizer_status") != "committed"
        ):
            attempt["finalizer_status"] = "unknown"
            self._save_attempts()

    def authorize_recovered_camera_denial(
        self, message: InboundMessage, candidate_id: str, original_turn: str
    ) -> str:
        """Bind a current authenticated denial only after the old meal is proven."""
        attempt = self._attempts.get(candidate_id)
        correction_turn = attempt.get("camera_correction_turn_id") if attempt else None
        if (
            attempt is None
            or not isinstance(attempt.get("camera_commit"), dict)
            or attempt.get("camera_correction") != "reconciling_original"
            or original_turn not in {attempt.get("answer_turn_id"), attempt.get("final_turn_id")}
            or message.channel != "telegram"
            or str(message.chat_id) != self.config.chat_id
            or message.sender_id.split("|", 1)[0] != self.config.principal
            or message.metadata.get("_camera_answer") != "no"
            or not isinstance(correction_turn, str)
            or not correction_turn
        ):
            raise ValueError("Camera denial is not bound to a recovered owner operation")
        attempt["camera_correction"] = "answering"
        message.metadata.pop("_camera_legacy_reconcile", None)
        message.metadata.pop("_camera_reconcile_then_correction", None)
        message.metadata["_camera_correction"] = CAMERA_AUTHORITY
        message.metadata["_camera_turn_id"] = correction_turn
        self._save_attempts()
        return correction_turn

    def mark_reconciled_meal_ready(self, candidate_id: str, turn_id: str) -> None:
        attempt = self._attempts.get(candidate_id)
        if (
            attempt is None
            or not isinstance(attempt.get("camera_commit"), dict)
            or turn_id not in {attempt.get("answer_turn_id"), attempt.get("final_turn_id")}
        ):
            raise ValueError("Camera commit is not bound to the retained finalizer")
        attempt["finalizer_status"] = "committed"
        attempt["attention_active"] = False
        if attempt.get("state") == "answering":
            attempt["final_turn_id"] = turn_id
            attempt["state"] = "final_queued"
        self._save_attempts()

    def recover_legacy_committed_meal(self, candidate_id: str, turn_id: str, receipt: object) -> dict:
        attempt = self._attempts.get(candidate_id)
        if (
            attempt is None or attempt.get("state") != "completed"
            or turn_id not in {attempt.get("answer_turn_id"), attempt.get("final_turn_id")}
        ):
            raise ValueError("legacy Camera operation has no retained stable turn identity")
        message = InboundMessage(
            channel="telegram", sender_id=self.config.principal, chat_id=self.config.chat_id,
            content="", metadata={
                "_camera_authority": CAMERA_AUTHORITY,
                "_camera_candidate_id": candidate_id,
                "_camera_answer": "yes",
                "_camera_turn_id": turn_id,
            },
        )
        return self.record_committed_meal(
            message, receipt, None, legacy_reconciliation=True
        )

    def record_committed_correction(
        self,
        message: InboundMessage,
        receipt: object,
        expected_nutrition: Mapping[str, object] | None = None,
    ) -> dict:
        original = self.validate_correction_binding(
            message,
            principal=self.config.principal,
            chat_id=getattr(self.config, "chat_id", message.chat_id),
            tenant_id=self.config.tenant_id,
        )
        metadata = getattr(receipt, "assistant_metadata", None)
        candidate_id = message.metadata.get("_camera_candidate_id")
        turn_id = message.metadata.get("_camera_turn_id")
        attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
        assistant_id = getattr(receipt, "assistant_message_id", None)
        assistant_op = getattr(receipt, "assistant_client_op_id", None)
        if (
            not isinstance(metadata, Mapping)
            or not isinstance(original, dict)
            or attempt.get("camera_correction") != "answering"
            or attempt.get("camera_correction_turn_id") != turn_id
            or not isinstance(assistant_id, str) or not assistant_id
            or not isinstance(assistant_op, str) or not assistant_op
            or getattr(receipt, "user_client_op_id", None) != f"{turn_id}:user"
            or assistant_op != f"{turn_id}:assistant"
            or metadata.get("role") != "assistant"
            or metadata.get("client_op_id") != assistant_op
            or metadata.get("logical_turn_id") != turn_id
            or metadata.get("tenant_id") != original.get("tenant_id")
            or metadata.get("source_principal") != f"telegram:{original.get('principal')}"
            or metadata.get("camera_candidate_id") != candidate_id
            or metadata.get("camera_operation_id") != candidate_id
            or metadata.get("camera_answer_bound") != "no"
            or metadata.get("camera_correction_bound") is not True
            or metadata.get("camera_original_event_id") != original.get("event_id")
            or metadata.get("ingest_source") != "dropbox_camera"
            or metadata.get("confirmation_required") is not True
            or metadata.get("reply_to_source_message_id") != original.get("source_message_id")
        ):
            raise ValueError("Camera correction receipt does not match its committed target")
        trace = metadata.get("decision_trace")
        annotations = trace.get("annotations") if isinstance(trace, Mapping) else None
        nutrition = annotations.get("nutrition") if isinstance(annotations, Mapping) else None
        if (
            not isinstance(nutrition, Mapping)
            or nutrition.get("record_type") != "meal_correction"
            or nutrition.get("consumption_status") != "not_consumed"
            or "consumption_status" not in nutrition.get("changed_fields", [])
        ):
            raise ValueError("Camera correction receipt is not a validated denial patch")
        observed_annotation = _validate_camera_correction_annotation(nutrition)
        if expected_nutrition is not None:
            expected_annotation = _validate_camera_correction_annotation(expected_nutrition)
            expected_changes = list(expected_annotation.changed_fields)
            if (
                observed_annotation.schema_version != expected_annotation.schema_version
                or observed_annotation.record_type != expected_annotation.record_type
                or observed_annotation.consumption_status != expected_annotation.consumption_status
                or list(observed_annotation.changed_fields) != expected_changes
                or any(
                    getattr(observed_annotation, field_name)
                    != getattr(expected_annotation, field_name)
                    for field_name in expected_changes
                )
            ):
                raise ValueError("Camera correction receipt differs from the validated correction")
        correction = {
            "event_id": assistant_id,
            "source_message_id": metadata.get("source_message_id"),
            "client_op_id": assistant_op,
            "target_event_id": original["event_id"],
            "target_source_message_id": original["source_message_id"],
        }
        previous_correction = attempt.get("camera_correction_commit")
        if isinstance(previous_correction, dict) and previous_correction != correction:
            raise ValueError("Camera operation has conflicting correction commit evidence")
        attempt["camera_correction_commit"] = correction
        attempt["camera_correction"] = "final_queued"
        attempt["camera_correction_final_turn_id"] = turn_id
        self._save_attempts()
        return correction

    def validate_correction_binding(
        self,
        message: InboundMessage,
        *,
        principal: str,
        chat_id: str,
        tenant_id: str,
    ) -> dict:
        """Check the retained correction target against the current trusted scope."""
        metadata = message.metadata
        candidate_id = metadata.get("_camera_candidate_id")
        attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
        original = attempt.get("camera_commit") if attempt is not None else None
        turn_id = metadata.get("_camera_turn_id")
        if (
            not isinstance(original, dict)
            or metadata.get("_camera_authority") is not CAMERA_AUTHORITY
            or metadata.get("_camera_correction") is not CAMERA_AUTHORITY
            or metadata.get("_camera_answer") != "no"
            or attempt.get("state") not in {"answering", "final_queued", "completed", "delivery_unknown"}
            or attempt.get("camera_correction") != "answering"
            or attempt.get("camera_correction_turn_id") != turn_id
            or not isinstance(turn_id, str)
            or not turn_id
            or message.channel != "telegram"
            or str(message.chat_id) != str(chat_id)
            or message.sender_id.split("|", 1)[0] != str(principal)
            or original.get("candidate_id") != candidate_id
            or not isinstance(original.get("event_id"), str)
            or not original.get("event_id")
            or not isinstance(original.get("source_message_id"), str)
            or not original.get("source_message_id")
            or not isinstance(original.get("client_op_id"), str)
            or not original.get("client_op_id")
            or original.get("client_op_id") not in {
                f"{attempt.get('answer_turn_id')}:assistant",
                f"{attempt.get('final_turn_id')}:assistant",
            }
            or original.get("principal") != str(principal)
            or original.get("tenant_id") != str(tenant_id)
        ):
            raise ValueError("Camera correction is not bound to the current committed target")
        return original

    def _load_attempts(self) -> tuple[dict[str, dict], dict | None]:
        self._journal_present = False
        self._journal_schema_version = None
        try:
            state_fd = self._open_state_dir(create=False)
        except FileNotFoundError:
            return {}, None
        try:
            try:
                payload = json.loads(_read_regular("attempts.json", _MAX_JOURNAL, dir_fd=state_fd))
            except FileNotFoundError:
                return {}, None
        finally:
            os.close(state_fd)
        self._journal_present = True
        session = None
        if isinstance(payload, dict) and payload.get("schema_version") == 2:
            self._journal_schema_version = 2
            if set(payload) != {"schema_version", "attempts", "session"}:
                raise ValueError("camera attempt journal is invalid")
            attempts = payload["attempts"]
            session = payload["session"]
            if (
                not isinstance(session, dict)
                or _SESSION_ID.fullmatch(session.get("session_id", "")) is None
                or _SESSION_ID.fullmatch(session.get("epoch", "")) is None
                or not isinstance(session.get("committed_seq"), int)
                or session["committed_seq"] < 0
            ):
                raise ValueError("camera session journal is invalid")
        else:
            # Read the previous candidate-keyed journal during the protocol upgrade.
            attempts = payload
        if not isinstance(attempts, dict) or len(attempts) > _MAX_ATTEMPTS:
            raise ValueError("camera attempt journal is invalid")
        self._journal_migrated = False
        for key, value in attempts.items():
            validate_candidate_id(key)
            if not isinstance(value, dict) or value.get("state") not in {
                "admitted",
                "duplicate",
                "retired",
                "photo_sent",
                "clarifying",
                "answering",
                "final_queued",
                "completed",
                "delivery_unknown",
            }:
                raise ValueError("camera attempt journal is invalid")
            if "_camera_caption" in value and (
                not isinstance(value["_camera_caption"], str)
                or not value["_camera_caption"].startswith("Съели ли вы это? ")
                or len(value["_camera_caption"]) > 256
            ):
                raise ValueError("camera attempt caption is invalid")
            if "confirmed_camera_context" in value and (
                not isinstance(value["confirmed_camera_context"], str)
                or not value["confirmed_camera_context"].strip()
                or len(value["confirmed_camera_context"]) > 2048
                or value.get("prompt_edit_claimed") is not True
                or value.get("photo_delivery_confirmed") is not True
            ):
                raise ValueError("camera attempt context is invalid")
            # Older tombstones have no capture evidence. Keep them, but never
            # infer a meal time from admission or reply arrival.
            if "capture_time" in value or "capture_time_authority" in value:
                if self._attempt_capture_time(value) is None:
                    raise ValueError("camera attempt journal capture time is invalid")
            if "image_sha256" in value and (
                not isinstance(value["image_sha256"], str)
                or _SHA256.fullmatch(value["image_sha256"]) is None
            ):
                raise ValueError("camera attempt journal image digest is invalid")
            if value.get("phash_algorithm") is not None and (
                value.get("phash_algorithm") != PHASH_ALGORITHM
                or not isinstance(value.get("image_phash"), str)
                or re.fullmatch(r"[0-9a-f]{64}", value["image_phash"]) is None
            ):
                raise ValueError("camera attempt journal pHash is invalid")
        now_iso = datetime.now(timezone.utc).isoformat()
        for candidate_id, value in attempts.items():
            if value.get("state") == "retired":
                try:
                    retired_time = datetime.fromisoformat(value["retired_at"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("Camera retired record timestamp is invalid") from error
                if retired_time.tzinfo is None or retired_time.utcoffset() is None:
                    raise ValueError("Camera retired record timestamp has no timezone")
            elif not isinstance(value.get("admitted_at"), str):
                # Legacy in-flight entries adopt this process's start as their
                # admission time, so the pending TTL has a bounded runway.
                value["admitted_at"] = now_iso
                self._journal_migrated = True
            # Safe in-place conversion for journals written before attention
            # was separated from retained operation identity.
            if "attention_active" not in value:
                value["attention_active"] = value.get("state") not in {
                    "completed", "delivery_unknown"
                }
                self._journal_migrated = True
            if "photo_delivery_confirmed" not in value:
                photo_id = value.get("photo_id")
                value["photo_delivery_confirmed"] = (
                    isinstance(photo_id, int) and not isinstance(photo_id, bool) and photo_id > 0
                )
                self._journal_migrated = True
            if type(value.get("attention_active")) is not bool or type(
                value.get("photo_delivery_confirmed")
            ) is not bool:
                raise ValueError("camera attempt journal state is invalid")
            if "context_interrupted" in value and type(value["context_interrupted"]) is not bool:
                raise ValueError("camera attempt context state is invalid")
            request_identity = value.get("request_identity")
            request_ack = value.get("request_ack")
            if request_identity is not None:
                try:
                    bound_request = CameraCandidateRequest.model_validate(request_identity)
                except ValidationError as error:
                    raise ValueError("camera attempt request identity is invalid") from error
                if bound_request.candidate_id != candidate_id:
                    raise ValueError("camera attempt request identity has the wrong candidate")
                if request_ack is not None and (
                    not isinstance(request_ack, dict)
                    or type(request_ack.get("status")) is not int
                    or not isinstance(request_ack.get("body"), dict)
                    or request_ack["body"].get("candidate_id") != bound_request.candidate_id
                    or request_ack["body"].get("session_id") != bound_request.session_id
                    or request_ack["body"].get("epoch") != bound_request.epoch
                    or request_ack["body"].get("ack_seq") != bound_request.seq
                ):
                    raise ValueError("camera attempt request ACK is invalid")
            elif request_ack is not None:
                raise ValueError("camera attempt ACK has no request identity")
            if value.get("reference_source") is not None:
                try:
                    source = CameraReferenceSourceObservation.model_validate(
                        value["reference_source"]
                    )
                except ValidationError as error:
                    raise ValueError("Camera current reference source is invalid") from error
                if (
                    source.candidate_id != candidate_id
                    or candidate_id_for(source.file_id, source.source_revision) != candidate_id
                    or value.get("photo_delivery_confirmed") is not True
                    or type(value.get("photo_id")) is not int
                    or value["photo_id"] <= 0
                ):
                    raise ValueError("Camera current reference source is not receipt-bound")
                retained_capture = self._attempt_capture_time(value)
                if retained_capture is not None and (
                    retained_capture.astimezone(timezone.utc)
                    != source.capture_time.astimezone(timezone.utc)
                    or value.get("capture_time_authority") != source.capture_time_authority
                ):
                    raise ValueError("Camera current reference source contradicts capture evidence")
                if value.get("image_sha256") not in {None, source.image_sha256}:
                    raise ValueError("Camera current reference source contradicts image evidence")
                if request_identity is not None:
                    if (
                        bound_request.source_revision != source.source_revision
                        or (
                            value.get("manifest_sha256") is not None
                            and bound_request.manifest_sha256 != value.get("manifest_sha256")
                        )
                        or bound_request.image_sha256 != source.image_sha256
                        or bound_request.capture_time.astimezone(timezone.utc)
                        != source.capture_time.astimezone(timezone.utc)
                        or bound_request.capture_time_authority != source.capture_time_authority
                    ):
                        raise ValueError("Camera current reference source contradicts request evidence")
                if source.snapshot is not None:
                    snapshot_path = Path(source.snapshot)
                    expected_name = self._safe_snapshot_name(value, snapshot_path.suffix.lower())
                    if (
                        snapshot_path.parent != self._state_dir / "snapshots"
                        or snapshot_path.name != expected_name
                    ):
                        raise ValueError("Camera current reference snapshot path is invalid")
            if value.get("state") == "retired" and (
                request_identity is None
                or not isinstance(request_ack, dict)
                or request_ack.get("status") != 200
                or not isinstance(request_ack.get("body"), dict)
                or request_ack.get("body", {}).get("status") != "retired"
            ):
                raise ValueError("Camera retired record is incomplete")
            if value["photo_delivery_confirmed"] is True and "image_sha256" not in value:
                snapshot = value.get("snapshot")
                admission_id = value.get("admission_id")
                if isinstance(snapshot, str) and isinstance(admission_id, str):
                    path = Path(snapshot)
                    try:
                        if (
                            path.parent == self._state_dir / "snapshots"
                            and path.name in {
                                f"{admission_id}.jpg",
                                f"{admission_id}.jpeg",
                                f"{admission_id}.png",
                                f"{admission_id}.webp",
                            }
                        ):
                            retained_bytes = _read_regular(path, _MAX_IMAGE)
                            fingerprint = fingerprint_image_bytes(retained_bytes)
                            value["image_sha256"] = hashlib.sha256(retained_bytes).hexdigest()
                            value["image_phash"] = (
                                fingerprint.get("phash") if fingerprint is not None else None
                            )
                            value["phash_algorithm"] = (
                                fingerprint.get("phash_algorithm") if fingerprint is not None else None
                            )
                            self._journal_migrated = True
                    except (OSError, ValueError):
                        pass
        # The exact pre-gen5 schema kept the last producer ACK only in the
        # session journal and omitted request/artifact hashes from attempts.
        # Preserve a narrowly provable admitted ACK before __init__ rotates
        # the lease epoch.  The snapshot digest is explicitly observed now;
        # it is not represented as a historical journal field.
        if isinstance(session, dict):
            last_ack = session.get("last_ack")
            body = last_ack.get("body") if isinstance(last_ack, dict) else None
            if (
                isinstance(last_ack, dict)
                and last_ack.get("status") == 202
                and isinstance(body, dict)
                and body.get("status") == "admitted"
                and isinstance(body.get("candidate_id"), str)
                and isinstance(body.get("admission_id"), str)
                and body.get("session_id") == session.get("session_id")
                and body.get("epoch") == session.get("epoch")
                and body.get("ack_seq") == last_ack.get("seq")
                and last_ack.get("seq") == session.get("committed_seq")
                and isinstance(last_ack.get("seq"), int)
                and not isinstance(last_ack.get("seq"), bool)
                and last_ack["seq"] > 0
            ):
                candidate_id = body["candidate_id"]
                attempt = attempts.get(candidate_id)
                if (
                    isinstance(attempt, dict)
                    and attempt.get("request_identity") is None
                    and attempt.get("state") in {
                        "admitted", "photo_sent", "answering", "final_queued",
                        "completed", "delivery_unknown",
                    }
                    and attempt.get("admission_id") == body["admission_id"]
                    and attempt.get("capture_time_authority") in {"exif", "filename"}
                    and self._attempt_capture_time(attempt) is not None
                ):
                    snapshot = attempt.get("snapshot")
                    admission_id = attempt.get("admission_id")
                    if isinstance(snapshot, str) and isinstance(admission_id, str):
                        path = Path(snapshot)
                        if (
                            path.parent == self._state_dir / "snapshots"
                            and path.name in {
                                f"{admission_id}.jpg", f"{admission_id}.jpeg",
                                f"{admission_id}.png", f"{admission_id}.webp",
                            }
                        ):
                            try:
                                image = _read_regular(path, _MAX_IMAGE)
                                attempt["legacy_base_recovery"] = {
                                    "proof": "base_session_ack_and_verified_managed_snapshot",
                                    "snapshot_sha256_observed": hashlib.sha256(image).hexdigest(),
                                    "ack_status": last_ack["status"],
                                    "ack_body": body,
                                    "session_id": session["session_id"],
                                    "epoch": session["epoch"],
                                    "seq": last_ack["seq"],
                                }
                                self._journal_migrated = True
                            except (OSError, ValueError):
                                pass
        for candidate_id, attempt in attempts.items():
            recovery = attempt.get("legacy_base_recovery")
            identity_proof = attempt.get("request_identity_proof")
            if recovery is None:
                if identity_proof is not None:
                    raise ValueError("Camera legacy recovery proof is incomplete")
                continue
            if (
                not isinstance(recovery, dict)
                or recovery.get("proof") != "base_session_ack_and_verified_managed_snapshot"
                or not isinstance(recovery.get("snapshot_sha256_observed"), str)
                or _SHA256.fullmatch(recovery["snapshot_sha256_observed"]) is None
                or recovery.get("ack_status") != 202
                or not isinstance(recovery.get("ack_body"), dict)
                or recovery["ack_body"].get("status") != "admitted"
                or recovery["ack_body"].get("candidate_id") != candidate_id
                or recovery["ack_body"].get("admission_id") != attempt.get("admission_id")
                or recovery["ack_body"].get("session_id") != recovery.get("session_id")
                or recovery["ack_body"].get("epoch") != recovery.get("epoch")
                or recovery["ack_body"].get("ack_seq") != recovery.get("seq")
                or not isinstance(recovery.get("session_id"), str)
                or not isinstance(recovery.get("epoch"), str)
                or _SESSION_ID.fullmatch(recovery.get("session_id", "")) is None
                or _SESSION_ID.fullmatch(recovery.get("epoch", "")) is None
                or not isinstance(recovery.get("seq"), int)
                or isinstance(recovery.get("seq"), bool)
                or recovery["seq"] < 1
            ):
                raise ValueError("Camera legacy recovery evidence is invalid")
            if identity_proof is None:
                if attempt.get("request_identity") is not None or attempt.get("request_ack") is not None:
                    raise ValueError("Camera legacy recovery request binding is incomplete")
                continue
            if identity_proof != "legacy_ack_snapshot_sha_and_observed_recovery_input":
                raise ValueError("Camera legacy recovery request proof is invalid")
            if not isinstance(attempt.get("request_identity"), dict):
                raise ValueError("Camera legacy recovery request identity is absent")
            bound_request = CameraCandidateRequest.model_validate(attempt["request_identity"])
            ack = attempt.get("request_ack")
            capture = self._attempt_capture_time(attempt)
            if (
                bound_request.candidate_id != candidate_id
                or bound_request.session_id != recovery["session_id"]
                or bound_request.epoch != recovery["epoch"]
                or bound_request.seq != recovery["seq"]
                or capture is None
                or bound_request.capture_time.astimezone(timezone.utc)
                != capture.astimezone(timezone.utc)
                or bound_request.capture_time_authority != attempt.get("capture_time_authority")
                or bound_request.image_sha256 != recovery["snapshot_sha256_observed"]
                or not isinstance(ack, dict)
                or ack.get("status") != recovery["ack_status"]
                or ack.get("body") != recovery["ack_body"]
            ):
                raise ValueError("Camera legacy recovery request binding is invalid")
        return attempts, session

    def _open_state_dir(self, *, create: bool) -> int:
        workspace_fd = os.open(self._workspace, _directory_flags())
        try:
            return _child_directory(workspace_fd, "camera_ingress", create=create)
        finally:
            os.close(workspace_fd)

    def _save_attempts(self) -> None:
        state_fd = self._open_state_dir(create=True)
        temporary = f".{uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=state_fd
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "schema_version": 2,
                        "attempts": self._attempts,
                        "session": self._session,
                    },
                    stream,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, "attempts.json", src_dir_fd=state_fd, dst_dir_fd=state_fd)
            os.fsync(state_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=state_fd)
            except FileNotFoundError:
                pass
            os.close(state_fd)

    _TTL_SWEEPABLE_STATES = frozenset({"photo_sent", "clarifying", "answering", "final_queued"})

    def _sweep_expired_attempts(self) -> None:
        """Release attention after the TTL while retaining addressable operations.

        One ignored photo must not gate the whole chat forever: after the TTL
        the camera flow degrades to ordinary text turns instead of marking
        every inbound message unbound. ``admitted`` and ``delivery_unknown``
        tombstones are never swept: they are the at-most-once admission
        markers for sends whose outcome is unknown.
        """
        now = datetime.now(timezone.utc)
        expired = [
            value
            for value in self._attempts.values()
            if value.get("attention_active", True)
            and value.get("state") in self._TTL_SWEEPABLE_STATES
            and _attempt_age_seconds(value, now) >= _PENDING_TTL_SECONDS
        ]
        if not expired:
            return
        for value in expired:
            value["attention_active"] = False
        try:
            self._save_attempts()
        except OSError:
            logging.getLogger(__name__).warning(
                "failed to persist Camera attempt expiry", exc_info=True
            )

    def _interrupt_untargeted_camera_context(self) -> None:
        changed = False
        for attempt in self._attempts.values():
            if attempt.get("state") in {"photo_sent", "clarifying"} and not attempt.get(
                "context_interrupted", False
            ):
                attempt["context_interrupted"] = True
                changed = True
        if changed:
            try:
                self._save_attempts()
            except OSError:
                logger.warning("failed to persist Camera context interruption", exc_info=True)

    def _remove_snapshot_files(self, attempt: dict) -> None:
        admission_id = attempt.get("admission_id")
        if not isinstance(admission_id, str):
            return
        snapshots = [attempt.get("snapshot")]
        raw_source = attempt.get("reference_source")
        if isinstance(raw_source, dict):
            snapshots.append(raw_source.get("snapshot"))
        valid_names = set()
        for snapshot in snapshots:
            if not isinstance(snapshot, str):
                continue
            name = Path(snapshot)
            if name.parent == self._state_dir / "snapshots" and name.name in {
                f"{admission_id}.jpg",
                f"{admission_id}.jpeg",
                f"{admission_id}.png",
                f"{admission_id}.webp",
            }:
                valid_names.add(name.name)
        if not valid_names:
            return
        try:
            state_fd = self._open_state_dir(create=False)
            try:
                snapshots_fd = _child_directory(state_fd, "snapshots", create=False)
                try:
                    for name in valid_names:
                        try:
                            os.unlink(name, dir_fd=snapshots_fd)
                            os.fsync(snapshots_fd)
                        except FileNotFoundError:
                            pass
                finally:
                    os.close(snapshots_fd)
            finally:
                os.close(state_fd)
        except FileNotFoundError:
            return
        except OSError:
            logging.getLogger(__name__).warning("failed to remove Camera snapshot", exc_info=True)

    def _remove_completed_snapshots(self) -> None:
        """Prune completed originals only after their reference window expires.

        Confirmed Camera photos remain canonical repeat-suppression references
        for seven days from trusted source capture. Delivery-unknown attempts
        are not completed and retain their originals through the existing
        state handling; this cleanup does not broaden their lifetime policy.
        """
        now = datetime.now(timezone.utc)
        oldest_reference = now - _DEDUP_WINDOW
        for attempt in list(self._attempts.values()):
            if attempt.get("state") != "completed":
                continue
            captured = self._attempt_capture_time(attempt)
            raw_source = attempt.get("reference_source")
            if captured is None and isinstance(raw_source, dict):
                try:
                    source = CameraReferenceSourceObservation.model_validate(raw_source)
                except ValidationError:
                    source = None
                if source is not None and source.scope == self._reference_scope():
                    captured = source.capture_time
            if captured is not None:
                captured = captured.astimezone(timezone.utc)
                if oldest_reference <= captured <= now:
                    continue
            self._remove_snapshot_files(attempt)

    def mark_restart_unknown(self) -> None:
        """Mark uncertain sends unknown; retain confirmed native photo receipts."""
        changed = False
        for attempt in self._attempts.values():
            if attempt["state"] == "admitted":
                attempt["state"] = "delivery_unknown"
                changed = True
            elif attempt["state"] in {"answering", "final_queued"}:
                if attempt.get("finalizer_status") != "committed":
                    attempt["finalizer_status"] = "unknown"
                    changed = True
        if changed:
            self._save_attempts()

    def _token(self) -> bytes:
        path = self.config.bearer_token_file
        if path is None or path.is_symlink():
            raise ValueError("camera bearer credential is unavailable")
        metadata = path.stat()
        if (
            metadata.st_mode & 0o077
            or metadata.st_uid != getattr(os, "getuid", lambda: metadata.st_uid)()
        ):
            raise ValueError("camera bearer credential must be owner-only")
        token = _read_regular(path, 4096).strip()
        if len(token) < 32 or b"\n" in token or b"\r" in token:
            raise ValueError("camera bearer credential is invalid")
        return token

    def _refresh_expired_session(self) -> bool:
        expires_at = datetime.fromisoformat(self._session["expires_at"])
        if datetime.now(timezone.utc) < expires_at:
            return False
        self._session = self._new_session(session_id=self._session["session_id"])
        return True

    async def lease(self, authorization: str | None) -> tuple[int, dict]:
        """Return the current producer lease, rotating its epoch on expiration."""
        if not self.config.enabled:
            return self._error(403, "camera_disabled")
        try:
            expected = self._token()
        except (OSError, ValueError):
            return self._error(503, "pre_admission_unavailable")
        prefix = "Bearer "
        if (
            not isinstance(authorization, str)
            or not authorization.startswith(prefix)
            or not hmac.compare_digest(authorization[len(prefix) :].encode(), expected)
        ):
            return self._error(401, "unauthorized")
        async with self._lock:
            try:
                self._refresh_expired_session()
                self._save_attempts()
            except (OSError, ValueError):
                return self._error(503, "pre_admission_unavailable")
            return 200, {
                "session_id": self._session["session_id"],
                "epoch": self._session["epoch"],
                "committed_seq": self._session["committed_seq"],
                "expires_at": self._session["expires_at"],
            }

    def _commit_sequence(
        self,
        request: CameraCandidateRequest,
        status: int,
        response: dict,
        *,
        attempt: dict | None = None,
        advance: bool = True,
        update_last_ack: bool | None = None,
    ) -> tuple[int, dict]:
        acknowledged = {
            **response,
            "session_id": request.session_id,
            "epoch": request.epoch,
            "ack_seq": request.seq,
        }
        request_identity = self._request_identity(request)
        if update_last_ack is None:
            update_last_ack = advance
        if advance:
            self._session["committed_seq"] = request.seq
        if update_last_ack:
            self._session["last_ack"] = {
                "seq": request.seq,
                "candidate_id": request.candidate_id,
                "request_identity": request_identity,
                "status": status,
                "body": acknowledged,
            }
        if attempt is not None:
            attempt["request_identity"] = request_identity
            attempt["request_ack"] = {"status": status, "body": acknowledged}
        self._save_attempts()
        return status, acknowledged

    @staticmethod
    def _request_identity(request: CameraCandidateRequest) -> dict[str, object]:
        """Exact candidate artifact and original lease/sequence identity."""
        return request.model_dump(mode="json")

    @classmethod
    def _request_matches_attempt(
        cls, attempt: dict, request: CameraCandidateRequest
    ) -> bool:
        return attempt.get("request_identity") == cls._request_identity(request)

    @staticmethod
    def _validate_source_artifacts(request: object, upload: CameraCandidateUpload):
        """Validate the immutable publication artifacts shared by both routes."""
        manifest_bytes = upload.manifest_bytes
        if hashlib.sha256(manifest_bytes).hexdigest() != request.manifest_sha256:
            raise ValueError("candidate manifest digest differs")
        manifest = ManifestV2.model_validate_json(manifest_bytes)
        if manifest.normalized_capture_time is None:
            raise ValueError("candidate manifest capture time is absent")
        manifest_capture_time = manifest.normalized_capture_time
        if (
            manifest_capture_time.tzinfo is None
            or manifest_capture_time.utcoffset() is None
            or manifest_capture_time.astimezone(timezone.utc)
            != request.capture_time.astimezone(timezone.utc)
        ):
            raise ValueError("candidate manifest capture time differs from request")
        if (
            manifest.candidate_id != request.candidate_id
            or candidate_id_for(manifest.file_id, manifest.rev) != request.candidate_id
            or manifest.rev != request.source_revision
            or manifest.original_sha256 != request.image_sha256
            or manifest.capture_time_authority != request.capture_time_authority
        ):
            raise ValueError("candidate manifest differs from request")
        if manifest.event_id != f"{request.candidate_id}:manifest:v1":
            raise ValueError("candidate manifest event identity differs")
        _published_positive(upload.producer_sidecar_bytes, manifest)
        image_name = upload.image_filename
        suffix = Path(image_name).suffix.lower()
        if (
            Path(image_name).name != image_name
            or "/" in image_name
            or "\\" in image_name
            or image_name != manifest.original_filename
            or suffix
            not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }
        ):
            raise ValueError("candidate image name is invalid")
        mime_for_suffix = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        if (
            manifest.mime_type != mime_for_suffix[suffix]
            or upload.image_content_type != manifest.mime_type
        ):
            raise ValueError("candidate image MIME type differs from filename")
        image_bytes = upload.image_bytes
        if (
            not 0 < len(image_bytes) <= _MAX_IMAGE
            or len(image_bytes) != manifest.original_size_bytes
            or hashlib.sha256(image_bytes).hexdigest() != request.image_sha256
        ):
            raise ValueError("candidate image digest differs")
        return image_bytes, suffix, manifest

    @classmethod
    def _validate_upload(
        cls, request: CameraCandidateRequest, upload: CameraCandidateUpload
    ) -> tuple[bytes, str]:
        image, suffix, _manifest = cls._validate_source_artifacts(request, upload)
        return image, suffix

    async def admit(self, authorization: str | None, upload: object) -> tuple[int, dict]:
        if not self.config.enabled:
            return self._error(403, "camera_disabled")
        try:
            expected = self._token()
        except (OSError, ValueError):
            return self._error(503, "pre_admission_unavailable")
        prefix = "Bearer "
        if (
            not isinstance(authorization, str)
            or not authorization.startswith(prefix)
            or not hmac.compare_digest(authorization[len(prefix) :].encode(), expected)
        ):
            return self._error(401, "unauthorized")
        if not isinstance(upload, CameraCandidateUpload):
            return self._error(400, "invalid_request")
        try:
            request = CameraCandidateRequest.model_validate(upload.request)
        except ValidationError:
            return self._error(400, "invalid_request")
        async with self._lock:
            if self._refresh_expired_session():
                try:
                    self._save_attempts()
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
                return self._error(409, "session_expired")
            if (
                request.session_id != self._session["session_id"]
                or request.epoch != self._session["epoch"]
            ):
                return self._error(409, "session_expired")
            committed_seq = self._session["committed_seq"]
            if request.seq <= committed_seq:
                last_ack = self._session["last_ack"]
                if (
                    isinstance(last_ack, dict)
                    and last_ack.get("seq") == request.seq
                    and last_ack.get("candidate_id") == request.candidate_id
                    and last_ack.get("request_identity") == self._request_identity(request)
                ):
                    try:
                        self._validate_upload(request, upload)
                    except (OSError, ValueError, TypeError, ValidationError):
                        cached_body = last_ack.get("body")
                        if (
                            last_ack.get("status") == 422
                            and isinstance(cached_body, dict)
                            and isinstance(cached_body.get("error"), dict)
                            and cached_body["error"].get("code") == "candidate_evidence_mismatch"
                        ):
                            return last_ack["status"], cached_body
                        return self._error(422, "candidate_evidence_mismatch")
                    try:
                        # A prior write may have failed after mutating memory;
                        # persist the exact ACK before repeating it.
                        self._save_attempts()
                    except OSError:
                        return self._error(503, "pre_admission_unavailable")
                    return last_ack["status"], last_ack["body"]
                return 200, {
                    "status": "stale_sequence",
                    "session_id": self._session["session_id"],
                    "epoch": self._session["epoch"],
                    "ack_seq": committed_seq,
                }
            if request.seq > committed_seq + 1:
                return 409, {
                    "error": {"code": "expected_seq"},
                    "expected_seq": committed_seq + 1,
                }
            previous_attempt = self._attempts.get(request.candidate_id)
            if previous_attempt is not None:
                if previous_attempt.get("state") == "retired":
                    return self._error(409, "candidate_retired")
                if not self._request_matches_attempt(previous_attempt, request):
                    return self._error(409, "candidate_request_mismatch")
                try:
                    self._validate_upload(request, upload)
                except (OSError, ValueError, TypeError, ValidationError):
                    return self._error(422, "candidate_evidence_mismatch")
                stored_ack = previous_attempt.get("request_ack")
                if isinstance(stored_ack, dict) and isinstance(stored_ack.get("body"), dict):
                    return stored_ack["status"], stored_ack["body"]
                if previous_attempt.get("state") == "duplicate":
                    response = {
                        "status": "duplicate",
                        "candidate_id": request.candidate_id,
                        "reason": previous_attempt.get("duplicate_reason"),
                        "duplicate_of": previous_attempt.get("duplicate_of"),
                    }
                    try:
                        return self._commit_sequence(request, 200, response, attempt=previous_attempt)
                    except OSError:
                        return self._error(503, "pre_admission_unavailable")
                admission_id = previous_attempt.get("admission_id")
                if isinstance(admission_id, str) and admission_id:
                    response = {
                        "status": "admitted",
                        "candidate_id": request.candidate_id,
                        "admission_id": admission_id,
                        "delivery_semantics": "in_process_only",
                    }
                    try:
                        return self._commit_sequence(request, 202, response, attempt=previous_attempt)
                    except OSError:
                        return self._error(503, "pre_admission_unavailable")
                try:
                    return self._commit_sequence(
                        request, 409, self._error(409, "candidate_already_attempted")[1]
                    )
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
            if len(self._attempts) >= _MAX_ATTEMPTS:
                return self._error(503, "pre_admission_unavailable")
            self._sweep_expired_attempts()
            try:
                image_bytes, suffix, manifest = self._validate_source_artifacts(request, upload)
            except (OSError, ValueError, TypeError, ValidationError):
                status, response = self._error(422, "candidate_evidence_mismatch")
                try:
                    return self._commit_sequence(request, status, response)
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
            capture_time = request.capture_time.astimezone(timezone.utc)
            now = datetime.now(timezone.utc)
            if capture_time < now - _DEDUP_WINDOW or capture_time > now + timedelta(minutes=5):
                status, response = self._error(422, "candidate_capture_time_out_of_window")
                try:
                    return self._commit_sequence(request, status, response)
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
            descriptor = fingerprint_image_bytes(image_bytes)
            if descriptor is None:
                descriptor = {"sha256": hashlib.sha256(image_bytes).hexdigest()}
            duplicate_of = None
            reason = None
            for prior_id, prior in self._attempts.items():
                prior_time = self._attempt_capture_time(prior)
                try:
                    prior_source = self._reference_source_record(prior_id, prior)
                except ValueError:
                    return self._error(503, "duplicate_evidence_unavailable")
                if prior_time is None and prior_source is not None:
                    prior_time = prior_source.capture_time
                if prior_time is None:
                    if prior_id in self._reference_source_pending_commit:
                        return self._error(503, "duplicate_evidence_unavailable")
                    continue
                if prior.get("photo_delivery_confirmed") is not True:
                    continue
                if abs((capture_time - prior_time.astimezone(timezone.utc)).total_seconds()) > _DEDUP_WINDOW.total_seconds():
                    continue
                source_outside_current_retention = (
                    prior_source is not None
                    and prior_source.capture_time.astimezone(timezone.utc)
                    < now - _DEDUP_WINDOW
                )
                if (
                    prior_id in self._reference_source_pending_commit
                    and not source_outside_current_retention
                ):
                    return self._error(503, "duplicate_evidence_unavailable")
                prior_sha = prior.get("image_sha256")
                if not isinstance(prior_sha, str) and prior_source is not None:
                    prior_sha = prior_source.image_sha256
                prior_phash = prior.get("image_phash")
                prior_algorithm = prior.get("phash_algorithm")
                if not isinstance(prior_sha, str):
                    # Legacy journal evidence may retain one exact snapshot path;
                    # follow only that journal reference, never scan the filesystem.
                    snapshot = prior.get("snapshot")
                    admission_id = prior.get("admission_id")
                    if isinstance(snapshot, str) and isinstance(admission_id, str):
                        path = Path(snapshot)
                        try:
                            if (
                                path.parent == self._state_dir / "snapshots"
                                and path.name in {f"{admission_id}.jpg", f"{admission_id}.jpeg", f"{admission_id}.png", f"{admission_id}.webp"}
                            ):
                                retained = _read_regular(path, _MAX_IMAGE)
                                legacy = fingerprint_image_bytes(retained)
                                prior_sha = hashlib.sha256(retained).hexdigest()
                                if legacy is not None:
                                    prior_phash = legacy.get("phash")
                                    prior_algorithm = legacy.get("phash_algorithm")
                        except (OSError, ValueError):
                            pass
                if prior_sha == descriptor["sha256"]:
                    duplicate_of, reason = prior_id, "image_already_delivered"
                    break
                old_phash = prior_phash
                if (
                    prior_algorithm == PHASH_ALGORITHM
                    and isinstance(old_phash, str)
                    and isinstance(descriptor.get("phash"), str)
                    and phash_hamming_distance(old_phash, descriptor["phash"])
                    is not None
                    and phash_hamming_distance(old_phash, descriptor["phash"])
                    <= PHASH_HAMMING_THRESHOLD
                ):
                    duplicate_of, reason = prior_id, "image_already_delivered"
                    break
            if duplicate_of is None and self._recent_attachments is not None:
                remote_history_incomplete = False
                try:
                    history = await self._recent_attachments(
                        since=capture_time - _DEDUP_WINDOW,
                        until=min(now, capture_time + _DEDUP_WINDOW),
                    )
                    duplicate_of = _find_recent_attachment_duplicate(
                        history,
                        descriptor,
                        since=capture_time - _DEDUP_WINDOW,
                        until=min(now, capture_time + _DEDUP_WINDOW),
                        principal=f"telegram:{self.config.principal}",
                        expected_session=self._recent_session,
                        expected_peer=self._recent_peer,
                        session_key=self.config.session_key,
                        chat_id=self.config.chat_id,
                        expected_tenant=self.config.tenant_id,
                    )
                    if duplicate_of is not None:
                        reason = "human_photo_already_seen"
                except IncompleteAttachmentHistory:
                    # Honcho's bounded metadata projection can omit the tail of
                    # a larger album. Only the exact configured local snapshot
                    # may complete that known partial source.
                    remote_history_incomplete = True
                except Exception:
                    logger.exception("Camera duplicate history unavailable; candidate not sent")
                    return self._error(503, "duplicate_evidence_unavailable")
            if duplicate_of is None and self._retained_attachments is not None:
                try:
                    local_history = await self._retained_attachments(
                        since=capture_time - _DEDUP_WINDOW,
                        until=min(now, capture_time + _DEDUP_WINDOW),
                    )
                    local_duplicate = _find_recent_attachment_duplicate(
                        local_history,
                        descriptor,
                        since=capture_time - _DEDUP_WINDOW,
                        until=min(now, capture_time + _DEDUP_WINDOW),
                        principal=f"telegram:{self.config.principal}",
                        expected_session=None,
                        expected_peer=None,
                        session_key=self.config.session_key,
                        chat_id=self.config.chat_id,
                        local_snapshot=True,
                    )
                    if local_duplicate is not None:
                        duplicate_of, reason = local_duplicate, "human_photo_already_seen"
                except Exception:
                    logger.exception("Camera retained attachment history unavailable; candidate not sent")
                    return self._error(503, "duplicate_evidence_unavailable")
            if (
                duplicate_of is None
                and self._recent_attachments is not None
                and remote_history_incomplete
            ):
                return self._error(503, "duplicate_evidence_unavailable")
            if (
                duplicate_of is None
                and self._retained_history_required
                and self._retained_attachments is None
            ):
                return self._error(503, "duplicate_evidence_unavailable")
            if (
                duplicate_of is None
                and self._history_required
                and self._recent_attachments is None
                and self._retained_attachments is None
            ):
                return self._error(503, "duplicate_evidence_unavailable")
            if duplicate_of is not None:
                if not duplicate_of or len(duplicate_of) > 256:
                    return self._error(503, "duplicate_evidence_invalid")
                terminal = {
                    "state": "duplicate",
                    "admitted_at": now.isoformat(),
                    "attention_active": False,
                    "photo_delivery_confirmed": False,
                    "capture_time": request.capture_time.isoformat(),
                    "capture_time_authority": request.capture_time_authority,
                    "image_sha256": descriptor["sha256"],
                    "image_phash": descriptor.get("phash"),
                    "phash_algorithm": descriptor.get("phash_algorithm"),
                    "duplicate_of": duplicate_of,
                    "duplicate_reason": reason,
                }
                self._attempts[request.candidate_id] = terminal
                response = {
                    "status": "duplicate",
                    "candidate_id": request.candidate_id,
                    "reason": reason,
                    "duplicate_of": duplicate_of,
                }
                try:
                    return self._commit_sequence(request, 200, response, attempt=terminal)
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
            if any(
                item.get("attention_active", True)
                or item.get("state") == "admitted"
                or (item.get("state") == "delivery_unknown" and not item.get("photo_delivery_confirmed"))
                for item in self._attempts.values()
            ):
                return self._error(409, "unresolved_candidate")
            if not self._telegram or not getattr(self._telegram, "polling_started", False):
                return self._error(503, "pre_admission_unavailable")
            try:
                if not isinstance(upload, CameraCandidateUpload):
                    raise ValueError("Camera image and producer evidence were not uploaded")
                image_bytes, suffix, manifest = self._validate_source_artifacts(request, upload)
            except (OSError, ValueError, TypeError, ValidationError):
                status, response = self._error(422, "candidate_evidence_mismatch")
                try:
                    return self._commit_sequence(request, status, response)
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
            admission_id = f"cam1-{uuid4().hex}"
            try:
                state_fd = self._open_state_dir(create=True)
                try:
                    snapshot_fd = _child_directory(state_fd, "snapshots", create=True)
                    try:
                        snapshot = self._state_dir / "snapshots" / f"{admission_id}{suffix}"
                        snapshot_descriptor = os.open(
                            snapshot.name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                            0o600,
                            dir_fd=snapshot_fd,
                        )
                    finally:
                        os.close(snapshot_fd)
                finally:
                    os.close(state_fd)
                with os.fdopen(snapshot_descriptor, "wb") as stream:
                    stream.write(image_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                attempt = {
                    "state": "admitted",
                    "admission_id": admission_id,
                    "snapshot": str(snapshot),
                    "photo_id": None,
                    "photo_delivery_confirmed": False,
                    "reply_ids": [],
                    "final_turn_id": None,
                    "admitted_at": datetime.now(timezone.utc).isoformat(),
                    "attention_active": True,
                    "capture_time": request.capture_time.isoformat(),
                    "capture_time_authority": request.capture_time_authority,
                    "image_sha256": descriptor["sha256"],
                    "image_phash": descriptor.get("phash"),
                    "phash_algorithm": descriptor.get("phash_algorithm"),
                    # Old journal entries without this decision remain unknown.
                    "classifier_decision": manifest.classifier_output.decision,
                }
                self._attempts[request.candidate_id] = attempt
                status, response = self._commit_sequence(
                    request,
                    202,
                    {
                        "status": "admitted",
                        "candidate_id": request.candidate_id,
                        "admission_id": admission_id,
                        "delivery_semantics": "in_process_only",
                    },
                    attempt=attempt,
                )
            except OSError:
                # A journal fsync can fail after replace. Retain any in-memory
                # tombstone rather than risking a second send in this process.
                return self._error(503, "pre_admission_unavailable")
            task = asyncio.create_task(self._deliver(request.candidate_id), name=admission_id)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            return status, response

    async def reconcile(
        self,
        authorization: str | None,
        upload: object,
        *,
        purpose: str = "retire_ineligible",
    ) -> tuple[int, dict]:
        """Replay exact outcomes or retire an ineligible request, without delivery."""
        if not isinstance(purpose, str) or purpose not in {
            "existing_outcome_only", "retire_ineligible", "prove_not_admitted"
        }:
            return self._error(400, "invalid_request")
        if not self.config.enabled:
            return self._error(403, "camera_disabled")
        try:
            expected = self._token()
        except (OSError, ValueError):
            return self._error(503, "pre_admission_unavailable")
        prefix = "Bearer "
        if (
            not isinstance(authorization, str)
            or not authorization.startswith(prefix)
            or not hmac.compare_digest(authorization[len(prefix) :].encode(), expected)
        ):
            return self._error(401, "unauthorized")
        if not isinstance(upload, CameraCandidateUpload):
            return self._error(400, "invalid_request")
        try:
            request = CameraCandidateRequest.model_validate(upload.request)
        except ValidationError:
            return self._error(400, "invalid_request")

        async with self._lock:
            if purpose == "existing_outcome_only":
                # This operation is a journal lookup, so it must not refresh a
                # lease or require a write before returning an already durable
                # outcome. Artifact validation remains inside the admission lock.
                return self._lookup_existing_outcome(request, upload)
            if purpose == "prove_not_admitted":
                if type(upload.request.get("seq")) is not int:
                    return self._error(400, "invalid_request")
                return self._prove_not_admitted(request, upload)

            if self._refresh_expired_session():
                try:
                    self._save_attempts()
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
            try:
                # Artifact evidence is exact and validated even when the
                # capture is now stale or the request epoch has rotated.
                self._validate_upload(request, upload)
            except (OSError, ValueError, TypeError, ValidationError):
                return self._error(422, "candidate_evidence_mismatch")

            identity = self._request_identity(request)
            previous = self._attempts.get(request.candidate_id)
            if previous is not None:
                legacy = previous.get("legacy_base_recovery")
                if (
                    previous.get("request_identity") is None
                    and isinstance(legacy, dict)
                    and legacy.get("proof") == "base_session_ack_and_verified_managed_snapshot"
                ):
                    capture = self._attempt_capture_time(previous)
                    try:
                        _, _ = self._validate_upload(request, upload)
                    except (OSError, ValueError, TypeError, ValidationError):
                        return self._error(422, "candidate_evidence_mismatch")
                    if (
                        request.session_id != legacy.get("session_id")
                        or request.epoch != legacy.get("epoch")
                        or request.seq != legacy.get("seq")
                        or capture is None
                        or request.capture_time.astimezone(timezone.utc)
                        != capture.astimezone(timezone.utc)
                        or request.capture_time_authority
                        != previous.get("capture_time_authority")
                        or request.image_sha256 != legacy.get("snapshot_sha256_observed")
                    ):
                        return self._error(409, "candidate_request_mismatch")
                    ack = {
                        "status": legacy["ack_status"],
                        "body": dict(legacy["ack_body"]),
                    }
                    # Manifest SHA/revision were not persisted by the base.
                    # Store this submission as observed recovery input, without
                    # asserting it was the original historical digest.
                    previous["request_identity"] = identity
                    previous["request_ack"] = ack
                    previous["request_identity_proof"] = (
                        "legacy_ack_snapshot_sha_and_observed_recovery_input"
                    )
                    try:
                        self._save_attempts()
                    except OSError:
                        return self._error(503, "pre_admission_unavailable")
                    return ack["status"], ack["body"]
                if previous.get("request_identity") != identity:
                    return self._error(409, "candidate_request_mismatch")
                ack = previous.get("request_ack")
                if not isinstance(ack, dict) or not isinstance(ack.get("body"), dict):
                    return self._error(409, "candidate_ack_unavailable")
                if previous.get("request_identity_proof") != (
                    "legacy_ack_snapshot_sha_and_observed_recovery_input"
                ) and request.session_id != self._session["session_id"]:
                    return self._error(409, "session_expired")
                try:
                    self._save_attempts()
                except OSError:
                    return self._error(503, "pre_admission_unavailable")
                return ack["status"], ack["body"]

            if request.session_id != self._session["session_id"]:
                return self._error(409, "session_expired")

            if len(self._attempts) >= _MAX_ATTEMPTS:
                return self._error(503, "pre_admission_unavailable")

            # A previous sequence ACK without a candidate journal record can
            # be retired only if its exact request identity owns that sequence.
            same_epoch = request.epoch == self._session["epoch"]
            advance = False
            update_last_ack = None
            if same_epoch:
                committed = self._session["committed_seq"]
                if request.seq == committed + 1:
                    advance = True
                elif (
                    request.seq == committed
                    and isinstance(self._session.get("last_ack"), dict)
                    and self._session["last_ack"].get("candidate_id") == request.candidate_id
                    and self._session["last_ack"].get("request_identity") == identity
                ):
                    update_last_ack = True
                else:
                    return self._error(409, "conflicting_sequence_owner")

            for candidate_id, attempt in self._attempts.items():
                owner = attempt.get("request_identity")
                if (
                    candidate_id != request.candidate_id
                    and isinstance(owner, dict)
                    and owner.get("session_id") == request.session_id
                    and owner.get("epoch") == request.epoch
                    and owner.get("seq") == request.seq
                ):
                    return self._error(409, "conflicting_sequence_owner")

            retired = {
                "state": "retired",
                "retired_at": datetime.now(timezone.utc).isoformat(),
                "attention_active": False,
                "photo_delivery_confirmed": False,
                "capture_time": request.capture_time.isoformat(),
                "capture_time_authority": request.capture_time_authority,
                "image_sha256": request.image_sha256,
                "source_revision": request.source_revision,
                "manifest_sha256": request.manifest_sha256,
            }
            self._attempts[request.candidate_id] = retired
            response = {
                "status": "retired",
                "candidate_id": request.candidate_id,
                "reason": "candidate_no_longer_eligible",
            }
            try:
                return self._commit_sequence(
                    request,
                    200,
                    response,
                    attempt=retired,
                    advance=advance,
                    update_last_ack=update_last_ack,
                )
            except OSError:
                return self._error(503, "pre_admission_unavailable")

    def _lookup_existing_outcome(
        self, request: CameraCandidateRequest, upload: CameraCandidateUpload
    ) -> tuple[int, dict]:
        """Return only an exact durable ACK, without changing journal state."""
        try:
            self._validate_upload(request, upload)
        except (OSError, ValueError, TypeError, ValidationError):
            return self._error(422, "candidate_evidence_mismatch")

        previous = self._attempts.get(request.candidate_id)
        if previous is None:
            return self._error(503, "unknown_original_outcome")

        identity = self._request_identity(request)
        legacy = previous.get("legacy_base_recovery")
        if previous.get("request_identity") is None:
            if (
                not isinstance(legacy, dict)
                or legacy.get("proof") != "base_session_ack_and_verified_managed_snapshot"
            ):
                return self._error(503, "unknown_original_outcome")
            capture = self._attempt_capture_time(previous)
            if (
                request.session_id != legacy.get("session_id")
                or request.epoch != legacy.get("epoch")
                or request.seq != legacy.get("seq")
                or capture is None
                or request.capture_time.astimezone(timezone.utc)
                != capture.astimezone(timezone.utc)
                or request.capture_time_authority != previous.get("capture_time_authority")
            ):
                return self._error(409, "candidate_request_mismatch")
            if request.image_sha256 != legacy.get("snapshot_sha256_observed"):
                # A legacy ACK does not include a durable image digest. If the
                # managed snapshot cannot prove today's claimed SHA, the old
                # request outcome is unknown; do not turn missing proof into a
                # definitive request-mismatch response.
                return self._error(503, "unknown_original_outcome")
            status = legacy.get("ack_status")
            body = legacy.get("ack_body")
            if (
                type(status) is not int
                or not isinstance(body, dict)
                or body.get("candidate_id") != request.candidate_id
                or body.get("session_id") != request.session_id
                or body.get("epoch") != request.epoch
                or body.get("ack_seq") != request.seq
            ):
                return self._error(503, "unknown_original_outcome")
            # The base did not retain manifest/revision identity. The snapshot
            # and original ACK prove only the fields checked above; do not save
            # today's manifest as if it were historical evidence.
            return status, dict(body)

        if previous.get("request_identity") != identity:
            return self._error(503, "unknown_original_outcome")
        ack = previous.get("request_ack")
        if (
            not isinstance(ack, dict)
            or type(ack.get("status")) is not int
            or not isinstance(ack.get("body"), dict)
        ):
            return self._error(503, "unknown_original_outcome")
        return ack["status"], ack["body"]

    def _prove_not_admitted(
        self, request: CameraCandidateRequest, upload: CameraCandidateUpload
    ) -> tuple[int, dict]:
        """Prove only that this exact candidate was not admitted in this lineage.

        The v1 journal atomically retains an attempt tombstone before dispatch.
        A fresh read is required here: an in-memory absence is not evidence if
        the backing journal disappeared, changed schema, or cannot be parsed.
        """
        try:
            self._validate_upload(request, upload)
        except (OSError, ValueError, TypeError, ValidationError):
            return self._error(422, "candidate_evidence_mismatch")

        try:
            expires_at = datetime.fromisoformat(self._session["expires_at"])
        except (KeyError, TypeError, ValueError):
            return self._error(503, "unknown_original_outcome")
        if (
            expires_at.tzinfo is None
            or expires_at.utcoffset() is None
            or datetime.now(timezone.utc) >= expires_at
            or request.session_id != self._session["session_id"]
            or request.epoch == self._session["epoch"]
        ):
            return self._error(503, "unknown_original_outcome")

        migrated_before = getattr(self, "_journal_migrated", False)
        try:
            backing_attempts, backing_session = self._load_attempts()
        except (OSError, ValueError, TypeError, ValidationError):
            return self._error(503, "unknown_original_outcome")
        finally:
            self._journal_migrated = migrated_before

        if (
            not self._journal_present
            or self._journal_schema_version != 2
            or not isinstance(backing_session, dict)
            or set(backing_session)
            != {"session_id", "epoch", "committed_seq", "expires_at", "last_ack"}
            or type(backing_session.get("committed_seq")) is not int
            or type(self._session.get("committed_seq")) is not int
            or backing_session != self._session
            or request.candidate_id in self._attempts
            or request.candidate_id in backing_attempts
        ):
            return self._error(503, "unknown_original_outcome")

        # Refuse any recorded owner of the original sequence. Unrelated legacy
        # candidate tombstones remain useful coverage even when their older
        # schema did not retain request_identity: candidate IDs are the durable
        # admission keys and tombstones are never removed.
        for candidate_id, attempt in (*self._attempts.items(), *backing_attempts.items()):
            identity = attempt.get("request_identity")
            recovery = attempt.get("legacy_base_recovery")
            if isinstance(identity, dict) and type(identity.get("seq")) is not int:
                return self._error(503, "unknown_original_outcome")
            if (
                isinstance(identity, dict)
                and identity.get("session_id") == request.session_id
                and identity.get("epoch") == request.epoch
                and identity.get("seq") == request.seq
            ) or (
                isinstance(recovery, dict)
                and recovery.get("session_id") == request.session_id
                and recovery.get("epoch") == request.epoch
                and recovery.get("seq") == request.seq
            ):
                return self._error(503, "unknown_original_outcome")

        last_ack = backing_session.get("last_ack")
        committed_seq = backing_session["committed_seq"]
        if committed_seq == 0:
            if last_ack is not None:
                return self._error(503, "unknown_original_outcome")
        else:
            if not isinstance(last_ack, dict) or set(last_ack) != {
                "seq", "candidate_id", "request_identity", "status", "body"
            }:
                return self._error(503, "unknown_original_outcome")
            identity = last_ack.get("request_identity")
            body = last_ack.get("body")
            if (
                type(last_ack.get("seq")) is not int
                or last_ack["seq"] != committed_seq
                or type(last_ack.get("status")) is not int
                or not isinstance(last_ack.get("candidate_id"), str)
                or not isinstance(identity, dict)
                or type(identity.get("seq")) is not int
                or not isinstance(body, dict)
                or identity.get("candidate_id") != last_ack["candidate_id"]
                or identity.get("session_id") != self._session["session_id"]
                or identity.get("epoch") != self._session["epoch"]
                or identity.get("seq") != committed_seq
                or (
                    body.get("candidate_id") is not None
                    and body.get("candidate_id") != last_ack["candidate_id"]
                )
                or body.get("session_id") != self._session["session_id"]
                or body.get("epoch") != self._session["epoch"]
                or type(body.get("ack_seq")) is not int
                or body.get("ack_seq") != committed_seq
                or last_ack["candidate_id"] == request.candidate_id
            ):
                return self._error(503, "unknown_original_outcome")
            try:
                ack_request = CameraCandidateRequest.model_validate(identity)
            except ValidationError:
                return self._error(503, "unknown_original_outcome")
            if self._request_identity(ack_request) != identity:
                return self._error(503, "unknown_original_outcome")

        return 200, {
            "status": "not_admitted",
            "proof": "same_session_durable_journal_v1",
            "request_identity": self._request_identity(request),
            "current_lease": {
                "session_id": self._session["session_id"],
                "epoch": self._session["epoch"],
                "committed_seq": self._session["committed_seq"],
                "expires_at": self._session["expires_at"],
            },
        }

    @staticmethod
    def _error(status: int, code: str) -> tuple[int, dict]:
        return status, {"error": {"code": code}}

    async def _deliver(self, candidate_id: str) -> None:
        attempt = self._attempts[candidate_id]
        try:
            captured = self._attempt_capture_time(attempt)
            if "_camera_caption" not in attempt:
                attempt["_camera_caption"] = _camera_caption(
                    captured, datetime.now(timezone.utc)
                )
                # Freeze the exact text before the external delivery boundary.
                self._save_attempts()
            caption = _attempt_caption(attempt)
            receipt: OutboundDeliveryReceipt = await self._telegram.send_camera_photo(
                chat_id=self.config.chat_id,
                image_path=attempt["snapshot"],
                caption=caption,
                buttons=[],
            )
            if (
                receipt.channel != "telegram"
                or str(receipt.chat_id) != self.config.chat_id
                or len(receipt.native_message_ids) != 1
                or not isinstance(receipt.native_message_ids[0], int)
                or isinstance(receipt.native_message_ids[0], bool)
                or receipt.native_message_ids[0] <= 0
            ):
                raise ValueError("native Camera photo receipt is invalid")
            attempt["photo_id"] = receipt.native_message_ids[0]
            attempt["photo_delivery_confirmed"] = True
            attempt["state"] = "photo_sent"
            self._save_attempts()
            await self._bus.publish_inbound(
                InboundMessage(
                    channel="telegram",
                    sender_id="__camera__",
                    chat_id=self.config.chat_id,
                    content=(
                        "Проанализируй снимок Camera и кратко опиши видимую еду без выдуманных "
                        "ккал или макронутриентов. Исходная подпись уже содержит вопрос «Съели ли "
                        "вы это?» и дату; не повторяй его. Выбери полезный вопрос и конкретные "
                        "варианты ответа через [[ask: вопрос | вариант | вариант]]. Если ответ о "
                        "порции важен, варианты должны различать порции. Это только анализ: не "
                        "записывай приём пищи без явного ответа пользователя. Считай целые "
                        "продукты, а не кусочки, нарезанные из одного продукта. Подбирай варианты "
                        "по видимой еде и возможной порции; не указывай точное число в варианте, "
                        "если снимок надёжно его не подтверждает. Подтверждение целой порции "
                        "подтверждает порцию, но не превращает твоё предположительное число в "
                        "независимое количество, названное владельцем. Не выспрашивай точные "
                        "граммы только для обычной записи.\n"
                        + (
                            "Проверенное решение классификатора: ambiguous. Добавь вариант «Это не еда»."
                            if attempt.get("classifier_decision") == "ambiguous"
                            else "Проверенное решение классификатора: food. Не добавляй вариант «Это не еда»."
                            if attempt.get("classifier_decision") == "food"
                            else "Решение классификатора неизвестно. Не добавляй вариант «Это не еда»."
                        )
                    ),
                    media=[attempt["snapshot"]],
                    metadata={
                        "_synthetic": True,
                        "_camera_authority": CAMERA_AUTHORITY,
                        "_camera_candidate_id": candidate_id,
                        "_camera_photo_id": attempt["photo_id"],
                        "_camera_classifier_decision": attempt.get("classifier_decision"),
                        "_camera_caption": caption,
                        "_camera_initial_prompt": CAMERA_AUTHORITY,
                        "is_group": False,
                        "chat_type": "private",
                    },
                )
            )
        except Exception:
            # A lost response could still mean Telegram accepted the photo.
            # Keep the attempt unresolved; never send it again automatically.
            attempt["state"] = "delivery_unknown"
            self._save_attempts()

    async def claim_initial_prompt_edit(self, message: OutboundMessage, chat_id: str) -> bool:
        """Claim one initial prompt edit only for its confirmed, live photo."""
        md = message.metadata
        if (
            message.channel != "telegram"
            or str(message.chat_id) != self.config.chat_id
            or md.get("_camera_authority") is not CAMERA_AUTHORITY
            or md.get("_camera_final") is not CAMERA_AUTHORITY
            or md.get("_camera_edit_existing_photo") is not CAMERA_AUTHORITY
            or md.get("_camera_initial_prompt") is not CAMERA_AUTHORITY
            or not isinstance(chat_id, (str, int)) or str(chat_id) != self.config.chat_id
        ):
            return False
        candidate_id = md.get("_camera_candidate_id")
        photo_id = md.get("_camera_photo_id")
        if not isinstance(candidate_id, str) or type(photo_id) is not int or photo_id <= 0:
            return False
        async with self._lock:
            attempt = self._attempts.get(candidate_id)
            expected_caption = _attempt_caption(attempt) if isinstance(attempt, dict) else None
            if (
                not isinstance(attempt, dict)
                or attempt.get("state") != "photo_sent"
                or attempt.get("photo_delivery_confirmed") is not True
                or attempt.get("photo_id") != photo_id
                or md.get("_camera_caption") != expected_caption
                or not isinstance(message.buttons, list)
                or not 2 <= len(message.buttons) <= 8
                or any(not isinstance(option, str) or not option.strip() for option in message.buttons)
                or attempt.get("prompt_edit_claimed") is True
            ):
                return False
            attempt["prompt_edit_claimed"] = True
            self._save_attempts()
            return True

    def process_real_inbound(self, message: InboundMessage) -> None:
        if not self.config.enabled or message.channel != "telegram":
            return
        if (
            str(message.chat_id) != self.config.chat_id
            or message.sender_id.split("|", 1)[0] != self.config.principal
        ):
            return
        metadata = message.metadata
        self._sweep_expired_attempts()
        target = (
            metadata.get("native_message_id")
            if metadata.get("callback_query")
            else metadata.get("reply_to_message_id")
        )
        raw_text = metadata.get("_telegram_raw_text", message.content)
        callback = bool(metadata.get("callback_query"))
        target_matches = [
            (key, value) for key, value in self._attempts.items()
            if target is not None and str(target) in {
                str(value.get("photo_id")), *map(str, value.get("reply_ids", []))
            }
        ]
        if target is None and message.media:
            self._interrupt_untargeted_camera_context()
        # Persist a production-owned eligibility fact for calibration. The
        # callback's visible text is never enough to turn a stale Camera tap
        # into useful-click evidence after the Camera journal rejects it.
        if callback and len(target_matches) == 1:
            metadata["_camera_ingress_callback_candidate"] = target_matches[0][0]
            metadata["_camera_ingress_callback_eligible"] = False
        if message.media:
            # Incoming user media always keeps its ordinary intake path unless
            # it explicitly replies to a Camera prompt, which is not an answer.
            if target_matches:
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        if callback and not (
            target is not None
            and isinstance(metadata.get("callback_data"), str)
            and metadata["callback_data"].startswith("ask:")
        ):
            if target_matches:
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        if target is not None and not target_matches:
            # Preserve ordinary Telegram replies. Only a target retained by
            # this Camera journal can carry Camera authority or fail closed.
            return
        # A clearly new, dated owner meal is ordinary intake. Keep the old
        # Camera operation addressable for a later reply to its source.
        if (
            target is None
            and not message.media
            and _classify_answer(raw_text, anchored=False) == "yes"
            and (
                _CLARIFICATION_NEW_MEAL_RE.search(str(raw_text))
                or _CLARIFICATION_EXPLICIT_DATE_RE.search(str(raw_text))
            )
        ):
            self._interrupt_untargeted_camera_context()
            return
        if target is None and not message.media and isinstance(raw_text, str):
            open_camera = [
                item for item in self._attempts.values()
                if item.get("state") in {"photo_sent", "clarifying"}
            ]
            if (
                open_camera
                and _CAMERA_UNRELATED_CONTEXT_RE.search(raw_text)
                and _camera_context_answer_kind(raw_text) is None
                and not _CLARIFICATION_NEW_MEAL_RE.search(raw_text)
            ):
                # Let the owner handle the unrelated topic normally, but keep
                # its model trace from accidentally becoming a meal record.
                self._interrupt_untargeted_camera_context()
                metadata["_camera_context_unrelated"] = CAMERA_AUTHORITY
                return
        if len(target_matches) > 1:
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return

        intent = _classify_answer(raw_text, anchored=target is not None)
        if (
            not callback
            and intent == "yes"
            and isinstance(raw_text, str)
            and _NON_CONSUMPTION_SCOPE_BUTTON_RE.search(raw_text)
        ):
            intent = None
        if target is None and intent is None:
            intent = _camera_context_answer_kind(raw_text)
        if intent == "not_food":
            if not callback or len(target_matches) != 1:
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
            candidate_id, attempt = target_matches[0]
            options = metadata.get("native_keyboard_options")
            selected_index = metadata.get("native_keyboard_selected_index")
            selected_label = metadata.get("native_keyboard_selected_label")
            callback_data = metadata.get("callback_data")
            native_source_id = _source_message_id(metadata.get("message_id"))
            if (
                attempt.get("classifier_decision") != "ambiguous"
                or not isinstance(options, list)
                or not 2 <= len(options) <= 8
                or type(selected_index) is not int
                or not 0 <= selected_index < len(options)
                or selected_label != options[selected_index]
                or selected_label != "Это не еда"
                or callback_data != f"ask:{selected_index}"
                or native_source_id is None
                or native_source_id != str(target)
            ):
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
            source_id = _source_message_id(metadata.get("callback_query_id"))
            if (
                attempt.get("state") != "photo_sent" or target is None or source_id is None
                or attempt.get("classifier_feedback") is not None
            ):
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
            feedback = attempt.get("classifier_feedback")
            if feedback is None:
                prior_attention = attempt.get("attention_active", True)
                attempt["classifier_feedback"] = {
                    "candidate_id": candidate_id,
                    "photo_id": attempt.get("photo_id"),
                    "source_id": source_id,
                    "verdict": "not_food",
                    "owner_id": self.config.principal,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "classifier": _DEEPSEEK_RELEASE,
                }
                attempt["attention_active"] = False
                try:
                    self._save_attempts()
                except OSError:
                    attempt.pop("classifier_feedback", None)
                    attempt["attention_active"] = prior_attention
                    metadata["_camera_unbound"] = CAMERA_AUTHORITY
                    return
            # Attention expiry stops untargeted follow-up prompts; it does not
            # retire a still-addressable photo or invalidate a first native
            # owner callback that has just been durably accepted.
            metadata["_camera_ingress_callback_eligible"] = True
            metadata["_camera_feedback_receipt"] = {
                "kind": "not_food",
                "candidate_id": candidate_id,
                "tenant_id": self.config.tenant_id,
                "owner_principal": f"telegram:{self.config.principal}",
                "native_photo_id": attempt["photo_id"],
                "native_message_id": str(target),
                "source_message_id": native_source_id,
                "callback_query_id": source_id,
                "operation_id": f"{candidate_id}:classifier_feedback:{source_id}",
            }
            metadata["_camera_classifier_feedback"] = CAMERA_AUTHORITY
            return
        inbound_source_id = _source_message_id(metadata.get("message_id"))
        for replay_id, replay_attempt in self._attempts.items():
            if (
                inbound_source_id is not None
                and replay_attempt.get("state") == "clarifying"
                and replay_attempt.get("clarification_source_message_id") == inbound_source_id
                and isinstance(replay_attempt.get("context_question_turn_id"), str)
            ):
                metadata.update(
                    _camera_authority=CAMERA_AUTHORITY,
                    _camera_candidate_id=replay_id,
                    _camera_turn_id=replay_attempt["context_question_turn_id"],
                    _camera_context_question=CAMERA_CONTEXT_QUESTION_AUTHORITY,
                    _camera_context_question_reconcile=True,
                )
                if replay_attempt.get("snapshot"):
                    message.media.append(replay_attempt["snapshot"])
                return
            replay_source = replay_attempt.get("answer_source_message_id")
            # A source still pending in ``answering`` belongs to the current
            # answer operation. ``context_question_turn_id`` may intentionally
            # retain a prior completed question for this same candidate.
            context_turn = replay_attempt.get("answer_turn_id")
            if (
                inbound_source_id is not None
                and replay_source == inbound_source_id
                and replay_attempt.get("answer_kind") == "context"
                and replay_attempt.get("state") in {"answering", "delivery_unknown", "final_queued"}
                and isinstance(context_turn, str)
            ):
                metadata.update(
                    _camera_authority=CAMERA_AUTHORITY,
                    _camera_candidate_id=replay_id,
                    _camera_turn_id=context_turn,
                    _camera_context_question=CAMERA_CONTEXT_QUESTION_AUTHORITY,
                    _camera_context_question_reconcile=True,
                )
                if replay_attempt.get("snapshot"):
                    message.media.append(replay_attempt["snapshot"])
                return
            if (
                inbound_source_id is not None
                and replay_source == inbound_source_id
                and replay_attempt.get("state") in {"answering", "delivery_unknown", "final_queued"}
            ):
                metadata.update(
                    _camera_authority=CAMERA_AUTHORITY,
                    _camera_candidate_id=replay_id,
                    _camera_answer=replay_attempt.get("answer_kind"),
                    _camera_turn_id=replay_attempt.get("answer_turn_id"),
                    _camera_reconcile_only=True,
                )
                if replay_attempt.get("answer_kind") == "yes":
                    message.media.append(replay_attempt["snapshot"])
                return

        def bind_denial(candidate_id: str, attempt: dict, route: str) -> bool:
            state = attempt.get("state")
            has_commit = isinstance(attempt.get("camera_commit"), dict)
            has_turn = any(
                isinstance(attempt.get(name), str) and attempt[name]
                for name in ("answer_turn_id", "final_turn_id")
            )
            if state not in {"completed", "delivery_unknown", "answering", "final_queued"}:
                return False
            if not (has_commit or has_turn):
                return False
            correction_state = attempt.get("camera_correction")
            original_turn = attempt.get("answer_turn_id") or attempt.get("final_turn_id")
            if has_commit and correction_state is None:
                correction_turn = attempt.get("camera_correction_turn_id")
                if not isinstance(correction_turn, str) or not correction_turn:
                    correction_turn = uuid4().hex
                    attempt["camera_correction_turn_id"] = correction_turn
                attempt["camera_correction"] = "answering"
                self._save_attempts()
                metadata.update(
                    _camera_authority=CAMERA_AUTHORITY,
                    _camera_candidate_id=candidate_id,
                    _camera_answer="no",
                    _camera_correction=CAMERA_AUTHORITY,
                    _camera_turn_id=correction_turn,
                    _camera_route=route,
                )
            elif has_commit and correction_state == "answering":
                correction_turn = attempt.get("camera_correction_turn_id")
                if not isinstance(correction_turn, str) or not correction_turn:
                    metadata["_camera_unbound"] = CAMERA_AUTHORITY
                    return True
                metadata.update(
                    _camera_authority=CAMERA_AUTHORITY,
                    _camera_candidate_id=candidate_id,
                    _camera_answer="no",
                    _camera_correction=CAMERA_AUTHORITY,
                    _camera_turn_id=correction_turn,
                    _camera_route=route,
                    _camera_reconcile_only=True,
                )
            elif (
                not has_commit
                and state in {"answering", "final_queued", "delivery_unknown"}
                and attempt.get("answer_kind") == "yes"
                and isinstance(original_turn, str)
            ):
                correction_turn = attempt.get("camera_correction_turn_id")
                if not isinstance(correction_turn, str) or not correction_turn:
                    correction_turn = uuid4().hex
                    attempt["camera_correction_turn_id"] = correction_turn
                attempt["camera_correction"] = "reconciling_original"
                self._save_attempts()
                metadata.update(
                    _camera_authority=CAMERA_AUTHORITY,
                    _camera_candidate_id=candidate_id,
                    _camera_answer="no",
                    _camera_turn_id=original_turn,
                    _camera_original_turn_id=original_turn,
                    _camera_correction_turn_id=correction_turn,
                    _camera_reconcile_then_correction=CAMERA_AUTHORITY,
                    _camera_route=route,
                )
            elif (
                not has_commit
                and state == "completed"
                and attempt.get("answer_kind") == "yes"
                and isinstance(original_turn, str)
            ):
                correction_turn = attempt.get("camera_correction_turn_id")
                if not isinstance(correction_turn, str) or not correction_turn:
                    correction_turn = uuid4().hex
                    attempt["camera_correction_turn_id"] = correction_turn
                attempt["camera_correction"] = "reconciling_original"
                self._save_attempts()
                metadata.update(
                    _camera_authority=CAMERA_AUTHORITY,
                    _camera_candidate_id=candidate_id,
                    _camera_answer="no",
                    _camera_turn_id=original_turn,
                    _camera_correction_turn_id=correction_turn,
                    _camera_legacy_reconcile=True,
                    _camera_route=route,
                )
            else:
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return True
            if target is not None:
                metadata["_camera_native_binding"] = str(target)
            return True

        if intent == "no":
            if target_matches:
                candidate_id, attempt = target_matches[0]
                if attempt.get("state") == "clarifying":
                    answer = "no"
                    route = "callback" if callback else "reply"
                    self._bind_clarification_answer(message, candidate_id, attempt, answer, target, route)
                    return
                if bind_denial(candidate_id, attempt, "callback" if callback else "reply"):
                    return
            elif target is None:
                active_contexts = [
                    value for value in self._attempts.values()
                    if value.get("state") in {
                        "photo_sent", "clarifying", "answering", "final_queued"
                    }
                ]
                if (
                    len(active_contexts) == 1
                    and _bare_context_answer(
                        raw_text,
                        source_context=active_contexts[0].get("confirmed_camera_context"),
                    )
                    and active_contexts[0].get("context_interrupted", False)
                ):
                    metadata["_camera_context_unrelated"] = CAMERA_AUTHORITY
                    return
                contextual_candidates = [
                    (key, value) for key, value in self._attempts.items()
                    if value.get("state") in {
                        "photo_sent", "clarifying", "answering", "final_queued", "completed", "delivery_unknown"
                    }
                    and (
                        value.get("state") in {"photo_sent", "clarifying", "answering", "final_queued"}
                        or isinstance(value.get("camera_commit"), dict)
                        or isinstance(value.get("answer_turn_id"), str)
                        or isinstance(value.get("final_turn_id"), str)
                    )
                ]
                if len(contextual_candidates) > 1:
                    metadata["_camera_unbound"] = CAMERA_AUTHORITY
                    return
                if len(contextual_candidates) == 1:
                    candidate_id, attempt = contextual_candidates[0]
                    if attempt.get("state") == "photo_sent":
                        pass  # The original first-answer path below handles this.
                    elif attempt.get("state") == "clarifying":
                        self._bind_clarification_answer(message, candidate_id, attempt, "no", target, "context")
                        return
                    elif bind_denial(candidate_id, attempt, "context"):
                        return
        if intent != "no" and target_matches:
            candidate_id, attempt = target_matches[0]
            if (
                callback
                and attempt.get("state") == "completed"
                and isinstance(attempt.get("camera_commit"), dict)
                and target is not None
                and _native_button_answer_kind(metadata, raw_text) == "yes"
                and isinstance(attempt.get("answer_turn_id"), str)
                and attempt.get("finalizer_status") == "committed"
                and _source_message_id(metadata.get("callback_query_id")) is not None
                and not isinstance(attempt.get("camera_correction_commit"), dict)
                and attempt.get("camera_correction") is None
            ):
                # A completed affirmative callback may be a deliberate second
                # tap of the same portion. Runtime must reconcile the exact
                # saved exchange and compare its durable user text before
                # reporting the existing meal; ingress alone cannot assert it.
                metadata.update(
                    _camera_authority=CAMERA_AUTHORITY,
                    _camera_candidate_id=candidate_id,
                    _camera_answer="yes",
                    _camera_turn_id=attempt["answer_turn_id"],
                    _camera_route="callback",
                    _camera_native_binding=str(target),
                    _camera_existing_meal_replay=True,
                    _camera_ingress_callback_eligible=True,
                )
                return
            if (
                attempt.get("state") in {"completed", "delivery_unknown"}
                and (
                    isinstance(attempt.get("camera_commit"), dict)
                    or isinstance(attempt.get("answer_turn_id"), str)
                    or isinstance(attempt.get("final_turn_id"), str)
                )
            ) or (
                attempt.get("state") in {"answering", "final_queued"}
                and intent != attempt.get("answer_kind")
            ):
                # A thank-you, repeated yes, or unrelated callback is never a
                # denial authorization for a retained Camera meal.
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
        if (
            self._attempts
            and not message.media
            and target is None
            and not any(item["state"] != "completed" for item in self._attempts.values())
            and isinstance(raw_text, str)
            and raw_text.strip().casefold() in {*_YES, *_NO, "да", "yes", "ага"}
        ):
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        addressable = [
            (key, value) for key, value in self._attempts.items()
            if value["state"] in {"photo_sent", "clarifying", "answering", "final_queued"}
        ]
        matches = [
            (key, value) for key, value in addressable
            if target is not None and str(target) in {
                str(value.get("photo_id")), *map(str, value.get("reply_ids", []))
            }
        ]
        if target is not None and not matches and any(
            str(target) in {str(value.get("photo_id")), *map(str, value.get("reply_ids", []))}
            and not (
                value.get("state") == "completed"
                and (
                    isinstance(value.get("camera_commit"), dict)
                    or isinstance(value.get("answer_turn_id"), str)
                    or isinstance(value.get("final_turn_id"), str)
                )
            )
            for value in self._attempts.values()
        ):
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        if target is not None and matches:
            if len(matches) != 1:
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
            pending = matches
        else:
            if len(addressable) == 1:
                # A single retained operation remains context-addressable after
                # its attention timer has expired.
                if (
                    not addressable[0][1].get("attention_active", True)
                    and _camera_context_answer_kind(
                        raw_text,
                        source_context=addressable[0][1].get("confirmed_camera_context"),
                    ) is None
                    and not (
                        addressable[0][1].get("state") == "clarifying"
                        and _clarification_related(raw_text)
                    )
                    and not _camera_food_context_hint(raw_text)
                ):
                    self._interrupt_untargeted_camera_context()
                    return
                pending = addressable
            elif len(addressable) > 1:
                if (
                    _camera_context_answer_kind(raw_text) is not None
                    or _camera_composition_description(raw_text)
                ):
                    metadata["_camera_unbound"] = CAMERA_AUTHORITY
                else:
                    self._interrupt_untargeted_camera_context()
                return
            else:
                return
        candidate_id, attempt = pending[0]
        if (
            target is None
            and _bare_context_answer(
                raw_text, source_context=attempt.get("confirmed_camera_context")
            )
            and attempt.get("context_interrupted", False)
        ):
            metadata["_camera_context_unrelated"] = CAMERA_AUTHORITY
            return
        if attempt["state"] == "clarifying":
            source_id = _source_message_id(metadata.get("message_id"))
            if source_id is not None and source_id == attempt.get("clarification_source_message_id"):
                metadata.update(
                    _camera_authority=CAMERA_AUTHORITY,
                    _camera_candidate_id=candidate_id,
                    _camera_answer="yes",
                    _camera_turn_id=attempt.get("answer_turn_id"),
                    _camera_reconcile_only=True,
                    _camera_clarification_reconcile=True,
                )
                if attempt.get("reply_ids"):
                    metadata["_camera_duplicate_clarification_replay"] = True
                return
            text = raw_text if isinstance(raw_text, str) else ""
            if _CLARIFICATION_NEW_MEAL_RE.search(text):
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
            if callback:
                # This journal state exists only after a confirmed consumed
                # answer; it makes portion options safe even when the follow-up
                # itself asks only which size.
                metadata["_camera_known_consumption_clarification"] = CAMERA_AUTHORITY
            answer = (
                _native_button_answer_kind(metadata, raw_text)
                if callback else _camera_context_answer_kind(
                    text, source_context=attempt.get("confirmed_camera_context")
                )
            )
            if answer is None:
                if target is None and _camera_food_context_hint(text):
                    self._bind_context_question(message, candidate_id, attempt)
                    return
                if target is None:
                    self._interrupt_untargeted_camera_context()
                    metadata["_camera_context_unrelated"] = CAMERA_AUTHORITY
                return
            if callback and len(target_matches) == 1:
                metadata["_camera_ingress_callback_eligible"] = True
            self._bind_clarification_answer(
                message, candidate_id, attempt, answer, target,
                "callback" if callback else ("reply" if target is not None else "context"),
            )
            return
        if attempt["state"] in {"answering", "final_queued"}:
            answer_kind = attempt.get("answer_kind")
            turn_id = attempt.get("answer_turn_id")
            current_intent = _classify_answer(raw_text, anchored=target is not None)
            if (
                answer_kind not in {"yes", "no"}
                or not isinstance(turn_id, str)
                or current_intent != answer_kind
            ):
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
            metadata["_camera_authority"] = CAMERA_AUTHORITY
            metadata["_camera_candidate_id"] = candidate_id
            metadata["_camera_answer"] = answer_kind
            metadata["_camera_turn_id"] = turn_id
            metadata["_camera_route"] = "callback" if metadata.get("callback_query") else ("reply" if target is not None else "context")
            metadata["_camera_reconcile_only"] = True
            if target is not None:
                metadata["_camera_native_binding"] = str(target)
            if answer_kind == "yes":
                message.media.append(attempt["snapshot"])
            return
        if message.media:
            # A new, unrelated manual photo keeps its ordinary path. A photo
            # replied to this Camera operation must not bypass its answer gate.
            if str(target) in {str(attempt["photo_id"]), *map(str, attempt["reply_ids"])}:
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        if attempt["state"] != "photo_sent":
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        anchored_ids = {str(attempt["photo_id"]), *map(str, attempt["reply_ids"])}
        if target is not None and str(target) not in anchored_ids:
            # A reply to some other message answers nothing about this photo.
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        if metadata.get("callback_query"):
            # The agent's own [[ask:]] keyboard hangs on this operation's
            # photo or question message; a tap there is a first-party answer.
            # Any other callback (stale, foreign keyboard) confers nothing.
            callback_data = metadata.get("callback_data")
            if not (isinstance(callback_data, str) and callback_data.startswith("ask:")):
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
                return
            classified = _native_button_answer_kind(metadata, raw_text)
        elif target is not None:
            classified = _camera_context_answer_kind(
                raw_text, source_context=attempt.get("confirmed_camera_context")
            )
        else:
            classified = _camera_context_answer_kind(
                raw_text, source_context=attempt.get("confirmed_camera_context")
            )
        if classified is None:
            if target is None:
                if _camera_food_context_hint(raw_text):
                    self._bind_context_question(message, candidate_id, attempt)
                    return
                self._interrupt_untargeted_camera_context()
                metadata["_camera_context_unrelated"] = CAMERA_AUTHORITY
            else:
                metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        if callback and len(target_matches) == 1:
            metadata["_camera_ingress_callback_eligible"] = True
        attempt["state"] = "answering"
        attempt["attention_active"] = False
        attempt["answer_turn_id"] = uuid4().hex
        attempt["answer_kind"] = classified
        attempt["answer_source_message_id"] = _source_message_id(metadata.get("message_id"))
        attempt["finalizer_status"] = "pending"
        try:
            self._save_attempts()
        except OSError:
            metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        metadata["_camera_authority"] = CAMERA_AUTHORITY
        metadata["_camera_candidate_id"] = candidate_id
        metadata["_camera_answer"] = classified
        metadata["_camera_turn_id"] = attempt["answer_turn_id"]
        if type(attempt.get("photo_id")) is int and attempt["photo_id"] > 0:
            metadata["_camera_photo_id"] = attempt["photo_id"]
            metadata["_camera_caption"] = _attempt_caption(attempt)
        # A trusted affirmative may still need quantity clarification. The
        # finalizer must independently report missing/not-applicable nutrition;
        # invalid or non-consumed annotations remain rejected by runtime.
        if classified == "yes":
            metadata["_camera_clarification_allowed"] = CAMERA_AUTHORITY
        metadata["_camera_route"] = (
            "callback" if metadata.get("callback_query") else
            "reply" if target is not None else "context"
        )
        if target is not None:
            metadata["_camera_native_binding"] = str(target)
        if classified == "yes":
            message.media.append(attempt["snapshot"])
        return

    def note_assistant_receipt(
        self, message: OutboundMessage, receipt: OutboundDeliveryReceipt | None
    ) -> None:
        if message.metadata.get("_camera_authority") is not CAMERA_AUTHORITY:
            return
        candidate_id = message.metadata.get("_camera_candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in self._attempts:
            return
        attempt = self._attempts[candidate_id]
        turn_id = message.metadata.get("_camera_turn_id")
        is_correction = message.metadata.get("_camera_correction") is CAMERA_AUTHORITY
        expected_turn_id = (
            attempt.get("camera_correction_final_turn_id")
            if is_correction else attempt.get("final_turn_id")
        )
        final = (
            message.metadata.get("_camera_final") is CAMERA_AUTHORITY
            and isinstance(turn_id, str)
            and turn_id
            and turn_id == expected_turn_id
        )
        valid_receipt = (
            receipt is not None
            and receipt.channel == "telegram"
            and str(receipt.chat_id) == self.config.chat_id
            and bool(receipt.native_message_ids)
            and all(
                isinstance(native_id, int) and not isinstance(native_id, bool) and native_id > 0
                for native_id in receipt.native_message_ids
            )
        )
        if not valid_receipt:
            if final and is_correction:
                attempt["camera_correction"] = "delivery_unknown"
                self._save_attempts()
            elif final and attempt["state"] == "final_queued":
                attempt["state"] = "delivery_unknown"
                self._save_attempts()
            return
        if (
            message.channel == "telegram"
            and str(message.chat_id) == self.config.chat_id
            and message.metadata.get("_camera_final") is CAMERA_AUTHORITY
            and message.metadata.get("_camera_edit_existing_photo") is CAMERA_AUTHORITY
            and message.metadata.get("_camera_initial_prompt") is CAMERA_AUTHORITY
            and message.metadata.get("_camera_photo_id") == attempt.get("photo_id")
            and message.metadata.get("_camera_caption") == _attempt_caption(attempt)
            and attempt.get("state") == "photo_sent"
            and attempt.get("photo_delivery_confirmed") is True
            and attempt.get("prompt_edit_claimed") is True
            and isinstance(message.content, str)
            and 0 < len(message.content.strip().split("\n\n", 1)[0].strip()) <= 2048
        ):
            # The bridge appends the current question after a blank line. Retain
            # only the delivered source analysis, so question wording cannot
            # ground an otherwise arbitrary composition answer.
            attempt["confirmed_camera_context"] = (
                message.content.strip().split("\n\n", 1)[0].strip()
            )
        for native_id in receipt.native_message_ids:
            if native_id not in attempt["reply_ids"]:
                attempt["reply_ids"].append(native_id)
        if final and is_correction:
            attempt["camera_correction"] = "completed"
            if attempt["state"] in {"final_queued", "delivery_unknown"}:
                attempt["state"] = "completed"
                attempt["attention_active"] = False
        elif final and attempt["state"] in {"final_queued", "delivery_unknown"}:
            attempt["state"] = "completed"
            attempt["attention_active"] = False
        self._save_attempts()
        if final and attempt["state"] == "completed":
            self._remove_completed_snapshots()

    def note_assistant_failure(self, message: OutboundMessage) -> None:
        if message.metadata.get("_camera_final") is not CAMERA_AUTHORITY:
            return
        if message.metadata.get("_camera_authority") is not CAMERA_AUTHORITY:
            return
        candidate_id = message.metadata.get("_camera_candidate_id")
        attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
        if (
            attempt is not None
            and message.metadata.get("_camera_correction") is CAMERA_AUTHORITY
            and message.metadata.get("_camera_turn_id")
            == attempt.get("camera_correction_final_turn_id")
        ):
            attempt["camera_correction"] = "delivery_unknown"
            self._save_attempts()
            return
        if (
            attempt is not None
            and attempt["state"] == "final_queued"
            and isinstance(message.metadata.get("_camera_turn_id"), str)
            and message.metadata["_camera_turn_id"] == attempt.get("final_turn_id")
        ):
            attempt["state"] = "delivery_unknown"
            self._save_attempts()

    def complete(
        self,
        message: InboundMessage,
        *,
        recorded: bool,
        clarification: bool = False,
    ) -> None:
        """Arm completion; only a native final-send receipt releases the candidate."""
        if message.metadata.get("_camera_authority") is not CAMERA_AUTHORITY:
            return
        is_context_question = (
            message.metadata.get("_camera_context_question")
            is CAMERA_CONTEXT_QUESTION_AUTHORITY
        )
        if clarification and (
            message.metadata.get("_camera_answer") == "yes" or is_context_question
        ):
            candidate_id = message.metadata.get("_camera_candidate_id")
            turn_id = message.metadata.get("_camera_turn_id")
            attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
            if (
                attempt is not None
                and attempt["state"] in {"answering", "delivery_unknown"}
                and attempt.get("answer_turn_id") == turn_id
            ):
                attempt["state"] = "clarifying"
                attempt["attention_active"] = True
                attempt["finalizer_status"] = "clarification"
                attempt["clarification_source_message_id"] = _source_message_id(
                    message.metadata.get("message_id")
                )
                if is_context_question:
                    attempt["context_question_turn_id"] = turn_id
                self._save_attempts()
            return
        if message.metadata.get("_camera_answer") == "no" or recorded:
            candidate_id = message.metadata.get("_camera_candidate_id")
            turn_id = message.metadata.get("_camera_turn_id")
            attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
            if (
                attempt is not None
                and message.metadata.get("_camera_correction") is CAMERA_AUTHORITY
                and attempt.get("camera_correction") == "answering"
                and isinstance(turn_id, str)
                and turn_id
            ):
                if recorded:
                    attempt["camera_correction"] = "final_queued"
                    attempt["camera_correction_final_turn_id"] = turn_id
                    self._save_attempts()
            elif (
                attempt is not None
                and attempt["state"] == "answering"
                and isinstance(turn_id, str)
                and turn_id
            ):
                attempt["final_turn_id"] = turn_id
                attempt["state"] = "final_queued"
                attempt["attention_active"] = False
                self._save_attempts()
            if (
                attempt is not None
                and message.metadata.get("_camera_correction") is CAMERA_AUTHORITY
                and attempt.get("camera_correction") == "answering"
                and isinstance(turn_id, str)
                and turn_id
                and recorded
            ):
                attempt["camera_correction"] = "final_queued"
                attempt["camera_correction_final_turn_id"] = turn_id
                self._save_attempts()

    def _bind_clarification_answer(
        self, message: InboundMessage, candidate_id: str, attempt: dict,
        answer: str, target: object, route: str,
    ) -> None:
        attempt["state"] = "answering"
        attempt["answer_turn_id"] = uuid4().hex
        attempt["answer_kind"] = answer
        attempt["answer_source_message_id"] = _source_message_id(
            message.metadata.get("message_id")
        )
        attempt["finalizer_status"] = "pending"
        attempt["attention_active"] = False
        try:
            self._save_attempts()
        except OSError:
            message.metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        message.metadata.update(
            _camera_authority=CAMERA_AUTHORITY,
            _camera_candidate_id=candidate_id,
            _camera_answer=answer,
            _camera_turn_id=attempt["answer_turn_id"],
            _camera_route=route,
        )
        text = message.metadata.get("_telegram_raw_text", message.content)
        if answer == "yes" or _CLARIFICATION_ONLY_PART_RE.search(text if isinstance(text, str) else "") or (
            answer == "yes"
            and (
                _camera_food_context_hint(text)
                or _CLARIFICATION_PLATE_RE.fullmatch(text.strip() if isinstance(text, str) else "")
            )
        ):
            message.metadata["_camera_clarification_allowed"] = CAMERA_AUTHORITY
        if answer == "yes" and isinstance(text, str) and _CLARIFICATION_QUANTITY_RE.search(text):
            message.metadata["_camera_context_hint"] = CAMERA_AUTHORITY
        if target is not None:
            message.metadata["_camera_native_binding"] = str(target)
        if answer == "yes" and attempt.get("snapshot"):
            message.media.append(attempt["snapshot"])

    def _bind_context_question(
        self, message: InboundMessage, candidate_id: str, attempt: dict
    ) -> None:
        """Bind model-visible context without granting consumption authority."""
        turn_id = uuid4().hex
        source_id = _source_message_id(message.metadata.get("message_id"))
        attempt.update(
            state="answering",
            answer_turn_id=turn_id,
            answer_kind="context",
            answer_source_message_id=source_id,
            finalizer_status="context_question_pending",
            attention_active=False,
        )
        try:
            self._save_attempts()
        except OSError:
            message.metadata["_camera_unbound"] = CAMERA_AUTHORITY
            return
        message.metadata.update(
            _camera_authority=CAMERA_AUTHORITY,
            _camera_candidate_id=candidate_id,
            _camera_turn_id=turn_id,
            _camera_context_hint=CAMERA_AUTHORITY,
            _camera_context_question=CAMERA_CONTEXT_QUESTION_AUTHORITY,
            _camera_route="context",
        )
        if attempt.get("snapshot"):
            message.media.append(attempt["snapshot"])

    def validate_context_question_binding(
        self,
        message: InboundMessage,
        *,
        principal: str,
        chat_id: str,
        tenant_id: str,
    ) -> None:
        candidate_id = message.metadata.get("_camera_candidate_id")
        turn_id = message.metadata.get("_camera_turn_id")
        attempt = self._attempts.get(candidate_id) if isinstance(candidate_id, str) else None
        if (
            message.metadata.get("_camera_context_question")
            is not CAMERA_CONTEXT_QUESTION_AUTHORITY
            or message.metadata.get("_camera_authority") is not CAMERA_AUTHORITY
            or message.metadata.get("is_group") is True
            or message.sender_id.split("|", 1)[0] != principal
            or str(message.chat_id) != chat_id
            or self.config.principal != principal
            or self.config.chat_id != chat_id
            or self.config.tenant_id != tenant_id
            or message.metadata.get("_camera_answer") is not None
            or attempt is None
            or attempt.get("state") != "answering"
            or attempt.get("answer_kind") != "context"
            or attempt.get("answer_turn_id") != turn_id
            or attempt.get("answer_source_message_id")
            != _source_message_id(message.metadata.get("message_id"))
            or not isinstance(attempt.get("snapshot"), str)
            or attempt["snapshot"] not in message.media
        ):
            raise ValueError("Camera context question is not bound to the authorized owner source")

    async def close(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


async def serve_camera_http(
    ingress: CameraIngress, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """One bounded HTTP/1.1 request per private connection; never log credentials."""
    status, response = 400, {"error": {"code": "invalid_request"}}
    try:
        header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
        if len(header) > 8192:
            raise ValueError("large headers")
        lines = header.decode("ascii").split("\r\n")
        method, path, version = lines[0].split(" ")
        fields: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            key, value = line.split(":", 1)
            key = key.lower().strip()
            if key in fields:
                raise ValueError("duplicate header")
            fields[key] = value.strip()
        if method == "POST" and path == "/internal/v1/camera/session" and version == "HTTP/1.1":
            if fields.get("transfer-encoding") or fields.get("content-length") != "0":
                status, response = ingress._error(400, "invalid_request")
            else:
                status, response = await ingress.lease(fields.get("authorization"))
        elif method == "POST" and path == "/internal/v1/camera/references" and version == "HTTP/1.1":
            allowed_headers = {
                "authorization", "content-length", "content-type", "connection",
                "host", "accept", "accept-encoding", "user-agent",
            }
            if any(key not in allowed_headers for key in fields):
                status, response = ingress._error(400, "invalid_request")
            elif not fields.get("authorization", "").startswith("Bearer "):
                status, response = ingress._error(401, "unauthorized")
            elif not ingress.config.enabled:
                status, response = ingress._error(403, "camera_disabled")
            elif (
                fields.get("transfer-encoding")
                or not fields.get("content-length", "").isdigit()
                or fields.get("content-type", "").lower() != "application/json"
            ):
                status, response = ingress._error(400, "invalid_request")
            else:
                length = int(fields["content-length"])
                if not 0 < length <= _MAX_REQUEST:
                    status, response = ingress._error(400, "invalid_request")
                else:
                    body = await asyncio.wait_for(reader.readexactly(length), timeout=10)
                    try:
                        payload = json.loads(body, object_pairs_hook=_unique_json)
                        if isinstance(payload, dict) and "schema_version" in payload:
                            request = CameraReferencesRequestV2.model_validate(payload)
                        else:
                            request = CameraReferencesRequest.model_validate(payload)
                    except (UnicodeDecodeError, ValueError, TypeError, ValidationError):
                        status, response = ingress._error(400, "invalid_request")
                    else:
                        status, response = await ingress.camera_references(
                            fields.get("authorization"), request
                        )
        elif (
            method == "POST"
            and path == "/internal/v1/camera/reference-source"
            and version == "HTTP/1.1"
        ):
            allowed_headers = {
                "authorization", "content-length", "content-type", "connection",
                "host", "accept", "accept-encoding", "user-agent",
            }
            if any(key not in allowed_headers for key in fields):
                status, response = ingress._error(400, "invalid_request")
            elif not fields.get("authorization", "").startswith("Bearer "):
                status, response = ingress._error(401, "unauthorized")
            elif not ingress.config.enabled:
                status, response = ingress._error(403, "camera_disabled")
            elif fields.get("transfer-encoding") or not fields.get("content-length", "").isdigit():
                status, response = ingress._error(400, "invalid_request")
            else:
                length = int(fields["content-length"])
                content_type = fields.get("content-type", "")
                if (
                    not 0 < length <= _MAX_HTTP_BODY
                    or not content_type
                    or not content_type.isascii()
                ):
                    status, response = ingress._error(400, "invalid_request")
                else:
                    body = await asyncio.wait_for(reader.readexactly(length), timeout=30)
                    try:
                        upload = _parse_camera_upload(content_type, body)
                    except (UnicodeDecodeError, ValueError, TypeError):
                        status, response = ingress._error(400, "invalid_request")
                    else:
                        status, response = await ingress.reference_source(
                            fields.get("authorization"), upload
                        )
        elif (
            method == "POST"
            and path in {
                "/internal/v1/camera/candidates",
                "/internal/v1/camera/reconcile",
            }
            and version == "HTTP/1.1"
        ):
            reconcile_route = path == "/internal/v1/camera/reconcile"
            reconcile_purpose = fields.get("x-camera-reconcile-purpose")
            invalid_reconcile_header = False
            if reconcile_route:
                if reconcile_purpose is None:
                    reconcile_purpose = "retire_ineligible"
                if reconcile_purpose not in {
                    "existing_outcome_only", "retire_ineligible", "prove_not_admitted"
                }:
                    status, response = ingress._error(400, "invalid_request")
                    reconcile_purpose = None
                    invalid_reconcile_header = True
            elif reconcile_purpose is not None:
                status, response = ingress._error(400, "invalid_request")
                invalid_reconcile_header = True
            if not ingress.config.enabled:
                if invalid_reconcile_header:
                    pass
                else:
                    status, response = ingress._error(403, "camera_disabled")
            elif invalid_reconcile_header:
                pass
            elif not fields.get("authorization", "").startswith("Bearer "):
                status, response = ingress._error(401, "unauthorized")
            elif fields.get("transfer-encoding") or not fields.get("content-length", "").isdigit():
                status, response = ingress._error(400, "invalid_request")
            else:
                length = int(fields["content-length"])
                content_type = fields.get("content-type", "")
                if (
                    not 0 < length <= _MAX_HTTP_BODY
                    or not content_type
                    or not content_type.isascii()
                ):
                    status, response = ingress._error(400, "invalid_request")
                else:
                    body = await asyncio.wait_for(reader.readexactly(length), timeout=30)
                    try:
                        upload = _parse_camera_upload(content_type, body)
                    except (UnicodeDecodeError, ValueError, TypeError):
                        status, response = ingress._error(400, "invalid_request")
                    else:
                        if reconcile_route:
                            status, response = await ingress.reconcile(
                                fields.get("authorization"), upload,
                                purpose=reconcile_purpose,
                            )
                        else:
                            status, response = await ingress.admit(
                                fields.get("authorization"), upload
                            )
    except (
        asyncio.IncompleteReadError,
        asyncio.LimitOverrunError,
        asyncio.TimeoutError,
        UnicodeError,
        ValueError,
    ):
        pass
    body = json.dumps(response, separators=(",", ":")).encode()
    writer.write(
        f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
        + body
    )
    try:
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()
