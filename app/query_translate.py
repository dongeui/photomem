"""Query expansion helpers for Korean-first CLIP search."""
from __future__ import annotations

import os
import re
from functools import lru_cache

LEXICON = {
    "자전거": "bicycle bike cycling",
    "자동차": "car automobile vehicle",
    "차": "car vehicle",
    "사람": "person people human",
    "남자": "man male person",
    "여자": "woman female person",
    "얼굴": "face portrait",
    "셀카": "selfie portrait",
    "강아지": "dog puppy",
    "고양이": "cat",
    "음식": "food meal",
    "영수증": "receipt document",
    "문서": "document paper page",
    "화면": "screen screenshot",
    "오류": "error failure warning",
    "전송": "send transfer submit",
    "실패": "failure failed error",
    "동의": "agree consent accept",
    "확인": "confirm ok check",
    "취소": "cancel",
    "버튼": "button",
    "대화": "chat conversation message",
    "메시지": "message chat",
}


def expand_for_clip(query: str) -> list[str]:
    """Return original query plus optional English variants for CLIP."""
    cleaned = query.strip()
    if not cleaned:
        return []

    variants = [cleaned]
    translated = translate_to_english(cleaned)
    if translated and translated.casefold() != cleaned.casefold():
        variants.append(translated)
    return _dedupe(variants)


def translate_to_english(query: str) -> str | None:
    """Translate Korean query to English when possible.

    The default path is a small deterministic lexicon, so the app stays fast and
    offline even before optional translation models are installed. Set
    PHOTOMEM_TRANSLATOR=opus to use a local HuggingFace MarianMT model when the
    dependencies and model cache are available.
    """
    if os.environ.get("PHOTOMEM_TRANSLATOR", "lexicon").casefold() == "opus":
        translated = _translate_with_opus(query)
        if translated:
            return translated
    return _translate_with_lexicon(query)


def _translate_with_lexicon(query: str) -> str | None:
    hits = [english for korean, english in LEXICON.items() if korean in query]
    if not hits:
        return None
    return " ".join(hits)


@lru_cache(maxsize=1)
def _opus_pipeline():
    from transformers import MarianMTModel, MarianTokenizer

    model_name = os.environ.get("PHOTOMEM_TRANSLATOR_MODEL", "Helsinki-NLP/opus-mt-ko-en")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _translate_with_opus(query: str) -> str | None:
    if not _has_hangul(query):
        return None
    try:
        import torch

        tokenizer, model = _opus_pipeline()
        tokens = tokenizer([query], return_tensors="pt", padding=True)
        with torch.no_grad():
            output = model.generate(**tokens, max_new_tokens=48)
        translated = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return translated or None
    except Exception:
        return None


def _has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
