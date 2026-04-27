"""
OCR helpers built on Tesseract.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

OCR_LANG = os.environ.get("PHOTOMEM_OCR_LANG", "kor+eng")


def is_enabled() -> bool:
    return bool(OCR_LANG.strip())


def extract_text(image_path: str) -> str:
    if not is_enabled():
        return ""

    path = Path(image_path)
    if not path.exists():
        return ""

    try:
        with Image.open(path) as image:
            candidates = []
            for prepared, config in (
                (_prepare_image(image), "--oem 1 --psm 6"),
                (_prepare_threshold_image(image), "--oem 1 --psm 11"),
            ):
                text = pytesseract.image_to_string(prepared, lang=OCR_LANG, config=config)
                normalized = _normalize_text(text)
                if normalized:
                    candidates.append(normalized)
    except Exception:
        return ""

    if not candidates:
        return ""
    return _merge_candidates(candidates)


def _prepare_image(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    boosted = ImageOps.autocontrast(grayscale)
    if min(boosted.size) < 1200:
        boosted = boosted.resize((boosted.width * 2, boosted.height * 2))
    return boosted


def _prepare_threshold_image(image: Image.Image) -> Image.Image:
    prepared = _prepare_image(image)
    thresholded = prepared.point(lambda px: 255 if px > 170 else 0)
    return thresholded


def _normalize_text(text: str) -> str:
    parts = [line.strip() for line in text.splitlines()]
    compact = [part for part in parts if part]
    return "\n".join(compact)


def _merge_candidates(candidates: list[str]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for line in candidate.splitlines():
            normalized = line.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            lines.append(normalized)
    return "\n".join(lines)
