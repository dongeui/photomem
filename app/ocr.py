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
            prepared = _prepare_image(image)
            text = pytesseract.image_to_string(prepared, lang=OCR_LANG, config="--psm 6")
    except Exception:
        return ""

    return _normalize_text(text)


def _prepare_image(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    boosted = ImageOps.autocontrast(grayscale)
    if min(boosted.size) < 1200:
        boosted = boosted.resize((boosted.width * 2, boosted.height * 2))
    return boosted


def _normalize_text(text: str) -> str:
    parts = [line.strip() for line in text.splitlines()]
    compact = [part for part in parts if part]
    return "\n".join(compact)
