"""Thumbnail generation and serving. Thumbnails are pre-generated and cached."""
import os
from pathlib import Path

from PIL import Image

THUMB_DIR = Path(os.environ.get("PHOTOMEM_CACHE", "/app/cache")) / "thumbnails"
THUMB_SIZE = (300, 300)
THUMB_QUALITY = 75


def thumb_path(photo_id: int) -> Path:
    bucket = photo_id % 256  # spread across 256 subdirs to avoid huge directories
    return THUMB_DIR / f"{bucket:02x}" / f"{photo_id}.jpg"


def generate_thumbnail(photo_id: int, source_path: str) -> Path | None:
    dest = thumb_path(photo_id)
    if dest.exists():
        return dest
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(source_path)
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        # Convert to RGB to ensure JPEG compatibility (e.g. RGBA PNGs)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dest, "JPEG", quality=THUMB_QUALITY, optimize=True)
        return dest
    except Exception:
        return None


def thumbnail_exists(photo_id: int) -> bool:
    return thumb_path(photo_id).exists()
