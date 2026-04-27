"""Benchmark configured OCR engines on a folder of screenshots.

Usage:
    python scripts/benchmark_ocr.py C:\path\to\samples

The script does not require ground-truth labels. It reports speed, extracted
character count, and average confidence so engines can be compared on the same
sample set before changing production defaults.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import ocr  # noqa: E402

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/benchmark_ocr.py <image-folder>")
        return 2

    folder = Path(sys.argv[1])
    paths = [path for path in folder.rglob("*") if path.suffix.lower() in SUPPORTED]
    if not paths:
        print(f"No supported images found in {folder}")
        return 1

    engine = os.environ.get("PHOTOMEM_OCR_ENGINE", "tesseract")
    durations = []
    char_counts = []
    confidences = []

    print(f"engine={engine} samples={len(paths)}")
    for path in paths:
        started = time.perf_counter()
        result = ocr.extract(str(path))
        elapsed = time.perf_counter() - started
        durations.append(elapsed)
        char_counts.append(len(result.text))
        block_conf = [block.confidence for block in result.blocks if block.confidence >= 0]
        if block_conf:
            confidences.append(statistics.mean(block_conf))
        print(f"{path.name}\t{elapsed:.2f}s\tchars={len(result.text)}\tblocks={len(result.blocks)}")

    print("---")
    print(f"avg_time={statistics.mean(durations):.2f}s")
    print(f"avg_chars={statistics.mean(char_counts):.1f}")
    if confidences:
        print(f"avg_confidence={statistics.mean(confidences):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
