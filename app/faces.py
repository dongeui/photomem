"""
Face detection helpers built on OpenCV Haar cascades.
"""
from __future__ import annotations

from pathlib import Path

import cv2

_cascade = None


def detect_faces(image_path: str) -> int:
    cascade = _get_cascade()
    image = cv2.imread(str(Path(image_path)), cv2.IMREAD_COLOR)
    if image is None:
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    return int(len(faces))


def _get_cascade():
    global _cascade
    if _cascade is None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(str(cascade_path))
        if _cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return _cascade
