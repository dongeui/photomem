"""
Face detection helpers built on OpenCV YuNet.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = float(os.environ.get("PHOTOMEM_FACE_SCORE_THRESHOLD", "0.88"))
NMS_THRESHOLD = float(os.environ.get("PHOTOMEM_FACE_NMS_THRESHOLD", "0.3"))
TOP_K = int(os.environ.get("PHOTOMEM_FACE_TOP_K", "5000"))
MIN_FACE_SIZE = int(os.environ.get("PHOTOMEM_FACE_MIN_SIZE", "40"))
MIN_FACE_AREA_RATIO = float(os.environ.get("PHOTOMEM_FACE_MIN_AREA_RATIO", "0.002"))

_detector = None


def detect_faces(image_path: str) -> int:
    image = cv2.imread(str(Path(image_path)), cv2.IMREAD_COLOR)
    if image is None:
        return 0

    detector = _get_detector()
    detector.setInputSize((int(image.shape[1]), int(image.shape[0])))
    _status, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return 0

    return _count_valid_faces(faces, image.shape[1], image.shape[0])


def _count_valid_faces(faces: np.ndarray, image_width: int, image_height: int) -> int:
    if faces is None or len(faces) == 0:
        return 0

    min_area = max(1.0, float(image_width * image_height) * MIN_FACE_AREA_RATIO)
    valid = 0
    for face in faces:
        width = float(face[2])
        height = float(face[3])
        score = float(face[-1])
        area = width * height
        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue
        if area < min_area:
            continue
        if score < SCORE_THRESHOLD:
            continue
        valid += 1
    return valid


def _get_detector():
    global _detector
    if _detector is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"YuNet model not found at {MODEL_PATH}")
        _detector = _create_detector()
    return _detector


def _create_detector():
    if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
        return cv2.FaceDetectorYN.create(
            str(MODEL_PATH),
            "",
            (320, 320),
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            TOP_K,
        )
    if hasattr(cv2, "FaceDetectorYN_create"):
        return cv2.FaceDetectorYN_create(
            str(MODEL_PATH),
            "",
            (320, 320),
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            TOP_K,
        )
    raise RuntimeError("OpenCV YuNet face detector is not available in this build")
