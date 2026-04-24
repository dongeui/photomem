"""
CLIP ViT-B/32 via open_clip_torch.

Model is downloaded from HuggingFace on first run (~350MB) and cached by open_clip.
Visual encoder runs in ProcessPoolExecutor workers (one per worker process).
Text encoder runs in the main process for search queries.
"""
from __future__ import annotations

import struct
import numpy as np

# ── single shared model (both image and text encoding in main process) ───────
# Using one process avoids loading PyTorch twice (saves ~800MB RAM on NAS).
# asyncio.to_thread() keeps the event loop responsive during CPU inference.

_model = None
_preprocess = None
_tokenizer = None


def ensure_models() -> None:
    """Load CLIP model once at startup. Shared for both image and text encoding."""
    global _model, _preprocess, _tokenizer
    import open_clip
    print("[photomem] Loading CLIP model (downloads ~350MB on first run) ...", flush=True)
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    _model.eval()
    for p in _model.parameters():
        p.requires_grad_(False)
    _tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print("[photomem] Model ready.", flush=True)


def worker_init() -> None:
    """No-op — model is loaded in the main process. Kept for API compatibility."""
    pass


def load_text_encoder() -> None:
    """No-op — model loaded by ensure_models(). Kept for API compatibility."""
    pass


# ── embedding helpers ─────────────────────────────────────────────────────────

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


def embedding_to_bytes(v: np.ndarray) -> bytes:
    flat = _l2_normalize(v.flatten().astype(np.float32))
    return struct.pack(f"{len(flat)}f", *flat)


# ── inference ─────────────────────────────────────────────────────────────────

def encode_image(image_path: str) -> bytes:
    """Encode image → embedding bytes. Call via asyncio.to_thread() from async code."""
    if _model is None or _preprocess is None:
        raise RuntimeError("ensure_models() was not called")
    import torch
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = _model.encode_image(tensor)
    return embedding_to_bytes(features.numpy())


def encode_text(query: str) -> bytes:
    """Encode text query → embedding bytes."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("ensure_models() was not called")
    import torch
    tokens = _tokenizer([query])
    with torch.no_grad():
        features = _model.encode_text(tokens)
    return embedding_to_bytes(features.numpy())
