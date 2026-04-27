"""Text-query search: encode query → KNN in sqlite-vec → return photo rows."""
from __future__ import annotations

import logging

from app import db, models

logger = logging.getLogger("photomem.search")

FACE_HINTS = {
    "face", "faces", "person", "people", "portrait", "selfie",
    "얼굴", "사람", "인물", "셀카", "남자", "여자",
}


def search(
    query: str,
    limit: int = 20,
    city_filter: str | None = None,
    date_from: int | None = None,
    date_to: int | None = None,
    mode: str = "hybrid",
) -> list[dict]:
    """
    Encode query text with CLIP text encoder, run KNN, return photo dicts.
    Caller is responsible for opening/closing the connection.
    """
    if not query.strip():
        return []

    cleaned = query.strip()
    normalized_mode = mode if mode in {"hybrid", "ocr", "semantic"} else "hybrid"
    conn = db.get_connection()
    try:
        ocr_results = []
        clip_results = []

        if normalized_mode in {"hybrid", "ocr"}:
            ocr_results = db.search_by_ocr(conn, cleaned, limit=limit)

        if normalized_mode in {"hybrid", "semantic"}:
            try:
                query_bytes = models.encode_text(cleaned)
                clip_results = db.search_by_embedding(
                    conn,
                    query_bytes,
                    limit=limit,
                    city_filter=city_filter,
                    date_from=date_from,
                    date_to=date_to,
                )
            except Exception as exc:
                logger.error("Text encoding failed: %s", exc)
                clip_results = []
    finally:
        conn.close()

    merged: list[dict] = []
    seen_ids: set[int] = set()

    # Normalize BM25 scores (negative, more negative = better) to [0.6, 1.0].
    # This lets strong CLIP matches (≥0.65) beat weak OCR matches.
    if ocr_results:
        best_bm25 = min(r["ocr_rank"] for r in ocr_results)  # most negative = best
        worst_bm25 = max(r["ocr_rank"] for r in ocr_results)
        bm25_spread = worst_bm25 - best_bm25  # positive: range of BM25 scores
        for result in ocr_results:
            if bm25_spread < 1e-9:
                # Single result or all identical — use 0.8 (below ocr+clip combined hits at 1.0)
                result["rank_score"] = 0.8
            else:
                normalized = (worst_bm25 - result["ocr_rank"]) / bm25_spread  # 0=worst, 1=best
                result["rank_score"] = 0.6 + 0.4 * normalized
            merged.append(result)
            seen_ids.add(result["id"])

    merged_by_id: dict[int, dict] = {r["id"]: r for r in merged}

    for result in clip_results:
        result["match_reason"] = result.get("match_reason") or "clip"
        # Convert L2 distance to score. For L2-normalized vectors:
        # cosine_sim = 1 - dist^2/2. Cap at 0.65 so OCR hits can compete.
        dist = float(result["distance"]) if result.get("distance") is not None else 1.5
        result["rank_score"] = max(0.0, min(0.65, 1.0 - (dist ** 2) / 2))
        if result["id"] in seen_ids:
            existing = merged_by_id[result["id"]]
            existing["match_reason"] = "ocr+clip"
            existing["distance"] = result.get("distance")
            existing["rank_score"] = 1.0
            if result.get("ocr_text") and not existing.get("ocr_text"):
                existing["ocr_text"] = result["ocr_text"]
            continue
        merged.append(result)
        merged_by_id[result["id"]] = result
        seen_ids.add(result["id"])

    _apply_face_boost(cleaned, merged)
    merged.sort(key=lambda item: item.get("rank_score", 0.0), reverse=True)
    return merged[:limit]


def _apply_face_boost(query: str, results: list[dict]) -> None:
    lowered = query.lower()
    if not any(hint in lowered for hint in FACE_HINTS):
        return

    for result in results:
        face_count = int(result.get("face_count") or 0)
        if face_count > 0:
            boost = min(0.25, 0.12 + (0.05 * min(face_count, 3)))
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + boost)
            result["face_match"] = True
        else:
            result["rank_score"] = max(0.0, float(result.get("rank_score", 0.0)) - 0.18)
