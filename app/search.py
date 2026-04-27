"""Text-query search: encode query → KNN in sqlite-vec → return photo rows."""
from __future__ import annotations

import logging

from app import db, models

logger = logging.getLogger("photomem.search")


def search(
    query: str,
    limit: int = 20,
    city_filter: str | None = None,
    date_from: int | None = None,
    date_to: int | None = None,
) -> list[dict]:
    """
    Encode query text with CLIP text encoder, run KNN, return photo dicts.
    Caller is responsible for opening/closing the connection.
    """
    if not query.strip():
        return []

    cleaned = query.strip()
    conn = db.get_connection()
    try:
        ocr_results = db.search_by_ocr(conn, cleaned, limit=limit)
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
        bm25_spread = max(abs(best_bm25 - worst_bm25), 1e-9)
        for result in ocr_results:
            normalized = (result["ocr_rank"] - worst_bm25) / bm25_spread  # 0=worst, 1=best
            result["rank_score"] = 0.6 + 0.4 * normalized
            merged.append(result)
            seen_ids.add(result["id"])

    for index, result in enumerate(clip_results):
        result["match_reason"] = result.get("match_reason") or "clip"
        result["rank_score"] = max(0.0, 0.65 - (index * 0.01))
        if result["id"] in seen_ids:
            existing = next(item for item in merged if item["id"] == result["id"])
            existing["match_reason"] = "ocr+clip"
            existing["distance"] = result.get("distance")
            existing["rank_score"] = 1.0
            if result.get("ocr_text") and not existing.get("ocr_text"):
                existing["ocr_text"] = result["ocr_text"]
            continue
        merged.append(result)
        seen_ids.add(result["id"])

    return merged[:limit]
