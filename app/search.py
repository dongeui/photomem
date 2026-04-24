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

    try:
        query_bytes = models.encode_text(query.strip())
    except Exception as exc:
        logger.error("Text encoding failed: %s", exc)
        return []

    conn = db.get_connection()
    try:
        results = db.search_by_embedding(
            conn,
            query_bytes,
            limit=limit,
            city_filter=city_filter,
            date_from=date_from,
            date_to=date_to,
        )
    finally:
        conn.close()

    return results
