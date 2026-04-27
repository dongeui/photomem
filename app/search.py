"""Merged OCR and semantic search with lightweight reranking."""
from __future__ import annotations

import logging

from app import db, models

logger = logging.getLogger("photomem.search")

FACE_HINTS = {
    "face", "faces", "person", "people", "portrait", "selfie",
    "얼굴", "사람", "인물", "셀카", "남자", "여자",
}
TEXT_HINTS = {
    "text", "ocr", "document", "receipt", "error", "dialog", "message", "screen",
    "텍스트", "글씨", "문서", "영수증", "오류", "대화", "메시지", "화면",
}
SCREEN_HINTS = {"screenshot", "screen", "ui", "스크린샷", "화면", "앱"}


def search(
    query: str,
    limit: int = 20,
    city_filter: str | None = None,
    date_from: int | None = None,
    date_to: int | None = None,
    mode: str = "hybrid",
) -> list[dict]:
    results, _meta = search_with_meta(
        query,
        limit=limit,
        city_filter=city_filter,
        date_from=date_from,
        date_to=date_to,
        mode=mode,
    )
    return results


def search_with_meta(
    query: str,
    limit: int = 20,
    city_filter: str | None = None,
    date_from: int | None = None,
    date_to: int | None = None,
    mode: str = "hybrid",
) -> tuple[list[dict], dict]:
    if not query.strip():
        return [], {"effective_mode": mode, "intent_reason": "empty"}

    cleaned = query.strip()
    normalized_mode = mode if mode in {"hybrid", "ocr", "semantic"} else "hybrid"
    conn = db.get_connection()
    try:
        ocr_results = []
        clip_results = []

        if normalized_mode in {"hybrid", "ocr"}:
            ocr_results = db.search_by_ocr(conn, cleaned, limit=limit)

        effective_mode, intent_reason = _resolve_effective_mode(cleaned, normalized_mode, ocr_results)

        if effective_mode in {"hybrid", "semantic"}:
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
    merged_by_id: dict[int, dict] = {}

    if ocr_results:
        best_bm25 = min(r["ocr_rank"] for r in ocr_results)
        worst_bm25 = max(r["ocr_rank"] for r in ocr_results)
        bm25_spread = worst_bm25 - best_bm25
        for result in ocr_results:
            if bm25_spread < 1e-9:
                result["rank_score"] = 0.8
            else:
                normalized = (worst_bm25 - result["ocr_rank"]) / bm25_spread
                result["rank_score"] = 0.6 + 0.4 * normalized
            result["rank_score"] = min(1.0, result["rank_score"] + _ocr_match_boost(result.get("ocr_match_kind")))
            result["match_reason"] = result.get("match_reason") or "ocr"
            result["effective_mode"] = effective_mode
            merged.append(result)
            seen_ids.add(result["id"])
            merged_by_id[result["id"]] = result

    for result in clip_results:
        result["match_reason"] = result.get("match_reason") or "clip"
        result["effective_mode"] = effective_mode
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

    _apply_exact_ocr_boost(cleaned, merged)
    _apply_face_boost(cleaned, merged)
    _apply_analysis_boost(cleaned, normalized_mode, merged)
    _set_ocr_excerpt(cleaned, merged)
    merged.sort(key=lambda item: item.get("rank_score", 0.0), reverse=True)
    return merged[:limit], {"effective_mode": effective_mode, "intent_reason": intent_reason}


def _apply_exact_ocr_boost(query: str, results: list[dict]) -> None:
    lowered = query.casefold()
    tokens = [token for token in lowered.split() if token]
    if not lowered:
        return

    for result in results:
        ocr_text = str(result.get("ocr_text") or "")
        if not ocr_text:
            continue
        ocr_lower = ocr_text.casefold()
        if lowered in ocr_lower:
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + 0.22)
            result["ocr_exact_match"] = True
        elif tokens and all(token in ocr_lower for token in tokens):
            result["rank_score"] = min(1.0, float(result.get("rank_score", 0.0)) + 0.12)


def _ocr_match_boost(match_kind: str | None) -> float:
    if match_kind == "word":
        return 0.22
    if match_kind == "phrase":
        return 0.12
    if match_kind == "tokens":
        return 0.04
    return 0.0


def _apply_face_boost(query: str, results: list[dict]) -> None:
    lowered = query.casefold()
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


def _apply_analysis_boost(query: str, mode: str, results: list[dict]) -> None:
    lowered = query.casefold()
    wants_text = mode == "ocr" or any(hint in lowered for hint in TEXT_HINTS)
    wants_screen = any(hint in lowered for hint in SCREEN_HINTS)
    wants_faces = any(hint in lowered for hint in FACE_HINTS)

    for result in results:
        rank_score = float(result.get("rank_score", 0.0))
        text_heavy = bool(result.get("is_text_heavy"))
        document_like = bool(result.get("is_document_like"))
        screenshot_like = bool(result.get("is_screenshot_like"))
        face_count = int(result.get("face_count") or 0)

        if wants_text:
            if text_heavy:
                rank_score += 0.08
            if document_like:
                rank_score += 0.08
            if screenshot_like:
                rank_score += 0.04

        if wants_screen and screenshot_like:
            rank_score += 0.06

        if wants_faces and text_heavy and face_count == 0:
            rank_score -= 0.08

        result["rank_score"] = max(0.0, min(1.0, rank_score))


def _set_ocr_excerpt(query: str, results: list[dict]) -> None:
    lowered = query.casefold()
    for result in results:
        text = str(result.get("ocr_text") or "")
        if not text:
            continue
        normalized = " ".join(part.strip() for part in text.splitlines() if part.strip())
        if lowered and lowered in normalized.casefold():
            start = normalized.casefold().find(lowered)
            excerpt_start = max(0, start - 36)
            excerpt_end = min(len(normalized), start + len(query) + 48)
            result["ocr_excerpt"] = normalized[excerpt_start:excerpt_end].strip()
        else:
            result["ocr_excerpt"] = normalized[:120].strip()


def _resolve_effective_mode(query: str, requested_mode: str, ocr_results: list[dict]) -> tuple[str, str]:
    if requested_mode != "hybrid":
        return requested_mode, "manual"

    lowered = query.casefold()
    if any(hint in lowered for hint in FACE_HINTS):
        return "hybrid", "visual-hint"

    word_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "word"]
    phrase_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "phrase"]
    has_code_like_text = any(ch.isdigit() for ch in query) or any(ch in query for ch in "-_:/[]()#")

    if word_hits and (len(query.strip()) <= 12 or len(word_hits) >= 2 or has_code_like_text):
        return "ocr", "auto-word-match"
    if phrase_hits and has_code_like_text:
        return "ocr", "auto-phrase-code"

    return "hybrid", "fallback"
