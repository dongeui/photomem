"""Merged OCR and semantic search with lightweight reranking."""
from __future__ import annotations

import logging

from app import db, models

logger = logging.getLogger("photomem.search")
RRF_K = 60.0

FACE_HINTS = {
    "face", "faces", "person", "people", "portrait", "selfie",
    "얼굴", "사람", "인물", "셀카", "남자", "여자",
}
TEXT_HINTS = {
    "text", "ocr", "document", "receipt", "error", "dialog", "message", "screen",
    "텍스트", "글씨", "문서", "영수증", "오류", "대화", "메시지", "화면",
}
SCREEN_HINTS = {"screenshot", "screen", "ui", "chat", "popup", "스크린샷", "화면", "앱", "대화창", "팝업"}


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

    merged = _fuse_ranked_results(
        cleaned,
        effective_mode,
        intent_reason,
        ocr_results if effective_mode in {"hybrid", "ocr"} else [],
        clip_results,
    )

    _apply_exact_ocr_boost(cleaned, merged)
    _apply_face_boost(cleaned, merged)
    _apply_analysis_boost(cleaned, effective_mode, merged)
    _set_ocr_excerpt(cleaned, merged)
    merged.sort(key=lambda item: item.get("rank_score", 0.0), reverse=True)
    return merged[:limit], {"effective_mode": effective_mode, "intent_reason": intent_reason}


def _fuse_ranked_results(
    query: str,
    effective_mode: str,
    intent_reason: str,
    ocr_results: list[dict],
    clip_results: list[dict],
) -> list[dict]:
    weights = _intent_weights(effective_mode, intent_reason)
    candidates: dict[int, dict] = {}
    channel_hits: dict[int, set[str]] = {}

    def merge_result(result: dict, channel: str, rank: int) -> None:
        photo_id = int(result["id"])
        existing = candidates.setdefault(photo_id, dict(result))
        if existing is not result:
            for key, value in result.items():
                if key not in existing or existing[key] in (None, ""):
                    existing[key] = value
        existing["effective_mode"] = effective_mode
        existing[f"{channel}_rank"] = rank
        existing[f"rrf_{channel}"] = weights[channel] / (RRF_K + rank)
        channel_hits.setdefault(photo_id, set()).add(channel)

    for rank, result in enumerate(ocr_results, start=1):
        result["match_reason"] = result.get("match_reason") or "ocr"
        merge_result(result, "ocr", rank)

    for rank, result in enumerate(clip_results, start=1):
        result["match_reason"] = result.get("match_reason") or "clip"
        merge_result(result, "clip", rank)

    analysis_ranked = _analysis_ranked_candidates(query, effective_mode, list(candidates.values()))
    for rank, result in enumerate(analysis_ranked, start=1):
        result["analysis_rank"] = rank
        result["rrf_analysis"] = weights["analysis"] / (RRF_K + rank)

    fused = []
    for photo_id, result in candidates.items():
        hits = channel_hits.get(photo_id, set())
        result["match_reason"] = _combined_match_reason(hits)
        raw_score = (
            float(result.get("rrf_ocr") or 0.0)
            + float(result.get("rrf_clip") or 0.0)
            + float(result.get("rrf_analysis") or 0.0)
        )
        result["rrf_score"] = raw_score
        fused.append(result)

    if not fused:
        return []

    max_score = max(float(item.get("rrf_score") or 0.0) for item in fused) or 1.0
    for result in fused:
        result["rank_score"] = max(0.0, min(1.0, float(result.get("rrf_score") or 0.0) / max_score))

    fused.sort(key=lambda item: item.get("rank_score", 0.0), reverse=True)
    return fused


def _intent_weights(effective_mode: str, intent_reason: str) -> dict[str, float]:
    if effective_mode == "ocr":
        return {"ocr": 0.74, "clip": 0.08, "analysis": 0.18}
    if effective_mode == "semantic":
        return {"ocr": 0.05, "clip": 0.72, "analysis": 0.23}
    if intent_reason == "auto-mixed":
        return {"ocr": 0.45, "clip": 0.40, "analysis": 0.15}
    return {"ocr": 0.42, "clip": 0.43, "analysis": 0.15}


def _combined_match_reason(hits: set[str]) -> str:
    if "ocr" in hits and "clip" in hits:
        return "ocr+clip"
    if "ocr" in hits:
        return "ocr"
    if "clip" in hits:
        return "clip"
    return "analysis"


def _analysis_ranked_candidates(query: str, effective_mode: str, results: list[dict]) -> list[dict]:
    scored = [
        (score, result)
        for result in results
        if (score := _analysis_signal_score(query, effective_mode, result)) > 0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    for score, result in scored:
        result["analysis_score"] = score
    return [result for _score, result in scored]


def _analysis_signal_score(query: str, effective_mode: str, result: dict) -> float:
    lowered = query.casefold()
    wants_text = effective_mode == "ocr" or any(hint in lowered for hint in TEXT_HINTS)
    wants_screen = any(hint in lowered for hint in SCREEN_HINTS)
    wants_faces = effective_mode == "semantic" or any(hint in lowered for hint in FACE_HINTS)

    text_heavy = bool(result.get("is_text_heavy"))
    document_like = bool(result.get("is_document_like"))
    screenshot_like = bool(result.get("is_screenshot_like"))
    face_count = int(result.get("face_count") or 0)

    score = 0.0
    if wants_text:
        score += 1.2 if text_heavy else 0.0
        score += 1.0 if document_like else 0.0
        score += 0.6 if screenshot_like else 0.0
        score += 0.4 if result.get("ocr_text") else 0.0

    if wants_screen:
        score += 1.0 if screenshot_like else 0.0
        score += 0.5 if document_like else 0.0

    if wants_faces:
        score += min(3.0, face_count * 1.4)
        if text_heavy and face_count == 0:
            score -= 1.0

    return max(0.0, score)


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
    has_face_hint = any(hint in lowered for hint in FACE_HINTS)
    has_text_hint = any(hint in lowered for hint in TEXT_HINTS)
    has_screen_hint = any(hint in lowered for hint in SCREEN_HINTS)

    word_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "word"]
    phrase_hits = [result for result in ocr_results if result.get("ocr_match_kind") == "phrase"]
    has_code_like_text = any(ch.isdigit() for ch in query) or any(ch in query for ch in "-_:/[]()#")
    is_short_query = len(query.strip()) <= 12

    if has_face_hint and (has_text_hint or has_screen_hint or has_code_like_text):
        return "hybrid", "auto-mixed"
    if has_face_hint:
        return "semantic", "auto-face"
    if has_text_hint and not ocr_results:
        return "ocr", "auto-text-hint"
    if has_screen_hint and (word_hits or phrase_hits or is_short_query):
        return "ocr", "auto-screen-text"
    if has_code_like_text:
        return "ocr", "auto-code"
    if word_hits and (is_short_query or len(word_hits) >= 2 or has_code_like_text):
        return "ocr", "auto-word-match"
    if phrase_hits and has_code_like_text:
        return "ocr", "auto-phrase-code"

    return "hybrid", "fallback"
