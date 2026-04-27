from app import search


def test_search_ocr_mode_skips_clip(monkeypatch):
    calls = {"ocr": 0, "clip": 0}

    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())

    def fake_ocr(_conn, query, limit=20):
        calls["ocr"] += 1
        return [{"id": 1, "ocr_rank": -5.0, "ocr_text": query, "match_reason": "ocr"}]

    def fake_clip(*args, **kwargs):
        calls["clip"] += 1
        return []

    monkeypatch.setattr(search.db, "search_by_ocr", fake_ocr)
    monkeypatch.setattr(search.db, "search_by_embedding", fake_clip)
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"ignored")

    results = search.search("\uB3D9\uC758", mode="ocr", limit=5)

    assert len(results) == 1
    assert calls["ocr"] == 1
    assert calls["clip"] == 0
    assert results[0]["match_reason"] == "ocr"


def test_hybrid_auto_routes_to_ocr_on_strong_word_hits(monkeypatch):
    calls = {"clip": 0}

    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())
    monkeypatch.setattr(
        search.db,
        "search_by_ocr",
        lambda _conn, _query, limit=20: [
            {"id": 1, "ocr_rank": -5.0, "ocr_text": "\uB3D9\uC758 \uD544\uC694", "ocr_match_kind": "word", "match_reason": "ocr"},
            {"id": 2, "ocr_rank": -4.0, "ocr_text": "\uC57D\uAD00 \uB3D9\uC758", "ocr_match_kind": "word", "match_reason": "ocr"},
        ],
    )

    def fake_clip(*args, **kwargs):
        calls["clip"] += 1
        return []

    monkeypatch.setattr(search.db, "search_by_embedding", fake_clip)
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"ignored")

    results, meta = search.search_with_meta("\uB3D9\uC758", mode="hybrid", limit=5)

    assert len(results) == 2
    assert meta["effective_mode"] == "ocr"
    assert meta["intent_reason"] == "auto-word-match"
    assert calls["clip"] == 0


def test_hybrid_auto_routes_face_queries_to_semantic(monkeypatch):
    calls = {"clip": 0}

    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())
    monkeypatch.setattr(
        search.db,
        "search_by_ocr",
        lambda _conn, _query, limit=20: [
            {"id": 1, "ocr_rank": -5.0, "ocr_text": "\ub0a8\uc790 \uc5bc\uad74", "ocr_match_kind": "word", "match_reason": "ocr"},
        ],
    )

    def fake_clip(*args, **kwargs):
        calls["clip"] += 1
        return [{"id": 2, "distance": 0.2, "face_count": 1}]

    monkeypatch.setattr(search.db, "search_by_embedding", fake_clip)
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"ignored")

    results, meta = search.search_with_meta("\ub0a8\uc790 \uc5bc\uad74", mode="hybrid", limit=5)

    assert meta["effective_mode"] == "semantic"
    assert meta["intent_reason"] == "auto-face"
    assert calls["clip"] == 1
    assert results[0]["id"] == 2
    assert results[0]["match_reason"] == "clip"


def test_hybrid_keeps_mixed_queries_in_hybrid(monkeypatch):
    calls = {"clip": 0}

    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())
    monkeypatch.setattr(
        search.db,
        "search_by_ocr",
        lambda _conn, _query, limit=20: [
            {"id": 1, "ocr_rank": -5.0, "ocr_text": "\ub0a8\uc790 \uba54\uc2dc\uc9c0", "ocr_match_kind": "word", "match_reason": "ocr"},
        ],
    )

    def fake_clip(*args, **kwargs):
        calls["clip"] += 1
        return [{"id": 2, "distance": 0.25, "face_count": 1}]

    monkeypatch.setattr(search.db, "search_by_embedding", fake_clip)
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"ignored")

    _results, meta = search.search_with_meta("\ub0a8\uc790 \uba54\uc2dc\uc9c0", mode="hybrid", limit=5)

    assert meta["effective_mode"] == "hybrid"
    assert meta["intent_reason"] == "auto-mixed"
    assert calls["clip"] == 1


def test_hybrid_auto_routes_code_queries_to_ocr(monkeypatch):
    calls = {"clip": 0}

    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())
    monkeypatch.setattr(search.db, "search_by_ocr", lambda _conn, _query, limit=20: [])

    def fake_clip(*args, **kwargs):
        calls["clip"] += 1
        return []

    monkeypatch.setattr(search.db, "search_by_embedding", fake_clip)
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"ignored")

    results, meta = search.search_with_meta("T001", mode="hybrid", limit=5)

    assert results == []
    assert meta["effective_mode"] == "ocr"
    assert meta["intent_reason"] == "auto-code"
    assert calls["clip"] == 0


def test_search_semantic_mode_skips_ocr(monkeypatch):
    calls = {"ocr": 0, "clip": 0}

    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())

    def fake_ocr(*args, **kwargs):
        calls["ocr"] += 1
        return []

    def fake_clip(_conn, _embedding, limit=20, city_filter=None, date_from=None, date_to=None):
        calls["clip"] += 1
        return [{"id": 7, "distance": 0.5, "match_reason": "clip"}]

    monkeypatch.setattr(search.db, "search_by_ocr", fake_ocr)
    monkeypatch.setattr(search.db, "search_by_embedding", fake_clip)
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"embedding")

    results = search.search("\uB0A8\uC790 \uC5BC\uAD74", mode="semantic", limit=5)

    assert len(results) == 1
    assert calls["ocr"] == 0
    assert calls["clip"] == 1
    assert results[0]["match_reason"] == "clip"


def test_search_hybrid_merges_ocr_and_clip(monkeypatch):
    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())
    monkeypatch.setattr(
        search.db,
        "search_by_ocr",
        lambda _conn, _query, limit=20: [
            {"id": 1, "ocr_rank": -5.0, "ocr_text": "verified domain issue", "match_reason": "ocr", "ocr_match_kind": "word"},
        ],
    )
    monkeypatch.setattr(
        search.db,
        "search_by_embedding",
        lambda _conn, _embedding, limit=20, city_filter=None, date_from=None, date_to=None: [
            {"id": 1, "distance": 0.2, "ocr_text": "verified domain issue"},
            {"id": 2, "distance": 0.7},
        ],
    )
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"embedding")

    results = search.search("verified domain issue", mode="hybrid", limit=5)

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["match_reason"] == "ocr+clip"
    assert results[0]["rank_score"] == 1.0
    assert results[1]["id"] == 2


def test_hybrid_rrf_prefers_cross_channel_agreement(monkeypatch):
    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())
    monkeypatch.setattr(
        search.db,
        "search_by_ocr",
        lambda _conn, _query, limit=20: [
            {"id": 2, "ocr_rank": -4.0, "ocr_text": "dialog screen", "match_reason": "ocr", "ocr_match_kind": "word"},
        ],
    )
    monkeypatch.setattr(
        search.db,
        "search_by_embedding",
        lambda _conn, _embedding, limit=20, city_filter=None, date_from=None, date_to=None: [
            {"id": 3, "distance": 0.1},
            {"id": 2, "distance": 0.2},
        ],
    )
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"embedding")

    results = search.search("dialog issue today", mode="hybrid", limit=5)

    assert results[0]["id"] == 2
    assert results[0]["match_reason"] == "ocr+clip"
    assert results[0]["rank_score"] == 1.0


def test_face_hint_query_boosts_face_results(monkeypatch):
    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())
    monkeypatch.setattr(search.db, "search_by_ocr", lambda _conn, _query, limit=20: [])
    monkeypatch.setattr(
        search.db,
        "search_by_embedding",
        lambda _conn, _embedding, limit=20, city_filter=None, date_from=None, date_to=None: [
            {"id": 1, "distance": 0.45, "face_count": 0},
            {"id": 2, "distance": 0.48, "face_count": 2},
        ],
    )
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"embedding")

    results = search.search("\uB0A8\uC790 \uC5BC\uAD74", mode="semantic", limit=5)

    assert results[0]["id"] == 2
    assert results[0]["face_match"] is True
    assert results[0]["rank_score"] > results[1]["rank_score"]


def test_exact_ocr_match_boosts_text_result(monkeypatch):
    class DummyConn:
        def close(self):
            pass

    monkeypatch.setattr(search.db, "get_connection", lambda: DummyConn())
    monkeypatch.setattr(
        search.db,
        "search_by_ocr",
        lambda _conn, _query, limit=20: [
            {
                "id": 3,
                "ocr_rank": -5.0,
                "ocr_text": "\uC804\uC1A1 \uC2E4\uD328 \uC624\uB958\uCF54\uB4DC T001",
                "ocr_match_kind": "word",
                "text_char_count": 12,
                "text_line_count": 1,
                "is_text_heavy": 1,
                "is_document_like": 1,
                "is_screenshot_like": 1,
            },
        ],
    )
    monkeypatch.setattr(search.db, "search_by_embedding", lambda *args, **kwargs: [])
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"embedding")

    results = search.search("\uC804\uC1A1 \uC2E4\uD328", mode="ocr", limit=5)

    assert results[0]["ocr_exact_match"] is True
    assert "\uC804\uC1A1 \uC2E4\uD328" in results[0]["ocr_excerpt"]
    assert results[0]["rank_score"] > 0.9


def test_ocr_word_match_beats_phrase_match(initialized_db):
    from app import db as real_db

    conn = initialized_db
    first_id = real_db.upsert_photo(conn, "/fake/word-hit.png", "wordhit")
    second_id = real_db.upsert_photo(conn, "/fake/phrase-hit.png", "phrasehit")
    real_db.update_photo_indexed(conn, first_id, 1, None, None, None, None, b"\x00" * 512 * 4)
    real_db.update_photo_indexed(conn, second_id, 1, None, None, None, None, b"\x00" * 512 * 4)
    real_db.update_photo_ocr(conn, first_id, "\uB3D9\uC758 \uD544\uC694\uD569\uB2C8\uB2E4")
    real_db.update_photo_ocr(conn, second_id, "\uC815\uB3D9\uC758\uC778 \uD14C\uC2A4\uD2B8")

    results = real_db.search_by_ocr(conn, "\uB3D9\uC758", limit=50)

    by_id = {result["id"]: result for result in results}
    assert by_id[first_id]["ocr_match_kind"] == "word"
    assert by_id[second_id]["ocr_match_kind"] == "phrase"
    assert results.index(by_id[first_id]) < results.index(by_id[second_id])
