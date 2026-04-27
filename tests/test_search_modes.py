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

    results = search.search("동의", mode="ocr", limit=5)

    assert len(results) == 1
    assert calls["ocr"] == 1
    assert calls["clip"] == 0
    assert results[0]["match_reason"] == "ocr"


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

    results = search.search("남자 얼굴", mode="semantic", limit=5)

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
            {"id": 1, "ocr_rank": -5.0, "ocr_text": "verified", "match_reason": "ocr"},
        ],
    )
    monkeypatch.setattr(
        search.db,
        "search_by_embedding",
        lambda _conn, _embedding, limit=20, city_filter=None, date_from=None, date_to=None: [
            {"id": 1, "distance": 0.2, "ocr_text": "verified"},
            {"id": 2, "distance": 0.7},
        ],
    )
    monkeypatch.setattr(search.models, "encode_text", lambda _query: b"embedding")

    results = search.search("verified", mode="hybrid", limit=5)

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["match_reason"] == "ocr+clip"
    assert results[0]["rank_score"] == 1.0
    assert results[1]["id"] == 2


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

    results = search.search("남자 얼굴", mode="semantic", limit=5)

    assert results[0]["id"] == 2
    assert results[0]["face_match"] is True
    assert results[0]["rank_score"] > results[1]["rank_score"]
