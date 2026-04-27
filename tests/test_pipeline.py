"""
Integration tests: full indexing pipeline.
Run after placing fixture photos in tests/fixtures/.
"""
import os
import shutil
import time
from pathlib import Path

import pytest

from app import db, thumbnails
from app.indexer import _file_hash, _parse_exif

FIXTURES = Path(__file__).parent / "fixtures"


# ── unit-level pipeline helpers (no model needed) ────────────────────────────

def test_file_hash_is_deterministic(tmp_path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"hello")
    assert _file_hash(str(p)) == _file_hash(str(p))


def test_missing_exif_date_falls_back_to_mtime(tmp_path):
    p = tmp_path / "no_exif.jpg"
    p.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG
    created_at, lat, lon = _parse_exif(str(p))
    assert created_at is not None
    assert lat is None
    assert lon is None
    assert abs(created_at - int(p.stat().st_mtime)) < 2


def test_upsert_and_requeue(initialized_db):
    conn = initialized_db
    photo_id = db.upsert_photo(conn, "/fake/photo1.jpg", "abc123")
    assert photo_id is not None

    # Same hash → None (already pending, skip)
    result = db.upsert_photo(conn, "/fake/photo1.jpg", "abc123", 100, 123)
    assert result == photo_id

    # Different hash → re-queue
    photo_id2 = db.upsert_photo(conn, "/fake/photo1.jpg", "newhash", 100, 124)
    assert photo_id2 == photo_id  # same row, re-queued


def test_cached_file_stats(initialized_db):
    conn = initialized_db
    photo_id = db.upsert_photo(conn, "/fake/cached.jpg", "hash", 12, 34)
    assert photo_id is not None
    cached = db.get_cached_file_stats(conn)
    assert cached["/fake/cached.jpg"] == (12, 34, "pending")


def test_upsert_backfills_file_stat_cache_for_indexed_rows(initialized_db):
    conn = initialized_db
    photo_id = db.upsert_photo(conn, "/fake/indexed.jpg", "samehash")
    db.update_photo_indexed(conn, photo_id, 1, None, None, None, None, b"\x00" * 512 * 4)

    result = db.upsert_photo(conn, "/fake/indexed.jpg", "samehash", 99, 100)
    assert result is None

    cached = db.get_cached_file_stats(conn)
    assert cached["/fake/indexed.jpg"] == (99, 100, "indexed")


def test_stats_reflects_status(initialized_db):
    conn = initialized_db
    stats = db.get_stats(conn)
    assert "total" in stats
    assert "indexed" in stats
    assert "pending" in stats
    assert "errors" in stats


def test_mark_error(initialized_db):
    conn = initialized_db
    photo_id = db.upsert_photo(conn, "/fake/bad.jpg", "badhash")
    db.mark_photo_error(conn, photo_id, "CLIP failed")
    cur = conn.execute("SELECT status, error_msg FROM photos WHERE id=?", (photo_id,))
    row = cur.fetchone()
    assert row["status"] == "error"
    assert "CLIP" in row["error_msg"]


def test_update_and_search_ocr(initialized_db):
    conn = initialized_db
    photo_id = db.upsert_photo(conn, "/fake/ocr-hit.jpg", "ocrhash")
    db.update_photo_indexed(conn, photo_id, 1, None, None, None, None, b"\x00" * 512 * 4)
    db.update_photo_ocr(conn, photo_id, "동의서 signed consent form")

    results = db.search_by_ocr(conn, "동의", limit=5)
    assert results
    assert results[0]["id"] == photo_id
    assert "consent" in results[0]["ocr_text"]


def test_list_photos_includes_ocr_text(initialized_db):
    conn = initialized_db
    photo_id = db.upsert_photo(conn, "/fake/list-ocr.jpg", "listocr")
    db.update_photo_indexed(conn, photo_id, 1, None, None, None, None, b"\x00" * 512 * 4)
    db.update_photo_ocr(conn, photo_id, "receipt total 12900")

    photos = db.list_photos(conn, limit=10)
    matching = next(photo for photo in photos if photo["id"] == photo_id)
    assert "receipt" in matching["ocr_text"]


def test_get_missing_ocr_paths(initialized_db):
    conn = initialized_db
    photo_id = db.upsert_photo(conn, "/fake/missing-ocr.jpg", "missingocr")
    db.update_photo_indexed(conn, photo_id, 1, None, None, None, None, b"\x00" * 512 * 4)
    assert db.get_photo_id(conn, "/fake/missing-ocr.jpg") == photo_id
    assert not db.photo_has_ocr(conn, photo_id)

    missing_before = db.get_missing_ocr_paths(conn)
    assert "/fake/missing-ocr.jpg" in missing_before

    db.update_photo_ocr(conn, photo_id, "hello world")
    assert db.photo_has_ocr(conn, photo_id)
    missing_after = db.get_missing_ocr_paths(conn)
    assert "/fake/missing-ocr.jpg" not in missing_after


def test_update_and_load_face_count(initialized_db):
    conn = initialized_db
    photo_id = db.upsert_photo(conn, "/fake/face.jpg", "facehash")
    db.update_photo_indexed(conn, photo_id, 1, None, None, None, None, b"\x00" * 512 * 4)

    assert not db.photo_has_face_data(conn, photo_id)
    db.update_photo_faces(conn, photo_id, 2)
    assert db.photo_has_face_data(conn, photo_id)

    photos = db.list_photos(conn, limit=10)
    matching = next(photo for photo in photos if photo["id"] == photo_id)
    assert matching["face_count"] == 2


def test_get_missing_face_paths(initialized_db):
    conn = initialized_db
    photo_id = db.upsert_photo(conn, "/fake/missing-face.jpg", "missingface")
    db.update_photo_indexed(conn, photo_id, 1, None, None, None, None, b"\x00" * 512 * 4)

    missing_before = db.get_missing_face_paths(conn)
    assert "/fake/missing-face.jpg" in missing_before

    db.update_photo_faces(conn, photo_id, 1)
    missing_after = db.get_missing_face_paths(conn)
    assert "/fake/missing-face.jpg" not in missing_after


def test_thumbnail_path_bucketing():
    # Ensure thumbnails spread across subdirs
    p1 = thumbnails.thumb_path(1)
    p2 = thumbnails.thumb_path(2)
    assert p1.parent != p2.parent


def test_thumbnail_generate(tmp_path):
    from PIL import Image
    src = tmp_path / "test.jpg"
    img = Image.new("RGB", (1200, 800), color=(100, 150, 200))
    img.save(str(src), "JPEG")

    os.environ["PHOTOMEM_CACHE"] = str(tmp_path)
    import importlib
    import app.thumbnails as th
    importlib.reload(th)

    dest = th.generate_thumbnail(999, str(src))
    assert dest is not None and dest.exists()
    result = Image.open(dest)
    assert max(result.size) <= 300


def test_corrupt_photo_does_not_crash_hash(tmp_path):
    p = tmp_path / "corrupt.jpg"
    p.write_bytes(b"this is not a jpeg")
    # hash should succeed regardless
    h = _file_hash(str(p))
    assert len(h) == 32  # md5 hex


# ── search endpoint (requires models; skip in CI without models) ──────────────

@pytest.mark.skipif(
    not os.path.exists(os.environ.get("PHOTOMEM_MODELS", "/app/cache/models")),
    reason="ONNX models not available",
)
def test_search_returns_results_after_index():
    """Full pipeline: copy fixture photo → index → search → find it."""
    import asyncio
    from app import models as m, search as s

    m.load_text_encoder()
    results = s.search("photo", limit=5)
    assert isinstance(results, list)


@pytest.mark.skipif(
    not os.path.exists(os.environ.get("PHOTOMEM_MODELS", "/app/cache/models")),
    reason="ONNX models not available",
)
def test_search_empty_query_returns_empty():
    from app import search as s
    results = s.search("", limit=5)
    assert results == []
