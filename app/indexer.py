"""
Background indexer: watches the mounted photo folder, encodes photos with CLIP,
and writes metadata plus embeddings to SQLite.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import piexif
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app import analysis, db, faces, geocoder, models, ocr, thumbnails

logger = logging.getLogger("photomem.indexer")

PHOTOS_DIR = Path(os.environ.get("PHOTOMEM_PHOTOS", "/data/photos"))
SCAN_INTERVAL = int(os.environ.get("PHOTOMEM_SCAN_INTERVAL", "300"))  # seconds
INDEX_WORKERS = max(1, int(os.environ.get("PHOTOMEM_INDEX_WORKERS", "1")))

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".tif"}

# Shared state. These are mutated only from the main asyncio loop.
_queue: asyncio.Queue[str] = asyncio.Queue()
_queued_paths: set[str] = set()
_running = False
_last_heartbeat = 0.0
_active_paths: dict[int, tuple[str, float]] = {}
_last_completed_at: float | None = None
_last_scan_at: float | None = None
_last_scan_enqueued = 0


def _file_hash(path: str) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_exif(path: str) -> tuple[int | None, float | None, float | None]:
    """Return (unix_timestamp, lat, lon). Falls back to mtime if no EXIF date."""
    lat = lon = None
    created_at = None
    try:
        exif = piexif.load(path)
        raw_date = (
            exif.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal)
            or exif.get("0th", {}).get(piexif.ImageIFD.DateTime)
        )
        if raw_date:
            from datetime import datetime

            try:
                dt = datetime.strptime(raw_date.decode(), "%Y:%m:%d %H:%M:%S")
                created_at = int(dt.timestamp())
            except Exception:
                pass

        gps = exif.get("GPS", {})
        if gps:

            def _rational(v):
                return (
                    v[0][0] / v[0][1]
                    + v[1][0] / (v[1][1] * 60)
                    + v[2][0] / (v[2][1] * 3600)
                )

            lat_raw = gps.get(piexif.GPSIFD.GPSLatitude)
            lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef)
            lon_raw = gps.get(piexif.GPSIFD.GPSLongitude)
            lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef)
            if lat_raw and lon_raw:
                lat = _rational(lat_raw)
                lon = _rational(lon_raw)
                if lat_ref and lat_ref.decode() == "S":
                    lat = -lat
                if lon_ref and lon_ref.decode() == "W":
                    lon = -lon
    except Exception:
        pass

    if created_at is None:
        created_at = int(os.path.getmtime(path))

    return created_at, lat, lon


def _enqueue_path_nowait(path: str) -> bool:
    if Path(path).suffix.lower() not in SUPPORTED_EXT:
        return False
    if path in _queued_paths:
        return False
    _queued_paths.add(path)
    _queue.put_nowait(path)
    return True


class _PhotoHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _enqueue(self, path: str) -> None:
        self._loop.call_soon_threadsafe(_enqueue_path_nowait, path)

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_moved(self, event) -> None:
        if not event.is_directory:
            self._enqueue(event.dest_path)


async def _process_path(
    conn,
    executor: ThreadPoolExecutor,
    worker_id: int,
    path: str,
) -> None:
    global _last_completed_at, _last_heartbeat

    if not os.path.isfile(path):
        return

    try:
        stat = os.stat(path)
        file_size = stat.st_size
        modified_at = int(stat.st_mtime)
    except OSError:
        return

    try:
        file_hash = _file_hash(path)
    except OSError:
        return

    photo_id = db.upsert_photo(conn, path, file_hash, file_size, modified_at)
    if photo_id is None:
        existing_id = db.get_photo_id(conn, path)
        if existing_id is not None:
            thumbnails.generate_thumbnail(existing_id, path)
            ocr_text = None
            if not db.photo_has_ocr(conn, existing_id):
                ocr_result = ocr.extract(path)
                ocr_text = ocr_result.text
                db.update_photo_ocr(conn, existing_id, ocr_result.text, ocr_result.blocks, ocr_result.engine)
            if not db.photo_has_face_data(conn, existing_id):
                db.update_photo_faces(conn, existing_id, faces.detect_faces(path))
            if not db.photo_has_analysis_data(conn, existing_id):
                if ocr_text is None:
                    ocr_text = ocr.extract_text(path)
                db.update_photo_analysis(conn, existing_id, analysis.extract_analysis(path, ocr_text))
        return

    thumbnails.generate_thumbnail(photo_id, path)
    ocr_result = ocr.extract(path)
    ocr_text = ocr_result.text
    db.update_photo_ocr(conn, photo_id, ocr_result.text, ocr_result.blocks, ocr_result.engine)
    db.update_photo_faces(conn, photo_id, faces.detect_faces(path))
    db.update_photo_analysis(conn, photo_id, analysis.extract_analysis(path, ocr_text))
    created_at, lat, lon = _parse_exif(path)

    city = country = None
    if lat is not None and lon is not None:
        city, country = geocoder.reverse_geocode(lat, lon)

    _active_paths[worker_id] = (path, time.time())
    try:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(executor, models.encode_image, path)
    except Exception as exc:
        db.mark_photo_error(conn, photo_id, str(exc))
        logger.warning("CLIP failed for %s: %s", path, exc)
        return
    finally:
        _active_paths.pop(worker_id, None)

    db.update_photo_indexed(conn, photo_id, created_at, lat, lon, city, country, embedding)
    _last_completed_at = time.time()
    _last_heartbeat = _last_completed_at


async def _scan_directory() -> int:
    """Scan PHOTOS_DIR and enqueue new or changed files. Returns count enqueued."""
    global _last_scan_at, _last_scan_enqueued

    count = 0
    conn = db.get_connection()
    try:
        cached = db.get_cached_file_stats(conn)
    finally:
        conn.close()

    for root, _dirs, files in os.walk(str(PHOTOS_DIR)):
        for fname in files:
            path = os.path.join(root, fname)
            if Path(path).suffix.lower() not in SUPPORTED_EXT:
                continue

            try:
                stat = os.stat(path)
            except OSError:
                continue

            cached_entry = cached.get(path)
            if cached_entry:
                cached_size, cached_mtime, cached_status = cached_entry
                unchanged = cached_size == stat.st_size and cached_mtime == int(stat.st_mtime)
                if unchanged and cached_status == "indexed":
                    continue

            if _enqueue_path_nowait(path):
                count += 1

    _last_scan_at = time.time()
    _last_scan_enqueued = count
    logger.info("Scanned photos: enqueued %d new or changed file(s)", count)
    return count


async def run_indexer() -> None:
    global _running, _last_heartbeat
    _running = True
    _last_heartbeat = time.time()

    loop = asyncio.get_running_loop()
    setup_conn = db.get_connection()
    try:
        pending_count = db.requeue_pending(setup_conn)
        for path in db.get_pending_paths(setup_conn):
            _enqueue_path_nowait(path)
        missing_ocr_paths = db.get_missing_ocr_paths(setup_conn)
        for path in missing_ocr_paths:
            _enqueue_path_nowait(path)
        missing_face_paths = db.get_missing_face_paths(setup_conn)
        for path in missing_face_paths:
            _enqueue_path_nowait(path)
        missing_analysis_paths = db.get_missing_analysis_paths(setup_conn)
        for path in missing_analysis_paths:
            _enqueue_path_nowait(path)
        if pending_count:
            logger.info("Re-queued %d pending photos from previous run", pending_count)
        if missing_ocr_paths:
            logger.info("Queued %d indexed photo(s) for OCR backfill", len(missing_ocr_paths))
        if missing_face_paths:
            logger.info("Queued %d indexed photo(s) for face backfill", len(missing_face_paths))
        if missing_analysis_paths:
            logger.info("Queued %d indexed photo(s) for analysis backfill", len(missing_analysis_paths))
    finally:
        setup_conn.close()

    observer = Observer()
    observer.schedule(_PhotoHandler(loop), str(PHOTOS_DIR), recursive=True)
    observer.start()

    executor = ThreadPoolExecutor(max_workers=INDEX_WORKERS, thread_name_prefix="clip")

    await _scan_directory()

    async def _periodic_scan():
        while _running:
            await asyncio.sleep(SCAN_INTERVAL)
            await _scan_directory()

    periodic_task = asyncio.create_task(_periodic_scan())

    async def _worker(worker_id: int):
        global _last_heartbeat
        conn = db.get_connection()
        try:
            while _running:
                try:
                    path = await asyncio.wait_for(_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

                _queued_paths.discard(path)
                _last_heartbeat = time.time()
                try:
                    await _process_path(conn, executor, worker_id, path)
                except Exception:
                    logger.exception("Unexpected indexing failure for %s", path)
                finally:
                    _queue.task_done()
        finally:
            conn.close()

    worker_tasks = [asyncio.create_task(_worker(i)) for i in range(INDEX_WORKERS)]
    logger.info("Started %d index worker(s)", INDEX_WORKERS)

    try:
        await asyncio.gather(*worker_tasks)
    finally:
        periodic_task.cancel()
        for task in worker_tasks:
            task.cancel()
        observer.stop()
        observer.join()
        executor.shutdown(wait=False, cancel_futures=True)


def get_status() -> dict:
    conn = db.get_connection()
    stats = db.get_stats(conn)
    conn.close()

    now = time.time()
    heartbeat_age = now - _last_heartbeat if _last_heartbeat else None
    active = [
        {
            "worker_id": worker_id,
            "file": Path(path).name,
            "elapsed": int(now - started_at),
        }
        for worker_id, (path, started_at) in sorted(_active_paths.items())
    ]

    return {
        **stats,
        "running": _running,
        "index_workers": INDEX_WORKERS,
        "queue_size": _queue.qsize(),
        "active_workers": len(active),
        "active_files": active,
        "current_file": active[0]["file"] if active else None,
        "current_elapsed": active[0]["elapsed"] if active else None,
        "last_completed_at": int(_last_completed_at) if _last_completed_at else None,
        "scan_interval": SCAN_INTERVAL,
        "last_scan_at": int(_last_scan_at) if _last_scan_at else None,
        "last_scan_enqueued": _last_scan_enqueued,
        "worker_alive": bool(active) or (heartbeat_age is not None and heartbeat_age < 300),
    }
