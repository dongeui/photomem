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
from pathlib import Path

import piexif
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app import db, geocoder, models, thumbnails

logger = logging.getLogger("photomem.indexer")

PHOTOS_DIR = Path(os.environ.get("PHOTOMEM_PHOTOS", "/data/photos"))
SCAN_INTERVAL = int(os.environ.get("PHOTOMEM_SCAN_INTERVAL", "300"))  # seconds

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".tif"}

# Shared state. These are mutated only from the main asyncio loop.
_queue: asyncio.Queue[str] = asyncio.Queue()
_queued_paths: set[str] = set()
_running = False
_last_heartbeat = 0.0
_current_path: str | None = None
_current_started_at: float | None = None
_last_completed_at: float | None = None


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


async def _process_path(conn, path: str) -> None:
    global _current_path, _current_started_at, _last_completed_at, _last_heartbeat

    if not os.path.isfile(path):
        return

    try:
        file_hash = _file_hash(path)
    except OSError:
        return

    photo_id = db.upsert_photo(conn, path, file_hash)
    if photo_id is None:
        return

    thumbnails.generate_thumbnail(photo_id, path)
    created_at, lat, lon = _parse_exif(path)

    city = country = None
    if lat is not None and lon is not None:
        city, country = geocoder.reverse_geocode(lat, lon)

    _current_path = path
    _current_started_at = time.time()
    try:
        embedding = await asyncio.to_thread(models.encode_image, path)
    except Exception as exc:
        db.mark_photo_error(conn, photo_id, str(exc))
        logger.warning("CLIP failed for %s: %s", path, exc)
        return
    finally:
        _current_path = None
        _current_started_at = None

    db.update_photo_indexed(conn, photo_id, created_at, lat, lon, city, country, embedding)
    _last_completed_at = time.time()
    _last_heartbeat = _last_completed_at


async def _scan_directory(conn) -> int:
    """Scan PHOTOS_DIR and enqueue unindexed files. Returns count enqueued."""
    count = 0
    for root, _dirs, files in os.walk(str(PHOTOS_DIR)):
        for fname in files:
            if _enqueue_path_nowait(os.path.join(root, fname)):
                count += 1
    return count


async def run_indexer() -> None:
    global _running, _last_heartbeat
    _running = True
    _last_heartbeat = time.time()

    loop = asyncio.get_running_loop()
    conn = db.get_connection()

    pending_count = db.requeue_pending(conn)
    for path in db.get_pending_paths(conn):
        _enqueue_path_nowait(path)
    if pending_count:
        logger.info("Re-queued %d pending photos from previous run", pending_count)

    observer = Observer()
    observer.schedule(_PhotoHandler(loop), str(PHOTOS_DIR), recursive=True)
    observer.start()

    await _scan_directory(conn)

    async def _periodic_scan():
        while _running:
            await asyncio.sleep(SCAN_INTERVAL)
            await _scan_directory(conn)

    asyncio.create_task(_periodic_scan())

    try:
        while _running:
            try:
                path = await asyncio.wait_for(_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            _queued_paths.discard(path)
            _last_heartbeat = time.time()
            await _process_path(conn, path)
            _queue.task_done()
    finally:
        observer.stop()
        observer.join()
        conn.close()


def get_status() -> dict:
    conn = db.get_connection()
    stats = db.get_stats(conn)
    conn.close()

    now = time.time()
    heartbeat_age = now - _last_heartbeat if _last_heartbeat else None
    current_elapsed = now - _current_started_at if _current_started_at else None

    return {
        **stats,
        "running": _running,
        "queue_size": _queue.qsize(),
        "current_file": Path(_current_path).name if _current_path else None,
        "current_elapsed": int(current_elapsed) if current_elapsed is not None else None,
        "last_completed_at": int(_last_completed_at) if _last_completed_at else None,
        "worker_alive": bool(_current_path) or (heartbeat_age is not None and heartbeat_age < 300),
    }
