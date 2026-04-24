"""
Background indexer: watches /data/photos, encodes photos with CLIP, writes to SQLite.

Process model:
  - Main asyncio process: queue management, DB writes, watchdog observer
  - ProcessPoolExecutor workers: CPU-intensive CLIP inference (ONNX)
    Workers load the visual ONNX model via worker_init() — NOT passed from main process
    (ONNX sessions are not picklable; passing them silently breaks inference).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import piexif
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app import db, geocoder, models, thumbnails

if TYPE_CHECKING:
    pass

logger = logging.getLogger("photomem.indexer")

PHOTOS_DIR = Path(os.environ.get("PHOTOMEM_PHOTOS", "/data/photos"))
MAX_WORKERS = int(os.environ.get("PHOTOMEM_MAX_WORKERS", "1"))  # thread pool size
SCAN_INTERVAL = int(os.environ.get("PHOTOMEM_SCAN_INTERVAL", "300"))  # seconds
BATCH_SIZE = 8

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".tif"}

# Shared state (written only from the main asyncio loop)
_queue: asyncio.Queue[str] = asyncio.Queue()
_running = False
_last_heartbeat = 0.0


# ── file hash ─────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── EXIF extraction ───────────────────────────────────────────────────────────

def _parse_exif(path: str) -> tuple[int | None, float | None, float | None]:
    """Return (unix_timestamp, lat, lon). Falls back to mtime if no EXIF date."""
    lat = lon = None
    created_at = None
    try:
        exif = piexif.load(path)
        # Date
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

        # GPS
        gps = exif.get("GPS", {})
        if gps:
            def _rational(v):
                return v[0][0] / v[0][1] + v[1][0] / (v[1][1] * 60) + v[2][0] / (v[2][1] * 3600)
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


# ── watchdog ──────────────────────────────────────────────────────────────────

class _PhotoHandler(FileSystemEventHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        self._queue = queue
        self._loop = loop

    def _enqueue(self, path: str) -> None:
        if Path(path).suffix.lower() in SUPPORTED_EXT:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, path)

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_moved(self, event) -> None:
        if not event.is_directory:
            self._enqueue(event.dest_path)


# ── CLIP inference (runs in worker process) ───────────────────────────────────

def _encode_photo(path: str) -> bytes | None:
    """Called in ProcessPoolExecutor worker. worker_init() must have loaded the session."""
    try:
        return models.encode_image(path)
    except Exception as exc:
        raise RuntimeError(f"{path}: {exc}") from exc


# ── indexing loop ─────────────────────────────────────────────────────────────

async def _process_path(conn, loop: asyncio.AbstractEventLoop, path: str) -> None:
    global _last_heartbeat
    if not os.path.isfile(path):
        return

    try:
        file_hash = _file_hash(path)
    except OSError:
        return

    photo_id = db.upsert_photo(conn, path, file_hash)
    if photo_id is None:
        return  # already indexed, hash unchanged

    # Thumbnail (fast, in main process)
    thumbnails.generate_thumbnail(photo_id, path)

    # EXIF
    created_at, lat, lon = _parse_exif(path)

    # Geocode
    city = country = None
    if lat is not None and lon is not None:
        city, country = geocoder.reverse_geocode(lat, lon)

    # CLIP encoding — runs in thread pool to keep event loop responsive
    try:
        embedding: bytes = await asyncio.to_thread(models.encode_image, path)
    except Exception as exc:
        db.mark_photo_error(conn, photo_id, str(exc))
        logger.warning("CLIP failed for %s: %s", path, exc)
        return

    db.update_photo_indexed(conn, photo_id, created_at, lat, lon, city, country, embedding)
    _last_heartbeat = time.time()


async def _scan_directory(conn) -> int:
    """Scan PHOTOS_DIR and enqueue unindexed files. Returns count enqueued."""
    count = 0
    for root, _dirs, files in os.walk(str(PHOTOS_DIR)):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXT:
                await _queue.put(os.path.join(root, fname))
                count += 1
    return count


async def run_indexer() -> None:
    global _running, _last_heartbeat
    _running = True
    _last_heartbeat = time.time()

    conn = db.get_connection()

    # Re-queue pending rows from before any container restart
    pending_count = db.requeue_pending(conn)
    for path in db.get_pending_paths(conn):
        await _queue.put(path)
    if pending_count:
        logger.info("Re-queued %d pending photos from previous run", pending_count)

    # Start watchdog
    observer = Observer()
    observer.schedule(_PhotoHandler(_queue, loop), str(PHOTOS_DIR), recursive=True)
    observer.start()

    # Initial scan
    await _scan_directory(conn)

    # Periodic re-scan fallback (catches missed events on NAS)
    async def _periodic_scan():
        while _running:
            await asyncio.sleep(SCAN_INTERVAL)
            await _scan_directory(conn)

    asyncio.create_task(_periodic_scan())

    # Main processing loop
    try:
        while _running:
            try:
                path = await asyncio.wait_for(_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            _last_heartbeat = time.time()  # alive as long as queue is draining
            await _process_path(conn, loop, path)
            _queue.task_done()
    finally:
        observer.stop()
        observer.join()
        conn.close()


def get_status() -> dict:
    conn = db.get_connection()
    stats = db.get_stats(conn)
    conn.close()
    heartbeat_age = time.time() - _last_heartbeat if _last_heartbeat else None
    return {
        **stats,
        "running": _running,
        "queue_size": _queue.qsize(),
        "worker_alive": heartbeat_age is not None and heartbeat_age < 300,
    }
