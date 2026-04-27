import asyncio
import html as html_module
import logging
import mimetypes
import os
import sys
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db, indexer, models, search, thumbnails

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("photomem")

PHOTOS_DIR = Path(os.environ.get("PHOTOMEM_PHOTOS", "/data/photos"))
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="photomem")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Custom Jinja2 filter: unix timestamp → "YYYY-MM-DD"
def _strftime(ts: int) -> str:
    from datetime import datetime
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except Exception:
        return ""

templates.env.filters["strftime"] = _strftime


def _split_ranked_results(results: list[dict]) -> tuple[list[dict], list[dict]]:
    if not results:
        return [], []

    ranked = [result for result in results if result.get("rank_score") is not None]
    if ranked:
        for result in results:
            result["score"] = max(0, min(100, round(float(result.get("rank_score", 0.0)) * 100)))

        scores = [float(r.get("rank_score", 0.0)) for r in results]
        split_at = _gap_split(scores, max_close=12, min_threshold=0.40)
        return results[:split_at], results[split_at:]

    distances = [float(r.get("distance", 0.0)) for r in results if r.get("distance") is not None]
    if not distances:
        close_count = min(12, max(4, len(results) // 3))
        return results[:close_count], results[close_count:]

    best = min(distances)
    worst = max(distances)
    spread = max(worst - best, 1e-9)

    for result in results:
        distance = float(result.get("distance", worst))
        result["score"] = max(0, min(100, round((1 - ((distance - best) / spread)) * 100)))

    close_count = min(12, max(4, len(results) // 3))
    close_results = results[:close_count]
    other_results = results[close_count:]
    return close_results, other_results


def _gap_split(scores: list[float], max_close: int = 12, min_threshold: float = 0.40) -> int:
    """Return index where results should split into 'close' and 'other'.

    Finds the largest score drop among the first max_close+1 positions.
    Falls back to min_threshold if no significant gap (>0.10) exists.
    Always returns at least 1 and at most max_close.
    """
    cap = min(max_close, len(scores))
    best_gap = 0.0
    best_idx = cap  # default: put everything in close (up to cap)

    for i in range(1, cap + 1):
        prev = scores[i - 1]
        curr = scores[i] if i < len(scores) else 0.0
        gap = prev - curr
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    # Only honour the gap split if it's significant
    if best_gap < 0.10:
        # Fall back to threshold
        best_idx = sum(1 for s in scores if s >= min_threshold)

    return max(1, min(best_idx, max_close))


async def _load_models_and_start_indexer() -> None:
    try:
        await asyncio.to_thread(models.ensure_models)
    except Exception:
        logger.exception("Failed to load CLIP model")
        return

    asyncio.create_task(indexer.run_indexer())
    logger.info("photomem indexer started. Photos: %s", PHOTOS_DIR)


@app.on_event("startup")
async def startup() -> None:
    # Must-fix #4: validate photos mount
    if not PHOTOS_DIR.exists() or not any(PHOTOS_DIR.iterdir()):
        logger.error(
            "\n\n"
            "  ERROR: No photos found at %s\n"
            "  Make sure you've mounted your photo library:\n"
            "    docker run -v /your/photos:/data/photos ...\n"
            "  or in docker-compose.yml:\n"
            "    volumes:\n"
            "      - /your/photos:/data/photos:ro\n",
            PHOTOS_DIR,
        )
        sys.exit(1)

    db.init_db()
    asyncio.create_task(_load_models_and_start_indexer())
    logger.info("photomem started. Photos: %s", PHOTOS_DIR)


# ── pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


# ── HTMX fragments ────────────────────────────────────────────────────────────

@app.get("/search", response_class=HTMLResponse)
async def search_photos(
    request: Request,
    q: str = Query(""),
    city: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    mode: str = Query("hybrid"),
):
    if not q.strip():
        return await gallery_fragment(
            request,
            limit=60,
            page=1,
            sort="recent",
            filter_tag="all",
        )

    results = []
    error = None
    search_mode = mode if mode in {"hybrid", "ocr", "semantic"} else "hybrid"
    search_meta = {"effective_mode": search_mode, "intent_reason": "manual"}

    try:
        from_ts = _date_to_ts(date_from) if date_from else None
        to_ts = _date_to_ts(date_to, end_of_day=True) if date_to else None
        results, search_meta = search.search_with_meta(
            q.strip(),
            limit=40,
            city_filter=city or None,
            date_from=from_ts,
            date_to=to_ts,
            mode=search_mode,
        )
    except Exception as exc:
        error = str(exc)

    close_results, other_results = _split_ranked_results(results)

    return templates.TemplateResponse(
        "_results.html",
        {
            "request": request,
            "results": results,
            "close_results": close_results,
            "other_results": other_results,
            "query": q,
            "error": error,
            "mode": search_mode,
            "effective_mode": search_meta["effective_mode"],
            "intent_reason": search_meta["intent_reason"],
        },
    )


@app.get("/status", response_class=HTMLResponse)
async def status_fragment(request: Request):
    stats = {**models.status(), **indexer.get_status()}
    return templates.TemplateResponse("_status.html", {"request": request, **stats})


@app.get("/gallery", response_class=HTMLResponse)
async def gallery_fragment(
    request: Request,
    limit: int = Query(60, ge=12, le=120),
    page: int = Query(1, ge=1),
    sort: str = Query("recent"),
    filter_tag: str = Query("all"),
):
    sort_value = sort if sort in {"recent", "oldest", "faces", "text"} else "recent"
    filter_value = filter_tag if filter_tag in {"all", "faces", "text", "documents", "screens"} else "all"
    offset = (page - 1) * limit
    conn = db.get_connection()
    try:
        total = db.count_photos(conn, filter_tag=filter_value)
        photos = db.list_photos(
            conn,
            limit=limit,
            offset=offset,
            sort=sort_value,
            filter_tag=filter_value,
        )
    finally:
        conn.close()

    return templates.TemplateResponse(
        "_gallery.html",
        {
            "request": request,
            "photos": photos,
            "page": page,
            "limit": limit,
            "sort": sort_value,
            "filter_tag": filter_value,
            "total": total,
            "has_prev": page > 1,
            "has_next": offset + len(photos) < total,
        },
    )


# ── thumbnails ────────────────────────────────────────────────────────────────

@app.get("/thumb/{photo_id}")
async def thumbnail(photo_id: int):
    path = thumbnails.thumb_path(photo_id)
    if not path.exists():
        # Try to generate on demand (e.g. after a crash)
        conn = db.get_connection()
        try:
            cur = conn.execute("SELECT file_path FROM photos WHERE id=?", (photo_id,))
            row = cur.fetchone()
            if row:
                thumbnails.generate_thumbnail(photo_id, row["file_path"])
        finally:
            conn.close()

    if path.exists():
        return FileResponse(str(path), media_type="image/jpeg")

    # Generate a 1x1 grey placeholder with Pillow
    import io
    from PIL import Image as PilImage
    buf = io.BytesIO()
    PilImage.new("RGB", (1, 1), color=(40, 40, 40)).save(buf, "JPEG")
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.get("/photo/{photo_id}")
async def original_photo(photo_id: int):
    conn = db.get_connection()
    try:
        cur = conn.execute("SELECT file_path FROM photos WHERE id=?", (photo_id,))
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return Response(status_code=404)

    path = Path(row["file_path"])
    if not path.exists():
        return Response(status_code=404)

    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return FileResponse(str(path), media_type=media_type, filename=path.name)


# ── data endpoints ────────────────────────────────────────────────────────────

@app.get("/cities")
async def city_list():
    conn = db.get_connection()
    try:
        cur = conn.execute(
            "SELECT DISTINCT city FROM photos WHERE city IS NOT NULL AND city != '' ORDER BY city"
        )
        cities = [row["city"] for row in cur.fetchall()]
    finally:
        conn.close()
    options = "".join(f'<option value="{html_module.escape(c, quote=True)}">' for c in cities)
    return HTMLResponse(options)


# ── helpers ───────────────────────────────────────────────────────────────────

def _date_to_ts(date_str: str, end_of_day: bool = False) -> int | None:
    from datetime import datetime
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return int(dt.timestamp())
    except ValueError:
        return None
