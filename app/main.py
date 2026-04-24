import asyncio
import logging
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
):
    results = []
    error = None

    if q.strip():
        try:
            from_ts = _date_to_ts(date_from) if date_from else None
            to_ts = _date_to_ts(date_to, end_of_day=True) if date_to else None
            results = search.search(
                q.strip(),
                limit=40,
                city_filter=city or None,
                date_from=from_ts,
                date_to=to_ts,
            )
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        "_results.html",
        {
            "request": request,
            "results": results,
            "query": q,
            "error": error,
        },
    )


@app.get("/status", response_class=HTMLResponse)
async def status_fragment(request: Request):
    stats = {**models.status(), **indexer.get_status()}
    return templates.TemplateResponse("_status.html", {"request": request, **stats})


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


# ── helpers ───────────────────────────────────────────────────────────────────

def _date_to_ts(date_str: str, end_of_day: bool = False) -> int:
    from datetime import datetime
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if end_of_day:
            dt = dt.replace(hour=23, minute=59, second=59)
        return int(dt.timestamp())
    except ValueError:
        return 0
