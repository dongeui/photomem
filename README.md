# photomem

Private offline photo search with FastAPI, HTMX, SQLite, sqlite-vec, and CLIP.

## Run with Docker

1. Update the photo-library mount in `docker-compose.yml`.
2. Start the app:

```bash
docker compose up --build
```

3. Open `http://localhost:8000`.

The first run downloads the CLIP model and stores generated data in the `photomem_cache` Docker volume.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `PHOTOMEM_PHOTOS` | `/data/photos` | Mounted photo library path inside the container |
| `PHOTOMEM_DB` | `/app/cache/photomem.db` | SQLite database path |
| `PHOTOMEM_CACHE` | `/app/cache` | Cache directory for thumbnails and generated files |
| `PHOTOMEM_INDEX_WORKERS` | `1` | Concurrent CLIP image encoders |
| `PHOTOMEM_TORCH_THREADS` | auto | CPU threads per encoder; total CPU use is roughly workers x threads |
| `PHOTOMEM_SCAN_INTERVAL` | `300` | Seconds between folder scans for new or changed photos |
| `PHOTOMEM_OCR_ENGINE` | `tesseract` | OCR engine adapter: `tesseract`, optional `easyocr`, optional `paddleocr` |
| `PHOTOMEM_TRANSLATOR` | `lexicon` | CLIP query expansion: built-in Korean lexicon or optional `opus` MarianMT |

## Development

```bash
pip install -r requirements.txt
pytest
```

Some search tests are skipped unless CLIP model assets are already available.

## Search Quality Roadmap

The current search quality plan is tracked in [docs/search-quality-plan.md](docs/search-quality-plan.md).

## OCR Benchmark

Compare OCR engines on the same screenshot sample folder:

```bash
python scripts/benchmark_ocr.py C:\path\to\samples
```
