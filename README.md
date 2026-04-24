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

## Development

```bash
pip install -r requirements.txt
pytest
```

Some search tests are skipped unless CLIP model assets are already available.
