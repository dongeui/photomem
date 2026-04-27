## Current handoff snapshot (2026-04-27)

### What this app does now

- `photomem` is an offline screenshot/photo search app.
- Host screenshots are mounted into `/data/photos` through Docker.
- FastAPI serves the UI at `http://localhost:8000`.
- Search is CLIP text-to-image semantic search, not face detection or OCR-only filtering.
- The app keeps a local SQLite index with metadata in `photos` and embeddings in `vec_photos`.
- Thumbnails are cached under `/app/cache/thumbnails` and shown in both search results and gallery.

### Main runtime flow

1. `app/main.py` starts FastAPI, validates the photo mount, initializes SQLite, and loads the CLIP model.
2. `app/indexer.py` starts watchdog plus a periodic scan every `PHOTOMEM_SCAN_INTERVAL` seconds.
3. New or changed files are queued for indexing.
4. Each worker hashes the file, updates DB state, generates a thumbnail, reads EXIF date and GPS, reverse-geocodes location, then writes the CLIP image embedding.
5. `app/search.py` encodes search text with the CLIP text encoder and runs KNN search through `sqlite-vec`.

### Current implemented features

- Multi-threaded indexing via `PHOTOMEM_INDEX_WORKERS`
- Per-worker torch thread tuning via `PHOTOMEM_TORCH_THREADS`
- 5-minute folder rescans for new or changed screenshots
- Cached file stat tracking using `file_size` and `modified_at`
- Thumbnail gallery endpoint at `/gallery`
- OCR-backed text search (Tesseract kor+eng); BM25-normalized score [0.6, 1.0]
- CLIP results scored by L2→cosine distance, capped at 0.65
- `ocr+clip` combined hits score 1.0 (both signals agree)
- Split search UI: gap-based dynamic split (largest score drop wins)
- City autocomplete via `/cities` datalist loaded on page load

### Important files

- `app/main.py`: routes, startup, search result grouping, HTMX fragments, `_gap_split`
- `app/indexer.py`: watcher, queue, workers, rescan loop
- `app/db.py`: schema, migrations, stats, list/search queries, incremental FTS
- `app/models.py`: shared CLIP model loading and text/image encoding
- `app/search.py`: merged OCR + CLIP search with BM25/distance scoring
- `app/ocr.py`: Tesseract wrapper with image prep
- `app/templates/`: HTMX partials and card UI
- `docs/system-structure.svg`: one-page system diagram

### Known limits and next work ideas

- Search relevance is CLIP-only for image content. "Male face" = semantic guess, not face classification.
- No date-range pre-filter in the CLIP KNN step — all candidates fetched then filtered in Python.
- Gallery has no pagination — limited to 120 photos.
- No sort options (by date / by relevance) in the gallery view.
- Exact people/face filtering needs a separate face/person pipeline (detection + tagging/embeddings).

### Local run assumptions

- Docker is the primary run path.
- `.env` is expected to point `PHOTOMEM_HOST_PHOTOS` at the Windows screenshots folder.
- Current compose settings use `PHOTOMEM_INDEX_WORKERS=4` and `PHOTOMEM_SCAN_INTERVAL=300`.


## Project context

This is **photomem** — a private photo service with multilingual NL search (Korean/English/Japanese/Chinese), fully offline, Docker-first.

**Read `DESIGN.md` before writing any code.** The `GSTACK REVIEW REPORT` section at the bottom has the final architecture, 7 confirmed decisions (D2–D7), and 5 must-fix items for Sprint 1.

Key facts:
- Stack: Python + FastAPI + HTMX + ONNX Runtime (CLIP ViT-B/32) + sqlite-vec + SQLite WAL + Docker
- LLM/ollama is NOT Phase 1. Pure CLIP pipeline only.
- ProcessPoolExecutor workers must load ONNX model via `initializer=_worker_init` — not picklable
- Default `PHOTOMEM_MAX_WORKERS=1` for NAS memory constraints
- Distribution: `docker compose up` (primary), brew tap (v1.1+)

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. The
skill has multi-step workflows, checklists, and quality gates that produce better
results than an ad-hoc answer. When in doubt, invoke the skill. A false positive is
cheaper than a false negative.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke /office-hours
- Strategy, scope, "think bigger", "what should we build" → invoke /plan-ceo-review
- Architecture, "does this design make sense" → invoke /plan-eng-review
- Design system, brand, "how should this look" → invoke /design-consultation
- Design review of a plan → invoke /plan-design-review
- Developer experience of a plan → invoke /plan-devex-review
- "Review everything", full review pipeline → invoke /autoplan
- Bugs, errors, "why is this broken", "wtf", "this doesn't work" → invoke /investigate
- Test the site, find bugs, "does this work" → invoke /qa (or /qa-only for report only)
- Code review, check the diff, "look at my changes" → invoke /review
- Visual polish, design audit, "this looks off" → invoke /design-review
- Developer experience audit, try onboarding → invoke /devex-review
- Ship, deploy, create a PR, "send it" → invoke /ship
- Merge + deploy + verify → invoke /land-and-deploy
- Configure deployment → invoke /setup-deploy
- Post-deploy monitoring → invoke /canary
- Update docs after shipping → invoke /document-release
- Weekly retro, "how'd we do" → invoke /retro
- Second opinion, codex review → invoke /codex
- Safety mode, careful mode, lock it down → invoke /careful or /guard
- Restrict edits to a directory → invoke /freeze or /unfreeze
- Upgrade gstack → invoke /gstack-upgrade
- Save progress, "save my work" → invoke /context-save
- Resume, restore, "where was I" → invoke /context-restore
- Security audit, OWASP, "is this secure" → invoke /cso
- Make a PDF, document, publication → invoke /make-pdf
- Launch real browser for QA → invoke /open-gstack-browser
- Import cookies for authenticated testing → invoke /setup-browser-cookies
- Performance regression, page speed, benchmarks → invoke /benchmark
- Review what gstack has learned → invoke /learn
- Tune question sensitivity → invoke /plan-tune
- Code quality dashboard → invoke /health
