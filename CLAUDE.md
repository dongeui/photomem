## Current handoff snapshot (2026-04-27)

### What this app does now

- `photomem` is an offline screenshot/photo search app.
- Host screenshots are mounted into `/data/photos` through Docker.
- FastAPI serves the UI at `http://localhost:8000`.
- Search is now a hybrid of OCR, CLIP semantic search, shadow-document FTS, and lightweight analysis tags.
- The app keeps a local SQLite index with metadata in `photos`, embeddings in `vec_photos`, OCR text in `photo_ocr`, OCR blocks in `photo_ocr_blocks`, short Korean grams in `photo_ocr_grams`, and merged search docs in `photo_search_docs`.
- Thumbnails are cached under `/app/cache/thumbnails` and shown in both search results and gallery.

### Main runtime flow

1. `app/main.py` starts FastAPI, validates the photo mount, initializes SQLite, and loads the CLIP model.
2. `app/indexer.py` starts watchdog plus a periodic scan every `PHOTOMEM_SCAN_INTERVAL` seconds.
3. New or changed files are queued for indexing.
4. Each worker hashes the file, updates DB state, generates a thumbnail, extracts OCR text plus OCR blocks, computes face count and analysis tags, reads EXIF date and GPS, reverse-geocodes location, then writes the CLIP image embedding.
5. `app/search.py` routes intent across `ocr`, `semantic`, or `hybrid`, expands Korean CLIP queries with English variants, then fuses OCR, CLIP, shadow-doc, and analysis ranks with RRF.

### Current implemented features

- Multi-threaded indexing via `PHOTOMEM_INDEX_WORKERS`
- Per-worker torch thread tuning via `PHOTOMEM_TORCH_THREADS`
- 5-minute folder rescans for new or changed screenshots
- Cached file stat tracking using `file_size` and `modified_at`
- Thumbnail gallery endpoint at `/gallery`
- OCR-backed text search with FTS5, substring fallback, and 2-gram support for short Korean UI terms
- OCR engine abstraction: `tesseract` default, optional `easyocr` and `paddleocr`
- OCR block persistence for word/line boxes and confidence
- Korean-to-English query expansion for CLIP through built-in lexicon, with optional `opus` translator path
- CLIP, OCR, shadow-doc, and analysis ranking fused with RRF
- Split search UI: gap-based dynamic split (largest score drop wins)
- City autocomplete via `/cities` datalist loaded on page load
- Result cards include match explanations such as exact OCR hit, semantic expansion, or structural match
- Search mode auto-routing across `hybrid`, `ocr`, and `semantic`

### Important files

- `app/main.py`: routes, startup, search result grouping, HTMX fragments, `_gap_split`
- `app/indexer.py`: watcher, queue, workers, rescan loop
- `app/db.py`: schema, migrations, OCR blocks/grams/search-docs, list/search queries, incremental FTS
- `app/models.py`: shared CLIP model loading and text/image encoding
- `app/search.py`: intent routing, query expansion, RRF fusion, match explanations
- `app/ocr.py`: OCR engine abstraction and structured OCR extraction
- `app/query_translate.py`: Korean CLIP query expansion and optional translator hook
- `app/templates/`: HTMX partials and card UI
- `docs/system-structure.svg`: one-page system diagram
- `docs/search-quality-plan.md`: current search quality roadmap and status

### Known limits and next work ideas

- CLIP image indexing is still monolingual at the embedding level; current Korean support relies on query-time expansion rather than multilingual image re-embedding.
- OCR engine abstraction exists, but production default is still `tesseract` until a benchmark picks a better engine.
- Search cards now explain why they matched, but the "strong match vs candidate" visual split still needs a fuller UX pass.
- No translated image captions yet; shadow documents currently combine OCR and tags only.
- Exact person identity or face embedding search is still out of scope; current face support is count-based.

### Local run assumptions

- Docker is the primary run path.
- `.env` is expected to point `PHOTOMEM_HOST_PHOTOS` at the Windows screenshots folder.
- Current compose settings use `PHOTOMEM_INDEX_WORKERS=4` and `PHOTOMEM_SCAN_INTERVAL=300`.
- OCR benchmarking is done with `scripts/benchmark_ocr.py`.


## Project context

This is **photomem** — a private photo service with multilingual NL search (Korean/English/Japanese/Chinese), fully offline, Docker-first.

**Read `DESIGN.md` before writing any code.** The `GSTACK REVIEW REPORT` section at the bottom has the final architecture, 7 confirmed decisions (D2–D7), and 5 must-fix items for Sprint 1.

Key facts:
- Stack: Python + FastAPI + HTMX + open_clip + sqlite-vec + SQLite WAL + Docker
- Current search stack: OCR FTS + CLIP semantic retrieval + shadow docs + RRF fusion
- Translation path: built-in Korean lexicon by default, optional `opus` local translator
- Current worker model: thread-based CLIP encoding via `PHOTOMEM_INDEX_WORKERS`
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
