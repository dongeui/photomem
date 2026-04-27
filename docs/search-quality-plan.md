# Photomem Search Quality Plan

Updated: 2026-04-27

## Goal

Make Photomem feel like a reliable local memory search tool, not just a visual gallery. The system should handle Korean natural-language queries, exact UI text, screenshot content, people/face queries, and document-like screens with clearer ranking and clearer result explanations.

## Current Diagnosis

Photomem already has the right base pieces:

- OCR text extraction with Tesseract
- SQLite FTS5 and substring fallback for OCR search
- CLIP image embeddings through `open_clip`
- Face count metadata through OpenCV
- Basic screenshot/document/text-heavy analysis tags
- Intent routing across `hybrid`, `ocr`, and `semantic`

The remaining quality problem is not one single model. It is the ranking architecture:

- Korean CLIP queries such as `자전거` are weak because the current CLIP text encoder is English-centric.
- Short Korean UI text such as `동의` depends on OCR quality and exact text ranking.
- OCR and CLIP scores are different kinds of scores, so direct score mixing can let one channel dominate.
- Cards do not yet explain strongly enough why a result matched.

## Implementation Priorities

### 1. RRF Fusion

Replace direct score mixing with Reciprocal Rank Fusion (RRF).

Channels:

- OCR/FTS result rank
- CLIP semantic result rank
- Lightweight analysis/tag rank

Intent weights:

- Text/code intent: OCR high, CLIP low, tags medium
- Face/person intent: CLIP high, face tags medium, OCR low
- Mixed intent: OCR and CLIP both active

Expected result:

- `동의` should not be drowned out by semantic CLIP noise.
- `남자 얼굴` should favor visual/face results instead of screenshots containing the typed query.
- Mixed queries can still benefit from both channels.

### 2. Korean Query Translation for CLIP

Add a query-time Korean-to-English expansion path before CLIP encoding.

Candidate model:

- `Helsinki-NLP/opus-mt-ko-en`, CPU/offline

Implementation shape:

- Keep existing image embeddings.
- Translate only the search query.
- Run CLIP for original query and translated query.
- Fuse semantic candidates with RRF.

This does not require DB re-indexing.

Status: implemented as query-time expansion. The default path uses a small offline lexicon so Docker remains stable; `PHOTOMEM_TRANSLATOR=opus` enables the optional MarianMT path when dependencies/model cache are installed.

### 3. OCR Engine Benchmark and Abstraction

Do not replace Tesseract blindly. Add an OCR engine abstraction, then benchmark:

- Current Tesseract
- EasyOCR
- PaddleOCR

Benchmark target:

- 30-50 representative screenshots
- Korean UI text
- Small fonts
- Dark mode
- Mixed Korean/English/code text

Likely next engine to test first: PaddleOCR, because the project is screenshot-heavy and Korean UI-heavy.

Status: implemented as an OCR engine abstraction with Tesseract as the stable default, optional EasyOCR/PaddleOCR adapters, and `scripts/benchmark_ocr.py` for sample-folder comparison.

### 4. OCR Line/Word Storage

Extend OCR storage beyond a single text blob.

Store:

- Line text
- Word text
- Confidence
- Bounding box
- OCR engine/version

This enables:

- Ranking by confidence and text size/location
- Highlighting matched text on the original image
- Better result explanations

Status: implemented for Tesseract word boxes and optional engine line boxes. Stored in `photo_ocr_blocks`.

### 5. Short Korean Text Index

Add a 2-gram/materialized token table for short Korean UI terms.

Important terms:

- `동의`
- `확인`
- `취소`
- `전송`
- `오류`
- `실패`

FTS5 is useful, but short Korean UI strings need a dedicated fallback index.

Status: implemented as a materialized 2-gram table, `photo_ocr_grams`.

### 6. Shadow Document

Create one generated search document per image:

```text
ocr: 동의 확인 전송 실패
tags: screen document text-heavy face-0
caption: error dialog login screen
objects: button form dialog
```

Index it separately so text search can use OCR, tags, and future captions together.

Status: implemented as `photo_search_docs` FTS5, refreshed when OCR, face, or analysis metadata changes.

### 7. Result UX Trust

Cards should explain the strongest matching reason:

- `OCR: "동의" exact match`
- `Intent: text`
- `Face: 1 detected`
- `Structure: document screen`
- `Semantic: CLIP match`

The UI should distinguish strong matches from weak candidates more visually.

Status: partially implemented. Cards now show a match reason line. Strong/weak section styling still needs a fuller UX pass.

## Near-Term Development Order

1. Implement RRF fusion.
2. Add Korean-to-English CLIP query expansion.
3. Add OCR engine interface and benchmark command.
4. Extend OCR schema for line/word boxes.
5. Add short Korean 2-gram index.
6. Add shadow documents.
7. Improve result card explanation and filter-state UX.

## Deferred

- Full multilingual CLIP replacement: useful, but requires full image re-embedding.
- BLIP caption indexing: useful, but CPU-heavy and lower priority than OCR/search fusion.
- Heavy rerankers or ColPali-style document retrieval: promising, but should come after the simpler channels are reliable.
