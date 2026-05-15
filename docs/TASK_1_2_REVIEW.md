# Task 1.2 Review Report — PNG/PDF Fallback Vectorizer

Date: 2026-05-15
Status: **PASS** ✅

## Artifacts Produced

- `src/parser/png_vectorizer.py` — PNGVectorizer + PDFVectorizer (15KB, ~400 lines)
- `tests/test_png_vectorizer.py` — Unit tests (18 cases)

## Test Results

```
18 passed in 0.53s
- Config: 2/2
- Vectorizer init: 2/2
- Nearest standard sizes: 2/2
- Wall detection (BGR fix): 3/3
- Opening detection: 3/3
- Column detection: 2/2
- Room extraction: 1/1
- Skeletonization: 2/2
- Output schema compatibility: 1/1
```

## Implementation Summary

### Core Features
- `PNGVectorizer`: PNG → wall/door/window/column detection via OpenCV
  - Adaptive threshold + morphological closing for wall detection
  - Skeletonization (Zhang-Suen) for line extraction
  - Rectangular contour analysis for openings
  - MinAreaRect for rotated bounding boxes (door/window orientation)
- `PDFVectorizer`: PDF → PNG (poppler/pdftoppm) → PNGVectorizer
- Output format matches `DXFParseResult` schema (pipeline compatibility)
- Bounding box in mm via `pixels_per_mm` ratio (96 DPI)
- Room extraction via shapely (reuses DXF parser logic)

### Key Fixes Applied
- BGR/RGB image handling (OpenCV loads as BGR, tests create BGR explicitly)
- Window standard nearest logic corrected

## Sign-off

Ready for next task: **YES** (Task 1.3 — 2.5D ISO Renderer)

---

## Task 1.3 Plan

**Input:** `DXFParseResult` (from Task 1.1 or 1.2) — wall graph, rooms, columns, openings
**Output:** `base_render.png` + `depth_map.png` + `normal_map.png` (16-bit)

**Approach:**
1. Compute isometric projection matrix (30° standard ISO)
2. Extrude walls to fixed height (default 3000mm)
3. Render floor polygon (light gray), walls (medium gray), columns (dark gray)
4. Orthographic camera (no perspective)
5. Generate depth map: distance from camera plane per pixel (16-bit)
6. Generate normal map: surface normals per pixel (RGB encoded)

**Renderer:** Pure Python (PIL + numpy), no GPU needed for this stage

**Est. time: 2 days**