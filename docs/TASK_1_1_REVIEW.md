# Task 1.1 Review Report — DXF/DWG Parser + Layer Normalization

Date: 2026-05-15
Status: **PASS** ✅

## Artifacts Produced

- `src/parser/dxf_parser.py` — DXF/DWG parser (22KB, ~550 lines)
- `tests/test_dxf_parser.py` — Unit tests (22 cases)
- `docs/DEVELOPMENT_PLAN.md` — Project plan

## Test Results

```
22 passed in 0.27s
- Layer normalization: 5/5
- ParseResult dataclasses: 5/5
- Wall geometry: 3/3
- Room extraction: 3/3
- Opening estimation: 2/2
- Column detection: 2/2
- Bounding box: 1/1
- JSON export/load: 1/1
```

## Implementation Summary

### Core Features
- `DXFParser` class — reads DXF/DWG via ezdxf, extracts all structural entities
- Layer normalization: 20+ common AutoCAD layer patterns → WALL/DOOR/WINDOW/COLUMN/FLOOR/HATCH/DIMENSION/TEXT/UNKNOWN
- Entity parsing: LINE, LWPOLYLINE, POLYLINE, ARC, CIRCLE, SPLINE, INSERT (block refs)
- Room extraction via shapely `linemerge` + `polygonize` — closed polygons from wall segments
- Opening detection: INSERT block refs → DOOR/WINDOW with standard size estimation
- Column detection: CIRCLE → circle columns, LWPOLYLINE → rectangular columns, INSERT → block-based columns
- Bounding box calculation from all parsed points
- JSON export: full parse result + GeoJSON-style geometry export

### Key Fixes Applied
- `ezdxf.math.Vector` → `Vec3` (API change in newer ezdxf versions)

### File Structure
```
src/parser/
├── __init__.py
└── dxf_parser.py     # DXFParser, DXFParseResult, dataclasses

tests/
└── test_dxf_parser.py  # 22 unit tests
```

## Sign-off

Ready for next task: **YES** (Task 1.2 — PNG/PDF Fallback Vectorizer)

---

## Task 1.2 Plan

### Input: PNG/PDF raster image of floor plan
### Output: Vector geometry (wall graph, room polygons) as JSON

**Approach:**
1. **OpenCV** — grayscale threshold, contour detection
2. **Potrace** approach — bitmap → polygon paths via skimage + tracing
3. Convert raster contours → wall polygons, room outlines
4. Merge into structured JSON (same format as Task 1.1 output)

**Steps:**
1. `cv2.imread` → grayscale → adaptive threshold
2. `cv2.findContours` → external contours (wall outlines)
3. Polygon approximation via `cv2.approxPolyDP`
4. Classify by area/aspect ratio → rooms vs walls
5. Corridor detection (narrow connected regions)
6. Export same `DXFParseResult` JSON format for pipeline consistency

**Est. time: 1 day**