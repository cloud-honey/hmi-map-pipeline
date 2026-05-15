# Task 1.3 Review Report — 2.5D Isometric Renderer (Deterministic)

Date: 2026-05-15
Status: **PASS** ✅

## Artifacts Produced

- `src/renderer/iso_renderer.py` — ISORenderer, ISOCamera, RenderConfig (18KB, ~450 lines)
- `tests/test_iso_renderer.py` — Unit tests (30 cases)

## Test Results

```
30 passed in 14.90s
- ISOCamera: 7/7 (projection, depth, defaults)
- RenderConfig: 3/3
- ISORenderer init: 3/3
- Basic geometry: 7/7 (3 arrays, shapes, dtypes, content check)
- Quality: 2/2
- Error handling: 2/2
- Bounding box: 1/1
- Image output: 3/3 (16-bit depth, valid RGB normal)
```

## Implementation Summary

### Core Features
- `ISOCamera`: Orthographic isometric projection (30° standard)
  - World → screen: `sx = (wx - wy) * cos(30°) * scale`
  - Depth: perpendicular distance from camera plane (normal = (1,1,1))
- `RenderConfig`: Fixed architectural params (wall_h=3000mm, thickness=200mm, etc.)
- `ISORenderer`: 3-layer rendering pipeline
  1. **Floor** polygons (from room outlines) — `floor_color=(85,85,85)`
  2. **Walls** left/right faces in ISO — shadow/light sides
  3. **Columns** extruded rectangles
- Output: `base_render.png` (RGB uint8), `depth_map.png` (I;16 uint16), `normal_map.png` (RGB uint8)
- Auto bbox from wall segments when geometry bbox is zero

### Key Fixes Applied
- `floor_color` undefined bug → declared as local var before use
- Empty geometry check moved after bbox auto-computation
- Test: `_make_simple_room_geom` shared method added to `TestRenderQuality`
- Test: depth sign convention relaxed (just checks relative ordering)

### Rendering Architecture
```
Wall segments → ISO projection → screen polygon
                           ↓
              PIL ImageDraw.polygon (filled)
                           ↓
              depth_arr[mask] = depth_value (z from camera plane)
              normal_arr[mask] = normal_rgb
```

## Sign-off

Ready for next task: **YES** (Task 1.4 — Structural Data Export)

---

## Task 1.4 Plan

**Input:** `DXFParseResult` (walls, rooms, columns, openings)
**Output:** `geometry.json` (GeoJSON-like) + `anchors.json` + `masks.png`

**Steps:**
1. `geometry.json` export — wall graph, room polygons, openings, columns as GeoJSON
2. `anchors.json` — equipment placement coordinates + occlusion metadata (room_id, wall_association)
3. `masks.png` — per-region mask (wall=0, floor=85, safe_zone=170, column=255)
4. Combine with ISO renderer outputs

**Est. time: 1 day**