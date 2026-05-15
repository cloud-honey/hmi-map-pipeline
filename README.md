# HMI Map Automation Pipeline

AI-based HMI background map generator from CAD floor plans.

**Input:** DXF/DWG/PNG/PDF floor plan → **Output:** 3D Isometric HMI background map

---

## Overview

```
DXF/DWG/PNG/PDF
    │
    ▼ [Parser] ← DXFParser / PNGVectorizer
    DXFParseResult (wall graph, rooms, columns, openings)
    │
    ▼ [Renderer] ← ISORenderer (deterministic, no AI)
    base_render.png + depth_map.png + normal_map.png
    │
    ▼ [AI Refinement] ← AIRefiner (SDXL + ControlNet)
    ai_enhanced.png (optional)
    │
    ▼ [QA + Output]
    background.png, transparent_background.png, masks.png,
    anchors.json, metadata.json, qa_report.json
```

**Core Principle:** "Structure integrity via engineering, visual completeness via AI."

- Structure (walls, rooms, columns) → Deterministically rendered (no AI hallucination)
- Visual quality (textures, lighting) → AI-enhanced with low denoise (0.25–0.35)
- QA fallback → If AI fails, deterministic render is used

---

## Installation

### Requirements

- Python 3.12+
- PyTorch 2.x
- Key packages: `ezdxf`, `shapely`, `opencv-python-headless`, `Pillow`, `numpy`, `scipy`

```bash
# Install dependencies
pip install ezdxf shapely opencv-python-headless Pillow numpy scipy

# Optional: AI refinement (requires ~15GB disk space)
pip install diffusers transformers accelerate huggingface_hub

# Install ComfyUI (for GPU-accelerated AI)
python src/pipeline/install_comfyui.py --install --verify
```

---

## Quick Start

### 1. Create config file

```json
{
  "input": { "path": "samples/floorplan.dxf" },
  "output": { "directory": "./output" },
  "renderer": { "wall_height_mm": 3000 },
  "ai_refinement": { "enabled": false },
  "qa": { "enabled": true }
}
```

### 2. Run pipeline

```bash
# Deterministic only (no AI, CPU-capable)
python src/pipeline/main_pipeline.py --config config/pipeline.json

# With AI refinement (GPU required, ~8GB VRAM)
python src/pipeline/main_pipeline.py --config config/pipeline-full.json
```

### 3. Output

```
output/
├── background.png              # Final HMI background map
├── transparent_background.png   # RGBA version
├── masks.png                   # Region segmentation
├── anchors.json                # Equipment placement coords
├── metadata.json               # Parse stats + hashes
├── qa_report.json              # QA results
├── base_render.png             # Deterministic ISO render
├── geometry.json               # GeoJSON wall/room data
└── ai_enhanced.png            # AI-enhanced version (if enabled)
```

---

## Config Schema

See `config/config.schema.json` for full schema.

| Section | Key Parameters |
|---------|---------------|
| `input.path` | Path to DXF/DWG/PNG/PDF |
| `renderer.wall_height_mm` | Wall height (default: 3000) |
| `renderer.camera_pixels_per_mm` | ISO scale (default: 0.5) |
| `ai_refinement.enabled` | Enable AI (default: true) |
| `ai_refinement.denoise_strength` | Low = preserve structure (0.25–0.35) |
| `ai_refinement.tile_size` | Tiled rendering size (default: 512) |
| `qa.threshold_alignment` | Structure match threshold (default: 0.75) |

---

## Project Structure

```
hmi-map-pipeline/
├── config/
│   ├── config.schema.json      # JSON schema for config validation
│   └── pipeline.example.json   # Example config
├── docs/
│   ├── DEVELOPMENT_PLAN.md      # Full development plan
│   ├── TASK_1_1_REVIEW.md       # Task review reports
│   ├── TASK_1_2_REVIEW.md
│   ├── TASK_1_3_REVIEW.md
│   ├── TASK_1_4_REVIEW.md
│   ├── TASK_2_1_REVIEW.md
│   └── TASK_3_1_3_2_REVIEW.md
├── samples/                     # Sample input files
├── src/
│   ├── parser/
│   │   ├── dxf_parser.py        # DXF/DWG parser (ezdxf)
│   │   ├── png_vectorizer.py    # PNG/PDF raster vectorizer
│   │   └── structural_exporter.py # geometry/anchors/masks export
│   ├── renderer/
│   │   └── iso_renderer.py       # 2.5D Isometric renderer
│   └── pipeline/
│       ├── install_comfyui.py   # ComfyUI installer
│       ├── ai_refiner.py         # SDXL + ControlNet pipeline
│       ├── output_qa.py          # QA + output package generator
│       └── main_pipeline.py      # Main orchestrator
├── tests/
│   ├── test_dxf_parser.py        # 22 tests
│   ├── test_png_vectorizer.py    # 18 tests
│   ├── test_iso_renderer.py       # 30 tests
│   ├── test_structural_exporter.py # 27 tests
│   ├── test_ai_refiner.py        # 26 tests
│   └── test_output_qa.py          # 20 tests
└── README.md
```

**Total: 143 unit tests**

---

## Running Tests

```bash
cd hmi-map-pipeline
python -m pytest tests/ -v
```

---

## Development

### Task Progress

| Task | Status | Tests |
|------|--------|-------|
| 1.1 DXF/DWG Parser | ✅ DONE | 22 |
| 1.2 PNG/PDF Vectorizer | ✅ DONE | 18 |
| 1.3 2.5D ISO Renderer | ✅ DONE | 30 |
| 1.4 Structural Data Export | ✅ DONE | 27 |
| 2.1 ComfyUI + AI Refiner | ✅ DONE | 26 |
| 3.1 Output Package Generator | ✅ DONE | 20 |
| 3.2 Auto QA System | ✅ DONE | 20 |
| 4.1 Config-driven Pipeline | ✅ DONE | — (integration) |
| 4.2 README + Integration | ✅ DONE | 143 total |

---

## Notes

- **DXF Parser:** Requires `ezdxf` — parses LINE, LWPOLYLINE, ARC, CIRCLE, SPLINE, INSERT
- **PNG Vectorizer:** Requires `opencv-python-headless` — raster → vector via contour detection
- **AI Refinement:** Requires GPU with 8GB+ VRAM. Falls back to deterministic render if unavailable.
- **ComfyUI:** Alternative to native diffusers. Run `install_comfyui.py --install` on target PC.
- **VRAM Constraint:** 4GB XPS dev machine cannot run SDXL; 8GB target PC required.