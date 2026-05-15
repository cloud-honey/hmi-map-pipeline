# HMI Map Pipeline — Development Plan
Project: AI-based HMI Map Automation Pipeline (V3 Spec)
Target: 8GB RAM PC (GTX 1070 or better expected), XPS as dev controller
Created: 2026-05-15

---

## 0. Project Overview

**Goal:** CAD (DXF/DWG) → 3D Isometric HMI background maps via deterministic geometry + AI refinement.

**Core Principle:** "Structure integrity via engineering, visual completeness via AI."
- Structure (walls, rooms, columns) → Deterministically rendered (no AI hallucination)
- Visual quality (textures, lighting, mood) → AI-enhanced with low denoise strength

---

## 1. System Architecture

```
INPUT: DXF/DWG/PNG/PDF
  │
  ▼
[ Stage 1 ] Parser & Geometry Engine
  ├── DXF/DWG layer extraction (ezdxf)
  ├── Vectorization (PNG/PDF fallback)
  ├── Wall graph / room polygon / opening mask / column mask
  └── 2.5D Orthographic ISO renderer → base_render.png + depth/normal maps
  │
  ▼
[ Stage 2 ] AI Style Refinement
  ├── ComfyUI API or python diffusers pipeline
  ├── SDXL Base (quantized FP8 for 8GB VRAM)
  ├── Multi-ControlNet: Canny + Depth + Lineart (mask-separated)
  ├── Tiled rendering (4096x4096+ support)
  ├── LoRA: texture unification only (Clay Render style)
  └── Low denoise (0.25–0.35) — structural integrity preserved
  │
  ▼
[ Stage 3 ] Output & QA
  ├── background.png / transparent_background.png
  ├── masks.png (wall/floor/safe-zone)
  ├── anchors.json (equipment placement coords + occlusion metadata)
  ├── Auto QA: structure alignment, color rule check, artifact detection
  └── Fallback: save deterministic render if QA fails
  │
  ▼
[ Stage 4 ] Operations
  ├── External config.json injection
  ├── ComfyUI Workflow export
  ├── Webhook/polling-based external integration
  └── Metadata logging (checkpoint hash, seed, sampler, etc.)
```

---

## 2. Task Breakdown (8 Tasks → 4 Stages)

### Stage 1: Parser & Geometry Engine

**Task 1.1 — DXF/DWG Parser + Layer Normalization**
- Parse DXF/DWG via ezdxf
- Layer mapping: Wall, Door, Window, Column, Hatch, Dimension
- Entity extraction: LWPOLYLINE, LINE, ARC, CIRCLE, POLYLINE
- Normalize units to mm, 1:1 scale
- Clean non-structural elements (text, dimension, hatch)
- Output: wall_graph, room_polygons, opening_masks, column_masks (JSON/shapely)
- Review checkpoint: sample DXF parsing output verified against original

**Task 1.2 — PNG/PDF Fallback Vectorizer**
- Raster → Vector conversion (potrace / opencv contours)
- Structure reconstruction from bitmap
- Integrated as fallback when no DXF/DWG available
- Review checkpoint: vectorized output compared with original raster

**Task 1.3 — 2.5D ISO Renderer (Deterministic)**
- Orthographic camera preset (no perspective distortion)
- Fixed params: wall height=3000mm, floor thickness=200mm, door height=2100mm, window sill=900mm
- Camera: isometric angles (30° for standard ISO)
- Output: base_render.png + depth_map.png + normal_map.png (16-bit)
- Review checkpoint: rendered ISO view matches input geometry within 1% error

**Task 1.4 — Structural Data Export**
- Convert parsed geometry → structured JSON (wall_graph, room_data, anchors)
- Generate opening masks and column masks as PNG
- Review checkpoint: JSON structure validated, all rooms/columns present

### Stage 2: AI Style Refinement

**Task 2.1 — ComfyUI Installation + Custom Nodes**
- Install ComfyUI (port 8188)
- Install ComfyUI-Manager + required nodes:
  - ControlNet Tile (for tiled rendering)
  - ControlNet Canny / Depth / Lineart
  - SDXL Loader + FP8 quantization node
  - KSampler / Euler sampler
  - Image Composite (for tiled blending)
  - RGH Fated's "Clay Render Style" LoRA node (optional)
- Workflow export as JSON API format
- Review checkpoint: ComfyUI loads, SDXL generates test image

**Task 2.2 — AI Refinement Pipeline (Python diffusers fallback)**
- If ComfyUI unavailable: use diffusers Python pipeline
- SDXL + ControlNet Tile + Multi-ControlNet (Canny + Depth)
- Low denoise (0.25–0.35) strategy
- Output quality parity with ComfyUI workflow
- Review checkpoint: diffusers output matches ComfyUI output within 5% SSIM

**Task 2.3 — Tiled Rendering + QA**
- Large canvas (4096x4096+) tile splitting (512x512 or 768x768 tiles)
- Gaussian blending at tile boundaries
- VRAM optimization: model offload between tiles
- Review checkpoint: no visible seams in tiled output, VRAM < 8GB

### Stage 3: Output & QA

**Task 3.1 — Output Package Generator**
- Generate: background.png, transparent_background.png, masks.png, anchors.json
- All metadata embedded (hash, seed, sampler, steps, denoise, size, gen_time)
- Review checkpoint: all 4 output files exist, metadata valid JSON

**Task 3.2 — Auto QA System**
- Structure alignment: overlay AI output on original DXF → calculate deviation %
- Color rule check: HMI-friendly grayscale palette validation
- Artifact detection: detect AI-generated anomalies (floating objects, hallucinated walls)
- Fallback trigger: if QA fails → save deterministic render as final output
- Review checkpoint: QA correctly detects 3 known failure modes, fallback works

### Stage 4: Integration & Operations

**Task 4.1 — Config-driven Pipeline**
- External config.json: all pipeline parameters injectable without code changes
- Config schema: input_path, output_dir, wall_height, camera_angle, AI_params, etc.
- Review checkpoint: pipeline runs end-to-end with config only (no hardcoded values)

**Task 4.2 — Git Repository + Documentation**
- Initialize git repo: github.com/cloud-honey/hmi-map-pipeline
- Write README.md: installation, usage, config schema, output format
- Intermediate reports per task (this doc + per-task REVIEW reports)
- Review checkpoint: repo clone-and-run for new developer in < 30 min

---

## 3. Quality Gates (Per-Task Review)

After each task, generate a review report:

```
# Task X.Y Review Report

## Status: PASS / FAIL / NEEDS_REWORK

## Artifacts Produced
- list output files

## Test Results
- unit tests: X/Y passed
- integration check: PASS/FAIL

## Issues Found
- issue 1
- issue 2

## Sign-off
Ready for next task: YES/NO
```

---

## 4. Estimated Timeline

| Task | Description | Est. Time |
|------|-------------|-----------|
| 1.1 | DXF Parser | 2-3 days |
| 1.2 | PNG/PDF Vectorizer | 1 day |
| 1.3 | 2.5D ISO Renderer | 2 days |
| 1.4 | Structural Data Export | 1 day |
| 2.1 | ComfyUI Setup + Nodes | 1-2 days |
| 2.2 | AI Refinement Pipeline | 2-3 days |
| 2.3 | Tiled Rendering + QA | 1-2 days |
| 3.1 | Output Package Generator | 1 day |
| 3.2 | Auto QA System | 1-2 days |
| 4.1 | Config-driven Pipeline | 1-2 days |
| 4.2 | Git + Docs | 1 day |

**Total estimated: 14-22 working days**

---

## 5. Hardware & Environment

- Dev controller: XPS 15 (4GB VRAM — dev/test only, no heavy inference)
- Target runtime: 8GB+ RAM PC with GPU (GTX 1070 or equivalent)
- Python 3.12+, PyTorch 2.x (CUDA 12.x)
- Key packages: ezdxf, shapely, opencv-python, PIL, numpy, scipy, diffusers, transformers, accelerate
- ComfyUI standalone (port 8188) or Python pipeline fallback

---

## 6. Repository

```
github.com/cloud-honey/hmi-map-pipeline
```

## 7. External Dependencies

- SDXL Base 1.0: stabilityai/stable-diffusion-xl-base-1.0
- ControlNet Tile: lllyasviel/ControlNet-sdxl-tile
- ControlNet Canny/Depth/Lineart (SDXL version)
- LoRA (Clay texture): stabilityai/sdxl-vae (or equivalent)
- ComfyUI: comfyanonymous/ComfyUI