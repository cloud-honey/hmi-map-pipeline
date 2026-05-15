"""
Task 4.1: Config-driven Main Pipeline Orchestrator
HMI Map Automation Pipeline

End-to-end pipeline:
  DXF/PNG → Parser → Geometry → ISO Renderer → AI Refinement → QA → Output Package

Run: python src/pipeline/main_pipeline.py --config config/pipeline.json

Input types auto-detected: .dxf, .dwg, .png, .pdf
Output: background.png, transparent_background.png, masks.png,
        anchors.json, metadata.json, qa_report.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ─── Project root for imports ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from parser.dxf_parser import DXFParser, DXFParseResult
from parser.png_vectorizer import PNGVectorizer, PDFVectorizer
from renderer.iso_renderer import ISORenderer, ISOCamera, RenderConfig
from parser.structural_exporter import StructuralExporter
from pipeline.ai_refiner import AIRefiner, AIPipelineConfig
from pipeline.output_qa import AutoQA, OutputPackageGenerator, QAReport


# ─── Logging ────────────────────────────────────────────────────────────────
class SimpleLogger:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.steps = []

    def log(self, step: str, message: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [{step}] {message}"
        print(line)
        self.steps.append((ts, step, message))

    def summary(self):
        print("\n=== Pipeline Summary ===")
        for ts, step, msg in self.steps:
            print(f"  {step}: {msg}")


def load_config(config_path: str) -> dict:
    """Load and validate config JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        config = json.load(f)

    # Apply defaults
    config.setdefault("input", {})
    config.setdefault("output", {})
    config.setdefault("parser", {})
    config.setdefault("renderer", {})
    config.setdefault("ai_refinement", {})
    config.setdefault("qa", {})
    config.setdefault("pipeline", {})

    return config


def detect_input_type(path: str) -> str:
    """Auto-detect input file type from extension."""
    ext = Path(path).suffix.lower()
    type_map = {".dxf": "dxf", ".dwg": "dwg", ".png": "png", ".pdf": "pdf"}
    return type_map.get(ext, "auto")


# ─── Main Pipeline ─────────────────────────────────────────────────────────
class HMIMapPipeline:
    """
    End-to-end HMI map generation pipeline.

    Stages:
      1. Parse (DXF/DWG/PNG/PDF → DXFParseResult)
      2. Render (DXFParseResult → base ISO render + depth + normal)
      3. Export structural data (geometry.json, anchors.json, masks.png)
      4. AI refinement (optional, diffusers/ComfyUI)
      5. QA + output package
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = SimpleLogger(verbose=config.get("pipeline", {}).get("verbose", True))
        self.geometry: Optional[DXFParseResult] = None
        self.det_img: Optional["Image.Image"] = None  # type: ignore
        self.enh_img: Optional["Image.Image"] = None  # type: ignore

    def run(self, input_path: str = None) -> dict[str, str]:
        """Run full pipeline. Returns dict of output file paths."""
        input_path = input_path or self.config["input"]["path"]
        output_dir = self.config["output"].get("directory", "./output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.logger.log("INIT", f"Starting HMI Map Pipeline: {input_path}")

        # ─── Stage 1: Parse ───────────────────────────────────────────────
        stage_start = time.time()
        input_type = self.config["input"].get("type", "auto")
        if input_type == "auto":
            input_type = detect_input_type(input_path)

        self.logger.log("PARSER", f"Parsing {input_type.upper()} file: {input_path}")

        if input_type in ("dxf", "dwg"):
            parser = DXFParser(input_path)
            self.geometry = parser.parse()
        elif input_type == "png":
            vectorizer = PNGVectorizer()
            self.geometry = vectorizer.vectorize(input_path)
        elif input_type == "pdf":
            vectorizer = PDFVectorizer()
            self.geometry = vectorizer.vectorize(input_path)
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        self.logger.log("PARSER", f"  Walls: {len(self.geometry.wall_segments)}, "
                            f"Rooms: {len(self.geometry.rooms)}, "
                            f"Openings: {len(self.geometry.openings)}, "
                            f"Columns: {len(self.geometry.columns)}")

        # ─── Stage 2: ISO Renderer ─────────────────────────────────────────
        self.logger.log("RENDERER", "Generating deterministic ISO render...")
        r_cfg = self.config.get("renderer", {})
        cam = ISOCamera(pixels_per_mm=r_cfg.get("camera_pixels_per_mm", 0.5))
        r_conf = RenderConfig(
            wall_height_mm=r_cfg.get("wall_height_mm", 3000),
            wall_thickness_mm=r_cfg.get("wall_thickness_mm", 200),
            floor_thickness_mm=r_cfg.get("floor_thickness_mm", 200),
            door_height_mm=r_cfg.get("door_height_mm", 2100),
            window_sill_mm=r_cfg.get("window_sill_mm", 900),
        )
        renderer = ISORenderer(camera=cam, config=r_conf)
        outputs = renderer.render(self.geometry)
        det_img = outputs["render"]
        depth_img = outputs["depth"]
        normal_img = outputs["normal"]

        # Save deterministic render
        det_path = str(Path(output_dir) / "base_render.png")
        det_img.save(det_path)
        self.logger.log("RENDERER", f"  Saved: base_render.png ({det_img.size})")
        self.det_img = det_img

        # ─── Stage 3: Structural Export ─────────────────────────────────────
        self.logger.log("EXPORTER", "Exporting structural data...")
        exporter = StructuralExporter(self.geometry)
        export_results = exporter.export_all(output_dir)
        self.logger.log("EXPORTER", f"  geometry.json, anchors.json, masks.png, metadata.json saved")

        # ─── Stage 4: AI Refinement (optional) ───────────────────────────
        ai_cfg = self.config.get("ai_refinement", {})
        skip_ai = (
            self.config.get("pipeline", {}).get("skip_ai_refinement", False) or
            self.config.get("pipeline", {}).get("force_deterministic", False) or
            not ai_cfg.get("enabled", True)
        )

        if skip_ai:
            self.logger.log("AI_REFINE", "Skipped (disabled or force_deterministic=True)")
        else:
            self.logger.log("AI_REFINE", "Running AI refinement...")
            try:
                refiner = AIRefiner(config=AIPipelineConfig(
                    sdxl_base_model=ai_cfg.get("model_path", "stabilityai/stable-diffusion-xl-base-1.0"),
                    denoise_strength=ai_cfg.get("denoise_strength", 0.30),
                    num_steps=ai_cfg.get("num_steps", 25),
                    seed=ai_cfg.get("seed", 42),
                    tile_size=ai_cfg.get("tile_size", 512),
                    tile_overlap=ai_cfg.get("tile_overlap", 64),
                    controlnet_tile_scale=ai_cfg.get("controlnet_tile_scale", 0.7),
                    controlnet_canny_scale=ai_cfg.get("controlnet_canny_scale", 0.5),
                    enable_model_offload=ai_cfg.get("enable_model_offload", True),
                ))
                enh_img, metadata = refiner.refine(
                    base_render=det_img,
                    canny_map=None,  # auto-generated
                )
                enh_path = str(Path(output_dir) / "ai_enhanced.png")
                enh_img.save(enh_path)
                self.enh_img = enh_img
                self.logger.log("AI_REFINE", f"  Saved: ai_enhanced.png ({enh_img.size})")
            except Exception as e:
                self.logger.log("AI_REFINE", f"  Failed: {e} — using deterministic render")
                self.enh_img = None

        # ─── Stage 5: QA + Output Package ────────────────────────────────
        self.logger.log("QA", "Running automated QA...")
        qa_cfg = self.config.get("qa", {})
        qa = AutoQA(
            geometry=self.geometry,
            threshold_alignment=qa_cfg.get("threshold_alignment", 0.75),
            threshold_color=qa_cfg.get("threshold_color", 0.80),
            threshold_artifact=qa_cfg.get("threshold_artifact", 0.70),
        )

        check_img = self.enh_img if self.enh_img else det_img
        det_for_fallback = det_img if self.enh_img else None
        qa_report = qa.run_full_qa(check_img, det_for_fallback)

        self.logger.log("QA", f"  Passed: {qa_report.passed}, "
                            f"Overall: {qa_report.overall_score:.0%}, "
                            f"Fallback: {qa_report.fallback_triggered}")
        if qa_report.issues:
            for issue in qa_report.issues[:3]:
                self.logger.log("QA", f"  Issue: {issue}")

        # Generate final package
        gen = OutputPackageGenerator(geometry=self.geometry)
        final_results = gen.generate(
            enhanced_image=self.enh_img,
            deterministic_image=det_img,
            qa_report=qa_report,
            masks_path=export_results.get("masks"),
            anchors_path=export_results.get("anchors"),
            metadata_path=export_results.get("metadata"),
            output_dir=output_dir,
        )

        self.logger.log("COMPLETE", f"Output: {output_dir}")
        self.logger.summary()

        return final_results


# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HMI Map Automation Pipeline")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--input", help="Override input file path from config")
    parser.add_argument("--output", help="Override output directory from config")

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    if args.input:
        cfg["input"]["path"] = args.input
    if args.output:
        cfg["output"]["directory"] = args.output

    print(f"\n=== HMI Map Pipeline ===")
    print(f"Config: {args.config}")
    print(f"Input:  {cfg['input']['path']}")
    print(f"Output: {cfg['output'].get('directory', './output')}\n")

    pipeline = HMIMapPipeline(cfg)
    results = pipeline.run()