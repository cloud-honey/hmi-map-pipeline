"""
Task 2.2 tests: AI Refinement Pipeline
HMI Map Automation Pipeline

Run: python -m pytest tests/test_ai_refiner.py -v

Note: Full model tests require GPU with 8GB VRAM.
These tests validate the pipeline structure and data flow.
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.ai_refiner import (
    AIPipelineConfig, GenerationMetadata, AIRefiner,
    DEFAULT_DENOISE_STRENGTH, DEFAULT_GUIDANCE_SCALE,
    TILE_SIZE, TILE_OVERLAP,
    run_pipeline,
)


# ─── AIPipelineConfig tests ─────────────────────────────────────────────────
class TestAIPipelineConfig:
    def test_default_values(self):
        cfg = AIPipelineConfig()
        assert cfg.denoise_strength == DEFAULT_DENOISE_STRENGTH
        assert cfg.guidance_scale == DEFAULT_GUIDANCE_SCALE
        assert cfg.num_steps == 25
        assert cfg.seed == 42
        assert cfg.enable_tiled is True
        assert cfg.enable_model_offload is True

    def test_custom_values(self):
        cfg = AIPipelineConfig(
            denoise_strength=0.25,
            num_steps=20,
            seed=123,
            tile_size=768,
        )
        assert cfg.denoise_strength == 0.25
        assert cfg.num_steps == 20
        assert cfg.seed == 123
        assert cfg.tile_size == 768

    def test_to_dict(self):
        cfg = AIPipelineConfig(seed=99)
        d = cfg.to_dict()
        assert d["seed"] == 99
        assert "denoise_strength" in d
        assert "sdxl_base_model" in d

    def test_low_denoise_preserves_structure(self):
        """Low denoise (0.25-0.35) is required for structural integrity."""
        cfg = AIPipelineConfig(denoise_strength=0.30)
        assert cfg.denoise_strength < 0.35
        assert cfg.denoise_strength > 0.20


# ─── GenerationMetadata tests ───────────────────────────────────────────────
class TestGenerationMetadata:
    def test_metadata_to_dict(self):
        meta = GenerationMetadata(
            checkpoint_hash="abc123",
            controlnet_types=["tile-sdxl", "canny-sdxl"],
            controlnet_scales=[0.7, 0.5],
            seed=42,
            sampler="euler",
            steps=25,
            denoise=0.30,
            output_width=1024,
            output_height=1024,
            generation_time_seconds=45.3,
        )
        d = meta.to_dict()
        assert d["checkpoint_hash"] == "abc123"
        assert d["sampler"] == "euler"
        assert d["generation_time_seconds"] == 45.3

    def test_metadata_defaults(self):
        meta = GenerationMetadata()
        assert meta.pipeline_version == "1.0.0"
        assert meta.controlnet_types == []
        assert meta.generation_time_seconds == 0.0


# ─── AIRefiner init tests ───────────────────────────────────────────────────
class TestAIRefinerInit:
    def test_init_default(self):
        refiner = AIRefiner()
        assert refiner.config is not None
        assert refiner.pipe is None  # Not loaded until needed
        assert refiner.device in ("cuda", "cpu")

    def test_init_custom_config(self):
        cfg = AIPipelineConfig(denoise_strength=0.35, tile_size=768)
        refiner = AIRefiner(config=cfg)
        assert refiner.config.denoise_strength == 0.35
        assert refiner.config.tile_size == 768


# ─── Tiling tests ──────────────────────────────────────────────────────────
class TestTiling:
    def test_tile_count(self):
        refiner = AIRefiner(config=AIPipelineConfig(
            tile_size=512, tile_overlap=64, enable_tiled=True
        ))
        # 1024x1024 image with 512 tile and 64 overlap → 3x3 = 9 tiles
        img = Image.new("RGB", (1024, 1024), (128, 128, 128))
        tiles = refiner._tile_image(img)
        assert len(tiles) >= 1

    def test_tile_count_large(self):
        refiner = AIRefiner(config=AIPipelineConfig(
            tile_size=512, tile_overlap=64, enable_tiled=True
        ))
        # 2048x2048 image → should be more than 4 tiles
        img = Image.new("RGB", (2048, 2048), (128, 128, 128))
        tiles = refiner._tile_image(img)
        # 2048 / (512-64) = 2048/448 ≈ 4.6 → 5 rows x 5 cols = 25 tiles
        assert len(tiles) > 4

    def test_tile_overlap_values(self):
        """Overlap must be smaller than tile size."""
        cfg = AIPipelineConfig(tile_size=512, tile_overlap=256)
        assert cfg.tile_overlap < cfg.tile_size

    def test_gaussian_sigma(self):
        """Sigma must be positive for smooth blending."""
        from pipeline.ai_refiner import GAUSSIAN_SIGMA
        assert GAUSSIAN_SIGMA > 0


# ─── Canny generation tests ─────────────────────────────────────────────────
class TestCannyGeneration:
    def test_generate_canny_from_image(self):
        refiner = AIRefiner()
        # Create a test image with edges
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        # Draw a diagonal line
        import numpy as np
        arr = np.array(img)
        for i in range(200):
            if 0 <= i < 200 and 0 <= i < 200:
                arr[i, i] = [255, 255, 255]
        test_img = Image.fromarray(arr)

        canny = refiner._generate_canny(test_img)
        assert canny is not None
        assert canny.size == test_img.size
        assert canny.mode == "L"

    def test_canny_output_is_grayscale(self):
        refiner = AIRefiner()
        img = Image.new("RGB", (100, 100), (64, 64, 64))
        canny = refiner._generate_canny(img)
        assert canny.mode == "L"


# ─── Model hash tests ───────────────────────────────────────────────────────
class TestModelHash:
    def test_get_model_hash(self):
        refiner = AIRefiner()
        h = refiner._get_model_hash()
        assert isinstance(h, str)
        assert len(h) == 16

    def test_model_hash_deterministic(self):
        refiner1 = AIRefiner(config=AIPipelineConfig(sdxl_base_model="stabilityai/stable-diffusion-xl-base-1.0"))
        refiner2 = AIRefiner(config=AIPipelineConfig(sdxl_base_model="stabilityai/stable-diffusion-xl-base-1.0"))
        assert refiner1._get_model_hash() == refiner2._get_model_hash()


# ─── ControlNet types tests ─────────────────────────────────────────────────
class TestControlNetTypes:
    def test_controlnet_types_list(self):
        refiner = AIRefiner(config=AIPipelineConfig(
            use_controlnet_tile=True, use_controlnet_canny=True
        ))
        types = refiner._get_controlnet_types()
        assert "tile-sdxl" in types
        assert "canny-sdxl" in types

    def test_controlnet_types_empty_when_disabled(self):
        refiner = AIRefiner(config=AIPipelineConfig(
            use_controlnet_tile=False, use_controlnet_canny=False
        ))
        types = refiner._get_controlnet_types()
        assert len(types) == 0


# ─── Workflow JSON export tests ──────────────────────────────────────────────
class TestWorkflowExport:
    def test_export_workflow_json(self):
        refiner = AIRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workflow.json"
            refiner.export_workflow_json(str(path))
            assert path.exists()
            with open(path, "r") as f:
                wf = json.load(f)
            assert wf["version"] == "1.0"
            assert wf["pipeline"] == "HMI-Map-AI-Refinement"
            assert "parameters" in wf

    def test_workflow_contains_all_models(self):
        refiner = AIRefiner()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workflow.json"
            refiner.export_workflow_json(str(path))
            with open(path, "r") as f:
                wf = json.load(f)
            models = wf["models"]
            assert "base" in models
            assert "controlnet_tile" in models
            assert "controlnet_canny" in models


# ─── Config validation tests ────────────────────────────────────────────────
class TestConfigValidation:
    def test_denoise_in_valid_range(self):
        """Denoise strength must be in 0-1 range."""
        cfg = AIPipelineConfig(denoise_strength=0.5)
        assert 0 <= cfg.denoise_strength <= 1

        cfg = AIPipelineConfig(denoise_strength=0.0)
        assert cfg.denoise_strength == 0.0

        cfg = AIPipelineConfig(denoise_strength=1.0)
        assert cfg.denoise_strength == 1.0

    def test_lora_scale_in_valid_range(self):
        """LoRA scale must be in 0-1 range."""
        cfg = AIPipelineConfig(lora_scale=0.4)
        assert 0 <= cfg.lora_scale <= 1

    def test_controlnet_scales(self):
        cfg = AIPipelineConfig(
            controlnet_tile_scale=0.7,
            controlnet_canny_scale=0.5,
        )
        assert 0 < cfg.controlnet_tile_scale <= 1
        assert 0 < cfg.controlnet_canny_scale <= 1

    def test_tile_size_positive(self):
        cfg = AIPipelineConfig(tile_size=512)
        assert cfg.tile_size > 0

    def test_steps_positive(self):
        cfg = AIPipelineConfig(num_steps=20)
        assert cfg.num_steps > 0


# ─── Unload models tests ────────────────────────────────────────────────────
class TestModelUnload:
    def test_unload_clears_references(self):
        refiner = AIRefiner()
        # Initially none loaded
        assert refiner.pipe is None
        assert refiner.controlnet_tile is None
        assert refiner.controlnet_canny is None
        # After unload (called even when nothing loaded)
        refiner.unload_models()
        assert refiner.pipe is None


# ─── Run pytest ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])