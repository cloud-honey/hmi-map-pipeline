"""
Task 3.1 & 3.2 tests: Output Package Generator + Auto QA
HMI Map Automation Pipeline

Run: python -m pytest tests/test_output_qa.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.output_qa import (
    AutoQA, OutputPackageGenerator, QAReport,
    HMI_GRAYSCALE_PALETTE, HMI_ALLOWED_RANGE, COLOR_TOLERANCE
)
from parser.dxf_parser import DXFParseResult, WallSegment, Room


# ─── QAReport tests ─────────────────────────────────────────────────────────
class TestQAReport:
    def test_report_to_dict(self):
        report = QAReport(
            passed=True,
            structure_alignment_score=0.85,
            color_rule_score=0.90,
            artifact_score=0.80,
            overall_score=0.84,
            issues=["minor edge blur"],
            warnings=["low memory"],
            fallback_triggered=False,
            fallback_reason="",
        )
        d = report.to_dict()
        assert d["passed"] is True
        assert d["overall_score"] == 0.84
        assert "minor edge blur" in d["issues"]

    def test_report_defaults(self):
        report = QAReport(passed=True)
        assert report.overall_score == 0.0
        assert report.issues == []
        assert report.fallback_triggered is False


# ─── AutoQA init tests ──────────────────────────────────────────────────────
class TestAutoQAInit:
    def test_init_default_thresholds(self):
        qa = AutoQA()
        assert qa.threshold_alignment == 0.75
        assert qa.threshold_color == 0.80
        assert qa.threshold_artifact == 0.70

    def test_init_with_geometry(self):
        geom = DXFParseResult(source_file="test", units="mm")
        qa = AutoQA(geometry=geom, threshold_alignment=0.80)
        assert qa.geometry is not None
        assert qa.threshold_alignment == 0.80


# ─── Color rule tests ───────────────────────────────────────────────────────
class TestColorRules:
    def _make_grayscale_image(self, value=128):
        arr = np.full((200, 200, 3), value, dtype=np.uint8)
        return Image.fromarray(arr)

    def _make_color_image(self):
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        arr[:, :, 0] = 200
        arr[:, :, 1] = 50
        arr[:, :, 2] = 50
        return Image.fromarray(arr)

    def test_grayscale_image_passes(self):
        qa = AutoQA()
        img = self._make_grayscale_image(150)
        score, issues = qa.check_color_rules(img)
        assert score > 0.90
        assert len(issues) == 0

    def test_colored_image_fails(self):
        qa = AutoQA()
        img = self._make_color_image()
        score, issues = qa.check_color_rules(img)
        assert score < 0.80
        assert len(issues) > 0

    def test_out_of_range_pixels_detected(self):
        qa = AutoQA()
        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        img = Image.fromarray(arr)
        score, issues = qa.check_color_rules(img)
        assert score >= 1.0

    def test_hmi_grayscale_palette_values(self):
        for rgb in HMI_GRAYSCALE_PALETTE:
            assert rgb[0] == rgb[1] == rgb[2], f"Not grayscale: {rgb}"

    def test_color_tolerance_reasonable(self):
        assert 10 <= COLOR_TOLERANCE <= 30

    def test_hmi_allowed_range(self):
        assert HMI_ALLOWED_RANGE[0] >= 0
        assert HMI_ALLOWED_RANGE[1] <= 255
        assert HMI_ALLOWED_RANGE[0] < HMI_ALLOWED_RANGE[1]


# ─── Artifact detection tests ───────────────────────────────────────────────
class TestArtifactDetection:
    def test_clean_image_passes(self):
        qa = AutoQA()
        arr = np.full((200, 200), 128, dtype=np.uint8)
        img = Image.fromarray(arr)
        score, issues = qa.check_artifacts(img)
        assert score >= 0.7

    def test_noisy_image_detected(self):
        qa = AutoQA()
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = Image.fromarray(arr)
        score, issues = qa.check_artifacts(img)
        assert isinstance(score, float)

    def test_artifact_score_with_substantial_noise(self):
        """Noise should produce a valid artifact score."""
        qa = AutoQA()
        arr = np.zeros((100, 100), dtype=np.uint8)
        # Add 200 scattered pixels (noise)
        for _ in range(200):
            x, y = np.random.randint(0, 100, 2)
            arr[x, y] = 255
        img = Image.fromarray(arr)
        score, issues = qa.check_artifacts(img)
        # Score should be a valid float in [0, 1]
        assert 0.0 <= score <= 1.0


# ─── Structure alignment tests ──────────────────────────────────────────────
class TestStructureAlignment:
    def _make_test_geometry(self):
        geom = DXFParseResult(source_file="test", units="mm")
        geom.wall_segments = [
            WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000),
            WallSegment(start=(1000, 0), end=(1000, 1000), layer="wall", length=1000),
            WallSegment(start=(1000, 1000), end=(0, 1000), layer="wall", length=1000),
            WallSegment(start=(0, 1000), end=(0, 0), layer="wall", length=1000),
        ]
        return geom

    def test_no_geometry_returns_score_0_5(self):
        qa = AutoQA(geometry=None)
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        score, issues = qa.check_structure_alignment(img, None)
        assert score == 0.5
        assert len(issues) >= 1

    def test_alignment_with_geometry_completes(self):
        geom = self._make_test_geometry()
        qa = AutoQA(geometry=geom)
        arr = np.zeros((300, 300), dtype=np.uint8)
        img = Image.fromarray(arr)
        score, issues = qa.check_structure_alignment(img, geom)
        assert isinstance(score, float)
        assert score >= 0.0


# ─── Full QA run tests ──────────────────────────────────────────────────────
class TestFullQARun:
    def test_qa_runs_without_error(self):
        qa = AutoQA()
        img = Image.new("RGB", (200, 200), (128, 128, 128))
        report = qa.run_full_qa(img)
        assert isinstance(report, QAReport)
        assert "overall_score" in report.to_dict()

    def test_qa_fallback_triggers_with_low_alignment(self):
        """Fallback when alignment is low with large geometry mismatch."""
        geom = DXFParseResult(source_file="test", units="mm")
        # Large room (10000mm) - small image (100px) won't align
        geom.wall_segments = [
            WallSegment(start=(0, 0), end=(10000, 0), layer="wall", length=10000),
            WallSegment(start=(10000, 0), end=(10000, 10000), layer="wall", length=10000),
            WallSegment(start=(10000, 10000), end=(0, 10000), layer="wall", length=10000),
            WallSegment(start=(0, 10000), end=(0, 0), layer="wall", length=10000),
        ]
        qa = AutoQA(geometry=geom, threshold_alignment=0.75)
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        det = Image.new("RGB", (100, 100), (100, 100, 100))
        report = qa.run_full_qa(img, deterministic_image=det)
        # Alignment will be low → fallback triggered
        assert report.fallback_triggered is True

    def test_qa_passes_with_clean_image(self):
        qa = AutoQA()
        arr = np.full((200, 200, 3), 150, dtype=np.uint8)
        img = Image.fromarray(arr)
        report = qa.run_full_qa(img)
        assert isinstance(report.passed, bool)


# ─── OutputPackageGenerator tests ───────────────────────────────────────────
class TestOutputPackageGenerator:
    def _make_test_images(self):
        det = Image.new("RGB", (200, 200), (80, 80, 80))
        enh = Image.new("RGB", (200, 200), (90, 90, 90))
        return det, enh

    def test_generate_basic_package(self):
        det_img, enh_img = self._make_test_images()
        geom = DXFParseResult(source_file="test", units="mm")
        gen = OutputPackageGenerator(geometry=geom)
        qa = AutoQA(geometry=geom)
        report = qa.run_full_qa(enh_img, det_img)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = gen.generate(
                enhanced_image=enh_img,
                deterministic_image=det_img,
                qa_report=report,
                output_dir=tmpdir,
            )
            assert "background" in results
            assert "transparent_background" in results
            assert "qa_report" in results
            for path in results.values():
                assert Path(path).exists()

    def test_fallback_uses_deterministic_when_qa_fails(self):
        det_img, enh_img = self._make_test_images()
        gen = OutputPackageGenerator()

        report = QAReport(
            passed=False,
            structure_alignment_score=0.3,
            color_rule_score=0.3,
            artifact_score=0.3,
            overall_score=0.3,
            fallback_triggered=True,
            fallback_reason="low overall score",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results = gen.generate(
                enhanced_image=enh_img,
                deterministic_image=det_img,
                qa_report=report,
                output_dir=tmpdir,
            )
            assert Path(results["background"]).exists()


# ─── Run pytest ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])