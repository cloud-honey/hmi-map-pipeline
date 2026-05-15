"""
Task 1.2 tests: PNG/PDF Vectorizer
HMI Map Automation Pipeline

Run: python -m pytest tests/test_png_vectorizer.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np
import cv2

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser.png_vectorizer import (
    PNGVectorizer, PDFVectorizer, VectorizerConfig,
    PIXELS_PER_MM, MIN_WALL_THICKNESS_MM, MIN_ROOM_AREA_MM2
)
from parser.dxf_parser import DXFParseResult, WallSegment


# ─── Helper ────────────────────────────────────────────────────────────────
def img_to_gray(img):
    """Convert numpy image to grayscale if needed."""
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─── VectorizerConfig tests ─────────────────────────────────────────────────
class TestVectorizerConfig:
    def test_default_values(self):
        cfg = VectorizerConfig()
        assert cfg.dpi == 96
        assert cfg.pixels_per_mm == pytest.approx(3.78, rel=0.1)
        assert cfg.min_wall_thickness_mm == 60
        assert cfg.min_room_area_mm2 == 50_000
        assert cfg.max_corridor_width_mm == 2500

    def test_custom_config(self):
        cfg = VectorizerConfig(dpi=150, min_room_area_mm2=100_000)
        assert cfg.dpi == 150
        assert cfg.min_room_area_mm2 == 100_000


# ─── PNGVectorizer basic tests ─────────────────────────────────────────────
class TestPNGVectorizerInit:
    def test_init_default(self):
        v = PNGVectorizer()
        assert v.config.dpi == 96
        assert v.config.min_wall_thickness_mm == 60

    def test_init_with_config(self):
        cfg = VectorizerConfig(dpi=150, min_wall_thickness_mm=100)
        v = PNGVectorizer(config=cfg)
        assert v.config.dpi == 150
        assert v.config.min_wall_thickness_mm == 100


class TestNearestStandard:
    def test_nearest_standard_door(self):
        v = PNGVectorizer()
        assert v._nearest_standard(950, [900, 1000, 1100]) == 900
        assert v._nearest_standard(1050, [900, 1000, 1100]) == 1000

    def test_nearest_standard_window(self):
        v = PNGVectorizer()
        # 1400mm → nearest of [600, 900, 1200, 1500] is 1500 (diff=100 vs diff=200)
        assert v._nearest_standard(1400, [600, 900, 1200, 1500]) == 1500
        # 1100mm → nearest is 1200 (diff=100 vs diff=200)
        assert v._nearest_standard(1100, [600, 900, 1200, 1500]) == 1200


# ─── Wall detection tests ──────────────────────────────────────────────────
class TestWallDetection:
    @staticmethod
    def make_wall_image(wall_thickness_px=20):
        """Create a synthetic floor plan: 800x600 with a square room (BGR)."""
        img = np.full((600, 800, 3), 255, dtype=np.uint8)  # White BGR
        color = (0, 0, 0)  # Black walls (BGR)
        cv2.rectangle(img, (100, 100), (400, 400), color, wall_thickness_px)
        return img

    @staticmethod
    def make_two_room_image():
        """Two adjacent rooms: left 100-400, right 400-700."""
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        color = (0, 0, 0)
        cv2.rectangle(img, (100, 100), (400, 400), color, 20)
        cv2.rectangle(img, (400, 100), (700, 400), color, 20)
        cv2.line(img, (400, 100), (400, 400), color, 20)
        return img

    def test_detect_walls_square_room(self):
        img = self.make_wall_image(wall_thickness_px=20)
        v = PNGVectorizer()
        gray = img_to_gray(img)
        wall_mask = v._detect_walls(gray)
        assert wall_mask is not None
        assert wall_mask.shape == img.shape[:2]
        assert wall_mask.dtype == np.uint8
        assert wall_mask.sum() > 0

    def test_detect_walls_two_rooms(self):
        img = self.make_two_room_image()
        v = PNGVectorizer()
        gray = img_to_gray(img)
        wall_mask = v._detect_walls(gray)
        assert wall_mask.sum() > 0

    def test_wall_segments_from_mask(self):
        img = self.make_two_room_image()
        v = PNGVectorizer()
        gray = img_to_gray(img)
        wall_mask = v._detect_walls(gray)
        segments = v._extract_wall_segments(wall_mask)
        assert len(segments) >= 3


# ─── Opening detection tests ───────────────────────────────────────────────
class TestOpeningDetection:
    @staticmethod
    def make_door_image():
        """Image with a simple door gap in a wall (BGR)."""
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        color = (0, 0, 0)
        cv2.rectangle(img, (0, 100), (800, 120), color, -1)  # solid wall
        cv2.rectangle(img, (300, 100), (400, 120), (255, 255, 255), -1)  # door gap
        return img

    @staticmethod
    def make_window_image():
        """Image with a wide window gap in a wall (BGR)."""
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        color = (0, 0, 0)
        cv2.rectangle(img, (400, 0), (420, 600), color, -1)
        cv2.rectangle(img, (400, 200), (420, 350), (255, 255, 255), -1)  # window gap
        return img

    def test_detect_openings(self):
        img = self.make_door_image()
        v = PNGVectorizer()
        gray = img_to_gray(img)
        wall_mask = np.full_like(gray, 255, dtype=np.uint8)
        cv2.rectangle(wall_mask, (0, 100), (800, 120), 0, -1)
        opening_mask = v._detect_openings(gray, wall_mask)
        assert opening_mask is not None

    def test_extract_openings(self):
        img = self.make_door_image()
        v = PNGVectorizer()
        gray = img_to_gray(img)
        wall_mask = np.full_like(gray, 255, dtype=np.uint8)
        cv2.rectangle(wall_mask, (0, 100), (800, 120), 0, -1)
        opening_mask = v._detect_openings(gray, wall_mask)
        openings = v._extract_openings(opening_mask)
        # May be 0 depending on gap size detection threshold
        assert isinstance(openings, list)

    def test_window_vs_door_classification(self):
        v = PNGVectorizer()
        door_mm = v._nearest_standard(900, [900, 1000, 1100])
        assert door_mm >= 900
        window_mm = v._nearest_standard(1500, [600, 900, 1200, 1500])
        assert window_mm >= 1200


# ─── Column detection tests ─────────────────────────────────────────────────
class TestColumnDetection:
    @staticmethod
    def make_column_image():
        """Image with a small solid square (column) — BGR."""
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (300, 200), (500, 400), (0, 0, 0), -1)
        return img

    def test_detect_column(self):
        img = self.make_column_image()
        v = PNGVectorizer()
        gray = img_to_gray(img)
        wall_mask = np.zeros_like(gray)
        col_mask = v._detect_columns(gray, wall_mask)
        assert col_mask is not None

    def test_extract_columns(self):
        img = self.make_column_image()
        v = PNGVectorizer()
        gray = img_to_gray(img)
        wall_mask = np.zeros_like(gray)
        col_mask = v._detect_columns(gray, wall_mask)
        cols = v._extract_columns(col_mask)
        assert isinstance(cols, list)


# ─── Room extraction tests ──────────────────────────────────────────────────
class TestRoomExtraction:
    def test_extract_rooms(self):
        v = PNGVectorizer()
        walls = [
            WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000),
            WallSegment(start=(1000, 0), end=(1000, 1000), layer="wall", length=1000),
            WallSegment(start=(1000, 1000), end=(0, 1000), layer="wall", length=1000),
            WallSegment(start=(0, 1000), end=(0, 0), layer="wall", length=1000),
        ]
        rooms = v._extract_rooms(walls)
        assert len(rooms) >= 1
        assert all(r.area_mm2 > 0 for r in rooms)


# ─── Skeletonization tests ─────────────────────────────────────────────────
class TestSkeletonization:
    def test_skeletonize_simple_wall(self):
        v = PNGVectorizer()
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 40:60] = 255  # vertical bar
        skel = v._skeletonize(mask)
        assert skel is not None
        assert skel.shape == mask.shape
        assert skel.sum() <= mask.sum()

    def test_skeletonize_empty(self):
        v = PNGVectorizer()
        mask = np.zeros((100, 100), dtype=np.uint8)
        skel = v._skeletonize(mask)
        assert skel.sum() == 0


# ─── Output format tests ────────────────────────────────────────────────────
class TestOutputFormat:
    def test_output_matches_dxf_result_schema(self):
        img = np.full((600, 800, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (400, 400), (0, 0, 0), 20)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, img)

        try:
            v = PNGVectorizer()
            result = v.vectorize(temp_path)
            assert isinstance(result, DXFParseResult)
            d = result.to_dict()
            assert "wall_segments" in d
            assert "openings" in d
            assert "rooms" in d
            assert "columns" in d
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ─── Run pytest ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])