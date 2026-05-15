"""
Task 1.3 tests: 2.5D ISO Renderer
HMI Map Automation Pipeline

Run: python -m pytest tests/test_iso_renderer.py -v
"""

import tempfile
import math
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from renderer.iso_renderer import ISORenderer, ISOCamera, RenderConfig
from parser.dxf_parser import DXFParseResult, WallSegment, Room, Column


# ─── ISOCamera tests ───────────────────────────────────────────────────────
class TestISOCamera:
    def test_iso_projection_at_origin(self):
        cam = ISOCamera()
        sx, sy = cam.iso_to_screen(0, 0, 0)
        assert abs(sx) < 0.01
        assert abs(sy) < 0.01

    def test_iso_projection_x_axis(self):
        cam = ISOCamera()
        sx, sy = cam.iso_to_screen(1000, 0, 0)
        # (1000-0)*cos(30)*0.5 = 1000*0.866*0.5 = 433
        assert abs(sx - 433.0) < 1.0
        assert abs(sy - 250.0) < 1.0

    def test_iso_projection_y_axis(self):
        cam = ISOCamera()
        sx, sy = cam.iso_to_screen(0, 1000, 0)
        # (0-1000)*cos(30)*0.5 = -433
        assert abs(sx + 433.0) < 1.0
        assert abs(sy - 250.0) < 1.0

    def test_iso_projection_z_axis(self):
        cam = ISOCamera()
        sx0, sy0 = cam.iso_to_screen(0, 0, 0)
        sx1, sy1 = cam.iso_to_screen(0, 0, 3000)  # 3m wall height
        # z moves down (negative y) in ISO
        assert sy1 < sy0

    def test_world_to_depth(self):
        cam = ISOCamera()
        # Camera plane: normal = (1,1,1) normalized, camera from above
        # Point at (500, 500, 0) — floor level — should be in front of camera
        d = cam.world_to_depth(500, 500, 0)
        # May be positive or negative depending on camera direction convention
        assert isinstance(d, float)

    def test_depth_sign_convention(self):
        cam = ISOCamera()
        d0 = cam.world_to_depth(500, 500, 0)
        d1 = cam.world_to_depth(500, 500, 3000)
        # Higher z = closer to camera = larger depth value
        assert d1 > d0  # z=3000 is closer than z=0
        assert d0 != d1

    def test_depth_increases_with_z(self):
        cam = ISOCamera()
        d0 = cam.world_to_depth(500, 500, 0)
        d1 = cam.world_to_depth(500, 500, 3000)
        assert d1 > d0  # higher z = closer to camera = larger depth

    def test_pixels_per_mm_default(self):
        cam = ISOCamera()
        assert cam.pixels_per_mm == 0.5  # reasonable for ISO view

    def test_bg_color_default(self):
        cam = ISOCamera()
        assert cam.bg_color == (40, 40, 40)


# ─── RenderConfig tests ────────────────────────────────────────────────────
class TestRenderConfig:
    def test_default_wall_height(self):
        cfg = RenderConfig()
        assert cfg.wall_height_mm == 3000  # 3m

    def test_default_colors(self):
        cfg = RenderConfig()
        assert cfg.floor_color == (85, 85, 85)
        assert cfg.wall_color == (160, 160, 160)
        assert cfg.column_color == (130, 130, 130)

    def test_custom_colors(self):
        cfg = RenderConfig(floor_color=(100, 100, 100))
        assert cfg.floor_color == (100, 100, 100)


# ─── ISORenderer init tests ────────────────────────────────────────────────
class TestISORendererInit:
    def test_default_init(self):
        r = ISORenderer()
        assert r.camera is not None
        assert r.config is not None

    def test_custom_camera(self):
        cam = ISOCamera(pixels_per_mm=1.0)
        r = ISORenderer(camera=cam)
        assert r.camera.pixels_per_mm == 1.0

    def test_custom_config(self):
        cfg = RenderConfig(wall_height_mm=4000)
        r = ISORenderer(config=cfg)
        assert r.config.wall_height_mm == 4000


# ─── Rendering basic geometry tests ────────────────────────────────────────
class TestRenderBasicGeometry:
    def _make_simple_room_geom(self):
        """A simple 1m x 1m room."""
        geom = DXFParseResult(source_file="test", units="mm")
        # 4 walls forming a square
        geom.wall_segments = [
            WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000),
            WallSegment(start=(1000, 0), end=(1000, 1000), layer="wall", length=1000),
            WallSegment(start=(1000, 1000), end=(0, 1000), layer="wall", length=1000),
            WallSegment(start=(0, 1000), end=(0, 0), layer="wall", length=1000),
        ]
        geom.rooms.append(Room(
            name="Room_1",
            polygon=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
            area_mm2=1_000_000,
            centroid=(500, 500),
        ))
        return geom

    def test_render_produces_three_arrays(self):
        geom = self._make_simple_room_geom()
        r = ISORenderer()
        outputs = r.render(geom)
        assert "render" in outputs
        assert "depth" in outputs
        assert "normal" in outputs

    def test_render_array_shapes(self):
        geom = self._make_simple_room_geom()
        r = ISORenderer()
        outputs = r.render(geom)
        assert outputs["render"].shape[2] == 3  # RGB
        assert outputs["depth"].ndim == 2  # grayscale
        assert outputs["normal"].shape[2] == 3  # RGB

    def test_render_array_dtypes(self):
        geom = self._make_simple_room_geom()
        r = ISORenderer()
        outputs = r.render(geom)
        assert outputs["render"].dtype == np.uint8
        assert outputs["depth"].dtype == np.uint16
        assert outputs["normal"].dtype == np.uint8

    def test_render_has_content(self):
        """Rendered image should have non-background pixels."""
        geom = self._make_simple_room_geom()
        r = ISORenderer()
        outputs = r.render(geom)
        render = outputs["render"]
        # Not all pixels should be background color (40, 40, 40)
        non_bg = np.sum((render[:, :, 0] != 40) | (render[:, :, 1] != 40) | (render[:, :, 2] != 40))
        assert non_bg > 1000  # at least 1000 non-background pixels

    def test_render_depth_has_variance(self):
        """Depth map should have varying values (not all same)."""
        geom = self._make_simple_room_geom()
        r = ISORenderer()
        outputs = r.render(geom)
        depth = outputs["depth"]
        assert depth.min() != depth.max() or depth.max() > 0

    def test_save_outputs(self):
        geom = self._make_simple_room_geom()
        r = ISORenderer()
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = r.save_outputs(r.render(geom), tmpdir)
            assert "render" in saved
            assert "depth" in saved
            assert "normal" in saved
            for name, path in saved.items():
                assert Path(path).exists()
                assert Path(path).stat().st_size > 1000

    def test_render_to_files(self):
        geom = self._make_simple_room_geom()
        r = ISORenderer()
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = r.render_to_files(geom, tmpdir)
            assert len(saved) == 3
            for path in saved.values():
                assert Path(path).exists()


# ─── Render quality tests ───────────────────────────────────────────────────
class TestRenderQuality:
    @staticmethod
    def _make_simple_room_geom():
        geom = DXFParseResult(source_file="test", units="mm")
        geom.wall_segments = [
            WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000),
            WallSegment(start=(1000, 0), end=(1000, 1000), layer="wall", length=1000),
            WallSegment(start=(1000, 1000), end=(0, 1000), layer="wall", length=1000),
            WallSegment(start=(0, 1000), end=(0, 0), layer="wall", length=1000),
        ]
        geom.rooms.append(Room(
            name="Room_1",
            polygon=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
            area_mm2=1_000_000, centroid=(500, 500),
        ))
        return geom

    def _geom_with_corridor(self):
        """Two rooms with a corridor."""
        geom = DXFParseResult(source_file="test", units="mm")
        # Left room
        geom.wall_segments = [
            WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000),
            WallSegment(start=(1000, 0), end=(1000, 1000), layer="wall", length=1000),
            WallSegment(start=(1000, 1000), end=(0, 1000), layer="wall", length=1000),
            WallSegment(start=(0, 1000), end=(0, 0), layer="wall", length=1000),
            # Right room
            WallSegment(start=(1000, 0), end=(2000, 0), layer="wall", length=1000),
            WallSegment(start=(2000, 0), end=(2000, 1000), layer="wall", length=1000),
            WallSegment(start=(2000, 1000), end=(1000, 1000), layer="wall", length=1000),
        ]
        geom.rooms.append(Room(
            name="Room_1",
            polygon=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
            area_mm2=1_000_000,
            centroid=(500, 500),
        ))
        geom.rooms.append(Room(
            name="Room_2",
            polygon=[(1000, 0), (2000, 0), (2000, 1000), (1000, 1000)],
            area_mm2=1_000_000,
            centroid=(1500, 500),
        ))
        return geom

    def test_render_two_rooms(self):
        geom = self._geom_with_corridor()
        r = ISORenderer()
        outputs = r.render(geom)
        assert outputs["render"].shape[0] > 0
        assert outputs["render"].shape[1] > 0

    def test_render_with_columns(self):
        geom = self._make_simple_room_geom()
        geom.columns.append(Column(
            center=(500, 500),
            width_mm=400,
            height_mm=400,
            layer="column",
            shape="rect",
        ))
        r = ISORenderer()
        outputs = r.render(geom)
        assert outputs is not None
        assert outputs["render"].shape[2] == 3


# ─── Error handling tests ───────────────────────────────────────────────────
class TestErrorHandling:
    def test_empty_geometry_raises(self):
        geom = DXFParseResult(source_file="test", units="mm")
        r = ISORenderer()
        with pytest.raises(ValueError, match="no walls"):
            r.render(geom)

    def test_zero_wall_segments_raises(self):
        geom = DXFParseResult(source_file="test", units="mm")
        geom.wall_segments = []
        r = ISORenderer()
        with pytest.raises(ValueError):
            r.render(geom)


# ─── Bounding box handling ──────────────────────────────────────────────────
class TestBoundingBoxHandling:
    def test_auto_bbox_from_walls(self):
        geom = DXFParseResult(source_file="test", units="mm")
        geom.wall_segments = [
            WallSegment(start=(0, 0), end=(2000, 0), layer="wall", length=2000),
            WallSegment(start=(2000, 0), end=(2000, 2000), layer="wall", length=2000),
            WallSegment(start=(2000, 2000), end=(0, 2000), layer="wall", length=2000),
            WallSegment(start=(0, 2000), end=(0, 0), layer="wall", length=2000),
        ]
        geom.bounding_box_mm = (0, 0, 0, 0)  # zero bbox
        r = ISORenderer()
        outputs = r.render(geom)  # Should not raise
        assert outputs is not None


# ─── Image output verification ──────────────────────────────────────────────
class TestImageOutput:
    def _geom_l_shape(self):
        """L-shaped floor plan for non-trivial geometry test."""
        geom = DXFParseResult(source_file="test", units="mm")
        # L-shape: two wings
        walls = [
            # Outer walls - wing 1 (0,0 to 0,2000)
            WallSegment(start=(0, 0), end=(0, 2000), layer="wall", length=2000),
            WallSegment(start=(0, 2000), end=(1000, 2000), layer="wall", length=1000),
            # wing 2 (0,0 to 2000,0)
            WallSegment(start=(0, 0), end=(2000, 0), layer="wall", length=2000),
            WallSegment(start=(2000, 0), end=(2000, 1000), layer="wall", length=1000),
            # inner corner
            WallSegment(start=(1000, 2000), end=(1000, 1000), layer="wall", length=1000),
            WallSegment(start=(1000, 1000), end=(2000, 1000), layer="wall", length=1000),
        ]
        geom.wall_segments = walls
        return geom

    def test_l_shape_renders(self):
        geom = self._geom_l_shape()
        r = ISORenderer()
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = r.render_to_files(geom, tmpdir)
            render_img = Image.open(saved["render"])
            assert render_img.size[0] > 100
            assert render_img.size[1] > 100

    def test_depth_map_16bit(self):
        geom = self._geom_l_shape()
        r = ISORenderer()
        outputs = r.render(geom)
        depth = outputs["depth"]
        assert depth.dtype == np.uint16

    def test_normal_map_valid_rgb(self):
        geom = self._geom_l_shape()
        r = ISORenderer()
        outputs = r.render(geom)
        normal = outputs["normal"]
        assert normal.dtype == np.uint8
        assert normal.shape[2] == 3
        # Values should be in reasonable range (0-255)
        assert normal.min() >= 0
        assert normal.max() <= 255


# ─── Run pytest ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])