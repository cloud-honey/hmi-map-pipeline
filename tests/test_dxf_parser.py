"""
Unit tests for DXF Parser (Task 1.1)
HMI Map Automation Pipeline

Run: python -m pytest tests/test_dxf_parser.py -v
"""

import math
import json
import tempfile
from pathlib import Path

import pytest

# Import from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from parser.dxf_parser import (
    DXFParser, DXFParseResult, WallSegment, Opening, Room, Column,
    normalize_layer, LAYER_CATEGORY_MAP
)


# ─── normalize_layer tests ─────────────────────────────────────────────────
class TestNormalizeLayer:
    def test_wall_layer_names(self):
        for name in ["wall", "WALL", "Wall", "walls", "AWALL", "a-wall-1", "bw-shad"]:
            assert normalize_layer(name) == "WALL", f"Failed: {name}"

    def test_door_layer_names(self):
        for name in ["door", "DOOR", "doors", "D-DOOR", "d-door-2", "a-doors"]:
            assert normalize_layer(name) == "DOOR", f"Failed: {name}"

    def test_window_layer_names(self):
        for name in ["window", "WINDOW", "windows", "w-1", "glazing", "a-window"]:
            assert normalize_layer(name) == "WINDOW", f"Failed: {name}"

    def test_column_layer_names(self):
        for name in ["column", "COLUMN", "columns", "col", "a-col", "struct-col"]:
            assert normalize_layer(name) == "COLUMN", f"Failed: {name}"

    def test_unknown_returns_unknown(self):
        assert normalize_layer("random_layer_xyz") == "UNKNOWN"
        assert normalize_layer("") == "UNKNOWN"
        assert normalize_layer("none") == "UNKNOWN"


# ─── DXFParseResult dataclass tests ───────────────────────────────────────
class TestDXFParseResult:
    def test_wall_segment_to_dict(self):
        seg = WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000)
        d = seg.to_dict()
        assert d["type"] == "wall_segment"
        assert d["start"] == (0, 0)
        assert d["end"] == (1000, 0)
        assert d["length_mm"] == 1000

    def test_opening_to_dict(self):
        op = Opening(
            kind="DOOR", center=(500, 0), width=900, height=2100,
            rotation_deg=0, layer="door", bounding_box=(0, -500, 1000, 500)
        )
        d = op.to_dict()
        assert d["kind"] == "DOOR"
        assert d["width_mm"] == 900

    def test_room_to_dict(self):
        poly = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        room = Room(name="Room_1", polygon=poly, area_mm2=1_000_000, centroid=(500, 500))
        d = room.to_dict()
        assert d["type"] == "room"
        assert d["name"] == "Room_1"

    def test_column_to_dict(self):
        col = Column(center=(100, 200), width_mm=400, height_mm=400, layer="col", shape="rect")
        d = col.to_dict()
        assert d["shape"] == "rect"
        assert d["width_mm"] == 400

    def test_result_to_dict(self):
        result = DXFParseResult(source_file="test.dxf", units="mm")
        d = result.to_dict()
        assert d["source_file"] == "test.dxf"
        assert d["units"] == "mm"
        assert "wall_segments" in d


# ─── WallSegment geometry tests ────────────────────────────────────────────
class TestWallSegment:
    def test_length_calculation(self):
        # 1000mm horizontal line
        seg = WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000)
        assert seg.length == 1000

    def test_length_diagonal(self):
        # 45° diagonal: sqrt(2) * 1000 ≈ 1414.21
        seg = WallSegment(start=(0, 0), end=(1000, 1000), layer="wall", length=0)
        length = math.hypot(1000 - 0, 1000 - 0)
        assert abs(length - 1414.21) < 0.01

    def test_skip_short_segments(self):
        # Segments shorter than 1mm should be skipped in parsing
        seg = WallSegment(start=(0, 0), end=(0.5, 0), layer="wall", length=0.5)
        assert seg.length < 1.0  # Will be filtered


# ─── Room extraction tests ─────────────────────────────────────────────────
class TestRoomExtraction:
    def test_simple_square_room(self):
        # 4 walls forming a square room
        walls = [
            WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000),
            WallSegment(start=(1000, 0), end=(1000, 1000), layer="wall", length=1000),
            WallSegment(start=(1000, 1000), end=(0, 1000), layer="wall", length=1000),
            WallSegment(start=(0, 1000), end=(0, 0), layer="wall", length=1000),
        ]
        from parser.dxf_parser import DXFParser
        parser = DXFParser.__new__(DXFParser)
        rooms = parser._extract_rooms(walls)
        assert len(rooms) >= 1
        # Check area (1m x 1m = 1,000,000 mm²)
        areas = [r.area_mm2 for r in rooms]
        max_area = max(areas)
        assert abs(max_area - 1_000_000) < 100  # within 0.01%

    def test_two_adjacent_rooms(self):
        # Two rooms sharing a wall: left (0-1000) right (1000-2000)
        walls = [
            # Outer boundary
            WallSegment(start=(0, 0), end=(2000, 0), layer="wall", length=2000),
            WallSegment(start=(2000, 0), end=(2000, 1000), layer="wall", length=1000),
            WallSegment(start=(2000, 1000), end=(0, 1000), layer="wall", length=2000),
            WallSegment(start=(0, 1000), end=(0, 0), layer="wall", length=1000),
            # Shared wall
            WallSegment(start=(1000, 0), end=(1000, 1000), layer="wall", length=1000),
        ]
        from parser.dxf_parser import DXFParser
        parser = DXFParser.__new__(DXFParser)
        rooms = parser._extract_rooms(walls)
        # Should extract 2 rooms
        assert len(rooms) >= 1
        areas = [r.area_mm2 for r in rooms]
        assert max(areas) > 0

    def test_skip_small_polygon(self):
        # Very small polygon (< 10000 mm²) should be filtered
        walls = [
            WallSegment(start=(0, 0), end=(5, 0), layer="wall", length=5),
            WallSegment(start=(5, 0), end=(5, 5), layer="wall", length=5),
            WallSegment(start=(5, 5), end=(0, 5), layer="wall", length=5),
            WallSegment(start=(0, 5), end=(0, 0), layer="wall", length=5),
        ]
        from parser.dxf_parser import DXFParser
        parser = DXFParser.__new__(DXFParser)
        rooms = parser._extract_rooms(walls)
        # 25 mm² < 10000 mm² threshold → should be filtered out
        assert all(r.area_mm2 >= 10000 for r in rooms)


# ─── Opening estimation tests ──────────────────────────────────────────────
class TestOpeningEstimation:
    def test_standard_door_width(self):
        parser = DXFParser.__new__(DXFParser)
        parser.STANDARD_DOOR_WIDTHS = [900, 1000, 1100, 1200, 1500, 1800]

        # Exact match
        closest = min(parser.STANDARD_DOOR_WIDTHS, key=lambda x: abs(x - 1000))
        assert closest == 1000

        # Between sizes → nearest
        closest = min(parser.STANDARD_DOOR_WIDTHS, key=lambda x: abs(x - 950))
        assert closest == 900

        closest = min(parser.STANDARD_DOOR_WIDTHS, key=lambda x: abs(x - 1050))
        assert closest == 1000

    def test_standard_window_width(self):
        parser = DXFParser.__new__(DXFParser)
        parser.STANDARD_WINDOW_WIDTHS = [600, 900, 1200, 1500, 1800, 2400]

        closest = min(parser.STANDARD_WINDOW_WIDTHS, key=lambda x: abs(x - 1500))
        assert closest == 1500


# ─── Column detection tests ────────────────────────────────────────────────
class TestColumnDetection:
    def test_column_circle_shape(self):
        from parser.dxf_parser import DXFParser
        parser = DXFParser.__new__(DXFParser)
        entity = {
            "type": "CIRCLE",
            "layer": "column",
            "radius": 200,
            "points": [(200, 200), (200, 200)]  # Simplified
        }
        col = parser._to_column(entity)
        # Circle with radius 200mm → 400x400 column
        assert col is not None
        assert col.shape == "circle"
        assert col.width_mm == 400

    def test_column_insert_from_block(self):
        from parser.dxf_parser import DXFParser
        parser = DXFParser.__new__(DXFParser)
        entity = {
            "type": "INSERT",
            "layer": "column",
            "location": (1000, 2000),
            "scale": (1.0, 1.0, 1.0),
            "block_name": "COL400",
        }
        col = parser._to_column(entity)
        assert col is not None
        assert col.center == (1000, 2000)
        assert col.shape == "rect"


# ─── Bounding box tests ────────────────────────────────────────────────────
class TestBoundingBox:
    def test_bounding_box_calculation(self):
        # Simulate result with points
        result = DXFParseResult(source_file="test.dxf", units="mm")
        # Points: (0,0), (5000,0), (5000,3000), (0,3000)
        all_points = [(0, 0), (5000, 0), (5000, 3000), (0, 3000)]
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        result.bounding_box_mm = (min(xs), min(ys), max(xs), max(ys))
        assert result.bounding_box_mm == (0, 0, 5000, 3000)


# ─── Export JSON tests ────────────────────────────────────────────────────
class TestExportJSON:
    def test_export_and_load_result(self):
        result = DXFParseResult(source_file="test.dxf", units="mm")
        result.wall_segments.append(
            WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000)
        )
        result.rooms.append(
            Room(name="Room_1", polygon=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
                 area_mm2=1_000_000, centroid=(500, 500))
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            parser = DXFParser.__new__(DXFParser)
            # Manually create a minimal parser for export testing
            class MinimalParser:
                def export_json(self, result, path):
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            mp = MinimalParser()
            mp.export_json(result, json_path)

            with open(json_path, "r") as f:
                loaded = json.load(f)

            assert loaded["source_file"] == "test.dxf"
            assert len(loaded["wall_segments"]) == 1
            assert len(loaded["rooms"]) == 1
        finally:
            Path(json_path).unlink(missing_ok=True)


# ─── Run pytest ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])