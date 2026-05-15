"""
Task 1.4 tests: Structural Data Export
HMI Map Automation Pipeline

Run: python -m pytest tests/test_structural_exporter.py -v
"""

import json
import tempfile
import math
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser.structural_exporter import (
    StructuralExporter, Anchor, OcclusionZone,
    MASK_FLOOR, MASK_WALL, MASK_OPENING, MASK_COLUMN, MASK_SAFE
)
from parser.dxf_parser import DXFParseResult, WallSegment, Room, Column, Opening


# ─── Fixture ────────────────────────────────────────────────────────────────
def make_test_geometry():
    """A simple 1m x 1m room with one door opening."""
    geom = DXFParseResult(source_file="test.dxf", units="mm")
    geom.wall_segments = [
        WallSegment(start=(0, 0), end=(1000, 0), layer="wall", length=1000),
        WallSegment(start=(1000, 0), end=(1000, 1000), layer="wall", length=1000),
        WallSegment(start=(1000, 1000), end=(0, 1000), layer="wall", length=1000),
        WallSegment(start=(0, 1000), end=(0, 0), layer="wall", length=1000),
    ]
    geom.rooms.append(Room(
        name="Room_1", polygon=[(0, 0), (1000, 0), (1000, 1000), (0, 1000)],
        area_mm2=1_000_000, centroid=(500, 500),
    ))
    geom.columns.append(Column(
        center=(200, 200), width_mm=400, height_mm=400, layer="column", shape="rect",
    ))
    geom.openings.append(Opening(
        kind="DOOR", center=(500, 0), width=900, height=2100,
        rotation_deg=0, layer="door",
        bounding_box=(50, -500, 950, 500),
    ))
    return geom


# ─── Mask value constants tests ─────────────────────────────────────────────
class TestMaskValues:
    def test_mask_floor_is_85(self):
        assert MASK_FLOOR == 85

    def test_mask_wall_is_0(self):
        assert MASK_WALL == 0

    def test_mask_column_is_200(self):
        assert MASK_COLUMN == 200

    def test_mask_opening_is_128(self):
        assert MASK_OPENING == 128

    def test_all_masks_distinct(self):
        masks = {MASK_FLOOR, MASK_WALL, MASK_OPENING, MASK_COLUMN, MASK_SAFE}
        assert len(masks) == 5


# ─── Anchor dataclass tests ─────────────────────────────────────────────────
class TestAnchor:
    def test_anchor_to_dict(self):
        a = Anchor(
            id="anchor_0", anchor_type="room_center",
            room_id="room_1", room_name="Room_1",
            position_mm=(500.0, 500.0),
            facing_wall="north",
            occlusion_zone_mm=(0, 0, 1000, 1000),
            nearby_openings=["opening_0"],
            safe_distance_mm=500,
        )
        d = a.to_dict()
        assert d["id"] == "anchor_0"
        assert d["anchor_type"] == "room_center"
        assert d["position_mm"] == (500.0, 500.0)
        assert d["safe_distance_mm"] == 500


# ─── OcclusionZone tests ────────────────────────────────────────────────────
class TestOcclusionZone:
    def test_occlusion_zone_to_dict(self):
        z = OcclusionZone(
            id="occl_0", zone_type="column",
            bbox_mm=(100, 100, 300, 300),
            blocked=True,
        )
        d = z.to_dict()
        assert d["zone_type"] == "column"
        assert d["bbox_mm"] == (100, 100, 300, 300)
        assert d["blocked"] is True


# ─── StructuralExporter init tests ──────────────────────────────────────────
class TestStructuralExporterInit:
    def test_init_with_geometry(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)
        assert exporter.geometry is not None
        assert exporter.geometry.source_file == "test.dxf"

    def test_init_with_config(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom, config={"dpi": 150})
        assert exporter.config["dpi"] == 150


# ─── geometry.json export tests ─────────────────────────────────────────────
class TestGeometryJsonExport:
    def test_geometry_json_structure(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_geometry_json(str(Path(tmpdir) / "geometry.json"))
            with open(path, "r") as f:
                geo = json.load(f)

            assert geo["type"] == "FeatureCollection"
            assert "metadata" in geo
            assert "walls" in geo
            assert "rooms" in geo
            assert "openings" in geo
            assert "columns" in geo

    def test_walls_exported_correctly(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_geometry_json(str(Path(tmpdir) / "geometry.json"))
            with open(path, "r") as f:
                geo = json.load(f)

            assert len(geo["walls"]) == 4  # 4 wall segments
            assert geo["walls"][0]["geometry"]["type"] == "LineString"

    def test_rooms_exported_correctly(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_geometry_json(str(Path(tmpdir) / "geometry.json"))
            with open(path, "r") as f:
                geo = json.load(f)

            assert len(geo["rooms"]) == 1
            assert geo["rooms"][0]["properties"]["name"] == "Room_1"
            assert geo["rooms"][0]["properties"]["area_mm2"] == 1_000_000

    def test_openings_exported_correctly(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_geometry_json(str(Path(tmpdir) / "geometry.json"))
            with open(path, "r") as f:
                geo = json.load(f)

            assert len(geo["openings"]) == 1
            assert geo["openings"][0]["properties"]["kind"] == "DOOR"
            assert geo["openings"][0]["properties"]["width_mm"] == 900

    def test_columns_exported_correctly(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_geometry_json(str(Path(tmpdir) / "geometry.json"))
            with open(path, "r") as f:
                geo = json.load(f)

            assert len(geo["columns"]) == 1
            assert geo["columns"][0]["properties"]["shape"] == "rect"


# ─── anchors.json export tests ─────────────────────────────────────────────
class TestAnchorsJsonExport:
    def test_anchors_json_structure(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_anchors_json(str(Path(tmpdir) / "anchors.json"))
            with open(path, "r") as f:
                data = json.load(f)

            assert "metadata" in data
            assert "anchors" in data
            assert "occlusion_zones" in data

    def test_room_center_anchor_created(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_anchors_json(str(Path(tmpdir) / "anchors.json"))
            with open(path, "r") as f:
                data = json.load(f)

            anchor_types = [a["anchor_type"] for a in data["anchors"]]
            assert "room_center" in anchor_types

    def test_column_anchor_created(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_anchors_json(str(Path(tmpdir) / "anchors.json"))
            with open(path, "r") as f:
                data = json.load(f)

            anchor_types = [a["anchor_type"] for a in data["anchors"]]
            assert "column" in anchor_types

    def test_occlusion_zone_for_column(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_anchors_json(str(Path(tmpdir) / "anchors.json"))
            with open(path, "r") as f:
                data = json.load(f)

            occl_zones = data["occlusion_zones"]
            assert len(occl_zones) >= 1
            assert any(z["zone_type"] == "column" for z in occl_zones)

    def test_metadata_has_total_counts(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_anchors_json(str(Path(tmpdir) / "anchors.json"))
            with open(path, "r") as f:
                data = json.load(f)

            assert data["metadata"]["total_anchors"] > 0
            assert data["metadata"]["safe_distance_mm"] == 500


# ─── masks.png export tests ────────────────────────────────────────────────
class TestMasksPngExport:
    def test_masks_png_created(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_masks_png(str(Path(tmpdir) / "masks.png"))
            assert Path(path).exists()
            assert Path(path).stat().st_size > 100

    def test_masks_png_is_valid_image(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_masks_png(str(Path(tmpdir) / "masks.png"))
            img = Image.open(path)
            assert img.mode == "L"
            assert img.size[0] > 100
            assert img.size[1] > 100

    def test_mask_values_in_expected_range(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_masks_png(str(Path(tmpdir) / "masks.png"))
            img = Image.open(path)
            arr = np.array(img)
            unique_vals = np.unique(arr)
            # All values should be valid mask values (0, 85, 128, 170, 200)
            assert all(v in [0, 85, 128, 170, 200] for v in unique_vals)

    def test_masks_png_has_wall_pixels(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_masks_png(str(Path(tmpdir) / "masks.png"))
            img = Image.open(path)
            arr = np.array(img)
            assert MASK_WALL in arr  # wall pixels should be present


# ─── metadata.json export tests ────────────────────────────────────────────
class TestMetadataJsonExport:
    def test_metadata_json_structure(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_metadata_json(str(Path(tmpdir) / "metadata.json"))
            with open(path, "r") as f:
                meta = json.load(f)

            assert "pipeline_version" in meta
            assert "source_file" in meta
            assert "bounding_box_mm" in meta
            assert "geometry" in meta

    def test_geometry_counts_correct(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter.export_metadata_json(str(Path(tmpdir) / "metadata.json"))
            with open(path, "r") as f:
                meta = json.load(f)

            assert meta["geometry"]["total_walls"] == 4
            assert meta["geometry"]["total_rooms"] == 1
            assert meta["geometry"]["total_columns"] == 1
            assert meta["geometry"]["total_openings"] == 1


# ─── export_all tests ───────────────────────────────────────────────────────
class TestExportAll:
    def test_export_all_produces_4_files(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = exporter.export_all(tmpdir)

            assert len(results) == 4
            assert "geometry" in results
            assert "anchors" in results
            assert "masks" in results
            assert "metadata" in results

            for name, path in results.items():
                assert Path(path).exists()

    def test_all_files_valid_json_except_masks(self):
        geom = make_test_geometry()
        exporter = StructuralExporter(geom)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = exporter.export_all(tmpdir)

            for name, path in results.items():
                if name != "masks":
                    with open(path, "r") as f:
                        json.load(f)  # should not raise


# ─── Run pytest ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])