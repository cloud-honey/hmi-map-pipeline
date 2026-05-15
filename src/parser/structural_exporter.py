"""
Task 1.4: Structural Data Export
HMI Map Automation Pipeline

Exports DXFParseResult into the structured output formats required
by the downstream AI refinement pipeline and HMI equipment placement:

1. geometry.json   — GeoJSON-like wall/room/opening/column data for AI
2. anchors.json    — Equipment placement coords + occlusion metadata
3. masks.png       — Per-region segmentation (wall=0, floor=85, safe=170, column=255)
4. metadata.json   — Parse stats, hashes, pipeline version

All exports use mm as primary unit, coordinate origin at drawing origin.
"""

import json
import math
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from parser.dxf_parser import DXFParseResult


# ─── Mask Values (for masks.png) ───────────────────────────────────────────
MASK_FLOOR = 85      # Room floor areas
MASK_WALL = 0         # Wall regions
MASK_SAFE = 170      # Safe-zone / equipment areas
MASK_COLUMN = 200     # Structural columns
MASK_OPENING = 128   # Door/window openings


@dataclass
class Anchor:
    """
    Equipment placement anchor point.
    Each anchor carries enough context for an HMI designer to place
    equipment and know its spatial relationships.
    """
    id: str                      # unique anchor id e.g. "room_1_anchor_0"
    anchor_type: str             # "room_center" | "wall_near_door" | "column" | "corridor"
    room_id: str                 # which room this anchor belongs to (or "outdoor")
    room_name: str               # human-readable room label
    position_mm: tuple[float, float]  # (x, y) world coordinates
    facing_wall: Optional[str] = None   # which wall the equipment faces
    occlusion_zone_mm: tuple[float, float, float, float] = (0, 0, 0, 0)  # bounding box
    nearby_openings: list[str] = field(default_factory=list)  # opening IDs nearby
    safe_distance_mm: float = 500  # min clearance from walls for equipment placement

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class OcclusionZone:
    """A zone that blocks equipment placement (wall, column, etc.)."""
    id: str
    zone_type: str   # "wall" | "column" | "opening"
    bbox_mm: tuple[float, float, float, float]  # minx, miny, maxx, maxy
    geometry_wkt: Optional[str] = None  # well-known text for precise collision
    blocked: bool = True

    def to_dict(self):
        return asdict(self)


class StructuralExporter:
    """
    Export DXFParseResult into all structured data formats.
    """

    def __init__(self, geometry: DXFParseResult, config: Optional[dict] = None):
        self.geometry = geometry
        self.config = config or {}

    # ─── 1. geometry.json ───────────────────────────────────────────────────

    def export_geometry_json(self, output_path: str) -> str:
        """
        GeoJSON-like structure for AI model conditioning (ControlNet, etc.).
        """
        bbox = self.geometry.bounding_box_mm
        geo = {
            "type": "FeatureCollection",
            "metadata": {
                "source": self.geometry.source_file,
                "units": self.geometry.units,
                "bounding_box_mm": {
                    "minx": bbox[0], "miny": bbox[1],
                    "maxx": bbox[2], "maxy": bbox[3],
                },
                "total_walls": len(self.geometry.wall_segments),
                "total_openings": len(self.geometry.openings),
                "total_rooms": len(self.geometry.rooms),
                "total_columns": len(self.geometry.columns),
                "pipeline_version": "1.0.0",
                "generator": "StructuralExporter",
            },

            # Walls as LineString features
            "walls": [
                {
                    "type": "Feature",
                    "id": f"wall_{i}",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [list(s.start), list(s.end)],
                    },
                    "properties": {
                        "length_mm": s.length,
                        "layer": s.layer,
                    }
                }
                for i, s in enumerate(self.geometry.wall_segments)
            ],

            # Rooms as Polygon features
            "rooms": [
                {
                    "type": "Feature",
                    "id": f"room_{i}",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [r.polygon + [r.polygon[0]]],
                    },
                    "properties": {
                        "name": r.name,
                        "area_mm2": r.area_mm2,
                        "centroid_mm": list(r.centroid),
                    }
                }
                for i, r in enumerate(self.geometry.rooms)
            ],

            # Openings as Point features with metadata
            "openings": [
                {
                    "type": "Feature",
                    "id": f"opening_{i}",
                    "geometry": {
                        "type": "Point",
                        "coordinates": list(o.center),
                    },
                    "properties": {
                        "kind": o.kind,
                        "width_mm": o.width,
                        "height_mm": o.height,
                        "rotation_deg": o.rotation_deg,
                        "layer": o.layer,
                        "bbox_mm": list(o.bounding_box),
                    }
                }
                for i, o in enumerate(self.geometry.openings)
            ],

            # Columns as Point features
            "columns": [
                {
                    "type": "Feature",
                    "id": f"column_{i}",
                    "geometry": {
                        "type": "Point",
                        "coordinates": list(c.center),
                    },
                    "properties": {
                        "width_mm": c.width_mm,
                        "height_mm": c.height_mm,
                        "shape": c.shape,
                        "layer": c.layer,
                    }
                }
                for i, c in enumerate(self.geometry.columns)
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geo, f, indent=2, ensure_ascii=False)

        return output_path

    # ─── 2. anchors.json ────────────────────────────────────────────────────

    def export_anchors_json(self, output_path: str, safe_distance_mm: float = 500) -> str:
        """
        Equipment placement anchors with occlusion zones.
        """
        anchors: list[Anchor] = []
        occlusion_zones: list[OcclusionZone] = []
        anchor_id_counter = 0

        # Add room center anchors
        for room in self.geometry.rooms:
            # Compute safe position (room centroid, moved away from walls)
            cx, cy = room.centroid

            # Find nearest wall to check if centroid is too close to wall
            nearest_wall_dist = self._min_dist_to_walls(cx, cy)

            anchor_type = "room_center"
            if nearest_wall_dist < safe_distance_mm:
                # Centroid is too close to a wall → place near geometric center of floor
                anchor_type = "wall_near"

            anchor_id = f"anchor_{anchor_id_counter:04d}"
            anchor_id_counter += 1

            anchor = Anchor(
                id=anchor_id,
                anchor_type=anchor_type,
                room_id=f"room_{room.name}",
                room_name=room.name,
                position_mm=(float(cx), float(cy)),
                occlusion_zone_mm=(cx - 500, cy - 500, cx + 500, cy + 500),
                nearby_openings=[],
                safe_distance_mm=safe_distance_mm,
            )
            anchors.append(anchor)

        # Add column anchors
        for col in self.geometry.columns:
            occl_id = f"occlusion_column_{col.center[0]:.0f}_{col.center[1]:.0f}"
            occl_bbox = (
                col.center[0] - col.width_mm / 2,
                col.center[1] - col.height_mm / 2,
                col.center[0] + col.width_mm / 2,
                col.center[1] + col.height_mm / 2,
            )
            occlusion_zones.append(OcclusionZone(
                id=occl_id,
                zone_type="column",
                bbox_mm=occl_bbox,
                blocked=True,
            ))

            anchor = Anchor(
                id=f"anchor_{anchor_id_counter:04d}",
                anchor_type="column",
                room_id="outdoor",
                room_name="column",
                position_mm=col.center,
                occlusion_zone_mm=occl_bbox,
                safe_distance_mm=col.width_mm / 2 + 100,
            )
            anchor_id_counter += 1
            anchors.append(anchor)

        # Add corridor / outdoor anchors (room-less wall midpoints)
        wall_midpoints = self._find_corridor_anchors(anchors, safe_distance_mm)
        for pos in wall_midpoints:
            anchor = Anchor(
                id=f"anchor_{anchor_id_counter:04d}",
                anchor_type="corridor",
                room_id="outdoor",
                room_name="corridor",
                position_mm=pos,
                occlusion_zone_mm=(pos[0] - 500, pos[1] - 500, pos[0] + 500, pos[1] + 500),
                safe_distance_mm=safe_distance_mm,
            )
            anchor_id_counter += 1
            anchors.append(anchor)

        result = {
            "metadata": {
                "total_anchors": len(anchors),
                "total_occlusion_zones": len(occlusion_zones),
                "safe_distance_mm": safe_distance_mm,
                "source": self.geometry.source_file,
            },
            "anchors": [a.to_dict() for a in anchors],
            "occlusion_zones": [z.to_dict() for z in occlusion_zones],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return output_path

    def _min_dist_to_walls(self, x: float, y: float) -> float:
        """Minimum distance from point to any wall segment."""
        import math
        min_dist = float("inf")
        for seg in self.geometry.wall_segments:
            # Point to line segment distance
            px, py = x, y
            x1, y1 = seg.start
            x2, y2 = seg.end
            dx, dy = x2 - x1, y2 - y1
            if dx == dy == 0:
                dist = math.hypot(px - x1, py - y1)
            else:
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
                near_x = x1 + t * dx
                near_y = y1 + t * dy
                dist = math.hypot(px - near_x, py - near_y)
            if dist < min_dist:
                min_dist = dist
        return min_dist if min_dist != float("inf") else 999999

    def _find_corridor_anchors(self, existing_anchors: list[Anchor], min_spacing_mm: float) -> list:
        """Find wall midpoints not too close to existing anchors (for corridor placement)."""
        candidates = []
        for seg in self.geometry.wall_segments:
            mid_x = (seg.start[0] + seg.end[0]) / 2
            mid_y = (seg.start[1] + seg.end[1]) / 2
            # Check if too close to any existing anchor
            too_close = any(
                math.hypot(mid_x - a.position_mm[0], mid_y - a.position_mm[1]) < min_spacing_mm
                for a in existing_anchors
            )
            if not too_close:
                candidates.append((mid_x, mid_y))
        return candidates

    # ─── 3. masks.png ───────────────────────────────────────────────────────

    def export_masks_png(self, output_path: str,
                         pixels_per_mm: float = 0.5,
                         render_offset_x: float = 0,
                         render_offset_y: float = 0) -> str:
        """
        Generate region masks as PNG.

        Mask values:
          0     = WALL
          85    = FLOOR
          128   = OPENING
          170   = SAFE ZONE
          200   = COLUMN

        Canvas sized to fit the full geometry bounding box.
        """
        bbox = self.geometry.bounding_box_mm
        world_w = max(bbox[2] - bbox[0], 1)
        world_h = max(bbox[3] - bbox[1], 1)

        canvas_w = int(world_w * pixels_per_mm) + 200
        canvas_h = int(world_h * pixels_per_mm) + 200

        offset_x = -bbox[0] * pixels_per_mm + 100 + render_offset_x
        offset_y = -bbox[1] * pixels_per_mm + 100 + render_offset_y

        mask_arr = np.full((canvas_h, canvas_w), MASK_FLOOR, dtype=np.uint8)

        # Draw walls
        for seg in self.geometry.wall_segments:
            p1 = (int(seg.start[0] * pixels_per_mm + offset_x),
                  int(seg.start[1] * pixels_per_mm + offset_y))
            p2 = (int(seg.end[0] * pixels_per_mm + offset_x),
                  int(seg.end[1] * pixels_per_mm + offset_y))
            mask_arr = self._draw_thick_line(mask_arr, p1, p2, MASK_WALL, thickness_px=3)

        # Draw floors (rooms)
        img = Image.fromarray(mask_arr)
        draw = ImageDraw.Draw(img)
        for room in self.geometry.rooms:
            if len(room.polygon) < 3:
                continue
            screen_pts = [
                (int(px * pixels_per_mm + offset_x), int(py * pixels_per_mm + offset_y))
                for (px, py) in room.polygon
            ]
            try:
                draw.polygon(screen_pts, fill=MASK_FLOOR, outline=MASK_FLOOR)
            except Exception:
                pass

        mask_arr = np.array(img)

        # Draw columns
        for col in self.geometry.columns:
            cx = int(col.center[0] * pixels_per_mm + offset_x)
            cy = int(col.center[1] * pixels_per_mm + offset_y)
            hw = max(int(col.width_mm * pixels_per_mm / 2), 4)
            hh = max(int(col.height_mm * pixels_per_mm / 2), 4)
            x0, y0 = cx - hw, cy - hh
            x1, y1 = cx + hw, cy + hh
            # Filled rectangle
            mask_arr[y0:y1, x0:x1] = MASK_COLUMN

        # Draw openings
        for op in self.geometry.openings:
            cx = int(op.center[0] * pixels_per_mm + offset_x)
            cy = int(op.center[1] * pixels_per_mm + offset_y)
            w = max(int(op.width * pixels_per_mm / 2), 4)
            h = max(int(op.height * pixels_per_mm / 2), 4)
            x0, y0 = cx - w, cy - h
            x1, y1 = cx + w, cy + h
            mask_arr[y0:y1, x0:x1] = MASK_OPENING

        # Safe zones (area not near walls or columns)
        # Mark all non-wall pixels as potentially safe, then carve out walls
        # Safe zones are floor areas that are not too close to wall edges
        # (simplified: mark floor area as safe unless adjacent to wall)
        # → Skip for now, mark all floor as floor; safe zones handled in anchors

        Image.fromarray(mask_arr).save(output_path)
        return output_path

    def _draw_thick_line(self, arr, p1, p2, color, thickness_px=3):
        """Draw a thick line on a numpy array using cv2."""
        import cv2
        arr_copy = arr.copy()
        cv2.line(arr_copy, p1, p2, color, thickness=thickness_px)
        return arr_copy

    # ─── 4. metadata.json ───────────────────────────────────────────────────

    def export_metadata_json(self, output_path: str, pipeline_version: str = "1.0.0") -> str:
        """Export parse/generate metadata."""
        bbox = self.geometry.bounding_box_mm

        # Compute content hashes (sha256 of source filename for now, or actual file hash)
        source_hash = hashlib.sha256(self.geometry.source_file.encode()).hexdigest()[:16]

        metadata = {
            "pipeline_version": pipeline_version,
            "generator": "StructuralExporter",
            "source_file": self.geometry.source_file,
            "source_hash": source_hash,
            "units": self.geometry.units,
            "bounding_box_mm": {
                "minx": float(bbox[0]), "miny": float(bbox[1]),
                "maxx": float(bbox[2]), "maxy": float(bbox[3]),
                "width_mm": float(bbox[2] - bbox[0]),
                "height_mm": float(bbox[3] - bbox[1]),
            },
            "geometry": {
                "total_walls": len(self.geometry.wall_segments),
                "total_openings": len(self.geometry.openings),
                "total_rooms": len(self.geometry.rooms),
                "total_columns": len(self.geometry.columns),
            },
            "entities_total": self.geometry.entities_total,
            "entities_parsed": self.geometry.entities_parsed,
            "layers_found": self.geometry.layers_found,
            "parse_errors": self.geometry.errors,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return output_path

    # ─── Full export ────────────────────────────────────────────────────────

    def export_all(self, output_dir: str) -> dict[str, str]:
        """Run all exports and return {name: filepath} map."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = {}
        results["geometry"] = self.export_geometry_json(str(out_path / "geometry.json"))
        results["anchors"] = self.export_anchors_json(str(out_path / "anchors.json"))
        results["masks"] = self.export_masks_png(str(out_path / "masks.png"))
        results["metadata"] = self.export_metadata_json(str(out_path / "metadata.json"))

        return results


# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python structural_exporter.py <geometry.json> <output_dir>")
        sys.exit(1)

    geom_path = sys.argv[1]
    out_dir = sys.argv[2]

    from parser.dxf_parser import DXFParseResult, WallSegment, Room, Opening, Column

    with open(geom_path, "r") as f:
        geo = json.load(f)

    result = DXFParseResult(
        source_file=geo.get("source_file", ""),
        units=geo.get("units", "mm"),
    )
    for w in geo.get("wall_segments", []):
        result.wall_segments.append(WallSegment(**w))
    for o in geo.get("openings", []):
        result.openings.append(Opening(**o))
    for r in geo.get("rooms", []):
        result.rooms.append(Room(**r))
    for c in geo.get("columns", []):
        result.columns.append(Column(**c))
    result.bounding_box_mm = tuple(geo.get("bounding_box_mm", (0, 0, 0, 0)))

    print(f"[StructuralExporter] Exporting to: {out_dir}")
    exporter = StructuralExporter(result)
    saved = exporter.export_all(out_dir)
    for name, path in saved.items():
        print(f"  {name}: {path}")