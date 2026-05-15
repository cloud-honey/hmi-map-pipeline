"""
Task 1.1: DXF/DWG Parser + Layer Normalization
HMI Map Automation Pipeline

Parses DXF/DWG files via ezdxf, extracts structural elements,
normalizes layers, and outputs geometry as shapely objects + JSON.
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import ezdxf
from ezdxf.math import Vec3 as Vector
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point, shape
)
from shapely.ops import unary_union, linemerge
import numpy as np

# ─── Standard Layer Mapping ───────────────────────────────────────────────
# AutoCAD/DXF common layer name patterns → standardized categories
LAYER_CATEGORY_MAP = {
    # Wall
    "wall": "WALL",
    "walls": "WALL",
    "w-line": "WALL",
    "w-lines": "WALL",
    "a-wall": "WALL",
    "a-walls": "WALL",
    "a-wall-*": "WALL",
    "s-wall": "WALL",
    "bw": "WALL",
    "bw-*": "WALL",
    "aw": "WALL",
    "aw-*": "WALL",
    # Door
    "door": "DOOR",
    "doors": "DOOR",
    "d-door": "DOOR",
    "d-doors": "DOOR",
    "a-door": "DOOR",
    "a-doors": "DOOR",
    "d-*": "DOOR",
    # Window
    "window": "WINDOW",
    "windows": "WINDOW",
    "w-*": "WINDOW",
    "glazing": "WINDOW",
    "a-window": "WINDOW",
    "a-windows": "WINDOW",
    # Column
    "column": "COLUMN",
    "columns": "COLUMN",
    "col": "COLUMN",
    "cols": "COLUMN",
    "a-col": "COLUMN",
    "a-cols": "COLUMN",
    "a-column": "COLUMN",
    "a-columns": "COLUMN",
    "struct-col": "COLUMN",
    # Floor / slab
    "floor": "FLOOR",
    "slab": "FLOOR",
    "slabs": "FLOOR",
    "a-slab": "FLOOR",
    "floor-*": "FLOOR",
    # Hatch / fill
    "hatch": "HATCH",
    "hatching": "HATCH",
    "fill": "HATCH",
    "pattern": "HATCH",
    "a-hatch": "HATCH",
    # Dimension
    "dimension": "DIMENSION",
    "dim": "DIMENSION",
    "annotate": "DIMENSION",
    # Text
    "text": "TEXT",
    "mtext": "TEXT",
    "label": "TEXT",
    "title": "TEXT",
    # Outline
    "outline": "OUTLINE",
    "border": "OUTLINE",
    "boundary": "OUTLINE",
    "room-outline": "OUTLINE",
}


def normalize_layer(name: str) -> str:
    """Map DXF layer name to standardized category."""
    if not name:
        return "UNKNOWN"
    n = name.lower().strip()
    if n in LAYER_CATEGORY_MAP:
        return LAYER_CATEGORY_MAP[n]
    # Pattern match (e.g. "a-wall-1" → WALL)
    for pattern, cat in LAYER_CATEGORY_MAP.items():
        if pattern.endswith("-*"):
            prefix = pattern[:-2]
            if n.startswith(prefix):
                return cat
    return "UNKNOWN"


@dataclass
class WallSegment:
    """A single wall line segment."""
    start: tuple[float, float]  # (x, y)
    end: tuple[float, float]
    layer: str
    length: float

    def to_dict(self):
        return {
            "type": "wall_segment",
            "start": self.start,
            "end": self.end,
            "length_mm": self.length,
            "layer": self.layer,
        }


@dataclass
class Opening:
    """Door or Window entity."""
    kind: str          # "DOOR" or "WINDOW"
    center: tuple[float, float]
    width: float
    height: float
    rotation_deg: float
    layer: str
    bounding_box: tuple[float, float, float, float]  # minx, miny, maxx, maxy

    def to_dict(self):
        return {
            "type": "opening",
            "kind": self.kind,
            "center": self.center,
            "width_mm": self.width,
            "height_mm": self.height,
            "rotation_deg": self.rotation_deg,
            "layer": self.layer,
            "bbox": self.bounding_box,
        }


@dataclass
class Room:
    """A closed polygon room."""
    name: str
    polygon: list[tuple[float, float]]
    area_mm2: float
    centroid: tuple[float, float]

    def to_dict(self):
        return {
            "type": "room",
            "name": self.name,
            "polygon": self.polygon,
            "area_mm2": self.area_mm2,
            "centroid": self.centroid,
        }


@dataclass
class Column:
    """A structural column (point or small polygon)."""
    center: tuple[float, float]
    width_mm: float
    height_mm: float
    layer: str
    shape: str  # "circle" or "rect"

    def to_dict(self):
        return {
            "type": "column",
            "center": self.center,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "layer": self.layer,
            "shape": self.shape,
        }


@dataclass
class DXFParseResult:
    """Complete parsed result from a DXF file."""
    source_file: str
    units: str
    wall_segments: list[WallSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    columns: list[Column] = field(default_factory=list)
    entities_total: int = 0
    entities_parsed: int = 0
    layers_found: list[str] = field(default_factory=list)
    bounding_box_mm: tuple[float, float, float, float] = (0, 0, 0, 0)
    errors: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "source_file": self.source_file,
            "units": self.units,
            "wall_segments": [s.to_dict() for s in self.wall_segments],
            "openings": [o.to_dict() for o in self.openings],
            "rooms": [r.to_dict() for r in self.rooms],
            "columns": [c.to_dict() for c in self.columns],
            "entities_total": self.entities_total,
            "entities_parsed": self.entities_parsed,
            "layers_found": self.layers_found,
            "bounding_box_mm": self.bounding_box_mm,
            "errors": self.errors,
        }


class DXFParser:
    """Parse DXF/DWG files and extract structural geometry."""

    # Entities that form walls (line-like)
    WALL_ENTITY_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "SPLINE"}
    # Entities that form openings
    DOOR_ENTITY_TYPES = {"INSERT", "BLOCK"}
    WINDOW_ENTITY_TYPES = {"INSERT", "BLOCK"}

    # Standard opening sizes (mm)
    STANDARD_DOOR_WIDTHS = [900, 1000, 1100, 1200, 1500, 1800]
    STANDARD_DOOR_HEIGHTS = [2000, 2100, 2200, 2400]
    STANDARD_WINDOW_WIDTHS = [600, 900, 1200, 1500, 1800, 2400]
    STANDARD_WINDOW_HEIGHTS = [600, 900, 1000, 1200, 1500]

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        self.doc: Optional[ezdxf.document.Drawing] = None
        self.msp = None
        self._units = "mm"
        self._errors: list[str] = []

    def parse(self) -> DXFParseResult:
        """Full parse of the DXF file."""
        try:
            self.doc = ezdxf.readfile(str(self.filepath))
        except ezdxf.DXFStructureError as e:
            raise ValueError(f"Invalid DXF structure: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to open DXF: {e}")

        self.msp = self.doc.modelspace()
        self._units = self._detect_units()

        result = DXFParseResult(source_file=str(self.filepath), units=self._units)

        # Iterate all entities
        entities = list(self.msp)
        result.entities_total = len(entities)

        all_points = []
        layer_entities: dict[str, list] = {}

        for ent in entities:
            try:
                parsed = self._parse_entity(ent)
                if parsed:
                    layer = parsed.get("layer", "UNKNOWN")
                    if layer not in layer_entities:
                        layer_entities[layer] = []
                    layer_entities[layer].append(parsed)

                    # Collect points for bounding box
                    if "points" in parsed:
                        all_points.extend(parsed["points"])

                    result.entities_parsed += 1

            except Exception as e:
                self._errors.append(f"Entity parse error [{ent.dxftype}]: {e}")

        # Classify and assign to result
        for layer, entities in layer_entities.items():
            category = normalize_layer(layer)
            for e in entities:
                e["category"] = category

                if category == "WALL":
                    for seg in self._to_wall_segments(e):
                        result.wall_segments.append(seg)
                elif category == "DOOR":
                    op = self._to_opening(e, "DOOR")
                    if op:
                        result.openings.append(op)
                elif category == "WINDOW":
                    op = self._to_opening(e, "WINDOW")
                    if op:
                        result.openings.append(op)
                elif category == "COLUMN":
                    col = self._to_column(e)
                    if col:
                        result.columns.append(col)

        # Extract rooms from enclosed wall polygons
        result.rooms = self._extract_rooms(result.wall_segments)

        # Bounding box
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            result.bounding_box_mm = (min(xs), min(ys), max(xs), max(ys))

        result.layers_found = list(layer_entities.keys())
        result.errors = self._errors

        return result

    def _detect_units(self) -> str:
        """Detect drawing units from DXF header."""
        try:
            dxfvars = self.doc.dxfvars
            insunits = dxfvars.get("INSUNITS", 0)
            # ezdxf unit constants
            unit_map = {
                0: "unitless",
                1: "inches",
                2: "feet",
                4: "mm",
                5: "cm",
                6: "m",
            }
            return unit_map.get(insunits, "mm")
        except Exception:
            return "mm"

    def _parse_entity(self, ent) -> Optional[dict]:
        """Parse a single DXF entity into a dict."""
        dxftype = ent.dxftype()
        layer = str(ent.dxf.layer) if hasattr(ent.dxf, "layer") else "UNKNOWN"

        if dxftype == "LINE":
            return self._parse_line(ent, layer)
        elif dxftype == "LWPOLYLINE":
            return self._parse_lwpolyline(ent, layer)
        elif dxftype == "POLYLINE":
            return self._parse_polyline(ent, layer)
        elif dxftype == "ARC":
            return self._parse_arc(ent, layer)
        elif dxftype == "CIRCLE":
            return self._parse_circle(ent, layer)
        elif dxftype == "INSERT":
            return self._parse_insert(ent, layer)
        elif dxftype == "LINE":
            return self._parse_line(ent, layer)
        elif dxftype == "SPLINE":
            return self._parse_spline(ent, layer)
        else:
            return None

    def _parse_line(self, ent, layer: str) -> dict:
        """Parse LINE entity."""
        start = ent.dxf.start
        end = ent.dxf.end
        pts = [(float(start.x), float(start.y)), (float(end.x), float(end.y))]
        length = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        return {"type": "LINE", "layer": layer, "points": pts, "length": length}

    def _parse_lwpolyline(self, ent, layer: str) -> dict:
        """Parse LWPOLYLINE (closed polygon)."""
        pts = []
        for pt in ent.points():
            pts.append((float(pt.x), float(pt.y)))
        if len(pts) < 2:
            return None
        return {"type": "LWPOLYLINE", "layer": layer, "points": pts, "closed": ent.closed}

    def _parse_polyline(self, ent, layer: str) -> dict:
        """Parse POLYLINE (as vertices)."""
        pts = []
        # Collect all vertices
        if ent.is_2d_polyline:
            for v in ent.vertices:
                pts.append((float(v.dxf.location.x), float(v.dxf.location.y)))
        if len(pts) < 2:
            return None
        return {"type": "POLYLINE", "layer": layer, "points": pts, "closed": ent.is_closed}

    def _parse_arc(self, ent, layer: str) -> dict:
        """Parse ARC as a series of points."""
        center = Vector(ent.dxf.center)
        radius = float(ent.dxf.radius)
        start_angle = math.radians(float(ent.dxf.start_angle))
        end_angle = math.radians(float(ent.dxf.end_angle))

        # Sample arc as line segments
        pts = []
        num_segments = max(8, int(abs(end_angle - start_angle) / (math.pi / 16)))
        for i in range(num_segments + 1):
            angle = start_angle + (end_angle - start_angle) * (i / num_segments)
            pts.append((center.x + radius * math.cos(angle),
                        center.y + radius * math.sin(angle)))
        return {"type": "ARC", "layer": layer, "points": pts, "radius": radius}

    def _parse_circle(self, ent, layer: str) -> dict:
        """Parse CIRCLE as polygon approximation."""
        center = Vector(ent.dxf.center)
        radius = float(ent.dxf.radius)
        # Approximate as 32-gon
        pts = []
        for i in range(33):
            angle = 2 * math.pi * i / 32
            pts.append((center.x + radius * math.cos(angle),
                        center.y + radius * math.sin(angle)))
        return {"type": "CIRCLE", "layer": layer, "points": pts, "radius": radius}

    def _parse_insert(self, ent, layer: str) -> dict:
        """Parse INSERT (block reference) — doors/windows."""
        loc = ent.dxf.insert
        return {
            "type": "INSERT",
            "layer": layer,
            "location": (float(loc.x), float(loc.y)),
            "block_name": str(ent.dxf.name),
            "rotation": float(ent.dxf.rotation) if hasattr(ent.dxf, "rotation") else 0.0,
            "scale": tuple(ent.dxf.scale) if hasattr(ent.dxf, "scale") else (1, 1, 1),
        }

    def _parse_spline(self, ent, layer: str) -> dict:
        """Parse SPLINE as line string points."""
        pts = []
        try:
            # Sample spline control points / fit points
            control_points = ent.control_points()
            for cp in control_points:
                pts.append((float(cp.x), float(cp.y)))
        except Exception:
            pass
        if len(pts) < 2:
            return None
        return {"type": "SPLINE", "layer": layer, "points": pts}

    def _to_wall_segments(self, entity: dict) -> list[WallSegment]:
        """Convert entity to WallSegment objects."""
        pts = entity.get("points", [])
        layer = entity["layer"]
        segments = []
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if length < 1.0:  # Skip very short segments
                continue
            segments.append(WallSegment(start=p1, end=p2, layer=layer, length=length))
        return segments

    def _to_opening(self, entity: dict, kind: str) -> Optional[Opening]:
        """Convert INSERT entity to Opening."""
        if entity["type"] != "INSERT":
            return None
        loc = entity.get("location", (0, 0))
        rot = entity.get("rotation", 0.0)
        scale = entity.get("scale", (1, 1, 1))

        # Estimate size from scale
        width = scale[0] if scale[0] > 0.1 else 900
        height = scale[1] if scale[1] > 0.1 else 2100

        # Pick standard sizes
        if kind == "DOOR":
            width = min(self.STANDARD_DOOR_WIDTHS, key=lambda x: abs(x - width * 1000))
            height = 2100
        else:
            width = min(self.STANDARD_WINDOW_WIDTHS, key=lambda x: abs(x - width * 1000))
            height = 1200

        # Bounding box (axis-aligned)
        half_w = width / 2
        half_h = height / 2
        bbox = (loc[0] - half_w, loc[1] - half_h, loc[0] + half_w, loc[1] + half_h)

        return Opening(
            kind=kind,
            center=loc,
            width=width,
            height=height,
            rotation_deg=rot,
            layer=entity["layer"],
            bounding_box=bbox,
        )

    def _to_column(self, entity: dict) -> Optional[Column]:
        """Convert entity to Column object."""
        dxftype = entity["type"]

        if dxftype == "CIRCLE":
            r = entity.get("radius", 150)
            # Use centroid
            pts = entity.get("points", [])
            if pts:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                center = (cx, cy)
            else:
                return None
            return Column(
                center=center,
                width_mm=float(r * 2),
                height_mm=float(r * 2),
                layer=entity["layer"],
                shape="circle",
            )

        elif dxftype == "LWPOLYLINE" or dxftype == "POLYLINE" or dxftype == "CIRCLE":
            pts = entity.get("points", [])
            if len(pts) < 2:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            return Column(
                center=(cx, cy),
                width_mm=float(w) if w > 0 else 300,
                height_mm=float(h) if h > 0 else 300,
                layer=entity["layer"],
                shape="rect",
            )

        elif dxftype == "INSERT":
            loc = entity.get("location", (0, 0))
            scale = entity.get("scale", (1, 1, 1))
            # Assume standard column 400x400mm
            width = max(float(scale[0]) * 400, 200)
            height = max(float(scale[1]) * 400, 200)
            return Column(
                center=loc,
                width_mm=width,
                height_mm=height,
                layer=entity["layer"],
                shape="rect",
            )

        return None

    def _extract_rooms(self, wall_segments: list[WallSegment]) -> list[Room]:
        """Extract closed rooms from wall segments using shapely."""
        if not wall_segments:
            return []

        # Build shapely lines
        lines = []
        for seg in wall_segments:
            ls = LineString([seg.start, seg.end])
            if ls.is_valid and not ls.is_empty:
                lines.append(ls)

        if not lines:
            return []

        try:
            merged = linemerge(lines)
            if merged.is_empty:
                return []
        except Exception:
            return []

        # Find enclosed polygons using polygonize
        try:
            from shapely.ops import polygonize
            polygons = list(polygonize(merged))
        except Exception:
            return []

        rooms = []
        for i, poly in enumerate(polygons):
            if not poly.is_valid or poly.area < 10000:  # Skip very small (< 0.01 m²)
                continue
            coords = list(poly.exterior.coords)[:-1]  # Remove closing point
            centroid = (poly.centroid.x, poly.centroid.y)
            rooms.append(Room(
                name=f"Room_{i+1}",
                polygon=[(float(c[0]), float(c[1])) for c in coords],
                area_mm2=float(poly.area),
                centroid=(float(centroid[0]), float(centroid[1])),
            ))

        return rooms

    def export_json(self, result: DXFParseResult, output_path: str) -> None:
        """Save parse result to JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def export_geometry_json(self, result: DXFParseResult, output_path: str) -> None:
        """
        Export geometry as simplified GeoJSON-like structure
        for downstream renderer.
        """
        geo = {
            "type": "FeatureCollection",
            "metadata": {
                "source": result.source_file,
                "units": result.units,
                "bounding_box_mm": result.bounding_box_mm,
                "total_walls": len(result.wall_segments),
                "total_openings": len(result.openings),
                "total_rooms": len(result.rooms),
                "total_columns": len(result.columns),
            },
            "walls": [
                {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [s.start, s.end]}}
                for s in result.wall_segments
            ],
            "rooms": [
                {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [r.polygon + [r.polygon[0]]]}}
                for r in result.rooms
            ],
            "openings": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": o.center},
                 "properties": {"kind": o.kind, "width": o.width, "height": o.height}}
                for o in result.openings
            ],
            "columns": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": c.center},
                 "properties": {"width": c.width_mm, "height": c.height_mm, "shape": c.shape}}
                for c in result.columns
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geo, f, indent=2, ensure_ascii=False)


# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dxf_parser.py <input.dxf> [output.json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"[DXFParser] Parsing: {input_file}")
    parser = DXFParser(input_file)
    result = parser.parse()

    print(f"  Entities: {result.entities_parsed}/{result.entities_total}")
    print(f"  Walls: {len(result.wall_segments)}")
    print(f"  Openings: {len(result.openings)}")
    print(f"  Rooms: {len(result.rooms)}")
    print(f"  Columns: {len(result.columns)}")
    print(f"  Layers: {', '.join(result.layers_found)}")
    print(f"  Bounding box: {result.bounding_box_mm}")

    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for e in result.errors[:5]:
            print(f"    - {e}")

    if output_file:
        parser.export_json(result, output_file)
        print(f"  Saved: {output_file}")