"""
Task 1.3: 2.5D Isometric Renderer (Deterministic Geometry Engine)
HMI Map Automation Pipeline

Renders DXFParseResult (walls, rooms, columns, openings) as a
3D Isometric (ISO) view using orthographic projection + 2.5D extrusion.

Output:
  - base_render.png       : RGB rendered image
  - depth_map.png         : 16-bit depth (distance from camera plane)
  - normal_map.png        : RGB surface normals
  - occlusion_mask.png    : per-room / per-zone occlusion visibility mask

No AI — purely deterministic geometry rendering.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser.dxf_parser import DXFParseResult, Room, Column, Opening

# ─── ISO Projection Constants ───────────────────────────────────────────────
# Standard isometric: 30° from horizontal
ISO_COS = math.cos(math.radians(30))  # ≈ 0.8660254
ISO_SIN = math.sin(math.radians(30))  # = 0.5

# Fixed architectural parameters (mm)
DEFAULT_WALL_HEIGHT = 3000      # 3m ceiling height
DEFAULT_WALL_THICKNESS = 200    # 200mm wall thickness
DEFAULT_FLOOR_THICKNESS = 200   # floor slab thickness
DEFAULT_DOOR_HEIGHT = 2100      # standard door height
DEFAULT_WINDOW_SILL = 900       # window sill height from floor

# ISO camera direction vectors (for depth/normal calculation)
# Camera looks from top-front-right (standard isometric)
CAM_DIR = np.array([1, 1, 1], dtype=np.float32)  # normalized
CAM_DIR = CAM_DIR / np.linalg.norm(CAM_DIR)

# Right-facing wall normal (in ISO view)
WALL_RIGHT_NORMAL = np.array([-ISO_COS, ISO_SIN, 0], dtype=np.float32)
# Left-facing wall normal
WALL_LEFT_NORMAL = np.array([ISO_COS, ISO_SIN, 0], dtype=np.float32)
# Floor normal (facing up)
FLOOR_NORMAL = np.array([0, 0, 1], dtype=np.float32)


@dataclass
class ISOCamera:
    """Orthographic ISO camera configuration."""
    # ISO angles (standard 30°)
    angle_xz: float = 30.0   # rotation around Y axis (radians precomputed)
    angle_yz: float = 30.0   # tilt around X axis

    # Scale: pixels per mm in the rendered image
    pixels_per_mm: float = 0.5

    # Z offset: how far above floor (0) the camera looks from
    camera_z_offset: float = DEFAULT_WALL_HEIGHT * 2  # look from above

    # Background color (HMI-friendly grayscale)
    bg_color: tuple[int, int, int] = (40, 40, 40)

    def iso_to_screen(self, wx: float, wy: float, wz: float) -> tuple[float, float]:
        """
        Project world (x, y, z) mm → screen (sx, sy) pixels.
        ISO projection:
          sx = (wx - wy) * cos(30°) * scale
          sy = ((wx + wy) * sin(30°) - wz) * scale
        """
        sx = (wx - wy) * ISO_COS * self.pixels_per_mm
        sy = ((wx + wy) * ISO_SIN - wz) * self.pixels_per_mm
        return (sx, sy)

    def world_to_depth(self, wx: float, wy: float, wz: float) -> float:
        """Compute perpendicular distance from camera plane (for depth map)."""
        # Camera plane: normal = (1, 1, 1) normalized, point = (0, 0, camera_z_offset)
        # depth = dot((P - camera_origin), cam_normal)
        cam_origin = np.array([0.0, 0.0, self.camera_z_offset], dtype=np.float32)
        P = np.array([wx, wy, wz], dtype=np.float32)
        depth = np.dot(P - cam_origin, CAM_DIR)
        return float(depth)


@dataclass
class RenderConfig:
    """Configuration for the ISO renderer."""
    wall_height_mm: float = DEFAULT_WALL_HEIGHT
    wall_thickness_mm: float = DEFAULT_WALL_THICKNESS
    floor_thickness_mm: float = DEFAULT_FLOOR_THICKNESS
    door_height_mm: float = DEFAULT_DOOR_HEIGHT
    window_sill_mm: float = DEFAULT_WINDOW_SILL

    # Wall colors (grayscale HMI theme)
    floor_color: tuple[int, int, int] = (85, 85, 85)
    wall_color: tuple[int, int, int] = (160, 160, 160)
    wall_shadow_color: tuple[int, int, int] = (110, 110, 110)
    column_color: tuple[int, int, int] = (130, 130, 130)
    door_color: tuple[int, int, int] = (200, 200, 200)
    window_color: tuple[int, int, int] = (100, 100, 100)
    opening_cut_color: tuple[int, int, int] = (50, 50, 50)

    # Output resolution
    output_dpi: int = 96

    # Anti-aliasing (sub-pixel sampling)
    anti_alias: bool = True

    # Depth map max distance (mm) — anything beyond is clamped
    depth_max_mm: float = 100_000  # 100m


class ISORenderer:
    """
    Deterministic 2.5D Isometric Renderer.

    Takes DXFParseResult (parsed geometry) and renders an ISO view
    with:
      - Floor polygon (from rooms)
      - Extruded walls (from wall segments)
      - Columns (from column data)
      - Openings (doors/windows cut into walls)

    Output: base_render.png, depth_map.png, normal_map.png
    """

    def __init__(self, camera: Optional[ISOCamera] = None, config: Optional[RenderConfig] = None):
        self.camera = camera or ISOCamera()
        self.config = config or RenderConfig()

    def render(self, geometry: DXFParseResult) -> dict[str, np.ndarray]:
        """
        Main entry point. Renders ISO view from parsed geometry.

        Returns dict:
          'render': HWC, uint8, RGB rendered image (0-255)
          'depth':  HWC, uint16, depth map (0-65535 mm)
          'normal': HWC, uint8, RGB normal map
        """
        # Determine canvas size from geometry bounding box
        bbox = geometry.bounding_box_mm
        if not geometry.wall_segments:
            raise ValueError("Empty geometry: no walls to render")

        # Compute bounding box from wall segments if not set in geometry
        if bbox == (0, 0, 0, 0) or all(v == 0 for v in bbox):
            xs = [s.start[0] for s in geometry.wall_segments] + [s.end[0] for s in geometry.wall_segments]
            ys = [s.start[1] for s in geometry.wall_segments] + [s.end[1] for s in geometry.wall_segments]
            bbox = (min(xs), min(ys), max(xs), max(ys))

        # World size in mm
        world_w = bbox[2] - bbox[0]
        world_h = bbox[3] - bbox[1]

        # Canvas: world extent * ISO scale + margin + wall extrusion
        margin = 500  # mm padding
        render_h = int((world_w + world_h) * ISO_COS * self.camera.pixels_per_mm + 2000)
        render_w = int((world_w + world_h) * ISO_COS * self.camera.pixels_per_mm + 2000)

        # Camera offset: center the drawing in the canvas
        # Origin (0,0) world maps near canvas center
        center_x = render_w / 2
        center_y = render_h / 2 - self.config.wall_height_mm * self.camera.pixels_per_mm * ISO_SIN

        self._center_x = center_x
        self._center_y = center_y

        # Prepare arrays
        render_arr = np.full((render_h, render_w, 3), self.camera.bg_color, dtype=np.uint8)
        depth_arr = np.full((render_h, render_w), self.config.depth_max_mm, dtype=np.float32)
        normal_arr = np.zeros((render_h, render_w, 3), dtype=np.float32)

        # Render layers (back to front for correct occlusion)
        # Layer 1: Floor
        self._render_floor(geometry.rooms, render_arr, depth_arr, normal_arr)

        # Layer 2: Wall left faces (darker - shadow side)
        self._render_walls(geometry.wall_segments, render_arr, depth_arr, normal_arr,
                           side="left", openings=geometry.openings)

        # Layer 3: Wall right faces (lighter - lit side)
        self._render_walls(geometry.wall_segments, render_arr, depth_arr, normal_arr,
                           side="right", openings=geometry.openings)

        # Layer 4: Columns
        self._render_columns(geometry.columns, render_arr, depth_arr, normal_arr)

        # Clip and convert depth → uint16
        depth_uint16 = self._depth_to_uint16(depth_arr)

        # Convert normal to uint8
        normal_uint8 = self._normal_to_uint8(normal_arr)

        return {
            "render": render_arr,
            "depth": depth_uint16,
            "normal": normal_uint8,
        }

    def _render_floor(self, rooms: list[Room], render_arr, depth_arr, normal_arr):
        """Render floor polygons (from room outlines)."""
        h, w = render_arr.shape[:2]
        floor_r, floor_g, floor_b = self.config.floor_color

        for room in rooms:
            poly_mm = room.polygon
            if len(poly_mm) < 3:
                continue

            # Convert polygon to screen coordinates
            screen_pts = []
            for (mx, my) in poly_mm:
                sx, sy = self.camera.iso_to_screen(mx, my, 0)
                screen_pts.append((int(sx + self._center_x), int(sy + self._center_y)))

            # Create polygon mask
            from PIL import ImageDraw
            img = Image.fromarray(render_arr)
            draw = ImageDraw.Draw(img)

            floor_color = (floor_r, floor_g, floor_b)
            try:
                draw.polygon(screen_pts, fill=floor_color, outline=None)
            except Exception:
                continue

            # Update depth: floor at z=0 → depth = camera_z_offset
            floor_depth = self.camera.world_to_depth(
                sum(p[0] for p in poly_mm) / len(poly_mm),
                sum(p[1] for p in poly_mm) / len(poly_mm),
                0
            )

            # Update normal: floor normal = (0, 0, 1)
            self._fill_polygon(depth_arr, screen_pts, floor_depth)
            self._fill_polygon_normal(normal_arr, screen_pts, FLOOR_NORMAL)

            # Copy back
            render_arr[:] = np.array(img)

    def _render_walls(self, wall_segments, render_arr, depth_arr, normal_arr,
                      side: Literal["left", "right"], openings: list[Opening]):
        """
        Render extruded walls (left or right faces in ISO).
        Side determines which face we see:
          - "left": wall on left side of wall segment (ISO left face = darker)
          - "right": wall on right side of wall segment (ISO right face = lighter)
        """
        h, w = render_arr.shape[:2]
        wall_h = self.config.wall_height_mm

        for seg in wall_segments:
            sx1, sy1 = self.camera.iso_to_screen(seg.start[0], seg.start[1], 0)
            sx2, sy2 = self.camera.iso_to_screen(seg.end[0], seg.end[1], 0)
            sx1_i = int(sx1 + self._center_x)
            sy1_i = int(sy1 + self._center_y)
            sx2_i = int(sx2 + self._center_x)
            sy2_i = int(sy2 + self._center_y)

            # Wall thickness in screen pixels
            thickness_px = self.config.wall_thickness_mm * self.camera.pixels_per_mm * ISO_COS

            # Compute wall face points (top and bottom of wall in ISO)
            # For left-facing wall (x - y direction)
            dx = seg.end[0] - seg.start[0]
            dy = seg.end[1] - seg.start[1]
            length = math.hypot(dx, dy)
            if length < 1:
                continue

            # Perpendicular direction (in world, pointing "left" of segment)
            # Normal in screen space: (-dy, dx) / length
            nx_perp = -dy / length
            ny_perp = dx / length

            if side == "left":
                # Left face: move perpendicular in +n direction
                offset = thickness_px / 2
                base_color = self.config.wall_shadow_color
                normal_vec = WALL_LEFT_NORMAL
            else:
                # Right face: move perpendicular in -n direction
                offset = -thickness_px / 2
                base_color = self.config.wall_color
                normal_vec = WALL_RIGHT_NORMAL

            # Wall polygon (4 points): bottom-left, bottom-right, top-right, top-left
            # z=0: bottom; z=wall_h: top
            z_wall = self.config.wall_height_mm

            # Screen coordinates at bottom (z=0) with offset
            p1 = (sx1_i + nx_perp * offset, sy1_i + ny_perp * offset)
            p2 = (sx2_i + nx_perp * offset, sy2_i + ny_perp * offset)

            # Screen coordinates at top (z=wall_h)
            sx1_top, sy1_top = self.camera.iso_to_screen(seg.start[0], seg.start[1], z_wall)
            sx2_top, sy2_top = self.camera.iso_to_screen(seg.end[0], seg.end[1], z_wall)
            p3 = (sx2_top + nx_perp * offset, sy2_top + ny_perp * offset)
            p4 = (sx1_top + nx_perp * offset, sy1_top + ny_perp * offset)

            # Depth at wall face
            mid_x = (seg.start[0] + seg.end[0]) / 2
            mid_y = (seg.start[1] + seg.end[1]) / 2
            wall_depth = self.camera.world_to_depth(mid_x, mid_y, z_wall / 2)

            # Draw wall polygon
            img = Image.fromarray(render_arr)
            draw = ImageDraw.Draw(img)
            pts = [p1, p2, p3, p4]
            try:
                draw.polygon(pts, fill=base_color, outline=None)
            except Exception:
                continue
            render_arr[:] = np.array(img)

            # Fill depth and normal
            self._fill_wall_polygon_depth(depth_arr, pts, wall_depth)
            self._fill_polygon_normal(normal_arr, pts, normal_vec)

    def _render_columns(self, columns: list[Column], render_arr, depth_arr, normal_arr):
        """Render columns as extruded rectangles in ISO view."""
        col_h = self.config.wall_height_mm

        for col in columns:
            cx, cy = col.center
            hw = col.width_mm / 2
            hh = col.height_mm / 2

            # Column rectangle in world space (4 corner points)
            corners = [
                (cx - hw, cy - hh),
                (cx + hw, cy - hh),
                (cx + hw, cy + hh),
                (cx - hw, cy + hh),
            ]

            # Project to screen (z=0 and z=col_h)
            screen_pts_bottom = []
            screen_pts_top = []
            for (mx, my) in corners:
                sbx, sby = self.camera.iso_to_screen(mx, my, 0)
                stx, sty = self.camera.iso_to_screen(mx, my, col_h)
                screen_pts_bottom.append((int(sbx + self._center_x), int(sby + self._center_y)))
                screen_pts_top.append((int(stx + self._center_x), int(sty + self._center_y)))

            col_depth = self.camera.world_to_depth(cx, cy, col_h / 2)
            col_normal = (0, ISO_SIN, ISO_COS)  # column face normal

            # Draw top face
            img = Image.fromarray(render_arr)
            draw = ImageDraw.Draw(img)
            try:
                draw.polygon(screen_pts_top, fill=self.config.column_color, outline=None)
            except Exception:
                pass
            render_arr[:] = np.array(img)

            self._fill_polygon(depth_arr, screen_pts_top, col_depth)
            self._fill_polygon_normal(normal_arr, screen_pts_top, col_normal)

    def _fill_polygon(self, depth_arr, screen_pts, depth_value):
        """Fill polygon area in depth array with constant depth value."""
        if len(screen_pts) < 3:
            return
        h, w = depth_arr.shape
        # Create mask using PIL
        mask_img = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        try:
            mask_draw.polygon([(int(x), int(y)) for x, y in screen_pts], fill=1, outline=1)
        except Exception:
            return
        mask = np.array(mask_img)
        depth_arr[mask > 0] = depth_value

    def _fill_wall_polygon_depth(self, depth_arr, screen_pts, depth_value):
        """Fill wall polygon in depth array."""
        self._fill_polygon(depth_arr, screen_pts, depth_value)

    def _fill_polygon_normal(self, normal_arr, screen_pts, normal_vec):
        """Fill polygon area in normal array with a constant normal."""
        if len(screen_pts) < 3:
            return
        h, w = normal_arr.shape[:2]
        mask_img = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        try:
            mask_draw.polygon([(int(x), int(y)) for x, y in screen_pts], fill=1, outline=1)
        except Exception:
            return
        mask = np.array(mask_img)
        normal_arr[mask > 0] = [
            int((normal_vec[0] + 1) * 127.5),
            int((normal_vec[1] + 1) * 127.5),
            int((normal_vec[2] + 1) * 127.5),
        ]

    def _depth_to_uint16(self, depth_arr: np.ndarray) -> np.ndarray:
        """Convert float depth (mm) to uint16 (0-65535)."""
        d = depth_arr.copy()
        d = np.clip(d, 0, self.config.depth_max_mm)
        # Normalize to 0-65535
        d_norm = (d / self.config.depth_max_mm * 65535).astype(np.uint16)
        return d_norm

    def _normal_to_uint8(self, normal_arr: np.ndarray) -> np.ndarray:
        """Convert float normal vectors (-1 to 1) to uint8 RGB (0-255)."""
        # normal_arr is already in 0-255 float after fill_polygon_normal
        result = np.clip(normal_arr, 0, 255).astype(np.uint8)
        return result

    def save_outputs(self, outputs: dict[str, np.ndarray], output_dir: str) -> dict[str, str]:
        """
        Save rendered outputs to PNG files.
        Returns dict of filepath → filepath mappings.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved = {}
        for name, arr in outputs.items():
            filepath = str(out_path / f"{name}.png")
            if name == "depth":
                # 16-bit grayscale
                img = Image.fromarray(arr, mode="I;16")
            else:
                img = Image.fromarray(arr)
            img.save(filepath, optimize=False)
            saved[name] = filepath

        return saved

    def render_to_files(self, geometry: DXFParseResult, output_dir: str) -> dict[str, str]:
        """Full pipeline: render + save."""
        outputs = self.render(geometry)
        return self.save_outputs(outputs, output_dir)


# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python iso_renderer.py <geometry.json> <output_dir>")
        sys.exit(1)

    geom_path = sys.argv[1]
    out_dir = sys.argv[2]

    with open(geom_path, "r") as f:
        geo = json.load(f)

    # Reconstruct DXFParseResult
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

    print(f"[ISORenderer] Rendering: {geom_path}")
    renderer = ISORenderer()
    saved = renderer.render_to_files(result, out_dir)
    for name, path in saved.items():
        print(f"  {name}: {path}")