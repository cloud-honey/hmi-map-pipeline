"""
Task 1.2: PNG/PDF Fallback Vectorizer
HMI Map Automation Pipeline

Converts raster images (PNG/PDF snapshots of floor plans) into
vector geometry (wall graph, room polygons, openings) using OpenCV
contour detection and shapely polygon operations.

Input:  PNG/PDF floor plan image
Output: DXFParseResult-compatible JSON (same structure as Task 1.1)
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from parser.dxf_parser import (
    DXFParseResult, WallSegment, Opening, Room, Column,
    DXFParseResult as ParseResult  # alias for consistency
)

# ─── Constants ──────────────────────────────────────────────────────────────
PIXELS_PER_MM = 3.78  # ~96 DPI: 96/25.4 ≈ 3.78 px/mm

# Minimum wall thickness in mm (walls thinner than this are noise)
MIN_WALL_THICKNESS_MM = 60
# Minimum room area in mm² (smaller = noise)
MIN_ROOM_AREA_MM2 = 50_000  # 0.05 m²
# Corridor width max in mm (wider = open space, not corridor)
MAX_CORRIDOR_WIDTH_MM = 2500


@dataclass
class VectorizerConfig:
    """Configuration for the vectorizer."""
    dpi: int = 96
    pixels_per_mm: float = PIXELS_PER_MM
    min_wall_thickness_mm: float = MIN_WALL_THICKNESS_MM
    min_room_area_mm2: float = MIN_ROOM_AREA_MM2
    max_corridor_width_mm: float = MAX_CORRIDOR_WIDTH_MM
    morph_kernel_size: int = 5
    morph_iterations: int = 2


class PNGVectorizer:
    """Convert raster floor plan images to vector geometry."""

    def __init__(self, config: Optional[VectorizerConfig] = None):
        self.config = config or VectorizerConfig()

    def vectorize(self, image_path: str) -> DXFParseResult:
        """
        Main entry point. Load image, detect walls, extract rooms/columns/openings.
        Output matches DXFParseResult schema for pipeline compatibility.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 1: Detect walls (dark lines in architectural drawings)
        wall_mask = self._detect_walls(gray)
        wall_segments = self._extract_wall_segments(wall_mask)

        # Step 2: Detect openings (regular rectangular gaps in walls)
        opening_mask = self._detect_openings(gray, wall_mask)
        openings = self._extract_openings(opening_mask)

        # Step 3: Extract rooms from wall segments
        rooms = self._extract_rooms(wall_segments)

        # Step 4: Detect columns (small rectangular blocks)
        column_mask = self._detect_columns(gray, wall_mask)
        columns = self._extract_columns(column_mask)

        # Build result
        result = DXFParseResult(
            source_file=str(image_path),
            units="px_from_raster",
            wall_segments=wall_segments,
            openings=openings,
            rooms=rooms,
            columns=columns,
        )

        if wall_segments:
            all_pts = [(s.start + s.end) for s in wall_segments]
            xs = [p[0] for pt in all_pts for p in [pt]]
            ys = [p[1] for pt in all_pts for p in [pt]]
            result.bounding_box_mm = (
                min(xs) / self.config.pixels_per_mm,
                min(ys) / self.config.pixels_per_mm,
                max(xs) / self.config.pixels_per_mm,
                max(ys) / self.config.pixels_per_mm,
            )

        return result

    def _detect_walls(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect wall regions using adaptive threshold + morphological ops.
        Architectural drawings: walls = dark lines on white/light background.
        """
        # Adaptive threshold for varying background
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,
            C=5,
        )

        # Morphological closing to fill gaps in walls
        kernel = np.ones((self.config.morph_kernel_size, self.config.morph_kernel_size), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=self.config.morph_iterations)

        # Remove noise (very small blobs)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(closed)
        min_area_px = (self.config.min_wall_thickness_mm * self.config.pixels_per_mm) ** 2

        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area_px / 4:
                cv2.drawContours(mask, [cnt], -1, 255, -1)

        return mask

    def _extract_wall_segments(self, wall_mask: np.ndarray) -> list[WallSegment]:
        """
        Extract wall segments by skeletonizing the wall mask,
        then finding line segments from skeleton pixels.
        """
        # Skeletonize ( thinning to 1-pixel wide lines )
        skeleton = self._skeletonize(wall_mask)

        # Find contours of the wall regions (for boundary extraction)
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segments = []

        # For each contour, extract polyline segments
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, epsilon=3.0, closed=True)
            pts = [(float(p[0][0]), float(p[0][1])) for p in approx]

            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if length < 2.0:  # Skip very short
                    continue
                # Convert to mm
                p1_mm = (p1[0] / self.config.pixels_per_mm, p1[1] / self.config.pixels_per_mm)
                p2_mm = (p2[0] / self.config.pixels_per_mm, p2[1] / self.config.pixels_per_mm)
                segments.append(WallSegment(
                    start=p1_mm, end=p2_mm,
                    layer="wall", length=length / self.config.pixels_per_mm
                ))

        return segments

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Skeletonize binary mask using Zhang-Suen algorithm."""
        img = (mask // 255).astype(np.uint8)
        if img.sum() == 0:
            return mask

        skel = np.zeros_like(img)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(img, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if img.sum() == 0:
                break

        return skel * 255

    def _detect_openings(self, gray: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
        """
        Detect door/window openings: white rectangular gaps interrupting dark wall lines.
        Strategy: Find rectangular white regions inside wall mask (negative space).
        """
        # Invert: walls white → openings as gaps
        inverted = cv2.bitwise_not(wall_mask)

        # Detect rectangular white regions
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        opening_mask = np.zeros_like(gray)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            approx = cv2.approxPolyDP(cnt, epsilon=5.0, closed=True)

            # Only rectangular contours (4 vertices) in reasonable size range
            if len(approx) >= 4 and len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = max(w, h) / (min(w, h) + 1e-6)

                # Opening: narrow tall (doors) or wide short (windows)
                if 0.2 < aspect < 8:  # Not too square, not too extreme
                    # Area filter (typical door: 200×900 = 180000 px², window: 900×1200 = 1080000 px²)
                    min_area = 10_000  # px²
                    max_area = 5_000_000  # px²
                    if min_area < area < max_area:
                        cv2.drawContours(opening_mask, [cnt], -1, 255, -1)

        return opening_mask

    def _extract_openings(self, opening_mask: np.ndarray) -> list[Opening]:
        """Extract openings from mask — center, width, height, rotation."""
        contours, _ = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        openings = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Too small
                continue

            # Get rotated bounding box
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            # Convert to mm
            w_mm = w / self.config.pixels_per_mm
            h_mm = h / self.config.pixels_per_mm

            # Classify as door or window based on aspect ratio
            if h_mm > w_mm * 1.5:
                kind = "DOOR"
                # Standard door heights: 2100mm
                height = 2100
                width = min(self._nearest_standard(w_mm, [900, 1000, 1100, 1200]), w_mm)
            else:
                kind = "WINDOW"
                # Standard window heights: 900-1500mm
                height = self._nearest_standard(h_mm, [600, 900, 1000, 1200])
                width = min(self._nearest_standard(w_mm, [600, 900, 1200, 1500, 1800, 2400]), w_mm)

            # Center in mm
            cx_mm = cx / self.config.pixels_per_mm
            cy_mm = cy / self.config.pixels_per_mm

            # Rotation
            rot_deg = -angle if angle != 0 else 0

            bbox = (cx_mm - width / 2, cy_mm - height / 2, cx_mm + width / 2, cy_mm + height / 2)

            openings.append(Opening(
                kind=kind, center=(cx_mm, cy_mm),
                width=width, height=height,
                rotation_deg=rot_deg,
                layer=kind.lower(),
                bounding_box=bbox,
            ))

        return openings

    def _nearest_standard(self, value: float, standards: list[float]) -> float:
        """Find nearest standard size."""
        return min(standards, key=lambda x: abs(x - value))

    def _detect_columns(self, gray: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
        """Detect structural columns: solid rectangular blocks in the floor plan."""
        # Use template matching for common column shapes
        # Or: detect small solid rectangular contours not connected to walls

        # Detect filled rectangles (columns are solid fills in architectural drawings)
        # Threshold to get white/black separation
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        col_mask = np.zeros_like(gray)
        min_col_area_px = (150 * self.config.pixels_per_mm) ** 2  # Min 150x150mm column

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_col_area_px:
                continue

            approx = cv2.approxPolyDP(cnt, epsilon=5.0, closed=True)

            # Columns: roughly square, small, isolated
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = max(w, h) / (min(w, h) + 1e-6)
                if aspect < 1.4:  # Nearly square
                    # Check not inside a wall (low wall_mask value at center)
                    mx, my = x + w // 2, y + h // 2
                    if wall_mask[my, mx] < 128:  # Not wall area
                        cv2.drawContours(col_mask, [cnt], -1, 255, -1)

        return col_mask

    def _extract_columns(self, column_mask: np.ndarray) -> list[Column]:
        """Extract column positions and sizes."""
        contours, _ = cv2.findContours(column_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        columns = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)

            # Convert to mm
            w_mm = w / self.config.pixels_per_mm
            h_mm = h / self.config.pixels_per_mm

            # Center
            cx_mm = (x + w // 2) / self.config.pixels_per_mm
            cy_mm = (y + h // 2) / self.config.pixels_per_mm

            # Determine shape
            aspect = max(w, h) / (min(w, h) + 1e-6)
            shape = "circle" if aspect > 0.9 else "rect"

            columns.append(Column(
                center=(cx_mm, cy_mm),
                width_mm=w_mm, height_mm=h_mm,
                layer="column", shape=shape,
            ))

        return columns

    def _extract_rooms(self, wall_segments: list[WallSegment]) -> list[Room]:
        """Extract closed rooms from wall segments (same logic as DXF parser)."""
        from parser.dxf_parser import DXFParser
        # Use the same room extraction logic
        parser = DXFParser.__new__(DXFParser)
        return parser._extract_rooms(wall_segments)


class PDFVectorizer(PNGVectorizer):
    """PDF → PNG → PNGVectorizer pipeline."""

    def vectorize(self, pdf_path: str, dpi: int = 96) -> DXFParseResult:
        """Convert PDF to image first, then vectorize."""
        # Try pdf2image (poppler) if available
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=dpi)
            if not images:
                raise RuntimeError("PDF produced no pages")
            # Use first page
            img_path = Path(pdf_path).with_suffix(".png")
            images[0].save(str(img_path), "PNG")
            return super().vectorize(str(img_path))
        except ImportError:
            # Fallback: use pdftoppm (command line)
            import subprocess
            pdf_path_obj = Path(pdf_path)
            png_out = pdf_path_obj.with_suffix(".png")
            result = subprocess.run(
                ["pdftoppm", "-r", str(dpi), "-png", str(pdf_path), str(pdf_path_obj.stem)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"pdftoppm failed: {result.stderr}")
            # Find generated PNG (pdftoppm creates filename-1.png)
            png_candidates = list(Path(pdf_path).parent.glob(f"{pdf_path_obj.stem}-1.png"))
            if not png_candidates:
                raise RuntimeError(f"pdftoppm didn't produce output: {result.stderr}")
            return super().vectorize(str(png_candidates[0]))


# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python png_vectorizer.py <input.png|pdf> [output.json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"[Vectorizer] Processing: {input_file}")

    if input_file.lower().endswith(".pdf"):
        vectorizer = PDFVectorizer()
    else:
        vectorizer = PNGVectorizer()

    result = vectorizer.vectorize(input_file)

    print(f"  Walls: {len(result.wall_segments)}")
    print(f"  Openings: {len(result.openings)}")
    print(f"  Rooms: {len(result.rooms)}")
    print(f"  Columns: {len(result.columns)}")

    if output_file:
        vectorizer._export_json(result, output_file)  # type: ignore
        print(f"  Saved: {output_file}")