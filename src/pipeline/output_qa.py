"""
Task 3.1: Output Package Generator
Task 3.2: Auto QA System

HMI Map Automation Pipeline

Combines all stage outputs into the final delivery package:
  background.png, transparent_background.png, masks.png, anchors.json, metadata.json

Plus automated QA validation:
  1. Structure alignment (AI output vs original geometry)
  2. Color rule check (HMI grayscale palette)
  3. Artifact detection (hallucinated objects, floating walls)
  4. Fallback: save deterministic render if QA fails
"""

import json
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from PIL import Image

import cv2


# ─── QA Result ──────────────────────────────────────────────────────────────
@dataclass
class QAReport:
    """Result of automated quality checks."""
    passed: bool
    structure_alignment_score: float = 0.0   # 0-1 (higher = better)
    color_rule_score: float = 0.0            # 0-1 (higher = better)
    artifact_score: float = 0.0              # 0-1 (higher = better)
    overall_score: float = 0.0              # weighted average
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallback_triggered: bool = False
    fallback_reason: str = ""

    def to_dict(self):
        def _native(obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        return {
            "passed": _native(self.passed),
            "structure_alignment_score": round(float(self.structure_alignment_score), 4),
            "color_rule_score": round(float(self.color_rule_score), 4),
            "artifact_score": round(float(self.artifact_score), 4),
            "overall_score": round(float(self.overall_score), 4),
            "issues": self.issues,
            "warnings": self.warnings,
            "fallback_triggered": _native(self.fallback_triggered),
            "fallback_reason": self.fallback_reason,
        }


# ─── Color Palette ──────────────────────────────────────────────────────────
HMI_GRAYSCALE_PALETTE = [
    (30, 30, 30),
    (40, 40, 40),
    (50, 50, 50),
    (60, 60, 60),
    (70, 70, 70),
    (80, 80, 80),
    (90, 90, 90),
    (100, 100, 100),
    (110, 110, 110),
    (120, 120, 120),
    (130, 130, 130),
    (140, 140, 140),
    (150, 150, 150),
    (160, 160, 160),
    (170, 170, 170),
    (180, 180, 180),
    (190, 190, 190),
    (200, 200, 200),
    (210, 210, 210),
    (220, 220, 220),
    (230, 230, 230),
    (240, 240, 240),
    (250, 250, 250),
]

HMI_ALLOWED_RANGE = (25, 255)  # Min/max pixel value for grayscale
COLOR_TOLERANCE = 15  # Max deviation from nearest palette entry


class AutoQA:
    """
    Automated QA for HMI map output quality.
    """

    def __init__(self, geometry=None, threshold_alignment: float = 0.75,
                 threshold_color: float = 0.80, threshold_artifact: float = 0.70):
        self.geometry = geometry
        self.threshold_alignment = threshold_alignment
        self.threshold_color = threshold_color
        self.threshold_artifact = threshold_artifact

    # ─── 1. Structure Alignment Check ───────────────────────────────────────

    def check_structure_alignment(self, output_image: Image.Image,
                                  geometry) -> tuple[float, list[str]]:
        """
        Compare output image walls against original geometry.
        Uses edge detection + Hough lines to find wall lines in output,
        then compares against parsed wall segments.
        Returns (score 0-1, issues).
        """
        if geometry is None:
            return 0.5, ["No geometry reference — skipping alignment check"]

        # Convert to grayscale
        arr = np.array(output_image.convert("L"))

        # Edge detection
        edges = cv2.Canny(arr, 50, 150)

        # Hough lines to detect straight wall lines
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                                threshold=30, minLineLength=20, maxLineGap=5)

        if lines is None or len(lines) == 0:
            return 0.3, ["No strong edges detected in output — possible over-blur"]

        # Compare detected lines against geometry wall segments
        detected_walls = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_walls.append(((x1, y1), (x2, y2)))

        # Count walls within acceptable angle/distance tolerance
        matched = 0
        total = len(geometry.wall_segments)

        # For each geometry wall, check if there's a corresponding detected line
        for geom_wall in geometry.wall_segments:
            gx1, gy1 = geom_wall.start
            gx2, gy2 = geom_wall.end

            # Compute expected wall length and angle
            g_len = geom_wall.length
            g_angle = np.arctan2(gy2 - gy1, gx2 - gx1)

            for (dx1, dy1), (dx2, dy2) in detected_walls:
                d_len = np.hypot(dx2 - dx1, dy2 - dy1)
                d_angle = np.arctan2(dy2 - dy1, dx2 - dx1)

                # Angle match within 15°, length within 30%
                angle_diff = abs(g_angle - d_angle) % np.pi
                if angle_diff > np.pi / 2:
                    angle_diff = np.pi - angle_diff

                len_ratio = min(d_len, g_len) / max(d_len, g_len + 1e-6)

                if angle_diff < np.pi / 12 and len_ratio > 0.7:  # 15° and 70% length match
                    matched += 1
                    break

        score = matched / max(total, 1)
        issues = []
        if score < self.threshold_alignment:
            issues.append(f"Structure alignment {score:.0%} below threshold {self.threshold_alignment:.0%}")

        return score, issues

    # ─── 2. Color Rule Check ───────────────────────────────────────────────

    def check_color_rules(self, output_image: Image.Image) -> tuple[float, list[str]]:
        """
        Check HMI grayscale palette compliance.
        Any pixel with RGB values outside the allowed range → violation.
        """
        arr = np.array(output_image)

        # Check for non-grayscale (colored) pixels
        if len(arr.shape) == 3:
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            # Grayscale = R≈G≈B
            grayscale_deviation = np.maximum.reduce([
                np.abs(r.astype(int) - g.astype(int)),
                np.abs(g.astype(int) - b.astype(int)),
                np.abs(b.astype(int) - r.astype(int))
            ])
            non_grayscale_pct = np.sum(grayscale_deviation > COLOR_TOLERANCE) / arr.shape[0] / arr.shape[1]
        else:
            non_grayscale_pct = 0.0

        # Check pixel value range
        if len(arr.shape) == 3:
            all_vals = arr.reshape(-1, 3).flatten()
        else:
            all_vals = arr.flatten()

        val_min, val_max = all_vals.min(), all_vals.max()
        out_of_range_pct = np.sum((all_vals < HMI_ALLOWED_RANGE[0]) | (all_vals > HMI_ALLOWED_RANGE[1])) / len(all_vals)

        # Score: penalize non-grayscale and out-of-range pixels
        score = 1.0 - (non_grayscale_pct * 0.5 + out_of_range_pct * 0.5)

        issues = []
        if non_grayscale_pct > 0.01:
            issues.append(f"{non_grayscale_pct:.1%} of pixels are non-grayscale (colored)")
        if out_of_range_pct > 0.05:
            issues.append(f"{out_of_range_pct:.1%} of pixels are out of allowed range {HMI_ALLOWED_RANGE}")

        return max(0.0, score), issues

    # ─── 3. Artifact Detection ─────────────────────────────────────────────

    def check_artifacts(self, output_image: Image.Image) -> tuple[float, list[str]]:
        """
        Detect AI artifacts:
        - Floating isolated objects (not connected to walls)
        - Hallucinated wall segments (walls in places with no geometry)
        - Orphan pixels (small clusters not part of structure)
        """
        arr = np.array(output_image.convert("L"))

        # Threshold to binary
        _, binary = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        issues = []
        artifact_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Very small contours (< 50px) are likely noise
            if area < 50:
                artifact_count += 1
            # Very large white areas with irregular shape = possible hallucination
            elif area > 50000:
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
                if circularity < 0.1:
                    issues.append(f"Large irregular artifact detected (area={area:.0f}px)")
                    artifact_count += 1

        # Hallucinated walls: detect suspicious long horizontal lines in areas with no geometry
        edges = cv2.Canny(arr, 30, 90)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                               threshold=50, minLineLength=100, maxLineGap=10)
        if lines is not None and len(lines) > 100:
            issues.append(f"Excessive number of lines detected ({len(lines)}) — possible over-generation")
            artifact_count += len(lines) - 100

        # Score: 1.0 minus normalized artifact penalty
        max_acceptable = 10
        score = max(0.0, 1.0 - (artifact_count / max_acceptable))

        if score < self.threshold_artifact:
            issues.append(f"Artifact score {score:.0%} below threshold {self.threshold_artifact:.0%}")

        return score, issues

    # ─── Full QA Run ───────────────────────────────────────────────────────

    def run_full_qa(self, output_image: Image.Image,
                    deterministic_image: Optional[Image.Image] = None) -> QAReport:
        """
        Run all QA checks and produce a report.
        If overall score is low, trigger fallback to deterministic render.
        """
        print("[AutoQA] Running QA checks...")
        issues = []
        warnings = []

        # Check 1: Structure alignment
        align_score, align_issues = self.check_structure_alignment(output_image, self.geometry)
        issues.extend(align_issues)

        # Check 2: Color rules
        color_score, color_issues = self.check_color_rules(output_image)
        issues.extend(color_issues)

        # Check 3: Artifacts
        artifact_score, artifact_issues = self.check_artifacts(output_image)
        issues.extend(artifact_issues)

        # Overall score (weighted: alignment 40%, color 30%, artifact 30%)
        overall = align_score * 0.4 + color_score * 0.3 + artifact_score * 0.3

        # Determine fallback
        fallback_triggered = False
        fallback_reason = ""
        if overall < 0.60 or align_score < 0.50:
            fallback_triggered = True
            fallback_reason = f"overall={overall:.2%}, align={align_score:.2%}"
            if deterministic_image is not None:
                warnings.append(f"Fallback triggered: using deterministic render instead")

        passed = (overall >= 0.70 and
                  align_score >= self.threshold_alignment * 0.9 and
                  color_score >= self.threshold_color * 0.9 and
                  artifact_score >= self.threshold_artifact * 0.9)

        print(f"  Alignment: {align_score:.0%}, Color: {color_score:.0%}, Artifact: {artifact_score:.0%}, Overall: {overall:.0%}")
        if fallback_triggered:
            print(f"  ⚠️ Fallback: {fallback_reason}")

        return QAReport(
            passed=passed,
            structure_alignment_score=align_score,
            color_rule_score=color_score,
            artifact_score=artifact_score,
            overall_score=overall,
            issues=issues,
            warnings=warnings,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
        )


class OutputPackageGenerator:
    """
    Generates the final output package:
      background.png, transparent_background.png, masks.png,
      anchors.json, metadata.json, qa_report.json
    """

    def __init__(self, geometry=None):
        self.geometry = geometry

    def generate(
        self,
        enhanced_image: Optional[Image.Image],
        deterministic_image: Image.Image,
        qa_report: QAReport,
        masks_path: Optional[str] = None,
        anchors_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        output_dir: str = "output",
    ) -> dict[str, str]:
        """
        Generate final output package.
        If QA fails, falls back to deterministic_image.
        Returns dict of output file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Decide which image to use
        if qa_report.fallback_triggered and deterministic_image is not None:
            final_img = deterministic_image
            print(f"[Output] QA failed — using deterministic render (fallback)")
        elif enhanced_image is not None:
            final_img = enhanced_image
        else:
            final_img = deterministic_image

        # 1. background.png
        bg_path = str(out_dir / "background.png")
        final_img.save(bg_path, optimize=True)
        results["background"] = bg_path

        # 2. transparent_background.png
        trans_path = str(out_dir / "transparent_background.png")
        trans_img = final_img.convert("RGBA")
        trans_img.save(trans_path, optimize=True)
        results["transparent_background"] = trans_path

        # 3. Copy masks if provided
        if masks_path and Path(masks_path).exists():
            import shutil
            masks_dest = str(out_dir / "masks.png")
            if masks_path != masks_dest:
                shutil.copy(masks_path, masks_dest)
            results["masks"] = masks_dest

        # 4. Copy anchors if provided
        if anchors_path and Path(anchors_path).exists():
            import shutil
            anchors_dest = str(out_dir / "anchors.json")
            if anchors_path != anchors_dest:
                shutil.copy(anchors_path, anchors_dest)
            results["anchors"] = anchors_dest

        # 5. Copy metadata if provided
        if metadata_path and Path(metadata_path).exists():
            import shutil
            meta_dest = str(out_dir / "metadata.json")
            if metadata_path != meta_dest:
                shutil.copy(metadata_path, meta_dest)
            results["metadata"] = meta_dest

        # 6. QA report
        qa_path = str(out_dir / "qa_report.json")
        with open(qa_path, "w") as f:
            json.dump(qa_report.to_dict(), f, indent=2, ensure_ascii=False)
        results["qa_report"] = qa_path

        return results


# ─── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto QA + Output Package Generator")
    parser.add_argument("--enhanced", help="Enhanced AI image path")
    parser.add_argument("--deterministic", required=True, help="Deterministic ISO render path")
    parser.add_argument("--geometry-json", help="Geometry JSON for alignment check")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--masks", help="Masks PNG path")
    parser.add_argument("--anchors", help="Anchors JSON path")
    parser.add_argument("--metadata", help="Metadata JSON path")

    args = parser.parse_args()

    # Load geometry if provided
    geometry = None
    if args.geometry_json:
        from parser.dxf_parser import DXFParseResult
        with open(args.geometry_json) as f:
            geo = json.load(f)
        geometry = DXFParseResult(source_file=geo.get("source_file", ""))
        from parser.dxf_parser import WallSegment, Room, Opening, Column
        for w in geo.get("wall_segments", []):
            geometry.wall_segments.append(WallSegment(**w))
        for r in geo.get("rooms", []):
            geometry.rooms.append(Room(**r))

    # Load images
    det_img = Image.open(args.deterministic).convert("RGB")
    enhanced_img = Image.open(args.enhanced).convert("RGB") if args.enhanced else None

    # Run QA
    qa = AutoQA(geometry=geometry)
    report = qa.run_full_qa(enhanced_img or det_img, det_img if enhanced_img else None)

    # Generate package
    gen = OutputPackageGenerator(geometry=geometry)
    results = gen.generate(
        enhanced_image=enhanced_img,
        deterministic_image=det_img,
        qa_report=report,
        masks_path=args.masks,
        anchors_path=args.anchors,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )

    print("\n=== QA Report ===")
    print(f"Passed: {report.passed}")
    print(f"Overall Score: {report.overall_score:.0%}")
    print(f"Fallback: {report.fallback_triggered} ({report.fallback_reason})")
    if report.issues:
        print("Issues:")
        for issue in report.issues:
            print(f"  - {issue}")

    print("\n=== Output Package ===")
    for name, path in results.items():
        print(f"  {name}: {path}")