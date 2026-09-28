"""
Object Detector — Stage 2 (Visual Perception), Object-State Level

Provides two implementations behind a common interface:
  1. HSVContourDetector  — zero-setup, uses color ranges to find red/yellow/large boxes
  2. YOLODetector        — wraps ultralytics YOLO for fine-tuned models

The factory function `create_detector(mode)` picks the right one based on config.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected object with bounding box and metadata."""
    label: str                          # e.g. "red_box", "yellow_box", "large_box", "sample_item"
    confidence: float                   # 0.0 – 1.0
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2) in pixel coords
    state: str = "unknown"              # e.g. "open", "closed", "in_hand"
    area: int = 0                       # pixel area of the detection
    center: Tuple[int, int] = (0, 0)    # center point


@dataclass
class DetectionResult:
    """All detections from a single frame."""
    detections: List[Detection] = field(default_factory=list)
    frame_annotated: Optional[np.ndarray] = None  # frame with overlays drawn


# ── Abstract base ───────────────────────────────────────────────────────────

class BaseDetector(ABC):
    """Interface that all detectors implement."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a BGR frame, return DetectionResult."""
        ...

    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and labels on a frame copy."""
        annotated = frame.copy()
        colors = {
            "red_box": (0, 0, 255),
            "yellow_box": (0, 255, 255),
            "large_box": (255, 180, 0),
            "sample_item": (0, 255, 0),
            "unknown": (200, 200, 200),
        }
        for det in result.detections:
            color = colors.get(det.label, (200, 200, 200))
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det.label} {det.confidence:.0%}"
            if det.state != "unknown":
                label_text += f" [{det.state}]"
            # Background for text
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return annotated


# ── HSV Color + Contour Detector (MVP fallback) ────────────────────────────

class HSVContourDetector(BaseDetector):
    """
    Detects red, yellow, and large (brown/cardboard) boxes using HSV color
    segmentation + contour analysis with skin-tone rejection.

    Key safeguards against false positives:
      - High saturation floors for red/yellow (S >= 150) exclude skin tones
      - Explicit skin-mask subtraction for brown/cardboard detections
      - Solidity + rectangularity filters reject irregular shapes (hands, faces)
    """

    # HSV ranges: (H_low, S_low, V_low, H_high, S_high, V_high)
    # OpenCV HSV: H 0-179, S 0-255, V 0-255
    # Saturation floors set HIGH so that skin tones (S ~ 30-130) are excluded.
    COLOR_RANGES = {
        "red_box": [
            # Pure saturated red — S >= 150 reliably excludes skin
            (0, 150, 80, 10, 255, 255),
            (165, 150, 80, 179, 255, 255),
        ],
        "yellow_box": [
            # Pure saturated yellow — S >= 140 excludes skin
            (20, 140, 100, 38, 255, 255),
        ],
        "large_box": [
            # Brown / cardboard — moderate saturation; uses skin masking
            (8, 50, 60, 22, 200, 220),
        ],
    }

    # Skin tone HSV range (subtracted from large_box mask)
    SKIN_HSV_RANGES = [
        (0, 20, 50, 35, 175, 255),
    ]

    MIN_AREA = 3500          # reject small blobs (was 2000)
    MAX_AREA = 250000        # reject frame-filling blobs
    MIN_AREA_LARGE_BOX = 8000  # large_box should be genuinely large
    MIN_SOLIDITY = 0.70      # boxes are solid (hands w/ spread fingers ~ 0.4-0.6)
    MIN_RECTANGULARITY = 0.55  # boxes fill their bounding rect well

    def _create_skin_mask(self, hsv: np.ndarray) -> np.ndarray:
        """Build a dilated binary mask of skin-toned pixels."""
        skin = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (hl, sl, vl, hh, sh, vh) in self.SKIN_HSV_RANGES:
            skin = cv2.bitwise_or(
                skin,
                cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh])),
            )
        # Dilate generously so we also cover edges around skin
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        return cv2.dilate(skin, kernel, iterations=2)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (7, 7), 0)

        # Pre-compute skin mask (used only for large_box disambiguation)
        skin_mask = self._create_skin_mask(hsv)
        not_skin = cv2.bitwise_not(skin_mask)

        detections: List[Detection] = []

        for label, ranges in self.COLOR_RANGES.items():
            combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (hl, sl, vl, hh, sh, vh) in ranges:
                mask = cv2.inRange(hsv, np.array([hl, sl, vl]), np.array([hh, sh, vh]))
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Subtract skin regions from brown/cardboard detections
            if label == "large_box":
                combined_mask = cv2.bitwise_and(combined_mask, not_skin)

            # Morphological cleanup (larger kernel for robustness)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            min_area = self.MIN_AREA_LARGE_BOX if label == "large_box" else self.MIN_AREA

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > self.MAX_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / max(h, 1)
                if aspect < 0.35 or aspect > 3.0:
                    continue

                # ── Shape-quality gates (reject non-box blobs) ──────────
                # Solidity = contour area / convex hull area
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = area / max(hull_area, 1)
                if solidity < self.MIN_SOLIDITY:
                    continue

                # Rectangularity = contour area / bounding-rect area
                rect_area = w * h
                rectangularity = area / max(rect_area, 1)
                if rectangularity < self.MIN_RECTANGULARITY:
                    continue

                confidence = min(1.0, rectangularity * solidity * 1.5)

                cx, cy = x + w // 2, y + h // 2
                detections.append(Detection(
                    label=label,
                    confidence=round(confidence, 3),
                    bbox=(x, y, x + w, y + h),
                    state="detected",
                    area=area,
                    center=(cx, cy),
                ))

        # Sort by area descending (large_box should be biggest)
        detections.sort(key=lambda d: d.area, reverse=True)

        # Infer basic spatial relationships
        detections = self._infer_containment(detections)

        return DetectionResult(detections=detections)

    def _infer_containment(self, dets: List[Detection]) -> List[Detection]:
        """
        Simple heuristic: if a smaller box's center is inside a larger box's
        bbox, mark the smaller box as 'inside_large_box'.
        """
        large_boxes = [d for d in dets if d.label == "large_box"]
        if not large_boxes:
            return dets

        lb = large_boxes[0]
        lx1, ly1, lx2, ly2 = lb.bbox

        for det in dets:
            if det.label in ("red_box", "yellow_box"):
                cx, cy = det.center
                if lx1 <= cx <= lx2 and ly1 <= cy <= ly2:
                    det.state = "inside_large_box"
                else:
                    det.state = "outside_large_box"

        return dets


# ── YOLO Detector (upgrade path) ───────────────────────────────────────────

class YOLODetector(BaseDetector):
    """
    Wraps ultralytics YOLO for fine-tuned object detection.
    Falls back to HSVContourDetector if ultralytics isn't installed or
    weights file is missing.
    """

    def __init__(self, weights_path: str = "models/yolo_finetuned.pt",
                 conf_threshold: float = 0.5):
        self._fallback = HSVContourDetector()
        self._model = None
        self._conf = conf_threshold

        try:
            from ultralytics import YOLO
            import os
            if os.path.exists(weights_path):
                self._model = YOLO(weights_path)
                logger.info("YOLO detector loaded from %s", weights_path)
            else:
                logger.warning("YOLO weights not found at %s — falling back to HSV",
                               weights_path)
        except ImportError:
            logger.warning("ultralytics not installed — falling back to HSV detector")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self._model is None:
            return self._fallback.detect(frame)

        results = self._model(frame, conf=self._conf, verbose=False)
        detections: List[Detection] = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = r.names.get(cls_id, f"class_{cls_id}")
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                detections.append(Detection(
                    label=label,
                    confidence=round(conf, 3),
                    bbox=(x1, y1, x2, y2),
                    state="detected",
                    area=(x2 - x1) * (y2 - y1),
                    center=(cx, cy),
                ))

        return DetectionResult(detections=detections)


# ── Factory ─────────────────────────────────────────────────────────────────

def create_detector(mode: str = "hsv", **kwargs) -> BaseDetector:
    """
    Factory function to create the appropriate detector.
    
    Args:
        mode: "hsv" for color+contour fallback, "yolo" for fine-tuned YOLO
    """
    if mode == "yolo":
        return YOLODetector(**kwargs)
    else:
        return HSVContourDetector()
