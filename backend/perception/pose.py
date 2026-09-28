"""
Pose & Hand Keypoint Estimation — Stage 2 (Visual Perception), Person Level

Uses MediaPipe Tasks API (v0.10.35+) for pose and hand landmark detection.
Falls back gracefully if MediaPipe is unavailable.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Landmark:
    """A single keypoint."""
    x: float   # normalized 0-1
    y: float
    z: float = 0.0
    visibility: float = 0.0
    name: str = ""


@dataclass
class HandResult:
    """Detected hand with landmarks."""
    handedness: str = "unknown"          # "Left" or "Right"
    landmarks: List[Landmark] = field(default_factory=list)
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2)
    wrist_pos: Optional[Tuple[int, int]] = None        # pixel (x,y)
    fingertip_positions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class PoseResult:
    """Combined pose + hand estimation result for a single frame."""
    person_detected: bool = False
    person_confidence: float = 0.0
    body_landmarks: List[Landmark] = field(default_factory=list)
    hands: List[HandResult] = field(default_factory=list)
    # Key body part positions in pixel coords
    left_wrist: Optional[Tuple[int, int]] = None
    right_wrist: Optional[Tuple[int, int]] = None
    body_bbox: Optional[Tuple[int, int, int, int]] = None


# ── Abstract base ───────────────────────────────────────────────────────────

class BasePoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray) -> PoseResult:
        ...

    def draw_pose(self, frame: np.ndarray, result: PoseResult) -> np.ndarray:
        """Draw pose landmarks on frame."""
        annotated = frame.copy()
        if not result.person_detected:
            cv2.putText(annotated, "No person detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated

        h, w = frame.shape[:2]

        # Draw body landmarks
        for lm in result.body_landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            if lm.visibility > 0.5:
                cv2.circle(annotated, (px, py), 3, (0, 255, 0), -1)

        # Draw hand landmarks
        for hand in result.hands:
            color = (255, 0, 0) if hand.handedness == "Left" else (0, 0, 255)
            for lm in hand.landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (px, py), 2, color, -1)
            if hand.wrist_pos:
                cv2.circle(annotated, hand.wrist_pos, 6, color, 2)
                label = f"{hand.handedness} hand"
                cv2.putText(annotated, label,
                            (hand.wrist_pos[0] + 10, hand.wrist_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw wrist positions
        if result.left_wrist:
            cv2.circle(annotated, result.left_wrist, 8, (255, 200, 0), 2)
        if result.right_wrist:
            cv2.circle(annotated, result.right_wrist, 8, (0, 200, 255), 2)

        return annotated


# ── MediaPipe Tasks API Implementation ─────────────────────────────────────

class MediaPipePoseEstimator(BasePoseEstimator):
    """
    Uses the new MediaPipe Tasks API (PoseLandmarker + HandLandmarker)
    for person-level and hand-level detection.
    """

    def __init__(self):
        self._pose_landmarker = None
        self._hand_landmarker = None
        self._initialized = False

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision, BaseOptions

            # Download model files if not present
            model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
            os.makedirs(model_dir, exist_ok=True)

            pose_model = os.path.join(model_dir, "pose_landmarker_lite.task")
            hand_model = os.path.join(model_dir, "hand_landmarker.task")

            # Download models if not present
            self._download_model(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                pose_model,
            )
            self._download_model(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                hand_model,
            )

            # Create Pose Landmarker
            pose_options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

            # Create Hand Landmarker
            hand_options = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=hand_model),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

            self._initialized = True
            logger.info("MediaPipe Tasks pose+hand estimator initialized")

        except Exception as e:
            logger.warning("MediaPipe Tasks init failed: %s", e)
            logger.info("Pose estimation will be unavailable — hand-object interaction "
                       "still works via object detection proximity")

    def _download_model(self, url: str, path: str):
        """Download model file if not present."""
        if os.path.exists(path):
            return
        logger.info("Downloading model: %s", os.path.basename(path))
        try:
            import urllib.request
            urllib.request.urlretrieve(url, path)
            logger.info("Downloaded: %s", os.path.basename(path))
        except Exception as e:
            logger.warning("Model download failed: %s", e)

    def estimate(self, frame: np.ndarray) -> PoseResult:
        if not self._initialized:
            return PoseResult()

        import mediapipe as mp

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = PoseResult()

        # Run pose detection
        try:
            pose_detection = self._pose_landmarker.detect(mp_image)

            if pose_detection.pose_landmarks and len(pose_detection.pose_landmarks) > 0:
                landmarks = pose_detection.pose_landmarks[0]
                pose_result.person_detected = True
                pose_result.person_confidence = 0.85

                body_lms = []
                xs, ys = [], []
                for i, lm in enumerate(landmarks):
                    vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility else 0.5
                    body_lms.append(Landmark(
                        x=lm.x, y=lm.y,
                        z=lm.z if hasattr(lm, 'z') else 0.0,
                        visibility=vis,
                    ))
                    if vis > 0.3:
                        xs.append(lm.x)
                        ys.append(lm.y)

                pose_result.body_landmarks = body_lms

                # Extract wrist positions (indices 15=left wrist, 16=right wrist)
                if len(landmarks) > 16:
                    lw = landmarks[15]
                    rw = landmarks[16]
                    lw_vis = lw.visibility if hasattr(lw, 'visibility') and lw.visibility else 0.5
                    rw_vis = rw.visibility if hasattr(rw, 'visibility') and rw.visibility else 0.5
                    if lw_vis > 0.3:
                        pose_result.left_wrist = (int(lw.x * w), int(lw.y * h))
                    if rw_vis > 0.3:
                        pose_result.right_wrist = (int(rw.x * w), int(rw.y * h))

                # Body bounding box
                if xs and ys:
                    pose_result.body_bbox = (
                        int(min(xs) * w), int(min(ys) * h),
                        int(max(xs) * w), int(max(ys) * h),
                    )
        except Exception as e:
            logger.debug("Pose detection error: %s", e)

        # Run hand detection
        try:
            hand_detection = self._hand_landmarker.detect(mp_image)

            if hand_detection.hand_landmarks:
                for i, hand_landmarks in enumerate(hand_detection.hand_landmarks):
                    handedness = "Right"
                    if hand_detection.handedness and i < len(hand_detection.handedness):
                        cats = hand_detection.handedness[i]
                        if cats:
                            handedness = cats[0].category_name

                    hand = self._extract_hand(hand_landmarks, handedness, w, h)
                    pose_result.hands.append(hand)

                    # Update wrist positions from hand detection (more accurate)
                    if hand.wrist_pos:
                        if handedness == "Left":
                            pose_result.left_wrist = hand.wrist_pos
                        else:
                            pose_result.right_wrist = hand.wrist_pos

        except Exception as e:
            logger.debug("Hand detection error: %s", e)

        return pose_result

    def _extract_hand(self, hand_landmarks, handedness: str,
                      w: int, h: int) -> HandResult:
        landmarks = []
        xs, ys = [], []
        for lm in hand_landmarks:
            landmarks.append(Landmark(x=lm.x, y=lm.y, z=lm.z if hasattr(lm, 'z') else 0.0))
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        wrist = hand_landmarks[0]
        wrist_pos = (int(wrist.x * w), int(wrist.y * h))

        # Fingertip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
        fingertips = []
        for idx in [4, 8, 12, 16, 20]:
            if idx < len(hand_landmarks):
                ft = hand_landmarks[idx]
                fingertips.append((int(ft.x * w), int(ft.y * h)))

        bbox = None
        if xs and ys:
            bbox = (min(xs), min(ys), max(xs), max(ys))

        return HandResult(
            handedness=handedness,
            landmarks=landmarks,
            bbox=bbox,
            wrist_pos=wrist_pos,
            fingertip_positions=fingertips,
        )

    def __del__(self):
        try:
            if self._pose_landmarker:
                self._pose_landmarker.close()
            if self._hand_landmarker:
                self._hand_landmarker.close()
        except Exception:
            pass


# ── Dummy Estimator (no-op fallback) ───────────────────────────────────────

class DummyPoseEstimator(BasePoseEstimator):
    """No-op pose estimator when MediaPipe is unavailable."""

    def estimate(self, frame: np.ndarray) -> PoseResult:
        return PoseResult()


# ── Factory ─────────────────────────────────────────────────────────────────

def create_pose_estimator(mode: str = "mediapipe") -> BasePoseEstimator:
    """
    Factory function.
    
    Args:
        mode: "mediapipe" for MediaPipe Tasks API
    """
    if mode == "mediapipe":
        estimator = MediaPipePoseEstimator()
        if estimator._initialized:
            return estimator
        logger.warning("MediaPipe init failed — using dummy pose estimator")
        return DummyPoseEstimator()
    else:
        return DummyPoseEstimator()
