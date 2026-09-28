"""
Hand-Object Interaction Tracker — Stage 3 (Spatial + Temporal Understanding)

Rule-based heuristic that fuses pose (hand wrist/fingertip positions) with
object detections to determine:
  - Which hand is near which object
  - Whether a grasp/place/open/close action is likely happening
  - Sliding-window temporal smoothing over the last N frames

This replaces ST-GCN++ / Mamba-2 for v0.1 — the interface is designed so a
trained model can be swapped in later without touching downstream code.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detector import Detection, DetectionResult
from .pose import PoseResult

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class InteractionEvent:
    """A single detected hand↔object interaction."""
    hand: str                    # "left" or "right"
    object_label: str            # e.g. "red_box"
    interaction_type: str        # "near", "touching", "grasping"
    distance: float              # pixels between hand and object center
    confidence: float            # 0.0 – 1.0


@dataclass
class ActionState:
    """Inferred action state from temporal analysis."""
    action: str                  # e.g. "reaching", "grasping", "placing", "opening", "closing", "idle"
    target_object: str           # which object the action is on
    confidence: float            # 0.0 – 1.0
    evidence: List[str] = field(default_factory=list)  # human-readable evidence strings


@dataclass
class InteractionResult:
    """Full interaction analysis for a single frame."""
    interactions: List[InteractionEvent] = field(default_factory=list)
    current_action: ActionState = field(default_factory=lambda: ActionState("idle", "none", 0.0))
    objects_in_scene: List[str] = field(default_factory=list)
    object_states: Dict[str, str] = field(default_factory=dict)  # label -> state


# ── Interaction Tracker ─────────────────────────────────────────────────────

class InteractionTracker:
    """
    Rule-based hand↔object interaction tracker with temporal smoothing.

    Distance thresholds (in pixels at 640×480):
      - NEAR:     < 120 px  — hand is approaching the object
      - TOUCHING: < 60 px   — hand is on/overlapping the object
      - GRASPING: < 40 px + hand closing heuristic
    """

    NEAR_THRESHOLD = 120
    TOUCHING_THRESHOLD = 60
    GRASP_THRESHOLD = 40

    WINDOW_SIZE = 15  # frames of temporal history

    def __init__(self, window_size: int = 15):
        self.WINDOW_SIZE = window_size
        self._history: deque = deque(maxlen=window_size)
        self._action_history: deque = deque(maxlen=window_size)
        self._prev_object_positions: Dict[str, Tuple[int, int]] = {}
        self._object_states: Dict[str, str] = {}
        self._last_update = time.time()

    def update(self, detections: DetectionResult, pose: PoseResult) -> InteractionResult:
        """
        Analyze hand-object interactions for the current frame.
        
        Args:
            detections: Object detection results
            pose: Pose estimation results
            
        Returns:
            InteractionResult with interactions, action state, and object states
        """
        result = InteractionResult()

        # Collect hand positions
        hand_positions: Dict[str, Tuple[int, int]] = {}
        if pose.left_wrist:
            hand_positions["left"] = pose.left_wrist
        if pose.right_wrist:
            hand_positions["right"] = pose.right_wrist

        # Also add fingertip positions for finer-grained interaction
        fingertip_positions: Dict[str, List[Tuple[int, int]]] = {"left": [], "right": []}
        for hand in pose.hands:
            key = hand.handedness.lower()
            if key in fingertip_positions:
                fingertip_positions[key] = hand.fingertip_positions

        # Objects in scene
        result.objects_in_scene = list(set(d.label for d in detections.detections))

        # Compute interactions
        for det in detections.detections:
            for hand_name, wrist_pos in hand_positions.items():
                dist = self._distance(wrist_pos, det.center)

                if dist < self.GRASP_THRESHOLD:
                    itype = "grasping"
                    conf = min(1.0, (self.GRASP_THRESHOLD - dist) / self.GRASP_THRESHOLD + 0.5)
                elif dist < self.TOUCHING_THRESHOLD:
                    itype = "touching"
                    conf = min(1.0, (self.TOUCHING_THRESHOLD - dist) / self.TOUCHING_THRESHOLD + 0.3)
                elif dist < self.NEAR_THRESHOLD:
                    itype = "near"
                    conf = min(1.0, (self.NEAR_THRESHOLD - dist) / self.NEAR_THRESHOLD + 0.1)
                else:
                    continue

                result.interactions.append(InteractionEvent(
                    hand=hand_name,
                    object_label=det.label,
                    interaction_type=itype,
                    distance=round(dist, 1),
                    confidence=round(conf, 3),
                ))

        # Update object states based on interactions and position changes
        self._update_object_states(detections, result.interactions)
        result.object_states = dict(self._object_states)

        # Infer current action from interactions + temporal context
        self._history.append(result.interactions)
        result.current_action = self._infer_action(result.interactions, detections)
        self._action_history.append(result.current_action)

        # Track object positions for motion detection
        for det in detections.detections:
            self._prev_object_positions[det.label] = det.center

        self._last_update = time.time()
        return result

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

    def _update_object_states(self, detections: DetectionResult,
                               interactions: List[InteractionEvent]):
        """Update tracked object states based on interactions."""
        for det in detections.detections:
            label = det.label
            # Initialize state if new
            if label not in self._object_states:
                self._object_states[label] = det.state if det.state != "unknown" else "detected"

            # Check if object is being grasped
            grasping = any(
                i.object_label == label and i.interaction_type == "grasping"
                for i in interactions
            )
            if grasping:
                self._object_states[label] = "in_hand"

            # Check position changes (object moved)
            if label in self._prev_object_positions:
                prev = self._prev_object_positions[label]
                curr = det.center
                movement = self._distance(prev, curr)
                if movement > 30:  # significant movement
                    if self._object_states[label] not in ("in_hand",):
                        self._object_states[label] = "moving"

            # Use detector's spatial state if available
            if det.state in ("inside_large_box", "outside_large_box"):
                if self._object_states[label] != "in_hand":
                    self._object_states[label] = det.state

    def _infer_action(self, interactions: List[InteractionEvent],
                      detections: DetectionResult) -> ActionState:
        """
        Infer the current high-level action from interactions and temporal history.
        Uses a simple priority-based rule system.
        """
        evidence = []

        # Check for grasping
        grasps = [i for i in interactions if i.interaction_type == "grasping"]
        if grasps:
            best_grasp = max(grasps, key=lambda i: i.confidence)
            evidence.append(f"hand_{best_grasp.hand}_grasping_{best_grasp.object_label}")

            # Check temporal: was the hand previously NOT grasping? → "grasp_start"
            recent_grasps = self._count_recent_grasps(best_grasp.object_label)

            if recent_grasps < 3:
                return ActionState(
                    action="grasp_remove",
                    target_object=best_grasp.object_label,
                    confidence=round(best_grasp.confidence, 3),
                    evidence=evidence,
                )
            else:
                # Sustained grasp — could be placing or holding
                return ActionState(
                    action="holding",
                    target_object=best_grasp.object_label,
                    confidence=round(best_grasp.confidence * 0.8, 3),
                    evidence=evidence,
                )

        # Check for touching (potential open/close)
        touches = [i for i in interactions if i.interaction_type == "touching"]
        if touches:
            best_touch = max(touches, key=lambda i: i.confidence)
            evidence.append(f"hand_touching_{best_touch.object_label}")

            # Determine if it's an open or close based on state
            obj_state = self._object_states.get(best_touch.object_label, "detected")
            if obj_state in ("detected", "closed"):
                action = "open"
            elif obj_state == "open":
                action = "close"
            else:
                action = "interact"

            return ActionState(
                action=action,
                target_object=best_touch.object_label,
                confidence=round(best_touch.confidence, 3),
                evidence=evidence,
            )

        # Check for reaching (near)
        nears = [i for i in interactions if i.interaction_type == "near"]
        if nears:
            best_near = max(nears, key=lambda i: i.confidence)
            evidence.append(f"hand_near_{best_near.object_label}")
            return ActionState(
                action="reaching",
                target_object=best_near.object_label,
                confidence=round(best_near.confidence, 3),
                evidence=evidence,
            )

        # No interactions detected
        return ActionState(action="idle", target_object="none", confidence=0.9)

    def _count_recent_grasps(self, object_label: str) -> int:
        """Count how many recent frames had a grasp on this object."""
        count = 0
        for frame_interactions in self._history:
            for i in frame_interactions:
                if i.object_label == object_label and i.interaction_type == "grasping":
                    count += 1
                    break
        return count

    def get_temporal_summary(self) -> Dict:
        """Get a summary of recent temporal activity for the GUI."""
        if not self._action_history:
            return {"dominant_action": "idle", "action_stability": 0.0}

        # Find dominant action in the window
        action_counts: Dict[str, int] = {}
        for action_state in self._action_history:
            key = f"{action_state.action}_{action_state.target_object}"
            action_counts[key] = action_counts.get(key, 0) + 1

        if action_counts:
            dominant = max(action_counts, key=action_counts.get)
            stability = action_counts[dominant] / len(self._action_history)
        else:
            dominant = "idle_none"
            stability = 0.0

        return {
            "dominant_action": dominant,
            "action_stability": round(stability, 3),
            "window_size": len(self._action_history),
        }

    def reset(self):
        """Reset all tracking state for a new session."""
        self._history.clear()
        self._action_history.clear()
        self._prev_object_positions.clear()
        self._object_states.clear()
