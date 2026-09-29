"""
Experiment State Machine — Stage 4 (Experiment State & Validation)

A finite-state machine that tracks experiment progress against the SOP.
Each step can be in one of these states:
  PENDING, IN_PROGRESS, VALID, SKIPPED, OUT_OF_ORDER, UNCERTAIN, TIMEOUT

v2 additions:
  - World-state tracking via initial_state / postconditions
  - Precondition checking before step transitions
  - next_step-aware advancement (supports "COMPLETE" sentinel)

The FSM emits status transitions that drive:
  - Voice alerts (on SKIPPED, OUT_OF_ORDER, TIMEOUT)
  - GUI updates (all transitions)
  - Log entries (all transitions)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from .sop_loader import SOP, SOPStep
from ..perception.interaction import ActionState, InteractionResult

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Status labels matching Section 10 of the spec."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    VALID = "VALID"
    SKIPPED = "SKIPPED"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    UNCERTAIN = "UNCERTAIN"
    TIMEOUT = "TIMEOUT"


@dataclass
class StepState:
    """Runtime state for a single SOP step."""
    step: SOPStep
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class TransitionEvent:
    """Emitted when a step's status changes."""
    step_id: int
    label: str
    description: str
    old_status: StepStatus
    new_status: StepStatus
    confidence: float
    evidence: List[str]
    timestamp: float
    requires_alert: bool = False   # True for SKIPPED, OUT_OF_ORDER, TIMEOUT
    alert_message: str = ""


@dataclass
class ExperimentState:
    """Complete experiment state snapshot (sent to GUI via WebSocket)."""
    experiment_id: str
    experiment_name: str
    current_step_id: int
    current_step_label: str
    current_step_description: str
    current_step_status: str
    next_step_description: str
    total_steps: int
    completed_steps: int
    step_states: List[Dict]          # serialized StepState list
    session_started_at: float
    session_elapsed_sec: float
    is_complete: bool = False
    latest_action: str = "idle"
    latest_action_target: str = "none"
    latest_action_confidence: float = 0.0
    detected_objects: List[str] = field(default_factory=list)
    object_states: Dict[str, str] = field(default_factory=dict)
    person_detected: bool = False
    person_confidence: float = 0.0
    # --- v2 ---
    world_state: Dict[str, str] = field(default_factory=dict)


class ExperimentStateMachine:
    """
    Finite-state machine that validates experiment progress against the SOP.

    Usage:
        sop = load_sop("config/sop.json")
        fsm = ExperimentStateMachine(sop)
        fsm.on_transition(my_callback)
        fsm.start_session()
        
        # In the processing loop:
        transition = fsm.update(interaction_result)
    """

    # Minimum confidence to commit to a step transition
    CONFIDENCE_THRESHOLD = 0.4
    # How many consecutive frames of matching evidence before transitioning
    STABILITY_FRAMES = 3

    def __init__(self, sop: SOP):
        self.sop = sop
        self._step_states: List[StepState] = []
        self._current_step_index: int = 0
        self._session_start: float = 0.0
        self._is_running: bool = False
        self._is_complete: bool = False
        self._callbacks: List[Callable[[TransitionEvent], None]] = []
        self._evidence_buffer: Dict[int, int] = {}  # step_id -> consecutive match count
        self._last_interaction: Optional[InteractionResult] = None

        # v2: world state — tracks object/item states across the experiment
        self._world_state: Dict[str, str] = dict(sop.initial_state)

        # Initialize step states
        for step in sop.steps:
            self._step_states.append(StepState(step=step))

    def on_transition(self, callback: Callable[[TransitionEvent], None]):
        """Register a callback for step transitions."""
        self._callbacks.append(callback)

    def start_session(self):
        """Start a new experiment session."""
        self._session_start = time.time()
        self._is_running = True
        self._is_complete = False
        self._current_step_index = 0
        self._evidence_buffer.clear()

        # v2: reset world state to initial
        self._world_state = dict(self.sop.initial_state)

        # Reset all steps
        for ss in self._step_states:
            ss.status = StepStatus.PENDING
            ss.started_at = None
            ss.completed_at = None
            ss.confidence = 0.0
            ss.evidence.clear()

        # Mark first step as in-progress
        if self._step_states:
            self._transition_step(0, StepStatus.IN_PROGRESS, 0.0, ["session_started"])

        logger.info("Experiment session started: %s", self.sop.experiment_name)

    def stop_session(self):
        """Stop the current session."""
        self._is_running = False
        logger.info("Experiment session stopped")

    def update(self, interaction: InteractionResult) -> Optional[TransitionEvent]:
        """
        Feed new interaction data into the FSM. Returns a TransitionEvent
        if a step status changed, None otherwise.

        This is called once per processed frame from the main pipeline loop.
        """
        if not self._is_running or self._is_complete:
            return None

        self._last_interaction = interaction

        current_ss = self._step_states[self._current_step_index]
        current_step = current_ss.step

        # Check for timeout on current step
        if current_ss.status == StepStatus.IN_PROGRESS and current_ss.started_at:
            elapsed = time.time() - current_ss.started_at
            max_duration = current_step.expected_duration_sec[1]
            if elapsed > max_duration:
                return self._transition_step(
                    self._current_step_index, StepStatus.TIMEOUT,
                    0.0, [f"timeout_after_{max_duration}s"],
                )

        # Try to match current evidence to SOP steps
        matched_step_index, match_confidence, match_evidence = self._match_evidence(interaction)

        if matched_step_index is None:
            # No clear match — might be transitional motion or idle
            if current_ss.status == StepStatus.IN_PROGRESS:
                # Stay in progress, don't change
                pass
            return None

        # Accumulate evidence stability
        step_id = self.sop.steps[matched_step_index].step_id
        self._evidence_buffer[step_id] = self._evidence_buffer.get(step_id, 0) + 1

        # Clear other step evidence when one step gets stronger
        for sid in list(self._evidence_buffer.keys()):
            if sid != step_id:
                self._evidence_buffer[sid] = max(0, self._evidence_buffer[sid] - 1)

        # Need enough consecutive frames before committing
        if self._evidence_buffer[step_id] < self.STABILITY_FRAMES:
            return None

        if match_confidence < self.CONFIDENCE_THRESHOLD:
            return None

        # Reset buffer after committing
        self._evidence_buffer[step_id] = 0

        # Determine transition type
        if matched_step_index == self._current_step_index:
            # Evidence matches current step — it's being completed
            if current_ss.status != StepStatus.VALID:
                return self._complete_current_step(match_confidence, match_evidence)

        elif matched_step_index == self._current_step_index + 1:
            # Evidence matches the NEXT step — current is being completed, advance
            event = self._complete_current_step(match_confidence, match_evidence)
            return event

        elif matched_step_index > self._current_step_index + 1:
            # Skipped one or more steps!
            return self._handle_skip(matched_step_index, match_confidence, match_evidence)

        elif matched_step_index < self._current_step_index:
            # Going backwards — out of order
            return self._handle_out_of_order(matched_step_index, match_confidence, match_evidence)

        return None

    def _match_evidence(self, interaction: InteractionResult):
        """
        Try to match the current interaction evidence to one of the SOP steps.
        Returns (step_index, confidence, evidence_list) or (None, 0, []).
        """
        action = interaction.current_action
        objects_in_scene = interaction.objects_in_scene
        object_states = interaction.object_states

        if action.action == "idle":
            return None, 0.0, []

        best_match_index = None
        best_confidence = 0.0
        best_evidence = []

        for i, step in enumerate(self.sop.steps):
            score = 0.0
            evidence = []
            max_possible = 3.0  # action + objects + states

            # 1. Action match
            if self._action_matches(action.action, step.expected_action):
                score += 1.0
                evidence.append(f"action_match:{action.action}")

            # 2. Required objects present
            objects_found = sum(1 for obj in step.required_objects if obj in objects_in_scene)
            if step.required_objects:
                obj_ratio = objects_found / len(step.required_objects)
                score += obj_ratio
                if obj_ratio > 0:
                    evidence.append(f"objects_found:{objects_found}/{len(step.required_objects)}")

            # 3. Object state match
            if step.expected_object_states:
                states_matched = 0
                for obj, expected_state in step.expected_object_states.items():
                    actual_state = object_states.get(obj, "unknown")
                    if actual_state == expected_state or self._state_compatible(actual_state, expected_state):
                        states_matched += 1
                state_ratio = states_matched / len(step.expected_object_states)
                score += state_ratio
                if state_ratio > 0:
                    evidence.append(f"state_match:{states_matched}/{len(step.expected_object_states)}")

            # 4. Also match the target object
            if action.target_object in step.required_objects:
                score += 0.5
                evidence.append(f"target_object_match:{action.target_object}")
                max_possible += 0.5

            confidence = score / max_possible if max_possible > 0 else 0.0

            if confidence > best_confidence and confidence > 0.2:
                best_match_index = i
                best_confidence = confidence
                best_evidence = evidence

        return best_match_index, round(best_confidence, 3), best_evidence

    def _action_matches(self, observed: str, expected: str) -> bool:
        """Check if observed action matches expected, with flexible matching."""
        if observed == expected:
            return True

        # Flexible matches
        matches = {
            ("grasp_remove", "grasp_remove"): True,
            ("grasping", "grasp_remove"): True,
            ("holding", "grasp_remove"): True,
            ("open", "open"): True,
            ("interact", "open"): True,
            ("close", "close"): True,
            ("interact", "close"): True,
            ("place", "place"): True,
            ("reaching", "grasp_remove"): True,
        }
        return matches.get((observed, expected), False)

    def _state_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual state is compatible with expected."""
        compatible = {
            ("in_hand", "outside_large_box"): True,
            ("moving", "outside_large_box"): True,
            ("detected", "outside_large_box"): True,
            ("in_hand", "in_hand"): True,
            ("grasping", "in_hand"): True,
        }
        return compatible.get((actual, expected), False)

    def _check_preconditions(self, step_index: int) -> bool:
        """Check whether the world state satisfies a step's preconditions.

        Returns True if all preconditions are met (or there are none).
        This is informational in v2 — the FSM logs warnings but does not
        block transitions, preserving existing v1 behaviour.
        """
        step = self._step_states[step_index].step
        if not step.preconditions:
            return True

        for key, expected in step.preconditions.items():
            actual = self._world_state.get(key, "unknown")
            if actual != expected:
                logger.warning(
                    "Step %d [%s] precondition UNMET: %s expected '%s', world has '%s'",
                    step.step_id, step.label, key, expected, actual,
                )
                return False
        return True

    def _apply_postconditions(self, step_index: int) -> None:
        """Apply a step's postconditions to the world state."""
        step = self._step_states[step_index].step
        if not step.postconditions:
            return
        for key, value in step.postconditions.items():
            old = self._world_state.get(key, "unknown")
            self._world_state[key] = value
            if old != value:
                logger.info(
                    "World state: %s: '%s' → '%s' (step %d)",
                    key, old, value, step.step_id,
                )

    def _complete_current_step(self, confidence: float,
                                evidence: List[str]) -> Optional[TransitionEvent]:
        """Mark current step as VALID and advance to next."""
        # v2: check preconditions (informational)
        self._check_preconditions(self._current_step_index)

        event = self._transition_step(
            self._current_step_index, StepStatus.VALID,
            confidence, evidence,
        )

        # v2: apply postconditions to world state
        self._apply_postconditions(self._current_step_index)

        # Advance to next step — v2 uses next_step field if available
        current_step = self._step_states[self._current_step_index].step
        if current_step.next_step == "COMPLETE":
            self._is_complete = True
            logger.info("Experiment COMPLETE — all steps validated")
        elif current_step.next_step is not None and isinstance(current_step.next_step, int):
            # Find the index of the next_step by step_id
            next_index = None
            for idx, ss in enumerate(self._step_states):
                if ss.step.step_id == current_step.next_step:
                    next_index = idx
                    break
            if next_index is not None:
                self._current_step_index = next_index
                self._transition_step(
                    self._current_step_index, StepStatus.IN_PROGRESS,
                    0.0, ["auto_advance"],
                )
            else:
                logger.error(
                    "next_step %d not found in SOP — falling back to sequential",
                    current_step.next_step,
                )
                if self._current_step_index + 1 < len(self._step_states):
                    self._current_step_index += 1
                    self._transition_step(
                        self._current_step_index, StepStatus.IN_PROGRESS,
                        0.0, ["auto_advance"],
                    )
                else:
                    self._is_complete = True
                    logger.info("Experiment COMPLETE — all steps validated")
        else:
            # v1 fallback: sequential advancement
            if self._current_step_index + 1 < len(self._step_states):
                self._current_step_index += 1
                self._transition_step(
                    self._current_step_index, StepStatus.IN_PROGRESS,
                    0.0, ["auto_advance"],
                )
            else:
                self._is_complete = True
                logger.info("Experiment COMPLETE — all steps validated")

        return event

    def _handle_skip(self, matched_index: int, confidence: float,
                     evidence: List[str]) -> TransitionEvent:
        """Handle when evidence shows steps were skipped."""
        # Mark skipped steps
        for i in range(self._current_step_index, matched_index):
            if self._step_states[i].status != StepStatus.VALID:
                self._transition_step(i, StepStatus.SKIPPED, 0.0,
                                      [f"skipped_in_favor_of_step_{matched_index + 1}"])

        # The matched step is out of order (since prior steps were skipped)
        event = self._transition_step(
            matched_index, StepStatus.OUT_OF_ORDER,
            confidence, evidence + ["prior_steps_skipped"],
        )

        self._current_step_index = matched_index
        return event

    def _handle_out_of_order(self, matched_index: int, confidence: float,
                              evidence: List[str]) -> TransitionEvent:
        """Handle when evidence matches a previous step (going backwards)."""
        return self._transition_step(
            matched_index, StepStatus.OUT_OF_ORDER,
            confidence, evidence + ["going_backwards"],
        )

    def _transition_step(self, step_index: int, new_status: StepStatus,
                          confidence: float, evidence: List[str]) -> TransitionEvent:
        """Execute a step status transition and fire callbacks."""
        ss = self._step_states[step_index]
        old_status = ss.status
        ss.status = new_status
        ss.confidence = confidence
        ss.evidence = evidence
        now = time.time()

        if new_status == StepStatus.IN_PROGRESS and ss.started_at is None:
            ss.started_at = now
        elif new_status in (StepStatus.VALID, StepStatus.SKIPPED,
                            StepStatus.OUT_OF_ORDER, StepStatus.TIMEOUT):
            ss.completed_at = now

        # Build alert message
        requires_alert = new_status in (
            StepStatus.SKIPPED, StepStatus.OUT_OF_ORDER, StepStatus.TIMEOUT,
        )
        alert_message = ""
        if new_status == StepStatus.SKIPPED:
            alert_message = f"Warning: Step {ss.step.step_id} was skipped — {ss.step.description}. Please complete it before continuing."
        elif new_status == StepStatus.OUT_OF_ORDER:
            alert_message = f"Warning: Step {ss.step.step_id} is out of order — {ss.step.description}."
        elif new_status == StepStatus.TIMEOUT:
            alert_message = f"Warning: Step {ss.step.step_id} timed out — {ss.step.description}. Please proceed."

        event = TransitionEvent(
            step_id=ss.step.step_id,
            label=ss.step.label,
            description=ss.step.description,
            old_status=old_status,
            new_status=new_status,
            confidence=confidence,
            evidence=evidence,
            timestamp=now,
            requires_alert=requires_alert,
            alert_message=alert_message,
        )

        # Fire callbacks
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error("Transition callback error: %s", e)

        logger.info("Step %d [%s]: %s → %s (conf=%.2f)",
                     ss.step.step_id, ss.step.label, old_status.value,
                     new_status.value, confidence)

        return event

    def get_state(self) -> ExperimentState:
        """Get the current full experiment state snapshot for the GUI."""
        current_ss = self._step_states[self._current_step_index] if self._step_states else None
        next_step = self.sop.get_next_step(
            current_ss.step.step_id if current_ss else 0
        )
        completed = sum(1 for ss in self._step_states if ss.status == StepStatus.VALID)

        # Get latest interaction info
        latest_action = "idle"
        latest_target = "none"
        latest_conf = 0.0
        detected_objects: List[str] = []
        object_states: Dict[str, str] = {}
        person_detected = False
        person_conf = 0.0

        if self._last_interaction:
            latest_action = self._last_interaction.current_action.action
            latest_target = self._last_interaction.current_action.target_object
            latest_conf = self._last_interaction.current_action.confidence
            detected_objects = self._last_interaction.objects_in_scene
            object_states = self._last_interaction.object_states

        step_states_data = []
        for ss in self._step_states:
            step_data = {
                "step_id": ss.step.step_id,
                "label": ss.step.label,
                "description": ss.step.description,
                "status": ss.status.value,
                "confidence": ss.confidence,
                "evidence": ss.evidence,
                "voice_prompt": ss.step.voice_prompt,
            }
            # v2: include action_type, preconditions, postconditions if present
            if ss.step.action_type:
                step_data["action_type"] = ss.step.action_type
            if ss.step.preconditions:
                step_data["preconditions"] = ss.step.preconditions
            if ss.step.postconditions:
                step_data["postconditions"] = ss.step.postconditions
            step_states_data.append(step_data)

        now = time.time()
        return ExperimentState(
            experiment_id=self.sop.experiment_id,
            experiment_name=self.sop.experiment_name,
            current_step_id=current_ss.step.step_id if current_ss else 0,
            current_step_label=current_ss.step.label if current_ss else "",
            current_step_description=current_ss.step.description if current_ss else "",
            current_step_status=current_ss.status.value if current_ss else "PENDING",
            next_step_description=next_step.description if next_step else "Experiment complete",
            total_steps=self.sop.total_steps,
            completed_steps=completed,
            step_states=step_states_data,
            session_started_at=self._session_start,
            session_elapsed_sec=round(now - self._session_start, 1) if self._session_start else 0.0,
            is_complete=self._is_complete,
            latest_action=latest_action,
            latest_action_target=latest_target,
            latest_action_confidence=latest_conf,
            detected_objects=detected_objects,
            object_states=object_states,
            person_detected=person_detected,
            person_confidence=person_conf,
            world_state=dict(self._world_state),
        )

    def force_complete_step(self, step_id: int) -> Optional[TransitionEvent]:
        """
        Manually mark a step as complete (for demo / testing).
        Useful during demo to advance steps with keyboard.
        """
        for i, ss in enumerate(self._step_states):
            if ss.step.step_id == step_id and ss.status != StepStatus.VALID:
                event = self._complete_current_step(1.0, ["manual_override"])
                return event
        return None

    def force_skip_to_step(self, step_id: int) -> Optional[TransitionEvent]:
        """
        Skip directly to a step (for demo / testing).
        """
        for i, ss in enumerate(self._step_states):
            if ss.step.step_id == step_id:
                # Skip everything before
                for j in range(self._current_step_index, i):
                    self._transition_step(j, StepStatus.SKIPPED, 0.0, ["manual_skip"])
                self._current_step_index = i
                return self._transition_step(i, StepStatus.IN_PROGRESS, 0.0, ["manual_jump"])
        return None
