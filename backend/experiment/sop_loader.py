"""
SOP Loader — loads and validates the experiment procedure from config/sop.json

The SOP file is the single source of truth for the experiment.  All step
logic in the state machine and validation code references this loaded data,
never hard-coded step IDs or labels.

v2 additions:
  - initial_state: world state before step 1
  - Per-step: action_type, object, source, target, preconditions,
    postconditions, next_step
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# --- Valid enumerations (v2) ---
VALID_ACTION_TYPES = frozenset({"BACKGROUND", "OPEN", "GRASP_REMOVE", "PLACE", "CLOSE"})
VALID_OBJECTS = frozenset({"astronaut", "large_box", "red_box", "yellow_box", "sample_item"})


@dataclass
class SOPStep:
    """A single step in the Standard Operating Procedure."""
    step_id: int
    label: str
    description: str
    required_objects: List[str]
    expected_action: str
    expected_object_states: Dict[str, str] = field(default_factory=dict)
    expected_duration_sec: Tuple[int, int] = (2, 15)
    voice_prompt: str = ""

    # --- v2 fields ---
    action_type: str = ""                               # e.g. "OPEN", "GRASP_REMOVE"
    object: str = ""                                    # primary object acted upon
    source: Optional[str] = None                        # where the object comes from
    target: Optional[str] = None                        # where the object goes
    preconditions: Dict[str, str] = field(default_factory=dict)   # world state required
    postconditions: Dict[str, str] = field(default_factory=dict)  # world state after step
    next_step: Union[int, str, None] = None             # step_id or "COMPLETE"


@dataclass
class SOP:
    """The full Standard Operating Procedure."""
    experiment_id: str
    experiment_name: str
    description: str
    steps: List[SOPStep]
    version: str = "1.0"

    # --- v2 fields ---
    initial_state: Dict[str, str] = field(default_factory=dict)
    objects: List[str] = field(default_factory=list)
    action_types: List[str] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def get_step(self, step_id: int) -> Optional[SOPStep]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_step_by_label(self, label: str) -> Optional[SOPStep]:
        """Get a step by its label."""
        for step in self.steps:
            if step.label == label:
                return step
        return None

    def get_next_step(self, current_step_id: int) -> Optional[SOPStep]:
        """Get the step after the given step ID.

        v2-aware: uses next_step field if available, otherwise falls back to
        sequential ordering.
        """
        current = self.get_step(current_step_id)
        if current and current.next_step is not None:
            if isinstance(current.next_step, int):
                return self.get_step(current.next_step)
            # "COMPLETE" — no next step
            return None

        # Fallback to sequential ordering (v1 behaviour)
        for i, step in enumerate(self.steps):
            if step.step_id == current_step_id and i + 1 < len(self.steps):
                return self.steps[i + 1]
        return None

    def get_all_required_objects(self) -> List[str]:
        """Get all unique objects referenced in the SOP."""
        objects = set()
        for step in self.steps:
            objects.update(step.required_objects)
        return sorted(objects)

    def get_full_text(self) -> str:
        """Get the full SOP as readable text (for the Q&A assistant)."""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Description: {self.description}",
            "",
            "Steps:",
        ]
        for step in self.steps:
            lines.append(f"  {step.step_id}. {step.description}")
            lines.append(f"     Required objects: {', '.join(step.required_objects)}")
            lines.append(f"     Expected action: {step.expected_action}")
            if step.action_type:
                lines.append(f"     Action type: {step.action_type}")
            if step.preconditions:
                pre = ", ".join(f"{k}=={v}" for k, v in step.preconditions.items())
                lines.append(f"     Preconditions: {pre}")
            if step.postconditions:
                post = ", ".join(f"{k}=={v}" for k, v in step.postconditions.items())
                lines.append(f"     Postconditions: {post}")
            lines.append(f"     Expected duration: {step.expected_duration_sec[0]}-{step.expected_duration_sec[1]} seconds")
            lines.append("")
        return "\n".join(lines)


def _validate_v2_step(step_data: dict, index: int) -> List[str]:
    """Validate v2-specific fields on a single step.  Returns list of warnings."""
    warnings: List[str] = []

    action_type = step_data.get("action_type", "")
    if action_type and action_type not in VALID_ACTION_TYPES:
        warnings.append(
            f"Step {index + 1}: action_type '{action_type}' not in {sorted(VALID_ACTION_TYPES)}"
        )

    obj = step_data.get("object", "")
    if obj and obj not in VALID_OBJECTS:
        warnings.append(
            f"Step {index + 1}: object '{obj}' not in {sorted(VALID_OBJECTS)}"
        )

    source = step_data.get("source")
    if source and source not in VALID_OBJECTS and source != "in_hand":
        warnings.append(
            f"Step {index + 1}: source '{source}' is not a known object or 'in_hand'"
        )

    target = step_data.get("target")
    if target and target not in VALID_OBJECTS:
        warnings.append(
            f"Step {index + 1}: target '{target}' not in {sorted(VALID_OBJECTS)}"
        )

    next_step = step_data.get("next_step")
    if next_step is not None and not isinstance(next_step, int) and next_step != "COMPLETE":
        warnings.append(
            f"Step {index + 1}: next_step must be an int or 'COMPLETE', got '{next_step}'"
        )

    return warnings


def load_sop(config_path: str = "config/sop.json") -> SOP:
    """
    Load the SOP from a JSON file.

    Args:
        config_path: Path to the SOP JSON file

    Returns:
        Parsed SOP object

    Raises:
        FileNotFoundError: if config file doesn't exist
        ValueError: if config file is malformed
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"SOP config not found at {path.absolute()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate required top-level fields
    for field_name in ("experiment_id", "experiment_name", "steps"):
        if field_name not in data:
            raise ValueError(f"SOP config missing required field: {field_name}")

    if not data["steps"]:
        raise ValueError("SOP config has no steps defined")

    # Detect version
    version = data.get("version", "1.0")
    is_v2 = version.startswith("2")

    # Parse initial_state (v2)
    initial_state: Dict[str, str] = data.get("initial_state", {})
    if is_v2 and not initial_state:
        logger.warning("SOP v2 config has no initial_state — world state tracking will be limited")

    # Parse steps
    steps: List[SOPStep] = []
    all_warnings: List[str] = []

    for i, step_data in enumerate(data["steps"]):
        for req_field in ("step_id", "label", "description", "expected_action"):
            if req_field not in step_data:
                raise ValueError(f"Step {i + 1} missing required field: {req_field}")

        duration = step_data.get("expected_duration_sec", [2, 15])
        if isinstance(duration, list) and len(duration) == 2:
            duration = tuple(duration)
        else:
            duration = (2, 15)

        # v2 validation
        if is_v2:
            all_warnings.extend(_validate_v2_step(step_data, i))

        # Parse next_step — can be int, "COMPLETE", or absent
        raw_next = step_data.get("next_step")
        if isinstance(raw_next, str) and raw_next != "COMPLETE":
            try:
                raw_next = int(raw_next)
            except ValueError:
                raise ValueError(
                    f"Step {i + 1}: next_step '{raw_next}' is not an integer or 'COMPLETE'"
                )

        steps.append(SOPStep(
            step_id=step_data["step_id"],
            label=step_data["label"],
            description=step_data["description"],
            required_objects=step_data.get("required_objects", []),
            expected_action=step_data["expected_action"],
            expected_object_states=step_data.get("expected_object_states", {}),
            expected_duration_sec=duration,
            voice_prompt=step_data.get("voice_prompt", f"Step {step_data['step_id']}: {step_data['description']}"),
            # v2 fields (default gracefully for v1 configs)
            action_type=step_data.get("action_type", ""),
            object=step_data.get("object", ""),
            source=step_data.get("source"),
            target=step_data.get("target"),
            preconditions=step_data.get("preconditions", {}),
            postconditions=step_data.get("postconditions", {}),
            next_step=raw_next,
        ))

    # Log any validation warnings
    for w in all_warnings:
        logger.warning("SOP validation: %s", w)

    sop = SOP(
        experiment_id=data["experiment_id"],
        experiment_name=data["experiment_name"],
        description=data.get("description", ""),
        steps=steps,
        version=version,
        initial_state=initial_state,
        objects=data.get("objects", []),
        action_types=data.get("action_types", []),
    )

    logger.info(
        "Loaded SOP '%s' v%s with %d steps, %d initial_state keys",
        sop.experiment_name, sop.version, sop.total_steps, len(sop.initial_state),
    )
    return sop
