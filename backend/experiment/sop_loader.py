"""
SOP Loader — loads and validates the experiment procedure from config/sop.json

The SOP file is the single source of truth for the experiment.  All step
logic in the state machine and validation code references this loaded data,
never hard-coded step IDs or labels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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


@dataclass
class SOP:
    """The full Standard Operating Procedure."""
    experiment_id: str
    experiment_name: str
    description: str
    steps: List[SOPStep]

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
        """Get the step after the given step ID."""
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
            lines.append(f"     Expected duration: {step.expected_duration_sec[0]}-{step.expected_duration_sec[1]} seconds")
            lines.append("")
        return "\n".join(lines)


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

    # Parse steps
    steps = []
    for i, step_data in enumerate(data["steps"]):
        for req_field in ("step_id", "label", "description", "expected_action"):
            if req_field not in step_data:
                raise ValueError(f"Step {i + 1} missing required field: {req_field}")

        duration = step_data.get("expected_duration_sec", [2, 15])
        if isinstance(duration, list) and len(duration) == 2:
            duration = tuple(duration)
        else:
            duration = (2, 15)

        steps.append(SOPStep(
            step_id=step_data["step_id"],
            label=step_data["label"],
            description=step_data["description"],
            required_objects=step_data.get("required_objects", []),
            expected_action=step_data["expected_action"],
            expected_object_states=step_data.get("expected_object_states", {}),
            expected_duration_sec=duration,
            voice_prompt=step_data.get("voice_prompt", f"Step {step_data['step_id']}: {step_data['description']}"),
        ))

    sop = SOP(
        experiment_id=data["experiment_id"],
        experiment_name=data["experiment_name"],
        description=data.get("description", ""),
        steps=steps,
    )

    logger.info("Loaded SOP '%s' with %d steps", sop.experiment_name, sop.total_steps)
    return sop
