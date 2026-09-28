"""
Session Logger — writes structured experiment logs.

Produces two files per session:
  1. logs/session_<timestamp>.jsonl   — one JSON object per event (machine-readable)
  2. logs/session_<timestamp>_summary.md — human-readable session report

The .jsonl format matches Section 11 of the spec:
  {"timestamp": "...", "step_id": 3, "label": "...", "status": "VALID", 
   "confidence": 0.91, "evidence": ["..."]}
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """A single log entry matching the spec format."""
    timestamp: str
    step_id: int
    label: str
    status: str
    confidence: float
    evidence: List[str]
    description: str = ""
    elapsed_sec: float = 0.0


class SessionLogger:
    """
    Manages structured logging for an experiment session.

    Usage:
        sl = SessionLogger(output_dir="logs")
        sl.start_session("sample_box_sorting_v1")
        sl.log_event(step_id=1, label="open_large_box", status="VALID", ...)
        sl.end_session()
    """

    def __init__(self, output_dir: str = "logs"):
        self._output_dir = output_dir
        self._session_id: str = ""
        self._jsonl_path: str = ""
        self._jsonl_file = None
        self._entries: List[LogEntry] = []
        self._session_start: float = 0.0
        self._experiment_name: str = ""
        self._is_active: bool = False

        os.makedirs(output_dir, exist_ok=True)

    def start_session(self, experiment_name: str = "") -> str:
        """Start a new logging session. Returns the session ID."""
        self._session_id = time.strftime("%Y%m%d_%H%M%S")
        self._experiment_name = experiment_name
        self._session_start = time.time()
        self._entries.clear()

        self._jsonl_path = os.path.join(
            self._output_dir, f"session_{self._session_id}.jsonl"
        )
        self._jsonl_file = open(self._jsonl_path, "w", encoding="utf-8")
        self._is_active = True

        logger.info("Session logging started: %s", self._jsonl_path)
        return self._session_id

    def log_event(self, step_id: int, label: str, status: str,
                  confidence: float = 0.0, evidence: Optional[List[str]] = None,
                  description: str = "") -> LogEntry:
        """Log a single event to the JSONL file."""
        if not self._is_active:
            logger.warning("Attempted to log event without active session")
            return LogEntry("", 0, "", "", 0.0, [])

        now = datetime.now(timezone.utc)
        elapsed = time.time() - self._session_start

        entry = LogEntry(
            timestamp=now.isoformat(),
            step_id=step_id,
            label=label,
            status=status,
            confidence=round(confidence, 3),
            evidence=evidence or [],
            description=description,
            elapsed_sec=round(elapsed, 1),
        )

        self._entries.append(entry)

        # Write to JSONL
        line = json.dumps({
            "timestamp": entry.timestamp,
            "step_id": entry.step_id,
            "label": entry.label,
            "status": entry.status,
            "confidence": entry.confidence,
            "evidence": entry.evidence,
        })
        self._jsonl_file.write(line + "\n")
        self._jsonl_file.flush()

        return entry

    def end_session(self) -> tuple[str, str]:
        """
        End the session, close the JSONL file, and write the summary.
        Returns (jsonl_path, summary_path).
        """
        if not self._is_active:
            return "", ""

        self._is_active = False

        if self._jsonl_file:
            self._jsonl_file.close()
            self._jsonl_file = None

        # Write summary
        summary_path = os.path.join(
            self._output_dir, f"session_{self._session_id}_summary.md"
        )
        self._write_summary(summary_path)

        logger.info("Session ended. Log: %s, Summary: %s",
                    self._jsonl_path, summary_path)
        return self._jsonl_path, summary_path

    def _write_summary(self, path: str):
        """Write a human-readable session summary."""
        total_duration = time.time() - self._session_start if self._session_start else 0

        # Count statuses
        status_counts: Dict[str, int] = {}
        for entry in self._entries:
            status_counts[entry.status] = status_counts.get(entry.status, 0) + 1

        valid_count = status_counts.get("VALID", 0)
        skipped_count = status_counts.get("SKIPPED", 0)
        ooo_count = status_counts.get("OUT_OF_ORDER", 0)
        timeout_count = status_counts.get("TIMEOUT", 0)

        lines = [
            f"# Session Summary — {self._session_id}",
            "",
            f"**Experiment:** {self._experiment_name}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Duration:** {total_duration:.1f} seconds",
            f"**Total events logged:** {len(self._entries)}",
            "",
            "## Status Breakdown",
            "",
            f"| Status | Count |",
            f"|--------|-------|",
        ]
        for status, count in sorted(status_counts.items()):
            lines.append(f"| {status} | {count} |")

        lines.extend([
            "",
            "## Result",
            "",
        ])

        if skipped_count == 0 and ooo_count == 0 and timeout_count == 0:
            lines.append("✅ **All steps completed in correct order.** No deviations detected.")
        else:
            lines.append("⚠️ **Deviations detected:**")
            if skipped_count > 0:
                lines.append(f"  - {skipped_count} step(s) skipped")
            if ooo_count > 0:
                lines.append(f"  - {ooo_count} out-of-order event(s)")
            if timeout_count > 0:
                lines.append(f"  - {timeout_count} timeout(s)")

        lines.extend([
            "",
            "## Event Timeline",
            "",
            "| Time (s) | Step | Label | Status | Confidence |",
            "|----------|------|-------|--------|------------|",
        ])

        for entry in self._entries:
            lines.append(
                f"| {entry.elapsed_sec:.1f} | {entry.step_id} | "
                f"{entry.label} | {entry.status} | {entry.confidence:.2f} |"
            )

        lines.extend([
            "",
            "---",
            f"*Generated by SIH v0.1 HAR System*",
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def get_recent_entries(self, n: int = 50) -> List[Dict]:
        """Get the last N log entries as dicts (for the GUI log feed)."""
        entries = self._entries[-n:]
        return [
            {
                "timestamp": e.timestamp,
                "step_id": e.step_id,
                "label": e.label,
                "status": e.status,
                "confidence": e.confidence,
                "evidence": e.evidence,
                "elapsed_sec": e.elapsed_sec,
            }
            for e in reversed(entries)
        ]

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def session_id(self) -> str:
        return self._session_id
