"""
Video Writer — local .mp4 session recording.

Always records the full session to data/videos/session_<timestamp>.mp4
regardless of whether network streaming is enabled.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoWriter:
    """
    Writes frames to a local .mp4 file using OpenCV's VideoWriter.
    Thread-safe for the write operation.
    """

    def __init__(self, output_dir: str = "data/videos",
                 fps: float = 15.0, resolution: tuple[int, int] = (640, 480)):
        self._output_dir = output_dir
        self._fps = fps
        self._resolution = resolution
        self._writer: Optional[cv2.VideoWriter] = None
        self._filepath: str = ""
        self._frame_count: int = 0
        self._is_recording: bool = False

        os.makedirs(output_dir, exist_ok=True)

    def start(self, session_id: str = "") -> str:
        """
        Start recording a new session.
        Returns the output file path.
        """
        if not session_id:
            session_id = time.strftime("%Y%m%d_%H%M%S")

        self._filepath = os.path.join(self._output_dir, f"session_{session_id}.mp4")

        # Use mp4v codec (widely supported)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(
            self._filepath, fourcc, self._fps,
            (self._resolution[0], self._resolution[1]),
        )

        if not self._writer.isOpened():
            logger.error("Failed to open VideoWriter at %s", self._filepath)
            self._writer = None
            return ""

        self._is_recording = True
        self._frame_count = 0
        logger.info("Recording started: %s", self._filepath)
        return self._filepath

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single frame. Resizes if needed."""
        if not self._is_recording or self._writer is None:
            return

        h, w = frame.shape[:2]
        if (w, h) != self._resolution:
            frame = cv2.resize(frame, self._resolution)

        self._writer.write(frame)
        self._frame_count += 1

    def stop(self) -> str:
        """Stop recording and release the writer. Returns the file path."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        self._is_recording = False
        logger.info("Recording stopped: %s (%d frames)",
                    self._filepath, self._frame_count)
        return self._filepath

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def frame_count(self) -> int:
        return self._frame_count
