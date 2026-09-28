"""
Speech-to-Text — Stage 5 (Local Assistant), optional

Provides voice input for the Q&A assistant ("what's next?").
Uses faster-whisper with CTranslate2 backend for CPU-friendly inference.
Falls back gracefully to disabled state if not installed.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class BaseSTT(ABC):
    @abstractmethod
    def start_listening(self, on_result: Callable[[str], None]) -> None:
        """Start listening for speech, call on_result with transcribed text."""
        ...

    @abstractmethod
    def stop_listening(self) -> None:
        """Stop listening."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this STT engine is actually functional."""
        ...


class DisabledSTT(BaseSTT):
    """Placeholder when STT is disabled or unavailable."""

    def start_listening(self, on_result):
        logger.info("STT is disabled")

    def stop_listening(self):
        pass

    def is_available(self) -> bool:
        return False


class FasterWhisperSTT(BaseSTT):
    """
    Uses faster-whisper for speech-to-text.
    Falls back to DisabledSTT if faster-whisper or audio input isn't available.
    """

    def __init__(self, model_size: str = "small"):
        self._model = None
        self._listening = False
        self._thread: Optional[threading.Thread] = None

        try:
            from faster_whisper import WhisperModel
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(model_size, device=device,
                                        compute_type=compute_type)
            logger.info("Faster-whisper loaded (model=%s, device=%s)",
                        model_size, device)
        except ImportError:
            logger.warning("faster-whisper not installed — STT disabled")
        except Exception as e:
            logger.warning("STT init failed: %s", e)

    def start_listening(self, on_result: Callable[[str], None]) -> None:
        if not self._model:
            return

        self._listening = True
        self._thread = threading.Thread(
            target=self._listen_loop, args=(on_result,), daemon=True
        )
        self._thread.start()

    def _listen_loop(self, on_result: Callable[[str], None]):
        """Record audio chunks and transcribe them."""
        try:
            import sounddevice as sd
            import numpy as np
            import tempfile
            import wave
            import os

            sample_rate = 16000
            chunk_duration = 5  # seconds

            while self._listening:
                # Record audio chunk
                audio = sd.rec(int(chunk_duration * sample_rate),
                              samplerate=sample_rate, channels=1, dtype='int16')
                sd.wait()

                if not self._listening:
                    break

                # Save to temp file for faster-whisper
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                    with wave.open(tmp_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio.tobytes())

                # Transcribe
                segments, _ = self._model.transcribe(tmp_path, language="en")
                text = " ".join(seg.text for seg in segments).strip()

                os.unlink(tmp_path)

                if text and len(text) > 2:
                    logger.info("STT result: %s", text)
                    on_result(text)

        except Exception as e:
            logger.error("STT listen error: %s", e)

    def stop_listening(self) -> None:
        self._listening = False
        if self._thread:
            self._thread.join(timeout=3)

    def is_available(self) -> bool:
        return self._model is not None


def create_stt(engine: str = "off", **kwargs) -> BaseSTT:
    """Factory function for STT engine."""
    if engine == "faster_whisper":
        stt = FasterWhisperSTT(**kwargs)
        if stt.is_available():
            return stt
    return DisabledSTT()
