"""
Text-to-Speech — Stage 5 (Local Assistant)

Provides voice output for:
  - Step completion guidance ("Step 2 complete. Next: Remove the yellow box.")
  - Alerts on SKIPPED / OUT_OF_ORDER / TIMEOUT

Two backends:
  1. pyttsx3   — zero setup, uses system speech engine (SAPI5 on Windows)
  2. KokoroTTS — higher quality, needs pre-downloaded ONNX weights

Runs TTS in a background thread to avoid blocking the pipeline.
"""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    """Interface for TTS engines."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Queue text for speech (non-blocking)."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop current speech and clear queue."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""
        ...


class Pyttsx3TTS(BaseTTS):
    """
    pyttsx3-based TTS. Uses the system speech engine (SAPI5 on Windows,
    espeak on Linux, nsss on macOS).  Zero downloads, works everywhere.
    """

    def __init__(self, rate: int = 170, volume: float = 0.9):
        self._queue: queue.Queue = queue.Queue()
        self._running = True
        self._rate = rate
        self._volume = volume
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("pyttsx3 TTS engine initialized (rate=%d)", rate)

    def _worker(self):
        """Background thread that processes speech queue."""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', self._rate)
        engine.setProperty('volume', self._volume)

        # Try to set a more natural voice
        voices = engine.getProperty('voices')
        if voices:
            # Prefer a female voice if available (typically index 1)
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            else:
                engine.setProperty('voice', voices[0].id)

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if text is None:
                    break
                logger.debug("TTS speaking: %s", text[:80])
                engine.say(text)
                engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("TTS error: %s", e)

        try:
            engine.stop()
        except Exception:
            pass

    def speak(self, text: str) -> None:
        self._queue.put(text)

    def stop(self) -> None:
        # Clear the queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        self._running = False
        self._queue.put(None)
        self._thread.join(timeout=3)


class KokoroTTS(BaseTTS):
    """
    Kokoro-82M ONNX TTS — higher quality neural voice.
    Falls back to pyttsx3 if kokoro-onnx isn't installed or weights missing.
    """

    def __init__(self, model_path: str = "models/kokoro-v0_19.onnx",
                 voices_path: str = "models/voices.bin"):
        self._fallback: Pyttsx3TTS | None = None
        self._kokoro = None
        self._queue: queue.Queue = queue.Queue()
        self._running = True

        try:
            from kokoro_onnx import Kokoro
            import os
            if os.path.exists(model_path) and os.path.exists(voices_path):
                self._kokoro = Kokoro(model_path, voices_path)
                self._thread = threading.Thread(target=self._worker_kokoro, daemon=True)
                self._thread.start()
                logger.info("Kokoro TTS engine initialized")
            else:
                raise FileNotFoundError("Kokoro weights not found")
        except Exception as e:
            logger.warning("Kokoro TTS not available (%s) — falling back to pyttsx3", e)
            self._fallback = Pyttsx3TTS()

    def _worker_kokoro(self):
        import sounddevice as sd
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                if text is None:
                    break
                samples, sample_rate = self._kokoro.create(text, voice="af_bella", speed=1.0)
                sd.play(samples, sample_rate)
                sd.wait()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Kokoro TTS error: %s", e)

    def speak(self, text: str) -> None:
        if self._fallback:
            self._fallback.speak(text)
        else:
            self._queue.put(text)

    def stop(self) -> None:
        if self._fallback:
            self._fallback.stop()
        else:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

    def shutdown(self) -> None:
        self._running = False
        if self._fallback:
            self._fallback.shutdown()
        else:
            self._queue.put(None)
            self._thread.join(timeout=3)


def create_tts(engine: str = "pyttsx3", **kwargs) -> BaseTTS:
    """Factory function for TTS engine."""
    if engine == "kokoro":
        return KokoroTTS(**kwargs)
    else:
        return Pyttsx3TTS(**kwargs)
