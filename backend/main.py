"""
SIH v0.1 — Main Application

FastAPI server that orchestrates the full pipeline:
  Input → Visual Perception → Spatial/Temporal → Experiment Validation
       → Voice Alerts → Streaming/Storage → WebSocket to GUI

Endpoints:
  GET  /api/health          — system health check
  POST /api/session/start   — start experiment session
  POST /api/session/stop    — stop experiment session  
  GET  /api/session/state   — get current experiment state
  GET  /api/session/logs    — get recent log entries
  POST /api/session/step/complete — manually complete current step (demo helper)
  POST /api/session/step/skip    — skip to a specific step (demo helper)
  POST /api/assistant/ask   — ask the Q&A assistant a question
  WS   /ws                  — WebSocket for live state/frame push
  GET  /api/video/frame     — latest JPEG frame (for frontend fallback)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# ── Setup paths ─────────────────────────────────────────────────────────────
# Add project root to path so imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# ── Import pipeline modules ────────────────────────────────────────────────
from backend.perception.detector import create_detector, DetectionResult
from backend.perception.pose import create_pose_estimator, PoseResult
from backend.perception.interaction import InteractionTracker, InteractionResult
from backend.experiment.sop_loader import load_sop, SOP
from backend.experiment.state_machine import (
    ExperimentStateMachine, StepStatus, TransitionEvent, ExperimentState,
)
from backend.assistant.tts import create_tts, BaseTTS
from backend.assistant.qa import create_qa, BaseQA
from backend.streaming.video_writer import VideoWriter
from backend.streaming.network_sink import NetworkSink
from backend.logging_.session_logger import SessionLogger

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sih.main")


# ── Configuration ───────────────────────────────────────────────────────────

class Config:
    """Load configuration from environment variables with defaults."""

    CAMERA_INDEX_1: int = int(os.getenv("CAMERA_INDEX_1", "0"))
    CAMERA_INDEX_2: int = int(os.getenv("CAMERA_INDEX_2", "1"))
    ENABLE_CAMERA_2: bool = os.getenv("ENABLE_CAMERA_2", "false").lower() == "true"
    TARGET_FPS: int = int(os.getenv("TARGET_FPS", "4"))
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "640"))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "480"))

    DETECTOR_MODE: str = os.getenv("DETECTOR_MODE", "hsv")
    YOLO_WEIGHTS_PATH: str = os.getenv("YOLO_WEIGHTS_PATH", "models/yolo_finetuned.pt")
    YOLO_CONF_THRESHOLD: float = float(os.getenv("YOLO_CONF_THRESHOLD", "0.5"))

    POSE_MODE: str = os.getenv("POSE_MODE", "mediapipe")

    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "pyttsx3")
    STT_ENGINE: str = os.getenv("STT_ENGINE", "off")
    ASSISTANT_MODE: str = os.getenv("ASSISTANT_MODE", "keyword")

    ENABLE_STREAM: bool = os.getenv("ENABLE_STREAM", "true").lower() == "true"
    STREAM_HOST: str = os.getenv("STREAM_HOST", "0.0.0.0")
    STREAM_PORT: int = int(os.getenv("STREAM_PORT", "8554"))

    SAVE_VIDEO: bool = os.getenv("SAVE_VIDEO", "true").lower() == "true"
    VIDEO_OUTPUT_DIR: str = os.getenv("VIDEO_OUTPUT_DIR", str(PROJECT_ROOT / "data" / "videos"))
    LOG_OUTPUT_DIR: str = os.getenv("LOG_OUTPUT_DIR", str(PROJECT_ROOT / "logs"))

    BACKEND_HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    BACKEND_PORT: int = int(os.getenv("BACKEND_PORT", "8000"))

    SOP_PATH: str = os.getenv("SOP_PATH", str(PROJECT_ROOT / "config" / "sop.json"))


cfg = Config()


# ── Pipeline State (module-level singleton) ─────────────────────────────────

class PipelineState:
    """Holds all pipeline components and runtime state."""

    def __init__(self):
        # Components (initialized in startup)
        self.sop: Optional[SOP] = None
        self.detector = None
        self.pose_estimator = None
        self.interaction_tracker: Optional[InteractionTracker] = None
        self.state_machine: Optional[ExperimentStateMachine] = None
        self.tts: Optional[BaseTTS] = None
        self.qa: Optional[BaseQA] = None
        self.video_writer: Optional[VideoWriter] = None
        self.network_sink: Optional[NetworkSink] = None
        self.session_logger: Optional[SessionLogger] = None

        # Camera
        self.cap1: Optional[cv2.VideoCapture] = None
        self.cap2: Optional[cv2.VideoCapture] = None

        # Runtime
        self.is_session_active: bool = False
        self.processing_thread: Optional[threading.Thread] = None
        self.processing_running: bool = False
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_state: Optional[ExperimentState] = None
        self.latest_detections: Optional[DetectionResult] = None
        self.latest_pose: Optional[PoseResult] = None
        self.latest_interaction: Optional[InteractionResult] = None
        self.frame_lock = threading.Lock()

        # Async event loop reference (set during startup)
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

        # WebSocket clients
        self.ws_clients: Set[WebSocket] = set()

        # Latest state as JSON string (built in processing thread, sent by WS)
        self.latest_state_json: str = ""

        # System status
        self.system_status = {
            "camera1": False,
            "camera2": False,
            "detector": False,
            "pose": False,
            "tts": False,
            "logging": False,
            "streaming": False,
            "recording": False,
        }


pipeline = PipelineState()


# ── Lifecycle ───────────────────────────────────────────────────────────────

def initialize_pipeline():
    """Initialize all pipeline components."""
    logger.info("=" * 60)
    logger.info("SIH v0.1 — AI Human Activity Recognition System")
    logger.info("=" * 60)

    # 1. Load SOP
    try:
        pipeline.sop = load_sop(cfg.SOP_PATH)
        logger.info("✓ SOP loaded: %s (%d steps)",
                    pipeline.sop.experiment_name, pipeline.sop.total_steps)
    except Exception as e:
        logger.error("✗ Failed to load SOP: %s", e)
        raise

    # 2. Initialize detector
    try:
        pipeline.detector = create_detector(
            mode=cfg.DETECTOR_MODE,
            weights_path=cfg.YOLO_WEIGHTS_PATH,
            conf_threshold=cfg.YOLO_CONF_THRESHOLD,
        )
        pipeline.system_status["detector"] = True
        logger.info("✓ Detector initialized: %s", cfg.DETECTOR_MODE)
    except Exception as e:
        logger.error("✗ Detector init failed: %s", e)

    # 3. Initialize pose estimator
    try:
        pipeline.pose_estimator = create_pose_estimator(mode=cfg.POSE_MODE)
        pipeline.system_status["pose"] = True
        logger.info("✓ Pose estimator initialized: %s", cfg.POSE_MODE)
    except Exception as e:
        logger.error("✗ Pose estimator init failed: %s", e)

    # 4. Initialize interaction tracker
    pipeline.interaction_tracker = InteractionTracker(window_size=15)
    logger.info("✓ Interaction tracker initialized")

    # 5. Initialize state machine
    pipeline.state_machine = ExperimentStateMachine(pipeline.sop)
    pipeline.state_machine.on_transition(on_step_transition)
    logger.info("✓ State machine initialized")

    # 6. Initialize TTS
    try:
        pipeline.tts = create_tts(engine=cfg.TTS_ENGINE)
        pipeline.system_status["tts"] = True
        logger.info("✓ TTS initialized: %s", cfg.TTS_ENGINE)
    except Exception as e:
        logger.warning("✗ TTS init failed: %s", e)

    # 7. Initialize Q&A assistant
    step_dicts = [
        {
            "step_id": s.step_id,
            "label": s.label,
            "description": s.description,
            "required_objects": s.required_objects,
            "expected_action": s.expected_action,
        }
        for s in pipeline.sop.steps
    ]
    pipeline.qa = create_qa(
        mode=cfg.ASSISTANT_MODE,
        sop_text=pipeline.sop.get_full_text(),
        step_descriptions=step_dicts,
    )
    logger.info("✓ Q&A assistant initialized: %s", cfg.ASSISTANT_MODE)

    # 8. Initialize video writer
    pipeline.video_writer = VideoWriter(
        output_dir=cfg.VIDEO_OUTPUT_DIR,
        fps=cfg.TARGET_FPS * 2,  # record at higher fps than processing
        resolution=(cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT),
    )
    logger.info("✓ Video writer initialized")

    # 9. Initialize network sink
    if cfg.ENABLE_STREAM:
        pipeline.network_sink = NetworkSink(
            host=cfg.STREAM_HOST, port=cfg.STREAM_PORT,
        )
        pipeline.network_sink.start()
        pipeline.system_status["streaming"] = True
        logger.info("✓ Network stream: http://%s:%d/video_feed",
                    cfg.STREAM_HOST, cfg.STREAM_PORT)

    # 10. Initialize session logger
    pipeline.session_logger = SessionLogger(output_dir=cfg.LOG_OUTPUT_DIR)
    logger.info("✓ Session logger initialized")

    # 11. Open camera(s)
    _open_cameras()

    logger.info("=" * 60)
    logger.info("Pipeline ready. Start a session via POST /api/session/start")
    logger.info("=" * 60)


def _open_cameras():
    """Open webcam(s)."""
    pipeline.cap1 = cv2.VideoCapture(cfg.CAMERA_INDEX_1)
    if pipeline.cap1.isOpened():
        pipeline.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
        pipeline.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
        pipeline.system_status["camera1"] = True
        logger.info("✓ Camera 1 opened (index %d)", cfg.CAMERA_INDEX_1)
    else:
        logger.error("✗ Camera 1 failed to open (index %d)", cfg.CAMERA_INDEX_1)

    if cfg.ENABLE_CAMERA_2:
        pipeline.cap2 = cv2.VideoCapture(cfg.CAMERA_INDEX_2)
        if pipeline.cap2.isOpened():
            pipeline.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
            pipeline.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
            pipeline.system_status["camera2"] = True
            logger.info("✓ Camera 2 opened (index %d)", cfg.CAMERA_INDEX_2)
        else:
            logger.warning("Camera 2 not available — continuing with single camera")


def shutdown_pipeline():
    """Clean up all resources."""
    logger.info("Shutting down pipeline...")

    pipeline.processing_running = False
    if pipeline.processing_thread:
        pipeline.processing_thread.join(timeout=5)

    if pipeline.tts:
        pipeline.tts.shutdown()
    if pipeline.network_sink:
        pipeline.network_sink.stop()
    if pipeline.video_writer and pipeline.video_writer.is_recording:
        pipeline.video_writer.stop()
    if pipeline.cap1:
        pipeline.cap1.release()
    if pipeline.cap2:
        pipeline.cap2.release()

    logger.info("Pipeline shutdown complete")


# ── Transition callback ────────────────────────────────────────────────────

def on_step_transition(event: TransitionEvent):
    """Called by the state machine when a step's status changes."""
    # Log the event
    if pipeline.session_logger and pipeline.session_logger.is_active:
        pipeline.session_logger.log_event(
            step_id=event.step_id,
            label=event.label,
            status=event.new_status.value,
            confidence=event.confidence,
            evidence=event.evidence,
            description=event.description,
        )

    # Voice alert on errors
    if event.requires_alert and pipeline.tts:
        pipeline.tts.speak(event.alert_message)
        logger.warning("🔊 ALERT: %s", event.alert_message)

    # Voice guidance on step completion
    if event.new_status == StepStatus.VALID and pipeline.tts:
        next_step = pipeline.sop.get_next_step(event.step_id)
        if next_step:
            pipeline.tts.speak(f"Step {event.step_id} complete. Next: {next_step.description}")
        else:
            pipeline.tts.speak("Experiment complete. All steps validated.")

    # Voice prompt when a new step becomes in-progress
    if event.new_status == StepStatus.IN_PROGRESS and pipeline.tts:
        step = pipeline.sop.get_step(event.step_id)
        if step and step.voice_prompt:
            pipeline.tts.speak(step.voice_prompt)


# ── Processing Loop ────────────────────────────────────────────────────────

def processing_loop():
    """
    Main processing loop — runs in a background thread.
    Captures frames, runs perception, updates state machine.
    """
    frame_interval = 1.0 / cfg.TARGET_FPS
    logger.info("Processing loop started (target %d FPS)", cfg.TARGET_FPS)

    while pipeline.processing_running:
        loop_start = time.time()

        # 1. Capture frame
        if not pipeline.cap1 or not pipeline.cap1.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = pipeline.cap1.read()
        if not ret:
            time.sleep(0.01)
            continue

        with pipeline.frame_lock:
            pipeline.latest_frame = frame.copy()

        # 2. Run object detection
        det_result = DetectionResult()
        if pipeline.detector:
            try:
                det_result = pipeline.detector.detect(frame)
                pipeline.latest_detections = det_result
            except Exception as e:
                logger.error("Detection error: %s", e)

        # 3. Run pose estimation
        pose_result = PoseResult()
        if pipeline.pose_estimator:
            try:
                pose_result = pipeline.pose_estimator.estimate(frame)
                pipeline.latest_pose = pose_result
            except Exception as e:
                logger.error("Pose error: %s", e)

        # 4. Run interaction tracking
        interaction_result = InteractionResult()
        if pipeline.interaction_tracker:
            try:
                interaction_result = pipeline.interaction_tracker.update(
                    det_result, pose_result
                )
                pipeline.latest_interaction = interaction_result
            except Exception as e:
                logger.error("Interaction error: %s", e)

        # 5. Update state machine (only during active session)
        if pipeline.is_session_active and pipeline.state_machine:
            try:
                pipeline.state_machine.update(interaction_result)
                pipeline.latest_state = pipeline.state_machine.get_state()
                # Update person detection info
                if pipeline.latest_state:
                    pipeline.latest_state.person_detected = pose_result.person_detected
                    pipeline.latest_state.person_confidence = pose_result.person_confidence
            except Exception as e:
                logger.error("State machine error: %s", e)
        elif pipeline.state_machine:
            # Even when no active session, keep state snapshot updated for GUI
            try:
                state = pipeline.state_machine.get_state()
                state.person_detected = pose_result.person_detected
                state.person_confidence = pose_result.person_confidence
                state.detected_objects = interaction_result.objects_in_scene
                state.object_states = interaction_result.object_states
                state.latest_action = interaction_result.current_action.action
                state.latest_action_target = interaction_result.current_action.target_object
                state.latest_action_confidence = interaction_result.current_action.confidence
                pipeline.latest_state = state
            except Exception:
                pass

        # 6. Draw annotations on frame
        annotated = frame.copy()
        if pipeline.detector:
            annotated = pipeline.detector.draw_detections(annotated, det_result)
        if pipeline.pose_estimator:
            annotated = pipeline.pose_estimator.draw_pose(annotated, pose_result)

        # Draw step info overlay
        if pipeline.latest_state:
            _draw_status_overlay(annotated, pipeline.latest_state)

        with pipeline.frame_lock:
            pipeline.latest_annotated_frame = annotated.copy()

        # 7. Write to video
        if pipeline.video_writer and pipeline.video_writer.is_recording:
            pipeline.video_writer.write_frame(annotated)

        # 8. Update network stream
        if pipeline.network_sink and pipeline.network_sink.is_running:
            pipeline.network_sink.update_frame(annotated)

        # 9. Build state JSON for WebSocket broadcast (done in this thread to avoid async issues)
        _build_state_json()

        # Rate limit
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    logger.info("Processing loop stopped")


def _draw_status_overlay(frame: np.ndarray, state: ExperimentState):
    """Draw experiment status info on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent background bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Status text
    status_colors = {
        "PENDING": (200, 200, 200),
        "IN_PROGRESS": (255, 200, 0),
        "VALID": (0, 255, 0),
        "SKIPPED": (0, 0, 255),
        "OUT_OF_ORDER": (0, 100, 255),
        "UNCERTAIN": (200, 200, 0),
        "TIMEOUT": (0, 0, 255),
    }
    color = status_colors.get(state.current_step_status, (255, 255, 255))

    step_text = f"Step {state.current_step_id}/{state.total_steps}: {state.current_step_description}"
    cv2.putText(frame, step_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    status_text = f"[{state.current_step_status}] | Action: {state.latest_action} | Elapsed: {state.session_elapsed_sec:.0f}s"
    cv2.putText(frame, status_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _build_state_json():
    """Build the state JSON string in the processing thread (thread-safe)."""
    if not pipeline.latest_state:
        return

    state_dict = {
        "type": "state_update",
        "data": {
            "experiment_id": pipeline.latest_state.experiment_id,
            "experiment_name": pipeline.latest_state.experiment_name,
            "current_step_id": pipeline.latest_state.current_step_id,
            "current_step_label": pipeline.latest_state.current_step_label,
            "current_step_description": pipeline.latest_state.current_step_description,
            "current_step_status": pipeline.latest_state.current_step_status,
            "next_step_description": pipeline.latest_state.next_step_description,
            "total_steps": pipeline.latest_state.total_steps,
            "completed_steps": pipeline.latest_state.completed_steps,
            "step_states": pipeline.latest_state.step_states,
            "session_elapsed_sec": pipeline.latest_state.session_elapsed_sec,
            "is_complete": pipeline.latest_state.is_complete,
            "latest_action": pipeline.latest_state.latest_action,
            "latest_action_target": pipeline.latest_state.latest_action_target,
            "latest_action_confidence": pipeline.latest_state.latest_action_confidence,
            "detected_objects": pipeline.latest_state.detected_objects,
            "object_states": pipeline.latest_state.object_states,
            "person_detected": pipeline.latest_state.person_detected,
            "person_confidence": pipeline.latest_state.person_confidence,
        },
        "session_active": pipeline.is_session_active,
        "system_status": pipeline.system_status,
    }

    # Include frame as base64 JPEG
    with pipeline.frame_lock:
        if pipeline.latest_annotated_frame is not None:
            _, jpeg = cv2.imencode('.jpg', pipeline.latest_annotated_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 60])
            state_dict["frame"] = base64.b64encode(jpeg.tobytes()).decode('ascii')

    pipeline.latest_state_json = json.dumps(state_dict)


# ── FastAPI App ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    initialize_pipeline()

    # Start processing loop in background thread
    pipeline.processing_running = True
    pipeline.processing_thread = threading.Thread(target=processing_loop, daemon=True)
    pipeline.processing_thread.start()

    yield

    shutdown_pipeline()


app = FastAPI(
    title="SIH v0.1 — AI Human Activity Recognition",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "system_status": pipeline.system_status,
        "session_active": pipeline.is_session_active,
    }


@app.post("/api/session/start")
async def start_session():
    """Start a new experiment session (or restart after completion)."""
    # Allow restart: if session was completed or stopped, reset it
    if pipeline.is_session_active:
        # Stop the old session first
        pipeline.is_session_active = False
        pipeline.state_machine.stop_session()
        if pipeline.session_logger and pipeline.session_logger.is_active:
            pipeline.session_logger.end_session()
        if pipeline.video_writer and pipeline.video_writer.is_recording:
            pipeline.video_writer.stop()

    # Recreate state machine for a clean start
    pipeline.state_machine = ExperimentStateMachine(pipeline.sop)
    pipeline.state_machine.on_transition(on_step_transition)

    # Start state machine
    pipeline.state_machine.start_session()
    pipeline.is_session_active = True

    # Start logging
    session_id = pipeline.session_logger.start_session(pipeline.sop.experiment_name)
    pipeline.system_status["logging"] = True

    # Start video recording
    if cfg.SAVE_VIDEO:
        video_path = pipeline.video_writer.start(session_id)
        pipeline.system_status["recording"] = True

    # Reset interaction tracker
    pipeline.interaction_tracker.reset()

    # Announce start
    if pipeline.tts:
        pipeline.tts.speak(f"Starting experiment: {pipeline.sop.experiment_name}. "
                          f"{pipeline.sop.steps[0].voice_prompt}")

    logger.info("Session started: %s", session_id)
    return {
        "status": "started",
        "session_id": session_id,
        "experiment": pipeline.sop.experiment_name,
        "total_steps": pipeline.sop.total_steps,
    }


@app.post("/api/session/stop")
async def stop_session():
    """Stop the current session."""
    if not pipeline.is_session_active:
        return JSONResponse({"error": "No active session"}, status_code=400)

    pipeline.is_session_active = False
    pipeline.state_machine.stop_session()

    # Stop logging
    jsonl_path, summary_path = pipeline.session_logger.end_session()
    pipeline.system_status["logging"] = False

    # Stop video recording
    video_path = ""
    if pipeline.video_writer and pipeline.video_writer.is_recording:
        video_path = pipeline.video_writer.stop()
        pipeline.system_status["recording"] = False

    if pipeline.tts:
        pipeline.tts.speak("Experiment session ended.")

    logger.info("Session stopped")
    return {
        "status": "stopped",
        "log_file": jsonl_path,
        "summary_file": summary_path,
        "video_file": video_path,
    }


@app.get("/api/session/state")
async def get_session_state():
    """Get the current experiment state."""
    if pipeline.latest_state:
        state = pipeline.latest_state
        return {
            "experiment_id": state.experiment_id,
            "experiment_name": state.experiment_name,
            "current_step_id": state.current_step_id,
            "current_step_label": state.current_step_label,
            "current_step_description": state.current_step_description,
            "current_step_status": state.current_step_status,
            "next_step_description": state.next_step_description,
            "total_steps": state.total_steps,
            "completed_steps": state.completed_steps,
            "step_states": state.step_states,
            "session_elapsed_sec": state.session_elapsed_sec,
            "is_complete": state.is_complete,
            "latest_action": state.latest_action,
            "latest_action_target": state.latest_action_target,
            "latest_action_confidence": state.latest_action_confidence,
            "detected_objects": state.detected_objects,
            "object_states": state.object_states,
            "person_detected": state.person_detected,
            "person_confidence": state.person_confidence,
            "system_status": pipeline.system_status,
        }
    return {"status": "no_active_session", "system_status": pipeline.system_status}


@app.get("/api/session/logs")
async def get_session_logs(n: int = 50):
    """Get recent log entries."""
    if pipeline.session_logger:
        return {"entries": pipeline.session_logger.get_recent_entries(n)}
    return {"entries": []}


@app.post("/api/session/step/complete")
async def complete_step():
    """Manually complete the current step (demo helper)."""
    if not pipeline.is_session_active:
        return JSONResponse({"error": "No active session"}, status_code=400)

    state = pipeline.state_machine.get_state()
    event = pipeline.state_machine.force_complete_step(state.current_step_id)
    if event:
        return {"status": "completed", "step_id": event.step_id, "label": event.label}
    return {"status": "no_change"}


@app.post("/api/session/step/skip")
async def skip_to_step(step_id: int = 0):
    """Skip to a specific step (demo helper for testing deviation alerts)."""
    if not pipeline.is_session_active:
        return JSONResponse({"error": "No active session"}, status_code=400)

    if step_id <= 0:
        return JSONResponse({"error": "Provide step_id query param"}, status_code=400)

    event = pipeline.state_machine.force_skip_to_step(step_id)
    if event:
        return {"status": "skipped", "step_id": event.step_id}
    return {"status": "no_change"}


@app.post("/api/assistant/ask")
async def ask_assistant(question: str = "", context: str = ""):
    """Ask the Q&A assistant a question."""
    if not question:
        return JSONResponse({"error": "Provide question"}, status_code=400)

    # Auto-inject current step context
    if not context and pipeline.latest_state:
        context = f"current_step_id: {pipeline.latest_state.current_step_id}"

    answer = pipeline.qa.answer(question, context)

    # Speak the answer
    if pipeline.tts:
        pipeline.tts.speak(answer)

    return {"question": question, "answer": answer}


@app.get("/api/video/frame")
async def get_frame():
    """Get the latest annotated frame as JPEG."""
    with pipeline.frame_lock:
        frame = pipeline.latest_annotated_frame
        if frame is None:
            frame = pipeline.latest_frame

    if frame is None:
        return Response(content=b"", media_type="image/jpeg", status_code=204)

    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


@app.get("/api/sop")
async def get_sop():
    """Get the loaded SOP data."""
    if pipeline.sop:
        return {
            "experiment_id": pipeline.sop.experiment_id,
            "experiment_name": pipeline.sop.experiment_name,
            "description": pipeline.sop.description,
            "total_steps": pipeline.sop.total_steps,
            "steps": [
                {
                    "step_id": s.step_id,
                    "label": s.label,
                    "description": s.description,
                    "required_objects": s.required_objects,
                    "expected_action": s.expected_action,
                    "voice_prompt": s.voice_prompt,
                }
                for s in pipeline.sop.steps
            ],
        }
    return {"error": "SOP not loaded"}


# ── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for live state + frame streaming to frontend."""
    await ws.accept()
    pipeline.ws_clients.add(ws)
    logger.info("WebSocket client connected (%d total)", len(pipeline.ws_clients))

    try:
        # Run two tasks concurrently:
        # 1. Push state updates to client on a timer
        # 2. Receive messages from client
        send_task = asyncio.create_task(_ws_sender(ws))
        recv_task = asyncio.create_task(_ws_receiver(ws))
        
        # Wait for either to finish (disconnection)
        done, pending = await asyncio.wait(
            {send_task, recv_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket error: %s", e)
    finally:
        pipeline.ws_clients.discard(ws)
        logger.info("WebSocket client disconnected (%d remaining)",
                    len(pipeline.ws_clients))


async def _ws_sender(ws: WebSocket):
    """Push state updates to a single WebSocket client at ~4 FPS."""
    try:
        while True:
            if pipeline.latest_state_json:
                await ws.send_text(pipeline.latest_state_json)
            await asyncio.sleep(0.25)  # 4 updates/sec
    except Exception:
        pass


async def _ws_receiver(ws: WebSocket):
    """Receive messages from a WebSocket client."""
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ask":
                    answer = pipeline.qa.answer(
                        msg.get("question", ""),
                        msg.get("context", ""),
                    )
                    await ws.send_text(json.dumps({
                        "type": "qa_response",
                        "answer": answer,
                    }))
                    if pipeline.tts:
                        pipeline.tts.speak(answer)
            except json.JSONDecodeError:
                pass
    except Exception:
        pass


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=cfg.BACKEND_HOST,
        port=cfg.BACKEND_PORT,
        reload=False,
        log_level="info",
    )
