# SIH v0.1 — AI Human Activity Recognition for On-board BAS Experiments

> **SIH Problem Statement 26174** | Theme: Space Technology  
> Webcam-based, fully offline prototype of an onboard AI HAR system for tracking astronaut experiment procedures.

## ✨ What It Does

1. **Watches** a person performing a multi-step experiment via webcam
2. **Detects** objects (red box, yellow box, large box) and hand/pose keypoints in real-time
3. **Tracks** progress against a predefined SOP (Standard Operating Procedure)
4. **Alerts** via voice when steps are skipped or done out of order
5. **Logs** every event to structured JSONL + human-readable summary
6. **Records** video locally and streams over HTTP to any viewer
7. **Displays** everything on a live Next.js dashboard

## 🏗️ Architecture

```
Input → Visual Perception → Spatial/Temporal → Experiment Validation → Voice Alerts → GUI
(Camera)  (HSV/YOLO+MediaPipe)  (Rule-based)     (Finite State Machine)  (pyttsx3/Kokoro)  (Next.js)
```

| Layer | MVP (v0.1) | Target (future) |
|-------|-----------|-----------------|
| Object Detection | HSV color+contour | YOLO11-S fine-tuned |
| Pose/Hands | MediaPipe Holistic | RTMPose-L via rtmlib |
| Spatial/Temporal | Rule-based proximity | ST-GCN++, Mamba-2 |
| Experiment Logic | Python FSM + JSON SOP | Symbolic graph + Qwen3-8B |
| TTS | pyttsx3 | Kokoro-82M |
| STT | off (optional) | faster-whisper large-v3-turbo |
| Assistant | Keyword search | Ollama + qwen2.5-7B |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Webcam

### 1. Clone & Install Backend
```bash
cd SIH
pip install -r requirements.txt
copy .env.example .env    # Windows
# cp .env.example .env    # Linux/Mac
```

### 2. Install Frontend
```bash
cd frontend
npm install
```

### 3. Run Backend
```bash
# From project root
python -m backend.main
```
Backend starts at `http://localhost:8000`.

### 4. Run Frontend
```bash
cd frontend
npm run dev
```
Dashboard at `http://localhost:3000`.

### 5. View MJPEG Stream (optional)
```bash
ffplay http://localhost:8554/video_feed
# Or open http://localhost:8554/ in a browser
```

## 🎬 Demo Script (Acceptance Test)

1. **Start system** — open http://localhost:3000, verify camera feed and PENDING status
2. **Click "Start Session"** — GUI shows Step 1 highlighted, voice speaks the first instruction
3. **Perform steps 1–2 correctly** — GUI updates to VALID, next-step guidance spoken
4. **Skip step 3, do step 4** — voice alert fires, GUI shows SKIPPED/OUT_OF_ORDER
5. **Correct course, finish** — all steps VALID, session ends
6. **Check logs** — open `logs/session_*.jsonl` and `*_summary.md`
7. **Check video** — open `data/videos/session_*.mp4`
8. **Check stream** — run `ffplay http://localhost:8554/video_feed`

### Demo Helper Endpoints
For reliable demo, use keyboard shortcuts or REST endpoints to manually advance steps:
```bash
# Complete current step
curl -X POST http://localhost:8000/api/session/step/complete

# Skip to step 4 (triggers SKIPPED alert for step 3)
curl -X POST "http://localhost:8000/api/session/step/skip?step_id=4"
```

## 📁 Project Structure

```
SIH/
├── backend/
│   ├── main.py                  # FastAPI app, pipeline orchestrator
│   ├── perception/
│   │   ├── detector.py          # Object detection (HSV / YOLO)
│   │   ├── pose.py              # Pose/hand keypoints (MediaPipe / rtmlib)
│   │   └── interaction.py       # Hand-object interaction tracker
│   ├── experiment/
│   │   ├── state_machine.py     # SOP-driven FSM
│   │   └── sop_loader.py        # JSON SOP parser
│   ├── assistant/
│   │   ├── tts.py               # Text-to-speech
│   │   ├── stt.py               # Speech-to-text (optional)
│   │   └── qa.py                # Q&A over SOP text
│   ├── streaming/
│   │   ├── video_writer.py      # Local .mp4 recording
│   │   └── network_sink.py      # MJPEG HTTP streaming
│   └── logging_/
│       └── session_logger.py    # JSONL + summary writer
├── frontend/                    # Next.js dashboard
├── config/
│   └── sop.json                 # Experiment procedure definition
├── data/
│   ├── raw/                     # Collected training frames
│   └── videos/                  # Session recordings
├── logs/                        # Session logs
├── scripts/
│   └── collect_data.py          # Data collection tool
├── requirements.txt
├── .env.example
└── README.md
```

## ⚙️ Configuration

All settings in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX_1` | `0` | Primary webcam index |
| `DETECTOR_MODE` | `hsv` | `hsv` or `yolo` |
| `POSE_MODE` | `mediapipe` | `mediapipe` or `rtmlib` |
| `TTS_ENGINE` | `pyttsx3` | `pyttsx3` or `kokoro` |
| `TARGET_FPS` | `4` | Processing framerate |
| `ENABLE_STREAM` | `true` | MJPEG network streaming |
| `STREAM_PORT` | `8554` | Stream server port |

## 📊 Data Collection

```bash
python scripts/collect_data.py --camera 0 --output data/raw
```
- Press **1–8** to select step, **SPACE** to pause/resume, **S** for snapshot, **Q** to quit
- Record ~15–30 repetitions per step for detector fine-tuning

## 🔧 Upgrade Path

The architecture is designed for seamless upgrades:
1. **Object Detection**: Set `DETECTOR_MODE=yolo` and provide weights at `YOLO_WEIGHTS_PATH`
2. **Pose**: Set `POSE_MODE=rtmlib` (install rtmlib first)
3. **TTS**: Set `TTS_ENGINE=kokoro` (download ONNX weights)
4. **STT**: Set `STT_ENGINE=faster_whisper` (install faster-whisper)
5. **Assistant**: Set `ASSISTANT_MODE=ollama` (install Ollama + model)

## 📝 License

MIT — Built for Smart India Hackathon 2026
