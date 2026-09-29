"use client";

import { useState, useEffect, useRef, useCallback } from "react";

// ── Types ──────────────────────────────────────────────────────────────────

interface StepState {
  step_id: number;
  label: string;
  description: string;
  status: string;
  confidence: number;
  evidence: string[];
  voice_prompt: string;
}

interface SystemStatus {
  camera1: boolean;
  camera2: boolean;
  detector: boolean;
  pose: boolean;
  tts: boolean;
  logging: boolean;
  streaming: boolean;
  recording: boolean;
}

interface ExperimentState {
  experiment_id: string;
  experiment_name: string;
  current_step_id: number;
  current_step_label: string;
  current_step_description: string;
  current_step_status: string;
  next_step_description: string;
  total_steps: number;
  completed_steps: number;
  step_states: StepState[];
  session_elapsed_sec: number;
  is_complete: boolean;
  latest_action: string;
  latest_action_target: string;
  latest_action_confidence: number;
  detected_objects: string[];
  object_states: Record<string, string>;
  person_detected: boolean;
  person_confidence: number;
}

interface LogEntry {
  timestamp: string;
  step_id: number;
  label: string;
  status: string;
  confidence: number;
  evidence: string[];
  elapsed_sec: number;
}

interface WSMessage {
  type: string;
  data?: ExperimentState;
  frame?: string;
  session_active?: boolean;
  system_status?: SystemStatus;
  answer?: string;
}

// ── Constants ──────────────────────────────────────────────────────────────

const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";

// ── Main Page ──────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [state, setState] = useState<ExperimentState | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    camera1: false,
    camera2: false,
    detector: false,
    pose: false,
    tts: false,
    logging: false,
    streaming: false,
    recording: false,
  });
  const [frameData, setFrameData] = useState<string>("");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [sessionActive, setSessionActive] = useState(false);
  const [connected, setConnected] = useState(false);
  const [qaQuestion, setQaQuestion] = useState("");
  const [qaAnswer, setQaAnswer] = useState("");
  const [qaLoading, setQaLoading] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── WebSocket connection ─────────────────────────────────────────────

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setConnected(true);
      console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);

        if (msg.type === "state_update" && msg.data) {
          setState(msg.data);
          if (msg.system_status) setSystemStatus(msg.system_status);
          if (typeof msg.session_active === "boolean") {
            setSessionActive(msg.session_active);
          }
        }

        if (msg.frame) {
          setFrameData(`data:image/jpeg;base64,${msg.frame}`);
        }

        if (msg.type === "qa_response" && msg.answer) {
          setQaAnswer(msg.answer);
          setQaLoading(false);
        }
      } catch (e) {
        console.error("WS parse error:", e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect
      reconnectTimer.current = setTimeout(connectWS, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connectWS();
    // Fetch logs periodically
    const logInterval = setInterval(fetchLogs, 3000);
    // Initial health check
    checkHealth();

    return () => {
      clearInterval(logInterval);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connectWS]);

  // ── API calls ────────────────────────────────────────────────────────

  async function checkHealth() {
    try {
      const res = await fetch(`${BACKEND_URL}/api/health`);
      const data = await res.json();
      setSystemStatus(data.system_status);
      setSessionActive(data.session_active);
    } catch {
      console.log("Backend not reachable");
    }
  }

  async function startSession() {
    try {
      const res = await fetch(`${BACKEND_URL}/api/session/start`, {
        method: "POST",
      });
      const data = await res.json();
      if (data.status === "started") setSessionActive(true);
    } catch (e) {
      console.error("Start session error:", e);
    }
  }

  async function stopSession() {
    try {
      const res = await fetch(`${BACKEND_URL}/api/session/stop`, {
        method: "POST",
      });
      await res.json();
      setSessionActive(false);
    } catch (e) {
      console.error("Stop session error:", e);
    }
  }

  async function completeStep() {
    try {
      await fetch(`${BACKEND_URL}/api/session/step/complete`, {
        method: "POST",
      });
    } catch (e) {
      console.error("Complete step error:", e);
    }
  }

  async function skipToStep(stepId: number) {
    try {
      await fetch(
        `${BACKEND_URL}/api/session/step/skip?step_id=${stepId}`,
        { method: "POST" }
      );
    } catch (e) {
      console.error("Skip step error:", e);
    }
  }

  async function fetchLogs() {
    try {
      const res = await fetch(`${BACKEND_URL}/api/session/logs?n=30`);
      const data = await res.json();
      if (data.entries) setLogs(data.entries);
    } catch {
      // Backend not available
    }
  }

  async function askQuestion() {
    if (!qaQuestion.trim()) return;
    setQaLoading(true);
    setQaAnswer("");

    // Try WebSocket first
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "ask",
          question: qaQuestion,
          context: state
            ? `current_step_id: ${state.current_step_id}`
            : "",
        })
      );
    } else {
      // Fallback to REST
      try {
        const res = await fetch(
          `${BACKEND_URL}/api/assistant/ask?question=${encodeURIComponent(
            qaQuestion
          )}&context=current_step_id:${state?.current_step_id || 1}`,
          { method: "POST" }
        );
        const data = await res.json();
        setQaAnswer(data.answer);
      } catch {
        setQaAnswer("Unable to reach assistant.");
      }
      setQaLoading(false);
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────

  function formatElapsed(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  }

  const statusIcon: Record<string, string> = {
    PENDING: "⏳",
    IN_PROGRESS: "🔄",
    VALID: "✅",
    SKIPPED: "⏭️",
    OUT_OF_ORDER: "⚠️",
    UNCERTAIN: "❓",
    TIMEOUT: "⏰",
  };

  // ── Render ───────────────────────────────────────────────────────────

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(180deg, #0a0e1a 0%, #0f172a 50%, #0a0e1a 100%)",
        padding: "16px",
      }}
    >
      {/* Header */}
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "16px",
          padding: "12px 20px",
          borderRadius: "12px",
          background: "rgba(17, 24, 39, 0.8)",
          backdropFilter: "blur(12px)",
          border: "1px solid rgba(148, 163, 184, 0.1)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "8px",
              background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "18px",
            }}
          >
            🛰️
          </div>
          <div>
            <h1
              style={{
                fontSize: "1.2rem",
                fontWeight: 700,
                background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}
            >
              SIH — Human Activity Recognition
            </h1>
            <p
              style={{
                fontSize: "0.75rem",
                color: "#64748b",
              }}
            >
              {state?.experiment_name || "AI HAR for On-board BAS Experiments"}
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {/* Connection indicator */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              fontSize: "0.75rem",
              color: connected ? "#10b981" : "#ef4444",
            }}
          >
            <div
              className={connected ? "indicator-dot indicator-green" : "indicator-dot indicator-red"}
            />
            {connected ? "Connected" : "Disconnected"}
          </div>

          {/* Session controls */}
          {!sessionActive ? (
            <button className="btn-primary" onClick={startSession}>
              ▶ Start Session
            </button>
          ) : (
            <div style={{ display: "flex", gap: "8px" }}>
              <button className="btn-outline" onClick={completeStep}>
                ✓ Complete Step
              </button>
              <button className="btn-danger" onClick={stopSession}>
                ■ Stop Session
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 360px",
          gridTemplateRows: "auto auto",
          gap: "16px",
          maxHeight: "calc(100vh - 100px)",
        }}
      >
        {/* Camera Panel */}
        <div className="glass-card" style={{ padding: "0", overflow: "hidden" }}>
          <div
            style={{
              padding: "12px 16px",
              borderBottom: "1px solid rgba(148, 163, 184, 0.1)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>
              📷 Camera 1 — Live Feed
            </span>
            {sessionActive && state && (
              <span style={{ fontSize: "0.8rem", color: "#94a3b8" }}>
                {formatElapsed(state.session_elapsed_sec)}
              </span>
            )}
          </div>
          <div
            style={{
              aspectRatio: "4/3",
              background: "#000",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              position: "relative",
            }}
          >
            {frameData ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={frameData}
                alt="Live camera feed"
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "contain",
                }}
              />
            ) : (
              <div
                style={{
                  color: "#64748b",
                  textAlign: "center",
                  fontSize: "0.9rem",
                }}
              >
                <div style={{ fontSize: "2rem", marginBottom: "8px" }}>📷</div>
                <div>Waiting for camera feed...</div>
                <div style={{ fontSize: "0.75rem", marginTop: "4px" }}>
                  Ensure backend is running at {BACKEND_URL}
                </div>
              </div>
            )}

            {/* Recording indicator */}
            {systemStatus.recording && (
              <div
                style={{
                  position: "absolute",
                  top: "12px",
                  right: "12px",
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  background: "rgba(239, 68, 68, 0.8)",
                  padding: "4px 10px",
                  borderRadius: "6px",
                  fontSize: "0.7rem",
                  fontWeight: 600,
                }}
              >
                <div
                  className="indicator-dot indicator-red"
                  style={{
                    animation: "pulse-glow 1s ease-in-out infinite",
                  }}
                />
                REC
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Step Checklist + Evidence */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "16px",
          }}
        >
          {/* Experiment State Panel */}
          <div className="glass-card" style={{ padding: "16px" }}>
            <h3
              style={{
                fontSize: "0.85rem",
                fontWeight: 600,
                marginBottom: "12px",
                color: "#94a3b8",
              }}
            >
              📋 Experiment Progress
            </h3>
            {state ? (
              <>
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "12px",
                  }}
                >
                  <span style={{ fontSize: "1.1rem", fontWeight: 700 }}>
                    Step {state.current_step_id} of {state.total_steps}
                  </span>
                  <span
                    className={`status-badge status-${state.current_step_status}`}
                  >
                    {statusIcon[state.current_step_status]}{" "}
                    {state.current_step_status}
                  </span>
                </div>
                <p
                  style={{
                    fontSize: "0.85rem",
                    color: "#e2e8f0",
                    marginBottom: "8px",
                  }}
                >
                  {state.current_step_description}
                </p>
                <p
                  style={{
                    fontSize: "0.75rem",
                    color: "#64748b",
                  }}
                >
                  Next: {state.next_step_description}
                </p>

                {/* Progress bar */}
                <div
                  style={{
                    marginTop: "12px",
                    height: "4px",
                    background: "rgba(148, 163, 184, 0.1)",
                    borderRadius: "2px",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${(state.completed_steps / state.total_steps) * 100}%`,
                      background:
                        "linear-gradient(90deg, #3b82f6, #10b981)",
                      borderRadius: "2px",
                      transition: "width 0.5s ease",
                    }}
                  />
                </div>
              </>
            ) : (
              <p style={{ color: "#64748b", fontSize: "0.85rem" }}>
                Start a session to begin tracking
              </p>
            )}
          </div>

          {/* Step Checklist */}
          <div
            className="glass-card"
            style={{
              padding: "16px",
              flex: 1,
              overflow: "auto",
              maxHeight: "320px",
            }}
          >
            <h3
              style={{
                fontSize: "0.85rem",
                fontWeight: 600,
                marginBottom: "12px",
                color: "#94a3b8",
              }}
            >
              ✓ Step Checklist
            </h3>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "6px",
              }}
            >
              {(state?.step_states || []).map((step) => (
                <div
                  key={step.step_id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    padding: "8px 12px",
                    borderRadius: "8px",
                    background:
                      step.status === "IN_PROGRESS"
                        ? "rgba(59, 130, 246, 0.1)"
                        : step.status === "VALID"
                        ? "rgba(16, 185, 129, 0.05)"
                        : "transparent",
                    border:
                      step.status === "IN_PROGRESS"
                        ? "1px solid rgba(59, 130, 246, 0.2)"
                        : "1px solid transparent",
                    transition: "all 0.3s ease",
                    cursor: "pointer",
                  }}
                  onClick={() =>
                    sessionActive && skipToStep(step.step_id)
                  }
                  title={`Click to skip to step ${step.step_id}`}
                >
                  <span style={{ fontSize: "1rem" }}>
                    {statusIcon[step.status] || "⏳"}
                  </span>
                  <div style={{ flex: 1 }}>
                    <span
                      style={{
                        fontSize: "0.8rem",
                        fontWeight:
                          step.status === "IN_PROGRESS" ? 600 : 400,
                        color:
                          step.status === "VALID"
                            ? "#34d399"
                            : step.status === "IN_PROGRESS"
                            ? "#60a5fa"
                            : step.status === "SKIPPED"
                            ? "#f87171"
                            : "#94a3b8",
                      }}
                    >
                      {step.step_id}. {step.description}
                    </span>
                  </div>
                  <span
                    className={`status-badge status-${step.status}`}
                    style={{ fontSize: "0.65rem", padding: "2px 8px" }}
                  >
                    {step.status}
                  </span>
                </div>
              ))}

              {(!state || state.step_states.length === 0) && (
                <p
                  style={{
                    color: "#64748b",
                    fontSize: "0.8rem",
                    textAlign: "center",
                    padding: "20px",
                  }}
                >
                  No steps loaded
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Bottom Row: Evidence + Logs + Assistant */}
        <div
          style={{
            gridColumn: "1 / -1",
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: "16px",
          }}
        >
          {/* Evidence Panels */}
          <div className="glass-card" style={{ padding: "16px" }}>
            <h3
              style={{
                fontSize: "0.85rem",
                fontWeight: 600,
                marginBottom: "12px",
                color: "#94a3b8",
              }}
            >
              🔍 Detection Evidence
            </h3>

            {/* Person level */}
            <div style={{ marginBottom: "12px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "0.75rem",
                  marginBottom: "4px",
                }}
              >
                <span style={{ color: "#94a3b8" }}>Person</span>
                <span
                  style={{
                    color: state?.person_detected ? "#10b981" : "#ef4444",
                  }}
                >
                  {state?.person_detected
                    ? `Detected (${((state?.person_confidence || 0) * 100).toFixed(0)}%)`
                    : "Not detected"}
                </span>
              </div>
            </div>

            {/* Action level */}
            <div style={{ marginBottom: "12px" }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "0.75rem",
                  marginBottom: "4px",
                }}
              >
                <span style={{ color: "#94a3b8" }}>Action</span>
                <span style={{ color: "#60a5fa" }}>
                  {state?.latest_action || "idle"}{" "}
                  {state?.latest_action_target !== "none" &&
                    `→ ${state?.latest_action_target}`}
                </span>
              </div>
              {state?.latest_action_confidence !== undefined && (
                <div
                  style={{
                    height: "3px",
                    background: "rgba(148, 163, 184, 0.1)",
                    borderRadius: "2px",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${(state.latest_action_confidence || 0) * 100}%`,
                      background: "#3b82f6",
                      borderRadius: "2px",
                      transition: "width 0.3s",
                    }}
                  />
                </div>
              )}
            </div>

            {/* Object level */}
            <div style={{ marginBottom: "12px" }}>
              <div
                style={{
                  fontSize: "0.75rem",
                  color: "#94a3b8",
                  marginBottom: "6px",
                }}
              >
                Objects
              </div>
              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "4px",
                }}
              >
                {(state?.detected_objects || []).length > 0 ? (
                  state?.detected_objects.map((obj) => (
                    <span
                      key={obj}
                      style={{
                        fontSize: "0.7rem",
                        padding: "2px 8px",
                        borderRadius: "4px",
                        background:
                          obj === "red_box"
                            ? "rgba(239, 68, 68, 0.2)"
                            : obj === "yellow_box"
                            ? "rgba(245, 158, 11, 0.2)"
                            : "rgba(59, 130, 246, 0.2)",
                        color:
                          obj === "red_box"
                            ? "#f87171"
                            : obj === "yellow_box"
                            ? "#fbbf24"
                            : "#60a5fa",
                      }}
                    >
                      {obj.replace(/_/g, " ")}
                      {state?.object_states[obj] &&
                        ` [${state.object_states[obj]}]`}
                    </span>
                  ))
                ) : (
                  <span style={{ fontSize: "0.7rem", color: "#64748b" }}>
                    No objects detected
                  </span>
                )}
              </div>
            </div>

            {/* System Status */}
            <div>
              <div
                style={{
                  fontSize: "0.75rem",
                  color: "#94a3b8",
                  marginBottom: "6px",
                }}
              >
                System
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "4px",
                }}
              >
                {Object.entries(systemStatus).map(([key, active]) => (
                  <div
                    key={key}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "6px",
                      fontSize: "0.7rem",
                    }}
                  >
                    <div
                      className={`indicator-dot ${
                        active ? "indicator-green" : "indicator-red"
                      }`}
                    />
                    <span style={{ color: "#94a3b8" }}>
                      {key.replace(/([A-Z0-9])/g, " $1").trim()}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Live Log Feed */}
          <div
            className="glass-card"
            style={{
              padding: "16px",
              overflow: "hidden",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <h3
              style={{
                fontSize: "0.85rem",
                fontWeight: 600,
                marginBottom: "12px",
                color: "#94a3b8",
              }}
            >
              📜 Live Log
            </h3>
            <div
              style={{
                flex: 1,
                overflow: "auto",
                maxHeight: "250px",
              }}
            >
              {logs.length > 0 ? (
                logs.map((entry, i) => (
                  <div
                    key={`${entry.timestamp}-${i}`}
                    className="log-entry"
                    style={{
                      fontSize: "0.7rem",
                      fontFamily: "'JetBrains Mono', monospace",
                      padding: "6px 8px",
                      borderBottom: "1px solid rgba(148, 163, 184, 0.05)",
                      display: "flex",
                      gap: "8px",
                    }}
                  >
                    <span style={{ color: "#64748b", whiteSpace: "nowrap" }}>
                      {entry.elapsed_sec.toFixed(1)}s
                    </span>
                    <span
                      style={{
                        color:
                          entry.status === "VALID"
                            ? "#34d399"
                            : entry.status === "SKIPPED"
                            ? "#f87171"
                            : entry.status === "OUT_OF_ORDER"
                            ? "#fbbf24"
                            : "#60a5fa",
                        fontWeight: 600,
                        minWidth: "90px",
                      }}
                    >
                      {entry.status}
                    </span>
                    <span style={{ color: "#94a3b8" }}>
                      Step {entry.step_id}: {entry.label}
                    </span>
                  </div>
                ))
              ) : (
                <p
                  style={{
                    color: "#64748b",
                    fontSize: "0.8rem",
                    textAlign: "center",
                    padding: "20px",
                  }}
                >
                  No log entries yet
                </p>
              )}
            </div>
          </div>

          {/* Voice Assistant Widget */}
          <div className="glass-card" style={{ padding: "16px" }}>
            <h3
              style={{
                fontSize: "0.85rem",
                fontWeight: 600,
                marginBottom: "12px",
                color: "#94a3b8",
              }}
            >
              🎙️ Voice Assistant
            </h3>
            <div
              style={{
                display: "flex",
                gap: "8px",
                marginBottom: "12px",
              }}
            >
              <input
                type="text"
                value={qaQuestion}
                onChange={(e) => setQaQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && askQuestion()}
                placeholder="Ask about the experiment..."
                style={{
                  flex: 1,
                  background: "rgba(148, 163, 184, 0.05)",
                  border: "1px solid rgba(148, 163, 184, 0.15)",
                  borderRadius: "8px",
                  padding: "8px 12px",
                  color: "#f1f5f9",
                  fontSize: "0.8rem",
                  outline: "none",
                }}
              />
              <button
                className="btn-primary"
                onClick={askQuestion}
                disabled={qaLoading}
                style={{
                  padding: "8px 16px",
                  fontSize: "0.8rem",
                  opacity: qaLoading ? 0.6 : 1,
                }}
              >
                {qaLoading ? "..." : "Ask"}
              </button>
            </div>

            {qaAnswer && (
              <div
                className="fade-in"
                style={{
                  background: "rgba(59, 130, 246, 0.05)",
                  border: "1px solid rgba(59, 130, 246, 0.15)",
                  borderRadius: "8px",
                  padding: "12px",
                  fontSize: "0.8rem",
                  color: "#e2e8f0",
                  lineHeight: "1.5",
                  whiteSpace: "pre-line",
                }}
              >
                {qaAnswer}
              </div>
            )}

            {/* Quick questions */}
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "4px",
                marginTop: "12px",
              }}
            >
              {[
                "What's next?",
                "What objects do I need?",
                "List all steps",
                "Help",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    setQaQuestion(q);
                    setTimeout(() => askQuestion(), 100);
                  }}
                  style={{
                    fontSize: "0.65rem",
                    padding: "3px 8px",
                    borderRadius: "4px",
                    background: "rgba(148, 163, 184, 0.08)",
                    border: "1px solid rgba(148, 163, 184, 0.1)",
                    color: "#94a3b8",
                    cursor: "pointer",
                    transition: "all 0.2s",
                  }}
                  onMouseOver={(e) =>
                    ((e.target as HTMLElement).style.background =
                      "rgba(59, 130, 246, 0.15)")
                  }
                  onMouseOut={(e) =>
                    ((e.target as HTMLElement).style.background =
                      "rgba(148, 163, 184, 0.08)")
                  }
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
