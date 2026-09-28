"""
Network Sink — MJPEG-over-HTTP streaming server.

Streams annotated frames to any browser or player at:
  http://<STREAM_HOST>:<STREAM_PORT>/video_feed

Receive with: ffplay http://localhost:8554/video_feed
Or open in any browser.
"""

from __future__ import annotations

import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP request handler that serves MJPEG stream."""

    # Class-level shared frame (set by NetworkSink)
    _current_frame: Optional[bytes] = None
    _frame_lock = threading.Lock()

    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            try:
                while True:
                    with MJPEGHandler._frame_lock:
                        frame_data = MJPEGHandler._current_frame

                    if frame_data is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(frame_data)}\r\n'.encode())
                        self.wfile.write(b'\r\n')
                        self.wfile.write(frame_data)
                        self.wfile.write(b'\r\n')

                    time.sleep(0.066)  # ~15 FPS cap

            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        elif self.path == '/':
            # Simple HTML page with embedded video
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html = (
                '<html><head><title>SIH Live Stream</title></head>'
                '<body style="background:#111;display:flex;justify-content:center;'
                'align-items:center;height:100vh;margin:0">'
                '<img src="/video_feed" style="max-width:100%;max-height:100%">'
                '</body></html>'
            )
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default HTTP logging
        pass


class NetworkSink:
    """
    MJPEG-over-HTTP network sink. Runs an HTTP server in a background thread
    that serves the latest annotated frame.

    Usage:
        sink = NetworkSink(host="0.0.0.0", port=8554)
        sink.start()
        # In the loop:
        sink.update_frame(annotated_bgr_frame)
        # When done:
        sink.stop()

    Receive: ffplay http://localhost:8554/video_feed
             or open http://localhost:8554/ in a browser
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8554):
        self._host = host
        self._port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the MJPEG streaming server."""
        try:
            self._server = HTTPServer((self._host, self._port), MJPEGHandler)
            self._running = True
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            logger.info("MJPEG stream server started at http://%s:%d/video_feed",
                        self._host, self._port)
        except OSError as e:
            logger.error("Failed to start stream server on port %d: %s",
                         self._port, e)
            self._running = False

    def _serve(self):
        while self._running:
            self._server.handle_request()

    def update_frame(self, frame: np.ndarray) -> None:
        """Update the current frame to be streamed."""
        if not self._running:
            return

        # Encode to JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with MJPEGHandler._frame_lock:
            MJPEGHandler._current_frame = jpeg.tobytes()

    def stop(self):
        """Stop the streaming server."""
        self._running = False
        if self._server:
            self._server.shutdown()
        logger.info("MJPEG stream server stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stream_url(self) -> str:
        return f"http://{self._host}:{self._port}/video_feed"
