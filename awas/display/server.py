"""The device serves its own display app (plan §7.5).

A local web page, so one codebase covers both stages: the prototype's 7"
screen opens it in a full-screen browser on the Pi, and a production head unit
opens it over the device's own network. Standard library only.

The link is one-way in authority. The page reads derived numbers (§8: never
frames, never landmarks) and can send back only what a driver would press on a
screen: a KSS rating, plus demo controls for speed and ignition while there is
no GPS. Losing the page costs nothing but the page.

The one exception is opt-in and for demos: with `camera_view=True` the debug
view (camera, eye points, readout) is served at /camera. That sends images of
the driver off the device, which §8 otherwise forbids, so it is off unless
asked for and should be used only with the consent of whoever is on camera.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

WEB_DIR = Path(__file__).parent / "web"
# The renderer is shared with the rev-B mockup so the two cannot drift apart.
SCREENS_JS = Path(__file__).resolve().parents[2] / "hardware" / "rev-b" / "mockup" / "app-screens.js"

_ALLOWED_COMMANDS = {"speed_delta", "ignition", "kss"}

_CAMERA_PAGE = b"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta name="google" content="notranslate"><title>AWAS camera</title>
<body style="margin:0;background:#0B0D0E;color:#8E9BA1;font:14px ui-monospace,monospace;text-align:center">
<p>AWAS debug view &middot; demo only &middot; images of the driver are leaving the device</p>
<img src="/camera.mjpg" style="max-width:100%;max-height:calc(100vh - 50px)"></body>"""


class DisplayServer:
    """Serves the page, the latest snapshot, and a small command inbox."""

    def __init__(self, port: int, host: str = "0.0.0.0", camera_view: bool = False):
        self.commands: "queue.Queue[Dict]" = queue.Queue()
        self._snapshot = b"{}"
        self._frame: Optional[bytes] = None
        self._frame_seq = 0
        self._lock = threading.Lock()
        self.camera_view = camera_view
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/state":
                    with server._lock:
                        body = server._snapshot
                    self._send(200, body, "application/json")
                elif path in ("/", "/index.html"):
                    self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/app-screens.js":
                    self._send(200, SCREENS_JS.read_bytes(), "text/javascript; charset=utf-8")
                elif path == "/camera" and server.camera_view:
                    self._send(200, _CAMERA_PAGE, "text/html; charset=utf-8")
                elif path == "/camera.mjpg" and server.camera_view:
                    self._stream()
                else:
                    self._send(404, b"not found", "text/plain")

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with server._lock:
                            jpg, seq = server._frame, server._frame_seq
                        if jpg is not None and seq != seen:
                            seen = seq
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self):
                if self.path != "/command":
                    self._send(404, b"not found", "text/plain")
                    return
                try:
                    length = min(int(self.headers.get("Content-Length", 0)), 1024)
                    cmd = json.loads(self.rfile.read(length) or b"{}")
                except (ValueError, json.JSONDecodeError):
                    self._send(400, b"bad json", "text/plain")
                    return
                if not isinstance(cmd, dict) or not set(cmd) <= _ALLOWED_COMMANDS:
                    self._send(400, b"unknown command", "text/plain")
                    return
                server.commands.put(cmd)
                self._send(204, b"", "text/plain")

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> "DisplayServer":
        self._thread.start()
        return self

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def publish(self, snapshot: Dict) -> None:
        body = json.dumps(snapshot, separators=(",", ":")).encode()
        with self._lock:
            self._snapshot = body

    def publish_frame(self, jpeg: bytes) -> None:
        """Latest debug-view frame; ignored unless camera_view is on."""
        if not self.camera_view:
            return
        with self._lock:
            self._frame = jpeg
            self._frame_seq += 1

    def next_command(self) -> Optional[Dict]:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
