"""The device serves its own display app (plan §7.5).

A local web page, so one codebase covers both stages: the prototype's 7"
screen opens it in a full-screen browser on the Pi, and a production head unit
opens it over the device's own network. Standard library only.

The link is one-way in authority. The page reads derived numbers (§8: never
frames, never landmarks) and can send back only what a driver would press on a
screen: a KSS rating, plus demo controls for speed and ignition while there is
no GPS. Losing the page costs nothing but the page.

Two exceptions send camera images to the page, both bounded:

* During guided calibration (`set_camera_live(True)`) the driver sees
  themselves to get into position. They have just consented, and the stream
  stops the moment calibration ends — an open stream is cut, not left running.
* `camera_view=True`, opt-in and for demos: the debug view (camera, eye
  points, readout) at /camera. That sends images of the driver off the device,
  which §8 otherwise forbids, so use it only with the consent of whoever is
  on camera.

Sound, also opt-in and for demos (`enable_sound`): a prototype with no speaker
of its own can have the page play what the device decided to say. The device
still decides, times and logs every word; the page is only a loudspeaker on a
cable, and it plays clips from the same pre-rendered bank by id — nothing is
generated there. A shipping unit speaks through its own speaker (§7.2).
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

_ALLOWED_COMMANDS = {
    "speed_delta", "ignition", "kss",                       # driving
    "new_driver", "name", "consent", "select_driver",       # who's driving (setup_flow)
    "delete_driver", "calibrate", "start", "back", "switch_driver",
}

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
        self.camera_live = False          # guided calibration only
        # The latest thing to say, for a page acting as the speaker. Only the
        # newest matters: the executor has already settled what may interrupt
        # what, so a page that missed one poll must not replay old alerts.
        self._sound: Dict = {"seq": 0, "clips": []}
        self._phrase_files: Dict[str, Path] = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, body: bytes, ctype: str, cache: str = "no-store") -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", cache)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/state":
                    with server._lock:
                        body = server._snapshot
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    # Which page and renderer this server would hand out now.
                    # A kiosk left open across an update sees it change and
                    # reloads, instead of drawing new screens with old code.
                    self.send_header("X-AWAS-UI", server.ui_version())
                    if server._phrase_files:
                        # Cheap to poll: the page fetches /sound only when this moves.
                        self.send_header("X-AWAS-Sound", str(server._sound["seq"]))
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path in ("/", "/index.html"):
                    self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/app-screens.js":
                    self._send(200, SCREENS_JS.read_bytes(), "text/javascript; charset=utf-8")
                elif path == "/sound" and server._phrase_files:
                    with server._lock:
                        body = json.dumps(server._sound).encode()
                    self._send(200, body, "application/json")
                elif path.startswith("/phrases/") and path.endswith(".wav") \
                        and path[len("/phrases/"):-4] in server._phrase_files:
                    # By clip id from the bank's own manifest, never a path.
                    wav = server._phrase_files[path[len("/phrases/"):-4]].read_bytes()
                    self._send(200, wav, "audio/wav", cache="max-age=3600")
                elif path == "/camera" and server.camera_view:
                    self._send(200, _CAMERA_PAGE, "text/html; charset=utf-8")
                elif path == "/camera.mjpg" and server.camera_allowed:
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
                    while server.camera_allowed:  # cut the moment calibration ends
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

    @staticmethod
    def ui_version() -> str:
        """Changes whenever the page or the renderer on disk changes."""
        return "-".join(str(int(p.stat().st_mtime_ns // 1_000_000))
                        for p in (WEB_DIR / "index.html", SCREENS_JS))

    @property
    def camera_allowed(self) -> bool:
        return self.camera_view or self.camera_live

    def set_camera_live(self, on: bool) -> None:
        if self.camera_live and not on:
            with self._lock:
                self._frame = None          # no stale image of the driver lingers
        self.camera_live = on

    def publish_frame(self, jpeg: bytes) -> None:
        """Latest camera frame; ignored unless a camera view is allowed."""
        if not self.camera_allowed:
            return
        with self._lock:
            self._frame = jpeg
            self._frame_seq += 1

    def enable_sound(self, phrase_files: Dict[str, Path]) -> None:
        """Let the page play the bank: {clip id: wav file}."""
        self._phrase_files = dict(phrase_files)

    def play_sound(self, clip_ids) -> None:
        with self._lock:
            self._sound = {"seq": self._sound["seq"] + 1, "clips": list(clip_ids)}

    def next_command(self) -> Optional[Dict]:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
