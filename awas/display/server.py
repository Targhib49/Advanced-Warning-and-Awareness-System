"""The device serves its own display app (plan §7.5).

A local web page, so one codebase covers both stages: the prototype's 7"
screen opens it in a full-screen browser on the Pi, and a production head unit
opens it over the device's own network. Standard library only.

The link is one-way in authority. The page reads derived numbers (§8: never
frames, never landmarks) and can send back only what a driver would press on a
screen: a KSS rating, plus demo controls for speed and ignition while there is
no GPS. Losing the page costs nothing but the page.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

WEB_DIR = Path(__file__).parent / "web"
# The renderer is shared with the rev-B mockup so the two cannot drift apart.
SCREENS_JS = Path(__file__).resolve().parents[2] / "hardware" / "rev-b" / "mockup" / "app-screens.js"

_ALLOWED_COMMANDS = {"speed_delta", "ignition", "kss"}


class DisplayServer:
    """Serves the page, the latest snapshot, and a small command inbox."""

    def __init__(self, port: int, host: str = "0.0.0.0"):
        self.commands: "queue.Queue[Dict]" = queue.Queue()
        self._snapshot = b"{}"
        self._lock = threading.Lock()
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
                else:
                    self._send(404, b"not found", "text/plain")

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

    def next_command(self) -> Optional[Dict]:
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
