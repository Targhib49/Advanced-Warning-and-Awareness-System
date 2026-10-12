"""Driving-simulator road source: ETS2 telemetry plus its rendered road view.

The simulator runs on a separate Windows machine. A small bridge there
(`bridge/awas_bridge.py`) reads the game's telemetry SDK and captures the game
window, and sends both here over UDP. The device still does every bit of
sensing and deciding (plan §7.5); the simulator only stands in for the truck.

What the SDK gives directly: speed, steering, engine, pause. What it does not:
lane position. So lane position comes the same way it would in a real cab —
from the forward view, through the same `CameraRoadSource` a dashcam would
feed. The game's rendered road simply takes the place of the camera.

Wire format, one UDP datagram per message (kept in step with the bridge):

    b"AWS1T" + JSON        telemetry, ~20 Hz
    b"AWS1F" + JPEG        one road frame, ~12 Hz, under 60 KB

and the device answers each sender about once a second with

    b"AWS1A" + JSON        what it is receiving and whether it sees the lane
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from awas.road.camera import CameraRoadSource
from awas.road.lane_detect import LaneCalibration
from awas.road.signals import RoadState

MAGIC = b"AWS1"
MAX_DATAGRAM = 65_507


@dataclass(frozen=True)
class SimConfig:
    port: int = 8097
    # Telemetry older than this means the bridge or the game has gone; the
    # vehicle channel then reports nothing rather than a frozen last value.
    stale_after_sec: float = 1.0
    # Steering arrives as -1..1. Degrees of steering-wheel angle are what the
    # reversal deadband is written in; ETS2's default wheel is 900° lock to lock.
    steering_range_deg: float = 900.0
    ack_every_sec: float = 1.0


@dataclass
class SimTelemetry:
    """The fields the bridge sends that AWAS uses."""

    speed_kmh: float = 0.0
    steer: float = 0.0               # game steering, -1 (left) .. 1 (right)
    engine_on: bool = False
    paused: bool = False
    speed_limit_kmh: Optional[float] = None
    source: str = "ets2"

    @classmethod
    def from_json(cls, raw: bytes) -> "SimTelemetry":
        d = json.loads(raw)
        if not isinstance(d, dict):
            raise ValueError("telemetry must be a JSON object")
        limit = d.get("speed_limit_kmh")
        return cls(
            speed_kmh=abs(float(d.get("speed_kmh", 0.0))),
            steer=max(-1.0, min(1.0, float(d.get("steer", 0.0)))),
            engine_on=bool(d.get("engine_on", False)),
            paused=bool(d.get("paused", False)),
            speed_limit_kmh=float(limit) if limit else None,
            source=str(d.get("source", "ets2"))[:16],
        )


class SimRoadSource:
    """RoadSource fed by the simulator bridge."""

    def __init__(self, config: Optional[SimConfig] = None,
                 calibration: Optional[LaneCalibration] = None,
                 camera: Optional[CameraRoadSource] = None,
                 bind: str = "0.0.0.0"):
        self.cfg = config or SimConfig()
        self.camera = camera or CameraRoadSource(calibration=calibration)

        self._lock = threading.Lock()
        self._telemetry: Optional[SimTelemetry] = None
        self._telemetry_at = 0.0
        self._frame: Optional[bytes] = None
        self._frame_seq = 0
        self._used_seq = 0
        self._peers: dict = {}
        self.counts = {"telemetry": 0, "frames": 0, "rejected": 0}
        # Set by the owner (the demo's [ / ] keys) so a bench run with no game
        # connected still has a speed — reported as "manual", never as "sim".
        self.manual_speed_kmh: Optional[float] = None
        self.lane_seen = 0
        self.lane_tried = 0

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind, self.cfg.port))
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    # ---------------- receiving ----------------

    def _listen(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(MAX_DATAGRAM)
            except socket.timeout:
                continue
            except OSError:
                break
            self.handle(data, addr)

    def handle(self, data: bytes, addr=None) -> bool:
        """Accept one datagram. Public so tests can feed it without a socket."""
        if len(data) < 5 or data[:4] != MAGIC:
            self.counts["rejected"] += 1
            return False
        kind, payload = data[4:5], data[5:]
        now = time.monotonic()
        try:
            if kind == b"T":
                tel = SimTelemetry.from_json(payload)
                with self._lock:
                    self._telemetry, self._telemetry_at = tel, now
                self.counts["telemetry"] += 1
            elif kind == b"F" and payload[:2] == b"\xff\xd8":     # a JPEG
                with self._lock:
                    self._frame = payload
                    self._frame_seq += 1
                self.counts["frames"] += 1
            else:
                raise ValueError("unknown message")
        except (ValueError, TypeError, json.JSONDecodeError):
            self.counts["rejected"] += 1
            return False
        if addr is not None:
            self._maybe_ack(addr, now)
        return True

    def _maybe_ack(self, addr, now: float) -> None:
        if now - self._peers.get(addr, 0.0) < self.cfg.ack_every_sec:
            return
        self._peers[addr] = now
        status = self.status()
        try:
            self._sock.sendto(MAGIC + b"A" + json.dumps(status).encode(), addr)
        except OSError:
            pass

    # ---------------- RoadSource ----------------

    @property
    def connected(self) -> bool:
        with self._lock:
            return (self._telemetry is not None
                    and time.monotonic() - self._telemetry_at < self.cfg.stale_after_sec)

    @property
    def ignition_on(self) -> Optional[bool]:
        """The game's engine state, or None when there is no live game."""
        with self._lock:
            tel, at = self._telemetry, self._telemetry_at
        if tel is None or time.monotonic() - at >= self.cfg.stale_after_sec:
            return None
        return tel.engine_on

    def read(self, t: float) -> RoadState:
        with self._lock:
            tel, at = self._telemetry, self._telemetry_at
            frame, seq = self._frame, self._frame_seq

        if tel is None or time.monotonic() - at >= self.cfg.stale_after_sec:
            # No live game: never replay a frozen speed. Fall back to the
            # manual demo speed if there is one, labelled as such.
            if self.manual_speed_kmh is not None:
                return RoadState(t=t, speed_kmh=self.manual_speed_kmh, speed_source="manual")
            return RoadState(t=t, speed_kmh=0.0, speed_source="none")

        speed = 0.0 if tel.paused else tel.speed_kmh
        state = RoadState(
            t=t, speed_kmh=speed, speed_source="sim",
            steering_angle_deg=tel.steer * self.cfg.steering_range_deg / 2.0,
            speed_limit_kmh=tel.speed_limit_kmh,
        )

        # Only the newest frame matters; older ones were superseded in flight.
        if frame is not None and seq != self._used_seq and not tel.paused:
            self._used_seq = seq
            image = _decode(frame, self.camera.detector.cal)
            if image is not None:
                self.camera.speed_kmh = speed
                cam = self.camera.push(image, t)
                self.lane_tried += 1
                if cam.lane_offset_m is not None:
                    self.lane_seen += 1
                state.lane_offset_m = cam.lane_offset_m
                state.lane_departure = cam.lane_departure
        return state

    def status(self) -> dict:
        tel = self._telemetry
        det = self.camera.last
        return {
            "connected": self.connected,
            "telemetry": self.counts["telemetry"],
            "frames": self.counts["frames"],
            "rejected": self.counts["rejected"],
            "speed_kmh": round(tel.speed_kmh, 1) if tel else None,
            "lane_found": det.found,
            "lane_offset_m": round(det.offset_m, 3) if det.found else None,
            "lane_reason": det.reason,
            "lane_seen_pct": round(100.0 * self.lane_seen / self.lane_tried) if self.lane_tried else None,
            "calibrated": self.camera.calibrated,
        }

    def close(self) -> None:
        self._running = False
        try:
            self._sock.close()
        finally:
            self.camera.close()


def _decode(jpeg: bytes, cal: LaneCalibration) -> Optional[np.ndarray]:
    """JPEG to a frame at the size the lane calibration was made for.

    The calibration's trapezoid is in pixels of that size, so a game window of
    any other size is scaled to it first. Calibrate from recorded frames
    (`bridge/awas_bridge.py --record`) passed through this same function.
    """
    import cv2
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[:2] != (cal.height, cal.width):
        img = cv2.resize(img, (cal.width, cal.height), interpolation=cv2.INTER_AREA)
    return img
