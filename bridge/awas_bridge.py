"""AWAS simulator bridge: ETS2 telemetry and road view -> the AWAS device.

Runs on the machine with Euro Truck Simulator 2. Reads the game's telemetry
through a telemetry plugin — scs-sdk-plugin's shared memory on Windows
(install_windows.ps1), AWAS's own plugin over local UDP on macOS
(mac/install_mac.sh) — captures the game window, and sends both to the
Raspberry Pi over UDP. The Pi does all the
sensing and deciding; this program only stands in for the truck's wiring.

    python awas_bridge.py                       # send to awas.local
    python awas_bridge.py --pi 10.199.54.202    # if .local names don't resolve
    python awas_bridge.py --check               # live readout, to set things up
    python awas_bridge.py --record trip1        # also save video + telemetry
    python awas_bridge.py --source mock --capture mock   # no game needed

Standalone on purpose: copy this one file and its requirements anywhere.
The wire format must match awas/road/sim.py on the device.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import time
from pathlib import Path

MAGIC = b"AWS1"
MAX_JPEG = 60_000
MS_TO_KMH = 3.6


# ---------------------------------------------------------------- telemetry

class Ets2Telemetry:
    """The game's state via truck_telemetry (scs-sdk-plugin shared memory)."""

    def __init__(self):
        import truck_telemetry
        self._tt = truck_telemetry
        self._ready = False

    def _init(self) -> bool:
        try:
            self._tt.init()
            self._ready = True
        except FileNotFoundError:
            self._ready = False      # game not running, or plugin not loaded
        except Exception as exc:     # an unsupported plugin version, usually
            print(f"telemetry: {exc}", file=sys.stderr)
            self._ready = False
        return self._ready

    def read(self):
        if not self._ready and not self._init():
            return None
        try:
            d = self._tt.get_data()
        except Exception:
            self._ready = False
            return None
        if not d.get("sdkActive", True):
            return None
        limit = d.get("speedLimit", 0.0) * MS_TO_KMH
        return {
            "source": "ets2",
            "speed_kmh": round(abs(d.get("speed", 0.0)) * MS_TO_KMH, 2),
            "steer": round(d.get("gameSteer", 0.0), 4),
            "engine_on": bool(d.get("engineEnabled", False)),
            "paused": bool(d.get("paused", False)),
            "speed_limit_kmh": round(limit, 1) if limit > 1 else None,
            # SCS SDK truck.engine.gear: <0 reverse, 0 neutral, >0 forward. The
            # driver is looking back while reversing, so AWAS pauses its
            # face-based alerts.
            "gear": int(d.get("gear", 0)),
            # "Active" is the SDK's truck.lblinker: true for as long as the
            # stalk is set, without blinking — what "signalled" means. ("On" is
            # truck.light.lblinker, the lamp itself, which flashes.)
            "blinker_left": bool(d.get("blinkerLeftActive", False)),
            "blinker_right": bool(d.get("blinkerRightActive", False)),
        }


class PluginUdpTelemetry:
    """The game's state from mac/awas_telemetry.dylib, which sends one JSON
    line per ~50 ms to 127.0.0.1:8098. (scs-sdk-plugin has no macOS build.)"""

    PORT = 8098
    STALE_SEC = 2.0

    def __init__(self, port: int = PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(("127.0.0.1", port))
        except OSError:
            sys.exit(f"Port {port} is in use — is another AWAS bridge already running?")
        self.sock.setblocking(False)
        self._last, self._at = None, 0.0

    def read(self):
        while True:
            try:
                data = self.sock.recv(4096)
            except (BlockingIOError, OSError):
                break
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if msg.get("bye"):
                self._last = None            # the game closed
                continue
            self._last, self._at = msg, time.monotonic()
        d = self._last
        if d is None:
            return None
        # A paused game may stop sending frames; its last word stands. A
        # running one that went quiet has crashed or unloaded the plugin.
        if time.monotonic() - self._at > self.STALE_SEC and not d.get("paused"):
            return None
        limit = d.get("speed_limit", 0.0) * MS_TO_KMH
        return {
            "source": "ets2",
            "speed_kmh": round(abs(d.get("speed", 0.0)) * MS_TO_KMH, 2),
            "steer": round(d.get("steer", 0.0), 4),
            "engine_on": bool(d.get("engine", False)),
            "paused": bool(d.get("paused", False)),
            "speed_limit_kmh": round(limit, 1) if limit > 1 else None,
            "gear": int(d.get("gear", 0)),
            "blinker_left": bool(d.get("lblinker", False)),     # the stalk, as on Windows
            "blinker_right": bool(d.get("rblinker", False)),
        }


def ets2_telemetry():
    return Ets2Telemetry() if sys.platform == "win32" else PluginUdpTelemetry()


class MockTelemetry:
    """A steady motorway drive with a little steering wander."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.rng = random.Random(1)

    def read(self):
        t = time.monotonic() - self.t0
        return {
            "source": "mock",
            "speed_kmh": round(80 + 4 * math.sin(t / 20), 2),
            "steer": round(0.02 * math.sin(t / 3) + self.rng.gauss(0, 0.004), 4),
            "engine_on": True, "paused": False, "speed_limit_kmh": 90.0,
            "gear": 6, "blinker_left": False, "blinker_right": False,
        }

    def lane_offset(self) -> float:
        t = time.monotonic() - self.t0
        return 0.25 * math.sin(t / 6) + 0.08 * math.sin(t / 1.7)


# ---------------------------------------------------------------- capture

class ScreenCapture:
    """The game window (Windows, macOS), or a whole monitor, via mss."""

    def __init__(self, mode: str, title: str, monitor: int, region):
        import mss
        self.mss = mss.mss()
        self.mode, self.title, self.monitor, self.region = mode, title, monitor, region
        self._rect, self._found_at, self._warned = None, 0.0, False
        if sys.platform == "win32":
            try:   # physical pixels, so window rectangles match what mss grabs
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass

    def _window_rect(self):
        if time.monotonic() - self._found_at < 2.0 and self._rect:
            return self._rect
        self._found_at = time.monotonic()
        if sys.platform == "darwin":
            return self._mac_window_rect()
        try:
            import pygetwindow
            wins = [w for w in pygetwindow.getWindowsWithTitle(self.title)
                    if w.width > 200 and w.height > 200 and not w.isMinimized]
        except Exception:
            wins = []
        if not wins:
            if not self._warned:
                print(f"capture: no window titled '{self.title}' yet; capturing the monitor",
                      file=sys.stderr)
                self._warned = True
            self._rect = None
            return None
        w = wins[0]
        self._warned = False
        # Trim the title bar and borders of a windowed game.
        self._rect = {"left": w.left + 8, "top": w.top + 31,
                      "width": max(1, w.width - 16), "height": max(1, w.height - 39)}
        return self._rect

    def _mac_window_rect(self):
        # Quartz names windows by owning app; reading other apps' window
        # titles would need the Screen Recording permission anyway.
        try:
            import Quartz
            infos = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID) or []
        except Exception:
            infos = []
        # Any window level: full screen, ETS2 draws at a raised level (25),
        # windowed at the normal one (0). Its small menu-bar strips are
        # dropped by size below.
        wins = [w for w in infos
                if self.title.lower() in str(w.get("kCGWindowOwnerName", "")).lower()]
        bounds = [w["kCGWindowBounds"] for w in wins]
        bounds = [b for b in bounds if b["Width"] > 200 and b["Height"] > 200]
        if not bounds:
            if not self._warned:
                print(f"capture: no '{self.title}' window yet; capturing the monitor",
                      file=sys.stderr)
                self._warned = True
            self._rect = None
            return None
        self._warned = False
        b = max(bounds, key=lambda b: b["Width"] * b["Height"])
        x, y, w, h = int(b["X"]), int(b["Y"]), int(b["Width"]), int(b["Height"])
        # Points, as mss expects on macOS. A windowed game has a title bar; a
        # full-screen one is exactly a monitor's size and has none.
        full = any(m["width"] == w and m["height"] == h for m in self.mss.monitors[1:])
        bar = 0 if full else 28
        self._rect = {"left": x, "top": y + bar, "width": w, "height": max(1, h - bar)}
        return self._rect

    def grab(self):
        import numpy as np
        if self.region:
            box = dict(zip(("left", "top", "width", "height"), self.region))
        elif self.mode == "window" and sys.platform in ("win32", "darwin") and self._window_rect():
            box = self._rect
        else:
            box = self.mss.monitors[self.monitor]
        shot = self.mss.grab(box)
        return np.asarray(shot)[:, :, :3]        # BGRA -> BGR


class MockRoad:
    """A road drawn in perspective, the car wandering in its lane."""

    def __init__(self, telemetry: MockTelemetry):
        self.tel = telemetry
        self.n = 0

    def grab(self):
        import cv2
        import numpy as np
        w, h = 960, 540
        img = np.zeros((h, w, 3), np.uint8)
        img[: h // 2] = (200, 160, 110)                     # sky
        img[h // 2:] = (70, 70, 70)                         # asphalt
        vx, vy = w // 2, int(h * 0.5)
        px_per_m = w / 6.0                                  # at the bottom edge
        centre = w / 2 - self.tel.lane_offset() * px_per_m
        self.n += 1
        for side, dashed in ((-1, False), (1, True)):
            xb = centre + side * 1.75 * px_per_m
            if not dashed:
                cv2.line(img, (int(xb), h), (vx, vy), (235, 235, 235), 6)
                continue
            for k in range(12):
                f0 = ((k * 2 + self.n * 0.4) % 24) / 24
                f1 = min(1.0, f0 + 1 / 24)
                p0 = (int(xb + (vx - xb) * f0 ** 0.5), int(h + (vy - h) * f0 ** 0.5))
                p1 = (int(xb + (vx - xb) * f1 ** 0.5), int(h + (vy - h) * f1 ** 0.5))
                cv2.line(img, p0, p1, (235, 235, 235), 6)
        return img


def encode(frame, width: int, quality: int):
    """Resize to `width` and JPEG-encode under one UDP datagram."""
    import cv2
    h, w = frame.shape[:2]
    if w != width:
        frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    q = quality
    while q >= 30:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and len(buf) <= MAX_JPEG:
            return buf.tobytes(), frame
        q -= 10
    return None, frame


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Send ETS2 telemetry and road view to AWAS.")
    ap.add_argument("--pi", default="awas.local", help="the AWAS device (name or IP)")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--source", choices=("ets2", "mock"), default="ets2")
    ap.add_argument("--capture", choices=("window", "monitor", "mock", "none"), default="window")
    ap.add_argument("--window-title", default="Euro Truck Simulator 2")
    ap.add_argument("--monitor", type=int, default=1, help="mss monitor index for --capture monitor")
    ap.add_argument("--region", default=None, help="explicit capture box: left,top,width,height")
    ap.add_argument("--fps", type=float, default=12.0, help="road frames per second")
    ap.add_argument("--rate", type=float, default=20.0, help="telemetry messages per second")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--check", action="store_true", help="print a live readout every second")
    ap.add_argument("--record", default=None, metavar="DIR", help="also save road.avi + telemetry.jsonl")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds")
    args = ap.parse_args()

    try:
        pi_ip = socket.gethostbyname(args.pi)
    except socket.gaierror:
        sys.exit(f"Cannot find '{args.pi}'. Use the Pi's IP instead: --pi 10.x.x.x "
                 f"(shown in the phone's hotspot device list).")
    dest = (pi_ip, args.port)

    tel = MockTelemetry() if args.source == "mock" else ets2_telemetry()
    if args.capture == "mock":
        cap = MockRoad(tel if isinstance(tel, MockTelemetry) else MockTelemetry())
    elif args.capture == "none":
        cap = None
    else:
        region = tuple(int(v) for v in args.region.split(",")) if args.region else None
        cap = ScreenCapture(args.capture, args.window_title, args.monitor, region)
        if sys.platform == "darwin" and not _mac_can_capture():
            print("capture: macOS has not allowed this app to record the screen. Allow it in "
                  "System Settings > Privacy & Security > Screen & System Audio Recording "
                  "(for Terminal), then restart the bridge. Until then the Pi sees no road.",
                  file=sys.stderr)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # macOS caps a datagram at the send buffer, 9 KB by default; a road frame
    # is up to MAX_JPEG. (Windows and Linux already allow ~64 KB.)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * MAX_JPEG)
    sock.setblocking(False)

    rec_video = rec_tel = None
    if args.record:
        out = Path(args.record)
        out.mkdir(parents=True, exist_ok=True)
        rec_tel = open(out / "telemetry.jsonl", "a", encoding="utf-8")
        print(f"recording to {out.resolve()}")

    print(f"sending to {args.pi} ({pi_ip}) UDP {args.port} — Ctrl+C to stop")
    t0 = time.monotonic()
    next_tel = next_frame = next_print = t0
    sent = {"telemetry": 0, "frames": 0, "no_game": 0}
    last_tel, last_ack, frame_bytes = None, None, 0

    try:
        while not args.duration or time.monotonic() - t0 < args.duration:
            now = time.monotonic()

            if now >= next_tel:
                next_tel += 1.0 / args.rate
                last_tel = tel.read()
                if last_tel is None:
                    sent["no_game"] += 1
                else:
                    sock.sendto(MAGIC + b"T" + json.dumps(last_tel).encode(), dest)
                    sent["telemetry"] += 1
                    if rec_tel:
                        rec_tel.write(json.dumps({"t": round(now - t0, 3), **last_tel}) + "\n")

            if cap is not None and now >= next_frame:
                next_frame += 1.0 / args.fps
                jpeg, small = encode(cap.grab(), args.width, args.quality)
                if jpeg:
                    sock.sendto(MAGIC + b"F" + jpeg, dest)
                    sent["frames"] += 1
                    frame_bytes = len(jpeg)
                    if args.record:
                        import cv2
                        if rec_video is None:
                            h, w = small.shape[:2]
                            rec_video = cv2.VideoWriter(str(Path(args.record) / "road.avi"),
                                                        cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (w, h))
                        rec_video.write(small)

            try:
                data, _ = sock.recvfrom(4096)
                if data[:5] == MAGIC + b"A":
                    last_ack = (now, json.loads(data[5:]))
            except (BlockingIOError, ConnectionResetError, OSError):
                pass

            if now >= next_print and (args.check or now - t0 < 1 or int(now - t0) % 10 == 0):
                next_print = now + 1.0
                _readout(sent, last_tel, last_ack, frame_bytes, now, args.source)

            wait = min(next_tel, next_frame if cap else next_tel) - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, 0.02))
    except KeyboardInterrupt:
        pass
    finally:
        if rec_video is not None:
            rec_video.release()
        if rec_tel:
            rec_tel.close()
        print(f"\nsent {sent['telemetry']} telemetry, {sent['frames']} frames")


def _mac_can_capture() -> bool:
    try:
        import Quartz
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        return True          # cannot tell (older macOS, or no pyobjc); assume yes


def _readout(sent, tel, ack, frame_bytes, now, source) -> None:
    if tel is None:
        game = ("waiting for ETS2 — is the game running, and was the SDK dialog accepted?"
                if source == "ets2" else "no telemetry")
    else:
        gear = tel.get("gear", 0)
        signal = "<" if tel.get("blinker_left") else ">" if tel.get("blinker_right") else "-"
        game = (f"{tel['speed_kmh']:5.1f} km/h  steer {tel['steer']:+.2f}  "
                f"gear {'R' if gear < 0 else gear}  signal {signal}  "
                f"engine {'on ' if tel['engine_on'] else 'off'}{'  PAUSED' if tel['paused'] else ''}")
    if ack is None or now - ack[0] > 3:
        pi = "Pi: no reply yet (is AWAS running with --sim?)"
    else:
        a = ack[1]
        lane = (f"lane {a['lane_offset_m']:+.2f} m" if a.get("lane_found")
                else f"lane not found ({a.get('lane_reason') or '-'})")
        seen = f", seen in {a['lane_seen_pct']}% of frames" if a.get("lane_seen_pct") is not None else ""
        pi = f"Pi: receiving ({a['frames']} frames)  {lane}{seen}"
    print(f"[{sent['telemetry']:6d} tel {sent['frames']:5d} frm {frame_bytes // 1000:2d} KB]  {game}  |  {pi}",
          flush=True)


if __name__ == "__main__":
    main()
