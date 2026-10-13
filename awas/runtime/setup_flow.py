"""Before driving: who is driving, their consent, and their calibration.

    select ──new──▶ new ──name──▶ consent ──agree──▶ calibrating ──▶ result ──start──▶ driving
      │                              └─decline─▶ select (nothing kept)         │
      ├──pick a calibrated driver──▶ ready ──start──────────────────────────────┤
      │                                └─recalibrate──▶ calibrating             │
      └──pick an uncalibrated one──▶ calibrating                                │
    driving ──switch driver──▶ select ◀──────────────────────────────────────────┘

The screens are commands from the display (a tap or a key); this module only
moves between them and says when driving starts or stops. While anything but
`driving` is showing, no alerts fire and nothing is logged as driving
(run_webcam enforces that), and the camera image goes to the display only
during `calibrating` (plan §8 — the driver is positioning themselves).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from awas.driver.guided import Frame, GuidedCalibration, profile_from
from awas.runtime.drivers import Driver, DriverNameError, DriverRegistry

Event = Tuple[str, Optional[Driver]]     # ("drive", driver) | ("end_drive", None)


@dataclass
class SetupFlow:
    registry: DriverRegistry
    screen: str = "select"
    driver: Optional[Driver] = None
    error: str = ""
    calibration: Optional[GuidedCalibration] = None
    _fresh: bool = False                  # created in this flow, nothing recorded yet
    events: List[Event] = field(default_factory=list)

    @property
    def driving(self) -> bool:
        return self.screen == "driving"

    @property
    def camera_wanted(self) -> bool:
        return self.screen == "calibrating"

    # ---------------- commands ----------------

    def command(self, cmd: dict) -> List[Event]:
        """Apply one display command; return any drive start/stop events."""
        self.error, self.events = "", []
        s = self.screen
        if "back" in cmd and s in ("new", "consent", "ready", "calibrating", "result"):
            self._back()
        elif "new_driver" in cmd and s == "select":
            self.screen = "new"
        elif "name" in cmd and s == "new":
            self._create(cmd["name"])
        elif "consent" in cmd and s == "consent":
            self._consent(bool(cmd["consent"]))
        elif "select_driver" in cmd and s == "select":
            self._select(str(cmd["select_driver"]))
        elif "delete_driver" in cmd and s == "select":
            self._delete(str(cmd["delete_driver"]))
        elif "calibrate" in cmd and s in ("ready", "result"):
            self._start_calibration()
        elif "start" in cmd and s in ("ready", "result") and self.driver and self.driver.calibrated:
            self.registry.touch(self.driver.id)
            self.screen = "driving"
            self.events.append(("drive", self.driver))
        elif "switch_driver" in cmd and s == "driving":
            self.events.append(("end_drive", self.driver))
            self.driver, self.screen = None, "select"
        return self.events

    def drive_now(self, driver_id: str) -> List[Event]:
        """Skip the screens: start straight away as this (calibrated) driver."""
        d = self.registry.get(driver_id)
        if d is None or not d.calibrated:
            raise KeyError(f"{driver_id!r} is not a calibrated driver")
        self.driver, self.screen = d, "driving"
        self.registry.touch(d.id)
        return [("drive", d)]

    def _back(self) -> None:
        if self._fresh and self.driver is not None:
            # Walked away before agreeing: keep nothing about them.
            self.registry.delete(self.driver.id)
        self.driver, self.calibration, self._fresh, self.screen = None, None, False, "select"

    def _create(self, raw) -> None:
        try:
            self.driver = self.registry.create(str(raw))
        except DriverNameError as exc:
            self.error = str(exc)
            return
        self._fresh, self.screen = True, "consent"

    def _consent(self, agreed: bool) -> None:
        if not agreed:
            self._back()
            return
        self.registry.record_consent(self.driver.id)
        self._fresh = False
        self._start_calibration()

    def _select(self, driver_id: str) -> None:
        d = self.registry.get(driver_id)
        if d is None:
            self.error = "That driver no longer exists."
            return
        self.driver = d
        if d.consent_at is None:
            self.screen = "consent"
        elif d.calibrated:
            self.screen = "ready"
        else:
            self._start_calibration()

    def _delete(self, driver_id: str) -> None:
        if self.registry.get(driver_id) is not None:
            self.registry.delete(driver_id)

    def _start_calibration(self) -> None:
        self.calibration, self.screen = GuidedCalibration(), "calibrating"

    # ---------------- calibration frames ----------------

    def feed(self, frame: Frame, profiles_dir: Path) -> None:
        """One camera frame while calibrating. Saves the profile on success."""
        if self.screen != "calibrating" or self.calibration is None:
            return
        self.calibration.feed(frame)
        result = self.calibration.result
        if result is None:
            return
        if result.ok:
            profile_from(result, self.driver.id).save(profiles_dir)
            self.registry.record_calibration(self.driver.id, result.eye_signal, result.summary())
            self.driver = self.registry.get(self.driver.id)
        self.screen = "result"

    # ---------------- display ----------------

    def snapshot(self) -> dict:
        snap = {"mode": "setup", "screen": self.screen, "error": self.error}
        if self.screen == "select":
            snap["drivers"] = [{"id": d.id, "name": d.name, "calibrated": d.calibrated,
                                "eye_signal": d.eye_signal} for d in self.registry.list()]
        if self.driver is not None:
            snap["driver"] = {"id": self.driver.id, "name": self.driver.name,
                              "calibrated": self.driver.calibrated,
                              "eye_signal": self.driver.eye_signal}
        cal = self.calibration
        if self.screen == "calibrating" and cal is not None:
            snap["calibration"] = {"step": cal.step, "checks": cal.checks, "paused": cal.paused,
                                   "progress": round(cal.progress, 2), "rounds": cal.rounds_done,
                                   "rounds_needed": cal.cfg.rounds, "closing": cal.closing_now}
        if self.screen == "result" and cal is not None and cal.result is not None:
            snap["result"] = cal.result.summary()
        return snap
