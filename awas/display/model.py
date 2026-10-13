"""What the centre screen shows, decided from live state (plan §7.5).

The screen is a view, never a component: it reads what Layer 1 and the state
machine already decided and picks one of the rev-B screens for it. Everything
here is pure, so the choice of screen is testable without a browser.

While the car moves the screen carries almost nothing to read — a band colour,
one lit segment, one word. Detail (the trip summary, the KSS question) appears
only when parked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from awas.core.states import DrivingState

# Suppression reasons that are housekeeping, not a decision worth showing.
_QUIET_SUPPRESSIONS = ("muted", "no clip", "clip missing", "urgency")

PARKED_STATES = (DrivingState.PARKED, DrivingState.SHUTDOWN)


def choose_screen(
    driving_state: DrivingState,
    level: int,
    face_present: bool,
    calibrating: bool,
    held_reason: Optional[str] = None,
    lane_available: bool = False,
    camera_ok: bool = True,
    manoeuvring: bool = False,
) -> str:
    """Pick the screen key. Order matters: safety outranks bookkeeping.

    A missing camera outranks everything — even the parked summary — because
    it is a fault someone has to fix before AWAS can do its job, and a calm
    trip screen would hide it. (It cannot hide a critical level: with no face
    the level is capped at 3, plan §5.4.)
    """
    if not camera_ok:
        return "nocamera"
    if driving_state in PARKED_STATES:
        return "parked"
    if manoeuvring:
        # The state machine is deliberately quiet; say so (plan §5.4).
        return "reversing"
    if level >= 4:
        return "critical"
    if not face_present:
        # With the lane still read, the vehicle channel carries on alone —
        # the degradation ladder's "lane only" row (plan §5.4), not blindness.
        return "occluded" if lane_available else "noface"
    if held_reason and level >= 1:
        return "gate"
    if calibrating:
        return "calibrating"
    return {0: "monitoring", 1: "tired", 2: "engaging", 3: "insist"}.get(level, "monitoring")


def is_display_worthy(suppressed: Optional[str]) -> bool:
    """A held-back alert is worth showing only when a gate decided it."""
    return bool(suppressed) and not any(q in suppressed for q in _QUIET_SUPPRESSIONS)


@dataclass
class TripTracker:
    """Running trip numbers for the parked screen.

    Distance integrates the speed the system was given; with no GPS yet that
    is the manual demo speed, and the screen is only as honest as its input.
    """

    bucket_sec: float = 10.0
    started_wall: float = field(default_factory=time.time)
    driving_sec: float = 0.0
    distance_km: float = 0.0
    nudges: int = 0
    _levels: List[int] = field(default_factory=list)   # worst level per bucket
    _bucket_start: Optional[float] = None
    _last_t: Optional[float] = None

    def update(self, t: float, level: int, speed_kmh: float, moving: bool) -> None:
        if self._last_t is not None:
            dt = max(0.0, t - self._last_t)
            if moving:
                self.driving_sec += dt
                self.distance_km += speed_kmh * dt / 3600.0
        self._last_t = t
        if self._bucket_start is None or t - self._bucket_start >= self.bucket_sec:
            self._bucket_start = t
            self._levels.append(level)
        else:
            self._levels[-1] = max(self._levels[-1], level)

    def note_alert(self) -> None:
        self.nudges += 1

    def chart(self, max_points: int = 60) -> List[float]:
        """Worst level per bucket, scaled 0..1, downsampled to fit the chart."""
        pts = self._levels
        if len(pts) > max_points:
            step = len(pts) / max_points
            pts = [max(pts[int(i * step):int((i + 1) * step)] or [0]) for i in range(max_points)]
        return [p / 4.0 for p in pts]

    def worst(self) -> Optional[Tuple[int, int]]:
        """Index range of the worst stretch in chart(), or None if it never rose."""
        pts = self.chart()
        if not pts or max(pts) == 0:
            return None
        peak = max(pts)
        i = pts.index(peak)
        j = i
        while j + 1 < len(pts) and pts[j + 1] == peak:
            j += 1
        return i, j


def _hhmm(epoch: float) -> str:
    return time.strftime("%H:%M", time.localtime(epoch))


def snapshot(
    *,
    driving_state: DrivingState,
    level: int,
    driver,                       # awas.driver.pipeline.DriverState
    speed_kmh: float,
    workload: Optional[float],
    lane_available: bool,
    held_reason: Optional[str],
    trip: TripTracker,
    kss: Optional[int],
    now: Optional[float] = None,
    sdlp_m: Optional[float] = None,
    sdlp_baseline_m: Optional[float] = None,
    fused=None,                   # awas.fusion.core.FusedEstimate
    camera_ok: bool = True,
    manoeuvring: bool = False,
) -> Dict:
    """Everything the page needs, as plain JSON."""
    now = time.time() if now is None else now
    worst = trip.worst()
    return {
        "screen": choose_screen(driving_state, level, driver.face_present,
                                driver.calibrating, held_reason, lane_available,
                                camera_ok, manoeuvring),
        "manoeuvring": manoeuvring,
        "camera_ok": camera_ok,
        "level": level,
        "clock": _hhmm(now),
        "driving_state": driving_state.value,
        "speed_kmh": round(speed_kmh),
        "workload": None if workload is None else round(workload, 2),
        "lane": lane_available,
        "face_present": driver.face_present,
        "confidence": round(driver.confidence, 2),
        "calibrating": driver.calibrating,
        "calibration_progress": round(driver.calibration_progress, 2),
        "microsleeps": driver.microsleep_count,
        "longest_closure_s": round(driver.longest_closure_ms / 1000.0, 1),
        "perclos": round(driver.perclos, 3),
        "held_reason": held_reason,
        "eye_source": getattr(driver, "eye_source", None),
        "sdlp_m": None if sdlp_m is None else round(sdlp_m, 3),
        "sdlp_baseline_m": None if sdlp_baseline_m is None else round(sdlp_baseline_m, 3),
        "fusion": None if fused is None else {
            "explain": fused.explain(),
            "agreement": None if fused.agreement is None else round(fused.agreement, 2),
            "confidence": round(fused.confidence, 2),
        },
        "trip": {
            "driving_min": int(trip.driving_sec // 60),
            "driving_sec": int(trip.driving_sec % 60),
            "distance_km": round(trip.distance_km, 1),
            "nudges": trip.nudges,
            "chart": trip.chart(),
            "worst": list(worst) if worst else None,
            "start": _hhmm(trip.started_wall),
            "end": _hhmm(now),
        },
        "kss": kss,
    }
