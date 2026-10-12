"""Layer 2 signals — the road and vehicle channel (plan §4.2).

Deliberately narrow. Every field here exists because it changes an intervention
decision; nothing is collected because it would be interesting. Pedestrian
detection, obstacle detection and collision prediction are ADAS functions and
are explicitly out of scope.

Sources are interchangeable behind `RoadSource`: a scripted mock for tests, a
driving simulator's telemetry, GPS alone, or a forward camera. The rest of the
system never learns which one it got.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Protocol


@dataclass
class RoadState:
    """What is known about the road and the vehicle at one instant."""

    t: float = 0.0

    # --- vehicle motion ---
    speed_kmh: float = 0.0
    speed_source: str = "none"        # "obd" | "gps" | "sim" | "mock" | "manual"

    # --- lane keeping: drowsiness evidence, not just a safety feature ---
    # Signed offset from lane centre in metres; positive is to the right.
    lane_offset_m: Optional[float] = None
    lane_departure: bool = False

    # --- steering ---
    steering_angle_deg: Optional[float] = None

    # --- traffic context ---
    vehicle_ahead_distance_m: Optional[float] = None
    traffic_light_color: Optional[str] = None
    road_class: Optional[str] = None  # "highway" | "arterial" | "urban"
    speed_limit_kmh: Optional[float] = None

    @property
    def moving(self) -> bool:
        return self.speed_kmh > 5.0

    @property
    def has_lane(self) -> bool:
        return self.lane_offset_m is not None


class RoadSource(Protocol):
    """Anything that can report road state."""

    def read(self, t: float) -> RoadState: ...
    def close(self) -> None: ...


class NullRoadSource:
    """No road channel at all — the honest Phase 1 state.

    Returns a state with speed from whatever the caller sets and nothing else.
    Downstream code must treat every `None` as "unknown", never as zero: a
    missing lane reading is not a centred vehicle.
    """

    def __init__(self, speed_kmh: float = 0.0):
        self.speed_kmh = speed_kmh

    def read(self, t: float) -> RoadState:
        return RoadState(t=t, speed_kmh=self.speed_kmh, speed_source="manual")

    def close(self) -> None:
        pass


@dataclass
class RoadScene:
    """One step of a scripted road scenario."""

    duration_sec: float
    label: str = ""
    speed_kmh: float = 0.0
    lane_offset_m: Optional[float] = None
    lane_departure: bool = False
    steering_angle_deg: Optional[float] = None
    vehicle_ahead_distance_m: Optional[float] = None
    traffic_light_color: Optional[str] = None
    road_class: Optional[str] = None
    speed_limit_kmh: Optional[float] = None


class MockRoadSource:
    """Replays scripted road scenes. Deterministic, no clock, no hardware."""

    def __init__(self, scenes: list, jitter_m: float = 0.0, seed: int = 0):
        if not scenes:
            raise ValueError("MockRoadSource needs at least one scene")
        self.scenes = scenes
        self.jitter_m = jitter_m
        import random
        self._rng = random.Random(seed)

    def read(self, t: float) -> RoadState:
        scene = self._scene_at(t)
        offset = scene.lane_offset_m
        if offset is not None and self.jitter_m:
            # Lane position is never perfectly steady; the wander is the signal.
            offset += self._rng.gauss(0.0, self.jitter_m)
        return RoadState(
            t=t, speed_kmh=scene.speed_kmh, speed_source="mock",
            lane_offset_m=offset, lane_departure=scene.lane_departure,
            steering_angle_deg=scene.steering_angle_deg,
            vehicle_ahead_distance_m=scene.vehicle_ahead_distance_m,
            traffic_light_color=scene.traffic_light_color,
            road_class=scene.road_class, speed_limit_kmh=scene.speed_limit_kmh,
        )

    def _scene_at(self, t: float) -> RoadScene:
        elapsed = 0.0
        for scene in self.scenes:
            if t < elapsed + scene.duration_sec:
                return scene
            elapsed += scene.duration_sec
        return self.scenes[-1]

    def close(self) -> None:
        pass


# ---------------- lane-keeping statistics ----------------

@dataclass(frozen=True)
class LaneConfig:
    window_sec: float = 60.0
    min_samples: int = 30
    capacity: int = 4000
    # A reversal counts only past this amplitude, so sensor noise does not
    # register as steering activity.
    reversal_deadband_deg: float = 2.0


@dataclass
class LaneStats:
    sdlp: Optional[float] = None            # metres, std dev of lane position
    steering_reversal_rate: Optional[float] = None   # reversals per minute
    departures: int = 0
    samples: int = 0

    @property
    def usable(self) -> bool:
        return self.sdlp is not None


class LaneTracker:
    """Accumulates lane position into SDLP — the vehicle channel's core measure.

    Standard Deviation of Lane Position is the most established behavioural
    measure of driver fatigue: a tired driver's lane keeping degrades before
    their face shows much. That is why this channel matters and why fusing it
    with the facial channel is the actual research claim (H2), rather than a
    second opinion on something already known.
    """

    def __init__(self, cfg: Optional[LaneConfig] = None):
        self.cfg = cfg or LaneConfig()
        self._pos: Deque[tuple] = deque(maxlen=self.cfg.capacity)
        self._steer: Deque[tuple] = deque(maxlen=self.cfg.capacity)
        self._departures = 0

    def add(self, state: RoadState) -> None:
        if state.lane_offset_m is not None:
            self._pos.append((state.t, state.lane_offset_m))
        if state.steering_angle_deg is not None:
            self._steer.append((state.t, state.steering_angle_deg))
        if state.lane_departure:
            self._departures += 1
        self._evict(state.t)

    def _evict(self, now: float) -> None:
        cutoff = now - self.cfg.window_sec
        for buf in (self._pos, self._steer):
            while buf and buf[0][0] < cutoff:
                buf.popleft()

    def stats(self) -> LaneStats:
        out = LaneStats(samples=len(self._pos), departures=self._departures)
        if len(self._pos) >= self.cfg.min_samples:
            import statistics
            out.sdlp = statistics.pstdev([p for _, p in self._pos])
        if len(self._steer) >= self.cfg.min_samples:
            out.steering_reversal_rate = self._reversal_rate()
        return out

    def _reversal_rate(self) -> Optional[float]:
        """Direction changes per minute, ignoring anything inside the deadband."""
        angles = [a for _, a in self._steer]
        span = self._steer[-1][0] - self._steer[0][0]
        if span <= 0:
            return None

        reversals, direction, anchor = 0, 0, angles[0]
        for a in angles[1:]:
            delta = a - anchor
            if abs(delta) < self.cfg.reversal_deadband_deg:
                continue
            new_dir = 1 if delta > 0 else -1
            if direction and new_dir != direction:
                reversals += 1
            direction, anchor = new_dir, a
        return reversals / (span / 60.0)

    def reset(self) -> None:
        self._pos.clear()
        self._steer.clear()
        self._departures = 0


# ---------------- per-driver SDLP baseline ----------------

@dataclass(frozen=True)
class SdlpCalibrationConfig:
    """Mirrors the facial channel's calibration, for the same reason.

    Lane-keeping precision is as personal as eye shape: a driver whose alert
    SDLP is 0.12 m and one whose alert SDLP is 0.20 m are both driving well.
    Comparing either against one absolute threshold repeats exactly the mistake
    a fixed EAR threshold makes.
    """

    calibration_window_sec: float = 900.0   # presumed alert this long into a trip
    min_windows: int = 8                    # SDLP estimates before a baseline
    quantile: float = 0.5                   # median of alert-period SDLP
    min_baseline_m: float = 0.04            # below this, the lane data is suspect
    max_baseline_m: float = 0.30            # above this, they were not alert


@dataclass
class SdlpProfile:
    name: str
    baseline_m: float
    samples: int

    def usable(self, cfg: SdlpCalibrationConfig) -> bool:
        return (self.samples >= cfg.min_windows
                and cfg.min_baseline_m <= self.baseline_m <= cfg.max_baseline_m)


class SdlpCalibrator:
    """Learns a driver's alert-driving SDLP, for the personal vehicle scale."""

    def __init__(self, name: str = "default",
                 cfg: Optional[SdlpCalibrationConfig] = None):
        self.name = name
        self.cfg = cfg or SdlpCalibrationConfig()
        self._values: list = []

    def observe(self, sdlp: Optional[float], trip_elapsed_sec: float,
                drowsiness_level: int = 0) -> bool:
        """Offer one SDLP estimate. Returns whether it was admitted.

        Gated like the facial baseline: early in the trip, and only while
        nothing already suggests drowsiness. A baseline learned from a tired
        driver is worse than no baseline, because it normalises the very
        degradation it is supposed to detect.
        """
        if sdlp is None or drowsiness_level > 0:
            return False
        if trip_elapsed_sec > self.cfg.calibration_window_sec:
            return False
        if not (self.cfg.min_baseline_m <= sdlp <= self.cfg.max_baseline_m):
            return False
        self._values.append(sdlp)
        return True

    def build(self) -> Optional[SdlpProfile]:
        if len(self._values) < self.cfg.min_windows:
            return None
        import statistics
        p = SdlpProfile(self.name,
                        baseline_m=statistics.median(self._values),
                        samples=len(self._values))
        return p if p.usable(self.cfg) else None

    @property
    def progress(self) -> float:
        return min(len(self._values) / max(self.cfg.min_windows, 1), 1.0)

    def reset(self) -> None:
        self._values.clear()
