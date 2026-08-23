"""Scripted mock source — replays a predefined sequence of telemetry.

Time is virtual: it advances by `tick_sec` on every `read()`, so tests are
deterministic and a scenario runs faster than real time.
"""

from dataclasses import dataclass, fields
from typing import List, Optional

from awas.core.telemetry import Distraction, Telemetry
from awas.sources.base import Source


@dataclass
class Scene:
    """One step in a scripted scenario."""

    duration_sec: float
    label: str = ""

    speed_kmh: float = 0.0
    ignition_on: bool = True
    parking_brake: Optional[bool] = False

    drowsiness_level: int = 0
    confidence: float = 1.0
    workload: float = 0.0
    distraction: Distraction = Distraction.NONE
    face_present: bool = True

    traffic_light_color: Optional[str] = None
    vehicle_ahead_distance_m: Optional[float] = None
    lane_departure: bool = False


# Scene fields that map straight onto Telemetry fields of the same name.
_PASSTHROUGH = {f.name for f in fields(Scene)} - {"duration_sec", "label"}


class MockSource(Source):
    def __init__(self, scenes: List[Scene], tick_sec: float = 1.0):
        if not scenes:
            raise ValueError("MockSource needs at least one scene")
        self.scenes = scenes
        self.tick_sec = tick_sec
        self.now: float = 0.0

    def read(self) -> Telemetry:
        scene = self._scene_at(self.now)
        t = Telemetry(
            timestamp=self.now,
            speed_source="mock",
            **{name: getattr(scene, name) for name in _PASSTHROUGH},
        )
        self.now += self.tick_sec
        return t

    def current_label(self) -> str:
        # read() already advanced the clock, so look back one tick.
        return self._scene_at(self.now - self.tick_sec).label

    def is_done(self) -> bool:
        return self.now > sum(s.duration_sec for s in self.scenes)

    def _scene_at(self, t: float) -> Scene:
        elapsed = 0.0
        for scene in self.scenes:
            if t < elapsed + scene.duration_sec:
                return scene
            elapsed += scene.duration_sec
        return self.scenes[-1]  # past the end: hold the last scene
