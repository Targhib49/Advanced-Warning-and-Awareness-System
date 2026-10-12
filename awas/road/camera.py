"""Forward-camera road source — real lane data for the vehicle channel.

Turns a stream of road-facing frames into `RoadState`. This is what finally
gives the road channel something measured rather than scripted.

Deliberately conservative about failure. A frame where the lane cannot be found
yields `lane_offset_m = None`, and the lane tracker simply receives no sample —
so SDLP is computed over the frames that were actually seen rather than being
diluted by invented zeros. A stretch of road with no visible markings should
lower the vehicle channel's confidence, not fake its steadiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from awas.road.calibrate import load as load_calibration
from awas.road.lane_detect import LaneCalibration, LaneDetection, LaneDetector
from awas.road.objects import RoadObjectDetector, RoadObjects
from awas.road.signals import RoadState


@dataclass(frozen=True)
class CameraRoadConfig:
    # A wheel crosses the line at roughly half a lane minus half a vehicle:
    # (3.5 / 2) - (1.8 / 2) = 0.85 m of offset. Slightly conservative.
    departure_offset_m: float = 0.80

    # A departure must persist, so that one bad frame is a glitch and not an
    # alert. At 15 fps this is about a fifth of a second.
    departure_frames: int = 3

    # Consecutive failures before the road channel admits it is blind.
    blind_after_frames: int = 30


DEFAULT_CALIBRATION_PATH = Path("config/lane_calibration.json")


def default_calibration() -> Optional[LaneCalibration]:
    """A saved calibration if one exists, otherwise None.

    Returning None rather than a default is deliberate: the caller then decides
    whether to fall back to the built-in guess, and can warn that it is a guess.
    """
    return load_calibration(DEFAULT_CALIBRATION_PATH)


class CameraRoadSource:
    """Road state from a forward-facing camera.

    Frames are pushed in by whatever owns the capture, so this works equally
    over a live camera, a recorded clip, or a simulator's rendered output —
    and stays testable with synthetic frames.
    """

    def __init__(
        self,
        calibration: Optional[LaneCalibration] = None,
        config: Optional[CameraRoadConfig] = None,
        detector: Optional[LaneDetector] = None,
        objects: Optional[RoadObjectDetector] = None,
        speed_kmh: float = 0.0,
        road_class: Optional[str] = None,
    ):
        self.cfg = config or CameraRoadConfig()
        self.calibrated = calibration is not None
        if calibration is None:
            calibration = default_calibration()
            self.calibrated = calibration is not None
        self.detector = detector or LaneDetector(calibration)
        self.speed_kmh = speed_kmh
        self.road_class = road_class

        # Optional: without it, lanes still work and the traffic-light and
        # following-distance fields simply stay None, which is what the state
        # machine already reads as "not fitted".
        self.objects = objects
        self.last_objects = RoadObjects()

        self.last: LaneDetection = LaneDetection()
        self._departing = 0
        self._blind = 0

    def push(self, frame: np.ndarray, t: float) -> RoadState:
        """Process one forward frame and return the road state it implies."""
        det = self.detector.detect(frame)
        self.last = det

        objs = self.objects.detect(frame, t) if self.objects else RoadObjects()
        self.last_objects = objs
        light = objs.traffic_light_color if objs.traffic_light_score >= 0.4 else None

        if not det.found:
            self._blind += 1
            self._departing = 0
            return RoadState(
                t=t, speed_kmh=self.speed_kmh, speed_source="manual",
                vehicle_ahead_distance_m=objs.lead_vehicle_distance_m,
                traffic_light_color=light, road_class=self.road_class,
            )

        self._blind = 0
        offset = det.offset_m

        if abs(offset) >= self.cfg.departure_offset_m:
            self._departing += 1
        else:
            self._departing = 0

        return RoadState(
            t=t,
            speed_kmh=self.speed_kmh,
            speed_source="manual",
            lane_offset_m=offset,
            lane_departure=self._departing >= self.cfg.departure_frames,
            vehicle_ahead_distance_m=objs.lead_vehicle_distance_m,
            traffic_light_color=light,
            road_class=self.road_class,
        )

    @property
    def blind(self) -> bool:
        """True when the lane has been unreadable long enough to matter."""
        return self._blind >= self.cfg.blind_after_frames

    def read(self, t: float) -> RoadState:
        """RoadSource protocol. Without a pushed frame there is nothing to say."""
        return RoadState(t=t, speed_kmh=self.speed_kmh, speed_source="manual")

    def close(self) -> None:
        if self.objects is not None:
            self.objects.close()
