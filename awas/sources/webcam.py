"""Webcam telemetry source — the Phase 1 demo path (plan v3 §11).

Camera and landmark model live here; all the actual reasoning is in
`awas.driver.pipeline`, which needs neither. Produces `Telemetry` so the
state machine cannot tell a webcam from the mock source.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

from awas.core.telemetry import Telemetry
from awas.driver.landmarks import (
    DEFAULT_MODEL_NAME,
    Landmarker,
    MediaPipeLandmarker,
)
from awas.driver.pipeline import DriverPipeline, DriverState
from awas.fusion.telemetry import build_telemetry, fuse_channels
from awas.road.signals import LaneTracker, NullRoadSource, RoadSource
from awas.road.workload import WorkloadEstimator
from awas.sources.base import Source


def default_model_path() -> Path:
    return Path(os.environ.get(
        "AWAS_FACE_MODEL", str(Path("models") / DEFAULT_MODEL_NAME)
    )).expanduser()


class WebcamSource(Source):
    """Live camera -> Telemetry.

    The vehicle fields are stubs: Phase 1 has no OBD-II and no road camera, so
    speed is supplied by the caller (a keyboard control in the demo) rather
    than invented. `parking_brake` stays None, which the state machine already
    reads as "no OBD-II present" (plan §7.4).
    """

    def __init__(
        self,
        camera_index=0,          # int index, or a path to a video file
        driver_name: str = "default",
        profile_dir: Optional[Path] = None,
        model_path: Optional[Path] = None,
        landmarker: Optional[Landmarker] = None,
        road: Optional[RoadSource] = None,
        width: int = 640,
        height: int = 480,
    ):
        import cv2

        self.pipeline = DriverPipeline(driver_name=driver_name, profile_dir=profile_dir)
        self.landmarker = landmarker or MediaPipeLandmarker(
            model_path or default_model_path()
        )

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open capture source {camera_index!r}. For a camera "
                f"on macOS, the terminal needs camera permission in "
                f"System Settings > Privacy & Security > Camera."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # The road channel. NullRoadSource is the honest Phase 1 default: speed
        # comes from wherever the caller sets it and nothing else is known.
        self.road_source = road or NullRoadSource()
        self.lane = LaneTracker()
        self.workload_est = WorkloadEstimator()

        self.t0 = time.monotonic()
        self.speed_kmh = 60.0        # driven by the demo's keyboard controls
        self.ignition_on = True
        self.last_state: Optional[DriverState] = None
        self.last_frame = None
        self.last_road = None
        self.last_workload = None
        self.last_fused = None

    def read(self) -> Telemetry:
        import cv2

        t = time.monotonic() - self.t0
        ok, frame = self.cap.read()
        if not ok:
            # A dropped frame is an absent face, never a closed eye.
            state = self.pipeline.process(_no_face(t), t)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame = frame
            state = self.pipeline.process(self.landmarker.observe(rgb, t), t)

        self.last_state = state

        if isinstance(self.road_source, NullRoadSource):
            self.road_source.speed_kmh = self.speed_kmh
        elif hasattr(self.road_source, "manual_speed_kmh"):
            self.road_source.manual_speed_kmh = self.speed_kmh   # used only with no simulator
        road = self.road_source.read(t)
        self.lane.add(road)
        lane_stats = self.lane.stats()
        workload = self.workload_est.update(road)

        self.last_road = road
        self.last_workload = workload
        self.last_fused = fuse_channels(state, lane_stats)

        # A road source that knows the engine (a simulator, later OBD-II) wins
        # over the demo's manual ignition key.
        ignition = getattr(self.road_source, "ignition_on", None)
        return build_telemetry(
            driver=state, road=road, lane=lane_stats, workload=workload,
            ignition_on=self.ignition_on if ignition is None else ignition,
            fused=self.last_fused,
        )

    def close(self) -> None:
        try:
            self.pipeline.save_profile()
        finally:
            self.landmarker.close()
            self.road_source.close()
            if self.cap is not None:
                self.cap.release()
                self.cap = None


def _no_face(t: float):
    from awas.driver.landmarks import FaceObservation
    return FaceObservation(t=t, face_present=False)


def to_telemetry(state: DriverState, speed_kmh: float, ignition_on: bool) -> Telemetry:
    """Facial-channel-only telemetry — condition E, and the Phase 1 default.

    Kept because it is exactly the ablation H2 compares against: the same
    driver state with no vehicle channel to corroborate or contradict it.
    """
    from awas.road.signals import RoadState
    return build_telemetry(
        driver=state,
        road=RoadState(t=state.t, speed_kmh=speed_kmh, speed_source="manual"),
        lane=None, workload=None, ignition_on=ignition_on,
    )
