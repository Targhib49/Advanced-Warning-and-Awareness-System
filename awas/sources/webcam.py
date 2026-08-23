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

from awas.core.telemetry import Distraction, Telemetry
from awas.driver.landmarks import (
    DEFAULT_MODEL_NAME,
    Landmarker,
    MediaPipeLandmarker,
)
from awas.driver.pipeline import DriverPipeline, DriverState
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

        self.t0 = time.monotonic()
        self.speed_kmh = 60.0        # driven by the demo's keyboard controls
        self.ignition_on = True
        self.last_state: Optional[DriverState] = None
        self.last_frame = None

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
        return to_telemetry(state, self.speed_kmh, self.ignition_on)

    def close(self) -> None:
        try:
            self.pipeline.save_profile()
        finally:
            self.landmarker.close()
            if self.cap is not None:
                self.cap.release()
                self.cap = None


def _no_face(t: float):
    from awas.driver.landmarks import FaceObservation
    return FaceObservation(t=t, face_present=False)


def to_telemetry(state: DriverState, speed_kmh: float, ignition_on: bool) -> Telemetry:
    """Map driver state onto the snapshot the state machine reads."""
    return Telemetry(
        timestamp=state.t,
        drowsiness_level=state.drowsiness_level,
        confidence=state.confidence,
        workload=0.0,               # needs the road channel — Phase 2
        distraction=Distraction.NONE,
        perclos=state.perclos,
        microsleep_count=state.microsleep_count,
        blink_duration_mean_ms=state.blink_duration_mean_ms,
        face_present=state.face_present,
        eye_aspect_ratio=state.eye_aspect_ratio,
        head_pitch_deg=state.head_pitch_deg,
        head_yaw_deg=state.head_yaw_deg,
        speed_kmh=speed_kmh,
        speed_source="manual",
        parking_brake=None,         # no OBD-II in Phase 1
        ignition_on=ignition_on,
    )
