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
from awas.road.signals import (
    LaneTracker, NullRoadSource, RoadSource, SdlpCalibrator, SdlpProfile,
)
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
        eye_model=None,          # awas.driver.eyestate.EyeStateClassifier, or None for EAR only
        fusion_mode: str = "weighted",
    ):
        import cv2

        self.eye_model = eye_model
        self.fusion_mode = fusion_mode
        # The vehicle channel's personal scale (fusion.core: excess over the
        # driver's own alert SDLP). Offered one SDLP estimate per interval —
        # consecutive per-frame values share almost the whole window and would
        # count one minute of driving many times over.
        self.sdlp_cal = SdlpCalibrator(driver_name)
        self.sdlp_baseline: Optional[SdlpProfile] = None
        self._sdlp_offered_at: Optional[float] = None
        self.sdlp_offer_every_sec = 30.0
        self.pipeline = DriverPipeline(driver_name=driver_name, profile_dir=profile_dir,
                                       eye_signal="cnn" if eye_model else "ear")
        self.landmarker = landmarker or MediaPipeLandmarker(
            model_path or default_model_path()
        )

        # A live camera may be missing at start or unplugged mid-trip. That is
        # a monitoring failure to SHOW, not a reason to exit: the dashboard and
        # the vehicle channel keep running while the camera is retried. A video
        # file that will not open is a mistake in the command, and still fails.
        self.camera_index = camera_index
        self.width, self.height = width, height
        self.cap = None
        self.camera_ok = False
        self._read_failures = 0
        self._next_open_at = 0.0
        self._open_camera()
        if self.cap is None and not isinstance(camera_index, int):
            raise RuntimeError(f"Could not open video file {camera_index!r}.")

        # The road channel. NullRoadSource is the honest Phase 1 default: speed
        # comes from wherever the caller sets it and nothing else is known.
        self.road_source = road or NullRoadSource()
        self.lane = LaneTracker()
        self.workload_est = WorkloadEstimator()

        self.t0 = time.monotonic()
        self.trip_started = 0.0      # seconds after t0 that this driver's trip began
        self.last_obs = None
        self.speed_kmh = 60.0        # driven by the demo's keyboard controls
        self.ignition_on = True
        self.last_state: Optional[DriverState] = None
        self.last_frame = None
        self.last_road = None
        self.last_workload = None
        self.last_fused = None
        self.last_lane_stats = None

    # ---------------- camera ----------------

    RETRY_EVERY_SEC = 2.0
    # Consecutive failed reads before a live camera is declared gone. Isolated
    # failures happen; a run of them at 15-30 fps is an unplugged cable.
    FAILURES_TO_LOSE = 30

    def _open_camera(self) -> bool:
        import cv2
        import sys

        self._next_open_at = time.monotonic() + self.RETRY_EVERY_SEC
        idx = self.camera_index
        # On Linux, probing a /dev/video node that is not there makes OpenCV
        # print a warning on every retry; check first and stay quiet.
        if isinstance(idx, int) and sys.platform.startswith("linux"):
            if not os.path.exists(f"/dev/video{idx}"):
                return False
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap, self.camera_ok, self._read_failures = cap, True, 0
        return True

    def _drop_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap, self.camera_ok = None, False
        self._next_open_at = time.monotonic() + self.RETRY_EVERY_SEC

    def _grab(self):
        """(ok, frame), retrying a missing live camera without spinning."""
        if self.cap is None:
            if isinstance(self.camera_index, int) and time.monotonic() >= self._next_open_at:
                self._open_camera()
            if self.cap is None:
                time.sleep(0.1)        # nothing to wait on; keep ~10 Hz, not 100% CPU
                return False, None
        ok, frame = self.cap.read()
        if ok:
            self._read_failures = 0
            return True, frame
        self._read_failures += 1
        if isinstance(self.camera_index, int) and self._read_failures >= self.FAILURES_TO_LOSE:
            self._drop_camera()
        return False, None

    def read(self) -> Telemetry:
        import cv2

        # `t` keeps rising for the life of the process (MediaPipe's video mode
        # refuses a timestamp that goes back); `trip_t` restarts per driver,
        # for the calibration windows that mean "early in this trip".
        t = time.monotonic() - self.t0
        trip_t = t - self.trip_started
        ok, frame = self._grab()
        if not ok:
            # A dropped frame is an absent face, never a closed eye.
            obs = _no_face(t)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.last_frame = frame
            obs = self.landmarker.observe(rgb, t)
            if self.eye_model is not None and obs.face_present and obs.landmarks:
                from awas.driver.eyestate import combine
                eyes = self.eye_model.per_eye(rgb, obs.landmarks)
                obs.p_closed = combine(eyes)
                obs.p_closed_eyes = tuple(None if e is None else round(e[0], 3) for e in eyes)
        self.last_obs = obs
        state = self.pipeline.process(obs, trip_t)

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
        self._learn_sdlp_baseline(lane_stats.sdlp, trip_t, state.drowsiness_level)
        baseline = self.sdlp_baseline.baseline_m if self.sdlp_baseline else None
        self.last_lane_stats = lane_stats
        self.last_fused = fuse_channels(state, lane_stats, sdlp_baseline_m=baseline,
                                        mode=self.fusion_mode)

        # A road source that knows the engine (a simulator, later OBD-II) wins
        # over the demo's manual ignition key.
        ignition = getattr(self.road_source, "ignition_on", None)
        return build_telemetry(
            driver=state, road=road, lane=lane_stats, workload=workload,
            ignition_on=self.ignition_on if ignition is None else ignition,
            fused=self.last_fused,
        )

    def set_driver(self, driver_name: str, profile_dir: Optional[Path], eye_signal: Optional[str],
                   neutral_pose: Optional[Tuple[float, float]] = None,
                   cnn_scale: Optional[Tuple[float, float]] = None,
                   signal_confidence: Optional[float] = None) -> None:
        """Start a new trip for another driver, without reopening the camera.

        Saves what the outgoing driver's trip learned first. `profile_dir` None
        means a placeholder (the setup screens) whose observations are never
        persisted — they belong to whoever happens to be in front of the camera.
        """
        self.pipeline.save_profile()
        signal = eye_signal if (eye_signal and self.eye_model is not None) else "ear"
        self.pipeline = DriverPipeline(driver_name=driver_name, profile_dir=profile_dir,
                                       eye_signal=signal, neutral_pose=neutral_pose,
                                       cnn_scale=cnn_scale, signal_confidence=signal_confidence)
        self.sdlp_cal = SdlpCalibrator(driver_name)
        self.sdlp_baseline, self._sdlp_offered_at = None, None
        self.lane.reset()
        self.trip_started = time.monotonic() - self.t0

    def _learn_sdlp_baseline(self, sdlp, t: float, level: int) -> None:
        if sdlp is None:
            return
        if self._sdlp_offered_at is not None and t - self._sdlp_offered_at < self.sdlp_offer_every_sec:
            return
        self._sdlp_offered_at = t
        # The calibrator itself refuses values outside the early-trip window,
        # while any drowsiness is showing, or outside a plausible range.
        if self.sdlp_cal.observe(sdlp, trip_elapsed_sec=t, drowsiness_level=level):
            self.sdlp_baseline = self.sdlp_cal.build() or self.sdlp_baseline

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
