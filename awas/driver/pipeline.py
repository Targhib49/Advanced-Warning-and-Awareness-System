"""Layer 1 end to end: face observation in, driver state out (plan §4.1).

    FaceObservation -> calibration -> personal openness -> window -> level

Owns no camera and no model. Everything here is deterministic given a sequence
of observations and their timestamps, which is what makes the whole perception
path testable without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from awas.driver.calibration import (
    CalibrationConfig,
    DriverProfile,
    EyeOpennessCalibrator,
    OpennessNormalizer,
    SampleContext,
)
from awas.driver.landmarks import FaceObservation
from awas.driver.window import (
    FeatureWindow,
    Hysteresis,
    LevelConfig,
    Proposal,
    WindowConfig,
    WindowFeatures,
    propose_level,
)


@dataclass
class DriverState:
    """What Layer 1 hands to the fusion core."""

    t: float
    drowsiness_level: int = 0
    confidence: float = 0.0

    face_present: bool = False
    eye_aspect_ratio: float = 0.0
    openness: float = 0.0
    head_pitch_deg: float = 0.0
    head_yaw_deg: float = 0.0

    perclos: float = 0.0
    microsleep_count: int = 0
    blink_duration_mean_ms: float = 0.0
    blink_rate_per_min: float = 0.0
    longest_closure_ms: float = 0.0

    calibrating: bool = True
    calibration_progress: float = 0.0
    personalized: bool = False
    reason: str = ""

    # Which eye signal drove `openness` this frame: "cnn" (T1) or "ear".
    # Both are carried so a session log can compare them on the same frames.
    eye_source: str = "ear"
    p_closed: Optional[float] = None
    ear_openness: float = 0.0
    cnn_openness: Optional[float] = None
    p_closed_eyes: Optional[tuple] = None    # (left, right), for the log
    iris_drop: Optional[tuple] = None        # (left, right), eye widths below the corner line
    # Whether the feature window holds enough observation to judge. The
    # pipeline sets it; a hand-built state is taken as usable, as before.
    window_usable: bool = True
    # Face found but turned away from road-facing (mirror, shoulder check):
    # the driver is present, their eyes are not measurable this frame.
    eyes_observable: bool = True

    # Carried through for the debug overlay only; nothing downstream reads it.
    landmarks: list = field(default_factory=list)
    landmarks_drawn: bool = False


class DriverPipeline:
    """Stateful per-trip driver-state estimator."""

    def __init__(
        self,
        driver_name: str = "default",
        profile_dir: Optional[Path] = None,
        cal_cfg: Optional[CalibrationConfig] = None,
        win_cfg: Optional[WindowConfig] = None,
        lvl_cfg: Optional[LevelConfig] = None,
        update_hz: float = 1.0,
        eye_signal: str = "cnn",
        cnn_confidence: float = 0.80,
        cnn_scale: Optional[tuple] = None,
        neutral_pose: Optional[tuple] = None,
        max_eye_yaw_deg: float = 25.0,
        max_eye_pitch_deg: float = 20.0,
    ):
        """`eye_signal`: "cnn" uses T1's P(closed) whenever a frame has one and
        falls back to EAR when it does not; "ear" is the v2 path, kept as the
        control condition; "both" averages the two openness readings, so one
        signal failing cannot hide a closure the other sees (bench drive 2:
        EAR read ten seconds of closed eyes while the CNN hovered near 0.6).

        `cnn_scale` is (P(closed) open, P(closed) closed) from this driver's
        guided calibration: their personal CNN scale, exactly as EAR has one.
        Without it the raw 1 - P(closed) stands in.

        `cnn_confidence` replaces the EAR calibration's confidence while the
        CNN is driving. It is UNFITTED: the model scored 98.6% on unseen MRL
        subjects, but MRL is infrared and a cabin camera in daylight is not, so
        it starts below that until it is measured on cabin footage.

        `neutral_pose` is (yaw, pitch) while this driver looks at the road,
        from guided calibration. A frame whose head pose is further from it
        than `max_eye_yaw_deg` / `max_eye_pitch_deg` (UNFITTED) is a mirror
        check, a shoulder check, a look down at the instruments: the driver is
        there, but their eyes cannot be measured — EAR foreshortens and the eye
        crop turns to profile. Such frames are left out of drowsiness exactly
        like frames with no face, rather than read as drooping lids.

        Without a neutral (no guided calibration, or one measured with the
        retired solvePnP pose) there is no reference to judge a turn against,
        and no frame is excluded for its pose — a wrong reference throws away
        exactly the frames that matter, which the first bench drive showed.
        """
        if eye_signal not in ("cnn", "ear", "both"):
            raise ValueError(f"eye_signal must be 'cnn', 'ear' or 'both', got {eye_signal!r}")
        self.eye_signal = eye_signal
        self.cnn_scale = cnn_scale
        self.cnn_confidence = cnn_confidence
        self.neutral_pose = neutral_pose
        self.neutral_yaw, self.neutral_pitch = neutral_pose or (0.0, 0.0)
        self.max_eye_yaw_deg = max_eye_yaw_deg
        self.max_eye_pitch_deg = max_eye_pitch_deg
        self.driver_name = driver_name
        self.profile_dir = Path(profile_dir) if profile_dir else None
        self.cal_cfg = cal_cfg or CalibrationConfig()
        self.lvl_cfg = lvl_cfg or LevelConfig()
        self.update_hz = update_hz

        stored = (
            DriverProfile.load(self.profile_dir, driver_name)
            if self.profile_dir else None
        )
        self.stored_profile = stored
        self.normalizer = OpennessNormalizer(stored, self.cal_cfg)

        self.calibrator = EyeOpennessCalibrator(driver_name, self.cal_cfg)
        self.window = FeatureWindow(win_cfg)
        self.hysteresis = Hysteresis(self.lvl_cfg)

        self._last_estimate_t: Optional[float] = None
        self._last_features = WindowFeatures()
        self._last_proposal = Proposal(0, "starting")
        self._rebuild_every = 150      # frames between profile rebuilds

    # ---------------- main entry ----------------

    def eyes_observable(self, obs: FaceObservation) -> bool:
        """Face found AND turned no further than the limits from road-facing."""
        if self.neutral_pose is None:
            return obs.face_present
        return (obs.face_present
                and abs(obs.yaw_deg - self.neutral_yaw) <= self.max_eye_yaw_deg
                and abs(obs.pitch_deg - self.neutral_pitch) <= self.max_eye_pitch_deg)

    def cnn_openness(self, p_closed: Optional[float]) -> Optional[float]:
        """1 = open, 0 = closed, on this driver's CNN scale when there is one."""
        if p_closed is None:
            return None
        if self.cnn_scale is not None:
            p_open, p_shut = self.cnn_scale
            if p_shut - p_open >= 0.2:       # a scale too narrow to divide by is no scale
                return float(min(max((p_shut - p_closed) / (p_shut - p_open), 0.0), 1.0))
        return 1.0 - p_closed

    def process(self, obs: FaceObservation, trip_elapsed_sec: float) -> DriverState:
        observable = self.eyes_observable(obs)
        if observable and obs.ear > 0.0:
            self._calibrate(obs, trip_elapsed_sec)

        # EAR is always computed and its calibration always runs: it is the
        # fallback when the CNN cannot crop an eye, and the control to compare
        # the CNN against.
        ear_openness = self.normalizer.openness(obs.ear) if obs.face_present else 0.0
        cnn_openness = self.cnn_openness(obs.p_closed) if obs.face_present else None
        if self.eye_signal == "cnn" and cnn_openness is not None:
            openness, source = cnn_openness, "cnn"
        elif self.eye_signal == "both" and cnn_openness is not None:
            openness, source = 0.5 * (cnn_openness + ear_openness), "both"
        else:
            openness, source = ear_openness, "ear"
        use_cnn = source != "ear"
        # Unobservable frames count as "no face" for the window: neither open
        # nor closed, just not seen.
        self.window.add(obs.t, openness, observable)

        # The level ticks at update_hz, NOT per frame. Hysteresis counts these
        # ticks, so running it at frame rate would turn "3 consecutive updates"
        # into a tenth of a second and defeat the damping entirely.
        if self._due(obs.t):
            self._last_estimate_t = obs.t
            self._last_features = self.window.features()
            self._last_proposal = propose_level(self._last_features, self.lvl_cfg)
            self.hysteresis.update(self._last_proposal)

        f = self._last_features
        # The CNN is population-trained, so it needs no personal calibration
        # before it can be acted on; EAR does.
        calibrating = not self.normalizer.personalized and not use_cnn
        return DriverState(
            t=obs.t,
            drowsiness_level=self.hysteresis.level,
            confidence=self._confidence(f, obs, use_cnn),
            face_present=obs.face_present,
            eye_aspect_ratio=obs.ear,
            openness=openness,
            head_pitch_deg=obs.pitch_deg,
            head_yaw_deg=obs.yaw_deg,
            perclos=f.perclos,
            microsleep_count=f.microsleep_count,
            blink_duration_mean_ms=f.mean_blink_duration_ms,
            blink_rate_per_min=f.blink_rate_per_min,
            longest_closure_ms=f.longest_closure_ms,
            calibrating=calibrating,
            calibration_progress=self.calibration_progress,
            personalized=self.normalizer.personalized,
            reason=self._last_proposal.reason,
            landmarks=obs.landmarks,
            eye_source=source,
            cnn_openness=cnn_openness,
            p_closed_eyes=getattr(obs, "p_closed_eyes", None),
            iris_drop=getattr(obs, "iris_drop", None),
            p_closed=obs.p_closed,
            ear_openness=ear_openness,
            window_usable=f.usable,
            eyes_observable=observable,
        )

    # ---------------- calibration ----------------

    def _calibrate(self, obs: FaceObservation, trip_elapsed_sec: float) -> None:
        ctx = SampleContext(
            trip_elapsed_sec=trip_elapsed_sec,
            face_present=obs.face_present,
            # Relative to road-facing, so an off-axis mount does not make every
            # frame look like a turned head to the passive calibrator.
            head_pitch_deg=obs.pitch_deg - self.neutral_pitch,
            head_yaw_deg=obs.yaw_deg - self.neutral_yaw,
            # Feeding the current level back in is what keeps a drowsy stretch
            # out of the baseline (calibration.is_known_alert).
            drowsiness_level=self.hysteresis.level,
        )
        if not self.calibrator.observe(obs.ear, ctx):
            return
        if self.calibrator.n_admitted % self._rebuild_every == 0:
            self._refresh_profile()

    def _refresh_profile(self) -> None:
        fresh = self.calibrator.build()
        if fresh is None:
            return
        merged = self.stored_profile.merged_with(fresh, self.cal_cfg) if self.stored_profile else fresh
        self.normalizer = OpennessNormalizer(merged, self.cal_cfg)

    @property
    def calibration_progress(self) -> float:
        if self.normalizer.personalized:
            return 1.0
        return min(self.calibrator.n_admitted / max(self.cal_cfg.min_samples, 1), 1.0)

    def save_profile(self) -> Optional[Path]:
        """Persist what this trip learned. Call once at shutdown."""
        if self.profile_dir is None:
            return None
        fresh = self.calibrator.build()
        if fresh is None:
            return None
        merged = self.stored_profile.merged_with(fresh, self.cal_cfg) if self.stored_profile else fresh
        return merged.save(self.profile_dir)

    # ---------------- helpers ----------------

    def _due(self, t: float) -> bool:
        if self._last_estimate_t is None:
            return True
        return (t - self._last_estimate_t) >= (1.0 / self.update_hz)

    def _confidence(self, f: WindowFeatures, obs: FaceObservation, use_cnn: bool = False) -> float:
        """How much the state machine should trust this level.

        Three things pull it down, and each one is a real reason to be quieter:
        an uncalibrated driver (EAR only — the CNN needs no calibration), a
        window with too little observation in it, and a face we are only
        intermittently seeing.
        """
        c = self.cnn_confidence if use_cnn else self.normalizer.confidence
        if not f.usable:
            c *= 0.5
        c *= max(f.face_present_ratio, 0.25) if f.n_samples > 1 else 0.5
        if not self.eyes_observable(obs):
            c *= 0.7
        return float(min(max(c, 0.0), 1.0))
