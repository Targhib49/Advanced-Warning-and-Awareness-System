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
    # Whether the feature window holds enough observation to judge. The
    # pipeline sets it; a hand-built state is taken as usable, as before.
    window_usable: bool = True

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
    ):
        """`eye_signal`: "cnn" uses T1's P(closed) whenever a frame has one and
        falls back to EAR when it does not; "ear" is the v2 path, kept as the
        control condition.

        `cnn_confidence` replaces the EAR calibration's confidence while the
        CNN is driving. It is UNFITTED: the model scored 98.6% on unseen MRL
        subjects, but MRL is infrared and a cabin camera in daylight is not, so
        it starts below that until it is measured on cabin footage.
        """
        if eye_signal not in ("cnn", "ear"):
            raise ValueError(f"eye_signal must be 'cnn' or 'ear', got {eye_signal!r}")
        self.eye_signal = eye_signal
        self.cnn_confidence = cnn_confidence
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

    def process(self, obs: FaceObservation, trip_elapsed_sec: float) -> DriverState:
        if obs.face_present and obs.ear > 0.0:
            self._calibrate(obs, trip_elapsed_sec)

        # EAR is always computed and its calibration always runs: it is the
        # fallback when the CNN cannot crop an eye, and the control to compare
        # the CNN against.
        ear_openness = self.normalizer.openness(obs.ear) if obs.face_present else 0.0
        use_cnn = (self.eye_signal == "cnn" and obs.face_present
                   and obs.p_closed is not None)
        openness = (1.0 - obs.p_closed) if use_cnn else ear_openness
        self.window.add(obs.t, openness, obs.face_present)

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
            eye_source="cnn" if use_cnn else "ear",
            p_closed=obs.p_closed,
            ear_openness=ear_openness,
            window_usable=f.usable,
        )

    # ---------------- calibration ----------------

    def _calibrate(self, obs: FaceObservation, trip_elapsed_sec: float) -> None:
        ctx = SampleContext(
            trip_elapsed_sec=trip_elapsed_sec,
            face_present=obs.face_present,
            head_pitch_deg=obs.pitch_deg,
            head_yaw_deg=obs.yaw_deg,
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
        if not obs.face_present:
            c *= 0.7
        return float(min(max(c, 0.0), 1.0))
