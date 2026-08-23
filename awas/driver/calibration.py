"""Per-driver eye calibration — personalization without machine learning.

The single largest error source in threshold-based drowsiness detection is that
baseline eye aperture varies two- to threefold across people. A universal
`EAR < 0.20` rule is therefore simultaneously too twitchy for one driver and
too deaf for the next, and no amount of tuning fixes both.

The fix is arithmetic, not a model. Learn two reference points per driver while
they are known to be alert, then measure every later frame against *their* scale:

    openness = (ear - closed_ref) / (open_ref - closed_ref)     clamped to [0, 1]

This is worth doing carefully because it lines the measure up with PERCLOS's own
definition. PERCLOS P80 asks what fraction of a window the eye was more than 80%
closed — that is literally `openness < 0.20` on this scale, whereas "EAR below
some constant" only approximates it and does so differently for every face.

Two reference points rather than one, because EAR does not reach zero on a
closed eye: eyelid thickness and landmark noise leave a floor, and that floor is
also personal. Calibrating only the open end would leave the bottom of every
driver's range wrong by a different amount.

Plan v3 §3 tier T3 — the no-ML portion, buildable in Phase 1.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

# --- Population fallbacks, used only until a driver has their own profile. ---
# Measured on mediapipe 0.10.35, ONE subject, live camera. That is n=1 and it
# needs widening, but it beats the dlib-derived guesses these replace: the old
# closed value of 0.08 was 4.5x too high, because a MediaPipe mesh drives EAR
# far closer to zero on a shut eye than a 68-point dlib fit does. The effect
# was a cold-start false-alarm band around EAR 0.08-0.12 — the system calling
# eyes shut while they were still a third open, in exactly the window before
# calibration completes and a real driver has no personal scale yet.
#
# Still replace these with a proper multi-subject measurement before reporting
# anything (plan §4.1: thresholds are fitted, not asserted). Note also that
# closed_ref sits at the 2nd percentile, so it can absorb partial landmark
# failures as if they were closures — worth re-checking with more subjects.
POPULATION_OPEN_EAR = 0.28
POPULATION_CLOSED_EAR = 0.02

PROFILE_VERSION = 1


@dataclass(frozen=True)
class CalibrationConfig:
    # How long after trip start we still treat the driver as presumed alert.
    calibration_window_sec: float = 600.0

    # Head pose beyond this foreshortens EAR enough to corrupt the baseline.
    max_pitch_deg: float = 20.0
    max_yaw_deg: float = 25.0

    # Quantiles over admitted samples. The median is dominated by open frames
    # (an alert person's eyes are open the large majority of the time); the low
    # quantile sits inside the blink minima.
    open_quantile: float = 0.50
    closed_quantile: float = 0.02

    # Sample counts, in frames.
    min_samples: int = 900          # ~30 s at 30 fps: enough to contain blinks
    full_confidence_samples: int = 9000   # ~5 min at 30 fps
    ring_capacity: int = 36000            # ~20 min at 30 fps

    # A profile whose endpoints are this close together means the driver barely
    # blinked, or the landmarks are not tracking the lids. Refuse it rather than
    # dividing by a near-zero range.
    min_dynamic_range: float = 0.04

    # Confidence floor while running on population defaults.
    cold_start_confidence: float = 0.35

    # Re-calibration trigger: current open median this far from the stored one
    # (as a fraction of the stored value) means something changed — glasses on,
    # camera knocked, a different person in the seat.
    drift_fraction: float = 0.25

    # Caps how much history can outvote a fresh session when profiles merge.
    max_effective_samples: int = 45000


@dataclass
class SampleContext:
    """What the calibrator needs to know to decide if a frame is admissible."""

    trip_elapsed_sec: float
    face_present: bool = True
    head_pitch_deg: float = 0.0
    head_yaw_deg: float = 0.0
    drowsiness_level: int = 0


def is_known_alert(ctx: SampleContext, cfg: CalibrationConfig) -> bool:
    """Whether this frame may contribute to the baseline.

    Conservative on purpose. A baseline polluted with drowsy frames drags
    `open_ref` down, which raises the apparent openness of a genuinely drooping
    eye — the failure is silent and it makes the system less sensitive exactly
    when it needs to be more so.

    Phase 1 has no road-facing camera, so lane-position variance is not
    available as corroboration (plan v3 §4.2). What is available: early in the
    trip, face tracked, head roughly forward, nothing already suggesting
    drowsiness.
    """
    if not ctx.face_present:
        return False
    if ctx.drowsiness_level > 0:
        return False
    if ctx.trip_elapsed_sec > cfg.calibration_window_sec:
        return False
    if abs(ctx.head_pitch_deg) > cfg.max_pitch_deg:
        return False
    if abs(ctx.head_yaw_deg) > cfg.max_yaw_deg:
        return False
    return True


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile over an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("quantile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


@dataclass
class DriverProfile:
    """One driver's personal eye scale, persisted between trips."""

    name: str
    open_ref: float
    closed_ref: float
    n_samples: int
    updated_at: float = field(default_factory=time.time)
    version: int = PROFILE_VERSION

    @property
    def dynamic_range(self) -> float:
        return self.open_ref - self.closed_ref

    def is_usable(self, cfg: CalibrationConfig) -> bool:
        return (
            self.n_samples >= cfg.min_samples
            and self.dynamic_range >= cfg.min_dynamic_range
        )

    def confidence(self, cfg: CalibrationConfig) -> float:
        """How much the fusion core should trust levels derived from this.

        Feeds `Telemetry.confidence`, which the state machine already uses to
        downgrade an under-evidenced level rather than acting on it at full
        strength (plan §5.2). A driver the system has just met gets a quieter
        system, by construction.
        """
        if not self.is_usable(cfg):
            return cfg.cold_start_confidence
        span = max(cfg.full_confidence_samples - cfg.min_samples, 1)
        maturity = (self.n_samples - cfg.min_samples) / span
        ramped = cfg.cold_start_confidence + (1.0 - cfg.cold_start_confidence) * maturity
        return float(min(max(ramped, cfg.cold_start_confidence), 1.0))

    def merged_with(self, other: "DriverProfile", cfg: CalibrationConfig) -> "DriverProfile":
        """Combine a stored profile with a fresh session, weighted by samples.

        History is capped at `max_effective_samples` so a long-established
        profile still moves when something real changes, instead of being
        frozen by sheer accumulated weight.
        """
        w_old = min(self.n_samples, cfg.max_effective_samples)
        w_new = other.n_samples
        total = w_old + w_new
        if total == 0:
            return self
        return DriverProfile(
            name=self.name,
            open_ref=(self.open_ref * w_old + other.open_ref * w_new) / total,
            closed_ref=(self.closed_ref * w_old + other.closed_ref * w_new) / total,
            n_samples=self.n_samples + other.n_samples,
            updated_at=time.time(),
        )

    def has_drifted(self, current_open: float, cfg: CalibrationConfig) -> bool:
        """True when the live open median has moved away from the stored one."""
        if self.open_ref <= 0:
            return False
        return abs(current_open - self.open_ref) / self.open_ref > cfg.drift_fraction

    # -------- persistence --------

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, directory: Path, name: str) -> Optional["DriverProfile"]:
        path = Path(directory) / f"{name}.json"
        if not path.exists():
            return None
        blob = json.loads(path.read_text())
        if blob.get("version") != PROFILE_VERSION:
            # A stale profile is worse than none: the scale it encodes may have
            # been produced by a different landmark model entirely.
            return None
        blob.pop("version", None)
        return cls(**blob, version=PROFILE_VERSION)


class EyeOpennessCalibrator:
    """Accumulates EAR during known-alert frames and derives a profile."""

    def __init__(self, name: str, cfg: Optional[CalibrationConfig] = None):
        self.name = name
        self.cfg = cfg or CalibrationConfig()
        self._ring: Deque[float] = deque(maxlen=self.cfg.ring_capacity)
        self._rejected = 0

    def observe(self, ear: float, ctx: SampleContext) -> bool:
        """Offer one frame. Returns whether it was admitted."""
        # 0.0 means the landmark fit failed. Admitting it would drag closed_ref
        # to zero and silently widen every driver's range.
        if ear <= 0.0 or not is_known_alert(ctx, self.cfg):
            self._rejected += 1
            return False
        self._ring.append(ear)
        return True

    @property
    def n_admitted(self) -> int:
        return len(self._ring)

    @property
    def n_rejected(self) -> int:
        return self._rejected

    def current_open_median(self) -> Optional[float]:
        if not self._ring:
            return None
        return quantile(sorted(self._ring), self.cfg.open_quantile)

    def build(self) -> Optional[DriverProfile]:
        """Derive a profile from what has been admitted, or None if not ready."""
        if len(self._ring) < self.cfg.min_samples:
            return None
        values = sorted(self._ring)
        profile = DriverProfile(
            name=self.name,
            open_ref=quantile(values, self.cfg.open_quantile),
            closed_ref=quantile(values, self.cfg.closed_quantile),
            n_samples=len(values),
        )
        if profile.dynamic_range < self.cfg.min_dynamic_range:
            # Too little separation to divide by. Usually means the driver
            # barely blinked in the window, or the lids are not being tracked.
            return None
        return profile

    def reset(self) -> None:
        self._ring.clear()
        self._rejected = 0


class OpennessNormalizer:
    """Maps raw EAR onto a driver's personal 0-1 openness scale."""

    def __init__(
        self,
        profile: Optional[DriverProfile] = None,
        cfg: Optional[CalibrationConfig] = None,
    ):
        self.cfg = cfg or CalibrationConfig()
        self.profile = profile if (profile and profile.is_usable(self.cfg)) else None
        self.personalized = self.profile is not None

    @property
    def open_ref(self) -> float:
        return self.profile.open_ref if self.profile else POPULATION_OPEN_EAR

    @property
    def closed_ref(self) -> float:
        return self.profile.closed_ref if self.profile else POPULATION_CLOSED_EAR

    @property
    def confidence(self) -> float:
        if self.profile is None:
            return self.cfg.cold_start_confidence
        return self.profile.confidence(self.cfg)

    def openness(self, ear: float) -> float:
        """0.0 = fully closed on this driver's scale, 1.0 = fully open.

        A failed landmark fit (`ear <= 0`) returns 0.0, so callers must gate on
        `face_present` rather than reading a dropped frame as a closed eye.
        """
        if ear <= 0.0:
            return 0.0
        span = self.open_ref - self.closed_ref
        if span <= 1e-9:
            return 0.0
        return float(min(max((ear - self.closed_ref) / span, 0.0), 1.0))

    def is_closed(self, ear: float, perclos_p: float = 0.80) -> bool:
        """PERCLOS-style closure test.

        P80 — the standard — means "more than 80% closed", i.e. openness below
        0.20. Because openness is already normalized per driver, this threshold
        is the same number for everyone, which is the entire point.
        """
        return self.openness(ear) < (1.0 - perclos_p)


def explain(ear: float, normalizer: OpennessNormalizer) -> Dict[str, object]:
    """Side-by-side of the population verdict and the personal one.

    This is the demo panel: the same frame that a fixed threshold calls fine,
    a personal scale can call severe, and vice versa.
    """
    population = OpennessNormalizer(profile=None, cfg=normalizer.cfg)
    return {
        "ear": round(ear, 4),
        "population": {
            "open_ref": population.open_ref,
            "openness": round(population.openness(ear), 3),
            "closed_p80": population.is_closed(ear),
        },
        "personal": {
            "open_ref": round(normalizer.open_ref, 4),
            "closed_ref": round(normalizer.closed_ref, 4),
            "openness": round(normalizer.openness(ear), 3),
            "closed_p80": normalizer.is_closed(ear),
            "personalized": normalizer.personalized,
            "confidence": round(normalizer.confidence, 3),
        },
    }
