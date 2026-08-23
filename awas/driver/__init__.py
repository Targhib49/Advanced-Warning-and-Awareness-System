"""Layer 1 — driver state monitoring (plan §4.1).

Camera -> face mesh -> per-frame features (EAR, MAR, head pose, iris offset)
-> 60 s sliding window (PERCLOS, microsleeps, blink duration, yawn rate,
head nods, gaze-off-road ratio).

Produces evidence for the fusion core. Does not produce the drowsiness level
itself — that is `awas.fusion`'s job, because the level depends on vehicle
evidence too.
"""

from awas.driver.calibration import (  # noqa: E402,F401
    CalibrationConfig,
    DriverProfile,
    EyeOpennessCalibrator,
    OpennessNormalizer,
    SampleContext,
    is_known_alert,
)
from awas.driver.ear import (  # noqa: E402,F401
    ear_from_landmarks,
    eye_aspect_ratio,
    mean_ear,
)

__all__ = [
    "CalibrationConfig",
    "DriverProfile",
    "EyeOpennessCalibrator",
    "OpennessNormalizer",
    "SampleContext",
    "is_known_alert",
    "ear_from_landmarks",
    "eye_aspect_ratio",
    "mean_ear",
]
