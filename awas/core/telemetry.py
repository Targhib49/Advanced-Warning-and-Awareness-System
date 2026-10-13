"""Sensor snapshot — the single source of truth for current world state.

Sensors fill this in; the fusion core derives `drowsiness_level`, `confidence`,
and `workload` from it; the state machine reads all of it and decides.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Distraction(Enum):
    """Distraction is a separate channel from drowsiness (plan §4.1).

    They have different dynamics and demand different responses, so they must
    not be collapsed into one scalar.
    """

    NONE = "none"
    GAZE_OFF_ROAD = "gaze_off_road"
    PHONE = "phone"
    PASSENGER = "passenger"


@dataclass
class Telemetry:
    """A snapshot of everything known at one point in time."""

    # --- timing ---
    timestamp: float = 0.0

    # --- fusion core output (plan §4.3) ---
    drowsiness_level: int = 0        # 0=alert 1=mild 2=moderate 3=severe 4=critical
    confidence: float = 1.0          # 0-1, how much to trust the level
    workload: float = 0.0            # 0-1, how busy the driving task is
    distraction: Distraction = Distraction.NONE

    # --- Layer 1: driver state, windowed features (diagnostic + fusion input) ---
    perclos: float = 0.0             # fraction of window with eyes >80% closed
    microsleep_count: int = 0        # closures >=500ms in window
    eyes_closed_ms: float = 0.0      # the closure in progress now, 0 if open
    blink_duration_mean_ms: float = 0.0
    yawn_rate_per_min: float = 0.0
    head_nod_count: int = 0
    gaze_off_road_ratio: float = 0.0
    face_present: bool = True        # False => occluded/absent, drop confidence

    # --- Layer 1: per-frame diagnostics ---
    eye_aspect_ratio: float = 0.3
    mouth_aspect_ratio: float = 0.0
    head_pitch_deg: float = 0.0
    head_yaw_deg: float = 0.0

    # --- Layer 2: vehicle state ---
    speed_kmh: float = 0.0
    speed_source: str = "gps"        # "obd" | "gps" | "sim" | "mock"
    parking_brake: Optional[bool] = None   # None => OBD-II unavailable (plan §7.4)
    ignition_on: bool = False
    reversing: bool = False          # only a source that knows the gear can say so
    turn_signal: Optional[str] = None  # "left" | "right" | None (or unknown)

    # --- Layer 2: vehicle behaviour, drowsiness evidence (plan §4.3) ---
    sdlp: Optional[float] = None            # std-dev of lane position, metres
    steering_reversal_rate: Optional[float] = None  # reversals per minute
    lane_departure: bool = False

    # --- Layer 2: road context ---
    traffic_light_color: Optional[str] = None      # "red" | "yellow" | "green" | None
    vehicle_ahead_distance_m: Optional[float] = None
    road_class: Optional[str] = None               # "highway" | "arterial" | "urban"

    # --- GPS ---
    latitude: float = 0.0
    longitude: float = 0.0
    gps_fix: bool = False
