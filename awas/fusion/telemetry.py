"""Assemble the snapshot the state machine reads (plan §4.3 -> §5).

A pure function on purpose: given a driver state, a road state and the derived
lane and workload figures, there is exactly one Telemetry it produces. That
makes the whole perception-to-decision join testable without a camera, a road,
or a clock.
"""

from __future__ import annotations

from typing import Optional

from awas.core.telemetry import Distraction, Telemetry
from awas.driver.pipeline import DriverState
from awas.fusion.core import FusedEstimate, facial_channel, fuse, vehicle_channel
from awas.road.signals import LaneStats, RoadState
from awas.road.workload import Workload


def fuse_channels(
    driver: DriverState,
    lane: Optional[LaneStats] = None,
    sdlp_baseline_m: Optional[float] = None,
    mode: str = "weighted",
) -> FusedEstimate:
    """Combine the facial and vehicle channels into one estimate.

    `mode` is a deployment decision, not a tuning knob (results §D-3): the
    weighted rule suppresses a facial false alarm the lane contradicts, the
    asymmetric rule keeps an early warning either channel raises. No single
    operator does both.
    """
    channels = [
        facial_channel(
            level=driver.drowsiness_level,
            confidence=driver.confidence,
            face_present=driver.face_present,
            window_usable=getattr(driver, "window_usable", True),
        )
    ]
    if lane is not None:
        channels.append(vehicle_channel(
            sdlp=lane.sdlp, departures=lane.departures,
            samples=lane.samples, sdlp_baseline_m=sdlp_baseline_m,
        ))
    return fuse(channels, mode=mode)


def build_telemetry(
    driver: DriverState,
    road: Optional[RoadState] = None,
    lane: Optional[LaneStats] = None,
    workload: Optional[Workload] = None,
    ignition_on: bool = True,
    fused: Optional[FusedEstimate] = None,
) -> Telemetry:
    """One snapshot, assembled from every channel that had something to say.

    Where the road channel is absent the vehicle fields stay `None` rather than
    zero, because the state machine already reads `None` as "not fitted" and
    zero as "measured zero" — and confusing the two would make a car with no
    OBD-II look like a car that is stationary with the handbrake off.
    """
    fused = fused or fuse_channels(driver, lane)
    road = road or RoadState()
    workload_value = workload.value if workload is not None else 0.0

    return Telemetry(
        timestamp=driver.t,

        # fusion output
        drowsiness_level=fused.level,
        confidence=fused.confidence,
        workload=workload_value,
        distraction=Distraction.NONE,      # Phase 2: gaze and phone channels

        # facial channel, for diagnostics and the log
        perclos=driver.perclos,
        microsleep_count=driver.microsleep_count,
        eyes_closed_ms=getattr(driver, "current_closure_ms", 0.0),
        blink_duration_mean_ms=driver.blink_duration_mean_ms,
        face_present=driver.face_present,
        eye_aspect_ratio=driver.eye_aspect_ratio,
        head_pitch_deg=driver.head_pitch_deg,
        head_yaw_deg=driver.head_yaw_deg,

        # vehicle channel
        speed_kmh=road.speed_kmh,
        speed_source=road.speed_source,
        parking_brake=None,                # no OBD-II (plan §7.4)
        ignition_on=ignition_on,
        sdlp=lane.sdlp if lane else None,
        steering_reversal_rate=lane.steering_reversal_rate if lane else None,
        lane_departure=road.lane_departure,
        reversing=bool(road.reversing),
        turn_signal=road.turn_signal,

        # road context
        traffic_light_color=road.traffic_light_color,
        vehicle_ahead_distance_m=road.vehicle_ahead_distance_m,
        road_class=road.road_class,
    )
