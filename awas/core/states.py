"""Operational driving states."""

from enum import Enum


class DrivingState(Enum):
    """High-level operational state of the vehicle/system."""

    SHUTDOWN = "shutdown"
    PARKED = "parked"
    STOPPED_AT_LIGHT = "stopped_at_light"
    TRAFFIC_JAM = "traffic_jam"
    MOVING = "moving"
