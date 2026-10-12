"""How busy is the driving task right now? (plan §4.2)

This is the number that decides when AWAS shuts up. The state machine already
gates every conversational intervention on it — a level-1 check-in needs
workload below 0.4, level 2 below 0.7 — but until now nothing computed it and
the field was hardcoded to zero, so every gate passed. The "knows when to be
quiet" property existed only on paper.

Deliberately transparent rather than learned. Each contribution is named and
reported, so the debug overlay and the session log can say *why* the system
went quiet — which matters both for debugging and for defending the design.
Hypothesis H3 tests whether this gating helps at all; a black box would make
that test uninterpretable.

Every threshold here is a starting point to be fitted (plan §4.1), not a value
anyone measured.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from awas.road.signals import RoadState

# Road classes differ in how much attention they demand moment to moment.
# Urban driving is event-dense; a highway is fast but monotonous, which is
# precisely why it is where drivers fall asleep.
ROAD_DEMAND: Dict[str, float] = {
    "urban": 0.55,
    "arterial": 0.35,
    "highway": 0.20,
}
DEFAULT_ROAD_DEMAND = 0.30


@dataclass(frozen=True)
class WorkloadConfig:
    window_sec: float = 20.0
    capacity: int = 2000

    # Speed contribution saturates: past this, more speed does not mean
    # proportionally more moment-to-moment demand.
    speed_saturate_kmh: float = 100.0
    speed_weight: float = 0.30

    # Stop-go traffic is far more demanding than steady motion at the same
    # average speed, so variability is weighted more heavily than speed itself.
    speed_variability_weight: float = 0.35
    variability_saturate_kmh: float = 18.0

    # Close following demands continuous attention.
    following_close_m: float = 12.0
    following_weight: float = 0.25

    road_class_weight: float = 0.30

    # A recent departure means the driver is already struggling; do not talk
    # over that.
    departure_spike: float = 0.45
    departure_decay_sec: float = 20.0

    # Manoeuvring at very low speed in traffic is busy; being stationary is not.
    stopped_kmh: float = 3.0


@dataclass
class Workload:
    value: float = 0.0
    parts: Dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        if not self.parts:
            return f"{self.value:.2f} (no road signals)"
        top = sorted(self.parts.items(), key=lambda kv: -kv[1])
        return f"{self.value:.2f} — " + ", ".join(
            f"{k} {v:+.2f}" for k, v in top if abs(v) > 0.01
        )


class WorkloadEstimator:
    """Rolling estimate of driving-task demand in [0, 1]."""

    def __init__(self, cfg: Optional[WorkloadConfig] = None):
        self.cfg = cfg or WorkloadConfig()
        self._speeds: Deque[tuple] = deque(maxlen=self.cfg.capacity)
        self._last_departure_t: Optional[float] = None

    def update(self, state: RoadState) -> Workload:
        cfg = self.cfg
        self._speeds.append((state.t, state.speed_kmh))
        while self._speeds and self._speeds[0][0] < state.t - cfg.window_sec:
            self._speeds.popleft()
        if state.lane_departure:
            self._last_departure_t = state.t

        # A stationary vehicle imposes no driving demand at all. This is what
        # makes the system willing to talk at a red light and unwilling to
        # talk while overtaking.
        if state.speed_kmh < cfg.stopped_kmh:
            return Workload(value=0.0, parts={"stopped": 0.0})

        parts: Dict[str, float] = {}

        speed_norm = min(state.speed_kmh / cfg.speed_saturate_kmh, 1.0)
        parts["speed"] = speed_norm * cfg.speed_weight

        variability = self._speed_variability()
        if variability is not None:
            v = min(variability / cfg.variability_saturate_kmh, 1.0)
            parts["stop-go"] = v * cfg.speed_variability_weight

        if state.vehicle_ahead_distance_m is not None:
            gap = state.vehicle_ahead_distance_m
            closeness = max(0.0, 1.0 - gap / max(cfg.following_close_m, 1e-6))
            if closeness > 0:
                parts["following"] = closeness * cfg.following_weight

        demand = ROAD_DEMAND.get(state.road_class or "", DEFAULT_ROAD_DEMAND)
        parts["road"] = demand * cfg.road_class_weight

        if self._last_departure_t is not None:
            age = state.t - self._last_departure_t
            if 0 <= age < cfg.departure_decay_sec:
                decay = 1.0 - age / cfg.departure_decay_sec
                parts["recent departure"] = cfg.departure_spike * decay

        total = min(sum(parts.values()), 1.0)
        return Workload(value=total, parts=parts)

    def _speed_variability(self) -> Optional[float]:
        if len(self._speeds) < 5:
            return None
        import statistics
        return statistics.pstdev([s for _, s in self._speeds])

    def reset(self) -> None:
        self._speeds.clear()
        self._last_departure_t = None
