"""Show what the vehicle channel and workload gating actually change.

    python -m scripts.demo_fusion

Same facial evidence throughout. What differs is whether the road channel is
listening, and what the road is doing.
"""

from __future__ import annotations

from awas.driver.pipeline import DriverState
from awas.fusion.telemetry import build_telemetry, fuse_channels
from awas.road.signals import LaneTracker, RoadState
from awas.road.workload import WorkloadEstimator


def lane_after(jitter: float, seed: int = 1, n: int = 600):
    import random
    rng = random.Random(seed)
    tr, t = LaneTracker(), 0.0
    for _ in range(n):
        tr.add(RoadState(t=t, speed_kmh=90.0, lane_offset_m=rng.gauss(0, jitter)))
        t += 0.1
    return tr.stats()


def driver(level: int, conf: float = 0.85, face: bool = True) -> DriverState:
    return DriverState(t=100.0, drowsiness_level=level, confidence=conf,
                       face_present=face, perclos=0.05 * level)


def main() -> None:
    tidy = lane_after(0.05)
    wander = lane_after(0.35)

    print("H2 — does the vehicle channel change the verdict?\n")
    print(f"  {'situation':44} {'fused':>26}")
    rows = [
        ("face drowsy, lane wandering  (agree)",  driver(3), wander),
        ("face drowsy, lane tight      (clash)",  driver(3), tidy),
        ("face alert,  lane wandering  (clash)",  driver(0), wander),
        ("face occluded, lane wandering",         driver(0, 0.0, False), wander),
        ("face drowsy, no road channel",          driver(3), None),
    ]
    for label, d, lane in rows:
        print(f"  {label:44} {fuse_channels(d, lane).explain()}")

    print(f"\n  SDLP: tidy {tidy.sdlp:.3f} m, wandering {wander.sdlp:.3f} m\n")

    print("H3 — when does workload silence a check-in?")
    print("  (a level-1 intervention needs workload below 0.40)\n")
    situations = [
        ("stopped at a red light",      [0.0] * 20, {}),
        ("quiet highway cruise",        [90.0] * 20, {"road_class": "highway"}),
        ("urban stop-go",               [5, 40, 10, 35, 0, 30] * 4, {"road_class": "urban"}),
        ("tailgating on the motorway",  [90.0] * 20,
         {"road_class": "highway", "vehicle_ahead_distance_m": 5.0}),
        ("just drifted out of lane",    [90.0] * 20,
         {"road_class": "highway", "lane_departure": True}),
    ]
    for label, speeds, kw in situations:
        est, w, t = WorkloadEstimator(), None, 0.0
        for s in speeds:
            w = est.update(RoadState(t=t, speed_kmh=s, **kw))
            t += 1.0
        gate = "speaks" if w.value < 0.40 else "STAYS QUIET"
        print(f"  {label:30} {w.value:.2f}  {gate:11}  {w.explain().split('— ')[-1]}")

    print("\nUntil now workload was hardcoded to 0.00, so every one of these")
    print("said 'speaks'. The gating existed in the state machine and in the")
    print("plan, and had never once run.")


if __name__ == "__main__":
    main()
