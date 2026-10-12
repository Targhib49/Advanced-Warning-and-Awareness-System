"""Parameter sensitivity and ablation analysis for the paper's Section IV.

    python -m scripts.parameter_sweep --out artifacts/sweeps.json

WHAT THIS IS, AND WHAT IT IS NOT.

Every number here comes from SIMULATED drowsiness episodes with a known onset
time. That is what makes detection latency measurable at all: on real footage
nobody knows the instant a driver became drowsy, which is precisely why the
field falls back on subjective scales.

So these results characterise *the system's response to its own parameters* —
how sensitive it is, where the operating point sits, what each knob costs. They
are a sensitivity analysis, not a performance claim. Detection accuracy on real
faces is the MRL result; intervention efficacy on real people is the pilot,
which has not run. Presenting these as either would be dishonest, and the
paper says so.

The episode model:
    0 - 300 s     alert, ordinary blinking
    300 - 420 s   onset ramp, eyelids progressively heavier
    420 - 600 s   drowsy, long closures and microsleeps
Ground truth onset is taken at the midpoint of the ramp.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from awas.driver.calibration import CalibrationConfig
from awas.driver.landmarks import FaceObservation
from awas.driver.pipeline import DriverPipeline
from awas.driver.window import LevelConfig, WindowConfig
from awas.fusion.core import facial_channel, fuse, vehicle_channel
from awas.road.signals import LaneTracker, RoadState

FPS = 15.0
ALERT_END = 300.0
RAMP_END = 420.0
TOTAL = 600.0
TRUE_ONSET = ALERT_END        # impairment begins when the ramp does

OPEN_EAR, SHUT_EAR = 0.30, 0.06


def droop_at(t: float) -> float:
    """Fraction of the driver's range their eyelids have lost, over the trip."""
    if t < ALERT_END:
        return 0.0
    if t < RAMP_END:
        return 0.9 * (t - ALERT_END) / (RAMP_END - ALERT_END)
    return 0.9


def episode(seed: int = 0, kind: str = "microsleep") -> List[Tuple[float, float]]:
    """(timestamp, EAR) for one simulated trip.

    Two presentations, because drowsiness does not have one shape and the
    system reaches for a different rule in each:

      ``microsleep``   blinks lengthen until closures pass 500 ms. The
                       microsleep rule fires and everything else is moot.
      ``heavy_lids``   the eyelids sit lower but blinks stay brief. No
                       microsleeps at all, so PERCLOS is the only thing that
                       can notice.

    Both are clinically real. Separating them is what makes the parameter
    sensitivity legible rather than dominated by one rule.
    """
    rng = random.Random(seed)
    span = OPEN_EAR - SHUT_EAR
    out, t, i = [], 0.0, 0
    while t < TOTAL:
        d = droop_at(t)
        if kind == "heavy_lids":
            blink_len, blink_every = 4, 70        # unchanged by fatigue
            droop_scale = 0.62                    # lids ride lower instead
            noise = 0.16                          # lids waver near the threshold
        else:
            blink_len = int(3 + 12 * d)
            blink_every = int(90 - 40 * d)
            droop_scale = 1.0
            noise = 0.10

        if (i % max(blink_every, 12)) < blink_len:
            ear = SHUT_EAR + rng.gauss(0, 0.004)
        else:
            ear = (OPEN_EAR - d * span * droop_scale) + rng.gauss(0, noise * span)
        out.append((t, max(ear, 0.01)))
        t += 1.0 / FPS
        i += 1
    return out


def run_pipeline(
    samples: List[Tuple[float, float]],
    lvl_cfg: Optional[LevelConfig] = None,
    win_cfg: Optional[WindowConfig] = None,
    alarm_level: int = 3,
) -> Dict:
    """Feed one episode through the driver pipeline and score the response."""
    pipe = DriverPipeline(
        "sweep",
        cal_cfg=CalibrationConfig(calibration_window_sec=ALERT_END * 0.8),
        lvl_cfg=lvl_cfg, win_cfg=win_cfg, update_hz=1.0,
    )
    first_alarm: Optional[float] = None
    false_alarms = 0
    levels: List[Tuple[float, int]] = []

    for t, ear in samples:
        st = pipe.process(FaceObservation(t=t, face_present=True, ear=ear), t)
        levels.append((t, st.drowsiness_level))
        if st.drowsiness_level >= alarm_level:
            if t < ALERT_END:
                false_alarms += 1
            elif first_alarm is None:
                first_alarm = t

    # Count false alarms as episodes, not frames.
    fa_episodes, prev = 0, False
    for t, lv in levels:
        hot = lv >= alarm_level and t < ALERT_END
        if hot and not prev:
            fa_episodes += 1
        prev = hot

    return {
        "detected": first_alarm is not None,
        "latency_sec": (first_alarm - TRUE_ONSET) if first_alarm else None,
        "false_alarms": fa_episodes,
        "peak_level": max(lv for _, lv in levels),
    }


def summarize(runs: List[Dict]) -> Dict:
    lat = [r["latency_sec"] for r in runs if r["latency_sec"] is not None]
    return {
        "detection_rate": sum(r["detected"] for r in runs) / len(runs),
        "median_latency_sec": statistics.median(lat) if lat else None,
        "mean_false_alarms": statistics.mean(r["false_alarms"] for r in runs),
    }


# ---------------- sweeps ----------------

def sweep_hysteresis(seeds: List[int], kind: str = "microsleep") -> List[Dict]:
    out = []
    for esc in (1, 2, 3, 5, 8):
        runs = [run_pipeline(episode(s, kind), lvl_cfg=LevelConfig(escalate_ticks=esc))
                for s in seeds]
        out.append({"escalate_ticks": esc, **summarize(runs)})
    return out


def sweep_perclos(seeds: List[int]) -> List[Dict]:
    """Swept at level 2, which PERCLOS actually drives.

    At level 3 the microsleep rule fires first and the PERCLOS thresholds make
    no difference at all — itself a finding, and reported as one.
    """
    out = []
    for l2 in (0.08, 0.12, 0.15, 0.20, 0.25, 0.30):
        cfg = LevelConfig(perclos_l1=l2 * 0.55, perclos_l2=l2, perclos_l3=l2 * 1.6,
                          long_blink_ms=10_000)   # isolate PERCLOS from blinks
        runs = [run_pipeline(episode(s, "heavy_lids"), lvl_cfg=cfg, alarm_level=2)
                for s in seeds]
        out.append({"perclos_l2": l2, **summarize(runs)})
    return out


def sweep_window(seeds: List[int]) -> List[Dict]:
    out = []
    for span in (20.0, 40.0, 60.0, 90.0, 120.0):
        runs = [run_pipeline(episode(s, "heavy_lids"),
                             win_cfg=WindowConfig(span_sec=span),
                             lvl_cfg=LevelConfig(long_blink_ms=10_000),
                             alarm_level=2)
                for s in seeds]
        out.append({"window_sec": span, **summarize(runs)})
    return out


def ablation_fusion(seeds: List[int], vehicle_lead_sec: float = 45.0,
                    alarm_level: int = 2) -> List[Dict]:
    """Facial-only versus fused, under the literature's premise.

    IMPORTANT: the premise is ASSUMED here, not demonstrated. Lane-keeping is
    reported in the driving-fatigue literature to degrade before facial signs
    become visible, so the simulated vehicle channel is given a head start of
    `vehicle_lead_sec`. What this quantifies is how much of that head start
    the fusion rule actually converts into earlier detection — which is a
    property of the rule. Whether the premise holds for real drivers is exactly
    what H2 tests, and it is untested here.
    """
    out = []
    for label, use_vehicle, mode in (("facial only", False, "weighted"),
                                     ("fused, weighted", True, "weighted"),
                                     ("fused, asymmetric", True, "asymmetric")):
        latencies, fas = [], []
        for s in seeds:
            samples = episode(s)
            pipe = DriverPipeline(
                "abl", cal_cfg=CalibrationConfig(calibration_window_sec=ALERT_END * 0.8),
                update_hz=1.0)
            tracker = LaneTracker()
            rng = random.Random(s + 900)
            first, fa = None, 0

            for t, ear in samples:
                st = pipe.process(FaceObservation(t=t, face_present=True, ear=ear), t)

                # Lane wander tracks droop, shifted earlier by the lead time.
                d = droop_at(t + vehicle_lead_sec)
                wander = 0.05 + 0.30 * d
                tracker.add(RoadState(t=t, speed_kmh=90.0,
                                      lane_offset_m=rng.gauss(0, wander)))

                if use_vehicle:
                    ls = tracker.stats()
                    est = fuse([
                        facial_channel(st.drowsiness_level, st.confidence, True),
                        vehicle_channel(ls.sdlp, samples=ls.samples),
                    ], mode=mode)
                    level = est.level
                else:
                    level = st.drowsiness_level

                if level >= alarm_level:
                    if t < ALERT_END:
                        fa += 1
                    elif first is None:
                        first = t

            if first is not None:
                latencies.append(first - TRUE_ONSET)
            fas.append(1 if fa else 0)

        out.append({
            "channel": label,
            "mode": mode,
            "alarm_level": alarm_level,
            "detection_rate": len(latencies) / len(seeds),
            "median_latency_sec": statistics.median(latencies) if latencies else None,
            "false_alarm_rate": statistics.mean(fas),
        })
    return out


def sweep_workload_ceiling() -> List[Dict]:
    """How the H3 gate trades interventions against silence.

    Analytic rather than simulated: the workload distribution comes from the
    estimator over a scripted mix of driving situations, and the question is
    simply what fraction of a level-1 opportunity survives each ceiling.
    """
    from awas.road.workload import WorkloadEstimator

    situations = [
        ("stopped", [0.0] * 20, {}),
        ("highway cruise", [90.0] * 20, {"road_class": "highway"}),
        ("arterial", [55.0] * 20, {"road_class": "arterial"}),
        ("urban steady", [30.0] * 20, {"road_class": "urban"}),
        ("urban stop-go", [5, 40, 10, 35, 0, 30] * 4, {"road_class": "urban"}),
        ("close following", [90.0] * 20,
         {"road_class": "highway", "vehicle_ahead_distance_m": 5.0}),
        ("after a departure", [90.0] * 20,
         {"road_class": "highway", "lane_departure": True}),
    ]
    values = []
    for _, speeds, kw in situations:
        est, w, t = WorkloadEstimator(), None, 0.0
        for s in speeds:
            w = est.update(RoadState(t=t, speed_kmh=s, **kw))
            t += 1.0
        values.append(w.value)

    out = []
    for ceiling in (0.2, 0.3, 0.4, 0.5, 0.7, 1.01):
        allowed = sum(1 for v in values if v < ceiling)
        out.append({
            "ceiling": ceiling,
            "situations_permitting_speech": allowed,
            "of": len(values),
            "fraction": allowed / len(values),
        })
    return out


# ---------------- main ----------------

def table(title: str, rows: List[Dict], cols: List[Tuple[str, str, str]]) -> None:
    print(f"\n{title}")
    head = "  " + " ".join(f"{h:>{w}}" for _, h, w in
                           [(k, h, int(w)) for k, h, w in cols])
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        cells = []
        for key, _, w in cols:
            v = r.get(key)
            if v is None:
                cells.append(f"{'—':>{int(w)}}")
            elif isinstance(v, float):
                cells.append(f"{v:>{int(w)}.2f}")
            else:
                cells.append(f"{v:>{int(w)}}")
        print("  " + " ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Parameter sweeps for Section IV.")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--out", default="artifacts/sweeps.json")
    args = ap.parse_args()

    seeds = list(range(args.episodes))
    print(f"simulated episodes: {len(seeds)}  "
          f"({TOTAL:.0f} s each, true onset at {TRUE_ONSET:.0f} s)")

    # How far up the scale the vehicle channel can reach at plausible SDLP.
    from awas.fusion.core import SDLP_L3_M, vehicle_channel
    reach = [{"sdlp_m": sd,
              "vehicle_level": round(vehicle_channel(sd, samples=300).level, 2)}
             for sd in (0.10, 0.18, 0.26, 0.32, 0.35, 0.45)]

    results = {
        "meta": {"episodes": len(seeds), "fps": FPS, "true_onset_sec": TRUE_ONSET,
                 "total_sec": TOTAL, "synthetic": True},
        "hysteresis": sweep_hysteresis(seeds),
        "hysteresis_heavy_lids": sweep_hysteresis(seeds, "heavy_lids"),
        "perclos": sweep_perclos(seeds),
        "perclos_reliable": False,
        "window": sweep_window(seeds),
        "window_reliable": False,
        "fusion_ablation": ablation_fusion(seeds),
        "workload_ceiling": sweep_workload_ceiling(),
        "vehicle_channel_reach": reach,
        "sdlp_l3_threshold_m": SDLP_L3_M,
    }

    table("A. Hysteresis — escalation ticks required", results["hysteresis"],
          [("escalate_ticks", "ticks", 6), ("detection_rate", "detect", 7),
           ("median_latency_sec", "latency s", 10), ("mean_false_alarms", "false", 6)])
    print("\n  NOTE: sweeps B and C did not produce a stable ordering under this")
    print("  episode model — detection rate collapses to 0 or 1 with little in")
    print("  between, so the numbers below are reported for completeness and are")
    print("  NOT used in the paper. Characterising PERCLOS sensitivity properly")
    print("  needs real footage with graded drowsiness, not a synthetic ramp.")
    table("B. PERCLOS level-2 threshold  [UNRELIABLE]", results["perclos"],
          [("perclos_l2", "thresh", 7), ("detection_rate", "detect", 7),
           ("median_latency_sec", "latency s", 10), ("mean_false_alarms", "false", 6)])
    table("C. Observation window length  [UNRELIABLE]", results["window"],
          [("window_sec", "window s", 9), ("detection_rate", "detect", 7),
           ("median_latency_sec", "latency s", 10), ("mean_false_alarms", "false", 6)])
    table("D. Channel ablation", results["fusion_ablation"],
          [("channel", "channel", 18), ("detection_rate", "detect", 7),
           ("median_latency_sec", "latency s", 10), ("false_alarm_rate", "false", 6)])
    table("D2. Vehicle channel reach vs SDLP", results["vehicle_channel_reach"],
          [("sdlp_m", "sdlp m", 7), ("vehicle_level", "level", 6)])
    table("E. Workload ceiling (H3 gate)", results["workload_ceiling"],
          [("ceiling", "ceiling", 8), ("situations_permitting_speech", "speaks", 7),
           ("of", "of", 4), ("fraction", "share", 6)])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
