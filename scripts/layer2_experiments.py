"""Experiments for the road-awareness contribution (paper Section IV).

    python -m scripts.layer2_experiments --out artifacts/layer2.json

Three questions, each isolating something the facial channel alone cannot do:

  E1  Does the vehicle channel detect onset earlier?
  E2  Does it keep working when the face is unavailable?
  E3  Does disagreement between channels suppress false alarms?

As with the sensitivity sweeps, these run on SIMULATED episodes with a known
onset, because detection latency is undefined without ground truth for when
impairment began. E1 additionally ASSUMES the literature's premise — that lane
keeping degrades before facial signs appear — and measures how much of that
lead the fusion rule converts into earlier detection. It does not establish the
premise; that is hypothesis H2 and it requires drivers.

E2 assumes nothing. Occlusion either loses the facial channel or it does not,
and what the vehicle channel does about it is fully determined here.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from awas.driver.calibration import CalibrationConfig
from awas.driver.landmarks import FaceObservation
from awas.driver.pipeline import DriverPipeline
from awas.fusion.core import facial_channel, fuse, vehicle_channel
from awas.road.signals import LaneTracker, RoadState

FPS = 15.0
ALERT_END = 300.0
RAMP_END = 420.0
TOTAL = 600.0
ONSET = ALERT_END
OPEN_EAR, SHUT_EAR = 0.30, 0.06
SPAN = OPEN_EAR - SHUT_EAR


def droop(t: float) -> float:
    if t < ALERT_END:
        return 0.0
    if t < RAMP_END:
        return 0.9 * (t - ALERT_END) / (RAMP_END - ALERT_END)
    return 0.9


def ear_at(t: float, d: float, i: int, rng) -> float:
    blink_len = int(3 + 12 * d)
    every = max(int(90 - 40 * d), 12)
    if (i % every) < blink_len:
        return SHUT_EAR + rng.gauss(0, 0.004)
    return max((OPEN_EAR - d * SPAN) + rng.gauss(0, 0.10 * SPAN), 0.01)


def wander_at(d: float) -> float:
    """Lane wander as a function of droop. Baseline 0.13 m, +0.16 m impaired."""
    return 0.13 + 0.16 * d


def run_trial(
    seed: int,
    use_vehicle: bool,
    vehicle_lead_sec: float = 0.0,
    occlude_from: Optional[float] = None,
    baseline_m: Optional[float] = 0.13,
    alarm_level: int = 3,
    mode: str = "asymmetric",
) -> Dict:
    rng = random.Random(seed)
    rng_lane = random.Random(seed + 900)
    pipe = DriverPipeline("e", cal_cfg=CalibrationConfig(
        calibration_window_sec=ALERT_END * 0.8), update_hz=1.0)
    tracker = LaneTracker()

    first: Optional[float] = None
    false_frames = 0
    covered = 0
    total = 0
    t, i = 0.0, 0

    while t < TOTAL:
        d = droop(t)
        occluded = occlude_from is not None and t >= occlude_from
        obs = FaceObservation(
            t=t, face_present=not occluded,
            ear=0.0 if occluded else ear_at(t, d, i, rng))
        st = pipe.process(obs, t)

        lane_d = droop(t + vehicle_lead_sec)
        tracker.add(RoadState(t=t, speed_kmh=90.0,
                              lane_offset_m=rng_lane.gauss(0, wander_at(lane_d))))

        channels = [facial_channel(st.drowsiness_level, st.confidence,
                                   obs.face_present)]
        if use_vehicle:
            ls = tracker.stats()
            channels.append(vehicle_channel(ls.sdlp, samples=ls.samples,
                                            sdlp_baseline_m=baseline_m))
        est = fuse(channels, mode=mode)

        total += 1
        if est.sources:
            covered += 1
        if est.level >= alarm_level:
            if t < ALERT_END:
                false_frames += 1
            elif first is None:
                first = t

        t += 1.0 / FPS
        i += 1

    return {
        "detected": first is not None,
        "latency_sec": (first - ONSET) if first else None,
        "false_frames": false_frames,
        "coverage": covered / total,
    }


def summarize(runs: List[Dict]) -> Dict:
    lat = [r["latency_sec"] for r in runs if r["latency_sec"] is not None]
    return {
        "detection_rate": round(sum(r["detected"] for r in runs) / len(runs), 3),
        "median_latency_sec": round(statistics.median(lat), 1) if lat else None,
        "mean_coverage": round(statistics.mean(r["coverage"] for r in runs), 3),
        "false_alarm_rate": round(
            sum(1 for r in runs if r["false_frames"]) / len(runs), 3),
    }


def e1_earlier_detection(seeds, lead=45.0) -> List[Dict]:
    return [
        {"config": "facial only",
         **summarize([run_trial(s, False) for s in seeds])},
        {"config": f"+ vehicle (leads {lead:.0f}s)",
         **summarize([run_trial(s, True, vehicle_lead_sec=lead) for s in seeds])},
    ]


def e2_occlusion(seeds, occlude_from=330.0) -> List[Dict]:
    return [
        {"config": "facial only",
         **summarize([run_trial(s, False, occlude_from=occlude_from)
                      for s in seeds])},
        {"config": "+ vehicle",
         **summarize([run_trial(s, True, occlude_from=occlude_from)
                      for s in seeds])},
    ]


def e3_false_alarms(seeds) -> List[Dict]:
    """A facial channel that misreads an alert driver, with lanes held steady.

    Simulates the glasses/glare failure mode: the face looks drowsy, the
    vehicle is not wandering. Weighted fusion should damp it; asymmetric should
    not, which is the cost of the asymmetric mode stated honestly.
    """
    out = []
    for label, mode, use_vehicle in (("facial only", "weighted", False),
                                     ("+ vehicle, asymmetric", "asymmetric", True),
                                     ("+ vehicle, weighted", "weighted", True)):
        levels = []
        for s in seeds:
            ls_tracker = LaneTracker()
            rng = random.Random(s)
            t = 0.0
            while t < 120.0:
                ls_tracker.add(RoadState(t=t, speed_kmh=90.0,
                                         lane_offset_m=rng.gauss(0, 0.13)))
                t += 1.0 / FPS
            ls = ls_tracker.stats()
            channels = [facial_channel(3, 0.85, True)]   # face misreads as drowsy
            if use_vehicle:
                channels.append(vehicle_channel(ls.sdlp, samples=ls.samples,
                                                sdlp_baseline_m=0.13))
            est = fuse(channels, mode=mode)
            levels.append(est.level)
        out.append({"config": label,
                    "median_level": statistics.median(levels),
                    "fires_at_l3": round(
                        sum(1 for l in levels if l >= 3) / len(levels), 3)})
    return out


def table(title, rows, cols):
    print(f"\n{title}")
    head = "  " + " ".join(f"{h:>{w}}" for _, h, w in cols)
    print(head + "\n  " + "-" * (len(head) - 2))
    for r in rows:
        cells = []
        for k, _, w in cols:
            v = r.get(k)
            cells.append(f"{'—':>{w}}" if v is None else
                         (f"{v:>{w}.2f}" if isinstance(v, float) else f"{v:>{w}}"))
        print("  " + " ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser(description="Road-awareness experiments.")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="artifacts/layer2.json")
    args = ap.parse_args()
    seeds = list(range(args.episodes))

    res = {
        "meta": {"episodes": len(seeds), "synthetic": True,
                 "onset_sec": ONSET, "total_sec": TOTAL},
        "e1_earlier_detection": e1_earlier_detection(seeds),
        "e2_occlusion": e2_occlusion(seeds),
        "e3_false_alarms": e3_false_alarms(seeds),
    }

    cols = [("config", "configuration", 24), ("detection_rate", "detect", 7),
            ("median_latency_sec", "latency s", 10), ("mean_coverage", "coverage", 9)]
    table("E1. Onset detection, vehicle channel leading", res["e1_earlier_detection"], cols)
    table("E2. Face occluded from t=330 s", res["e2_occlusion"], cols)
    table("E3. Facial false positive, lanes steady", res["e3_false_alarms"],
          [("config", "configuration", 24), ("median_level", "level", 6),
           ("fires_at_l3", "fires L3", 9)])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
