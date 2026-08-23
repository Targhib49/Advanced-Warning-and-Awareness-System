"""Show what per-driver calibration actually buys, on two different faces.

    python -m scripts.demo_calibration

Two drivers with genuinely different eye geometry drive the same trip and get
drowsy in exactly the same way. A fixed population threshold gets one of them
wrong in each direction; a personal scale gets both right.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from awas.driver.calibration import (
    CalibrationConfig,
    DriverProfile,
    EyeOpennessCalibrator,
    OpennessNormalizer,
    SampleContext,
)

FPS = 30.0
CFG = CalibrationConfig()


@dataclass
class Driver:
    name: str
    open_ear: float      # their EAR with eyes comfortably open
    closed_ear: float    # their EAR mid-blink
    note: str


# Ranges shaped like a real MediaPipe mesh: a shut eye lands near zero, not
# near 0.07. Measured open_ref on one real subject was 0.282 with closed_ref
# 0.018 (awas/driver/calibration.py), which is what these bracket.
DRIVERS = [
    Driver("Hendra", open_ear=0.38, closed_ear=0.025, note="wide-set eyes"),
    Driver("Rina",   open_ear=0.20, closed_ear=0.015, note="narrower aperture"),
]


def trace(driver: Driver, n: int, droop: float, seed: int) -> List[float]:
    """EAR samples for one minute of driving.

    `droop` is the fraction of this driver's own range their eyelids have lost,
    so both drivers are made *equally* drowsy in physiological terms — which is
    exactly the comparison a fixed threshold cannot make.

    Two things scale with droop, both of them real: the resting aperture falls,
    and blinks lengthen. Aperture also wanders frame to frame rather than
    sitting still, which is what makes PERCLOS ramp instead of switching.
    """
    rng = random.Random(seed)
    span = driver.open_ear - driver.closed_ear
    base = driver.open_ear - droop * span

    blink_len = int(4 + 10 * droop)        # blinks lengthen with fatigue
    blink_every = int(90 - 30 * droop)     # and come a little more often

    out = []
    for i in range(n):
        if (i % blink_every) < blink_len:
            ear = driver.closed_ear + rng.gauss(0, 0.01 * span)
        else:
            ear = base + rng.gauss(0, 0.13 * span)
        out.append(max(ear, 0.01))
    return out


def median(values: List[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2]


def perclos(samples: List[float], normalizer: OpennessNormalizer) -> float:
    """Fraction of the window spent more than 80% closed."""
    closed = sum(1 for e in samples if normalizer.is_closed(e))
    return closed / max(len(samples), 1)


def run() -> None:
    population = OpennessNormalizer(None)
    print(f"population scale: open {population.open_ref}  closed {population.closed_ref}\n")

    for driver in DRIVERS:
        print(f"--- {driver.name} ({driver.note}) ---")

        # 1. Calibrate on the first minutes of the trip, driver alert.
        cal = EyeOpennessCalibrator(driver.name)
        for i, ear in enumerate(trace(driver, 4000, droop=0.0, seed=1)):
            cal.observe(ear, SampleContext(trip_elapsed_sec=i / FPS))
        profile = cal.build()
        if profile is None:
            print("  calibration failed\n")
            continue

        personal = OpennessNormalizer(profile)
        print(f"  learned:  open {profile.open_ref:.3f}   closed {profile.closed_ref:.3f}"
              f"   confidence {personal.confidence:.2f}")

        # 2. The same physiological drowsiness, measured both ways.
        print(f"  {'droop':>7} {'med EAR':>8} {'PERCLOS pop':>13} {'PERCLOS personal':>18}   verdict")
        for droop in (0.0, 0.3, 0.55, 0.75, 0.9):
            samples = trace(driver, 1800, droop=droop, seed=2)
            p_pop = perclos(samples, population)
            p_per = perclos(samples, personal)

            pop_alarm = p_pop > 0.25      # plan §4.1: level 3 at PERCLOS > 25%
            per_alarm = p_per > 0.25
            if pop_alarm == per_alarm:
                verdict = "agree"
            elif per_alarm:
                verdict = "population MISSES it"
            else:
                verdict = "population FALSE ALARM"

            print(f"  {droop:7.0%} {median(samples):8.3f} "
                  f"{p_pop:12.1%} {p_per:17.1%}   {verdict}")
        print()

    print("Read the two 'PERCLOS personal' columns against each other: they are")
    print("identical, because both drivers are equally drowsy and the measure now")
    print("says so. The population column is not — the same true state produces")
    print("different readings depending only on the shape of the driver's eyes.")
    print()
    print("No model, no training data, no inference cost. Two numbers per driver.")


if __name__ == "__main__":
    run()
