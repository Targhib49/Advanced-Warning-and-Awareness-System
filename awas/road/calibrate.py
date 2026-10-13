"""Work out the lane trapezoid from footage instead of guessing it.

`LaneCalibration`'s defaults assume a dashcam near the rear-view mirror. Any
other mounting — higher, lower, tilted, a different lens — and the perspective
warp is wrong, detection rate collapses, and every offset it does produce is
scaled incorrectly. Neither failure announces itself.

Two stages, because neither alone is reliable:

  1. **Vanishing point.** Lane lines are parallel in the world, so in the image
     they converge on the horizon. Finding where they meet gives the horizon
     height directly, which is the single most important parameter.

  2. **Scored search.** With a starting point in hand, try nearby trapezoids
     and keep whichever actually detects best. This optimises the thing we
     care about — usable lane readings — rather than a geometric proxy.

The scoring function is the honest part of this module: a calibration is good
if it finds the lane often, at a consistent width, with a steady offset. All
three are measurable without knowing the right answer, which is what makes
automatic calibration possible at all.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from awas.road.lane_detect import LaneCalibration, LaneDetector, mask


@dataclass
class CalibrationScore:
    detection_rate: float = 0.0
    width_consistency: float = 0.0    # 1.0 = perfectly steady lane width
    offset_stability: float = 0.0     # 1.0 = no jitter frame to frame
    mean_confidence: float = 0.0
    frames: int = 0

    @property
    def total(self) -> float:
        """One number to rank candidates by.

        Detection rate dominates: a calibration that finds the lane in a fifth
        of frames is useless however tidy those few readings are.
        """
        return (0.50 * self.detection_rate
                + 0.20 * self.width_consistency
                + 0.20 * self.offset_stability
                + 0.10 * self.mean_confidence)

    def __str__(self) -> str:
        return (f"score {self.total:.3f}  "
                f"found {self.detection_rate:.0%}  "
                f"width {self.width_consistency:.2f}  "
                f"steady {self.offset_stability:.2f}  "
                f"conf {self.mean_confidence:.2f}")


def score_calibration(cal: LaneCalibration, frames: Sequence[np.ndarray]) -> CalibrationScore:
    """How well does this trapezoid work on these frames?"""
    det = LaneDetector(cal)
    offsets: List[float] = []
    widths: List[float] = []
    confs: List[float] = []

    for frame in frames:
        r = det.detect(frame)
        if r.found:
            offsets.append(r.offset_m)
            widths.append(r.lane_width_px)
            confs.append(r.confidence)

    n = len(frames)
    out = CalibrationScore(frames=n)
    if not offsets:
        return out

    out.detection_rate = len(offsets) / n
    out.mean_confidence = float(np.mean(confs))

    if len(widths) > 1 and statistics.mean(widths) > 0:
        cv = statistics.pstdev(widths) / statistics.mean(widths)
        out.width_consistency = float(max(0.0, 1.0 - cv * 4.0))
    else:
        out.width_consistency = 1.0

    if len(offsets) > 2:
        # Frame-to-frame jump, not overall spread: a vehicle genuinely moving
        # across the lane should not be punished, but a calibration that makes
        # the estimate flicker should be.
        jumps = [abs(b - a) for a, b in zip(offsets, offsets[1:])]
        out.offset_stability = float(max(0.0, 1.0 - statistics.mean(jumps) / 0.5))
    else:
        out.offset_stability = 0.5

    return out


# ---------------- stage 1: vanishing point ----------------

def find_vanishing_point(
    frames: Sequence[np.ndarray],
    min_slope: float = 0.30,
    max_slope: float = 4.0,
) -> Optional[Tuple[float, float]]:
    """Where the lane lines converge, in pixels.

    Lines that are nearly horizontal are usually the horizon, a bridge or a
    shadow; lines that are nearly vertical are usually poles or the bonnet.
    Only the diagonals between those extremes behave like lane markings from a
    forward camera, so the slope band is the filter that makes this work.
    """
    import cv2

    xs: List[float] = []
    ys: List[float] = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
        edges[: int(h * 0.35), :] = 0        # ignore sky

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=55,
                                minLineLength=int(h * 0.12), maxLineGap=40)
        if lines is None:
            continue

        left, right = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if not (min_slope <= abs(slope) <= max_slope):
                continue
            intercept = y1 - slope * x1
            (left if slope < 0 else right).append((slope, intercept))

        # A vanishing point needs lines converging from both sides.
        if not left or not right:
            continue
        for ms, bs in ((left, right), ):
            for s1, b1 in ms:
                for s2, b2 in bs:
                    if abs(s1 - s2) < 1e-6:
                        continue
                    x = (b2 - b1) / (s1 - s2)
                    y = s1 * x + b1
                    if 0 <= x <= w and 0 <= y <= h:
                        xs.append(x)
                        ys.append(y)

    if len(xs) < 5:
        return None
    return float(np.median(xs)), float(np.median(ys))


def calibration_from_vanishing_point(
    vp: Tuple[float, float], width: int, height: int,
    base: Optional[LaneCalibration] = None,
) -> LaneCalibration:
    """Turn a vanishing point into a starting trapezoid.

    The top edge sits a little below the horizon — exactly at it the lane is
    infinitely far away and a pixel spans metres, which makes the warp
    numerically hopeless.
    """
    base = base or LaneCalibration()
    _, vy = vp
    horizon = vy / height
    top_y = min(max(horizon + 0.06, 0.45), 0.80)
    return replace(base, width=width, height=height, top_y=float(top_y))


# ---------------- stage 2: scored search ----------------

def refine(
    start: LaneCalibration,
    frames: Sequence[np.ndarray],
    verbose: bool = False,
) -> Tuple[LaneCalibration, CalibrationScore]:
    """Coarse-to-fine search over the parameters that actually matter.

    Only three are searched. The warp size and lane-width assumption are not
    geometry to be discovered — they are decisions, and `lane_width_m` in
    particular must be set from the road being driven, because nothing in the
    image reveals it.
    """
    best, best_score = start, score_calibration(start, frames)
    if verbose:
        print(f"  start   top_y={start.top_y:.2f} "
              f"tw={start.top_half_width:.2f} bw={start.bottom_half_width:.2f}  {best_score}")

    grids = [
        # (top_y deltas, top-half-width options, bottom-half-width options)
        ([-0.08, -0.04, 0.0, 0.04, 0.08], [0.05, 0.09, 0.14, 0.20], [0.55, 0.72, 0.90]),
        ([-0.03, -0.015, 0.0, 0.015, 0.03], [-0.03, 0.0, 0.03], [-0.10, 0.0, 0.10]),
    ]

    for stage, (dys, tws, bws) in enumerate(grids):
        for dy in dys:
            for tw in tws:
                for bw in bws:
                    cand = replace(
                        best,
                        top_y=float(np.clip(best.top_y + dy, 0.40, 0.85)),
                        top_half_width=float(np.clip(
                            tw if stage == 0 else best.top_half_width + tw, 0.02, 0.35)),
                        bottom_half_width=float(np.clip(
                            bw if stage == 0 else best.bottom_half_width + bw, 0.40, 1.0)),
                    )
                    sc = score_calibration(cand, frames)
                    if sc.total > best_score.total:
                        best, best_score = cand, sc
                        if verbose:
                            print(f"  better  top_y={cand.top_y:.3f} "
                                  f"tw={cand.top_half_width:.2f} "
                                  f"bw={cand.bottom_half_width:.2f}  {sc}")
    return best, best_score


def auto_calibrate(
    frames: Sequence[np.ndarray],
    lane_width_m: float = 3.5,
    verbose: bool = False,
    mask_boxes: tuple = (),
) -> Tuple[LaneCalibration, CalibrationScore]:
    """Vanishing point for a starting guess, then search for what works."""
    if not frames:
        raise ValueError("auto_calibrate needs at least one frame")

    h, w = frames[0].shape[:2]
    start = LaneCalibration(width=w, height=h, lane_width_m=lane_width_m,
                            mask_boxes=tuple(mask_boxes))
    if mask_boxes:
        frames = [mask(f, mask_boxes) for f in frames]

    vp = find_vanishing_point(frames)
    if vp is not None:
        start = calibration_from_vanishing_point(vp, w, h, start)
        start = replace(start, lane_width_m=lane_width_m, mask_boxes=tuple(mask_boxes))
        if verbose:
            print(f"  vanishing point at ({vp[0]:.0f}, {vp[1]:.0f}) "
                  f"-> horizon {vp[1] / h:.2f}, top_y {start.top_y:.2f}")
    elif verbose:
        print("  no vanishing point found; searching from the defaults")

    return refine(start, frames, verbose=verbose)


# ---------------- persistence ----------------

def save(cal: LaneCalibration, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cal), indent=2))
    return path


def load(path: Path) -> Optional[LaneCalibration]:
    try:
        raw = json.loads(Path(path).read_text())
        raw["mask_boxes"] = tuple(tuple(b) for b in raw.get("mask_boxes", ()))
        return LaneCalibration(**raw)
    except (OSError, json.JSONDecodeError, TypeError, AttributeError):
        return None
