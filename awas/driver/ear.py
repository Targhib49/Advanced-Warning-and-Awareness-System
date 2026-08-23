"""Eye Aspect Ratio — the geometric openness measure (plan §4.1).

EAR (Soukupova & Cech, 2016) over six landmarks per eye:

        p2   p3
    p1            p4          EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)
        p6   p5

p1 and p4 are the corners; the rest are lid points. Dividing by eye width makes
it scale-invariant, so it survives the driver moving toward or away from the
camera.

What it does NOT survive is head rotation: EAR is a ratio of 2D *projected*
distances, so yaw and pitch foreshorten the numerator and denominator by
different amounts and change the value with no eyelid movement at all. That,
plus two- to threefold variation in baseline across individuals, is why the
raw number is never thresholded directly here — see `calibration.py`.

No numpy, no mediapipe: this is pure geometry so it stays testable on its own.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

Point = Tuple[float, float]

# Six-point subsets of the MediaPipe Face Mesh, in p1..p6 order.
# VERIFIED on mediapipe 0.10.35 against a live camera: the points track the
# eyes and open-eye EAR reads ~0.25, inside the expected 0.2-0.4 band.
# Re-check if the mediapipe version moves — index sets have differed between
# releases, and a wrong index silently produces a plausible but meaningless
# ratio rather than an error.
MEDIAPIPE_LEFT_EYE: Tuple[int, ...] = (362, 385, 387, 263, 373, 380)
MEDIAPIPE_RIGHT_EYE: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(points: Sequence[Point]) -> float:
    """EAR for one eye from six (x, y) landmarks in p1..p6 order.

    Returns 0.0 for a degenerate eye (zero width), which is what a failed
    landmark fit looks like. Callers treat 0.0 as "no reading", not as
    "fully closed" — the difference matters, because a dropped fit must not
    look like a microsleep.
    """
    if len(points) != 6:
        raise ValueError(f"eye_aspect_ratio needs exactly 6 points, got {len(points)}")

    width = _dist(points[0], points[3])
    if width <= 1e-9:
        return 0.0

    vertical = _dist(points[1], points[5]) + _dist(points[2], points[4])
    return vertical / (2.0 * width)


def mean_ear(left: Sequence[Point], right: Sequence[Point]) -> float:
    """Mean EAR across both eyes.

    Both eyes are averaged rather than tracked separately. Per-eye tracking
    would handle head yaw better — the far eye foreshortens first — and is the
    obvious refinement once the road-facing channel exists to corroborate it.
    A single degenerate eye falls back to the other rather than halving the
    result.
    """
    l = eye_aspect_ratio(left)
    r = eye_aspect_ratio(right)
    if l <= 0.0 and r <= 0.0:
        return 0.0
    if l <= 0.0:
        return r
    if r <= 0.0:
        return l
    return (l + r) / 2.0


def ear_from_landmarks(
    landmarks: Sequence[Point],
    left_idx: Sequence[int] = MEDIAPIPE_LEFT_EYE,
    right_idx: Sequence[int] = MEDIAPIPE_RIGHT_EYE,
) -> float:
    """Mean EAR from a full face-mesh landmark array."""
    need = max(max(left_idx), max(right_idx))
    if len(landmarks) <= need:
        raise ValueError(
            f"landmark array has {len(landmarks)} points; index {need} required. "
            f"Check that the mesh model returned a full face."
        )
    return mean_ear(
        [landmarks[i] for i in left_idx],
        [landmarks[i] for i in right_idx],
    )
