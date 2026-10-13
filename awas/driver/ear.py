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


# Iris centres in the 478-point mesh (the 468 + 10 refinement points), matched
# to the EAR sets above: 473 sits in MEDIAPIPE_LEFT_EYE, 468 in the right.
IRIS_LEFT_CENTER = 473
IRIS_RIGHT_CENTER = 468


def iris_drop(landmarks: Sequence[Point], eye_idx: Sequence[int], iris_idx: int) -> float:
    """How far the iris sits below the line through the eye corners, as a
    fraction of eye width. Positive is downward.

    Looking down lowers the upper lid with the eye, and an eye-state model
    reads the lowered lid as closing (bench drive 3: a glance at the lap
    counted as a microsleep). The iris going with it is what tells a downward
    look from a closed eye. Rotation-invariant: measured along the normal to
    the corner line, so a tilted head does not read as a gaze shift.
    """
    if len(landmarks) <= max(max(eye_idx), iris_idx):
        raise ValueError("iris_drop needs the 478-point mesh")
    (x1, y1), (x4, y4) = landmarks[eye_idx[0]], landmarks[eye_idx[3]]
    dx, dy = x4 - x1, y4 - y1
    width = math.hypot(dx, dy)
    if width <= 0:
        return 0.0
    nx, ny = -dy / width, dx / width
    if ny < 0:                         # corners listed right-to-left: point the normal down
        nx, ny = -nx, -ny
    ix, iy = landmarks[iris_idx]
    mx, my = (x1 + x4) / 2.0, (y1 + y4) / 2.0
    return ((ix - mx) * nx + (iy - my) * ny) / width


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
