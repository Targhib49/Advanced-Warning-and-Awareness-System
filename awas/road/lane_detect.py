"""Lane detection — where the vehicle sits within its lane (plan §4.2).

Feeds `lane_offset_m` to the road channel, which is what SDLP is computed from,
which is the vehicle half of hypothesis H2. Without this the road channel has
nothing real to say.

Classic computer vision, no model, and that is a deliberate choice rather than
a limitation: lane markings are high-contrast painted lines with known geometry,
and a thresholded bird's-eye histogram finds them in about a millisecond on a
laptop. A learned lane segmenter would cost a training set we do not have, a
second model on the Pi's budget, and explainability — to detect the same lines.

The pipeline:

    frame -> perspective warp to bird's-eye -> binary threshold
          -> column histogram -> sliding windows -> polynomial fit
          -> offset from lane centre, in metres

Metres per pixel are derived per frame from the detected lane width against an
assumed real width, so the scale self-corrects if the camera is re-aimed. A
frame where the lanes are not confidently found returns `None`, never zero —
"I cannot see the lane" and "the vehicle is centred" must not look alike.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

Point = Tuple[float, float]


def mask(frame: np.ndarray, boxes) -> np.ndarray:
    """A copy of `frame` with each fractional box set to black."""
    out = frame.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in boxes:
        out[int(y0 * h):int(round(y1 * h)), int(x0 * w):int(round(x1 * w))] = 0
    return out


@dataclass(frozen=True)
class LaneCalibration:
    """Camera geometry. Depends on where the camera is mounted and aimed.

    The defaults suit a dashcam near the rear-view mirror on a 640x480 feed.
    They are a starting point: run `scripts/calibrate_lane.py` on real footage
    from the actual mounting before trusting an absolute offset.
    """

    width: int = 640
    height: int = 480

    # Trapezoid in the source image that maps to a rectangle when warped.
    # Fractions of the frame so the same calibration survives a resize.
    top_y: float = 0.62          # horizon-ish cut
    bottom_y: float = 0.96
    top_half_width: float = 0.09
    bottom_half_width: float = 0.72

    warp_width: int = 320
    warp_height: int = 480
    warp_margin: float = 0.25    # lane should land inside these margins

    # Typical lane width. Indonesian arterial lanes run about 3.0-3.5 m;
    # motorway lanes about 3.5 m.
    lane_width_m: float = 3.5

    # A lane narrower or wider than this in the warped view is a bad fit,
    # not a strange road.
    min_lane_px: int = 60
    max_lane_px: int = 300

    # Regions blanked before detection, as (x0, y0, x1, y1) fractions of the
    # frame: overlays that sit on the road in the image, such as a
    # simulator's speedometer and route advisor, whose bright edges would
    # otherwise be read as lane paint. Part of the calibration because they
    # belong to one view, as the trapezoid does.
    mask_boxes: Tuple[Tuple[float, float, float, float], ...] = ()

    def src_points(self) -> np.ndarray:
        w, h = self.width, self.height
        cx = w / 2.0
        return np.float32([
            [cx - self.bottom_half_width * cx, self.bottom_y * h],
            [cx + self.bottom_half_width * cx, self.bottom_y * h],
            [cx + self.top_half_width * cx, self.top_y * h],
            [cx - self.top_half_width * cx, self.top_y * h],
        ])

    def dst_points(self) -> np.ndarray:
        w, h = self.warp_width, self.warp_height
        m = self.warp_margin * w
        return np.float32([[m, h], [w - m, h], [w - m, 0], [m, 0]])


@dataclass
class LaneDetection:
    """One frame's answer."""

    offset_m: Optional[float] = None       # +ve = vehicle right of centre
    confidence: float = 0.0
    lane_width_px: Optional[float] = None
    left_x: Optional[float] = None
    right_x: Optional[float] = None
    reason: str = ""
    binary: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def found(self) -> bool:
        return self.offset_m is not None


class LaneDetector:
    def __init__(self, cal: Optional[LaneCalibration] = None,
                 min_confidence: float = 0.35, n_windows: int = 9):
        self.cal = cal or LaneCalibration()
        self.min_confidence = min_confidence
        self.n_windows = n_windows
        self._M: Optional[np.ndarray] = None

    # ---------------- stages ----------------

    def warp(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        if self._M is None:
            self._M = cv2.getPerspectiveTransform(
                self.cal.src_points(), self.cal.dst_points())
        return cv2.warpPerspective(
            frame, self._M, (self.cal.warp_width, self.cal.warp_height),
            flags=cv2.INTER_LINEAR)

    def threshold(self, warped: np.ndarray) -> np.ndarray:
        """Isolate lane paint: bright, and brighter than its surroundings.

        Two cues rather than one because either alone fails often — absolute
        brightness loses white paint in glare, and local contrast picks up
        every crack in the asphalt. Requiring both is far steadier.
        """
        import cv2
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY) if warped.ndim == 3 else warped
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Local contrast: brighter than the median of the row's neighbourhood.
        blur = cv2.blur(gray, (61, 5))
        contrast = cv2.subtract(gray, blur)
        _, by_contrast = cv2.threshold(contrast, 18, 255, cv2.THRESH_BINARY)

        # Absolute brightness, relative to this frame so it survives night.
        cut = max(np.percentile(gray, 92), 60)
        _, by_bright = cv2.threshold(gray, float(cut), 255, cv2.THRESH_BINARY)

        return cv2.bitwise_and(by_contrast, by_bright)

    def _base_columns(self, binary: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """Left and right lane positions at the bottom of the warped image."""
        bottom = binary[binary.shape[0] // 2:, :]
        hist = bottom.sum(axis=0).astype(np.float64)
        if hist.max() <= 0:
            return None, None
        mid = len(hist) // 2
        left = int(np.argmax(hist[:mid]))
        right = int(np.argmax(hist[mid:]) + mid)
        return (left if hist[left] > 0 else None,
                right if hist[right] > 0 else None)

    def _follow(self, binary: np.ndarray, base: int, margin: int = 45) -> List[float]:
        """Sliding windows up the image, tracking one lane line."""
        h = binary.shape[0]
        win_h = h // self.n_windows
        nz_y, nz_x = binary.nonzero()
        current, centres = base, []

        for i in range(self.n_windows):
            lo, hi = h - (i + 1) * win_h, h - i * win_h
            sel = ((nz_y >= lo) & (nz_y < hi) &
                   (nz_x >= current - margin) & (nz_x < current + margin))
            xs = nz_x[sel]
            if len(xs) > 30:
                current = int(xs.mean())
                centres.append(float(current))
        return centres

    # ---------------- public ----------------

    def detect(self, frame: np.ndarray, keep_binary: bool = False) -> LaneDetection:
        cal = self.cal
        if cal.mask_boxes:
            frame = mask(frame, cal.mask_boxes)
        warped = self.warp(frame)
        binary = self.threshold(warped)
        keep = binary if keep_binary else None

        left_base, right_base = self._base_columns(binary)
        if left_base is None or right_base is None:
            return LaneDetection(reason="no lane pixels", binary=keep)

        left_pts = self._follow(binary, left_base)
        right_pts = self._follow(binary, right_base)
        if not left_pts or not right_pts:
            return LaneDetection(reason="lane lines not traceable", binary=keep)

        left_x = float(np.mean(left_pts[:3]))    # near field, where we are now
        right_x = float(np.mean(right_pts[:3]))
        lane_px = right_x - left_x

        if not (cal.min_lane_px <= lane_px <= cal.max_lane_px):
            return LaneDetection(
                reason=f"implausible lane width ({lane_px:.0f} px)",
                lane_width_px=lane_px, left_x=left_x, right_x=right_x,
                binary=keep)

        # How much of the lane we actually traced, both sides.
        coverage = (len(left_pts) + len(right_pts)) / (2.0 * self.n_windows)
        confidence = float(np.clip(coverage, 0.0, 1.0))
        if confidence < self.min_confidence:
            return LaneDetection(
                reason=f"weak trace (coverage {coverage:.2f})",
                confidence=confidence, lane_width_px=lane_px,
                left_x=left_x, right_x=right_x, binary=keep)

        # Scale from the detected lane, so the answer is in metres even if the
        # warp parameters are imperfect.
        m_per_px = cal.lane_width_m / lane_px
        lane_centre = (left_x + right_x) / 2.0
        camera_x = cal.warp_width / 2.0
        offset_m = (camera_x - lane_centre) * m_per_px

        return LaneDetection(
            offset_m=offset_m, confidence=confidence, lane_width_px=lane_px,
            left_x=left_x, right_x=right_x, reason="ok", binary=keep)


# ---------------- synthetic frames, for tests and calibration ----------------

def synthetic_road(
    offset_m: float = 0.0,
    cal: Optional[LaneCalibration] = None,
    lane_width_m: float = 3.5,
    dashed_right: bool = True,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Render a straight road seen from a camera offset from the lane centre.

    Built by drawing the lane in bird's-eye space and inverse-warping it, so
    the geometry is exact by construction and a detector can be checked against
    a known answer rather than against an eyeball.
    """
    import cv2

    cal = cal or LaneCalibration()
    bev = np.full((cal.warp_height, cal.warp_width, 3), 60, np.uint8)

    lane_px = (1.0 - 2.0 * cal.warp_margin) * cal.warp_width
    m_per_px = lane_width_m / lane_px
    centre = cal.warp_width / 2.0 - offset_m / m_per_px
    left = centre - lane_px / 2.0
    right = centre + lane_px / 2.0

    cv2.line(bev, (int(left), 0), (int(left), cal.warp_height), (245, 245, 245), 7)
    if dashed_right:
        step = cal.warp_height // 8
        for i in range(0, cal.warp_height, step * 2):
            cv2.line(bev, (int(right), i), (int(right), i + step),
                     (245, 245, 245), 7)
    else:
        cv2.line(bev, (int(right), 0), (int(right), cal.warp_height),
                 (245, 245, 245), 7)

    if noise:
        rng = np.random.default_rng(seed)
        bev = np.clip(bev + rng.normal(0, noise, bev.shape), 0, 255).astype(np.uint8)

    M = cv2.getPerspectiveTransform(cal.dst_points(), cal.src_points())
    frame = cv2.warpPerspective(bev, M, (cal.width, cal.height))
    # Everything above the horizon is sky, not road.
    frame[:int(cal.top_y * cal.height), :] = 110
    return frame
