"""T1 in the live path: the learned eye-state CNN on aligned eye crops.

The model (training/model.py, EyeStateNet) was trained on MRL eye crops:
square, grayscale, the eye centred and spanning roughly three quarters of the
width, resized to 32x32 and scaled to [0, 1]. Everything here exists to make a
live crop look like that, because a model fed crops framed differently from
its training data is answering a question it was never asked.

The crop is ALIGNED (plan v3 §T0): rotated so the eye corners are level. A
tilted head otherwise shows the model a tilted eye, which MRL rarely contains.

EAR stays alongside as fallback and control (plan v3 §T1, "ship the honest
baseline too") — this module only adds P(closed); it replaces nothing.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from awas.driver.ear import MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE

Point = Tuple[float, float]

# Square crop side as a multiple of corner-to-corner eye width. MRL crops show
# the eye across about 70-80% of the frame, i.e. a side of ~1.3-1.4x.
CROP_SCALE = 1.4
CROP_PX = 32
# An eye narrower than this in the frame has too few pixels to classify; the
# answer would be noise dressed up as a probability.
MIN_EYE_PX = 12.0

DEFAULT_MODEL = Path("artifacts/eyestate_fp32.onnx")


def eye_geometry(landmarks: Sequence[Point], idx: Sequence[int]) -> Tuple[Point, float, float]:
    """Centre, corner-to-corner width and roll angle (radians) of one eye.

    `idx` is a six-point EAR subset in p1..p6 order: p1 and p4 are the corners.
    """
    pts = np.asarray([landmarks[i] for i in idx], dtype=np.float64)
    c1, c4 = pts[0], pts[3]
    centre = pts.mean(axis=0)
    width = float(np.hypot(*(c4 - c1)))
    angle = math.atan2(c4[1] - c1[1], c4[0] - c1[0])
    # Corners can come in either order (left eye vs right eye); level them
    # without flipping the image upside down.
    if angle > math.pi / 2:
        angle -= math.pi
    elif angle < -math.pi / 2:
        angle += math.pi
    return (float(centre[0]), float(centre[1])), width, angle


def aligned_crop(gray: np.ndarray, centre: Point, width: float, angle: float,
                 scale: float = CROP_SCALE, out_px: int = CROP_PX) -> Optional[np.ndarray]:
    """A square, level crop around one eye, as float32 in [0, 1].

    Returns None when the eye is too small to classify or the crop would fall
    mostly outside the frame.
    """
    import cv2

    if width < MIN_EYE_PX:
        return None
    side = width * scale
    h, w = gray.shape[:2]
    cx, cy = centre
    # A crop centred off the frame is mostly replicated border, not an eye.
    if not (0 <= cx < w and 0 <= cy < h):
        return None

    # One affine: rotate about the eye centre to level it, scale the square to
    # out_px, and translate it to the origin. INTER_AREA because this is a
    # large downscale and the model was trained on antialiased resizes.
    k = out_px / side
    cos, sin = math.cos(-angle) * k, math.sin(-angle) * k
    m = np.array([
        [cos, -sin, out_px / 2 - (cos * cx - sin * cy)],
        [sin, cos, out_px / 2 - (sin * cx + cos * cy)],
    ], dtype=np.float64)
    crop = cv2.warpAffine(gray, m, (out_px, out_px), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)
    return crop.astype(np.float32) / 255.0


class EyeStateClassifier:
    """P(closed) for both eyes from a frame and its face-mesh landmarks."""

    def __init__(self, model_path: Path = DEFAULT_MODEL, threads: int = 1):
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"eye-state model not found at {model_path}. Export it with "
                f"`python -m training.export`, or run with the EAR signal only."
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = Path(model_path)

    def crops(self, rgb: np.ndarray, landmarks: Sequence[Point]) -> list:
        import cv2
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
        out = []
        for idx in (MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE):
            centre, width, angle = eye_geometry(landmarks, idx)
            out.append(aligned_crop(gray, centre, width, angle))
        return out

    def p_closed(self, rgb: np.ndarray, landmarks: Sequence[Point]) -> Optional[float]:
        """Mean P(closed) over the eyes that could be cropped, or None."""
        crops = [c for c in self.crops(rgb, landmarks) if c is not None]
        if not crops:
            return None
        batch = np.stack(crops)[:, None, :, :]
        logits = self.session.run(None, {self.input_name: batch})[0]
        z = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return float(p[:, 0].mean())          # class 0 = closed
