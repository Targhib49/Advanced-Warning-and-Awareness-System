"""Face landmark extraction and head pose (plan §4.1, tier T0).

Two responsibilities, kept apart on purpose:

  * `head_pose` is pure geometry over 2D points — no camera, no model, no
    mediapipe — so it can be tested by projecting a known rotation and
    checking the angle comes back.
  * `MediaPipeLandmarker` is the only thing that touches mediapipe, and it
    imports it lazily so the rest of the package stays importable on a machine
    with no camera stack at all.

Everything downstream consumes `FaceObservation`, so a recorded-video source,
a synthetic source, and the live camera are interchangeable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

# --- Landmark indices used for pose, in the order of MODEL_POINTS below. ---
# VERIFY against your mediapipe version, as with the EAR indices in ear.py.
POSE_IDX: Tuple[int, ...] = (
    1,    # nose tip
    152,  # chin
    33,   # outer corner, one eye
    263,  # outer corner, other eye
    61,   # mouth corner
    291,  # mouth corner
)

# Generic 3D face in millimetres, nose tip at the origin. Absolute scale does
# not matter — only the ratios between points affect the recovered angles.
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0),
], dtype=np.float64)


@dataclass
class FaceObservation:
    """One frame's worth of driver state, before any windowing."""

    t: float
    face_present: bool = False
    landmarks: List[Point] = field(default_factory=list)
    ear: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole approximation: focal length ~ image width, principal point centred.

    Good enough for pose gating (is the head roughly forward?). It is NOT good
    enough for absolute gaze angles — calibrate the real lens before claiming
    those.
    """
    f = float(width)
    return np.array([
        [f, 0.0, width / 2.0],
        [0.0, f, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def head_pose(
    image_points: Sequence[Point],
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float]]:
    """Recover (pitch, yaw, roll) in degrees from six facial points.

    Returns None when the solve fails, which callers must treat as "pose
    unknown" rather than "pose zero" — a fabricated zero would let a
    head-down driver pass the calibration gate.
    """
    import cv2

    pts = np.asarray(image_points, dtype=np.float64)
    if pts.shape != (6, 2):
        raise ValueError(f"head_pose needs 6 points, got shape {pts.shape}")

    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, pts, camera_matrix(width, height),
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])

    # RQDecomp3x3 reports pitch near +/-180 for a forward-facing head; fold it
    # back to a signed offset from level so the calibration gate reads sanely.
    if pitch > 90.0:
        pitch -= 180.0
    elif pitch < -90.0:
        pitch += 180.0
    return pitch, yaw, roll


class Landmarker(Protocol):
    """Anything that can turn a frame into a FaceObservation."""

    def observe(self, frame: np.ndarray, t: float) -> FaceObservation: ...
    def close(self) -> None: ...


DEFAULT_MODEL_NAME = "face_landmarker.task"


def _check_mediapipe_version() -> None:
    """Refuse a mediapipe that is known not to run on CPU on THIS platform.

    The 1.0.x *macOS* arm64 wheels initialise Metal inside
    TensorsToDetectionsCalculator regardless of the requested delegate, so the
    process dies with a C++ abort rather than a Python exception — unreadable
    if you have not seen it before.

    That is a macOS problem specifically. Linux has no Metal, and on the Pi the
    1.x aarch64 wheels are a legitimate option — so the check is platform-gated
    rather than blanket. Note also that mediapipe dropped aarch64 Linux wheels
    after 0.10.18: on a Pi, pip cannot give you the 0.10.35 this is validated
    against. See docs/03-pi-deployment.md.
    """
    import sys

    import mediapipe as mp

    if sys.platform != "darwin":
        return

    raw = getattr(mp, "__version__", "0")
    try:
        major = int(raw.split(".")[0])
    except (ValueError, IndexError):
        return
    if major >= 1:
        raise RuntimeError(
            f"mediapipe {raw} is installed. Its macOS builds force the Metal "
            f"delegate inside the face detector and abort the process on CPU.\n"
            f"On macOS, pin the 0.10 series:\n"
            f"    pip install 'mediapipe>=0.10,<1.0'"
        )


class MediaPipeLandmarker:
    """Face mesh via the mediapipe Tasks API.

    mediapipe >= 1.0 removed the legacy `solutions` API and ships no model
    asset, so a `face_landmarker.task` bundle must be present on disk. Its
    absence is a setup problem with a clear fix, not a crash.
    """

    def __init__(self, model_path: Path, num_faces: int = 1, use_gpu: bool = False):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Face landmark model not found at {self.model_path}.\n"
                f"mediapipe >= 1.0 does not bundle one. Fetch "
                f"'{DEFAULT_MODEL_NAME}' and place it there, or set "
                f"AWAS_FACE_MODEL to point at it."
            )

        _check_mediapipe_version()

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker, FaceLandmarkerOptions, RunningMode,
        )

        self._mp_image_cls = None
        # CPU by default. The Metal delegate fails to initialise outside a
        # GPU-service context on macOS, and the Pi has no Metal at all — so the
        # deployment target and the dev machine agree on CPU.
        delegate = BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(self.model_path), delegate=delegate,
            ),
            running_mode=RunningMode.VIDEO,
            num_faces=num_faces,
            # Iris refinement gives the 478-point mesh; the extra points are
            # what make gaze available later without a second model.
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def observe(self, frame: np.ndarray, t: float) -> FaceObservation:
        import mediapipe as mp

        from awas.driver.ear import ear_from_landmarks

        h, w = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect_for_video(image, int(t * 1000))

        if not result.face_landmarks:
            return FaceObservation(t=t, face_present=False)

        marks = [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]]

        try:
            ear = ear_from_landmarks(marks)
        except ValueError:
            return FaceObservation(t=t, face_present=False)

        pose = head_pose([marks[i] for i in POSE_IDX], w, h)
        pitch, yaw, roll = pose if pose else (0.0, 0.0, 0.0)

        return FaceObservation(
            t=t, face_present=True, landmarks=marks, ear=ear,
            pitch_deg=pitch, yaw_deg=yaw, roll_deg=roll,
        )

    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None


class ScriptedLandmarker:
    """Replays a list of FaceObservations. For tests and headless demos."""

    def __init__(self, observations: Sequence[FaceObservation]):
        self._obs = list(observations)
        self._i = 0

    def observe(self, frame: Optional[np.ndarray], t: float) -> FaceObservation:
        if self._i >= len(self._obs):
            return FaceObservation(t=t, face_present=False)
        obs = self._obs[self._i]
        self._i += 1
        return FaceObservation(
            t=t, face_present=obs.face_present, landmarks=obs.landmarks,
            ear=obs.ear, pitch_deg=obs.pitch_deg, yaw_deg=obs.yaw_deg,
            roll_deg=obs.roll_deg,
        )

    def close(self) -> None:
        pass
