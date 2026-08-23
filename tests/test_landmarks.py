"""Head-pose geometry. Verified by projecting a known rotation and recovering it."""

import math

import numpy as np
import pytest

from awas.driver.landmarks import (
    MODEL_POINTS,
    FaceObservation,
    ScriptedLandmarker,
    camera_matrix,
    head_pose,
)

W, H = 640, 480


def project(pitch, yaw, roll, distance=1200.0):
    """Render the canonical face at a known orientation."""
    import cv2

    def rot(axis, deg):
        v = np.zeros((3, 1))
        v[axis] = math.radians(deg)
        return cv2.Rodrigues(v)[0]

    R = rot(2, roll) @ rot(1, yaw) @ rot(0, pitch)
    rvec, _ = cv2.Rodrigues(R)
    pts, _ = cv2.projectPoints(
        MODEL_POINTS, rvec, np.array([[0.0], [0.0], [distance]]),
        camera_matrix(W, H), np.zeros((4, 1)),
    )
    return [tuple(p[0]) for p in pts]


class TestCameraMatrix:
    def test_principal_point_is_centred(self):
        k = camera_matrix(W, H)
        assert k[0, 2] == W / 2 and k[1, 2] == H / 2

    def test_focal_length_tracks_width(self):
        assert camera_matrix(800, 600)[0, 0] == 800


class TestHeadPose:
    @pytest.mark.parametrize("pitch,yaw,roll", [
        (0, 0, 0), (15, 0, 0), (-20, 0, 0),
        (0, 25, 0), (0, -25, 0), (0, 0, 12),
        (-10, -15, 5), (12, 18, -8),
    ])
    def test_recovers_the_angle_it_was_given(self, pitch, yaw, roll):
        got = head_pose(project(pitch, yaw, roll), W, H)
        assert got is not None
        assert got[0] == pytest.approx(pitch, abs=1.5)
        assert got[1] == pytest.approx(yaw, abs=1.5)
        assert got[2] == pytest.approx(roll, abs=1.5)

    def test_level_head_reads_near_zero_not_near_180(self):
        """RQDecomp3x3 reports pitch near +/-180 for a forward head; we fold it."""
        pitch, _, _ = head_pose(project(0, 0, 0), W, H)
        assert abs(pitch) < 5.0

    def test_head_down_is_negative_or_positive_but_large(self):
        pitch, _, _ = head_pose(project(-30, 0, 0), W, H)
        assert pitch == pytest.approx(-30, abs=2.0)

    def test_rejects_the_wrong_number_of_points(self):
        with pytest.raises(ValueError, match="6 points"):
            head_pose([(0.0, 0.0)] * 4, W, H)


class TestScriptedLandmarker:
    def test_replays_in_order(self):
        obs = [
            FaceObservation(t=0.0, face_present=True, ear=0.30),
            FaceObservation(t=0.0, face_present=True, ear=0.10),
        ]
        lm = ScriptedLandmarker(obs)
        assert lm.observe(None, 0.0).ear == 0.30
        assert lm.observe(None, 1.0).ear == 0.10

    def test_runs_out_as_an_absent_face(self):
        lm = ScriptedLandmarker([FaceObservation(t=0.0, face_present=True, ear=0.3)])
        lm.observe(None, 0.0)
        assert lm.observe(None, 1.0).face_present is False

    def test_stamps_the_caller_timestamp(self):
        lm = ScriptedLandmarker([FaceObservation(t=99.0, face_present=True, ear=0.3)])
        assert lm.observe(None, 4.0).t == 4.0
