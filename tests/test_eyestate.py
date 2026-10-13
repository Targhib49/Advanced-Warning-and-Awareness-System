"""T1 in the live path: eye crops, the CNN signal, and its EAR fallback."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from awas.core.states import DrivingState
from awas.display.model import TripTracker, choose_screen, snapshot
from awas.driver.eyestate import (
    CROP_PX, DEFAULT_MODEL, MIN_EYE_PX, aligned_crop, eye_geometry,
)
from awas.driver.landmarks import FaceObservation
from awas.driver.pipeline import DriverPipeline, DriverState
from awas.fusion.core import ChannelEstimate
from awas.fusion.telemetry import fuse_channels
from awas.road.signals import LaneStats
from awas.runtime.session_log import SessionLog


# ---------------------------------------------------------------- geometry

def _eye(cx, cy, width, angle_deg, n=478):
    """Landmarks with one eye's six EAR points placed, the rest at the origin."""
    marks = [(0.0, 0.0)] * n
    a = math.radians(angle_deg)
    dx, dy = math.cos(a) * width / 2, math.sin(a) * width / 2
    from awas.driver.ear import MEDIAPIPE_LEFT_EYE as idx
    p1, p4 = (cx - dx, cy - dy), (cx + dx, cy + dy)
    lids = [(cx - dx / 3, cy - 3), (cx + dx / 3, cy - 3), (cx + dx / 3, cy + 3), (cx - dx / 3, cy + 3)]
    for i, p in zip(idx, [p1, lids[0], lids[1], p4, lids[2], lids[3]]):
        marks[i] = p
    return marks, idx


@pytest.mark.parametrize("angle", [0.0, 12.0, -20.0])
def test_geometry_recovers_width_and_roll(angle):
    marks, idx = _eye(300, 200, 60, angle)
    centre, width, a = eye_geometry(marks, idx)
    assert centre == pytest.approx((300, 200), abs=1e-6)
    assert width == pytest.approx(60)
    assert math.degrees(a) == pytest.approx(angle)


def test_geometry_does_not_turn_an_eye_upside_down():
    marks, idx = _eye(300, 200, 60, 180.0)       # corners listed right-to-left
    _, _, a = eye_geometry(marks, idx)
    assert abs(math.degrees(a)) < 1e-6


def test_crop_levels_a_tilted_eye():
    import cv2
    frame = np.full((480, 640), 200, np.uint8)
    # A dark bar 70 px long, tilted 15 degrees: a crude closed eyelid.
    a = math.radians(15)
    p0 = (int(320 - 35 * math.cos(a)), int(240 - 35 * math.sin(a)))
    p1 = (int(320 + 35 * math.cos(a)), int(240 + 35 * math.sin(a)))
    cv2.line(frame, p0, p1, 20, 5)
    crop = aligned_crop(frame, (320, 240), 70, a)
    assert crop.shape == (CROP_PX, CROP_PX) and crop.dtype == np.float32
    assert 0.0 <= crop.min() and crop.max() <= 1.0
    # Levelled: the dark line now lies along the middle row, not diagonally.
    row_means = crop.mean(axis=1)
    assert row_means.argmin() in range(CROP_PX // 2 - 2, CROP_PX // 2 + 2)
    assert row_means.min() < 0.6 * row_means.max()


def test_crop_refuses_tiny_or_offscreen_eyes():
    frame = np.zeros((480, 640), np.uint8)
    assert aligned_crop(frame, (320, 240), MIN_EYE_PX - 1, 0.0) is None
    assert aligned_crop(frame, (-10, 240), 60, 0.0) is None
    assert aligned_crop(frame, (320, 500), 60, 0.0) is None


@pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="exported eye-state model not present")
def test_classifier_returns_a_probability():
    from awas.driver.eyestate import EyeStateClassifier
    clf = EyeStateClassifier()
    rgb = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    marks, _ = _eye(300, 200, 60, 0.0)
    from awas.driver.ear import MEDIAPIPE_RIGHT_EYE
    for i, p in zip(MEDIAPIPE_RIGHT_EYE, [(380, 200), (390, 197), (400, 197), (420, 200), (400, 203), (390, 203)]):
        marks[i] = p
    p = clf.p_closed(rgb, marks)
    assert p is not None and 0.0 <= p <= 1.0


# ---------------------------------------------------------------- pipeline

def _obs(t, p_closed=None, ear=0.30, face=True):
    return FaceObservation(t=t, face_present=face, ear=ear, p_closed=p_closed)


def test_cnn_drives_openness_when_present():
    pipe = DriverPipeline(eye_signal="cnn")
    s = pipe.process(_obs(0.0, p_closed=0.9), 0.0)
    assert s.eye_source == "cnn"
    assert s.openness == pytest.approx(0.1)
    assert s.p_closed == 0.9
    assert not s.calibrating                       # population model: no wait


def test_ear_takes_over_when_the_cnn_has_nothing():
    pipe = DriverPipeline(eye_signal="cnn")
    s = pipe.process(_obs(0.0, p_closed=None), 0.0)
    assert s.eye_source == "ear"
    assert s.openness == pytest.approx(s.ear_openness)


def test_ear_control_condition_ignores_the_cnn():
    pipe = DriverPipeline(eye_signal="ear")
    s = pipe.process(_obs(0.0, p_closed=0.9), 0.0)
    assert s.eye_source == "ear" and s.openness == pytest.approx(s.ear_openness)


def test_unknown_eye_signal_is_refused():
    with pytest.raises(ValueError):
        DriverPipeline(eye_signal="vibes")


def test_closed_eyes_by_cnn_raise_perclos():
    pipe = DriverPipeline(eye_signal="cnn")
    s = None
    for i in range(300):                            # 10 s at 30 fps, eyes shut
        s = pipe.process(_obs(i / 30, p_closed=0.97), i / 30)
    assert s.perclos > 0.8


# ---------------------------------------------------------------- fusion

def _lane(sdlp):
    return LaneStats(sdlp=sdlp, samples=200, departures=0)


def test_fusion_mode_reaches_the_operator():
    driver = DriverState(t=1.0, drowsiness_level=0, confidence=0.8, face_present=True)
    weighted = fuse_channels(driver, _lane(0.30), mode="weighted")
    asymmetric = fuse_channels(driver, _lane(0.30), mode="asymmetric")
    assert asymmetric.level_continuous > weighted.level_continuous
    assert asymmetric.level_continuous == pytest.approx(3.13, abs=0.05)


def test_an_unusable_facial_window_is_not_an_opinion():
    driver = DriverState(t=1.0, drowsiness_level=3, confidence=0.8,
                         face_present=True, window_usable=False)
    fused = fuse_channels(driver, _lane(0.10))
    assert fused.sources == ["vehicle"]


def test_personal_baseline_changes_the_vehicle_verdict():
    driver = DriverState(t=1.0, face_present=False)
    absolute = fuse_channels(driver, _lane(0.20))
    personal = fuse_channels(driver, _lane(0.20), sdlp_baseline_m=0.19)
    assert absolute.level_continuous > personal.level_continuous == 0.0


# ---------------------------------------------------------------- display + log

def test_lost_face_with_a_lane_is_lane_only_not_blind():
    assert choose_screen(DrivingState.MOVING, 2, False, False, lane_available=True) == "occluded"
    assert choose_screen(DrivingState.MOVING, 2, False, False, lane_available=False) == "noface"


def test_snapshot_explains_the_fusion():
    driver = DriverState(t=1.0, face_present=False)
    fused = fuse_channels(driver, _lane(0.25))
    snap = snapshot(driving_state=DrivingState.MOVING, level=2, driver=driver, speed_kmh=80,
                    workload=0.3, lane_available=True, held_reason=None, trip=TripTracker(),
                    kss=None, sdlp_m=0.25, sdlp_baseline_m=0.18, fused=fused, now=0.0)
    assert snap["screen"] == "occluded"
    assert snap["sdlp_baseline_m"] == 0.18
    assert snap["fusion"]["explain"].startswith("vehicle L")
    json.dumps(snap)


def test_session_samples_carry_both_eye_signals_and_the_fusion(tmp_path):
    log = SessionLog(tmp_path, driver="t", sample_hz=0)
    state = DriverState(t=1.0, face_present=True, eye_source="cnn", p_closed=0.12,
                        ear_openness=0.8, drowsiness_level=1, confidence=0.7)
    fused = fuse_channels(state, _lane(0.21))
    log.sample(1.0, state, fused=fused)
    path = log.close(t=1.0)
    rec = [json.loads(l) for l in Path(path).read_text().splitlines() if '"sample"' in l][0]
    assert rec["eye_source"] == "cnn" and rec["p_closed"] == 0.12 and rec["ear_openness"] == 0.8
    assert set(rec["channels"]) == {"facial", "vehicle"}
    assert rec["fused_level"] == fused.level
