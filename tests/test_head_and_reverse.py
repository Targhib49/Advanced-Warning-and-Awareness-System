"""A turned head is not a drooping eye, and reversing is not a lapse."""

import json

import pytest

from awas.core.actions import ActionType
from awas.core.machine import StateMachine
from awas.core.states import DrivingState
from awas.core.telemetry import Telemetry
from awas.display.model import choose_screen
from awas.driver.guided import Frame, GuidedCalibration
from awas.driver.landmarks import FaceObservation
from awas.driver.pipeline import DriverPipeline
from awas.road.lane_detect import LaneCalibration
from awas.road.sim import MAGIC, SimConfig, SimRoadSource

FPS = 15


def _obs(t, yaw=0.0, pitch=0.0, p_closed=0.05, ear=0.27):
    return FaceObservation(t=t, face_present=True, ear=ear, yaw_deg=yaw, pitch_deg=pitch, p_closed=p_closed)


# ---------------------------------------------------------------- head turned

def test_a_long_look_over_the_shoulder_is_not_drowsiness():
    pipe = DriverPipeline(eye_signal="cnn", neutral_pose=(0.0, 0.0))
    s = None
    for i in range(10 * FPS):
        # Head turned 60 degrees: the eye crop reads "closed" — it is in profile.
        s = pipe.process(_obs(i / FPS, yaw=60.0, p_closed=0.95, ear=0.12), i / FPS)
    assert s.face_present and not s.eyes_observable
    assert s.perclos == 0.0 and s.longest_closure_ms == 0.0


def test_the_same_closed_eyes_facing_the_road_do_count():
    pipe = DriverPipeline(eye_signal="cnn", neutral_pose=(0.0, 0.0))
    s = None
    for i in range(10 * FPS):
        s = pipe.process(_obs(i / FPS, yaw=0.0, p_closed=0.95, ear=0.12), i / FPS)
    assert s.eyes_observable and s.perclos > 0.8


def test_limits_are_measured_from_the_drivers_own_road_facing_pose():
    # Camera left of the wheel: looking at the road reads as yaw -25.
    pipe = DriverPipeline(eye_signal="cnn", neutral_pose=(-25.0, 8.0))
    assert pipe.eyes_observable(_obs(0.0, yaw=-25.0, pitch=8.0))
    assert pipe.eyes_observable(_obs(0.0, yaw=-5.0, pitch=8.0))       # 20 deg off: still fine
    assert not pipe.eyes_observable(_obs(0.0, yaw=15.0, pitch=8.0))   # 40 deg off: a mirror check
    assert not pipe.eyes_observable(_obs(0.0, yaw=-25.0, pitch=35.0)) # looking down at the dash


def test_without_a_calibrated_neutral_no_frame_is_excluded_for_pose():
    # A wrong reference throws away exactly the frames that matter (bench drive 1).
    pipe = DriverPipeline(eye_signal="cnn")
    assert pipe.eyes_observable(_obs(0.0, yaw=-55.0, pitch=10.0))
    s = None
    for i in range(10 * FPS):
        s = pipe.process(_obs(i / FPS, yaw=-55.0, p_closed=0.95, ear=0.12), i / FPS)
    assert s.perclos > 0.8


def test_mediapipe_matrix_pose_is_continuous_and_signed():
    import math
    import numpy as np
    from awas.driver.landmarks import pose_from_matrix

    def rot_y(deg):
        a = math.radians(deg)
        m = np.eye(4)
        m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
        return m

    assert pose_from_matrix(np.eye(4)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    yaws = [pose_from_matrix(rot_y(d))[1] for d in range(-70, 71, 10)]
    assert yaws == pytest.approx(list(range(-70, 71, 10)), abs=1e-6)   # no flips, no jumps


def test_calibration_summary_names_its_pose_method():
    from awas.driver.guided import POSE_METHOD, GuidedResult
    assert GuidedResult(ok=True).summary()["pose_method"] == POSE_METHOD


def test_guided_calibration_accepts_an_off_axis_camera_and_records_neutral():
    cal = GuidedCalibration()
    t, frames = 0.0, []
    for sec, ear, p in ((2.5, .26, .05), (9.0, .26, .05), (2.0, .15, .93), (1.0, .26, .05),
                        (2.0, .15, .93), (1.0, .26, .05), (7.5, .22, .95), (7.5, .16, .30)):
        for i in range(int(sec * FPS)):
            frames.append(Frame(t=t + i / FPS, face=True, ear=ear, p_closed=p, eye_px=30,
                                yaw=-50.0, pitch=7.0, brightness=120))
        t += int(sec * FPS) / FPS
    for f in frames:
        cal.feed(f)
    r = cal.result
    assert r.ok and r.neutral_yaw == pytest.approx(-50.0) and r.neutral_pitch == pytest.approx(7.0)


# ---------------------------------------------------------------- reversing

def _moving(sm, t, **kw):
    args = dict(timestamp=t, ignition_on=True, speed_kmh=30.0, drowsiness_level=4, confidence=0.9)
    args.update(kw)
    return sm.update(Telemetry(**args))


def test_reversing_silences_even_a_critical_level():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    assert any(a.type == ActionType.ALERT for a in _moving(sm, 0.0))     # forward: it fires
    sm = StateMachine(initial_state=DrivingState.MOVING)
    assert _moving(sm, 0.0, reversing=True) == []
    assert sm.manoeuvring(Telemetry(timestamp=0.0))


def test_the_pause_holds_briefly_after_reversing_then_ends():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    _moving(sm, 0.0, reversing=True)
    assert _moving(sm, 4.0) == []                                        # still within the hold
    assert any(a.type == ActionType.ALERT for a in _moving(sm, 6.0))     # past it: back on duty


def test_no_gear_information_never_pauses_anything():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    assert not sm.manoeuvring(Telemetry(timestamp=0.0))


def test_dashboard_says_alerts_are_paused():
    assert choose_screen(DrivingState.MOVING, 3, True, False, manoeuvring=True) == "reversing"
    assert choose_screen(DrivingState.MOVING, 3, True, False, camera_ok=False, manoeuvring=True) == "nocamera"


# ---------------------------------------------------------------- simulator signals

@pytest.fixture
def sim():
    s = SimRoadSource(SimConfig(port=0), calibration=LaneCalibration(), bind="127.0.0.1")
    yield s
    s.close()


def _tel(**kw):
    d = {"speed_kmh": 8.0, "steer": 0.0, "engine_on": True}
    d.update(kw)
    return MAGIC + b"T" + json.dumps(d).encode()


def test_negative_gear_is_reversing(sim):
    sim.handle(_tel(gear=-1))
    assert sim.read(1.0).reversing is True
    sim.handle(_tel(gear=3))
    assert sim.read(1.1).reversing is False


def test_an_older_bridge_without_gear_says_unknown(sim):
    sim.handle(_tel())
    assert sim.read(1.0).reversing is None


@pytest.mark.parametrize("left,right,want", [(True, False, "left"), (False, True, "right"),
                                             (False, False, None), (True, True, None)])
def test_turn_signal_and_hazards(sim, left, right, want):
    sim.handle(_tel(gear=5, blinker_left=left, blinker_right=right))
    assert sim.read(1.0).turn_signal == want


def test_the_forward_view_is_ignored_while_reversing(sim):
    import cv2
    from awas.road.lane_detect import synthetic_road
    ok, buf = cv2.imencode(".jpg", synthetic_road(offset_m=0.2))
    sim.handle(_tel(gear=-1))
    sim.handle(MAGIC + b"F" + buf.tobytes())
    assert sim.read(1.0).lane_offset_m is None and sim.lane_tried == 0


# ---------------------------------------------------------------- the critical alarm answers events

def _count(actions, kind):
    return sum(1 for a in actions if a.type == kind)


def _drive_critical(sm, t, micro, level=4):
    return sm.update(Telemetry(timestamp=t, ignition_on=True, speed_kmh=60.0,
                               drowsiness_level=level, confidence=0.9, microsleep_count=micro))


def test_one_microsleep_held_in_the_window_alarms_once():
    # Bench drive 3: one 2 s closure, then a minute of level 4 from memory alone.
    sm = StateMachine(initial_state=DrivingState.MOVING)
    alarms = speech = 0
    for i in range(60):
        acts = _drive_critical(sm, float(i), micro=1)
        alarms += _count(acts, ActionType.ALERT)
        speech += _count(acts, ActionType.SPEAK)
    assert alarms == 1                       # was ~12 on a 5 s cooldown
    assert speech == 1                       # the calm reminder comes at 60 s, not every 10 s


def test_a_new_microsleep_alarms_again():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    alarms = []
    for i in range(30):
        micro = 1 if i < 20 else 2                 # a second microsleep at t=20
        if _count(_drive_critical(sm, float(i), micro), ActionType.ALERT):
            alarms.append(i)
    assert alarms == [0, 20]


def test_the_reminder_repeats_while_the_level_holds():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    spoken = [i for i in range(130)
              if _count(_drive_critical(sm, float(i), micro=1), ActionType.SPEAK)]
    assert spoken == [0, 60, 120]


def test_leaving_level_4_and_returning_is_a_new_event():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    assert _count(_drive_critical(sm, 0.0, micro=1), ActionType.ALERT) == 1
    _drive_critical(sm, 10.0, micro=0, level=2)
    assert _count(_drive_critical(sm, 20.0, micro=0), ActionType.ALERT) == 1
