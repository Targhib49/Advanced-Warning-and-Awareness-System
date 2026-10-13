"""Bench drive 5: a talkative recovery, and a head drop nobody mentioned."""

from awas.companion.phrasebank import ALERTS, ALERTS_WITH_TONE
from awas.core.actions import ActionType
from awas.core.machine import StateMachine
from awas.core.states import DrivingState
from awas.core.telemetry import Telemetry


def _tick(sm, t, **kw):
    args = dict(timestamp=t, ignition_on=True, speed_kmh=60.0, drowsiness_level=0,
                confidence=0.9, face_present=True)
    args.update(kw)
    return sm.update(Telemetry(**args))


def _spoken(actions):
    return [a for a in actions if a.type == ActionType.SPEAK]


def _alerts(actions):
    return [a.alert_id for a in actions if a.type == ActionType.ALERT]


# ---------------------------------------------------------------- recovery

def test_the_way_down_from_critical_is_quiet():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    for t in range(0, 70):
        _tick(sm, float(t), drowsiness_level=4)
    said = []
    # The hysteresis steps down one level about every 10 s.
    for t, level in ((70, 3), (80, 2), (90, 1), (100, 0)):
        for dt in range(10):
            said += _spoken(_tick(sm, float(t + dt), drowsiness_level=level))
    assert said == []


def test_a_level_that_holds_on_the_way_down_still_speaks():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    _tick(sm, 0.0, drowsiness_level=4)
    said = []
    for t in range(1, 60):
        said += _spoken(_tick(sm, float(t), drowsiness_level=3))
    # Quiet for level 3's cooldown after the fall, then a still-drowsy
    # driver hears about it.
    assert [a.urgency for a in said] == [3]


def test_a_rising_level_still_speaks_at_once():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    _tick(sm, 0.0, drowsiness_level=0)
    assert _spoken(_tick(sm, 1.0, drowsiness_level=2))


# ---------------------------------------------------------------- face lost

def test_a_face_gone_while_moving_is_warned_of_once():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    _tick(sm, 0.0)
    fired = []
    for t in range(1, 30):
        fired += _alerts(_tick(sm, float(t), face_present=False))
    assert fired == ["face_not_seen"]
    assert "face_not_seen" in ALERTS and "face_not_seen" in ALERTS_WITH_TONE


def test_a_brief_glance_away_is_not():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    _tick(sm, 0.0)
    fired = []
    for t in range(1, 5):
        fired += _alerts(_tick(sm, float(t), face_present=False))
    fired += _alerts(_tick(sm, 5.0))
    assert fired == []


def test_each_new_disappearance_is_warned_of_again():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    _tick(sm, 0.0)
    fired = []
    for t in range(1, 10):
        fired += _alerts(_tick(sm, float(t), face_present=False))
    fired += _alerts(_tick(sm, 10.0))                                   # back
    for t in range(11, 20):
        fired += _alerts(_tick(sm, float(t), face_present=False))
    assert fired == ["face_not_seen", "face_not_seen"]


def test_looking_back_to_reverse_does_not_count():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    _tick(sm, 0.0)
    fired = []
    for t in range(1, 10):
        fired += _alerts(_tick(sm, float(t), face_present=False, reversing=True, speed_kmh=6.0))
    # The reverse hold (5 s) then the warn time (5 s) from the end of it.
    for t in range(10, 19):
        fired += _alerts(_tick(sm, float(t), face_present=False))
    assert fired == []
    assert _alerts(_tick(sm, 20.0, face_present=False)) == ["face_not_seen"]


def test_parked_says_nothing():
    sm = StateMachine(initial_state=DrivingState.PARKED)
    fired = []
    for t in range(0, 20):
        fired += _alerts(_tick(sm, float(t), face_present=False, speed_kmh=0.0))
    assert fired == []


# ---------------------------------------------------------------- precedence

def test_reaching_critical_is_announced_even_just_after_another_alert():
    # Bench drive 6: a lane alert 4.9 s earlier held the alarm back by the
    # shared alert cooldown, and level 4 was then treated as already announced.
    sm = StateMachine(initial_state=DrivingState.MOVING)
    assert _alerts(_tick(sm, 0.0, lane_departure=True)) == ["lane_departure"]
    assert "critical_drowsy" not in _alerts(_tick(sm, 4.9, drowsiness_level=4))
    assert "critical_drowsy" in _alerts(_tick(sm, 5.1, drowsiness_level=4))
