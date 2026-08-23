"""State machine tests — fast, deterministic, no I/O.

    python -m pytest tests/ -v
"""

import pytest

from awas.core.actions import ActionType, Intent
from awas.core.config import StateMachineConfig
from awas.core.machine import StateMachine
from awas.core.states import DrivingState
from awas.core.telemetry import Distraction, Telemetry


def drive(sm, **kwargs):
    """Build Telemetry from kwargs, feed it to the machine, return actions."""
    return sm.update(Telemetry(**kwargs))


def has_type(actions, action_type):
    return any(a.type == action_type for a in actions)


def of_type(actions, action_type):
    return [a for a in actions if a.type == action_type]


def moving(sm, t=0.0):
    """Put the machine into MOVING and return it."""
    drive(sm, timestamp=t, ignition_on=True, speed_kmh=30.0)
    return sm


# ---------------- state transitions ----------------

class TestBasicTransitions:
    def test_parked_to_moving(self):
        sm = StateMachine()
        assert sm.state == DrivingState.PARKED
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0)
        assert sm.state == DrivingState.MOVING

    def test_parked_stays_parked_with_brake(self):
        sm = StateMachine()
        drive(sm, timestamp=1.0, ignition_on=True, parking_brake=True)
        assert sm.state == DrivingState.PARKED

    def test_moving_to_stopped_at_light_on_red(self):
        sm = moving(StateMachine())
        drive(sm, timestamp=2.0, ignition_on=True, speed_kmh=2.0,
              traffic_light_color="red")
        assert sm.state == DrivingState.STOPPED_AT_LIGHT

    def test_stopped_at_light_to_moving_on_acceleration(self):
        sm = StateMachine(initial_state=DrivingState.STOPPED_AT_LIGHT)
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=20.0,
              traffic_light_color="green")
        assert sm.state == DrivingState.MOVING

    def test_moving_to_traffic_jam_after_30s_stopped(self):
        sm = moving(StateMachine())
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=0.0)
        assert sm.state == DrivingState.MOVING     # not yet
        drive(sm, timestamp=29.0, ignition_on=True, speed_kmh=0.0)
        assert sm.state == DrivingState.MOVING     # still not
        drive(sm, timestamp=32.0, ignition_on=True, speed_kmh=0.0)
        assert sm.state == DrivingState.TRAFFIC_JAM

    def test_light_becomes_jam_after_two_minutes(self):
        sm = StateMachine(initial_state=DrivingState.STOPPED_AT_LIGHT)
        drive(sm, timestamp=10.0, ignition_on=True, speed_kmh=0.0,
              traffic_light_color="red")
        drive(sm, timestamp=130.0, ignition_on=True, speed_kmh=0.0,
              traffic_light_color="red")
        assert sm.state == DrivingState.TRAFFIC_JAM

    def test_traffic_jam_to_moving_on_acceleration(self):
        sm = StateMachine(initial_state=DrivingState.TRAFFIC_JAM)
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=20.0)
        assert sm.state == DrivingState.MOVING

    def test_moving_to_parked_on_brake(self):
        sm = moving(StateMachine())
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=0.0,
              parking_brake=True)
        assert sm.state == DrivingState.PARKED


class TestShutdown:
    def test_ignition_off_30s_triggers_shutdown(self):
        sm = moving(StateMachine())
        drive(sm, timestamp=10.0, ignition_on=False)
        assert sm.state == DrivingState.MOVING     # not yet
        drive(sm, timestamp=45.0, ignition_on=False)
        assert sm.state == DrivingState.SHUTDOWN

    def test_shutdown_to_parked_on_ignition_on(self):
        sm = StateMachine(initial_state=DrivingState.SHUTDOWN)
        drive(sm, timestamp=1.0, ignition_on=True, parking_brake=True)
        assert sm.state == DrivingState.PARKED


class TestNoObdAvailable:
    """plan §7.4 — parking_brake is None when OBD-II is absent."""

    def test_can_still_reach_moving_without_obd(self):
        sm = StateMachine()
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
              parking_brake=None)
        assert sm.state == DrivingState.MOVING

    def test_no_obd_does_not_park_the_vehicle(self):
        sm = moving(StateMachine())
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=0.0,
              parking_brake=None)
        assert sm.state == DrivingState.MOVING


# ---------------- escalation ladder (plan §5.1) ----------------

class TestEscalationLadder:
    @pytest.mark.parametrize("level,intent", [
        (1, Intent.CHECK_IN),
        (2, Intent.ENGAGE),
        (3, Intent.INSIST),
    ])
    def test_level_maps_to_intent(self, level, intent):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=level)
        speak = of_type(actions, ActionType.SPEAK)
        assert len(speak) == 1
        assert speak[0].intent is intent
        assert speak[0].urgency == level

    def test_level_1_does_not_expect_a_reply(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=1)
        assert of_type(actions, ActionType.SPEAK)[0].expects_reply is False

    def test_level_2_expects_a_reply(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=2)
        assert of_type(actions, ActionType.SPEAK)[0].expects_reply is True

    def test_level_4_uses_the_reflex_path(self):
        """Invariant I1: a critical driver gets pre-rendered audio, not an LLM."""
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=4)
        alerts = of_type(actions, ActionType.ALERT)
        assert len(alerts) == 1
        assert alerts[0].alert_id == "critical_drowsy"
        assert alerts[0].urgency == 4

    def test_level_0_stays_silent(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0)
        assert actions == []


class TestCooldowns:
    def test_conversational_cooldown_is_per_level(self):
        sm = moving(StateMachine())
        a1 = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                   drowsiness_level=2)
        assert has_type(a1, ActionType.SPEAK)

        # Same level a few seconds later: silent.
        a2 = drive(sm, timestamp=5.0, ignition_on=True, speed_kmh=30.0,
                   drowsiness_level=2)
        assert not has_type(a2, ActionType.SPEAK)

        # Escalating to level 3 has its own cooldown, so it speaks immediately.
        a3 = drive(sm, timestamp=6.0, ignition_on=True, speed_kmh=30.0,
                   drowsiness_level=3)
        assert has_type(a3, ActionType.SPEAK)

    def test_level_2_repeats_after_its_cooldown(self):
        sm = moving(StateMachine())
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
              drowsiness_level=2)
        late = drive(sm, timestamp=62.0, ignition_on=True, speed_kmh=30.0,
                     drowsiness_level=2)
        assert has_type(late, ActionType.SPEAK)

    def test_alert_cooldown(self):
        """Regression: alerts must not fire on every tick."""
        sm = StateMachine(initial_state=DrivingState.STOPPED_AT_LIGHT)
        kw = dict(ignition_on=True, speed_kmh=2.0,
                  traffic_light_color="green", drowsiness_level=3)
        assert has_type(drive(sm, timestamp=1.0, **kw), ActionType.ALERT)
        assert not has_type(drive(sm, timestamp=2.0, **kw), ActionType.ALERT)
        assert has_type(drive(sm, timestamp=7.0, **kw), ActionType.ALERT)


# ---------------- context gating (plan §5.2) ----------------

class TestWorkloadGating:
    def test_high_workload_suppresses_a_mild_check_in(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=1, workload=0.9)
        assert not has_type(actions, ActionType.SPEAK)

    def test_low_workload_permits_a_mild_check_in(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=1, workload=0.1)
        assert has_type(actions, ActionType.SPEAK)

    def test_critical_ignores_workload_entirely(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=4, workload=1.0)
        assert has_type(actions, ActionType.ALERT)

    def test_level_3_is_deferred_not_suppressed(self):
        """plan §5.1: severe drowsiness waits for a gap, then fires anyway."""
        sm = moving(StateMachine())
        kw = dict(ignition_on=True, speed_kmh=30.0, drowsiness_level=3,
                  workload=0.95)
        assert not has_type(drive(sm, timestamp=1.0, **kw), ActionType.SPEAK)
        assert not has_type(drive(sm, timestamp=5.0, **kw), ActionType.SPEAK)
        # Past max_defer_sec, workload no longer holds it back.
        assert has_type(drive(sm, timestamp=12.0, **kw), ActionType.SPEAK)


class TestConfidence:
    def test_low_confidence_downgrades_the_level(self):
        """plan §5.2: a level-3 we do not trust is handled as a level 2."""
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=3, confidence=0.45)
        speak = of_type(actions, ActionType.SPEAK)
        assert len(speak) == 1
        assert speak[0].intent is Intent.ENGAGE  # not INSIST

    def test_very_low_confidence_stays_silent(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=3, confidence=0.05)
        assert actions == []

    def test_full_confidence_acts_at_full_strength(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=3, confidence=1.0)
        assert of_type(actions, ActionType.SPEAK)[0].intent is Intent.INSIST


# ---------------- degradation ladder (plan §5.4) ----------------

class TestFaceLost:
    def test_face_lost_caps_the_level(self):
        """Vehicle evidence alone must not raise a level-4 critical alert."""
        sm = moving(StateMachine())
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
              face_present=True)
        actions = drive(sm, timestamp=20.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=4, face_present=False)
        assert not has_type(actions, ActionType.ALERT)
        assert of_type(actions, ActionType.SPEAK)[0].intent is Intent.INSIST

    def test_face_lost_within_grace_still_acts_at_full_strength(self):
        sm = moving(StateMachine())
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
              face_present=True)
        actions = drive(sm, timestamp=5.0, ignition_on=True, speed_kmh=30.0,
                        drowsiness_level=4, face_present=False)
        assert has_type(actions, ActionType.ALERT)


# ---------------- distraction (a separate channel, plan §4.1) ----------------

class TestDistraction:
    def test_phone_emits_a_reflex_alert(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        distraction=Distraction.PHONE)
        alerts = of_type(actions, ActionType.ALERT)
        assert len(alerts) == 1
        assert alerts[0].alert_id == "phone_in_hand"

    def test_gaze_off_road_does_not_alert_on_its_own(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        distraction=Distraction.GAZE_OFF_ROAD)
        assert not has_type(actions, ActionType.ALERT)

    def test_lane_departure_alerts(self):
        sm = moving(StateMachine())
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=30.0,
                        lane_departure=True)
        assert of_type(actions, ActionType.ALERT)[0].alert_id == "lane_departure"


# ---------------- quiet states ----------------

class TestQuietStates:
    @pytest.mark.parametrize("state", [DrivingState.PARKED, DrivingState.SHUTDOWN])
    def test_no_conversation_when_not_driving(self, state):
        sm = StateMachine(initial_state=state)
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=0.0,
                        parking_brake=True, drowsiness_level=4)
        assert not has_type(actions, ActionType.SPEAK)
        assert not has_type(actions, ActionType.ALERT)


class TestTrafficJamNudge:
    def test_nudge_when_the_gap_opens(self):
        sm = StateMachine(initial_state=DrivingState.TRAFFIC_JAM)
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=0.0,
                        vehicle_ahead_distance_m=15.0)
        assert of_type(actions, ActionType.ALERT)[0].alert_id == "traffic_moving"

    def test_no_nudge_while_the_gap_is_closed(self):
        sm = StateMachine(initial_state=DrivingState.TRAFFIC_JAM)
        actions = drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=0.0,
                        vehicle_ahead_distance_m=3.0)
        assert not has_type(actions, ActionType.ALERT)

    def test_nudge_is_rate_limited(self):
        """Regression: the jam nudge once fired on every single tick."""
        sm = StateMachine(initial_state=DrivingState.TRAFFIC_JAM)
        kw = dict(ignition_on=True, speed_kmh=0.0, vehicle_ahead_distance_m=15.0)
        assert has_type(drive(sm, timestamp=1.0, **kw), ActionType.ALERT)
        assert not has_type(drive(sm, timestamp=2.0, **kw), ActionType.ALERT)
        assert not has_type(drive(sm, timestamp=10.0, **kw), ActionType.ALERT)
        assert has_type(drive(sm, timestamp=25.0, **kw), ActionType.ALERT)


class TestConfigurability:
    def test_thresholds_come_from_config(self):
        cfg = StateMachineConfig(traffic_jam_after_sec=5.0)
        sm = StateMachine(config=cfg)
        moving(sm)
        drive(sm, timestamp=1.0, ignition_on=True, speed_kmh=0.0)
        assert sm.state == DrivingState.MOVING     # timer just started
        drive(sm, timestamp=7.0, ignition_on=True, speed_kmh=0.0)
        assert sm.state == DrivingState.TRAFFIC_JAM
