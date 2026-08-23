"""State machine — pure logic, no I/O (invariant I2, plan §3).

Call `update(telemetry)` once per tick. Returns the Actions the runtime should
execute. Time arrives in the telemetry; this module never reads a clock, never
touches a device, and never calls a model.
"""

from typing import List, Optional

from .actions import Action, ActionType, Intent
from .config import StateMachineConfig
from .context import StateContext
from .states import DrivingState
from .telemetry import Distraction, Telemetry

# Which dialogue intent belongs to which escalation level (plan §5.1)
_INTENT_FOR_LEVEL = {
    1: Intent.CHECK_IN,
    2: Intent.ENGAGE,
    3: Intent.INSIST,
    4: Intent.INSIST,
}


class StateMachine:
    def __init__(
        self,
        initial_state: DrivingState = DrivingState.PARKED,
        config: Optional[StateMachineConfig] = None,
    ):
        self.state = initial_state
        self.cfg = config or StateMachineConfig()
        self.ctx = StateContext()

    # ---------------- public API ----------------

    def update(self, t: Telemetry) -> List[Action]:
        """Process one telemetry snapshot. Return actions to execute."""
        self._update_timers(t)

        next_state = self._next_state(t)
        if next_state != self.state:
            actions = self._on_exit(self.state, t)
            self.state = next_state
            self.ctx.state_entered_at = t.timestamp
            actions += self._on_enter(next_state, t)
            return actions

        return self._in_state_actions(t)

    # ---------------- effective level ----------------

    def effective_level(self, t: Telemetry) -> int:
        """The level we are willing to act on, after confidence and occlusion.

        A low-confidence level 3 is treated as a level 2 rather than acted on at
        full strength (plan §5.2). With the face lost past its grace period, the
        level is capped because only vehicle evidence remains (plan §5.4).
        """
        level = t.drowsiness_level

        while level > 0 and t.confidence < self.cfg.confidence_floor.get(level, 0.0):
            level -= 1

        if self._face_lost_for(t) > self.cfg.face_lost_grace_sec:
            level = min(level, self.cfg.face_lost_level_cap)

        return level

    # ---------------- timer bookkeeping ----------------

    def _update_timers(self, t: Telemetry) -> None:
        if t.speed_kmh < self.cfg.stopped_speed_kmh:
            if self.ctx.speed_dropped_at is None:
                self.ctx.speed_dropped_at = t.timestamp
        else:
            self.ctx.speed_dropped_at = None

        if not t.ignition_on:
            if self.ctx.ignition_off_at is None:
                self.ctx.ignition_off_at = t.timestamp
        else:
            self.ctx.ignition_off_at = None

        if t.face_present:
            self.ctx.face_last_seen_at = t.timestamp
        elif self.ctx.face_last_seen_at is None:
            self.ctx.face_last_seen_at = t.timestamp

    # ---------------- transitions ----------------

    def _next_state(self, t: Telemetry) -> DrivingState:
        # SHUTDOWN takes priority from any state
        if self.ctx.ignition_off_at is not None:
            if t.timestamp - self.ctx.ignition_off_at >= self.cfg.shutdown_after_off_sec:
                return DrivingState.SHUTDOWN

        if self.state == DrivingState.SHUTDOWN:
            if t.ignition_on:
                return DrivingState.PARKED  # warm up before MOVING

        elif self.state == DrivingState.PARKED:
            # `parking_brake is not True` so a None (no OBD-II) reading does not
            # pin us in PARKED forever — plan §7.4.
            if t.ignition_on and t.parking_brake is not True and t.speed_kmh > 0:
                return DrivingState.MOVING

        elif self.state == DrivingState.MOVING:
            if t.parking_brake is True and t.speed_kmh == 0:
                return DrivingState.PARKED
            if t.speed_kmh < self.cfg.stopped_speed_kmh:
                if t.traffic_light_color in ("red", "yellow"):
                    return DrivingState.STOPPED_AT_LIGHT
                if self._stopped_for(t) >= self.cfg.traffic_jam_after_sec:
                    return DrivingState.TRAFFIC_JAM

        elif self.state == DrivingState.STOPPED_AT_LIGHT:
            if t.speed_kmh > self.cfg.moving_speed_kmh:
                return DrivingState.MOVING
            if self._time_in_state(t) >= self.cfg.light_to_jam_sec:
                return DrivingState.TRAFFIC_JAM

        elif self.state == DrivingState.TRAFFIC_JAM:
            if t.speed_kmh > self.cfg.moving_speed_kmh:
                return DrivingState.MOVING
            if t.traffic_light_color in ("red", "yellow"):
                return DrivingState.STOPPED_AT_LIGHT

        return self.state

    # ---------------- in-state actions ----------------

    def _in_state_actions(self, t: Telemetry) -> List[Action]:
        if self.state == DrivingState.MOVING:
            return self._moving_actions(t)
        if self.state == DrivingState.STOPPED_AT_LIGHT:
            return self._light_actions(t)
        if self.state == DrivingState.TRAFFIC_JAM:
            return self._jam_actions(t)
        return []  # PARKED and SHUTDOWN stay quiet — plan §5.2

    def _moving_actions(self, t: Telemetry) -> List[Action]:
        actions: List[Action] = []
        level = self.effective_level(t)

        if level >= 4:
            # Reflex path only. Never generated, never gated by workload.
            if self._alert_cooldown_ok(t):
                actions.append(self._fire_alert(t, "critical_drowsy", urgency=4))
                # Conversation follows the alert, it does not replace it.
                if self._intervention_allowed(4, t):
                    actions.append(self._fire_speak(4, "moving_critical", t))
        elif level >= 1:
            if self._intervention_allowed(level, t):
                actions.append(self._fire_speak(level, "moving_drowsy", t))

        if t.distraction == Distraction.PHONE and self._alert_cooldown_ok(t):
            actions.append(self._fire_alert(t, "phone_in_hand", urgency=2))

        if t.lane_departure and self._alert_cooldown_ok(t):
            actions.append(self._fire_alert(t, "lane_departure", urgency=3))

        return actions

    def _light_actions(self, t: Telemetry) -> List[Action]:
        if t.traffic_light_color == "green" and self.effective_level(t) >= 3:
            if self._alert_cooldown_ok(t):
                return [self._fire_alert(t, "green_light_unresponsive", urgency=3)]
        return []

    def _jam_actions(self, t: Telemetry) -> List[Action]:
        # The gap ahead has opened but we have not moved: nudge, once, rate-limited.
        gap = t.vehicle_ahead_distance_m
        if gap is not None and gap > self.cfg.jam_gap_open_m and t.speed_kmh < 1.0:
            if self._cooldown_ok(
                self.ctx.last_alert_at, t.timestamp, self.cfg.jam_nudge_cooldown_sec
            ):
                return [self._fire_alert(t, "traffic_moving", urgency=1)]
        return []

    # ---------------- gating (plan §5.2) ----------------

    def _intervention_allowed(self, level: int, t: Telemetry) -> bool:
        """Cooldown + workload gate for the conversational path."""
        last = self.ctx.last_intervention_at.get(level)
        cooldown = self.cfg.intervention_cooldown_sec.get(level, 60.0)
        if not self._cooldown_ok(last, t.timestamp, cooldown):
            return False

        ceiling = self.cfg.workload_ceiling.get(level, 1.01)
        if t.workload < ceiling:
            self.ctx.deferred_since = None
            return True

        # Over the ceiling. Levels 1-2 are suppressed; 3+ is only deferred, and
        # fires anyway once max_defer_sec has passed (plan §5.1).
        if level < 3:
            return False
        if self.ctx.deferred_since is None:
            self.ctx.deferred_since = t.timestamp
            return False
        return t.timestamp - self.ctx.deferred_since >= self.cfg.max_defer_sec

    def _alert_cooldown_ok(self, t: Telemetry) -> bool:
        return self._cooldown_ok(
            self.ctx.last_alert_at, t.timestamp, self.cfg.alert_cooldown_sec
        )

    @staticmethod
    def _cooldown_ok(last_at: Optional[float], now: float, cooldown_sec: float) -> bool:
        """True if enough time has passed since `last_at` (or it never fired)."""
        if last_at is None:
            return True
        return now - last_at >= cooldown_sec

    # ---------------- action builders ----------------

    def _fire_alert(self, t: Telemetry, alert_id: str, urgency: int) -> Action:
        self.ctx.last_alert_at = t.timestamp
        return Action(
            type=ActionType.ALERT,
            urgency=urgency,
            alert_id=alert_id,
            message=f"reflex alert: {alert_id}",
        )

    def _fire_speak(self, level: int, context: str, t: Telemetry) -> Action:
        self.ctx.last_intervention_at[level] = t.timestamp
        self.ctx.last_intervention_level = level
        self.ctx.deferred_since = None
        intent = _INTENT_FOR_LEVEL.get(level, Intent.CHECK_IN)
        expects_reply = level >= 2
        if expects_reply:
            self.ctx.awaiting_reply_since = t.timestamp
        return Action(
            type=ActionType.SPEAK,
            urgency=level,
            intent=intent,
            context=context,
            expects_reply=expects_reply,
            message=f"dialogue: {intent.value} (level {level})",
        )

    # ---------------- enter/exit hooks ----------------

    def _on_enter(self, state: DrivingState, t: Telemetry) -> List[Action]:
        return [Action(
            type=ActionType.LOG,
            message=f"Entered {state.value} at t={t.timestamp:.2f}",
        )]

    def _on_exit(self, state: DrivingState, t: Telemetry) -> List[Action]:
        return []

    # ---------------- helpers ----------------

    def _time_in_state(self, t: Telemetry) -> float:
        return t.timestamp - self.ctx.state_entered_at

    def _stopped_for(self, t: Telemetry) -> float:
        if self.ctx.speed_dropped_at is None:
            return 0.0
        return t.timestamp - self.ctx.speed_dropped_at

    def _face_lost_for(self, t: Telemetry) -> float:
        if t.face_present or self.ctx.face_last_seen_at is None:
            return 0.0
        return t.timestamp - self.ctx.face_last_seen_at
