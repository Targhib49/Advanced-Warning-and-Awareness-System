"""Tunable thresholds for the state machine.

Kept in one dataclass so they can be loaded from config/default.yaml and, more
importantly, *fitted* on data rather than asserted (plan §4.1).
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class StateMachineConfig:
    # --- speed / state transitions ---
    stopped_speed_kmh: float = 5.0
    moving_speed_kmh: float = 5.0
    traffic_jam_after_sec: float = 30.0
    light_to_jam_sec: float = 120.0
    shutdown_after_off_sec: float = 30.0
    # Manoeuvring: reversing, and this long after, the driver is looking back
    # on purpose, so face-based interventions pause. UNFITTED.
    reverse_hold_sec: float = 5.0

    # --- cooldowns (plan §5.2) ---
    alert_cooldown_sec: float = 5.0
    jam_nudge_cooldown_sec: float = 20.0
    # Per-level conversational cooldown: higher urgency may repeat sooner.
    # Level 4 was 10 s; with the alarm now event-driven, its spoken "pull
    # over" is the periodic reminder while the level holds, and every 10 s for
    # a minute is nagging, not safety. UNFITTED.
    intervention_cooldown_sec: Dict[int, float] = field(
        default_factory=lambda: {1: 90.0, 2: 60.0, 3: 30.0, 4: 60.0}
    )

    # --- context gating (plan §5.2) ---
    # An intervention at level N fires only if workload < ceiling[N].
    workload_ceiling: Dict[int, float] = field(
        default_factory=lambda: {1: 0.4, 2: 0.7, 3: 0.85, 4: 1.01}
    )
    # Level 3+ is deferred, never suppressed: it fires anyway after this long.
    max_defer_sec: float = 10.0

    # A level whose confidence is below its floor is downgraded, not acted on
    # at full strength.
    confidence_floor: Dict[int, float] = field(
        default_factory=lambda: {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6}
    )

    # --- degradation ladder (plan §5.4) ---
    face_lost_grace_sec: float = 10.0
    face_lost_level_cap: int = 3
    # Face gone this long while moving: one spoken warning per disappearance.
    # A head dropped onto the chest takes the face out of view, and that is
    # the nod-off the eye signals then cannot see (bench drive 5). UNFITTED.
    face_lost_warn_sec: float = 5.0

    # Eyes that STAY shut keep the alarm going: past this closure, it repeats
    # every `closed_alarm_repeat_sec` until they open (an alarm clip is ~2.4 s,
    # so this is near-continuous). One 10 s closure used to get one alarm.
    # UNFITTED.
    closed_alarm_after_ms: float = 2000.0
    closed_alarm_repeat_sec: float = 3.0

    # --- jam nudge ---
    jam_gap_open_m: float = 8.0
