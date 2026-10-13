"""Internal bookkeeping for the state machine — timers and counters.

These are *internal*: not sensor data, not actions. Transitions like
"stopped for 30 seconds -> traffic jam" require tracking when things started.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class StateContext:
    # When we entered the current DrivingState
    state_entered_at: float = 0.0

    # When speed first dropped below the "stopped" threshold (None = moving)
    speed_dropped_at: Optional[float] = None

    # When ignition turned off (None = currently on)
    ignition_off_at: Optional[float] = None

    # Last fire time per escalation level, for per-level cooldowns (plan §5.2)
    last_intervention_at: Dict[int, float] = field(default_factory=dict)
    last_alert_at: Optional[float] = None

    # Efficacy loop (plan §5.3): what we did last and when, so the runtime can
    # measure time-to-relapse and response latency.
    last_intervention_level: Optional[int] = None
    # Set when a level-3+ intervention is held back by high workload; it
    # fires anyway once max_defer_sec has passed (plan §5.1).
    deferred_since: Optional[float] = None
    awaiting_reply_since: Optional[float] = None

    # When the face was last seen, for the degradation ladder (plan §5.4)
    face_last_seen_at: Optional[float] = None
    last_reversing_at: Optional[float] = None

    # Critical alarm is event-driven: at level 4 already, and the microsleep
    # count of the previous update (a rise is a new microsleep).
    critical_active: bool = False
    prev_microsleeps: int = 0

    # The level acted on at the previous update, to tell a falling level
    # (recovery) from a steady one; and whether this face loss was warned of.
    prev_level: int = 0
    face_lost_warned: bool = False
    last_critical_at: Optional[float] = None     # when the alarm last sounded
