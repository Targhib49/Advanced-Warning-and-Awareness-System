"""Actions emitted by the state machine for downstream execution.

Invariant I1 (plan §3): ALERT is the reflex path — pre-rendered audio selected
by `alert_id`, never generated, budget <300ms. SPEAK is the dialogue path —
template first, LLM variation optional, budget <3000ms with template fallback.
No generated text may ever be the only thing standing between a level-4 driver
and a warning.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(Enum):
    ALERT = "alert"   # reflex: pre-rendered audio, <300ms, never generated
    SPEAK = "speak"   # dialogue: template + optional LLM variation, <3000ms
    LOG = "log"       # internal only, no audio


class Intent(Enum):
    """What the dialogue manager is trying to accomplish (plan §4.4)."""

    CHECK_IN = "check_in"       # level 1: light ambient contact
    ENGAGE = "engage"           # level 2: question requiring a verbal reply
    INSIST = "insist"           # level 3: firm prompt, suggest rest
    CONFIRM_ALERT = "confirm"   # follow-up after a reflex alert


@dataclass
class Action:
    """One thing the state machine wants the executor to do."""

    type: ActionType
    urgency: int = 0                      # mirrors drowsiness level 0-4
    alert_id: str = ""                    # ALERT: which pre-rendered clip
    intent: Optional[Intent] = None       # SPEAK: what to accomplish
    context: str = ""                     # SPEAK: situation hint for template/LLM
    message: str = ""                     # LOG, and human-readable trace
    expects_reply: bool = False           # SPEAK: start the response-latency timer
