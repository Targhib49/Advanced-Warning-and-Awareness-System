"""Performs the actions the state machine decides on (plan §5, invariant I3).

The state machine is pure and decides; this is the only place that touches a
speaker or a disk. Two rules it enforces that the state machine cannot:

  * **Urgency wins over recency.** The audio layer interrupts by design, since
    a newer decision rests on newer evidence. But a level-1 check-in must never
    cut off a level-4 warning that is still sounding, so a lower-urgency action
    arriving mid-playback is dropped and the drop is recorded. Without this the
    cooldowns could still let a pleasantry truncate an emergency.

  * **Nothing happens unlogged.** Every action is written with the telemetry
    that caused it and what actually came out of the speaker, including the
    ones that were suppressed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from awas.companion.dialogue import DialogueManager
from awas.companion.phrasebank import text_for
from awas.core.actions import Action, ActionType
from awas.speech.audio import AudioOut, NullAudioOut, Playback


@dataclass
class Executed:
    """One action, and what became of it."""

    action: Action
    clips: List[str] = field(default_factory=list)
    playback: Optional[Playback] = None
    text: str = ""
    suppressed: str = ""      # empty when it was performed

    @property
    def performed(self) -> bool:
        return not self.suppressed and self.playback is not None


class ActionExecutor:
    def __init__(
        self,
        audio: Optional[AudioOut] = None,
        dialogue: Optional[DialogueManager] = None,
        log: Any = None,
        muted: bool = False,
    ):
        self.audio = audio or NullAudioOut()
        self.dialogue = dialogue or DialogueManager()
        self.log = log
        self.muted = muted

        self._busy_until = 0.0
        self._busy_urgency = -1
        self.history: List[Executed] = []

    def execute(
        self,
        actions: List[Action],
        t: float,
        telemetry: Any = None,
        driving_state: str = "",
    ) -> List[Executed]:
        out: List[Executed] = []
        for action in actions:
            out.append(self._one(action, t, telemetry, driving_state))
        return out

    # ---------------- one action ----------------

    def _one(self, action: Action, t: float, telemetry, driving_state: str) -> Executed:
        if action.type is ActionType.LOG:
            ex = Executed(action=action, text=action.message)
            self._record(ex, t, telemetry, driving_state)
            return ex

        clips = self.dialogue.clips_for(action)
        text = " ".join(text_for(c) for c in clips if text_for(c)).strip()
        ex = Executed(action=action, clips=clips, text=text)

        if not clips:
            ex.suppressed = "no clip for this action"
        elif self.muted:
            ex.suppressed = "muted"
        elif self._would_interrupt_something_louder(action):
            ex.suppressed = f"urgency {action.urgency} < playing {self._busy_urgency}"
        else:
            ex.playback = self.audio.play(clips)
            if ex.playback.played:
                now = time.monotonic()
                self._busy_until = now + ex.playback.duration_ms / 1000.0
                self._busy_urgency = action.urgency
            else:
                ex.suppressed = "clip missing from the bank"

        self._record(ex, t, telemetry, driving_state)
        self.history.append(ex)
        return ex

    def _would_interrupt_something_louder(self, action: Action) -> bool:
        if time.monotonic() >= self._busy_until:
            return False
        return action.urgency < self._busy_urgency

    # ---------------- logging ----------------

    def _record(self, ex: Executed, t: float, telemetry, driving_state: str) -> None:
        if self.log is None:
            return
        trigger = {}
        if telemetry is not None:
            trigger = {
                "level": getattr(telemetry, "drowsiness_level", None),
                "confidence": round(getattr(telemetry, "confidence", 0.0), 4),
                "perclos": round(getattr(telemetry, "perclos", 0.0), 4),
                "microsleeps": getattr(telemetry, "microsleep_count", None),
                "speed_kmh": round(getattr(telemetry, "speed_kmh", 0.0), 1),
                "face": getattr(telemetry, "face_present", None),
            }
        self.log.action(
            t=t, action=ex.action, clips=ex.clips, playback=ex.playback,
            driving_state=driving_state, trigger=trigger,
        )
        if ex.suppressed:
            self.log.annotation(t=t, kind="suppressed", text=ex.suppressed)

    def close(self) -> None:
        self.audio.close()
