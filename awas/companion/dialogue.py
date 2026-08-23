"""Choosing what to say (plan v3 §4.4, the L3b path).

Phase 1 selects from the pre-rendered bank. When the LLM lands in Milestone 4
it slots in *above* this — generating a variant, with this selection as the
fallback when the model misses its budget or fails the safety filter. The
interface does not change, which is the point of building it this way now.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from awas.companion.phrasebank import (
    ALERTS,
    ALERTS_WITH_TONE,
    TEMPLATES,
    alert_clip_id,
    template_clip_id,
)
from awas.core.actions import Action, ActionType, Intent

TONE_CLIP = "tone__alarm"


class DialogueManager:
    """Turns an Action into the clip sequence that should be played."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._last: Dict[Intent, int] = {}

    def clips_for(self, action: Action) -> List[str]:
        if action.type is ActionType.ALERT:
            return self._alert_clips(action.alert_id)
        if action.type is ActionType.SPEAK:
            return self._speak_clips(action.intent)
        return []

    # ---------------- reflex ----------------

    def _alert_clips(self, alert_id: str) -> List[str]:
        if alert_id not in ALERTS:
            return []
        clips = [alert_clip_id(alert_id)]
        if alert_id in ALERTS_WITH_TONE:
            # Tone first: at level 4 a driver may not parse words immediately,
            # and a rising pitch carries urgency before language does.
            clips.insert(0, TONE_CLIP)
        return clips

    # ---------------- dialogue ----------------

    def _speak_clips(self, intent: Optional[Intent]) -> List[str]:
        if intent is None:
            return []
        options = TEMPLATES.get(intent)
        if not options:
            return []

        # Never the same line twice running for one intent. A driver who hears
        # an identical sentence every time stops hearing it — habituation is
        # the failure mode of conventional alarms and the thing this avoids.
        index = self._rng.randrange(len(options))
        if len(options) > 1 and index == self._last.get(intent):
            index = (index + 1) % len(options)
        self._last[intent] = index

        return [template_clip_id(intent, index)]
