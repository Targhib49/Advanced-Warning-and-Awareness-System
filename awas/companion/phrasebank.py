"""Every line AWAS can say, as text (plan v3 §4.4).

All of it is PRE-RENDERED to audio before the system ever runs — see
`scripts/render_phrases.py`. At runtime the executor selects a clip by id and
plays bytes already sitting in memory. Nothing is synthesised while driving.

That is invariant I1 made structural rather than aspirational: there is no code
path from a generator to the speaker, so no future change can accidentally put
one between a level-4 driver and their warning.

The texts are English for Phase 1 because the demo and the judging are. A
shipping unit would carry Indonesian; the bank is the only file that changes,
and `render_phrases.py --voice` re-renders it.
"""

from __future__ import annotations

from typing import Dict, List

from awas.core.actions import Intent

# --- Reflex alerts. Short, imperative, understandable at a glance of hearing.
# Keyed by the alert_id the state machine emits.
ALERTS: Dict[str, str] = {
    "critical_drowsy":          "Wake up. Pull over now.",
    "phone_in_hand":            "Eyes on the road.",
    "lane_departure":           "Check your lane.",
    "green_light_unresponsive": "Light is green.",
    "traffic_moving":           "Traffic is moving.",
    "monitoring_lost":          "AWAS is not monitoring.",
}

# Alerts that get a rising alarm tone in front of the words. Reserved for the
# two cases where a driver may be too far gone to parse speech immediately.
ALERTS_WITH_TONE = {"critical_drowsy", "lane_departure"}

# --- Conversational templates, several per intent.
# Several because a driver who hears the same sentence every time stops hearing
# it at all — habituation is the documented failure of conventional alarms and
# the thing AWAS exists to avoid. The dialogue manager never repeats the line
# it used last for a given intent.
TEMPLATES: Dict[Intent, List[str]] = {
    Intent.CHECK_IN: [
        "Still doing all right up there?",
        "You have been driving a while. How are you feeling?",
        "How is the road looking?",
    ],
    Intent.ENGAGE: [
        "Talk to me a moment. Where are you headed tonight?",
        "Do me a favour and tell me the next town on your route.",
        "How much further have you got to go?",
    ],
    Intent.INSIST: [
        "You are not safe to keep driving. Find somewhere to stop.",
        "This is serious now. Pull over at the next opportunity.",
        "Stop driving and rest. It can wait.",
    ],
    Intent.CONFIRM_ALERT: [
        "Are you with me?",
        "Say something so I know you are awake.",
    ],
}


def alert_clip_id(alert_id: str) -> str:
    return f"alert__{alert_id}"


def template_clip_id(intent: Intent, index: int) -> str:
    return f"say__{intent.value}__{index}"


def all_clips() -> Dict[str, str]:
    """Every clip id mapped to the text it should be rendered from."""
    clips = {alert_clip_id(k): v for k, v in ALERTS.items()}
    for intent, options in TEMPLATES.items():
        for i, text in enumerate(options):
            clips[template_clip_id(intent, i)] = text
    return clips


def text_for(clip_id: str) -> str:
    """The words behind a clip id — for logs and for the debug overlay."""
    return all_clips().get(clip_id, "")
