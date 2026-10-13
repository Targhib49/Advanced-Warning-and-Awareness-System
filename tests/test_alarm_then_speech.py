"""Bench drive 7: the alarm was cut off by its own follow-up speech, and
closures inside the alert cooldown were never announced."""

import time

from awas.core.actions import Action, ActionType, Intent
from awas.core.machine import StateMachine
from awas.core.states import DrivingState
from awas.core.telemetry import Telemetry
from awas.runtime.executor import ActionExecutor
from awas.speech.audio import Playback


class TimedAudio:
    """Every utterance lasts `sec`; records what was played, in order."""

    def __init__(self, sec=0.15):
        self.sec, self.played = sec, []

    def play(self, clip_ids):
        self.played.append(list(clip_ids))
        return Playback(clip_id="+".join(clip_ids), latency_ms=1.0, duration_ms=self.sec * 1000)

    def is_busy(self):
        return False

    def close(self):
        pass


ALARM = Action(type=ActionType.ALERT, urgency=4, alert_id="critical_drowsy")


def speak(level=4):
    return Action(type=ActionType.SPEAK, urgency=level, intent=Intent.INSIST, context="moving_critical")


def test_speech_waits_for_the_alarm_then_plays():
    audio = TimedAudio()
    ex = ActionExecutor(audio=audio)
    alarm, talk = ex.execute([ALARM, speak()], t=0.0)
    assert alarm.performed and talk.deferred and not talk.performed
    assert len(audio.played) == 1                              # only the alarm so far
    assert ex.execute([], t=0.1) == []                         # alarm still sounding
    time.sleep(0.2)
    [later] = ex.execute([], t=0.3)
    assert later.performed and later.action.type is ActionType.SPEAK
    assert len(audio.played) == 2 and "critical_drowsy" in audio.played[0][-1]


def test_only_the_newest_speech_waits():
    ex = ActionExecutor(audio=TimedAudio())
    _, first = ex.execute([ALARM, speak(4)], t=0.0)
    [newer] = ex.execute([speak(4)], t=0.05)
    assert newer.deferred
    superseded = [h for h in ex.history if h.suppressed.startswith("superseded")]
    assert superseded == [first]


def test_a_less_urgent_speech_during_the_alarm_is_still_dropped():
    ex = ActionExecutor(audio=TimedAudio())
    ex.execute([ALARM], t=0.0)
    [pleasantry] = ex.execute([speak(1)], t=0.05)
    assert not pleasantry.deferred and "urgency 1" in pleasantry.suppressed


def test_a_new_alarm_still_interrupts_an_alarm():
    audio = TimedAudio()
    ex = ActionExecutor(audio=audio)
    ex.execute([ALARM], t=0.0)
    [again] = ex.execute([ALARM], t=0.05)
    assert again.performed and len(audio.played) == 2


def _tick(sm, t, **kw):
    args = dict(timestamp=t, ignition_on=True, speed_kmh=60.0, drowsiness_level=4,
                confidence=0.9, face_present=True)
    args.update(kw)
    return [a.alert_id for a in sm.update(Telemetry(**args)) if a.type == ActionType.ALERT]


def test_a_microsleep_inside_the_cooldown_sounds_when_it_ends():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    assert _tick(sm, 0.0, microsleep_count=1) == ["critical_drowsy"]
    assert _tick(sm, 2.0, microsleep_count=2) == []            # within the 5 s alert cooldown
    assert _tick(sm, 4.0, microsleep_count=2) == []
    assert _tick(sm, 5.1, microsleep_count=2) == ["critical_drowsy"]
    assert _tick(sm, 11.0, microsleep_count=2) == []           # nothing new since


def test_a_count_that_ages_out_then_rises_again_is_new():
    sm = StateMachine(initial_state=DrivingState.MOVING)
    _tick(sm, 0.0, microsleep_count=3)
    _tick(sm, 30.0, microsleep_count=2)                        # one left the 60 s window
    assert _tick(sm, 40.0, microsleep_count=3) == ["critical_drowsy"]
