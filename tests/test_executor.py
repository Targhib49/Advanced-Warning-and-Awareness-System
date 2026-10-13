"""Dialogue selection, action execution and session logging."""

import json
import time
from typing import List

import pytest

from awas.companion.dialogue import TONE_CLIP, DialogueManager
from awas.companion.phrasebank import (
    ALERTS,
    ALERTS_WITH_TONE,
    TEMPLATES,
    all_clips,
    text_for,
)
from awas.core.actions import Action, ActionType, Intent
from awas.runtime.executor import ActionExecutor
from awas.runtime.session_log import SCHEMA_VERSION, SessionLog
from awas.speech.audio import NullAudioOut, Playback


class FakeAudio:
    """Reports a real duration, so the urgency guard can actually be tested."""

    def __init__(self, duration_ms: float = 2000.0, known: bool = True):
        self.duration_ms = duration_ms
        self.known = known
        self.played: List[List[str]] = []

    def play(self, clip_ids):
        self.played.append(list(clip_ids))
        return Playback(clip_id="+".join(clip_ids), latency_ms=8.0,
                        duration_ms=self.duration_ms, played=self.known)

    def is_busy(self):
        return False

    def close(self):
        pass


def alert(alert_id="phone_in_hand", urgency=2):
    return Action(type=ActionType.ALERT, alert_id=alert_id, urgency=urgency)


def speak(intent=Intent.CHECK_IN, urgency=1):
    return Action(type=ActionType.SPEAK, intent=intent, urgency=urgency)


# ---------------- phrase bank ----------------

class TestPhraseBank:
    def test_every_alert_the_machine_emits_has_a_line(self):
        """A state machine alert_id with no phrase would be a silent warning."""
        emitted = {
            "critical_drowsy", "phone_in_hand", "lane_departure",
            "green_light_unresponsive", "traffic_moving", "face_not_seen",
        }
        assert emitted <= set(ALERTS)

    def test_every_intent_has_templates(self):
        for intent in Intent:
            assert TEMPLATES.get(intent), f"{intent} has no phrasing"

    def test_intents_have_variety(self):
        """One line per intent means the driver stops hearing it."""
        for intent in (Intent.CHECK_IN, Intent.ENGAGE, Intent.INSIST):
            assert len(TEMPLATES[intent]) >= 2

    def test_clip_ids_are_unique(self):
        clips = all_clips()
        assert len(clips) == len(set(clips))

    def test_text_lookup_round_trips(self):
        for clip_id, text in all_clips().items():
            assert text_for(clip_id) == text

    def test_unknown_clip_has_no_text(self):
        assert text_for("nonsense") == ""


# ---------------- dialogue ----------------

class TestDialogue:
    def test_critical_alert_leads_with_a_tone(self):
        clips = DialogueManager().clips_for(alert("critical_drowsy", 4))
        assert clips[0] == TONE_CLIP
        assert len(clips) == 2

    def test_ordinary_alert_is_speech_only(self):
        clips = DialogueManager().clips_for(alert("phone_in_hand", 2))
        assert TONE_CLIP not in clips

    def test_tone_set_matches_the_bank(self):
        assert ALERTS_WITH_TONE <= set(ALERTS)

    def test_unknown_alert_yields_nothing(self):
        assert DialogueManager().clips_for(alert("not_a_real_alert")) == []

    def test_log_actions_are_silent(self):
        a = Action(type=ActionType.LOG, message="entered moving")
        assert DialogueManager().clips_for(a) == []

    def test_never_repeats_a_line_back_to_back(self):
        d = DialogueManager(seed=3)
        picks = [d.clips_for(speak(Intent.ENGAGE))[0] for _ in range(30)]
        assert all(a != b for a, b in zip(picks, picks[1:]))

    def test_uses_the_whole_bank_over_time(self):
        d = DialogueManager(seed=3)
        picks = {d.clips_for(speak(Intent.INSIST))[0] for _ in range(40)}
        assert len(picks) == len(TEMPLATES[Intent.INSIST])

    def test_speak_without_an_intent_yields_nothing(self):
        assert DialogueManager().clips_for(
            Action(type=ActionType.SPEAK, intent=None)) == []


# ---------------- executor ----------------

class TestExecutor:
    def test_plays_an_alert(self):
        audio = FakeAudio()
        ex = ActionExecutor(audio=audio)
        [res] = ex.execute([alert()], t=1.0)
        assert res.performed
        assert audio.played == [["alert__phone_in_hand"]]
        assert res.text == ALERTS["phone_in_hand"]

    def test_log_actions_make_no_sound(self):
        audio = FakeAudio()
        ex = ActionExecutor(audio=audio)
        ex.execute([Action(type=ActionType.LOG, message="x")], t=1.0)
        assert audio.played == []

    def test_muted_suppresses_but_still_reports(self):
        ex = ActionExecutor(audio=FakeAudio(), muted=True)
        [res] = ex.execute([alert()], t=1.0)
        assert not res.performed
        assert res.suppressed == "muted"
        assert res.text        # the words are still known, for the overlay

    def test_missing_clip_is_reported_not_crashed(self):
        ex = ActionExecutor(audio=FakeAudio(known=False))
        [res] = ex.execute([alert()], t=1.0)
        assert not res.performed
        assert "missing" in res.suppressed


class TestUrgencyGuard:
    """A pleasantry must never truncate an emergency."""

    def test_quiet_action_cannot_interrupt_a_loud_one(self):
        ex = ActionExecutor(audio=FakeAudio(duration_ms=3000))
        ex.execute([alert("critical_drowsy", urgency=4)], t=1.0)
        [res] = ex.execute([speak(Intent.CHECK_IN, urgency=1)], t=1.2)
        assert not res.performed
        assert "urgency 1" in res.suppressed

    def test_louder_action_does_interrupt(self):
        ex = ActionExecutor(audio=FakeAudio(duration_ms=3000))
        ex.execute([speak(Intent.CHECK_IN, urgency=1)], t=1.0)
        [res] = ex.execute([alert("critical_drowsy", urgency=4)], t=1.2)
        assert res.performed

    def test_equal_urgency_is_allowed_through(self):
        ex = ActionExecutor(audio=FakeAudio(duration_ms=3000))
        ex.execute([alert("phone_in_hand", urgency=2)], t=1.0)
        [res] = ex.execute([alert("phone_in_hand", urgency=2)], t=1.2)
        assert res.performed

    def test_guard_lifts_once_playback_finishes(self):
        ex = ActionExecutor(audio=FakeAudio(duration_ms=20))
        ex.execute([alert("critical_drowsy", urgency=4)], t=1.0)
        time.sleep(0.05)
        [res] = ex.execute([speak(Intent.CHECK_IN, urgency=1)], t=1.5)
        assert res.performed


# ---------------- session log ----------------

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class FakeState:
    drowsiness_level = 2
    confidence = 0.81
    perclos = 0.184
    microsleep_count = 0
    blink_duration_mean_ms = 412.0
    blink_rate_per_min = 14.2
    longest_closure_ms = 260.0
    face_present = True
    eye_aspect_ratio = 0.19
    openness = 0.44
    head_pitch_deg = -12.0
    head_yaw_deg = 3.0
    personalized = True


class TestSessionLog:
    def test_opens_with_a_versioned_header(self, tmp_path):
        log = SessionLog(tmp_path, driver="budi")
        path = log.close()
        first = read(path)[0]
        assert first["type"] == "session_start"
        assert first["schema_version"] == SCHEMA_VERSION
        assert first["driver"] == "budi"

    def test_closes_with_a_summary(self, tmp_path):
        log = SessionLog(tmp_path)
        log.sample(0.0, FakeState())
        path = log.close(t=10.0)
        last = read(path)[-1]
        assert last["type"] == "session_end"
        assert last["counts"]["sample"] == 1

    def test_samples_are_rate_limited(self, tmp_path):
        log = SessionLog(tmp_path, sample_hz=1.0)
        written = [log.sample(t / 30.0, FakeState()) for t in range(90)]
        log.close()
        assert sum(written) == 3        # 3 seconds at 1 Hz, not 90 frames

    def test_sample_carries_the_fields_the_study_needs(self, tmp_path):
        log = SessionLog(tmp_path)
        log.sample(1.0, FakeState())
        path = log.close()
        s = [r for r in read(path) if r["type"] == "sample"][0]
        for key in ("level", "confidence", "perclos", "microsleeps", "face",
                    "openness", "personalized"):
            assert key in s, f"{key} missing — unrecoverable after the fact"

    def test_action_records_its_cause(self, tmp_path):
        """An intervention with no trigger cannot be correlated with anything."""
        log = SessionLog(tmp_path)
        ex = ActionExecutor(audio=FakeAudio(), log=log)
        telemetry = FakeState()
        telemetry.speed_kmh = 62.0
        ex.execute([alert("critical_drowsy", 4)], t=5.0,
                   telemetry=telemetry, driving_state="moving")
        path = log.close()
        a = [r for r in read(path) if r["type"] == "action"][0]
        assert a["alert_id"] == "critical_drowsy"
        assert a["driving_state"] == "moving"
        assert a["trigger"]["level"] == 2
        assert a["latency_ms"] == 8.0

    def test_suppressed_actions_are_still_recorded(self, tmp_path):
        log = SessionLog(tmp_path)
        ex = ActionExecutor(audio=FakeAudio(), log=log, muted=True)
        ex.execute([alert()], t=1.0)
        path = log.close()
        notes = [r for r in read(path) if r["type"] == "annotation"]
        assert any(r["text"] == "muted" for r in notes)

    def test_records_a_kss_answer(self, tmp_path):
        log = SessionLog(tmp_path)
        log.annotation(0.0, kind="kss", value=3)
        path = log.close()
        a = [r for r in read(path) if r.get("kind") == "kss"][0]
        assert a["value"] == 3

    def test_holds_no_imagery(self, tmp_path):
        """Invariant I5: derived numbers only, nothing reconstructable."""
        log = SessionLog(tmp_path)
        log.sample(0.0, FakeState())
        path = log.close()
        raw = path.read_text().lower()
        for banned in ("landmark", "frame", "image", "pixel", "crop", "jpeg"):
            assert banned not in raw

    def test_survives_being_used_as_a_context_manager(self, tmp_path):
        with SessionLog(tmp_path) as log:
            log.sample(0.0, FakeState())
            path = log.path
        assert read(path)[-1]["type"] == "session_end"


# ---------------- time-to-relapse (H1 primary outcome) ----------------

from scripts.analyze_session import time_to_relapse  # noqa: E402


def sample_rec(t, level):
    return {"type": "sample", "t": t, "level": level}


def action_rec(t, level, kind="speak", label="engage"):
    key = "alert_id" if kind == "alert" else "intent"
    return {"type": "action", "t": t, "kind": kind, key: label,
            "trigger": {"level": level}}


class TestTimeToRelapse:
    def test_measures_from_intervention_to_return(self):
        recs = [action_rec(10.0, 2)] + [
            sample_rec(t, 2 if t >= 40 else 0) for t in range(11, 60)
        ]
        [r] = time_to_relapse(recs)
        assert r["ttr_sec"] == pytest.approx(30.0)

    def test_requires_recovery_before_counting_a_relapse(self):
        """Still drowsy the whole time is one unbroken episode, not a relapse."""
        recs = [action_rec(10.0, 2)] + [sample_rec(t, 2) for t in range(11, 60)]
        [r] = time_to_relapse(recs)
        assert r["ttr_sec"] is None
        assert r["recovered"] is False

    def test_recovered_and_never_relapsed_is_censored_not_zero(self):
        """Treating this as 0 would make a working intervention look useless."""
        recs = [action_rec(10.0, 2)] + [sample_rec(t, 0) for t in range(11, 60)]
        [r] = time_to_relapse(recs)
        assert r["recovered"] is True
        assert r["censored"] is True
        assert r["ttr_sec"] is None

    def test_ignores_level_zero_actions(self):
        recs = [action_rec(10.0, 0)] + [sample_rec(t, 0) for t in range(11, 30)]
        assert time_to_relapse(recs) == []

    def test_honours_the_horizon(self):
        recs = [action_rec(0.0, 2)] + [
            sample_rec(t, 2 if t >= 500 else 0) for t in range(1, 600)
        ]
        assert time_to_relapse(recs, horizon_sec=100.0)[0]["ttr_sec"] is None
        assert time_to_relapse(recs, horizon_sec=600.0)[0]["ttr_sec"] is not None

    def test_labels_alerts_and_dialogue_separately(self):
        recs = [
            action_rec(0.0, 3, kind="alert", label="critical_drowsy"),
            action_rec(1.0, 2, kind="speak", label="engage"),
        ] + [sample_rec(t, 0) for t in range(2, 20)]
        labels = {r["label"] for r in time_to_relapse(recs)}
        assert labels == {"critical_drowsy", "engage"}

    def test_a_session_with_no_interventions_is_empty_not_an_error(self):
        assert time_to_relapse([sample_rec(t, 0) for t in range(10)]) == []


# ---------------- experimental conditions (plan §6.3) ----------------

from awas.companion.dialogue import TONE_CLIP  # noqa: E402
from awas.runtime.conditions import CONDITIONS, get  # noqa: E402


def run_under(condition_id, actions, audio=None):
    audio = audio or FakeAudio()
    ex = ActionExecutor(audio=audio, condition=get(condition_id))
    return ex.execute(actions, t=1.0), audio


class TestConditions:
    def test_all_four_conditions_exist(self):
        assert set(CONDITIONS) == {"A", "B", "C", "D"}

    def test_lookup_is_case_insensitive(self):
        assert get("c").id == "C"

    def test_unknown_condition_fails_loudly(self):
        with pytest.raises(SystemExit, match="Unknown condition"):
            get("Z")

    def test_a_is_silent(self):
        """The no-intervention control: detects, logs, never responds."""
        res, audio = run_under("A", [alert("critical_drowsy", 4), speak()])
        assert audio.played == []
        assert all(not r.performed for r in res)

    def test_a_still_records_what_it_would_have_done(self):
        """Otherwise the control is an empty log and useless as a comparison."""
        res, _ = run_under("A", [alert("critical_drowsy", 4)])
        assert "condition A" in res[0].suppressed
        assert res[0].action.alert_id == "critical_drowsy"

    def test_b_plays_a_tone_with_no_words(self):
        res, audio = run_under("B", [alert("critical_drowsy", 4)])
        assert audio.played == [[TONE_CLIP]]
        assert res[0].performed

    def test_b_gives_a_tone_even_to_alerts_that_normally_lack_one(self):
        """A crippled baseline would make H1 trivially true and worthless."""
        _, audio = run_under("B", [alert("phone_in_hand", 2)])
        assert audio.played == [[TONE_CLIP]]

    def test_b_holds_no_conversation(self):
        res, _ = run_under("B", [speak(Intent.ENGAGE, 2)])
        assert not res[0].performed
        assert "no dialogue" in res[0].suppressed

    def test_c_does_everything(self):
        # Zero duration so the urgency guard stays out of the way — this test
        # is about what the CONDITION permits, not about interruption.
        res, audio = run_under("C", [alert("critical_drowsy", 4), speak()],
                               audio=FakeAudio(duration_ms=0))
        assert all(r.performed for r in res)
        assert audio.played[0] == [TONE_CLIP, "alert__critical_drowsy"]

    def test_d_matches_c_at_the_executor(self):
        """D differs by gating, which lives in the state machine, not here."""
        c_res, _ = run_under("C", [alert("critical_drowsy", 4), speak()],
                             audio=FakeAudio(duration_ms=0))
        d_res, _ = run_under("D", [alert("critical_drowsy", 4), speak()],
                             audio=FakeAudio(duration_ms=0))
        assert [r.performed for r in c_res] == [r.performed for r in d_res]

    def test_gating_flag_differentiates_c_from_d(self):
        assert get("C").gating is True
        assert get("D").gating is False

    def test_perception_is_identical_across_conditions(self):
        """Same decisions in, different responses out — that is the whole point.

        If a condition changed what the state machine decided, any difference
        in outcome would be confounded by a difference in detection.
        """
        actions = [alert("critical_drowsy", 4), speak(Intent.ENGAGE, 2)]
        seen = {}
        for cid in "ABCD":
            res, _ = run_under(cid, list(actions))
            seen[cid] = [(r.action.type, r.action.urgency) for r in res]
        assert len(set(map(tuple, seen.values()))) == 1

    def test_condition_default_is_the_full_system(self):
        ex = ActionExecutor(audio=FakeAudio())
        assert ex.condition.id == "C"

    def test_silent_property(self):
        assert get("A").silent is True
        assert all(not get(c).silent for c in "BCD")
