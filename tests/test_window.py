"""Tests for the PERCLOS window and level estimation."""

import pytest

from awas.driver.window import (
    FeatureWindow,
    Hysteresis,
    LevelConfig,
    Proposal,
    WindowConfig,
    WindowFeatures,
    propose_level,
)

CFG = WindowConfig()
LVL = LevelConfig()
OPEN, SHUT = 0.9, 0.05
FPS = 30.0


def fill(w, n, closed_at=lambda i: False, face_at=lambda i: True, t0=0.0, fps=FPS):
    t = t0
    for i in range(n):
        w.add(t, SHUT if closed_at(i) else OPEN, face_at(i))
        t += 1.0 / fps
    return t


class TestWindowMechanics:
    def test_evicts_by_time_not_by_count(self):
        w = FeatureWindow(WindowConfig(span_sec=10.0))
        fill(w, 900)                      # 30 s at 30 fps
        assert w.features().observed_sec == pytest.approx(10.0, abs=0.2)

    def test_a_slow_camera_still_gives_a_full_window(self):
        """Frame count is not time. 5 fps must still fill 60 s."""
        w = FeatureWindow(WindowConfig(span_sec=60.0))
        fill(w, 400, fps=5.0)
        assert w.features().observed_sec == pytest.approx(60.0, abs=1.0)

    def test_empty_window_is_not_usable(self):
        assert FeatureWindow().features().usable is False

    def test_clear_empties(self):
        w = FeatureWindow()
        fill(w, 100)
        w.clear()
        assert len(w) == 0


class TestPerclos:
    def test_all_open_is_zero(self):
        w = FeatureWindow()
        fill(w, 900)
        assert w.features().perclos == pytest.approx(0.0)

    def test_all_closed_is_one(self):
        w = FeatureWindow()
        fill(w, 900, closed_at=lambda i: True)
        assert w.features().perclos == pytest.approx(1.0, abs=0.01)

    def test_half_closed_is_half(self):
        w = FeatureWindow()
        fill(w, 900, closed_at=lambda i: (i % 2 == 0))
        assert w.features().perclos == pytest.approx(0.5, abs=0.02)

    def test_is_time_weighted_not_frame_weighted(self):
        """One long closed gap must outweigh many short open frames."""
        w = FeatureWindow()
        w.add(0.0, SHUT, True)
        w.add(6.0, OPEN, True)       # eye was shut for 6 s
        for i in range(10):
            w.add(6.0 + i * 0.01, OPEN, True)
        assert w.features().perclos > 0.9


class TestFaceLoss:
    def test_absent_face_is_excluded_from_perclos(self):
        w = FeatureWindow()
        t = fill(w, 300)                                  # 10 s open, face on
        fill(w, 300, closed_at=lambda i: True,
             face_at=lambda i: False, t0=t)               # 10 s no face
        assert w.features().perclos == pytest.approx(0.0, abs=0.01)

    def test_absent_face_never_becomes_a_microsleep(self):
        """The false alarm that would train drivers to ignore the system."""
        w = FeatureWindow()
        fill(w, 900, closed_at=lambda i: True, face_at=lambda i: False)
        f = w.features()
        assert f.microsleep_count == 0
        assert f.longest_closure_ms == 0.0

    def test_face_present_ratio_is_reported(self):
        w = FeatureWindow()
        t = fill(w, 300)
        fill(w, 300, face_at=lambda i: False, t0=t)
        assert w.features().face_present_ratio == pytest.approx(0.5, abs=0.05)

    def test_mostly_absent_face_is_not_usable(self):
        w = FeatureWindow()
        fill(w, 900, face_at=lambda i: i % 10 == 0)
        assert w.features().usable is False


class TestEpisodes:
    def test_short_closure_is_a_blink_not_a_microsleep(self):
        w = FeatureWindow()
        fill(w, 900, closed_at=lambda i: (i % 90) < 4)     # ~133 ms
        f = w.features()
        assert f.blink_count > 0
        assert f.microsleep_count == 0

    def test_long_closure_is_a_microsleep_not_a_blink(self):
        w = FeatureWindow()
        fill(w, 900, closed_at=lambda i: (i % 300) < 30)   # 1 s
        f = w.features()
        assert f.microsleep_count == 3
        assert f.blink_count == 0

    def test_single_frame_flicker_is_ignored(self):
        """One stray closed frame is landmark noise, not a blink."""
        w = FeatureWindow()
        fill(w, 900, closed_at=lambda i: i == 100)
        assert w.features().blink_count == 0

    def test_longest_closure_is_reported(self):
        w = FeatureWindow()
        fill(w, 900, closed_at=lambda i: 100 <= i < 160)   # 2 s
        assert w.features().longest_closure_ms == pytest.approx(2000, abs=80)

    def test_blink_rate_is_per_minute(self):
        w = FeatureWindow(WindowConfig(span_sec=60.0))
        fill(w, 1800, closed_at=lambda i: (i % 90) < 4)    # 20 blinks in 60 s
        assert w.features().blink_rate_per_min == pytest.approx(20, abs=2)


class TestProposeLevel:
    def base(self, **kw):
        d = dict(observed_sec=60.0, face_present_ratio=1.0, n_samples=1800)
        d.update(kw)
        return WindowFeatures(**d)

    def test_insufficient_observation_proposes_zero(self):
        assert propose_level(WindowFeatures(observed_sec=1.0), LVL).level == 0

    def test_alert_window_proposes_zero(self):
        assert propose_level(self.base(perclos=0.02), LVL).level == 0

    @pytest.mark.parametrize("perclos,level", [(0.10, 1), (0.18, 2), (0.30, 3)])
    def test_perclos_ladder(self, perclos, level):
        assert propose_level(self.base(perclos=perclos), LVL).level == level

    def test_long_blinks_reach_level_2_before_perclos_does(self):
        p = propose_level(self.base(perclos=0.05, mean_blink_duration_ms=450), LVL)
        assert p.level == 2

    def test_one_microsleep_is_level_3(self):
        assert propose_level(self.base(microsleep_count=1), LVL).level == 3

    def test_two_microsleeps_are_critical_and_immediate(self):
        p = propose_level(self.base(microsleep_count=2), LVL)
        assert p.level == 4 and p.immediate is True

    def test_two_second_closure_is_critical_and_immediate(self):
        p = propose_level(self.base(longest_closure_ms=2100), LVL)
        assert p.level == 4 and p.immediate is True

    def test_microsleep_outranks_a_calm_perclos(self):
        assert propose_level(self.base(perclos=0.01, microsleep_count=1), LVL).level == 3


class TestHysteresis:
    def test_escalation_needs_consecutive_ticks(self):
        h = Hysteresis(LevelConfig(escalate_ticks=3))
        assert h.update(Proposal(2, "")) == 0
        assert h.update(Proposal(2, "")) == 0
        assert h.update(Proposal(2, "")) == 2

    def test_a_single_spike_does_not_escalate(self):
        h = Hysteresis(LevelConfig(escalate_ticks=3))
        h.update(Proposal(3, ""))
        h.update(Proposal(0, ""))
        assert h.level == 0

    def test_immediate_proposal_bypasses_hysteresis(self):
        """A two-second closure is the event, not a trend to smooth."""
        h = Hysteresis(LevelConfig(escalate_ticks=3))
        assert h.update(Proposal(4, "closure", immediate=True)) == 4

    def test_immediate_never_forces_a_downgrade(self):
        h = Hysteresis(LevelConfig(escalate_ticks=1))
        h.update(Proposal(4, "", immediate=True))
        assert h.update(Proposal(1, "", immediate=True)) == 4

    def test_deescalation_is_slower_than_escalation(self):
        h = Hysteresis(LevelConfig(escalate_ticks=3, deescalate_ticks=10))
        for _ in range(3):
            h.update(Proposal(3, ""))
        assert h.level == 3
        for _ in range(9):
            h.update(Proposal(0, ""))
        assert h.level == 3          # still not released
        h.update(Proposal(0, ""))
        assert h.level == 2          # one step only

    def test_deescalation_steps_one_level_at_a_time(self):
        h = Hysteresis(LevelConfig(escalate_ticks=1, deescalate_ticks=2))
        h.update(Proposal(4, ""))
        for _ in range(2):
            h.update(Proposal(0, ""))
        assert h.level == 3
        for _ in range(2):
            h.update(Proposal(0, ""))
        assert h.level == 2

    def test_matching_proposal_resets_the_counters(self):
        h = Hysteresis(LevelConfig(escalate_ticks=3))
        h.update(Proposal(2, ""))
        h.update(Proposal(0, ""))     # back to current level: reset
        h.update(Proposal(2, ""))
        h.update(Proposal(2, ""))
        assert h.level == 0
