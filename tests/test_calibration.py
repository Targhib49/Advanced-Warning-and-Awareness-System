"""Tests for EAR geometry and per-driver calibration."""

import json
import math
import random

import pytest

from awas.driver.calibration import (
    POPULATION_CLOSED_EAR,
    POPULATION_OPEN_EAR,
    PROFILE_VERSION,
    CalibrationConfig,
    DriverProfile,
    EyeOpennessCalibrator,
    OpennessNormalizer,
    SampleContext,
    explain,
    is_known_alert,
    quantile,
)
from awas.driver.ear import ear_from_landmarks, eye_aspect_ratio, mean_ear

CFG = CalibrationConfig()


def eye(openness: float, width: float = 8.0):
    """Six landmarks for an eye of a given aperture."""
    h = openness * width / 2.0
    return [(0.0, 0.0), (2.0, -h), (6.0, -h), (width, 0.0), (6.0, h), (2.0, h)]


def feed(cal: EyeOpennessCalibrator, n: int, open_ear: float, closed_ear: float,
         blink_every: int = 90, blink_len: int = 4, seed: int = 0, fps: float = 30.0):
    """Simulate an alert driver blinking at a steady rate."""
    rng = random.Random(seed)
    for i in range(n):
        blinking = (i % blink_every) < blink_len
        ear = (closed_ear if blinking else open_ear) + rng.gauss(0, 0.004)
        cal.observe(ear, SampleContext(trip_elapsed_sec=i / fps))


# ---------------- EAR geometry ----------------

class TestEyeAspectRatio:
    def test_open_eye_scores_higher_than_closed(self):
        assert eye_aspect_ratio(eye(0.8)) > eye_aspect_ratio(eye(0.1))

    def test_is_scale_invariant(self):
        """Moving toward the camera must not change the ratio."""
        near = eye_aspect_ratio(eye(0.5, width=8.0))
        far = eye_aspect_ratio(eye(0.5, width=16.0))
        assert math.isclose(near, far, rel_tol=1e-9)

    def test_degenerate_eye_returns_zero(self):
        assert eye_aspect_ratio([(0, 0)] * 6) == 0.0

    def test_rejects_wrong_point_count(self):
        with pytest.raises(ValueError, match="exactly 6"):
            eye_aspect_ratio([(0, 0)] * 5)

    def test_mean_uses_both_eyes(self):
        both = mean_ear(eye(0.8), eye(0.2))
        assert eye_aspect_ratio(eye(0.2)) < both < eye_aspect_ratio(eye(0.8))

    def test_mean_falls_back_when_one_eye_is_lost(self):
        """A dropped eye must not halve the reading into a fake closure."""
        good = eye_aspect_ratio(eye(0.8))
        assert math.isclose(mean_ear(eye(0.8), [(0, 0)] * 6), good, rel_tol=1e-9)

    def test_mean_of_two_lost_eyes_is_zero(self):
        assert mean_ear([(0, 0)] * 6, [(0, 0)] * 6) == 0.0

    def test_from_landmarks_needs_a_full_mesh(self):
        with pytest.raises(ValueError, match="required"):
            ear_from_landmarks([(0.0, 0.0)] * 100)

    def test_from_landmarks_reads_the_indexed_points(self):
        marks = [(0.0, 0.0)] * 478
        for i, p in zip((362, 385, 387, 263, 373, 380), eye(0.6)):
            marks[i] = p
        for i, p in zip((33, 160, 158, 133, 153, 144), eye(0.6)):
            marks[i] = p
        assert ear_from_landmarks(marks) == pytest.approx(eye_aspect_ratio(eye(0.6)))


# ---------------- quantile ----------------

class TestQuantile:
    def test_median_of_odd_length(self):
        assert quantile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_interpolates(self):
        assert quantile([0.0, 10.0], 0.5) == 5.0

    def test_endpoints(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        assert quantile(vals, 0.0) == 1.0
        assert quantile(vals, 1.0) == 4.0

    def test_single_value(self):
        assert quantile([7.0], 0.3) == 7.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            quantile([], 0.5)


# ---------------- the known-alert gate ----------------

class TestKnownAlertGate:
    def test_accepts_an_ordinary_early_frame(self):
        assert is_known_alert(SampleContext(trip_elapsed_sec=10.0), CFG)

    def test_rejects_after_the_calibration_window(self):
        assert not is_known_alert(SampleContext(trip_elapsed_sec=9999.0), CFG)

    def test_rejects_when_the_face_is_lost(self):
        assert not is_known_alert(
            SampleContext(trip_elapsed_sec=10.0, face_present=False), CFG)

    def test_rejects_when_already_drowsy(self):
        """A baseline polluted with drowsy frames desensitises the system."""
        assert not is_known_alert(
            SampleContext(trip_elapsed_sec=10.0, drowsiness_level=1), CFG)

    def test_rejects_extreme_pitch(self):
        assert not is_known_alert(
            SampleContext(trip_elapsed_sec=10.0, head_pitch_deg=35.0), CFG)

    def test_rejects_extreme_yaw(self):
        assert not is_known_alert(
            SampleContext(trip_elapsed_sec=10.0, head_yaw_deg=40.0), CFG)


# ---------------- calibrator ----------------

class TestCalibrator:
    def test_recovers_both_reference_points(self):
        cal = EyeOpennessCalibrator("a")
        feed(cal, 3000, open_ear=0.34, closed_ear=0.09)
        p = cal.build()
        assert p.open_ref == pytest.approx(0.34, abs=0.02)
        assert p.closed_ref == pytest.approx(0.09, abs=0.02)

    def test_returns_none_before_enough_samples(self):
        cal = EyeOpennessCalibrator("a")
        feed(cal, 100, 0.34, 0.09)
        assert cal.build() is None

    def test_rejects_a_profile_with_no_separation(self):
        """A driver who never blinked gives a range we cannot divide by."""
        cal = EyeOpennessCalibrator("a")
        feed(cal, 3000, open_ear=0.30, closed_ear=0.30, blink_every=10 ** 9)
        assert cal.build() is None

    def test_ignores_failed_landmark_fits(self):
        cal = EyeOpennessCalibrator("a")
        for _ in range(50):
            assert cal.observe(0.0, SampleContext(trip_elapsed_sec=1.0)) is False
        assert cal.n_admitted == 0
        assert cal.n_rejected == 50

    def test_ignores_frames_outside_the_gate(self):
        cal = EyeOpennessCalibrator("a")
        assert cal.observe(0.3, SampleContext(trip_elapsed_sec=1.0)) is True
        assert cal.observe(0.3, SampleContext(trip_elapsed_sec=1.0,
                                             drowsiness_level=2)) is False
        assert cal.n_admitted == 1

    def test_reset_clears_state(self):
        cal = EyeOpennessCalibrator("a")
        feed(cal, 1000, 0.34, 0.09)
        cal.reset()
        assert cal.n_admitted == 0 and cal.build() is None


# ---------------- profile ----------------

class TestDriverProfile:
    def test_confidence_rises_with_samples(self):
        few = DriverProfile("a", 0.34, 0.09, n_samples=1000)
        many = DriverProfile("a", 0.34, 0.09, n_samples=9000)
        assert many.confidence(CFG) > few.confidence(CFG)

    def test_confidence_is_capped_at_one(self):
        p = DriverProfile("a", 0.34, 0.09, n_samples=10 ** 7)
        assert p.confidence(CFG) <= 1.0

    def test_unusable_profile_reports_cold_start_confidence(self):
        p = DriverProfile("a", 0.34, 0.09, n_samples=10)
        assert p.confidence(CFG) == CFG.cold_start_confidence

    def test_merge_is_weighted_by_samples(self):
        old = DriverProfile("a", 0.30, 0.08, n_samples=9000)
        new = DriverProfile("a", 0.40, 0.10, n_samples=1000)
        merged = old.merged_with(new, CFG)
        assert 0.30 < merged.open_ref < 0.32     # history dominates, but moves
        assert merged.n_samples == 10000

    def test_merge_caps_how_much_history_can_outvote_a_session(self):
        ancient = DriverProfile("a", 0.30, 0.08, n_samples=10 ** 7)
        fresh = DriverProfile("a", 0.40, 0.10, n_samples=45000)
        merged = ancient.merged_with(fresh, CFG)
        assert merged.open_ref == pytest.approx(0.35, abs=0.01)

    def test_detects_drift(self):
        p = DriverProfile("a", 0.34, 0.09, n_samples=9000)
        assert p.has_drifted(0.20, CFG) is True
        assert p.has_drifted(0.33, CFG) is False

    def test_save_and_load_roundtrip(self, tmp_path):
        p = DriverProfile("budi", 0.3394, 0.0895, n_samples=3000)
        p.save(tmp_path)
        back = DriverProfile.load(tmp_path, "budi")
        assert back.open_ref == pytest.approx(p.open_ref)
        assert back.closed_ref == pytest.approx(p.closed_ref)
        assert back.n_samples == p.n_samples

    def test_load_returns_none_for_an_unknown_driver(self, tmp_path):
        assert DriverProfile.load(tmp_path, "nobody") is None

    def test_load_refuses_a_stale_version(self, tmp_path):
        """A profile from another landmark model encodes a different scale."""
        (tmp_path / "old.json").write_text(json.dumps({
            "name": "old", "open_ref": 0.3, "closed_ref": 0.08,
            "n_samples": 5000, "updated_at": 0.0,
            "version": PROFILE_VERSION + 1,
        }))
        assert DriverProfile.load(tmp_path, "old") is None


# ---------------- normalizer ----------------

class TestOpennessNormalizer:
    def test_open_reference_maps_to_one(self):
        n = OpennessNormalizer(DriverProfile("a", 0.34, 0.09, n_samples=9000))
        assert n.openness(0.34) == pytest.approx(1.0)

    def test_closed_reference_maps_to_zero(self):
        n = OpennessNormalizer(DriverProfile("a", 0.34, 0.09, n_samples=9000))
        assert n.openness(0.09) == pytest.approx(0.0)

    def test_clamps_outside_the_range(self):
        n = OpennessNormalizer(DriverProfile("a", 0.34, 0.09, n_samples=9000))
        assert n.openness(0.99) == 1.0
        assert n.openness(0.01) == 0.0

    def test_failed_fit_is_not_a_closure(self):
        n = OpennessNormalizer(DriverProfile("a", 0.34, 0.09, n_samples=9000))
        assert n.openness(0.0) == 0.0   # callers must gate on face_present

    def test_falls_back_to_population_without_a_profile(self):
        n = OpennessNormalizer(None)
        assert n.personalized is False
        assert n.open_ref == POPULATION_OPEN_EAR
        assert n.closed_ref == POPULATION_CLOSED_EAR
        assert n.confidence == CFG.cold_start_confidence

    def test_ignores_an_immature_profile(self):
        n = OpennessNormalizer(DriverProfile("a", 0.34, 0.09, n_samples=5))
        assert n.personalized is False

    def test_p80_matches_the_perclos_definition(self):
        """PERCLOS P80 is 'more than 80% closed' = openness below 0.20."""
        n = OpennessNormalizer(DriverProfile("a", 0.30, 0.10, n_samples=9000))
        assert n.is_closed(0.13) is True     # openness 0.15
        assert n.is_closed(0.16) is False    # openness 0.30


class TestPersonalizationActuallyChangesTheVerdict:
    """The whole reason this module exists."""

    # Ranges consistent with a MediaPipe mesh (closed EAR near zero), not with
    # the dlib-shaped numbers this file originally assumed.
    WIDE = DriverProfile("wide", open_ref=0.38, closed_ref=0.025, n_samples=9000)
    NARROW = DriverProfile("narrow", open_ref=0.20, closed_ref=0.015, n_samples=9000)

    def test_same_ear_reads_differently_for_different_eyes(self):
        ear = 0.15
        wide = OpennessNormalizer(self.WIDE).openness(ear)
        narrow = OpennessNormalizer(self.NARROW).openness(ear)
        assert wide < narrow                 # same number, opposite stories
        assert narrow - wide > 0.3

    def test_population_threshold_misses_a_wide_eyed_driver(self):
        """0.085 is 'fine' on the population scale but 80% shut for this driver."""
        ear = 0.085
        assert OpennessNormalizer(None).is_closed(ear) is False
        assert OpennessNormalizer(self.WIDE).is_closed(ear) is True

    def test_population_threshold_false_alarms_on_a_narrow_eyed_driver(self):
        ear = 0.060
        assert OpennessNormalizer(None).is_closed(ear) is True
        assert OpennessNormalizer(self.NARROW).is_closed(ear) is False

    def test_explain_reports_both_verdicts(self):
        out = explain(0.085, OpennessNormalizer(self.WIDE))
        assert out["population"]["closed_p80"] is False
        assert out["personal"]["closed_p80"] is True
        assert out["personal"]["personalized"] is True


class TestEndToEnd:
    def test_calibrate_persist_reload_and_use(self, tmp_path):
        cal = EyeOpennessCalibrator("hendra")
        feed(cal, 4000, open_ear=0.36, closed_ear=0.10)
        profile = cal.build()
        profile.save(tmp_path)

        reloaded = DriverProfile.load(tmp_path, "hendra")
        n = OpennessNormalizer(reloaded)
        assert n.personalized is True
        assert n.openness(0.36) == pytest.approx(1.0, abs=0.05)
        assert n.is_closed(0.13) is True
