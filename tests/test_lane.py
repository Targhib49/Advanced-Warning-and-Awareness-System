"""Lane detection, verified against synthetic roads with known geometry."""

import numpy as np
import pytest

from awas.road.camera import CameraRoadConfig, CameraRoadSource
from awas.road.lane_detect import LaneCalibration, LaneDetector, synthetic_road
from awas.road.signals import LaneTracker


@pytest.fixture
def det():
    return LaneDetector()


class TestGeometryRoundTrip:
    """Render a known offset, detect it, check the number comes back."""

    @pytest.mark.parametrize("offset", [-0.8, -0.4, -0.15, 0.0, 0.15, 0.4, 0.8])
    def test_recovers_the_offset_it_was_given(self, det, offset):
        r = det.detect(synthetic_road(offset_m=offset))
        assert r.found, r.reason
        assert r.offset_m == pytest.approx(offset, abs=0.06)

    def test_sign_convention_is_right_positive(self, det):
        assert det.detect(synthetic_road(offset_m=0.5)).offset_m > 0
        assert det.detect(synthetic_road(offset_m=-0.5)).offset_m < 0

    def test_lane_width_is_plausible(self, det):
        r = det.detect(synthetic_road(0.0))
        assert 100 < r.lane_width_px < 250

    def test_offset_is_recovered_whatever_the_lane_width(self, det):
        """Scale comes from the lane detected in-frame, so 3.0 m and 3.5 m
        roads both return the true offset rather than a pixel count."""
        cal3 = LaneCalibration(lane_width_m=3.0)
        wide = det.detect(synthetic_road(0.4, lane_width_m=3.5)).offset_m
        narrow = LaneDetector(cal3).detect(
            synthetic_road(0.4, cal=cal3, lane_width_m=3.0)).offset_m
        assert wide == pytest.approx(0.4, abs=0.06)
        assert narrow == pytest.approx(0.4, abs=0.06)

    def test_a_miscalibrated_lane_width_scales_the_error(self, det):
        """The calibration sensitivity, stated as a test.

        Assume 3.0 m on a road whose lanes are really 3.5 m and every offset
        comes back about 14% small. Nothing detects this on its own — which is
        why lane_width_m has to be set from the road being driven.
        """
        truth = det.detect(synthetic_road(0.7, lane_width_m=3.5)).offset_m
        wrong = LaneDetector(LaneCalibration(lane_width_m=3.0)).detect(
            synthetic_road(0.7, lane_width_m=3.5)).offset_m
        assert wrong == pytest.approx(truth * (3.0 / 3.5), abs=0.05)


class TestRobustness:
    def test_survives_sensor_noise(self, det):
        r = det.detect(synthetic_road(0.3, noise=14.0))
        assert r.found and r.offset_m == pytest.approx(0.3, abs=0.08)

    def test_handles_a_solid_line_on_both_sides(self, det):
        r = det.detect(synthetic_road(0.3, dashed_right=False))
        assert r.found and r.offset_m == pytest.approx(0.3, abs=0.08)

    def test_blank_road_finds_nothing(self, det):
        blank = np.full((480, 640, 3), 70, np.uint8)
        assert det.detect(blank).found is False

    def test_failure_returns_none_not_zero(self, det):
        """'I cannot see the lane' must never look like 'we are centred'."""
        r = det.detect(np.full((480, 640, 3), 70, np.uint8))
        assert r.offset_m is None
        assert r.reason

    def test_uniform_bright_frame_is_rejected(self, det):
        assert det.detect(np.full((480, 640, 3), 255, np.uint8)).found is False

    def test_reports_why_it_failed(self, det):
        r = det.detect(np.full((480, 640, 3), 70, np.uint8))
        assert len(r.reason) > 3


class TestCameraRoadSource:
    def push_all(self, src, offsets, hz=15.0):
        states, t = [], 0.0
        for o in offsets:
            states.append(src.push(synthetic_road(offset_m=o), t))
            t += 1.0 / hz
        return states

    def test_produces_lane_offsets(self):
        src = CameraRoadSource(speed_kmh=90.0)
        states = self.push_all(src, [0.0, 0.2, -0.2])
        assert all(s.lane_offset_m is not None for s in states)
        assert states[0].speed_kmh == 90.0

    def test_departure_needs_to_persist(self):
        """One bad frame is a glitch, not an alert."""
        src = CameraRoadSource(config=CameraRoadConfig(departure_frames=3))
        states = self.push_all(src, [0.0, 1.0, 0.0, 0.0])
        assert not any(s.lane_departure for s in states)

    def test_sustained_departure_fires(self):
        src = CameraRoadSource(config=CameraRoadConfig(departure_frames=3))
        states = self.push_all(src, [0.0] + [1.0] * 5)
        assert states[-1].lane_departure is True

    def test_returning_to_lane_clears_it(self):
        src = CameraRoadSource(config=CameraRoadConfig(departure_frames=3))
        states = self.push_all(src, [1.0] * 5 + [0.0] * 3)
        assert states[-1].lane_departure is False

    def test_blindness_is_detected(self):
        src = CameraRoadSource(config=CameraRoadConfig(blind_after_frames=5))
        blank = np.full((480, 640, 3), 70, np.uint8)
        for i in range(6):
            src.push(blank, i / 15.0)
        assert src.blind is True

    def test_unreadable_frames_yield_no_lane_sample(self):
        """SDLP must be computed over frames actually seen, not invented ones."""
        src = CameraRoadSource()
        blank = np.full((480, 640, 3), 70, np.uint8)
        s = src.push(blank, 0.0)
        assert s.lane_offset_m is None


class TestFeedsTheVehicleChannel:
    """The point of the whole module: real SDLP for hypothesis H2."""

    def sdlp_for(self, wander_m: float, seed: int = 0, n: int = 120) -> float:
        import random
        rng = random.Random(seed)
        src, tracker, t = CameraRoadSource(speed_kmh=90.0), LaneTracker(), 0.0
        for _ in range(n):
            frame = synthetic_road(offset_m=rng.gauss(0, wander_m))
            tracker.add(src.push(frame, t))
            t += 1.0 / 15.0
        return tracker.stats().sdlp

    def test_steady_driving_gives_low_sdlp(self):
        assert self.sdlp_for(0.05) < 0.12

    def test_wandering_gives_high_sdlp(self):
        assert self.sdlp_for(0.35) > 0.22

    def test_the_two_separate(self):
        assert self.sdlp_for(0.35, seed=2) > self.sdlp_for(0.05, seed=2) * 2.5


# ---------------- calibration ----------------

from dataclasses import replace  # noqa: E402

from awas.road.calibrate import (  # noqa: E402
    auto_calibrate,
    find_vanishing_point,
    load,
    save,
    score_calibration,
)


def footage(cal, offsets=(0.0, 0.15, -0.1, 0.25, -0.2, 0.05), noise=6.0):
    return [synthetic_road(offset_m=o, cal=cal, noise=noise, seed=i)
            for i, o in enumerate(offsets)]


class TestScoring:
    def test_a_matching_trapezoid_scores_well(self):
        cal = LaneCalibration()
        assert score_calibration(cal, footage(cal)).total > 0.7

    def test_a_mismatched_trapezoid_scores_badly(self):
        truth = LaneCalibration(top_y=0.70, bottom_half_width=0.62)
        assert score_calibration(LaneCalibration(), footage(truth)).total < 0.3

    def test_blank_footage_scores_zero(self):
        blank = [np.full((480, 640, 3), 70, np.uint8)] * 4
        assert score_calibration(LaneCalibration(), blank).total == 0.0

    def test_detection_rate_dominates(self):
        """A tidy calibration that rarely finds the lane is not a good one."""
        sc = score_calibration(LaneCalibration(),
                               footage(LaneCalibration(top_y=0.70)))
        assert sc.total < 0.5


class TestVanishingPoint:
    def test_finds_the_horizon(self):
        cal = LaneCalibration(top_y=0.70)
        vp = find_vanishing_point(footage(cal, noise=4.0))
        assert vp is not None
        assert vp[1] / cal.height == pytest.approx(0.63, abs=0.12)

    def test_returns_none_on_a_blank_frame(self):
        assert find_vanishing_point([np.full((480, 640, 3), 70, np.uint8)]) is None


class TestAutoCalibrate:
    def test_recovers_a_different_mounting(self):
        truth = LaneCalibration(top_y=0.70, top_half_width=0.14,
                                bottom_half_width=0.62)
        frames = footage(truth)
        best, sc = auto_calibrate(frames)
        assert best.top_y == pytest.approx(truth.top_y, abs=0.06)
        assert sc.detection_rate > 0.8

    def test_beats_the_default_it_started_from(self):
        truth = LaneCalibration(top_y=0.72, bottom_half_width=0.58)
        frames = footage(truth)
        before = score_calibration(LaneCalibration(), frames)
        _, after = auto_calibrate(frames)
        assert after.total > before.total

    def test_keeps_the_lane_width_it_was_given(self):
        best, _ = auto_calibrate(footage(LaneCalibration()), lane_width_m=3.0)
        assert best.lane_width_m == 3.0

    def test_needs_frames(self):
        with pytest.raises(ValueError):
            auto_calibrate([])

    def test_result_actually_detects(self):
        truth = LaneCalibration(top_y=0.70, bottom_half_width=0.62)
        best, _ = auto_calibrate(footage(truth))
        r = LaneDetector(best).detect(synthetic_road(0.3, cal=truth))
        assert r.found
        assert r.offset_m == pytest.approx(0.3, abs=0.15)


class TestPersistence:
    def test_round_trips(self, tmp_path):
        cal = LaneCalibration(top_y=0.71, lane_width_m=3.2)
        path = save(cal, tmp_path / "lane.json")
        back = load(path)
        assert back.top_y == pytest.approx(0.71)
        assert back.lane_width_m == pytest.approx(3.2)

    def test_missing_file_is_none_not_an_error(self):
        assert load("/nonexistent/lane.json") is None

    def test_corrupt_file_is_none(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert load(bad) is None

    def test_camera_source_reports_whether_it_is_calibrated(self):
        from awas.road.camera import CameraRoadSource
        src = CameraRoadSource(calibration=LaneCalibration(top_y=0.7))
        assert src.calibrated is True
