"""Layer 2: road signals, lane statistics, workload, and multimodal fusion."""

import pytest

from awas.driver.pipeline import DriverState
from awas.fusion.core import (
    SINGLE_CHANNEL_CAP,
    ChannelEstimate,
    facial_channel,
    fuse,
    vehicle_channel,
)
from awas.fusion.telemetry import build_telemetry, fuse_channels
from awas.road.signals import (
    LaneTracker,
    MockRoadSource,
    NullRoadSource,
    RoadScene,
    RoadState,
)
from awas.road.workload import WorkloadEstimator


def drive_lane(tracker, n, offset=0.0, jitter=0.0, seed=0, hz=10.0,
               steering=None, departures=0):
    import random
    rng = random.Random(seed)
    t = 0.0
    for i in range(n):
        tracker.add(RoadState(
            t=t, speed_kmh=90.0,
            lane_offset_m=offset + (rng.gauss(0, jitter) if jitter else 0.0),
            steering_angle_deg=steering(i) if steering else None,
            lane_departure=i < departures,
        ))
        t += 1.0 / hz
    return tracker.stats()


# ---------------- road sources ----------------

class TestRoadSources:
    def test_null_source_reports_only_speed(self):
        s = NullRoadSource(speed_kmh=62.0).read(1.0)
        assert s.speed_kmh == 62.0
        assert s.lane_offset_m is None
        assert s.has_lane is False

    def test_unknown_is_none_not_zero(self):
        """A missing lane reading is not a perfectly centred vehicle."""
        assert NullRoadSource().read(0.0).lane_offset_m is None

    def test_mock_replays_scenes_in_order(self):
        src = MockRoadSource([
            RoadScene(10, speed_kmh=90, road_class="highway"),
            RoadScene(10, speed_kmh=0, traffic_light_color="red"),
        ])
        assert src.read(5.0).speed_kmh == 90
        assert src.read(15.0).traffic_light_color == "red"

    def test_mock_holds_the_last_scene_past_the_end(self):
        src = MockRoadSource([RoadScene(5, speed_kmh=40)])
        assert src.read(999.0).speed_kmh == 40

    def test_mock_needs_a_scene(self):
        with pytest.raises(ValueError):
            MockRoadSource([])

    def test_moving_threshold(self):
        assert RoadState(speed_kmh=20).moving is True
        assert RoadState(speed_kmh=2).moving is False


# ---------------- lane statistics ----------------

class TestLaneTracker:
    def test_steady_driving_has_low_sdlp(self):
        assert drive_lane(LaneTracker(), 600, jitter=0.05).sdlp < 0.10

    def test_wandering_has_high_sdlp(self):
        assert drive_lane(LaneTracker(), 600, jitter=0.35).sdlp > 0.25

    def test_sdlp_separates_the_two(self):
        """The whole basis of the vehicle channel."""
        alert = drive_lane(LaneTracker(), 600, jitter=0.05, seed=1).sdlp
        tired = drive_lane(LaneTracker(), 600, jitter=0.35, seed=1).sdlp
        assert tired > alert * 3

    def test_needs_enough_samples(self):
        assert drive_lane(LaneTracker(), 5, jitter=0.2).sdlp is None

    def test_offset_does_not_change_sdlp(self):
        """SDLP measures wander, not where in the lane you sit."""
        centred = drive_lane(LaneTracker(), 600, offset=0.0, jitter=0.2, seed=3).sdlp
        left = drive_lane(LaneTracker(), 600, offset=-0.5, jitter=0.2, seed=3).sdlp
        assert centred == pytest.approx(left, abs=0.01)

    def test_counts_departures(self):
        assert drive_lane(LaneTracker(), 100, jitter=0.1, departures=4).departures == 4

    def test_window_evicts_old_samples(self):
        from awas.road.signals import LaneConfig
        tr = LaneTracker(LaneConfig(window_sec=5.0))
        drive_lane(tr, 600, jitter=0.2, hz=10.0)
        assert tr.stats().samples <= 60

    def test_steering_reversals_are_counted(self):
        import math
        tr = LaneTracker()
        stats = drive_lane(tr, 600, jitter=0.05, hz=10.0,
                           steering=lambda i: 12.0 * math.sin(i / 6.0))
        assert stats.steering_reversal_rate > 0

    def test_deadband_ignores_sensor_noise(self):
        import random
        rng = random.Random(4)
        tr = LaneTracker()
        stats = drive_lane(tr, 600, jitter=0.05,
                           steering=lambda i: rng.gauss(0, 0.4))
        assert stats.steering_reversal_rate == 0

    def test_reset_clears(self):
        tr = LaneTracker()
        drive_lane(tr, 600, jitter=0.2)
        tr.reset()
        assert tr.stats().sdlp is None


# ---------------- workload ----------------

class TestWorkload:
    def run(self, speeds, **kw):
        est = WorkloadEstimator()
        w, t = None, 0.0
        for s in speeds:
            w = est.update(RoadState(t=t, speed_kmh=s, **kw))
            t += 1.0
        return w

    def test_stationary_is_zero(self):
        """Which is what lets the system talk at a red light."""
        assert self.run([0.0] * 10).value == 0.0

    def test_highway_cruise_is_moderate(self):
        v = self.run([90.0] * 20, road_class="highway").value
        assert 0.15 < v < 0.55

    def test_stop_go_beats_steady_at_the_same_road_class(self):
        steady = self.run([30.0] * 20, road_class="urban").value
        stopgo = self.run([5, 40, 10, 35, 0, 30] * 4, road_class="urban").value
        assert stopgo > steady

    def test_close_following_raises_it(self):
        far = self.run([90.0] * 20, road_class="highway",
                       vehicle_ahead_distance_m=60.0).value
        near = self.run([90.0] * 20, road_class="highway",
                        vehicle_ahead_distance_m=5.0).value
        assert near > far

    def test_recent_departure_spikes_it(self):
        calm = self.run([90.0] * 20, road_class="highway").value
        after = self.run([90.0] * 20, road_class="highway",
                         lane_departure=True).value
        assert after > calm + 0.3

    def test_never_exceeds_one(self):
        v = self.run([120.0] * 20, road_class="urban",
                     vehicle_ahead_distance_m=1.0, lane_departure=True).value
        assert v <= 1.0

    def test_explains_itself(self):
        """The gating has to be defensible, so it must say why it fired."""
        w = self.run([90.0] * 20, road_class="highway")
        assert "speed" in w.explain()
        assert w.parts

    def test_urban_demands_more_than_highway_at_equal_speed(self):
        urban = self.run([50.0] * 20, road_class="urban").value
        highway = self.run([50.0] * 20, road_class="highway").value
        assert urban > highway


# ---------------- fusion ----------------

class TestFusion:
    def test_agreement_raises_confidence_above_either_channel(self):
        f = facial_channel(3, 0.80, True)
        v = vehicle_channel(0.28, samples=200)      # level 3 on the new scale
        out = fuse([f, v])
        assert out.level == 3
        assert out.confidence > max(f.confidence, v.confidence)

    def test_disagreement_lowers_confidence(self):
        agree = fuse([facial_channel(3, 0.85, True), vehicle_channel(0.28, samples=200)])
        clash = fuse([facial_channel(3, 0.85, True), vehicle_channel(0.05, samples=200)])
        assert clash.confidence < agree.confidence

    def test_disagreement_pulls_the_level_between_the_channels(self):
        """A drowsy-looking driver holding the lane precisely is probably fine."""
        out = fuse([facial_channel(4, 0.85, True), vehicle_channel(0.05, samples=200)])
        assert 0 < out.level < 4

    def test_lost_face_lets_the_vehicle_channel_carry_it(self):
        out = fuse([facial_channel(0, 0.0, face_present=False),
                    vehicle_channel(0.36, samples=200)])
        assert out.level >= 3          # not pinned to the threshold table
        assert out.sources == ["vehicle"]

    def test_no_lane_data_leaves_the_facial_channel_alone(self):
        out = fuse([facial_channel(3, 0.85, True), vehicle_channel(None, samples=0)])
        assert out.level == 3
        assert out.sources == ["facial"]

    def test_a_single_channel_cannot_be_certain(self):
        out = fuse([facial_channel(3, 1.0, True), vehicle_channel(None, samples=0)])
        assert out.confidence <= SINGLE_CHANNEL_CAP

    def test_two_channels_can_exceed_the_single_channel_cap(self):
        out = fuse([facial_channel(3, 0.85, True), vehicle_channel(0.28, samples=200)])
        assert out.confidence > SINGLE_CHANNEL_CAP

    def test_no_channels_at_all_is_level_zero_no_confidence(self):
        out = fuse([facial_channel(0, 0.0, False), vehicle_channel(None, samples=0)])
        assert out.level == 0 and out.confidence == 0.0

    def test_agreement_is_none_with_one_channel(self):
        assert fuse([facial_channel(2, 0.8, True)]).agreement is None

    def test_lane_departure_forces_the_vehicle_channel_up(self):
        calm = vehicle_channel(0.10, samples=200)
        gone = vehicle_channel(0.10, departures=1, samples=200)
        assert gone.level >= 3 > calm.level

    def test_short_windows_are_less_confident(self):
        brief = vehicle_channel(0.30, samples=40)
        full = vehicle_channel(0.30, samples=400)
        assert full.confidence > brief.confidence

    def test_personal_baseline_changes_the_verdict(self):
        """Same SDLP reads differently against different drivers' baselines."""
        tidy = vehicle_channel(0.30, samples=200, sdlp_baseline_m=0.28)
        loose = vehicle_channel(0.30, samples=200, sdlp_baseline_m=0.10)
        assert loose.level > tidy.level

    def test_equal_excess_gives_equal_level_whatever_the_baseline(self):
        """The point of the personal scale, mirroring the EAR calibration."""
        a = vehicle_channel(0.12 + 0.08, samples=300, sdlp_baseline_m=0.12)
        b = vehicle_channel(0.20 + 0.08, samples=300, sdlp_baseline_m=0.20)
        assert a.level == pytest.approx(b.level, abs=0.05)

    def test_vehicle_channel_can_now_reach_level_three(self):
        """It previously saturated near 2.7 and could never raise a level-3."""
        assert vehicle_channel(0.28, samples=300).level >= 3.0
        assert vehicle_channel(0.24, samples=300, sdlp_baseline_m=0.12).level >= 3.0

    def test_explain_names_the_channels(self):
        out = fuse([facial_channel(3, 0.85, True), vehicle_channel(0.28, samples=200)])
        assert "facial" in out.explain() and "vehicle" in out.explain()


# ---------------- telemetry assembly ----------------

class TestTelemetryAssembly:
    def state(self, **kw):
        d = dict(t=10.0, drowsiness_level=2, confidence=0.8, face_present=True,
                 perclos=0.18, eye_aspect_ratio=0.19)
        d.update(kw)
        return DriverState(**d)

    def test_workload_reaches_the_state_machine(self):
        """The bug this layer exists to fix: workload was hardcoded to zero."""
        est = WorkloadEstimator()
        w, t = None, 0.0
        for _ in range(20):
            w = est.update(RoadState(t=t, speed_kmh=90, road_class="highway"))
            t += 1.0
        tel = build_telemetry(self.state(), RoadState(speed_kmh=90), None, w)
        assert tel.workload > 0.0

    def test_absent_road_channel_leaves_fields_none(self):
        tel = build_telemetry(self.state())
        assert tel.sdlp is None
        assert tel.parking_brake is None
        assert tel.vehicle_ahead_distance_m is None

    def test_fused_level_replaces_the_facial_level(self):
        lane = LaneTracker()
        stats = drive_lane(lane, 600, jitter=0.35, seed=5)
        tel = build_telemetry(self.state(drowsiness_level=1, confidence=0.8),
                              RoadState(speed_kmh=90), stats)
        assert tel.drowsiness_level >= 1
        assert tel.sdlp is not None

    def test_facial_only_path_still_works(self):
        from awas.sources.webcam import to_telemetry
        tel = to_telemetry(self.state(), speed_kmh=62.0, ignition_on=True)
        assert tel.speed_kmh == 62.0
        assert tel.sdlp is None
        assert tel.workload == 0.0
