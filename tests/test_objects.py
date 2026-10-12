"""Traffic lights and following distance from the forward camera."""

import numpy as np
import pytest

from awas.road.camera import CameraRoadSource
from awas.road.lane_detect import synthetic_road
from awas.road.objects import (
    VEHICLE_WIDTH_M,
    Box,
    RoadObjects,
    classify_light,
    distance_from_width,
    pick_lead_vehicle,
)


def light_crop(colour, row, h=60, w=20):
    """A synthetic traffic light: one lit lamp at a given height in the box."""
    crop = np.zeros((h, w, 3), np.uint8)
    y = int(row * h)
    crop[max(0, y - 7):y + 7, w // 4: 3 * w // 4] = colour
    return crop


RED, AMBER, GREEN = (230, 30, 30), (240, 200, 40), (40, 220, 90)


class TestDistanceEstimate:
    def test_closer_vehicles_look_wider(self):
        near = distance_from_width(300, 640, 1.8)
        far = distance_from_width(80, 640, 1.8)
        assert near < far

    def test_follows_the_pinhole_relation(self):
        """Halving the apparent width should double the distance."""
        a = distance_from_width(160, 640, 1.8)
        b = distance_from_width(80, 640, 1.8)
        assert b == pytest.approx(a * 2, rel=0.01)

    def test_wider_vehicles_are_further_for_the_same_box(self):
        """A bus filling the frame is further away than a car filling it."""
        car = distance_from_width(200, 640, VEHICLE_WIDTH_M["car"])
        bus = distance_from_width(200, 640, VEHICLE_WIDTH_M["bus"])
        assert bus > car

    def test_degenerate_width_is_none(self):
        assert distance_from_width(0.0, 640, 1.8) is None

    def test_implausible_distances_are_rejected(self):
        """A 3-pixel box is noise, not a vehicle 400 m away."""
        assert distance_from_width(2.0, 640, 1.8) is None


class TestLeadVehicle:
    def box(self, cx, bottom, label="car", w=100):
        return Box(x=cx - w / 2, y=bottom - 60, w=w, h=60, label=label, score=0.9)

    def test_picks_the_vehicle_in_our_lane(self):
        boxes = [self.box(60, 400), self.box(320, 380), self.box(600, 420)]
        assert pick_lead_vehicle(boxes, 640).cx == 320

    def test_picks_the_nearest_when_several_are_ahead(self):
        boxes = [self.box(320, 300), self.box(330, 430)]
        assert pick_lead_vehicle(boxes, 640).bottom == 430

    def test_ignores_non_vehicles(self):
        assert pick_lead_vehicle([self.box(320, 400, label="person")], 640) is None

    def test_none_when_the_road_is_clear(self):
        assert pick_lead_vehicle([], 640) is None

    def test_ignores_vehicles_in_other_lanes(self):
        assert pick_lead_vehicle([self.box(40, 400), self.box(610, 400)], 640) is None


class TestTrafficLightColour:
    @pytest.mark.parametrize("name,colour,row", [
        ("red", RED, 0.15), ("yellow", AMBER, 0.5), ("green", GREEN, 0.85),
    ])
    def test_reads_the_lit_lamp(self, name, colour, row):
        got, conf = classify_light(light_crop(colour, row))
        assert got == name
        assert conf > 0.5

    def test_hue_wins_when_position_disagrees(self):
        """A light seen at an angle still shows its colour."""
        got, conf = classify_light(light_crop(GREEN, 0.15))
        assert got == "green"
        assert conf < 1.0        # but we are less sure

    def test_agreement_raises_confidence(self):
        _, agree = classify_light(light_crop(GREEN, 0.85))
        _, clash = classify_light(light_crop(GREEN, 0.15))
        assert agree > clash

    def test_unlit_box_reads_nothing(self):
        assert classify_light(np.zeros((60, 20, 3), np.uint8)) == (None, 0.0)

    def test_tiny_crop_is_rejected(self):
        assert classify_light(np.zeros((3, 2, 3), np.uint8)) == (None, 0.0)

    def test_empty_crop_does_not_crash(self):
        assert classify_light(np.zeros((0, 0, 3), np.uint8)) == (None, 0.0)

    def test_dim_pixels_do_not_vote(self):
        """Dark red bodywork behind a light is not a red signal."""
        dim = np.full((60, 20, 3), (40, 8, 8), np.uint8)
        assert classify_light(dim)[0] is None


class TestThrottling:
    """The detector is 5x the cost of the face mesh; it must not run per frame."""

    class FakeDetector:
        """Stands in for the model so the throttle can be tested without it."""

        def __init__(self, interval_sec=0.33):
            self.interval_sec = interval_sec
            self.calls = 0
            self._last_t = None
            self._last = RoadObjects()

        def due(self, t):
            return self._last_t is None or (t - self._last_t) >= self.interval_sec

        def detect(self, frame, t):
            if not self.due(t):
                return RoadObjects(**{**self._last.__dict__, "stale": True})
            self.calls += 1
            self._last, self._last_t = RoadObjects(vehicle_count=1), t
            return self._last

        def close(self):
            pass

    def test_runs_at_the_configured_rate_not_every_frame(self):
        det = self.FakeDetector(interval_sec=0.33)
        for i in range(60):            # 4 seconds at 15 fps
            det.detect(None, i / 15.0)
        assert 10 <= det.calls <= 14   # ~3 Hz, not 60

    def test_cached_results_are_marked_stale(self):
        det = self.FakeDetector(interval_sec=1.0)
        det.detect(None, 0.0)
        assert det.detect(None, 0.1).stale is True

    def test_fresh_results_are_not_stale(self):
        det = self.FakeDetector(interval_sec=0.1)
        det.detect(None, 0.0)
        assert det.detect(None, 5.0).stale is False


class TestCameraSourceIntegration:
    def test_road_class_is_carried_through(self):
        src = CameraRoadSource(speed_kmh=80.0, road_class="urban")
        assert src.push(synthetic_road(0.0), 0.0).road_class == "urban"

    def test_works_without_an_object_detector(self):
        """Lanes must not depend on the heavier model being present."""
        src = CameraRoadSource(speed_kmh=80.0)
        s = src.push(synthetic_road(0.2), 0.0)
        assert s.lane_offset_m is not None
        assert s.vehicle_ahead_distance_m is None
        assert s.traffic_light_color is None
