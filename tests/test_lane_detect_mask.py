"""Mask boxes: a simulator's HUD drawn over the road is not lane paint."""

import numpy as np

from awas.road.calibrate import load, save
from awas.road.lane_detect import LaneCalibration, LaneDetector, mask, synthetic_road


def test_mask_blacks_out_only_its_box():
    frame = np.full((100, 200, 3), 200, np.uint8)
    out = mask(frame, ((0.5, 0.5, 1.0, 1.0),))
    assert out[60:, 100:].max() == 0 and out[:50].min() == 200 and out[:, :100].min() == 200
    assert frame.min() == 200                                  # the original is untouched


def test_a_bright_overlay_on_the_road_is_ignored_once_masked():
    cal = LaneCalibration()
    road = synthetic_road(offset_m=0.0, cal=cal)
    hud = road.copy()
    hud[300:480, 560:640] = 255                                # a white HUD panel bottom right
    box = ((560 / 640, 300 / 480, 1.0, 1.0),)
    clean = LaneDetector(cal).detect(road)
    masked = LaneDetector(LaneCalibration(mask_boxes=box)).detect(hud)
    assert clean.found and masked.found
    assert abs(masked.offset_m - clean.offset_m) < 0.1


def test_mask_boxes_survive_save_and_load(tmp_path):
    cal = LaneCalibration(mask_boxes=((0.0, 0.77, 0.14, 1.0),))
    assert load(save(cal, tmp_path / "c.json")) == cal
