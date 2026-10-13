"""Simulator road source: bridge datagrams in, RoadState out."""

import json
import socket
import time

import cv2
import pytest

from awas.road.lane_detect import LaneCalibration, synthetic_road
from awas.road.sim import MAGIC, SimConfig, SimRoadSource


def tel(**kw):
    d = {"speed_kmh": 80.0, "steer": 0.1, "engine_on": True, "paused": False, "speed_limit_kmh": 90}
    d.update(kw)
    return MAGIC + b"T" + json.dumps(d).encode()


def frame(offset_m):
    ok, buf = cv2.imencode(".jpg", synthetic_road(offset_m=offset_m), [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return MAGIC + b"F" + buf.tobytes()


@pytest.fixture
def sim():
    s = SimRoadSource(SimConfig(port=0), calibration=LaneCalibration(), bind="127.0.0.1")
    yield s
    s.close()


def test_nothing_received_means_nothing_known(sim):
    state = sim.read(1.0)
    assert state.speed_source == "none" and state.speed_kmh == 0.0
    assert state.lane_offset_m is None
    assert sim.ignition_on is None and not sim.connected


def test_bench_run_without_a_game_uses_the_manual_speed(sim):
    sim.manual_speed_kmh = 60.0
    state = sim.read(1.0)
    assert state.speed_kmh == 60.0 and state.speed_source == "manual"
    sim.handle(tel(speed_kmh=80.0))                        # a game connects: it wins
    assert sim.read(1.1).speed_source == "sim"


def test_telemetry_becomes_speed_steering_and_ignition(sim):
    assert sim.handle(tel(speed_kmh=-72.0, steer=0.5, engine_on=False))
    state = sim.read(1.0)
    assert state.speed_source == "sim"
    assert state.speed_kmh == pytest.approx(72.0)          # reversing still counts as speed
    assert state.steering_angle_deg == pytest.approx(225.0)   # 0.5 of a 900° wheel's half-range
    assert state.speed_limit_kmh == 90.0
    assert sim.ignition_on is False


def test_paused_game_is_a_stopped_truck(sim):
    sim.handle(tel(paused=True))
    assert sim.read(1.0).speed_kmh == 0.0


def test_stale_telemetry_is_dropped(sim):
    sim.handle(tel())
    sim._telemetry_at -= 5.0
    assert sim.read(1.0).speed_source == "none"
    assert sim.ignition_on is None


@pytest.mark.parametrize("offset", [-0.4, 0.0, 0.3])
def test_road_frames_give_lane_position(sim, offset):
    sim.handle(tel())
    sim.handle(frame(offset))
    state = sim.read(1.0)
    assert state.lane_offset_m == pytest.approx(offset, abs=0.1)


def test_each_frame_is_used_once(sim):
    sim.handle(tel())
    sim.handle(frame(0.2))
    assert sim.read(1.0).lane_offset_m is not None
    assert sim.read(1.1).lane_offset_m is None              # no new frame, no new sample


def test_frames_of_another_size_are_scaled_to_the_calibration(sim):
    big = cv2.resize(synthetic_road(offset_m=0.25), (1280, 720))
    ok, buf = cv2.imencode(".jpg", big)
    sim.handle(tel())
    sim.handle(MAGIC + b"F" + buf.tobytes())
    assert sim.read(1.0).lane_offset_m is not None


@pytest.mark.parametrize("junk", [b"", b"hello", MAGIC + b"T{bad json", MAGIC + b"T[1,2]",
                                  MAGIC + b"Fnot a jpeg", MAGIC + b"X{}"])
def test_junk_is_rejected(sim, junk):
    assert not sim.handle(junk)
    assert sim.counts["rejected"] == 1


def test_real_socket_round_trip_with_acknowledgement(sim):
    port = sim._sock.getsockname()[1]
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2.0)
    try:
        client.sendto(tel(speed_kmh=55.0), ("127.0.0.1", port))
        data, _ = client.recvfrom(4096)
        assert data[:5] == MAGIC + b"A"
        assert json.loads(data[5:])["telemetry"] == 1
        deadline = time.time() + 2
        while not sim.connected and time.time() < deadline:
            time.sleep(0.01)
        assert sim.read(1.0).speed_kmh == pytest.approx(55.0)
    finally:
        client.close()


# Bench drive 6: an uncalibrated game view, crawling through a town, gave an
# SDLP of 1.2 m and "check your lane" every five seconds.

def test_slow_driving_gives_no_lane_sample(sim):
    sim.handle(tel(speed_kmh=12.0))
    sim.handle(frame(0.9))
    state = sim.read(1.0)
    assert state.lane_offset_m is None and not state.lane_departure
    assert sim.status()["lane_found"]                       # still shown to the bridge


def test_an_uncalibrated_view_gives_no_lane_sample(sim):
    sim.camera.calibrated = False
    sim.handle(tel())
    sim.handle(frame(0.2))
    assert sim.read(1.0).lane_offset_m is None
    assert sim.status()["lane_found"]
