"""Centre-screen app: screen choice, trip numbers, and the local server."""

import json
import urllib.error
import urllib.request

import pytest

from awas.core.states import DrivingState
from awas.display.model import TripTracker, choose_screen, is_display_worthy, snapshot
from awas.display.server import DisplayServer
from awas.driver.pipeline import DriverState

MOVING = DrivingState.MOVING


class TestChooseScreen:
    def test_levels_map_to_their_screens(self):
        names = [choose_screen(MOVING, lvl, True, False) for lvl in range(5)]
        assert names == ["monitoring", "tired", "engaging", "insist", "critical"]

    def test_parked_outranks_everything(self):
        assert choose_screen(DrivingState.PARKED, 4, False, True, "workload") == "parked"
        assert choose_screen(DrivingState.SHUTDOWN, 0, True, False) == "parked"

    def test_critical_outranks_a_lost_face(self):
        assert choose_screen(MOVING, 4, False, False) == "critical"

    def test_lost_face_is_said_plainly(self):
        assert choose_screen(MOVING, 2, False, False) == "noface"

    def test_held_alert_shows_the_gate_only_when_there_is_a_level(self):
        assert choose_screen(MOVING, 2, True, False, "workload 0.86") == "gate"
        assert choose_screen(MOVING, 0, True, False, "workload 0.86") == "monitoring"

    def test_calibrating_until_personal_scale_exists(self):
        assert choose_screen(MOVING, 0, True, True) == "calibrating"


def test_housekeeping_suppressions_are_not_shown_as_gates():
    assert not is_display_worthy("")
    assert not is_display_worthy("muted")
    assert not is_display_worthy("clip missing from the bank")
    assert not is_display_worthy("urgency 1 < playing 3")
    assert is_display_worthy("workload 0.82")


class TestTripTracker:
    def test_distance_and_time_only_accrue_while_moving(self):
        trip = TripTracker()
        trip.update(0.0, 0, 60.0, moving=True)
        trip.update(60.0, 0, 60.0, moving=True)      # one minute at 60 km/h
        trip.update(120.0, 0, 0.0, moving=False)     # parked minute
        assert trip.driving_sec == pytest.approx(60.0)
        assert trip.distance_km == pytest.approx(1.0)

    def test_chart_keeps_the_worst_level_in_each_bucket(self):
        trip = TripTracker(bucket_sec=10.0)
        for t, lvl in [(0, 0), (3, 2), (6, 1), (10, 0), (21, 4), (25, 0)]:
            trip.update(float(t), lvl, 50.0, moving=True)
        assert trip.chart() == [0.5, 0.0, 1.0]
        assert trip.worst() == (2, 2)

    def test_no_worst_stretch_on_an_uneventful_trip(self):
        trip = TripTracker()
        trip.update(0.0, 0, 50.0, moving=True)
        assert trip.worst() is None

    def test_long_trips_are_downsampled_to_fit_the_chart(self):
        trip = TripTracker(bucket_sec=1.0)
        for t in range(500):
            trip.update(float(t), 4 if t == 250 else 0, 50.0, moving=True)
        pts = trip.chart(max_points=60)
        assert len(pts) == 60 and max(pts) == 1.0


def test_snapshot_is_plain_json():
    driver = DriverState(t=1.0, face_present=True, calibrating=False, confidence=0.9)
    snap = snapshot(driving_state=MOVING, level=1, driver=driver, speed_kmh=62.4,
                    workload=0.31, lane_available=False, held_reason=None,
                    trip=TripTracker(), kss=None, now=0.0)
    assert snap["screen"] == "tired"
    assert snap["speed_kmh"] == 62
    assert json.loads(json.dumps(snap)) == snap


class TestServer:
    @pytest.fixture
    def server(self):
        s = DisplayServer(port=0, host="127.0.0.1").start()
        yield s
        s.close()

    def _url(self, server, path):
        return f"http://127.0.0.1:{server.port}{path}"

    def test_serves_page_renderer_and_state(self, server):
        server.publish({"screen": "monitoring", "level": 0})
        page = urllib.request.urlopen(self._url(server, "/")).read()
        js = urllib.request.urlopen(self._url(server, "/app-screens.js")).read()
        state = json.loads(urllib.request.urlopen(self._url(server, "/state")).read())
        assert b"drawScreen" in page and b"export function drawScreen" in js
        assert state == {"screen": "monitoring", "level": 0}

    def test_accepts_known_commands(self, server):
        req = urllib.request.Request(self._url(server, "/command"), data=b'{"kss": 6}',
                                     headers={"Content-Type": "application/json"})
        assert urllib.request.urlopen(req).status == 204
        assert server.next_command() == {"kss": 6}
        assert server.next_command() is None

    def test_camera_view_does_not_exist_unless_asked_for(self, server):
        server.publish_frame(b"\xff\xd8 a frame")              # dropped: view is off
        for path in ("/camera", "/camera.mjpg"):
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(self._url(server, path))
            assert err.value.code == 404

    def test_camera_view_streams_when_enabled(self):
        s = DisplayServer(port=0, host="127.0.0.1", camera_view=True).start()
        try:
            s.publish_frame(b"\xff\xd8 frame one")
            base = f"http://127.0.0.1:{s.port}"
            assert b"/camera.mjpg" in urllib.request.urlopen(base + "/camera").read()
            stream = urllib.request.urlopen(base + "/camera.mjpg", timeout=3)
            assert "multipart/x-mixed-replace" in stream.headers["Content-Type"]
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(200)
            while b"frame one" not in chunk:
                chunk += stream.read1(4096)
            assert b"--frame" in chunk
            stream.close()
        finally:
            s.close()

    def test_rejects_anything_else(self, server):
        req = urllib.request.Request(self._url(server, "/command"), data=b'{"shutdown": true}')
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        assert err.value.code == 400
        assert server.next_command() is None
