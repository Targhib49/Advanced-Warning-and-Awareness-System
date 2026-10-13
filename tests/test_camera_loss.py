"""A missing or unplugged camera is a fault to show, not a reason to exit."""

import numpy as np
import pytest

from awas.core.states import DrivingState
from awas.display.model import TripTracker, choose_screen, snapshot
from awas.driver.landmarks import FaceObservation, ScriptedLandmarker
from awas.driver.pipeline import DriverState
from awas.sources.webcam import WebcamSource

NO_SUCH_CAMERA = 97


class FakeCap:
    """A capture that works until told to fail."""

    def __init__(self):
        self.working = True
        self.released = False

    def read(self):
        if self.working:
            return True, np.zeros((480, 640, 3), np.uint8)
        return False, None

    def release(self):
        self.released = True


@pytest.fixture
def src(monkeypatch):
    s = WebcamSource(camera_index=NO_SUCH_CAMERA,
                     landmarker=ScriptedLandmarker([FaceObservation(t=0, face_present=True, ear=0.3)] * 500))
    monkeypatch.setattr("time.sleep", lambda _s: None)      # keep the tests fast
    yield s
    s.close()


def test_a_missing_camera_does_not_stop_the_system(src):
    assert not src.camera_ok and src.cap is None
    tel = src.read()                                        # still produces telemetry
    assert tel.face_present is False


def test_an_unplugged_camera_is_declared_lost_then_recovers(src, monkeypatch):
    cap = FakeCap()
    src.cap, src.camera_ok = cap, True
    src.read()
    assert src.camera_ok

    cap.working = False
    for _ in range(WebcamSource.FAILURES_TO_LOSE - 1):
        src.read()
    assert src.camera_ok                                    # a few bad frames are not a loss
    src.read()
    assert not src.camera_ok and cap.released and src.cap is None

    # Plugged back in: the next retry finds it.
    fresh = FakeCap()

    def reopen():
        src.cap, src.camera_ok, src._read_failures = fresh, True, 0
        return True

    monkeypatch.setattr(src, "_open_camera", reopen)
    src._next_open_at = 0.0
    src.read()
    assert src.camera_ok and src.cap is fresh


def test_a_video_file_that_will_not_open_still_fails_loudly():
    with pytest.raises(RuntimeError):
        WebcamSource(camera_index="/no/such/clip.mp4",
                     landmarker=ScriptedLandmarker([]))


def test_no_camera_outranks_every_other_screen():
    for ds, lvl in ((DrivingState.MOVING, 0), (DrivingState.MOVING, 3), (DrivingState.PARKED, 0)):
        assert choose_screen(ds, lvl, False, False, lane_available=True, camera_ok=False) == "nocamera"
    assert choose_screen(DrivingState.MOVING, 0, True, False, camera_ok=True) == "monitoring"


def test_snapshot_reports_the_camera():
    snap = snapshot(driving_state=DrivingState.MOVING, level=0, driver=DriverState(t=0.0),
                    speed_kmh=60, workload=None, lane_available=False, held_reason=None,
                    trip=TripTracker(), kss=None, camera_ok=False, now=0.0)
    assert snap["screen"] == "nocamera" and snap["camera_ok"] is False
