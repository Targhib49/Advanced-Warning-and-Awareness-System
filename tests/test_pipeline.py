"""Layer 1 end to end, without a camera or a model file."""

import math

import pytest

from awas.core.machine import StateMachine
from awas.core.states import DrivingState
from awas.driver.calibration import CalibrationConfig
from awas.driver.landmarks import FaceObservation
from awas.driver.pipeline import DriverPipeline
from awas.driver.window import LevelConfig, WindowConfig
from awas.sources.webcam import to_telemetry

FPS = 30.0
OPEN_EAR, SHUT_EAR = 0.34, 0.09


def drive(pipe, seconds, droop=0.0, t0=0.0, face=True, fps=FPS,
          open_ear=OPEN_EAR, shut_ear=SHUT_EAR, blink_len=4, blink_every=90):
    """Feed the pipeline a stretch of driving and return the last state."""
    span = open_ear - shut_ear
    base = open_ear - droop * span
    t, state = t0, None
    for i in range(int(seconds * fps)):
        blinking = (i % blink_every) < blink_len
        ear = shut_ear if blinking else base
        obs = FaceObservation(t=t, face_present=face, ear=ear if face else 0.0)
        state = pipe.process(obs, t)
        t += 1.0 / fps
    return state, t


class TestCalibrationInThePipeline:
    def test_starts_uncalibrated(self):
        s, _ = drive(DriverPipeline("a"), 1.0)
        assert s.calibrating is True
        assert s.personalized is False

    def test_progress_climbs(self):
        p = DriverPipeline("a")
        early, t = drive(p, 5.0)
        late, _ = drive(p, 20.0, t0=t)
        assert late.calibration_progress > early.calibration_progress

    def test_becomes_personalized(self):
        s, _ = drive(DriverPipeline("a"), 60.0)
        assert s.personalized is True
        assert s.calibrating is False

    def test_learns_this_driver_not_the_population(self):
        p = DriverPipeline("wide")
        drive(p, 60.0, open_ear=0.40, shut_ear=0.11)
        assert p.normalizer.open_ref == pytest.approx(0.40, abs=0.03)
        assert p.normalizer.closed_ref == pytest.approx(0.11, abs=0.03)

    def test_confidence_rises_once_calibrated(self):
        p = DriverPipeline("a")
        early, t = drive(p, 3.0)
        late, _ = drive(p, 60.0, t0=t)
        assert late.confidence > early.confidence


class TestLevelResponse:
    def test_alert_driver_stays_at_zero(self):
        s, _ = drive(DriverPipeline("a"), 90.0, droop=0.0)
        assert s.drowsiness_level == 0

    def test_heavy_droop_raises_the_level(self):
        p = DriverPipeline("a")
        _, t = drive(p, 60.0, droop=0.0)                 # calibrate while alert
        s, _ = drive(p, 90.0, droop=0.85, t0=t, blink_len=14, blink_every=60)
        assert s.drowsiness_level >= 2

    def test_a_long_closure_escalates_immediately(self):
        """No waiting for hysteresis on a two-second closure."""
        p = DriverPipeline("a")
        _, t = drive(p, 60.0)
        for i in range(int(3.0 * FPS)):                  # 3 s eyes shut
            s = p.process(FaceObservation(t=t, face_present=True, ear=SHUT_EAR), t)
            t += 1.0 / FPS
        assert s.drowsiness_level == 4

    def test_level_ticks_at_update_hz_not_frame_rate(self):
        """3 escalate ticks must mean ~3 seconds, not 3 frames."""
        p = DriverPipeline("a", lvl_cfg=LevelConfig(escalate_ticks=3))
        _, t = drive(p, 60.0)
        for i in range(3):                               # 3 frames of droop
            s = p.process(FaceObservation(t=t, face_present=True, ear=0.12), t)
            t += 1.0 / FPS
        assert s.drowsiness_level == 0


class TestDegradation:
    def test_losing_the_face_does_not_invent_a_microsleep(self):
        p = DriverPipeline("a")
        _, t = drive(p, 60.0)
        s, _ = drive(p, 30.0, t0=t, face=False)
        assert s.microsleep_count == 0
        assert s.drowsiness_level == 0

    def test_confidence_falls_when_the_face_is_gone(self):
        p = DriverPipeline("a")
        seen, t = drive(p, 60.0)
        gone, _ = drive(p, 30.0, t0=t, face=False)
        assert gone.confidence < seen.confidence

    def test_drowsy_frames_are_kept_out_of_the_baseline(self):
        """Calibrating on a drowsy stretch would desensitise the system."""
        p = DriverPipeline("a", cal_cfg=CalibrationConfig(calibration_window_sec=1e9))
        _, t = drive(p, 60.0, droop=0.0)
        clean = p.normalizer.open_ref
        drive(p, 120.0, droop=0.9, t0=t, blink_len=20, blink_every=50)
        assert p.normalizer.open_ref == pytest.approx(clean, abs=0.02)


class TestProfilePersistence:
    def test_profile_survives_a_trip(self, tmp_path):
        p1 = DriverPipeline("hendra", profile_dir=tmp_path)
        drive(p1, 60.0, open_ear=0.40, shut_ear=0.11)
        assert p1.save_profile() is not None

        p2 = DriverPipeline("hendra", profile_dir=tmp_path)
        assert p2.normalizer.personalized is True
        assert p2.normalizer.open_ref == pytest.approx(0.40, abs=0.03)

    def test_second_trip_starts_personalized(self, tmp_path):
        p1 = DriverPipeline("rina", profile_dir=tmp_path)
        drive(p1, 60.0)
        p1.save_profile()
        first_state, _ = drive(DriverPipeline("rina", profile_dir=tmp_path), 1.0)
        assert first_state.personalized is True

    def test_unknown_driver_gets_no_profile(self, tmp_path):
        assert DriverPipeline("stranger", profile_dir=tmp_path).normalizer.personalized is False


class TestTelemetryBridge:
    def test_maps_onto_the_snapshot_the_state_machine_reads(self):
        p = DriverPipeline("a")
        s, _ = drive(p, 60.0)
        t = to_telemetry(s, speed_kmh=62.0, ignition_on=True)
        assert t.drowsiness_level == s.drowsiness_level
        assert t.confidence == s.confidence
        assert t.perclos == s.perclos
        assert t.speed_kmh == 62.0

    def test_reports_no_obd(self):
        """Phase 1 has no OBD-II; None is how the state machine is told."""
        t = to_telemetry(DriverPipeline("a").process(
            FaceObservation(t=0.0, face_present=True, ear=0.3), 0.0), 60.0, True)
        assert t.parking_brake is None
        assert t.speed_source == "manual"


class TestFullLoop:
    def test_drowsy_driver_reaches_the_state_machine(self):
        """Webcam path -> telemetry -> state machine -> an actual action."""
        p = DriverPipeline("a")
        sm = StateMachine(initial_state=DrivingState.MOVING)
        _, t = drive(p, 60.0, droop=0.0)

        fired = []
        for i in range(int(120 * FPS)):
            blinking = (i % 50) < 18
            ear = SHUT_EAR if blinking else 0.13
            state = p.process(FaceObservation(t=t, face_present=True, ear=ear), t)
            fired.extend(sm.update(to_telemetry(state, 62.0, True)))
            t += 1.0 / FPS

        assert any(a.type.value in ("speak", "alert") for a in fired)
