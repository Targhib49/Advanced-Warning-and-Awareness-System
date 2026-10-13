"""Driver registry, guided calibration, and the screens before driving."""

import pytest

from awas.driver.calibration import CalibrationConfig, DriverProfile
from awas.driver.guided import Frame, GuidedCalibration, GuidedConfig, profile_from
from awas.runtime.drivers import DriverNameError, DriverRegistry, clean_name
from awas.runtime.setup_flow import SetupFlow

FPS = 15.0
GOOD = dict(face=True, eye_px=30.0, yaw=0.0, pitch=0.0, brightness=120.0)


def frames(t0, seconds, ear, p_closed=None, **kw):
    """Constant-state frames at FPS; returns (frames, end time)."""
    n = int(seconds * FPS)
    args = dict(GOOD, **kw)
    return [Frame(t=t0 + i / FPS, ear=ear, p_closed=p_closed, **args) for i in range(n)], t0 + n / FPS


# The two hard cases, (seconds, EAR, P(closed)), from the bench gaze test with
# the camera off to the side: looking down at the lap lowers EAR a little and
# fools the CNN into "closed"; nodding off lowers EAR fully and fools it into "open".
CNN_FOOLED = ((7.5, 0.22, 0.95), (7.5, 0.16, 0.30))
# The mirror image: EAR fooled both ways, the CNN right.
EAR_FOOLED = ((7.5, 0.15, 0.04), (7.5, 0.26, 0.95))
# Every signal right.
ALL_RIGHT = ((7.5, 0.24, 0.04), (7.5, 0.15, 0.95))


def run_session(cal, cnn=True, look_away_at_open=False, hard=CNN_FOOLED):
    """A cooperative driver: positions, looks ahead (with blinks), closes
    twice, then looks down and nods off."""
    t, seq = 0.0, []
    def add(sec, ear, p, **kw):
        nonlocal t
        f, t = frames(t, sec, ear, p if cnn else None, **kw)
        seq.extend(f)
    add(2.5, 0.26, 0.05)                                   # position, held
    for _ in range(4):                                      # open, with blinks
        add(1.8, 0.27, 0.04)
        add(0.15, 0.12, 0.95)
    if look_away_at_open:
        add(2.0, 0.26, 0.05, yaw=40.0)                      # pauses, keeps what it had
    add(2.0, 0.26, 0.05)
    for _ in range(2):                                      # two deliberate closures
        add(2.0, 0.15, 0.93)
        add(1.0, 0.26, 0.05)
    for sec, ear, p in hard:                                # look down, then nod off
        add(sec, ear, p)
    for f in seq:
        cal.feed(f)
    return cal


# ---------------------------------------------------------------- registry

def test_names_are_cleaned_and_policed():
    assert clean_name("  Budi   Santoso ") == "Budi Santoso"
    for bad in ("", "x" * 25, "Rm -rf /", "<script>"):
        with pytest.raises(DriverNameError):
            clean_name(bad)


def test_ids_are_unique_slugs(tmp_path):
    reg = DriverRegistry(tmp_path)
    a = reg.create("Siti Aminah")
    assert a.id == "siti-aminah"
    with pytest.raises(DriverNameError):
        reg.create("siti aminah")                           # same person, twice
    assert reg.create("Default").id == "default-driver"    # reserved word
    assert DriverRegistry(tmp_path).get("siti-aminah").name == "Siti Aminah"   # persisted


def test_delete_takes_profile_and_sessions_with_it(tmp_path):
    reg = DriverRegistry(tmp_path)
    d = reg.create("Budi")
    other = reg.create("Budi Dua")
    (tmp_path / "profiles").mkdir()
    (tmp_path / "sessions").mkdir()
    (tmp_path / "profiles" / "budi.json").write_text("{}")
    (tmp_path / "sessions" / "20261013T000000Z_budi_abc.jsonl").write_text("")
    (tmp_path / "sessions" / "20261013T000000Z_budi-dua_def.jsonl").write_text("")
    assert reg.delete(d.id) == 2
    assert reg.get("budi") is None and reg.get(other.id) is not None
    assert (tmp_path / "sessions" / "20261013T000000Z_budi-dua_def.jsonl").exists()


# ---------------------------------------------------------------- guided calibration

def test_a_cnn_fooled_by_the_hard_cases_loses_to_ear():
    # The bench result: both separate open from closed facing the road, only
    # EAR gets looking down and nodding off right.
    cal = run_session(GuidedCalibration())
    r = cal.result
    assert r.ok and r.eye_signal == "ear"
    assert r.open_ear == pytest.approx(0.27, abs=0.01)     # blinks ignored by the median
    assert r.closed_ear == pytest.approx(0.15)
    assert r.cnn_accuracy > 0.9 and cal.rounds_done == 2   # it passed the easy part
    # "open" includes the driver's ordinary blinks, so no signal scores 1.0 there.
    assert r.scores["ear"]["score"] > 0.95 and r.scores["ear"]["cases"]["nod"] == 1.0
    assert r.scores["cnn"]["cases"]["down"] == 0.0 and r.scores["cnn"]["cases"]["nod"] == 0.0
    assert r.down_frames > 0 and r.nod_frames > 0


def test_an_ear_fooled_by_the_hard_cases_loses_to_the_cnn():
    r = run_session(GuidedCalibration(), hard=EAR_FOOLED).result
    assert r.ok and r.eye_signal == "cnn"


def test_when_every_signal_is_right_the_simplest_is_kept():
    r = run_session(GuidedCalibration(), hard=ALL_RIGHT).result
    assert r.ok and r.eye_signal == "ear"
    assert len({v["score"] for v in r.scores.values()}) == 1      # a three-way tie


def test_when_no_signal_is_reliable_calibration_fails_with_a_reason():
    worst = ((7.5, 0.15, 0.95), (7.5, 0.27, 0.04))          # every signal wrong both times
    r = run_session(GuidedCalibration(), hard=worst).result
    assert not r.ok and "no eye signal" in r.reason


def test_no_cnn_falls_back_to_ear():
    r = run_session(GuidedCalibration(), cnn=False).result
    assert r.ok and r.eye_signal == "ear"


def test_a_cnn_that_cannot_tell_falls_back_to_ear():
    cal = GuidedCalibration()
    t, seq = 0.0, []
    for sec, ear in ((2.5, 0.26), (9.0, 0.26), (2.0, 0.15), (1.0, 0.26), (2.0, 0.15), (1.0, 0.26),
                     (7.5, 0.22), (7.5, 0.16)):
        f, t = frames(t, sec, ear, p_closed=0.5)            # the CNN shrugs at everything
        seq += f
    for f in seq:
        cal.feed(f)
    assert cal.result.ok and cal.result.eye_signal == "ear"


def test_position_must_hold_before_starting():
    cal = GuidedCalibration()
    f, _ = frames(0.0, 5.0, 0.26, eye_px=10.0)              # too far away
    for x in f:
        cal.feed(x)
    assert cal.step == "position" and not cal.checks["distance"] and cal.checks["face"]


def test_looking_away_pauses_without_losing_progress():
    assert run_session(GuidedCalibration(), look_away_at_open=True).result.ok


def test_blinks_do_not_count_as_closed_rounds():
    cal = GuidedCalibration()
    t, seq = 0.0, []
    for sec, ear in ((2.5, 0.26), (9.0, 0.26)):
        f, t = frames(t, sec, ear); seq += f
    for _ in range(10):                                     # ten ordinary blinks
        f, t = frames(t, 0.2, 0.12); seq += f
        f, t = frames(t, 1.0, 0.26); seq += f
    for f in seq:
        cal.feed(f)
    assert cal.step == "closed" and cal.rounds_done == 0


def test_never_closing_times_out_with_a_reason():
    cal = GuidedCalibration(GuidedConfig(closed_timeout_sec=5.0))
    f, _ = frames(0.0, 20.0, 0.26)
    for x in f:
        cal.feed(x)
    assert cal.result is not None and not cal.result.ok and "not detected" in cal.result.reason


def test_profile_from_a_guided_result_is_usable_at_once():
    r = run_session(GuidedCalibration()).result
    p = profile_from(r, "budi")
    cfg = CalibrationConfig()
    assert isinstance(p, DriverProfile) and p.is_usable(cfg)
    assert p.closed_ref == pytest.approx(0.15)


# ---------------------------------------------------------------- the flow

def test_new_driver_end_to_end(tmp_path):
    flow = SetupFlow(DriverRegistry(tmp_path))
    flow.command({"new_driver": True})
    flow.command({"name": "Budi"})
    assert flow.screen == "consent"
    flow.command({"consent": True})
    assert flow.screen == "calibrating" and flow.camera_wanted
    t, seq = 0.0, []
    for sec, ear, p in ((2.5, .26, .05), (9.0, .26, .05), (2.0, .15, .93), (1.0, .26, .05),
                        (2.0, .15, .93), (1.0, .26, .05), *CNN_FOOLED):
        f, t = frames(t, sec, ear, p); seq += f
    for f in seq:
        flow.feed(f, tmp_path / "profiles")
    assert flow.screen == "result" and flow.driver.calibrated and not flow.camera_wanted
    assert (tmp_path / "profiles" / "budi.json").exists()
    events = flow.command({"start": True})
    assert flow.driving and events == [("drive", flow.driver)]


def test_declining_consent_keeps_nothing(tmp_path):
    reg = DriverRegistry(tmp_path)
    flow = SetupFlow(reg)
    flow.command({"new_driver": True})
    flow.command({"name": "Tamu"})
    flow.command({"consent": False})
    assert flow.screen == "select" and reg.list() == []


def test_bad_name_stays_on_the_screen_with_a_message(tmp_path):
    flow = SetupFlow(DriverRegistry(tmp_path))
    flow.command({"new_driver": True})
    flow.command({"name": "   "})
    assert flow.screen == "new" and flow.error


def test_returning_driver_can_start_or_recalibrate(tmp_path):
    reg = DriverRegistry(tmp_path)
    d = reg.create("Siti")
    reg.record_consent(d.id)
    reg.record_calibration(d.id, "cnn", {"ok": True})
    flow = SetupFlow(reg)
    flow.command({"select_driver": d.id})
    assert flow.screen == "ready" and not flow.camera_wanted
    flow.command({"calibrate": True})
    assert flow.screen == "calibrating"
    flow.command({"back": True})
    assert flow.screen == "select" and reg.get(d.id) is not None   # back never deletes a real driver


def test_start_needs_a_calibrated_driver(tmp_path):
    reg = DriverRegistry(tmp_path)
    d = reg.create("Andi")
    reg.record_consent(d.id)
    flow = SetupFlow(reg)
    flow.command({"select_driver": d.id})
    assert flow.screen == "calibrating"                     # uncalibrated: straight to calibration
    assert flow.command({"start": True}) == [] and not flow.driving


def test_switching_driver_ends_the_drive(tmp_path):
    reg = DriverRegistry(tmp_path)
    d = reg.create("Siti")
    reg.record_consent(d.id)
    reg.record_calibration(d.id, "ear", {"ok": True})
    flow = SetupFlow(reg)
    assert flow.drive_now(d.id) == [("drive", reg.get(d.id))]
    assert flow.command({"switch_driver": True}) == [("end_drive", reg.get(d.id))]
    assert flow.screen == "select" and flow.driver is None


def test_commands_out_of_place_are_ignored(tmp_path):
    flow = SetupFlow(DriverRegistry(tmp_path))
    assert flow.command({"start": True}) == [] and flow.screen == "select"
    assert flow.command({"consent": True}) == [] and flow.screen == "select"


def test_cnn_alone_when_ear_cannot_separate():
    # Open and closed EAR nearly equal (lids not tracked), the CNN sure of both.
    cal = GuidedCalibration()
    t, seq = 0.0, []
    for sec, ear, p in ((2.5, .26, .05), (9.0, .26, .05), (2.0, .25, .95), (1.0, .26, .05),
                        (2.0, .25, .95), (1.0, .26, .05), (7.5, .26, .04), (7.5, .26, .95)):
        f, t = frames(t, sec, ear, p); seq += f
    for f in seq:
        cal.feed(f)
    assert cal.result.ok and cal.result.eye_signal == "cnn"


def test_snapshot_reports_how_long_the_eyes_have_been_closed(tmp_path):
    reg = DriverRegistry(tmp_path)
    d = reg.create("Andi"); reg.record_consent(d.id)
    flow = SetupFlow(reg)
    flow.command({"select_driver": d.id})
    t, seq = 0.0, []
    for sec, ear, p in ((2.5, .26, .05), (9.0, .26, .05), (1.5, .15, .93)):
        f, t = frames(t, sec, ear, p); seq += f
    for f in seq:
        flow.feed(f, tmp_path / "profiles")
    c = flow.snapshot()["calibration"]
    assert c["closing"] and c["closed_for"] >= 1.3 and c["min_closed_sec"] > 0
