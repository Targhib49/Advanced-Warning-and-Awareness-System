"""--browser-audio: the dashboard page as the speaker of a speakerless demo."""

import json
import struct
import urllib.error
import urllib.request
import wave

import pytest

from awas.display.server import DisplayServer
from awas.runtime.executor import ActionExecutor
from awas.core.actions import Action, ActionType
from awas.speech.audio import BrowserAudioOut, NullAudioOut, TeeAudioOut


def _wav(path, seconds, rate=22050):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))


@pytest.fixture
def bank(tmp_path):
    _wav(tmp_path / "tone.wav", 0.5)
    _wav(tmp_path / "wake.wav", 1.5)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "tone__alarm": {"file": "tone.wav"},
        "alert__critical_drowsy": {"file": "wake.wav"},
    }))
    return tmp_path


@pytest.fixture
def server():
    s = DisplayServer(port=0, host="127.0.0.1").start()
    yield s
    s.close()


def _get(server, path):
    return urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}")


def test_no_sound_unless_asked_for(server):
    assert _get(server, "/state").headers.get("X-AWAS-Sound") is None
    with pytest.raises(urllib.error.HTTPError):
        _get(server, "/sound")


def test_the_page_learns_of_each_utterance_and_its_clips(server, bank):
    out = BrowserAudioOut(bank, server)
    assert _get(server, "/state").headers["X-AWAS-Sound"] == "0"
    pb = out.play(["tone__alarm", "alert__critical_drowsy"])
    assert pb.played and pb.duration_ms == pytest.approx(2000, abs=1)
    assert pb.latency_ms == -1.0                      # not measurable here; never faked
    assert _get(server, "/state").headers["X-AWAS-Sound"] == "1"
    assert json.loads(_get(server, "/sound").read()) == {
        "seq": 1, "clips": ["tone__alarm", "alert__critical_drowsy"]}
    wav = _get(server, "/phrases/alert__critical_drowsy.wav")
    assert wav.headers["Content-Type"] == "audio/wav" and wav.read()[:4] == b"RIFF"


def test_only_clips_in_the_bank_are_served(server, bank):
    BrowserAudioOut(bank, server)
    for path in ("/phrases/nope.wav", "/phrases/../manifest.wav", "/phrases/manifest.json"):
        with pytest.raises(urllib.error.HTTPError):
            _get(server, path)


def test_an_unknown_clip_is_reported_unplayed(server, bank):
    pb = BrowserAudioOut(bank, server).play(["say__nothing__0"])
    assert not pb.played
    assert _get(server, "/state").headers["X-AWAS-Sound"] == "0"


def test_the_executor_keeps_its_interrupt_rule_through_the_page(server, bank):
    ex = ActionExecutor(audio=BrowserAudioOut(bank, server))
    [crit] = ex.execute([Action(type=ActionType.ALERT, urgency=4, alert_id="critical_drowsy")], t=0)
    assert crit.performed
    # Something less urgent while the alarm is still sounding waits its turn.
    [low] = ex.execute([Action(type=ActionType.ALERT, urgency=3, alert_id="critical_drowsy")], t=0.1)
    assert low.suppressed.startswith("urgency 3 < playing 4")


def test_tee_reports_the_device_and_feeds_the_page(server, bank):
    device = NullAudioOut()
    tee = TeeAudioOut(device, BrowserAudioOut(bank, server))
    tee.play(["tone__alarm"])
    assert device.history and _get(server, "/state").headers["X-AWAS-Sound"] == "1"
