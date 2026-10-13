"""The Mac bridge's reader for bridge/mac/awas_telemetry.dylib's UDP lines.

The bridge is a standalone script (it is copied to the laptop alone), so it is
loaded by path rather than imported from the package.
"""

import importlib.util
import json
import socket
import time
from pathlib import Path

import pytest

BRIDGE = Path(__file__).resolve().parents[1] / "bridge" / "awas_bridge.py"


@pytest.fixture
def bridge():
    spec = importlib.util.spec_from_file_location("awas_bridge", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def reader(bridge):
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    r = bridge.PluginUdpTelemetry(port=port)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(**msg):
        out.sendto(json.dumps({"v": 1, **msg}).encode(), ("127.0.0.1", port))
        time.sleep(0.02)

    yield r, send
    r.sock.close()
    out.close()


def test_nothing_before_the_game_speaks(reader):
    r, _ = reader
    assert r.read() is None


def test_plugin_line_maps_to_the_windows_fields(reader):
    r, send = reader
    send(speed=-2.5, steer=0.12345, engine=True, paused=False, gear=-1,
         lblinker=False, rblinker=True, speed_limit=22.2222)
    d = r.read()
    assert d == {
        "source": "ets2", "speed_kmh": 9.0, "steer": 0.1235, "engine_on": True,
        "paused": False, "speed_limit_kmh": 80.0, "gear": -1,
        "blinker_left": False, "blinker_right": True,
    }


def test_no_speed_limit_is_none(reader):
    r, send = reader
    send(speed=10.0, speed_limit=0.0)
    assert r.read()["speed_limit_kmh"] is None


def test_silence_while_running_means_gone_but_paused_holds(reader):
    r, send = reader
    send(speed=10.0, paused=False)
    assert r.read() is not None
    r._at -= r.STALE_SEC + 1
    assert r.read() is None
    send(speed=0.0, paused=True)
    assert r.read()["paused"] is True
    r._at -= r.STALE_SEC + 1
    assert r.read()["paused"] is True


def test_bye_means_the_game_closed(reader):
    r, send = reader
    send(speed=10.0)
    send(bye=True)
    assert r.read() is None
