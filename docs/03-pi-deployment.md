# Raspberry Pi Deployment

Milestone 5. Everything here is written and testable; **nothing in it has run
on a Pi yet**, because there was no Pi to run it on. Treat every number as a
prediction until `scripts/profile_device.py` has produced one on the board.

---

## 1. The finding that shapes this whole port

**mediapipe stopped publishing aarch64 Linux wheels after 0.10.18.**

| release | aarch64 Linux wheel |
|---|---|
| 0.10.35 — what macOS runs and what the code is validated against | **no** |
| 0.10.18 — newest 0.10.x with one | yes (cp39–cp312) |
| 1.0.x | yes, but see below |

So the Pi cannot run the same mediapipe as the development machine. There is no
pin that satisfies both, which is why `requirements-pi.txt` exists separately.

**Choice: 0.10.18.** Same Tasks API, same model asset, closest to what has
actually been validated. 1.0.x is available for aarch64 and would probably work
— its known defect is macOS-specific (it forces the Metal delegate, and Linux
has no Metal) — but moving two major versions away from the validated stack to
save nothing is a bad trade on a deadline.

**Consequence, and it is the first thing to check on the device:** eye landmark
indices have moved between mediapipe releases before. `MEDIAPIPE_LEFT_EYE` and
`MEDIAPIPE_RIGHT_EYE` in `awas/driver/ear.py` were verified against 0.10.35 on a
live face. They are **not** verified against 0.10.18.

```bash
./venv/bin/python -m scripts.profile_device --landmarks --camera 0
```

The EAR median must land roughly between 0.10 and 0.60, and the profiler says
so explicitly. A wrong index does not raise — it produces a plausible-looking
number that means nothing, and every measurement downstream inherits the error.

---

## 2. What to install

Raspberry Pi 5, 8 GB, **64-bit** Pi OS Bookworm. A 32-bit image finds no
mediapipe wheel at all.

```bash
git clone <repo> ~/awas && cd ~/awas
./scripts/setup_pi.sh
```

That installs system libraries (`libportaudio2` for audio, `libgl1` and
`libglib2.0-0` for OpenCV, `libatlas-base-dev` for numpy), creates the venv from
`requirements-pi.txt`, fetches the landmark model with its pinned checksum, runs
the tests, and profiles the device. It is idempotent.

**No torch on the Pi.** Inference only: the model arrives as ONNX, exported on a
development machine by `training/export.py`. That removes about a gigabyte of
dependency from the target.

### Speech

Piper is the Pi's offline TTS. Two ways to get a phrase bank:

```bash
PIPER_VOICE=/path/to/voice.onnx ./scripts/setup_pi.sh   # render on device
```

or simply copy `assets/phrases/` from a machine that has already rendered it —
the WAVs are portable, and this is a perfectly good shortcut for the
competition. Rendering happens at build time in either case, never while
driving, which is why the engines can differ between platforms at all.

Piper is also where **Indonesian** enters: pass an `id_ID` voice and the whole
bank re-renders in Indonesian with no other code change. The English text in
`awas/companion/phrasebank.py` is the only thing that would need translating.

---

## 3. Run it

```bash
./venv/bin/python -m scripts.run_webcam --driver <name>              # with a window
./venv/bin/python -m scripts.run_webcam --driver <name> --headless   # without
```

As a boot service:

```bash
sudo cp deploy/awas.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now awas
journalctl -u awas -f
```

The service handles SIGTERM properly, so `systemctl stop` closes the session log
with a `session_end` record instead of truncating it. `Restart=always` because a
safety system that dies quietly is worse than one that never started.

---

## 4. Profile before believing anything

```bash
./venv/bin/python -m scripts.profile_device --camera 0 --soak 600 --out artifacts/pi_profile.json
```

Reference figures from the development machine (Apple M1 Pro), for comparison —
**not** targets:

| stage | M1 Pro | budget (§7.2) |
|---|---|---|
| mesh + EAR + pose, p95 | 1.98 ms | 50 ms |
| pipeline logic, p95 | 0.01 ms | — |
| perception loop, p95 | 1.99 ms | 100 ms |
| reflex audio, worst | 10.6 ms | 300 ms |

Those were measured on blank frames, where no face is found and the mesh stage
is skipped. **Use `--camera 0` so a real face is in the frame**, or the numbers
flatter the system.

### The soak matters more than the benchmark

A Pi that passes a ten-second benchmark and thermally throttles after five
minutes has not passed anything. `--soak 600` holds the CPU busy and watches the
die temperature.

Cabin ambient in Indonesia can exceed 40 °C. **Test at that ambient**, not on a
desk in air conditioning — the plan calls for a sustained SoC temperature below
75 °C, and a bare Pi 5 under sustained inference will not manage it. The active
cooler in the revised BOM is not optional.

Also watch under-voltage: `vcgencmd get_throttled` should read `0x0`. Anything
else usually means the power supply, and it makes a Pi behave strangely in ways
that look like software bugs.

---

## 5. What is still unresolved

**The CSI camera path.** `cv2.VideoCapture` handles USB cameras. The Pi Camera
Module over CSI generally needs `picamera2`/libcamera, which is a different API.
A USB camera avoids the problem entirely and is the pragmatic choice for the
competition; the CSI path is worth doing only if the enclosure demands it.

**The IR illuminator.** The BOM's NoIR camera sees infrared but nothing in the
BOM emits any, so night operation does not work as specified (master plan §7.3).
Night is half the use case and the half where drowsiness matters most. This is a
hardware purchase, not a software fix.

**OBD-II.** Not needed. `parking_brake` is `Optional` and `None` already means
"no OBD-II present" throughout; GPS speed is the primary path by design
(§7.4).

**Audio device selection.** `config/pi.yaml` leaves it `null` for the system
default, which is usually right. If it is not, `sounddevice.query_devices()`
lists the indices.

---

## 6. Honest status

Written and locally verified: the platform-detection module, the device
profiler, headless mode with signal handling, the Piper renderer, the setup
script, the systemd unit, and the Pi configuration profile.

Not verified: any of it on actual Pi hardware. The first hour on the board will
find things this document does not predict. Start with the landmark check in §1
— everything else is recoverable, and that one silently poisons the data.
