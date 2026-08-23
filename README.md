# AWAS v2 — Advanced Warning & Awareness System

A retrofit driver-safety system that treats driving safety as a **human
cognitive-system problem**, not a vehicle-sensor problem. It fuses driver-state
monitoring with road context and intervenes through adaptive conversation —
a safety control mechanism, not entertainment — entirely offline on a
Raspberry Pi 5.

Research competition project, 2026 · Targhib & Fachry

## Read this first

Two documents, read in order:

**[`docs/00-master-plan.md`](docs/00-master-plan.md)** — thesis, evaluation
protocol, hardware, ethics, roadmap, risks. Section 1 lists everything that
changed from the original project brief and why.

**[`docs/03-pi-deployment.md`](docs/03-pi-deployment.md)** — Raspberry Pi
setup, profiling and the mediapipe version problem that shapes the port.

**[`docs/02-study-protocol.md`](docs/02-study-protocol.md)** — how to actually
run the H1 study: safety and ethics, session runbook, consent materials, and a
pre-registered analysis plan.

**[`docs/01-architecture-v3.md`](docs/01-architecture-v3.md)** — the current
architecture, superseding §3-5 of the master plan. A five-tier learned ML stack
in place of threshold detection, plus per-driver personalization, voice-channel
fatigue sensing, fatigue prediction, and learned intervention policy. Written
against a commercial bar rather than a competition one.

## Status

Milestone 1 — laptop webcam demo, end to end:

- [x] Core state machine: escalation ladder, per-level cooldowns, workload
      gating, confidence downgrade, occlusion capping
- [x] Mock telemetry source + scripted-drive demo
- [x] T1 eye-state training pipeline — MRL loader, subject-wise split,
      baselines, ONNX + int8 export, all verified
- [x] Per-driver eye calibration — personal openness scale, no ML
- [x] Layer 1 pipeline — landmarks, head pose, PERCLOS window, hysteresis,
      level estimation, webcam source, debug overlay (172 tests)
- [x] T1 trained on the full MRL set — **0.9859** on 7 unseen subjects vs
      0.8682 for logistic regression on pixels; int8 export costs no accuracy
- [x] Live camera verified end to end on a real face
- [x] Reflex alerts + template dialogue, all pre-rendered — **25 ms worst
      case against the 300 ms budget**
- [x] Session logging, with an analysis script that computes H1's primary
      outcome from it (214 tests)
- [x] Experimental conditions A/B/C/D wired end to end, so the same
      perception can drive different responses for a fair comparison
- [x] Raspberry Pi port — setup script, systemd service, headless mode,
      device profiler, Piper TTS (written, not yet run on hardware)
- [ ] Run the pilot study (see `docs/02-study-protocol.md`)
- [ ] Verify on actual Pi hardware, starting with the landmark indices
- [ ] Local LLM for dialogue variation (Milestone 4)
- [ ] Fusion core with hysteresis
- [ ] Reflex alert path, latency-measured
- [ ] Dialogue manager + template bank
- [ ] TTS output
- [ ] Live debug overlay
- [ ] Structured session logging

Milestones 2–6 (detection quality, simulator, companion LLM, Pi deployment,
study) are in the plan, §9.

## Running

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m scripts.render_phrases              # pre-render the phrase bank, once
python -m scripts.demo_state_machine --speak  # scripted drive, out loud
python -m scripts.demo_calibration            # why a fixed EAR threshold fails
python -m pytest tests/ -q                    # test suite
```

After a session, see what it bought you:

```bash
python -m scripts.analyze_session data/sessions/*.jsonl
python -m scripts.analyze_session --compare data/sessions/*.jsonl   # by condition
```

Run a study session under a specific condition:

```bash
python -m scripts.run_webcam --conditions          # list them
python -m scripts.run_webcam --driver P03 --condition B
```

The live camera path needs a face landmark model — mediapipe ships none:

```bash
./scripts/fetch_models.sh
python -m scripts.run_webcam --driver yourname
```

### Recorded clips beat the live camera

`--video` runs the identical pipeline over a file, and for a judged demo that
is the better option: reproducible, rehearsable, and it cannot fail on stage.

```bash
python -m scripts.run_webcam --driver yourname --video clips/drowsy.mov
```

Record with QuickTime Player or Photo Booth — apps that already hold camera
permission. A useful clip is about three minutes: a minute of alert,
eyes-forward driving posture so calibration completes, then progressively
heavier eyelids, then two or three closures held past two seconds to exercise
the critical path.

### Live camera

Needs camera permission for **the app that owns the session**, which is not
necessarily a terminal. On macOS, TCC attributes camera access to the
responsible app bundle, and an app only appears in **System Settings → Privacy
& Security → Camera** after it has successfully requested access.

Two traps worth knowing:

- **Terminal.app declares no `NSCameraUsageDescription`**, so it will never
  appear in that list. Looking for it is a dead end. VS Code does declare it.
- **A venv's `python` is ad-hoc signed** — `python -m venv` copies the binary
  and re-signs it without a team identifier. Under a hardened-runtime parent it
  does not inherit the parent's camera entitlement, so the request never
  registers with TCC and no list entry is ever created. This looks exactly like
  "permission denied" but no dialog appears and nothing can be toggled.

Two pins in `requirements.txt` exist for reasons that are not obvious and will
look like arbitrary caution later:

- **`mediapipe<1.0`** — the 1.0.x macOS arm64 wheels force the Metal delegate
  inside the face detector and abort the process on CPU regardless of the
  delegate you request. 0.10.x has a working CPU path. There is a version guard
  in `awas/driver/landmarks.py` that turns this into a readable error.
- **`numpy<2.0`** — several wheels in this stack are still built against 1.x
  and crash at import when mixed with 2.x.

Train and export the eye-state model (no dataset needed for the smoke run):

```bash
python -m training.train_eyestate --smoke   # verifies the whole pipeline
python -m training.export                   # ONNX + int8, numerically checked
```

See [`training/README.md`](training/README.md) for the real MRL run and the
Kaggle path.

## Layout

```
awas/
  core/        telemetry, states, actions, config, state machine — pure logic
  driver/      Layer 1  driver state monitoring (camera)
  road/        Layer 2  road & vehicle context
  fusion/      fusion core: drowsiness level + confidence + workload
  companion/   Layer 3  dialogue manager, template bank, LLM
  sources/     sensor adapters (mock, webcam, OBD, GPS, simulator)
  speech/      TTS out, STT in
  runtime/     main loop, logging
training/      T1 model training, baselines, ONNX export
deploy/        systemd unit for running on a Pi
design/        stage-by-stage UI mockups (.dc.html artboards)
docs/          the plan
config/        tunable thresholds
```

## Design invariants

These are enforced by code and checked by tests — see plan §3.

1. **No LLM output ever gates an emergency response.** Every line lives in
   `awas/companion/phrasebank.py` and is rendered to audio before the system
   runs, so there is no code path from a generator to the speaker at all.
   Measured worst case: 25 ms.
2. **The state machine is pure.** No I/O, no clock reads, no model calls.
   Time and sensor data arrive as arguments.
3. **Every intervention is rate-limited and logged** with the telemetry that
   triggered it.
4. **The system degrades, never crashes.** A dead sensor lowers confidence and
   narrows behaviour; it does not stop the loop.
5. **Raw video never leaves the device** and is never written to disk outside
   an explicit, opt-in research recording session.
