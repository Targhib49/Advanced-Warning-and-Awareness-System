"""Measure this machine against the latency and thermal budgets (plan §7.2).

    python -m scripts.profile_device                  # everything
    python -m scripts.profile_device --landmarks      # verify eye indices only
    python -m scripts.profile_device --soak 120       # sustained thermal run

Run it on the Pi before writing a single number into the report. The budgets in
§7.2 were set on a development machine; the only figures worth publishing are
the ones measured on the hardware that will be in the vehicle.

Every stage degrades independently: no camera still profiles the model, no
phrase bank still profiles perception, no thermal sensor still profiles
latency. A partial answer beats refusing to run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from awas.runtime.platform import describe, device_info, soc_temperature_c

# From plan §7.2.
BUDGETS = {
    "perception_loop_ms": 100.0,
    "reflex_alert_ms": 300.0,
    "mesh_features_ms": 50.0,
}


def pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(int(len(s) * q), len(s) - 1)]


def verdict(measured: float, budget: float) -> str:
    if measured != measured:      # NaN
        return "not measured"
    return "PASS" if measured <= budget else f"OVER by {measured - budget:.1f} ms"


# ---------------- stages ----------------

def profile_landmarks(model: Path, frames: int, image: Optional[np.ndarray]) -> Dict:
    from awas.driver.landmarks import MediaPipeLandmarker

    if not model.exists():
        return {"skipped": f"no model at {model} — run scripts/fetch_models.sh"}

    lm = MediaPipeLandmarker(model)
    try:
        probe = image if image is not None else np.zeros((480, 640, 3), np.uint8)
        lm.observe(probe, 0.0)                       # warm up

        times, seen, ears = [], 0, []
        for i in range(frames):
            t0 = time.perf_counter()
            obs = lm.observe(probe, 1.0 + i * 0.033)
            times.append((time.perf_counter() - t0) * 1000.0)
            if obs.face_present:
                seen += 1
                ears.append(obs.ear)

        out = {
            "frames": frames,
            "faces_detected": seen,
            "p50_ms": round(statistics.median(times), 2),
            "p95_ms": round(pct(times, 0.95), 2),
            "max_ms": round(max(times), 2),
        }
        if ears:
            out["ear_median"] = round(statistics.median(ears), 4)
            out["ear_plausible"] = 0.10 <= statistics.median(ears) <= 0.60
        return out
    finally:
        lm.close()


def profile_audio(phrases: Path, repeats: int) -> Dict:
    from awas.speech.audio import StreamAudioOut

    if not (phrases / "manifest.json").exists():
        return {"skipped": f"no phrase bank at {phrases} — "
                           f"run scripts/render_phrases.py"}
    try:
        audio = StreamAudioOut(phrases)
    except Exception as exc:                          # no output device, etc.
        return {"skipped": f"audio unavailable: {exc}"}

    try:
        lat = []
        for _ in range(repeats):
            pb = audio.play(["tone__alarm"])
            lat.append(pb.latency_ms)
            time.sleep(max(pb.duration_ms / 1000.0, 0.05) + 0.05)
        return {
            "repeats": repeats,
            "p50_ms": round(statistics.median(lat), 2),
            "p95_ms": round(pct(lat, 0.95), 2),
            "max_ms": round(max(lat), 2),
        }
    finally:
        audio.close()


def profile_pipeline(seconds: float, fps: float) -> Dict:
    """The pure-logic half of the loop: calibration, window, level, hysteresis."""
    from awas.driver.landmarks import FaceObservation
    from awas.driver.pipeline import DriverPipeline

    pipe = DriverPipeline("profile")
    times, t = [], 0.0
    for i in range(int(seconds * fps)):
        obs = FaceObservation(t=t, face_present=True,
                              ear=0.09 if (i % 90) < 4 else 0.30)
        t0 = time.perf_counter()
        pipe.process(obs, t)
        times.append((time.perf_counter() - t0) * 1000.0)
        t += 1.0 / fps
    return {
        "ticks": len(times),
        "p50_ms": round(statistics.median(times), 3),
        "p95_ms": round(pct(times, 0.95), 3),
        "max_ms": round(max(times), 3),
    }


def soak(seconds: float, model: Path) -> Dict:
    """Hold the CPU busy and watch the die temperature.

    A Pi that passes a ten-second benchmark and thermally throttles after five
    minutes has not passed anything. Cabin ambient in Indonesia can exceed
    40 C — test at that ambient, not on a desk in air conditioning.
    """
    start_t = soc_temperature_c()
    if start_t is None:
        return {"skipped": "no thermal sensor on this platform"}

    from awas.driver.landmarks import MediaPipeLandmarker
    if not model.exists():
        return {"skipped": "no landmark model to load the CPU with"}

    lm = MediaPipeLandmarker(model)
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    samples, t0, i = [], time.time(), 0
    try:
        while time.time() - t0 < seconds:
            lm.observe(frame, time.time() - t0)
            i += 1
            if i % 30 == 0:
                temp = soc_temperature_c()
                if temp is not None:
                    samples.append(temp)
    finally:
        lm.close()

    return {
        "seconds": seconds,
        "frames": i,
        "start_c": round(start_t, 1),
        "end_c": round(samples[-1], 1) if samples else None,
        "peak_c": round(max(samples), 1) if samples else None,
        "sustained_fps": round(i / seconds, 1),
    }


def grab_face_frame(index: int, model: Path, countdown: int) -> Optional[np.ndarray]:
    """Capture a frame that actually contains a face.

    Three things go wrong if you just call read() once, and all three make the
    profiler report flattering nonsense rather than failing:

      * a webcam's first frames are black or badly exposed while auto-exposure
        settles, so the mesh finds nothing;
      * the operator has not sat down yet;
      * and with no face detected, MediaPipe skips the landmark stage entirely,
        so the measured cost is of the detector alone — which is the cheap half.

    So: warm up, count down, then keep looking until a face appears.
    """
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"camera {index} would not open; profiling on a blank frame\n")
        return None

    try:
        for _ in range(15):          # let exposure settle
            cap.read()

        for n in range(countdown, 0, -1):
            print(f"  capturing in {n}…", flush=True)
            deadline = time.time() + 1.0
            while time.time() < deadline:
                cap.read()

        from awas.driver.landmarks import MediaPipeLandmarker

        if not model.exists():
            ok, frame = cap.read()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

        lm = MediaPipeLandmarker(model)
        try:
            for attempt in range(60):
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                obs = lm.observe(rgb, time.time())
                if obs.face_present:
                    print(f"  face found (EAR {obs.ear:.3f}, "
                          f"pitch {obs.pitch_deg:+.0f}, yaw {obs.yaw_deg:+.0f})\n")
                    return rgb
        finally:
            lm.close()

        print("  ** NO FACE FOUND in 60 frames **")
        print("  Profiling anyway, but the landmark numbers will measure the")
        print("  detector alone and understate the real cost. Check the lens,")
        print("  the lighting, and that someone is actually in shot.\n")
        ok, frame = cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None
    finally:
        cap.release()


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Profile this device against §7.2.")
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--phrases", default="assets/phrases")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--audio-repeats", type=int, default=10)
    ap.add_argument("--soak", type=float, default=0.0,
                    help="seconds of sustained load for a thermal check")
    ap.add_argument("--landmarks", action="store_true",
                    help="only verify the landmark model and eye indices")
    ap.add_argument("--camera", type=int, default=None,
                    help="grab one real frame from this camera to profile against")
    ap.add_argument("--countdown", type=int, default=5,
                    help="seconds to get into shot before the frame is captured")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(describe())
    print()

    image = None
    if args.camera is not None:
        image = grab_face_frame(args.camera, Path(args.model), args.countdown)

    report: Dict = {"device": device_info().__dict__}

    print("landmarks (mesh + EAR + pose)")
    lm = profile_landmarks(Path(args.model), args.frames, image)
    report["landmarks"] = lm
    if "skipped" in lm:
        print(f"  skipped: {lm['skipped']}")
    else:
        print(f"  p50 {lm['p50_ms']:.2f} ms   p95 {lm['p95_ms']:.2f} ms   "
              f"faces {lm['faces_detected']}/{lm['frames']}")
        if lm["faces_detected"] == 0:
            print("  ** no face in any frame — this measures the detector only,")
            print("     not the mesh. Do not quote these numbers. **")
        print(f"  budget {BUDGETS['mesh_features_ms']:.0f} ms -> "
              f"{verdict(lm['p95_ms'], BUDGETS['mesh_features_ms'])}")
        if "ear_median" in lm:
            ok = "plausible" if lm["ear_plausible"] else "** OUT OF RANGE — check the indices **"
            print(f"  EAR median {lm['ear_median']:.4f}  {ok}")

    if args.landmarks:
        return

    print("\npipeline (calibration + window + level)")
    pl = profile_pipeline(seconds=20.0, fps=30.0)
    report["pipeline"] = pl
    print(f"  p50 {pl['p50_ms']:.3f} ms   p95 {pl['p95_ms']:.3f} ms")

    if "skipped" not in lm:
        loop_p95 = lm["p95_ms"] + pl["p95_ms"]
        report["loop_p95_ms"] = round(loop_p95, 2)
        print(f"\nperception loop p95  {loop_p95:.2f} ms   "
              f"budget {BUDGETS['perception_loop_ms']:.0f} ms -> "
              f"{verdict(loop_p95, BUDGETS['perception_loop_ms'])}")
        print(f"  sustainable frame rate: {1000.0 / max(loop_p95, 0.01):.1f} fps")

    print("\nreflex audio")
    au = profile_audio(Path(args.phrases), args.audio_repeats)
    report["audio"] = au
    if "skipped" in au:
        print(f"  skipped: {au['skipped']}")
    else:
        print(f"  p50 {au['p50_ms']:.2f} ms   max {au['max_ms']:.2f} ms   "
              f"budget {BUDGETS['reflex_alert_ms']:.0f} ms -> "
              f"{verdict(au['max_ms'], BUDGETS['reflex_alert_ms'])}")

    if args.soak > 0:
        print(f"\nthermal soak ({args.soak:.0f} s)")
        sk = soak(args.soak, Path(args.model))
        report["soak"] = sk
        if "skipped" in sk:
            print(f"  skipped: {sk['skipped']}")
        else:
            print(f"  {sk['start_c']} C -> {sk['end_c']} C "
                  f"(peak {sk['peak_c']} C) at {sk['sustained_fps']} fps")
            if sk["peak_c"] and sk["peak_c"] > 75:
                print("  ** above the 75 C ceiling — check cooling before trusting "
                      "any sustained figure **")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
