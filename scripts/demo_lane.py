"""Lane detection end to end, and the SDLP it produces.

    python -m scripts.demo_lane
    python -m scripts.demo_lane --video dashcam.mp4     # real footage
"""

from __future__ import annotations

import argparse
import random
import time

from awas.fusion.core import facial_channel, fuse, vehicle_channel
from awas.road.camera import CameraRoadSource
from awas.road.lane_detect import LaneDetector, synthetic_road
from awas.road.signals import LaneTracker


def synthetic_run() -> None:
    print("accuracy — render a known offset, detect it\n")
    det = LaneDetector()
    print(f"  {'true':>8} {'detected':>10} {'error':>8} {'conf':>6}")
    for true in (-0.8, -0.4, 0.0, 0.4, 0.8):
        r = det.detect(synthetic_road(offset_m=true, noise=8.0))
        if r.found:
            print(f"  {true:+8.2f} {r.offset_m:+10.3f} "
                  f"{abs(r.offset_m - true):8.3f} {r.confidence:6.2f}")
        else:
            print(f"  {true:+8.2f} {'not found':>10}   ({r.reason})")

    frame = synthetic_road(0.3, noise=8.0)
    for _ in range(5):
        det.detect(frame)
    t0 = time.perf_counter()
    for _ in range(100):
        det.detect(frame)
    print(f"\n  {(time.perf_counter() - t0) / 100 * 1000:.2f} ms per frame\n")

    print("SDLP, and what it does to the fused estimate\n")
    print(f"  {'driving':22} {'SDLP':>8}   fused with a face that says 'alert'")
    for label, wander in (("steady", 0.05), ("mild wander", 0.18), ("drifting", 0.35)):
        rng = random.Random(7)
        src, tracker, t = CameraRoadSource(speed_kmh=90.0), LaneTracker(), 0.0
        for _ in range(120):
            tracker.add(src.push(synthetic_road(offset_m=rng.gauss(0, wander)), t))
            t += 1 / 15.0
        st = tracker.stats()
        out = fuse([facial_channel(0, 0.85, True),
                    vehicle_channel(st.sdlp, samples=st.samples)])
        print(f"  {label:22} {st.sdlp:7.3f} m   {out.explain()}")

    print("\n  A face that looks fine plus a lane that is wandering is exactly")
    print("  the case facial-only detection misses — hypothesis H2.")


def video_run(path: str, max_frames: int) -> None:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")

    src, tracker = CameraRoadSource(speed_kmh=90.0), LaneTracker()
    found = total = 0
    t = 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    while total < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        state = src.push(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), t)
        tracker.add(state)
        if state.lane_offset_m is not None:
            found += 1
        t += 1.0 / fps
    cap.release()

    stats = tracker.stats()
    print(f"  frames        {total}")
    print(f"  lane found    {found} ({found / max(total, 1):.0%})")
    print(f"  SDLP          {stats.sdlp:.3f} m" if stats.sdlp
          else "  SDLP          not enough samples")
    if found / max(total, 1) < 0.5:
        print("\n  Under half the frames yielded a lane. The calibration almost")
        print("  certainly does not match this camera — adjust LaneCalibration's")
        print("  trapezoid to the mounting before trusting any of this.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lane detection demo.")
    ap.add_argument("--video", default=None, help="dashcam footage to run on")
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()
    if args.video:
        video_run(args.video, args.frames)
    else:
        synthetic_run()


if __name__ == "__main__":
    main()
