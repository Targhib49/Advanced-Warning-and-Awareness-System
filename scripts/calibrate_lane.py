"""Derive the lane trapezoid from real footage.

    python -m scripts.calibrate_lane --video dashcam.mp4
    python -m scripts.calibrate_lane --video clip.mov --lane-width 3.0 --save
    python -m scripts.calibrate_lane --self-test

Run this once per camera mounting. Move the camera, re-run it. A calibration
that does not match the mounting fails silently: detection collapses to nothing,
or worse, still produces offsets that are quietly scaled wrong.

Pick frames on a straight, well-marked stretch at steady speed. Bends, junctions
and lane changes all make the trapezoid look wrong when it is not.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from awas.road.calibrate import (
    auto_calibrate,
    find_vanishing_point,
    save,
    score_calibration,
)
from awas.road.lane_detect import LaneCalibration, LaneDetector, synthetic_road

DEFAULT_OUT = Path("config/lane_calibration.json")


def sample_video(path: str, want: int, skip_start: float = 0.15) -> List[np.ndarray]:
    """Take frames spread across the clip, skipping the first moments.

    Spread out rather than consecutive: fifty frames from one second all show
    the same geometry, and a calibration that suits one second of road is not
    a calibration.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames: List[np.ndarray] = []
    if total > want * 2:
        start = int(total * skip_start)
        step = max((total - start) // want, 1)
        for i in range(want):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + i * step)
            ok, frame = cap.read()
            if ok:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        while len(frames) < want:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise SystemExit(f"No frames read from {path}")
    return frames


def write_overlay(cal: LaneCalibration, frame: np.ndarray, path: Path) -> None:
    """Draw the trapezoid and the warped view, so the result can be eyeballed."""
    import cv2

    annotated = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    pts = cal.src_points().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(annotated, [pts], True, (60, 220, 90), 2)

    det = LaneDetector(cal)
    warped = det.warp(frame)
    binary = det.threshold(warped)
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    h = max(annotated.shape[0], binary_bgr.shape[0])
    def pad(img):
        out = np.zeros((h, img.shape[1], 3), np.uint8)
        out[: img.shape[0]] = img
        return out

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack([pad(annotated), pad(binary_bgr)]))


def report(label: str, cal: LaneCalibration, frames) -> None:
    print(f"  {label:24} {score_calibration(cal, frames)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the lane trapezoid.")
    ap.add_argument("--video", help="dashcam footage from the real mounting")
    ap.add_argument("--frames", type=int, default=24, help="frames to sample")
    ap.add_argument("--lane-width", type=float, default=3.5,
                    help="real lane width in metres (Indonesian arterials ~3.0-3.5)")
    ap.add_argument("--save", action="store_true", help=f"write {DEFAULT_OUT}")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--overlay", default="artifacts/lane_calibration.png")
    ap.add_argument("--mask", action="append", default=[], metavar="X0,Y0,X1,Y1",
                    help="blank this box (fractions of the frame) before detection, e.g. a "
                         "simulator's speedometer; repeat for several")
    ap.add_argument("--self-test", action="store_true",
                    help="prove the calibrator on synthetic footage")
    args = ap.parse_args()

    if args.self_test:
        truth = LaneCalibration(top_y=0.70, top_half_width=0.14,
                                bottom_half_width=0.62)
        frames = [synthetic_road(offset_m=o, cal=truth, noise=6.0, seed=i)
                  for i, o in enumerate([0.0, 0.15, -0.1, 0.25, -0.2, 0.05])]
        print("synthetic footage from a camera mounted unlike the default\n")
        report("default trapezoid", LaneCalibration(), frames)
        best, sc = auto_calibrate(frames, args.lane_width)
        report("after calibration", best, frames)
        print(f"\n  true   top_y={truth.top_y:.2f} "
              f"tw={truth.top_half_width:.2f} bw={truth.bottom_half_width:.2f}")
        print(f"  found  top_y={best.top_y:.2f} "
              f"tw={best.top_half_width:.2f} bw={best.bottom_half_width:.2f}")
        return

    if not args.video:
        ap.error("give --video with real footage, or --self-test")

    boxes = tuple(tuple(float(v) for v in m.split(",")) for m in args.mask)
    if any(len(b) != 4 for b in boxes):
        ap.error("--mask takes four fractions: x0,y0,x1,y1")
    frames = sample_video(args.video, args.frames)
    h, w = frames[0].shape[:2]
    print(f"{len(frames)} frames sampled from {args.video} ({w}x{h})\n")

    vp = find_vanishing_point(frames)
    print(f"  vanishing point   {f'({vp[0]:.0f}, {vp[1]:.0f}), horizon at {vp[1]/h:.2f}' if vp else 'not found'}")

    report("default trapezoid", LaneCalibration(width=w, height=h, lane_width_m=args.lane_width,
                                                mask_boxes=boxes), frames)
    best, sc = auto_calibrate(frames, args.lane_width, verbose=False, mask_boxes=boxes)
    report("after calibration", best, frames)

    print(f"\n  top_y              {best.top_y:.3f}")
    print(f"  top_half_width     {best.top_half_width:.3f}")
    print(f"  bottom_half_width  {best.bottom_half_width:.3f}")
    print(f"  lane_width_m       {best.lane_width_m:.2f}   (you supplied this)")

    if sc.detection_rate < 0.6:
        print("\n  ** Detection is still under 60%. Something is off beyond the")
        print("     trapezoid: check the footage is a straight, marked road at")
        print("     steady speed, and that the markings are actually visible. **")

    if args.overlay:
        write_overlay(best, frames[len(frames) // 2], Path(args.overlay))
        print(f"\n  overlay written to {args.overlay} — green is the trapezoid,")
        print("  right-hand panel is what the detector actually sees")

    if args.save:
        print(f"  saved {save(best, Path(args.out))}")
    else:
        print("\n  (add --save to keep it)")


if __name__ == "__main__":
    main()
