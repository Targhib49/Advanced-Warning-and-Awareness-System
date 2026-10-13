"""The Phase 1 demo: webcam -> Layer 1 -> state machine -> actions.

    python -m scripts.run_webcam
    python -m scripts.run_webcam --driver budi --video clip.mp4

Keys:  q quit   [ / ] speed down/up   i toggle ignition   c clear profile

The overlay is the competition demo view. It shows the gating decisions, not
just the outcome — "L1 suppressed, workload 0.82" is what distinguishes this
from a beeper, and a judge cannot see it any other way.
"""

from __future__ import annotations

import argparse
import signal
import sys
from collections import deque
from pathlib import Path

from awas.companion.dialogue import DialogueManager
from awas.core.actions import ActionType
from awas.core.machine import StateMachine
from awas.core.states import DrivingState
from awas.display.model import TripTracker, is_display_worthy, snapshot
from awas.driver.ear import MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE
from awas.runtime.conditions import describe_all, get as get_condition
from awas.runtime.executor import ActionExecutor
from awas.runtime.session_log import SessionLog
from awas.sources.webcam import WebcamSource, default_model_path
from awas.speech.audio import NullAudioOut, StreamAudioOut

BG = (13, 12, 10)
DIM = (91, 138, 124)
TEXT = (239, 237, 232)
LEVEL_COLORS = [(155, 211, 56), (60, 169, 224), (62, 178, 242), (62, 122, 242), (56, 68, 240)]
LEVEL_NAMES = ["AWAKE", "TIRED", "DROWSY", "UNSAFE", "PULL OVER"]


def draw(cv2, frame, state, sm, telemetry, log, fps, fused=None):
    h, w = frame.shape[:2]
    # The level the state machine acts on (fused, then confidence- and
    # occlusion-gated), not the facial channel's opinion alone.
    lvl = sm.effective_level(telemetry)
    color = LEVEL_COLORS[min(lvl, 4)]

    if state.face_present and state.landmarks_drawn:
        pass  # landmarks are drawn by the caller, which owns the point list

    panel_h = 172
    cv2.rectangle(frame, (0, h - panel_h), (w, h), BG, -1)
    cv2.rectangle(frame, (0, 0), (w, 6), color, -1)

    def put(text, x, y, scale=0.5, col=TEXT, weight=1):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, weight, cv2.LINE_AA)

    y0 = h - panel_h + 26
    put(LEVEL_NAMES[min(lvl, 4)], 14, y0, 0.8, color, 2)
    put(f"conf {state.confidence:.2f}", 200, y0, 0.5, DIM)

    seg_w = (w - 28) // 5
    for i in range(5):
        c = LEVEL_COLORS[i] if i <= lvl else (34, 40, 43)
        x = 14 + i * seg_w
        cv2.rectangle(frame, (x, y0 + 12), (x + seg_w - 6, y0 + 20), c, -1)

    y1 = y0 + 46
    if state.eye_source == "cnn" and state.p_closed is not None:
        put(f"CNN P(closed) {state.p_closed:.2f}", 14, y1, 0.45, (155, 211, 56))
        put(f"EAR {state.eye_aspect_ratio:.3f}", 190, y1, 0.45, DIM)
    else:
        put(f"EAR {state.eye_aspect_ratio:.3f}", 14, y1, 0.45, DIM)
        put("(EAR only)", 120, y1, 0.45, DIM)
    put(f"PERCLOS {state.perclos:.1%}", 290, y1, 0.45, DIM)
    put(f"micro {state.microsleep_count}", 420, y1, 0.45, DIM)
    put(f"blink {state.blink_duration_mean_ms:.0f}ms", 505, y1, 0.45, DIM)

    y2 = y1 + 22
    put(f"{sm.state.value}", 14, y2, 0.45, DIM)
    put(f"{telemetry.speed_kmh:.0f} km/h", 150, y2, 0.45, DIM)
    put(f"{fps:.1f} fps", 240, y2, 0.45, DIM)
    put(f"pitch {state.head_pitch_deg:+.0f} yaw {state.head_yaw_deg:+.0f}", 330, y2, 0.45, DIM)

    if state.calibrating:
        pct = state.calibration_progress
        put(f"calibrating {pct:.0%}", 14, y2 + 22, 0.45, (60, 169, 224))
        cv2.rectangle(frame, (160, y2 + 14), (160 + int(200 * pct), y2 + 20), (60, 169, 224), -1)
    else:
        scale = "population CNN" if state.eye_source == "cnn" else "personal scale"
        put(f"{scale}  {state.reason}", 14, y2 + 22, 0.45, (155, 211, 56))

    # How the level that is acted on was reached: each channel's opinion and
    # whether they agree. This is the line that answers "why that level?".
    if fused is not None:
        lane = f"   SDLP {telemetry.sdlp:.3f} m" if telemetry.sdlp is not None else ""
        put(f"fusion: {fused.explain()}{lane}", 14, y2 + 44, 0.4, TEXT)

    for i, line in enumerate(list(log)[-3:]):
        put(line, 14, y2 + 62 + i * 15, 0.38, DIM)


CAMERA_VIEW_FPS = 10.0


def overlay_frame(cv2, raw, state, sm, telemetry, log, fps, mirror=True, fused=None):
    """The debug view: the camera, the 12 EAR points, and the readout panel.
    Draws on a copy; the source frame is left alone."""
    frame = cv2.flip(raw, 1) if mirror else raw.copy()
    h, w = frame.shape[:2]
    if state.face_present and state.landmarks:
        for idx in list(MEDIAPIPE_LEFT_EYE) + list(MEDIAPIPE_RIGHT_EYE):
            x, y = state.landmarks[idx]
            if mirror:
                x = w - x
            cv2.circle(frame, (int(x), int(y)), 2, LEVEL_COLORS[min(state.drowsiness_level, 4)], -1)
    state.landmarks_drawn = True
    draw(cv2, frame, state, sm, telemetry, log, fps, fused=fused)
    return frame


def _apply_commands(display, src, session, t, log, kss_given):
    """Drain what the screen sent. Speed and ignition stand in for GPS and the
    ignition line until the vehicle signals exist; a KSS tap is ground truth."""
    while (cmd := display.next_command()) is not None:
        if "speed_delta" in cmd:
            src.speed_kmh = max(0.0, src.speed_kmh + float(cmd["speed_delta"]))
        if "ignition" in cmd:
            src.ignition_on = not src.ignition_on
            if src.ignition_on:
                kss_given = None          # a new trip asks again
        if "kss" in cmd:
            k = int(cmd["kss"])
            if 1 <= k <= 9:
                kss_given = k
                if session:
                    session.annotation(t, kind="kss", value=k)
                log.append(f"t={t:6.1f} kss   {k} (screen)")
    return kss_given


def main() -> None:
    ap = argparse.ArgumentParser(description="AWAS Phase 1 webcam demo.")
    ap.add_argument("--driver", default="default")
    ap.add_argument("--video", default=None, help="run on a file instead of the camera")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--profiles", default="data/profiles")
    ap.add_argument("--model", default=None)
    ap.add_argument("--phrases", default="assets/phrases")
    ap.add_argument("--sessions", default="data/sessions")
    ap.add_argument("--condition", default="C",
                    help="experimental condition A/B/C/D — see --conditions")
    ap.add_argument("--conditions", action="store_true",
                    help="list the experimental conditions and exit")
    ap.add_argument("--mute", action="store_true", help="decide and log, but stay silent")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--headless", action="store_true",
                    help="no window — for a Pi running as a boot service")
    ap.add_argument("--dashboard", type=int, default=0, metavar="PORT",
                    help="serve the centre-screen app on this port (e.g. 8095); 0 = off")
    ap.add_argument("--eye-signal", choices=("cnn", "ear"), default="cnn",
                    help="eye openness from the T1 CNN (default; falls back to EAR per frame) "
                         "or from EAR alone, the v2 control condition")
    ap.add_argument("--eye-model", default="artifacts/eyestate_fp32.onnx",
                    help="the exported T1 eye-state model (training/export.py)")
    ap.add_argument("--fusion", choices=("weighted", "asymmetric"), default="weighted",
                    help="how the face and lane channels combine (results §D-3): weighted "
                         "suppresses false alarms the lane contradicts; asymmetric keeps "
                         "early warnings. A deployment decision — no mode does both")
    ap.add_argument("--camera-view", action="store_true",
                    help="with --dashboard, also serve the live camera and eye points at /camera. "
                         "For demos only: driver images leave the device, against plan §8, "
                         "so use it only with the consent of whoever is on camera")
    ap.add_argument("--sim", type=int, nargs="?", const=8097, default=0, metavar="PORT",
                    help="take speed, steering and lane from the ETS2 bridge (UDP, default 8097)")
    ap.add_argument("--sim-lane-cal", default="config/lane_calibration_sim.json",
                    help="lane calibration for the simulator's view")
    args = ap.parse_args()

    if args.conditions:
        print(describe_all())
        return

    condition = get_condition(args.condition)

    import cv2

    model = Path(args.model) if args.model else default_model_path()
    if not model.exists():
        raise SystemExit(
            f"Face landmark model not found at {model}.\n\n"
            f"mediapipe >= 1.0 ships no model asset. Place 'face_landmarker.task' "
            f"there, or point AWAS_FACE_MODEL at it.\n"
            f"Everything else runs without it:\n"
            f"    python -m scripts.demo_calibration\n"
            f"    python -m scripts.demo_state_machine"
        )

    road = None
    if args.sim:
        from awas.road.calibrate import load as load_lane_calibration
        from awas.road.sim import SimConfig, SimRoadSource
        lane_cal = load_lane_calibration(Path(args.sim_lane_cal))
        road = SimRoadSource(SimConfig(port=args.sim), calibration=lane_cal)
        print(f"simulator input on UDP {args.sim}"
              + ("" if lane_cal else f" (no {args.sim_lane_cal}: lane geometry is the built-in guess)"))

    eye_model = None
    if args.eye_signal == "cnn":
        try:
            from awas.driver.eyestate import EyeStateClassifier
            eye_model = EyeStateClassifier(Path(args.eye_model))
            print(f"eye signal: T1 CNN ({args.eye_model}), EAR as fallback and control")
        except (FileNotFoundError, ImportError) as exc:
            print(f"eye signal: EAR only — CNN unavailable ({exc})")
    else:
        print("eye signal: EAR only (control condition)")

    src = WebcamSource(
        camera_index=args.video if args.video else args.camera,
        driver_name=args.driver,
        profile_dir=Path(args.profiles),
        model_path=model,
        road=road,
        eye_model=eye_model,
        fusion_mode=args.fusion,
    )
    sm = StateMachine(initial_state=DrivingState.MOVING)

    # Audio is optional: a missing phrase bank should cost you sound, not the
    # session. Render it with `python -m scripts.render_phrases`.
    try:
        audio = NullAudioOut() if args.mute else StreamAudioOut(Path(args.phrases))
    except (FileNotFoundError, Exception) as exc:   # noqa: B014
        print(f"audio unavailable ({exc}); continuing muted")
        audio = NullAudioOut()

    session = None if args.no_log else SessionLog(
        Path(args.sessions), driver=args.driver,
        config={
            "source": args.video or f"camera{args.camera}",
            "muted": args.mute,
            "condition": condition.id,
            "condition_name": condition.name,
            # What produced the numbers, so sessions run differently are never
            # pooled by accident.
            "eye_signal": "cnn" if eye_model else "ear",
            "eye_model": args.eye_model if eye_model else None,
            "fusion": args.fusion,
            "road": "sim" if args.sim else "manual",
        },
    )
    executor = ActionExecutor(audio=audio, dialogue=DialogueManager(),
                              log=session, muted=args.mute, condition=condition)

    log = deque(maxlen=8)
    prev_t, fps = 0.0, 0.0

    # The centre screen is a view of decisions already made; it can be absent.
    display = trip = None
    held = {"reason": None, "t": -1e9}
    kss_given = None
    sim_noted = streamed_at = -1e9
    if args.dashboard:
        from awas.display.server import DisplayServer
        display = DisplayServer(args.dashboard, camera_view=args.camera_view).start()
        trip = TripTracker()
        print(f"dashboard on http://0.0.0.0:{display.port}"
              + ("  (camera view at /camera — driver images leave the device)" if args.camera_view else ""))

    # systemd stops a service with SIGTERM. Without this the process dies
    # mid-write and the session log has no session_end record, which the
    # analysis treats as a truncated file.
    stopping = {"now": False}

    def _stop(signum, _frame):
        stopping["now"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if session:
        print(f"logging to {session.path}")
    print(f"condition {condition.id} — {condition.name}: {condition.description}")
    print("running — q quit, [ / ] speed, i ignition, 1-9 log a KSS rating")
    try:
        while not stopping["now"]:
            telemetry = src.read()
            actions = sm.update(telemetry)
            state = src.last_state

            performed = executor.execute(
                actions, t=telemetry.timestamp, telemetry=telemetry,
                driving_state=sm.state.value,
            )
            for ex_ in performed:
                detail = ex_.text or ex_.action.message
                if ex_.suppressed:
                    detail = f"[{ex_.suppressed}] {detail}"
                elif ex_.playback:
                    detail = f"({ex_.playback.latency_ms:.0f}ms) {detail}"
                line = f"t={telemetry.timestamp:6.1f} {ex_.action.type.value:5} {detail}"
                log.append(line)
                print(" ", line)
                if display:
                    if is_display_worthy(ex_.suppressed):
                        held = {"reason": ex_.suppressed, "t": telemetry.timestamp}
                    elif ex_.action.type in (ActionType.ALERT, ActionType.SPEAK):
                        trip.note_alert()

            if session:
                session.sample(telemetry.timestamp, state, telemetry, fused=src.last_fused)

            if display:
                kss_given = _apply_commands(display, src, session, telemetry.timestamp, log, kss_given)
                level = sm.effective_level(telemetry)
                trip.update(telemetry.timestamp, level, telemetry.speed_kmh,
                            moving=sm.state in (DrivingState.MOVING, DrivingState.TRAFFIC_JAM))
                display.publish(snapshot(
                    driving_state=sm.state, level=level, driver=state,
                    speed_kmh=telemetry.speed_kmh, workload=telemetry.workload,
                    lane_available=telemetry.sdlp is not None,
                    held_reason=held["reason"] if telemetry.timestamp - held["t"] < 15 else None,
                    trip=trip, kss=kss_given,
                    sdlp_m=telemetry.sdlp,
                    sdlp_baseline_m=src.sdlp_baseline.baseline_m if src.sdlp_baseline else None,
                    fused=src.last_fused,
                ))

            if road is not None and telemetry.timestamp - sim_noted >= 10.0:
                sim_noted = telemetry.timestamp
                print("  sim  ", road.status())

            if state.t > prev_t:
                fps = 0.8 * fps + 0.2 / (state.t - prev_t)
            prev_t = state.t

            want_window = not args.headless
            want_stream = (display is not None and args.camera_view
                           and telemetry.timestamp - streamed_at >= 1.0 / CAMERA_VIEW_FPS)
            if (want_window or want_stream) and src.last_frame is not None:
                frame = overlay_frame(cv2, src.last_frame, state, sm, telemetry, log, fps,
                                      mirror=not args.video, fused=src.last_fused)
                if want_window:
                    cv2.imshow("AWAS - Phase 1", frame)
                if want_stream:
                    streamed_at = telemetry.timestamp
                    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        display.publish_frame(jpg.tobytes())

            if args.headless:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("["):
                src.speed_kmh = max(0.0, src.speed_kmh - 10)
            elif key == ord("]"):
                src.speed_kmh += 10
            elif key == ord("i"):
                src.ignition_on = not src.ignition_on
            elif ord("1") <= key <= ord("9") and session:
                # Karolinska Sleepiness Scale — the only ground truth we get,
                # and free to collect because the system can simply ask.
                session.annotation(telemetry.timestamp, kind="kss",
                                   value=key - ord("0"))
                log.append(f"t={telemetry.timestamp:6.1f} kss   {key - ord('0')}")
    finally:
        if display:
            display.close()
        executor.close()
        src.close()
        if not args.headless:
            cv2.destroyAllWindows()
        if session:
            path = session.close(t=prev_t)
            print(f"session log: {path}  ({session.counts})")
        print("profile saved" if src.pipeline.profile_dir else "no profile dir")


if __name__ == "__main__":
    main()
