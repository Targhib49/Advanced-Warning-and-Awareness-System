"""Run the state machine against a scripted drive and print what happens.

    python -m scripts.demo_state_machine
"""

import argparse
import time
from pathlib import Path

from awas.companion.dialogue import DialogueManager
from awas.core.machine import StateMachine
from awas.core.states import DrivingState
from awas.core.telemetry import Distraction
from awas.runtime.executor import ActionExecutor
from awas.runtime.session_log import SessionLog
from awas.sources.mock import MockSource, Scene
from awas.speech.audio import NullAudioOut, StreamAudioOut


def build_scenario() -> list[Scene]:
    """A drive that exercises every gate in the plan: escalation, workload
    suppression, deferral, confidence downgrade, and occlusion capping."""
    return [
        Scene(3, "parked, engine off",
              ignition_on=False, parking_brake=True),

        Scene(5, "engine on, brake released, pulling away",
              speed_kmh=20, parking_brake=False),

        Scene(10, "highway cruise, alert",
              speed_kmh=80, workload=0.3),

        Scene(20, "mild drowsiness, quiet road - check-in expected",
              speed_kmh=80, drowsiness_level=1, workload=0.2),

        Scene(20, "moderate drowsiness in heavy traffic - suppressed",
              speed_kmh=50, drowsiness_level=2, workload=0.9),

        Scene(20, "severe drowsiness, still heavy traffic - deferred, fires anyway",
              speed_kmh=50, drowsiness_level=3, workload=0.95),

        Scene(15, "severe reading but sunglasses glare - low confidence",
              speed_kmh=60, drowsiness_level=3, confidence=0.4, workload=0.2),

        Scene(20, "face occluded, vehicle signals only - level capped",
              speed_kmh=60, drowsiness_level=4, face_present=False, workload=0.2),

        Scene(10, "critical microsleep, face visible - reflex alert",
              speed_kmh=60, drowsiness_level=4, workload=0.5),

        Scene(8, "red light, driver still very drowsy",
              speed_kmh=2, traffic_light_color="red", drowsiness_level=3),

        Scene(4, "light turns green, driver unresponsive",
              speed_kmh=2, traffic_light_color="green", drowsiness_level=3),

        Scene(10, "driving again, reaching for the phone",
              speed_kmh=60, distraction=Distraction.PHONE, workload=0.4),

        Scene(40, "traffic stops - jam forming",
              speed_kmh=0, vehicle_ahead_distance_m=2.0),

        Scene(10, "gap ahead opens but we have not moved",
              speed_kmh=0, vehicle_ahead_distance_m=15.0),

        Scene(10, "traffic moving again",
              speed_kmh=15),

        Scene(5, "parking",
              speed_kmh=0, parking_brake=True),
    ]


def run(speak: bool = False, log_dir: str = "") -> None:
    source = MockSource(scenes=build_scenario(), tick_sec=1.0)
    sm = StateMachine(initial_state=DrivingState.PARKED)

    audio = StreamAudioOut(Path("assets/phrases")) if speak else NullAudioOut()
    session = SessionLog(Path(log_dir), driver="demo") if log_dir else None
    executor = ActionExecutor(audio=audio, dialogue=DialogueManager(seed=7),
                              log=session)

    last_label = ""
    while not source.is_done():
        t = source.read()
        actions = sm.update(t)

        label = source.current_label()
        if label != last_label:
            print(f"\n--- t={t.timestamp:.0f}s | {label} ---")
            last_label = label

        performed = executor.execute(actions, t=t.timestamp, telemetry=t,
                                     driving_state=sm.state.value)
        for ex in performed:
            detail = ex.text or ex.action.message
            if ex.suppressed:
                detail = f"[{ex.suppressed}] {detail}"
            print(f"  t={t.timestamp:5.1f}s [{sm.state.value:17}] "
                  f"{ex.action.type.value:5} urg={ex.action.urgency}  {detail}")
        if session:
            session.sample(t.timestamp, t, t)

        # Virtual time runs far faster than speech does. Without this the whole
        # drive fires in milliseconds and every line cuts off the one before —
        # correct behaviour for the executor, useless as a demo. Waiting
        # compresses the drive to just its audible parts.
        if speak:
            deadline = time.monotonic() + 6.0
            while audio.is_busy() and time.monotonic() < deadline:
                time.sleep(0.02)
            if any(ex.performed for ex in performed):
                time.sleep(0.35)


    executor.close()
    if session:
        print(f"\nsession log: {session.close(t=source.now)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scripted drive through every gate.")
    ap.add_argument("--speak", action="store_true", help="play the audio out loud")
    ap.add_argument("--log", default="", help="write a session log to this directory")
    a = ap.parse_args()
    run(speak=a.speak, log_dir=a.log)
