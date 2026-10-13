"""Guided calibration: a new driver's personal eye scale in about a minute.

Passive calibration (calibration.py) learns while the driver drives and takes
minutes; its "closed" reference is the bottom of their blinks. Guided
calibration asks for it directly — eyes open, then eyes closed — so the
closed reference is measured rather than inferred, and it doubles as the
check that the T1 CNN can actually tell THIS driver's eyes apart.

Steps:
    position  face found, close enough, facing forward, enough light; held
    open      look ahead normally (blinks included; medians ignore them)
    closed    close the eyes until a round is seen, twice
    done      a result, or a reason it failed

The closed step is self-paced, because a driver with their eyes shut cannot
read an instruction or a countdown, and the device may have no speaker. A
round counts once the eyes are seen to close for long enough and reopen.

Pure: frames in, state out. No camera, no clock.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class GuidedConfig:
    # position
    min_eye_px: float = 18.0          # corner-to-corner; below this the CNN crop is mush
    # Generous on purpose: the device sits left of the wheel (plan §7.5), so a
    # driver looking at the road is already turned 20-30 degrees from the
    # camera. "Facing" here means the camera has the face, not that the driver
    # stares into it. The pose they hold while looking ahead is recorded as
    # their neutral (GuidedResult.neutral_*), and driving is judged against it.
    max_yaw_deg: float = 35.0
    max_pitch_deg: float = 30.0
    min_brightness: float = 45.0      # mean gray around the eyes, 0-255
    max_brightness: float = 225.0
    hold_sec: float = 2.0             # all checks passing this long to move on
    # open
    open_sec: float = 8.0
    # closed
    rounds: int = 2
    closed_ratio: float = 0.80        # EAR below this x open median reads as closed
    reopen_ratio: float = 0.90
    cnn_closed_p: float = 0.70        # or the CNN this sure
    min_closed_sec: float = 1.2       # shorter is a blink, not a deliberate close
    reopen_sec: float = 0.3
    trim_sec: float = 0.2             # drop the lid-moving frames at each end
    closed_timeout_sec: float = 30.0
    # verdict
    min_ear_separation: float = 0.15  # (open - closed) / open
    cnn_min_accuracy: float = 0.90
    cnn_min_gap: float = 0.50         # median P(closed): closed minus open


@dataclass
class Frame:
    t: float
    face: bool
    ear: float = 0.0
    p_closed: Optional[float] = None
    eye_px: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    brightness: float = 128.0


@dataclass
class GuidedResult:
    ok: bool
    eye_signal: Optional[str] = None      # "cnn" | "ear" when ok
    reason: str = ""
    open_ear: Optional[float] = None
    closed_ear: Optional[float] = None
    ear_separation: Optional[float] = None
    cnn_open_p: Optional[float] = None
    cnn_closed_p: Optional[float] = None
    cnn_accuracy: Optional[float] = None
    open_frames: int = 0
    closed_frames: int = 0
    # Head pose while looking at the road: the driver's "eyes on the road"
    # direction, relative to wherever the camera happens to be mounted.
    neutral_yaw: Optional[float] = None
    neutral_pitch: Optional[float] = None

    def summary(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class GuidedCalibration:
    cfg: GuidedConfig = field(default_factory=GuidedConfig)
    step: str = "position"
    checks: dict = field(default_factory=lambda: dict(face=False, distance=False, facing=False, light=False))
    result: Optional[GuidedResult] = None

    _ok_since: Optional[float] = None
    _open: List[Frame] = field(default_factory=list)
    _open_started: Optional[float] = None
    _closed: List[Frame] = field(default_factory=list)
    _rounds_done: int = 0
    _run: List[Frame] = field(default_factory=list)
    _reopen_since: Optional[float] = None
    _closed_started: Optional[float] = None
    _open_ref: Optional[float] = None
    paused: bool = False

    # ---------------- feeding ----------------

    def feed(self, f: Frame) -> None:
        if self.step == "done":
            return
        self.checks = self._checks(f)
        if self.step == "position":
            self._position(f)
        elif self.step == "open":
            self._open_step(f)
        elif self.step == "closed":
            self._closed_step(f)

    def _checks(self, f: Frame) -> dict:
        c = self.cfg
        return dict(
            face=f.face,
            distance=f.face and f.eye_px >= c.min_eye_px,
            facing=f.face and abs(f.yaw) <= c.max_yaw_deg and abs(f.pitch) <= c.max_pitch_deg,
            light=f.face and c.min_brightness <= f.brightness <= c.max_brightness,
        )

    def _position(self, f: Frame) -> None:
        if all(self.checks.values()):
            self._ok_since = f.t if self._ok_since is None else self._ok_since
            if f.t - self._ok_since >= self.cfg.hold_sec:
                self.step, self._open_started = "open", f.t
        else:
            self._ok_since = None

    def _usable(self) -> bool:
        return self.checks["face"] and self.checks["facing"]

    def _open_step(self, f: Frame) -> None:
        # Looking away or leaving pauses the step; it does not throw away what
        # was already collected.
        self.paused = not self._usable()
        if self.paused:
            return
        self._open.append(f)
        if f.t - self._open[0].t >= self.cfg.open_sec:
            self._open_ref = statistics.median(x.ear for x in self._open)
            self.step, self._closed_started = "closed", f.t

    def _is_closed(self, f: Frame) -> bool:
        if f.p_closed is not None and f.p_closed >= self.cfg.cnn_closed_p:
            return True
        return f.ear > 0 and f.ear < self.cfg.closed_ratio * self._open_ref

    def _is_open(self, f: Frame) -> bool:
        return f.ear >= self.cfg.reopen_ratio * self._open_ref and not (
            f.p_closed is not None and f.p_closed >= self.cfg.cnn_closed_p)

    def _closed_step(self, f: Frame) -> None:
        c = self.cfg
        if f.t - self._closed_started > c.closed_timeout_sec:
            self._finish(GuidedResult(ok=False, reason="closed eyes were not detected"))
            return
        self.paused = not f.face
        if self.paused:
            self._run, self._reopen_since = [], None
            return
        if self._is_closed(f):
            self._run.append(f)
            self._reopen_since = None
            return
        if not self._run:
            return
        if self._is_open(f):
            self._reopen_since = f.t if self._reopen_since is None else self._reopen_since
            if f.t - self._reopen_since >= c.reopen_sec:
                span = self._run[-1].t - self._run[0].t
                if span >= c.min_closed_sec:
                    lo, hi = self._run[0].t + c.trim_sec, self._run[-1].t - c.trim_sec
                    self._closed += [x for x in self._run if lo <= x.t <= hi]
                    self._rounds_done += 1
                self._run, self._reopen_since = [], None
                if self._rounds_done >= c.rounds:
                    self._finish(self._verdict())

    # ---------------- verdict ----------------

    def _verdict(self) -> GuidedResult:
        c = self.cfg
        open_ear = statistics.median(x.ear for x in self._open)
        closed_ear = statistics.median(x.ear for x in self._closed)
        sep = (open_ear - closed_ear) / open_ear if open_ear > 0 else 0.0
        r = GuidedResult(ok=False, open_ear=open_ear, closed_ear=closed_ear, ear_separation=sep,
                         open_frames=len(self._open), closed_frames=len(self._closed),
                         neutral_yaw=statistics.median(x.yaw for x in self._open),
                         neutral_pitch=statistics.median(x.pitch for x in self._open))

        po = [x.p_closed for x in self._open if x.p_closed is not None]
        pc = [x.p_closed for x in self._closed if x.p_closed is not None]
        cnn_ok = False
        if po and pc:
            r.cnn_open_p, r.cnn_closed_p = statistics.median(po), statistics.median(pc)
            # Balanced accuracy at 0.5: open frames should read open, closed closed.
            r.cnn_accuracy = 0.5 * (sum(p < 0.5 for p in po) / len(po) + sum(p >= 0.5 for p in pc) / len(pc))
            cnn_ok = (r.cnn_accuracy >= c.cnn_min_accuracy
                      and r.cnn_closed_p - r.cnn_open_p >= c.cnn_min_gap)
        ear_ok = sep >= c.min_ear_separation

        if cnn_ok:
            r.ok, r.eye_signal = True, "cnn"
        elif ear_ok:
            r.ok, r.eye_signal = True, "ear"
            r.reason = "the CNN did not separate this driver's eyes; using EAR" if po else ""
        else:
            r.reason = "open and closed eyes looked too alike"
        return r

    def _finish(self, result: GuidedResult) -> None:
        self.result, self.step, self.paused = result, "done", False

    # ---------------- reporting ----------------

    @property
    def rounds_done(self) -> int:
        return self._rounds_done

    @property
    def progress(self) -> float:
        c = self.cfg
        if self.step == "position":
            return 0.0
        if self.step == "open":
            got = (self._open[-1].t - self._open[0].t) if self._open else 0.0
            return 0.1 + 0.5 * min(got / c.open_sec, 1.0)
        if self.step == "closed":
            return 0.6 + 0.4 * self._rounds_done / c.rounds
        return 1.0

    @property
    def closing_now(self) -> bool:
        return self.step == "closed" and bool(self._run)


def frame_from(obs, bgr) -> Frame:
    """A calibration frame from a FaceObservation and the camera image.

    Eye size is the mean corner-to-corner width in pixels; brightness is the
    mean gray level in a box around both eyes, which is the light that
    matters — a bright window behind the driver does not light their eyes.
    """
    if obs is None or not obs.face_present or not obs.landmarks or bgr is None:
        return Frame(t=getattr(obs, "t", 0.0), face=False)
    import numpy as np
    from awas.driver.ear import MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE
    from awas.driver.eyestate import eye_geometry

    geo = [eye_geometry(obs.landmarks, idx) for idx in (MEDIAPIPE_LEFT_EYE, MEDIAPIPE_RIGHT_EYE)]
    width = sum(g[1] for g in geo) / 2.0
    xs = [g[0][0] for g in geo]
    ys = [g[0][1] for g in geo]
    h, w = bgr.shape[:2]
    pad = width
    x0, x1 = int(max(min(xs) - pad, 0)), int(min(max(xs) + pad, w))
    y0, y1 = int(max(min(ys) - pad / 2, 0)), int(min(max(ys) + pad / 2, h))
    region = bgr[y0:y1, x0:x1]
    brightness = float(np.mean(region)) if region.size else 0.0
    return Frame(t=obs.t, face=True, ear=obs.ear, p_closed=obs.p_closed, eye_px=width,
                 yaw=obs.yaw_deg, pitch=obs.pitch_deg, brightness=brightness)


def profile_from(result: GuidedResult, driver_id: str, cal_cfg=None):
    """The driver's EAR profile from a passed guided calibration.

    It is credited with the passive calibrator's minimum sample count. That is
    a judgement, stated: a minute of deliberately open and deliberately closed
    eyes measures both ends of the scale directly, which thirty seconds of
    incidental blinks — the passive minimum — only estimates the bottom of.
    Driving then keeps refining it exactly as before (merged_with).
    """
    from awas.driver.calibration import CalibrationConfig, DriverProfile

    cal_cfg = cal_cfg or CalibrationConfig()
    if not result.ok:
        raise ValueError("cannot build a profile from a failed calibration")
    return DriverProfile(
        name=driver_id,
        open_ref=result.open_ear,
        closed_ref=result.closed_ear,
        n_samples=max(result.open_frames + result.closed_frames, cal_cfg.min_samples),
    )
