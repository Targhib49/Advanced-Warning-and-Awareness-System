"""Sliding feature window and level estimation (plan §4.1).

Turns a stream of per-frame openness readings into the windowed features the
fusion core consumes: PERCLOS, closure episodes, blink statistics.

Three things here are easy to get wrong and are handled deliberately:

1. **The window is measured in TIME, not frames.** PERCLOS is defined as a
   fraction of a time interval. A fixed frame count silently becomes a
   different interval whenever the camera drops frames, which it will.

2. **Closure is time-weighted.** PERCLOS divides closed time by observed time,
   so a stutter in frame rate does not distort the ratio.

3. **A lost face is not a closed eye.** Frames without a face are excluded from
   both numerator and denominator. Counting them as closure would turn every
   occlusion into a microsleep — the exact false alarm that trains drivers to
   ignore the system.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WindowConfig:
    span_sec: float = 60.0
    perclos_p: float = 0.80       # P80: "more than 80% closed"
    microsleep_ms: float = 500.0
    min_blink_ms: float = 50.0    # shorter than this is landmark noise
    critical_closure_ms: float = 2000.0
    capacity: int = 4000          # ring size; ~60 s at 60 fps with headroom


@dataclass(frozen=True)
class Sample:
    t: float
    openness: float
    face_present: bool


@dataclass(frozen=True)
class Episode:
    """One run of consecutive closed frames."""

    start: float
    end: float

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0


@dataclass(frozen=True)
class WindowFeatures:
    perclos: float = 0.0
    microsleep_count: int = 0
    blink_count: int = 0
    blink_rate_per_min: float = 0.0
    mean_blink_duration_ms: float = 0.0
    longest_closure_ms: float = 0.0
    face_present_ratio: float = 0.0
    observed_sec: float = 0.0
    n_samples: int = 0

    @property
    def usable(self) -> bool:
        """Whether there is enough observation to say anything at all."""
        return self.observed_sec >= 5.0 and self.face_present_ratio >= 0.5


class FeatureWindow:
    """Time-bounded ring of openness samples."""

    def __init__(self, cfg: Optional[WindowConfig] = None):
        self.cfg = cfg or WindowConfig()
        self._samples: Deque[Sample] = deque(maxlen=self.cfg.capacity)

    def add(self, t: float, openness: float, face_present: bool = True) -> None:
        self._samples.append(Sample(t, openness, face_present))
        self._evict(t)

    def _evict(self, now: float) -> None:
        cutoff = now - self.cfg.span_sec
        while self._samples and self._samples[0].t < cutoff:
            self._samples.popleft()

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)

    # ---------------- feature extraction ----------------

    def _is_closed(self, s: Sample) -> bool:
        return s.face_present and s.openness < (1.0 - self.cfg.perclos_p)

    def episodes(self) -> List[Episode]:
        """Closure episodes in the window.

        An episode ends at the timestamp of the first sample that is not closed,
        so its duration is the time the eye was actually observed shut rather
        than the span between closed samples — a single closed frame then has
        the duration of one frame interval, not zero.
        """
        out: List[Episode] = []
        start: Optional[float] = None
        samples = list(self._samples)

        for i, s in enumerate(samples):
            if self._is_closed(s):
                if start is None:
                    start = s.t
            elif start is not None:
                out.append(Episode(start, s.t))
                start = None

        if start is not None:
            # Still closed at the window edge. Close it at the last timestamp;
            # it will grow on the next update.
            last = samples[-1].t
            if last > start:
                out.append(Episode(start, last))
        return out

    def features(self) -> WindowFeatures:
        samples = list(self._samples)
        if len(samples) < 2:
            return WindowFeatures(n_samples=len(samples))

        observed = 0.0        # seconds with a face
        closed_time = 0.0     # seconds with a face AND eyes shut
        total = 0.0

        for a, b in zip(samples, samples[1:]):
            dt = b.t - a.t
            if dt <= 0:
                continue
            total += dt
            if a.face_present:
                observed += dt
                if self._is_closed(a):
                    closed_time += dt

        perclos = closed_time / observed if observed > 0 else 0.0
        face_ratio = observed / total if total > 0 else 0.0

        eps = self.episodes()
        micro = [e for e in eps if e.duration_ms >= self.cfg.microsleep_ms]
        blinks = [
            e for e in eps
            if self.cfg.min_blink_ms <= e.duration_ms < self.cfg.microsleep_ms
        ]
        longest = max((e.duration_ms for e in eps), default=0.0)
        mean_blink = sum(e.duration_ms for e in blinks) / len(blinks) if blinks else 0.0
        rate = len(blinks) / (observed / 60.0) if observed > 0 else 0.0

        return WindowFeatures(
            perclos=perclos,
            microsleep_count=len(micro),
            blink_count=len(blinks),
            blink_rate_per_min=rate,
            mean_blink_duration_ms=mean_blink,
            longest_closure_ms=longest,
            face_present_ratio=face_ratio,
            observed_sec=observed,
            n_samples=len(samples),
        )


# ---------------- level estimation ----------------

@dataclass(frozen=True)
class LevelConfig:
    # PERCLOS floors per level (plan §4.1 — fitted, not asserted)
    perclos_l1: float = 0.08
    perclos_l2: float = 0.15
    perclos_l3: float = 0.25

    # Blink lengthening precedes PERCLOS rising.
    long_blink_ms: float = 400.0

    escalate_ticks: int = 3
    deescalate_ticks: int = 10


@dataclass
class Proposal:
    level: int
    reason: str
    immediate: bool = False   # bypass hysteresis: unambiguous danger


def propose_level(f: WindowFeatures, cfg: LevelConfig) -> Proposal:
    """Map window features to a level, before hysteresis.

    Microsleeps outrank PERCLOS. A single two-second closure at speed is not a
    statistical trend to be smoothed — it is the event itself, and it bypasses
    the hysteresis that everything else goes through.
    """
    if not f.usable:
        return Proposal(0, "insufficient observation")

    if f.longest_closure_ms >= 2000.0:
        return Proposal(4, f"closure {f.longest_closure_ms:.0f} ms", immediate=True)
    if f.microsleep_count >= 2:
        return Proposal(4, f"{f.microsleep_count} microsleeps", immediate=True)
    if f.microsleep_count >= 1:
        return Proposal(3, "microsleep")

    if f.perclos >= cfg.perclos_l3:
        return Proposal(3, f"perclos {f.perclos:.1%}")
    if f.perclos >= cfg.perclos_l2:
        return Proposal(2, f"perclos {f.perclos:.1%}")
    if f.mean_blink_duration_ms >= cfg.long_blink_ms:
        return Proposal(2, f"blinks {f.mean_blink_duration_ms:.0f} ms")
    if f.perclos >= cfg.perclos_l1:
        return Proposal(1, f"perclos {f.perclos:.1%}")

    return Proposal(0, "alert")


class Hysteresis:
    """Damps level changes so the driver is not alerted and un-alerted.

    Deliberately asymmetric: escalation is fast and jumps straight to the
    proposed level, de-escalation is slow and steps down one level at a time.
    Being late to warn is a safety failure; being late to relax is only an
    annoyance.
    """

    def __init__(self, cfg: Optional[LevelConfig] = None):
        self.cfg = cfg or LevelConfig()
        self.level = 0
        self._up = 0
        self._down = 0

    def update(self, proposal: Proposal) -> int:
        if proposal.immediate and proposal.level > self.level:
            self.level = proposal.level
            self._up = self._down = 0
            return self.level

        if proposal.level > self.level:
            self._up += 1
            self._down = 0
            if self._up >= self.cfg.escalate_ticks:
                self.level = proposal.level
                self._up = 0
        elif proposal.level < self.level:
            self._down += 1
            self._up = 0
            if self._down >= self.cfg.deescalate_ticks:
                self.level -= 1        # one step at a time
                self._down = 0
        else:
            self._up = self._down = 0

        return self.level

    def reset(self) -> None:
        self.level = 0
        self._up = self._down = 0
