"""Multimodal fusion — the research contribution (plan v3 §4.3, tier T4).

Facial drowsiness detection has two well-known failures: it stops working when
the face is occluded, and it only sees drowsiness once the face shows it, which
is late. Lane keeping degrades earlier and survives sunglasses.

Fusing them is hypothesis **H2**: earlier onset detection at a matched
false-alarm rate, versus facial signals alone. This module is what makes that
testable — `E` (facial only) versus the fused estimate, on identical input.

Transparent weighted evidence, not a learned model, and deliberately so:
explainable to a judge, debuggable, needs no paired training data we do not
have, and gives an honest baseline for a learned fusion later. Every threshold
below is unfitted (plan §4.1) and marked as such.

The important behaviour is what happens when the channels DISAGREE. A driver
who looks drowsy but is holding the lane precisely is probably not drowsy —
so agreement raises confidence and disagreement lowers it, rather than the
estimate simply being averaged and asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --- Vehicle-channel thresholds ------------------------------------------
#
# Two scales, and the personal one is primary.
#
# ABSOLUTE (fallback, used before a driver has a baseline). Alert motorway
# driving sits around 0.15-0.20 m; the driving-fatigue literature reports
# increases of roughly 0.05-0.15 m under impairment. The earlier values here
# (L3 = 0.35 m) were set high enough that the channel saturated near level 2.7
# and could never raise a level-3 alarm — a ceiling the parameter sweep
# exposed. These are lowered to sit inside the range real wander occupies.
SDLP_L1_M = 0.16
SDLP_L2_M = 0.22
SDLP_L3_M = 0.28

# PERSONAL (primary, once an alert-driving baseline exists). Expressed as an
# EXCESS over that driver's own baseline rather than a ratio, because the
# literature reports fatigue effects as absolute increments in metres, and
# because a ratio punishes a driver whose baseline is already tidy. The
# +0.05 m step is roughly the increase associated with legally significant
# alcohol impairment, which makes it a defensible place to put level 2.
SDLP_EXCESS_L1_M = 0.03
SDLP_EXCESS_L2_M = 0.06
SDLP_EXCESS_L3_M = 0.11

# Confidence a single channel can reach on its own. Capped below 1.0 because
# one unopposed opinion is not corroboration.
SINGLE_CHANNEL_CAP = 0.75


@dataclass
class ChannelEstimate:
    """One channel's opinion, with how much it trusts itself."""

    source: str
    level: float = 0.0          # continuous 0-4; fusion needs the granularity
    confidence: float = 0.0     # 0-1
    available: bool = False
    detail: str = ""

    @classmethod
    def unavailable(cls, source: str, why: str = "") -> "ChannelEstimate":
        return cls(source=source, available=False, detail=why)


@dataclass
class FusedEstimate:
    level: int = 0
    level_continuous: float = 0.0
    confidence: float = 0.0
    agreement: Optional[float] = None       # None when only one channel spoke
    channels: List[ChannelEstimate] = field(default_factory=list)

    @property
    def sources(self) -> List[str]:
        return [c.source for c in self.channels if c.available]

    def explain(self) -> str:
        if not self.channels:
            return "no channels"
        bits = [f"{c.source} L{c.level:.1f}@{c.confidence:.2f}"
                for c in self.channels if c.available]
        if not bits:
            return "all channels unavailable"
        line = " + ".join(bits) + f" -> L{self.level} @{self.confidence:.2f}"
        if self.agreement is not None:
            line += f" (agreement {self.agreement:.2f})"
        return line


# ---------------- channel builders ----------------

def facial_channel(
    level: int,
    confidence: float,
    face_present: bool,
    window_usable: bool = True,
) -> ChannelEstimate:
    """Wrap the driver pipeline's output as a fusion channel."""
    if not face_present:
        return ChannelEstimate.unavailable("facial", "no face")
    if not window_usable:
        return ChannelEstimate.unavailable("facial", "window too short")
    return ChannelEstimate(source="facial", level=float(level),
                           confidence=confidence, available=True)


def vehicle_channel(
    sdlp: Optional[float],
    departures: int = 0,
    samples: int = 0,
    min_samples: int = 30,
    sdlp_baseline_m: Optional[float] = None,
) -> ChannelEstimate:
    """Lane-keeping degradation as drowsiness evidence.

    `sdlp_baseline_m` is the hook for personalisation: given a driver's own
    alert-driving SDLP, deviation from it is a far better signal than an
    absolute threshold — exactly as with EAR, and for the same reason. Until
    that is collected, absolute thresholds stand in.
    """
    if sdlp is None or samples < min_samples:
        return ChannelEstimate.unavailable("vehicle", "no lane data")

    if sdlp_baseline_m and sdlp_baseline_m > 0:
        excess = sdlp - sdlp_baseline_m
        if excess < SDLP_EXCESS_L1_M:
            level = 0.0
        elif excess < SDLP_EXCESS_L2_M:
            level = 1.0 + (excess - SDLP_EXCESS_L1_M) / (SDLP_EXCESS_L2_M - SDLP_EXCESS_L1_M)
        elif excess < SDLP_EXCESS_L3_M:
            level = 2.0 + (excess - SDLP_EXCESS_L2_M) / (SDLP_EXCESS_L3_M - SDLP_EXCESS_L2_M)
        else:
            level = min(3.0 + (excess - SDLP_EXCESS_L3_M) / 0.06, 4.0)
        detail = f"sdlp {sdlp:.3f} m = baseline +{excess * 100:.0f} cm"
    else:
        if sdlp < SDLP_L1_M:
            level = 0.0
        elif sdlp < SDLP_L2_M:
            level = 1.0 + (sdlp - SDLP_L1_M) / (SDLP_L2_M - SDLP_L1_M)
        elif sdlp < SDLP_L3_M:
            level = 2.0 + (sdlp - SDLP_L2_M) / (SDLP_L3_M - SDLP_L2_M)
        else:
            level = min(3.0 + (sdlp - SDLP_L3_M) / 0.15, 4.0)
        detail = f"sdlp {sdlp:.3f} m"

    # A lane departure is not a statistic, it is an event.
    if departures:
        level = max(level, 3.0)
        detail += f", {departures} departure(s)"

    # Confidence grows with how much of the window is actually populated:
    # SDLP over twelve seconds is a rumour, over a minute it is a measurement.
    confidence = min(samples / (min_samples * 6.0), 1.0) * 0.9
    return ChannelEstimate(source="vehicle", level=level,
                           confidence=confidence, available=True, detail=detail)


# ---------------- fusion ----------------

def fuse(channels: List[ChannelEstimate],
         disagreement_penalty: float = 0.35,
         mode: str = "weighted") -> FusedEstimate:
    """Confidence-weighted late fusion with an agreement term.

    Three properties that matter more than the arithmetic:

      * **A silent channel costs nothing.** With the face occluded, the vehicle
        channel simply carries the estimate — which is the graceful-degradation
        claim in §5.4, made real rather than asserted.
      * **One channel cannot be certain.** A lone opinion is capped, so the
        state machine's confidence gating stays cautious when nothing
        corroborates.
      * **Disagreement is informative, not noise.** Channels that contradict
        each other lower confidence rather than averaging into a confident
        wrong answer.

    Two modes, and the difference matters more than it looks:

      ``weighted``   confidence-weighted mean. Damps disagreement, but it
                     CANNOT detect earlier than the mean of its channels — an
                     early warning from one channel is diluted by the other's
                     silence. Fewer false alarms, later detection.

      ``asymmetric`` the level is the highest any channel claims, while
                     confidence still falls when they disagree. An early
                     warning survives; acting on it is then governed by the
                     state machine's confidence gate rather than by having been
                     averaged away.

    The parameter sweep in `scripts/parameter_sweep.py` is what exposed this:
    under ``weighted``, adding a vehicle channel that leads the facial one made
    detection LATER, which is the opposite of what fusing it is for.
    """
    live = [c for c in channels if c.available and c.confidence > 0]
    if not live:
        return FusedEstimate(level=0, confidence=0.0, channels=channels)

    total_w = sum(c.confidence for c in live)
    weighted_level = sum(c.level * c.confidence for c in live) / total_w
    level = max(c.level for c in live) if mode == "asymmetric" else weighted_level
    mean_conf = total_w / len(live)

    if len(live) == 1:
        return FusedEstimate(
            level=int(round(level)), level_continuous=level,
            confidence=min(mean_conf, SINGLE_CHANNEL_CAP),
            agreement=None, channels=channels,
        )

    # Spread across the full 0-4 scale, so 1.0 means total disagreement.
    spread = (max(c.level for c in live) - min(c.level for c in live)) / 4.0
    agreement = max(0.0, 1.0 - spread)
    confidence = mean_conf * (1.0 - disagreement_penalty * spread)

    # Corroboration is worth something: two channels that agree deserve more
    # confidence than either alone, which is the whole point of fusing.
    if agreement > 0.85:
        confidence = min(confidence * 1.15, 1.0)

    return FusedEstimate(
        level=int(round(level)), level_continuous=level,
        confidence=max(0.0, min(confidence, 1.0)),
        agreement=agreement, channels=channels,
    )
