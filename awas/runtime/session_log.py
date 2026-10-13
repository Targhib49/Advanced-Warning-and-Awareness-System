"""Structured session logging — the record the study in §6 will analyse.

This is the piece that cannot be added later. Time-to-relapse, intervention
efficacy and the Phase 4 hazard model are all computed from data that either
was or was not written down at the time. Recording it costs a few bytes per
second now; not recording it costs the experiment.

Format is JSON Lines: one self-describing record per line, appended, readable
while the session is still running and trivially loadable into pandas after.

PRIVACY (invariant I5, plan §8): derived numbers only. No frames, no crops, no
audio, no landmark coordinates — nothing here can reconstruct a picture of the
driver. What it does contain is what the driver's alertness was and what the
system did about it.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1


def _round(v: Any, places: int = 4) -> Any:
    return round(v, places) if isinstance(v, float) else v


class SessionLog:
    """Append-only JSONL writer for one driving session."""

    def __init__(
        self,
        directory: Path,
        driver: str = "default",
        sample_hz: float = 1.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.driver = driver
        self.sample_period = 1.0 / sample_hz if sample_hz > 0 else 0.0

        self.session_id = uuid.uuid4().hex[:12]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.dir / f"{stamp}_{driver}_{self.session_id}.jsonl"
        self._fh = self.path.open("w", encoding="utf-8")

        self._last_sample_t: Optional[float] = None
        self._t0 = time.time()
        self.counts: Dict[str, int] = {"sample": 0, "action": 0, "annotation": 0}

        self._write({
            "type": "session_start",
            "t": 0.0,
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "driver": driver,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": config or {},
        })

    # ---------------- writing ----------------

    def _write(self, record: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._fh.flush()   # a session that ends in a crash still has its data

    def sample(self, t: float, state: Any, telemetry: Any = None, fused: Any = None) -> bool:
        """Record a periodic snapshot. Rate-limited to `sample_hz`.

        Returns whether it was written, so callers can avoid building the
        record when it would be dropped.
        """
        if self.sample_period > 0 and self._last_sample_t is not None:
            if t - self._last_sample_t < self.sample_period:
                return False
        self._last_sample_t = t

        rec = {
            "type": "sample",
            "t": round(t, 3),
            "level": getattr(state, "drowsiness_level", None),
            "confidence": _round(getattr(state, "confidence", None)),
            "perclos": _round(getattr(state, "perclos", None)),
            "microsleeps": getattr(state, "microsleep_count", None),
            "blink_ms": _round(getattr(state, "blink_duration_mean_ms", None), 1),
            "blink_rate": _round(getattr(state, "blink_rate_per_min", None), 2),
            "longest_closure_ms": _round(getattr(state, "longest_closure_ms", None), 1),
            "face": getattr(state, "face_present", None),
            "ear": _round(getattr(state, "eye_aspect_ratio", None)),
            "openness": _round(getattr(state, "openness", None), 3),
            "pitch": _round(getattr(state, "head_pitch_deg", None), 1),
            "yaw": _round(getattr(state, "head_yaw_deg", None), 1),
            "personalized": getattr(state, "personalized", None),
            # Both eye signals on the same frame: the CNN-versus-EAR comparison
            # the architecture asks for comes straight out of these.
            "eye_source": getattr(state, "eye_source", None),
            "eyes_observable": getattr(state, "eyes_observable", None),
            "p_closed": _round(getattr(state, "p_closed", None), 3),
            "ear_openness": _round(getattr(state, "ear_openness", None), 3),
        }
        if telemetry is not None:
            rec["speed_kmh"] = _round(getattr(telemetry, "speed_kmh", None), 1)
            rec["speed_source"] = getattr(telemetry, "speed_source", None)
            rec["workload"] = _round(getattr(telemetry, "workload", None), 3)
            rec["sdlp"] = _round(getattr(telemetry, "sdlp", None), 4)
            rec["steering_reversals"] = _round(getattr(telemetry, "steering_reversal_rate", None), 2)
            rec["lane_departure"] = getattr(telemetry, "lane_departure", None)
            rec["reversing"] = getattr(telemetry, "reversing", None)
            rec["turn_signal"] = getattr(telemetry, "turn_signal", None)
        if fused is not None:
            # `level` above is the facial channel; this is what was acted on.
            rec["fused_level"] = getattr(fused, "level", None)
            rec["fused_confidence"] = _round(getattr(fused, "confidence", None))
            rec["agreement"] = _round(getattr(fused, "agreement", None))
            rec["channels"] = {
                c.source: [_round(c.level, 2), _round(c.confidence, 2)]
                for c in getattr(fused, "channels", []) if c.available
            }
        self._write(rec)
        self.counts["sample"] += 1
        return True

    def action(
        self,
        t: float,
        action: Any,
        clips: list,
        playback: Any = None,
        driving_state: str = "",
        trigger: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one action and what actually came out of the speaker.

        `trigger` carries the telemetry that caused it. Without that, an
        intervention in the log is an event with no cause and the efficacy
        analysis has nothing to correlate against.
        """
        rec = {
            "type": "action",
            "t": round(t, 3),
            "kind": getattr(action.type, "value", str(action.type)),
            "urgency": action.urgency,
            "alert_id": action.alert_id or None,
            "intent": getattr(action.intent, "value", None),
            "expects_reply": action.expects_reply,
            "driving_state": driving_state,
            "clips": clips,
            "trigger": trigger or {},
        }
        if playback is not None:
            rec["latency_ms"] = round(playback.latency_ms, 2)
            rec["duration_ms"] = round(playback.duration_ms, 1)
            rec["played"] = playback.played
        self._write(rec)
        self.counts["action"] += 1

    def annotation(self, t: float, kind: str, value: Any = None, text: str = "") -> None:
        """Driver self-report (KSS) or an experimenter's mark."""
        self._write({
            "type": "annotation", "t": round(t, 3),
            "kind": kind, "value": value, "text": text,
        })
        self.counts["annotation"] += 1

    def close(self, t: float = 0.0) -> Path:
        if self._fh is None:
            return self.path
        self._write({
            "type": "session_end",
            "t": round(t, 3),
            "wall_sec": round(time.time() - self._t0, 1),
            "counts": dict(self.counts),
        })
        self._fh.close()
        self._fh = None
        return self.path

    def __enter__(self) -> "SessionLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
