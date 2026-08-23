"""Audio output with a measurable latency budget (plan §7.2).

The reflex path has 300 ms from decision to first audible sound. Meeting that
rules out the obvious implementations: spawning `afplay` costs a process
launch, and opening a device on demand costs the driver hardware handshake.

So the stream is opened once at startup and left running, every clip is decoded
into memory before the vehicle moves, and playing one is a pointer swap under a
lock. The callback stamps the moment it first pulls samples from a new clip,
which makes the reported latency the real thing — decision to first frame
handed to the device — rather than the time to return from a function.
"""

from __future__ import annotations

import json
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

import numpy as np


@dataclass
class Playback:
    """What happened when a clip was played."""

    clip_id: str
    latency_ms: float
    duration_ms: float
    played: bool = True


class AudioOut(Protocol):
    def play(self, clip_ids: List[str]) -> Playback: ...
    def is_busy(self) -> bool: ...
    def close(self) -> None: ...


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Decode a 16-bit mono PCM WAV to float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16-bit PCM")
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1)
    return data, rate


class NullAudioOut:
    """Records what would have been played. For tests and headless runs."""

    def __init__(self) -> None:
        self.history: List[Playback] = []

    def play(self, clip_ids: List[str]) -> Playback:
        pb = Playback(clip_id="+".join(clip_ids), latency_ms=0.0, duration_ms=0.0)
        self.history.append(pb)
        return pb

    def is_busy(self) -> bool:
        return False

    def close(self) -> None:
        pass


class StreamAudioOut:
    """Pre-warmed output stream over an in-memory clip bank."""

    def __init__(self, phrase_dir: Path, device: Optional[int] = None,
                 blocksize: int = 256):
        import sounddevice as sd

        self.dir = Path(phrase_dir)
        manifest = self.dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"No phrase bank at {self.dir}. Render it first:\n"
                f"    python -m scripts.render_phrases"
            )

        self.manifest: Dict[str, dict] = json.loads(manifest.read_text())
        self.clips: Dict[str, np.ndarray] = {}
        rate = None
        for clip_id, meta in self.manifest.items():
            data, r = load_wav(self.dir / meta["file"])
            if rate is None:
                rate = r
            elif r != rate:
                raise ValueError(
                    f"{meta['file']} is {r} Hz but the bank is {rate} Hz. "
                    f"Re-render the whole bank so one stream can serve it."
                )
            self.clips[clip_id] = data
        self.rate = rate or 22050

        self._lock = threading.Lock()
        self._queue: List[np.ndarray] = []
        self._cursor = 0
        self._requested_at: Optional[float] = None
        self._started_at: Optional[float] = None

        self._stream = sd.OutputStream(
            samplerate=self.rate, channels=1, dtype="float32",
            blocksize=blocksize, device=device, callback=self._callback,
        )
        self._stream.start()

    # ---------------- device callback ----------------

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            if not self._queue:
                outdata[:] = 0.0
                return
            if self._started_at is None:
                # First frames of a new utterance actually reaching the device.
                self._started_at = time.perf_counter()

            filled = 0
            while filled < frames and self._queue:
                clip = self._queue[0]
                take = min(frames - filled, len(clip) - self._cursor)
                outdata[filled:filled + take, 0] = clip[self._cursor:self._cursor + take]
                filled += take
                self._cursor += take
                if self._cursor >= len(clip):
                    self._queue.pop(0)
                    self._cursor = 0
            if filled < frames:
                outdata[filled:, 0] = 0.0

    # ---------------- public ----------------

    def play(self, clip_ids: List[str]) -> Playback:
        """Queue clips back to back, interrupting anything already playing.

        Interrupting is deliberate: a newer decision is by definition based on
        newer evidence, and stacking warnings on top of each other is how a
        driver ends up listening to none of them.
        """
        missing = [c for c in clip_ids if c not in self.clips]
        if missing:
            return Playback(clip_id="+".join(clip_ids), latency_ms=0.0,
                            duration_ms=0.0, played=False)

        buffers = [self.clips[c] for c in clip_ids]
        duration_ms = sum(len(b) for b in buffers) / self.rate * 1000.0
        requested = time.perf_counter()

        with self._lock:
            self._queue = buffers
            self._cursor = 0
            self._requested_at = requested
            self._started_at = None

        # Wait briefly for the callback to stamp the true start, so the number
        # reported is measured rather than assumed.
        deadline = requested + 0.35
        started = None
        while time.perf_counter() < deadline:
            with self._lock:
                started = self._started_at
            if started is not None:
                break
            time.sleep(0.001)

        latency_ms = ((started or time.perf_counter()) - requested) * 1000.0
        return Playback(clip_id="+".join(clip_ids), latency_ms=latency_ms,
                        duration_ms=duration_ms)

    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._queue)

    def close(self) -> None:
        stream, self._stream = getattr(self, "_stream", None), None
        if stream is not None:
            stream.stop()
            stream.close()
