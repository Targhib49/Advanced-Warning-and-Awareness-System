"""Pre-render every phrase to WAV. Run once, before the system runs.

    python -m scripts.render_phrases
    python -m scripts.render_phrases --voice Daniel --force

Uses macOS `say` here and Piper on the Pi — the renderer is swappable because
this runs offline at build time, not on the road. Output lands in
`assets/phrases/` as 16-bit mono PCM, which loads with the standard library
and needs no audio codec at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

from awas.companion.phrasebank import ALERTS_WITH_TONE, all_clips

OUT_DIR = Path("assets/phrases")
SAMPLE_RATE = 22050
TONE_ID = "tone__alarm"


def write_wav(path: Path, samples: bytes, rate: int = SAMPLE_RATE) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples)


def make_alarm_tone(seconds: float = 0.45, rate: int = SAMPLE_RATE) -> bytes:
    """A rising two-tone alarm, generated rather than recorded.

    Deliberately not speech: at level 4 the driver may not parse words fast
    enough, and a rising pitch reads as urgent pre-linguistically.
    """
    n = int(rate * seconds)
    out = bytearray()
    for i in range(n):
        t = i / rate
        freq = 660.0 + 440.0 * (t / seconds)
        # Fade the last 15% so it does not click when the buffer ends.
        env = 1.0 if t < seconds * 0.85 else (seconds - t) / (seconds * 0.15)
        v = math.sin(2.0 * math.pi * freq * t) * 0.45 * env
        out += struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32767))
    return bytes(out)


def render_speech(text: str, path: Path, voice: str | None) -> None:
    if not shutil.which("say"):
        raise SystemExit(
            "`say` not found. On macOS it is built in; on the Pi use Piper and "
            "point this script at it instead."
        )
    cmd = ["say", "-o", str(path), "--data-format=LEI16@22050"]
    if voice:
        cmd += ["-v", voice]
    cmd += [text]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-render the phrase bank to WAV.")
    ap.add_argument("--voice", default=None, help="macOS voice name, e.g. Daniel")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true", help="re-render existing clips")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clips = all_clips()
    manifest = {}
    rendered = skipped = 0

    tone_path = out / f"{TONE_ID}.wav"
    if args.force or not tone_path.exists():
        write_wav(tone_path, make_alarm_tone())
        rendered += 1
    manifest[TONE_ID] = {"text": "(generated alarm tone)", "file": tone_path.name}

    for clip_id, text in sorted(clips.items()):
        path = out / f"{clip_id}.wav"
        if path.exists() and not args.force:
            skipped += 1
        else:
            render_speech(text, path, args.voice)
            rendered += 1
        manifest[clip_id] = {
            "text": text,
            "file": path.name,
            "tone_first": clip_id.replace("alert__", "") in ALERTS_WITH_TONE,
        }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    total_kb = sum(p.stat().st_size for p in out.glob("*.wav")) / 1024
    print(f"rendered {rendered}, skipped {skipped}  ->  {out}  ({total_kb:.0f} KB)")


if __name__ == "__main__":
    main()
