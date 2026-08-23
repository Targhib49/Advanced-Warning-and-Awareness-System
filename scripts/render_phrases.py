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


def render_with_say(text: str, path: Path, voice: str | None) -> None:
    """macOS built-in TTS."""
    cmd = ["say", "-o", str(path), "--data-format=LEI16@22050"]
    if voice:
        cmd += ["-v", voice]
    cmd += [text]
    subprocess.run(cmd, check=True, capture_output=True)


def render_with_piper(text: str, path: Path, model: str) -> None:
    """Piper: offline neural TTS, the Pi's answer to `say`.

    Piper voices are per-language, so this is also where an Indonesian build
    would switch over — pass an id_ID voice model and the whole bank re-renders
    in Indonesian with no other code change.
    """
    subprocess.run(
        ["piper", "--model", model, "--output_file", str(path)],
        input=text, text=True, check=True, capture_output=True,
    )


def render_speech(text: str, path: Path, voice: str | None,
                  piper_model: str | None) -> None:
    """Render one line with whichever engine this machine has.

    Both run offline and both happen at BUILD time, never while driving — which
    is why swapping engines between platforms is safe. The runtime only ever
    plays files.
    """
    if piper_model:
        if not shutil.which("piper"):
            raise SystemExit(
                "piper not found on PATH. Install it, or drop --piper-model to "
                "use the system TTS."
            )
        return render_with_piper(text, path, piper_model)

    if shutil.which("say"):
        return render_with_say(text, path, voice)

    if shutil.which("piper"):
        raise SystemExit(
            "piper is installed but no voice model was given. Pass one:\n"
            "    python -m scripts.render_phrases --piper-model /path/voice.onnx"
        )

    raise SystemExit(
        "No text-to-speech engine found.\n"
        "  macOS: `say` is built in and should already be here.\n"
        "  Pi:    install piper, then pass --piper-model <voice>.onnx"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-render the phrase bank to WAV.")
    ap.add_argument("--voice", default=None, help="macOS voice name, e.g. Daniel")
    ap.add_argument("--piper-model", default=None,
                    help="path to a Piper .onnx voice (Raspberry Pi)")
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
            render_speech(text, path, args.voice, args.piper_model)
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
