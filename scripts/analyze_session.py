"""Read a session log and compute the measures §6 is built on.

    python -m scripts.analyze_session data/sessions/*.jsonl

This exists as much to VALIDATE the log schema as to report numbers: if
time-to-relapse cannot be computed from a session file, the schema is wrong and
the time to find that out is now, not after the study has been run.

Time-to-relapse is the primary outcome for hypothesis H1 — seconds from an
intervention until drowsiness returns to the level that triggered it. Longer is
better. It is the number that says whether talking to the driver did anything,
and no system on the market records it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def load(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def time_to_relapse(
    records: List[dict], horizon_sec: float = 600.0
) -> List[dict]:
    """For each intervention, how long until drowsiness came back."""
    samples = [r for r in records if r["type"] == "sample" and r.get("level") is not None]
    actions = [r for r in records if r["type"] == "action" and r["kind"] in ("alert", "speak")]

    out = []
    for a in actions:
        trigger_level = a.get("trigger", {}).get("level")
        if trigger_level is None or trigger_level < 1:
            continue
        t0 = a["t"]

        # Must first drop BELOW the trigger level, or "relapse" is just the
        # same unbroken episode still in progress.
        recovered_at: Optional[float] = None
        relapsed_at: Optional[float] = None
        for s in samples:
            if s["t"] <= t0:
                continue
            if s["t"] - t0 > horizon_sec:
                break
            if recovered_at is None:
                if s["level"] < trigger_level:
                    recovered_at = s["t"]
            elif s["level"] >= trigger_level:
                relapsed_at = s["t"]
                break

        out.append({
            "t": t0,
            "kind": a["kind"],
            "label": a.get("alert_id") or a.get("intent") or "?",
            "level": trigger_level,
            "recovered": recovered_at is not None,
            "ttr_sec": (relapsed_at - t0) if relapsed_at else None,
            "censored": recovered_at is not None and relapsed_at is None,
        })
    return out


def summarize(path: Path) -> None:
    records = load(path)
    start = next((r for r in records if r["type"] == "session_start"), {})
    end = next((r for r in records if r["type"] == "session_end"), {})
    samples = [r for r in records if r["type"] == "sample"]
    actions = [r for r in records if r["type"] == "action" and r["kind"] in ("alert", "speak")]

    duration = max((r["t"] for r in records), default=0.0)
    hours = duration / 3600.0 if duration else 0.0

    print(f"session   {start.get('session_id','?')}  driver={start.get('driver','?')}")
    print(f"duration  {duration:.0f} s   samples={len(samples)}   interventions={len(actions)}")
    if hours:
        print(f"rate      {len(actions)/hours:.1f} interventions/hour")

    levels = [s["level"] for s in samples if s.get("level") is not None]
    if levels:
        worst = max(levels)
        share = sum(1 for l in levels if l >= 3) / len(levels)
        print(f"levels    peak {worst}   {share:.1%} of samples at level 3+")

    lat = [a["latency_ms"] for a in actions if a.get("latency_ms")]
    if lat:
        reflex = [a["latency_ms"] for a in actions
                  if a["kind"] == "alert" and a.get("latency_ms")]
        print(f"latency   median {statistics.median(lat):.1f} ms", end="")
        if reflex:
            worst_reflex = max(reflex)
            verdict = "within" if worst_reflex < 300 else "OVER"
            print(f"   worst reflex {worst_reflex:.1f} ms ({verdict} the 300 ms budget)")
        else:
            print()

    by_label: Dict[str, List[float]] = defaultdict(list)
    ttrs = time_to_relapse(records)
    for r in ttrs:
        if r["ttr_sec"] is not None:
            by_label[r["label"]].append(r["ttr_sec"])

    print()
    print("time-to-relapse by intervention   (H1 primary outcome)")
    if not ttrs:
        print("  no interventions above level 0 in this session")
        return
    for label, values in sorted(by_label.items(), key=lambda kv: -statistics.median(kv[1])):
        print(f"  {label:24} n={len(values):3d}   median {statistics.median(values):7.1f} s")

    censored = sum(1 for r in ttrs if r["censored"])
    never = sum(1 for r in ttrs if not r["recovered"])
    if censored:
        print(f"  {censored} intervention(s) never relapsed within the horizon "
              f"— censored, not zero")
    if never:
        print(f"  {never} intervention(s) never recovered at all "
              f"— the driver stayed drowsy through them")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize AWAS session logs.")
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for i, p in enumerate(args.paths):
        if i:
            print("\n" + "-" * 62 + "\n")
        summarize(Path(p))


if __name__ == "__main__":
    main()
