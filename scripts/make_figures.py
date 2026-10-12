"""Turn the measured artifacts into publication figures.

    python -m scripts.make_figures

Reads whatever exists in artifacts/ and data/sessions/ and writes PNG (300 dpi,
for slides and posters) plus PDF (vector, for a paper) into figures/. Anything
missing is skipped with a note rather than failing — a partial figure set beats
no figures.

Design constraints, because these end up printed and projected:
  * legible in GREYSCALE — competition posters get photocopied, and a palette
    that only separates by hue collapses to identical grey. Every series is
    separated by lightness as well as hue.
  * no chartjunk, no 3D, no gradients
  * every axis labelled with units, every claim traceable to a real file
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

# Lightness-separated so the figures survive greyscale reproduction.
INK      = "#1A1A1A"
GRID     = "#D8D8D8"
DARK     = "#1F3A5F"   # strongest series
MID      = "#5B8CB8"   # middle
LIGHT    = "#AFC8DD"   # weakest / baseline
WARN     = "#B4531A"   # the honest weakness
BUDGET   = "#8A8A8A"

FIGDIR = Path("figures")


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def save(fig, name: str) -> None:
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.png and .pdf")


def load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- fig 1

def fig_detection(metrics: dict) -> None:
    """Held-out accuracy against baselines, and where it breaks down."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0),
                                   gridspec_kw={"width_ratios": [1, 1.4],
                                                "wspace": 0.42})

    names = ["Majority\nclass", "Logistic\nregression", "EyeStateNet\n(CNN)"]
    vals = [metrics["test_acc_majority"], metrics["test_acc_logreg"],
            metrics["test_acc_cnn"]]
    bars = ax1.bar(names, [v * 100 for v in vals],
                   color=[LIGHT, MID, DARK], width=0.62)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5,
                 f"{v*100:.1f}", ha="center", fontsize=8.5, weight="bold")
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Held-out subjects", fontsize=9.5, loc="left", weight="bold")
    ax1.grid(axis="x", visible=False)

    slices, accs, cols = [], [], []
    order = [("glasses", "no_glasses", "no glasses"), ("glasses", "glasses", "glasses"),
             ("lighting", "good_light", "good light"), ("lighting", "bad_light", "bad light"),
             ("sensor", "realsense", "RealSense"), ("sensor", "ids", "IDS")]
    for key, val, label in order:
        d = metrics.get("by_slice", {}).get(key, {}).get(val)
        if d:
            slices.append(f"{label}\nn={d['n']:,}")
            accs.append(d["acc"] * 100)
            cols.append(WARN if val == "glasses" else MID)

    y = np.arange(len(slices))
    ax2.barh(y, accs, color=cols, height=0.6)
    ax2.set_yticks(y, slices, fontsize=7.6)
    ax2.invert_yaxis()
    ax2.set_xlim(90, 100.4)
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_title("By condition — glasses is the weakness",
                  fontsize=9.5, loc="left", weight="bold")
    for i, a in enumerate(accs):
        ax2.text(a + 0.12, i, f"{a:.1f}", va="center", fontsize=7.6)
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Eye-state classification, split by subject",
                 fontsize=10.5, weight="bold", x=0.005, ha="left", y=1.04)
    save(fig, "fig1_detection")


# ---------------------------------------------------------------- fig 2

def fig_latency(profile: dict) -> None:
    """Measured stages against the plan's budgets."""
    rows = []
    lm = profile.get("landmarks", {})
    if "p95_ms" in lm:
        rows.append(("Face mesh + EAR + pose", lm["p95_ms"], 50.0))
    if "loop_p95_ms" in profile:
        rows.append(("Full perception loop", profile["loop_p95_ms"], 100.0))
    au = profile.get("audio", {})
    if "max_ms" in au:
        rows.append(("Decision → audible alert", au["max_ms"], 300.0))
    if not rows:
        print("  skipped fig2: no latency measurements in the profile")
        return

    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    y = np.arange(len(rows))
    labels = [r[0] for r in rows]
    measured = [r[1] for r in rows]
    budgets = [r[2] for r in rows]
    used = [m / b * 100 for m, b in zip(measured, budgets)]

    # Percentage of each stage's OWN budget. Absolute milliseconds are not
    # comparable across rows (the budgets are 50, 100 and 300 ms), but
    # "how much of the allowance did it spend" is — and headroom is the thing
    # a reader actually wants to see.
    ax.barh(y, 100, color=GRID, height=0.52)
    ax.barh(y, used, color=DARK, height=0.52)

    for i, (m, b, u) in enumerate(zip(measured, budgets, used)):
        ax.text(u + 1.8, i, f"{m:.1f} ms of {b:.0f} ms  ({u:.0f}%)",
                va="center", fontsize=8.2, color=INK)

    ax.set_yticks(y, labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("percentage of the budget used")
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.grid(axis="y", visible=False)
    dev = profile.get("device", {})
    ax.set_title(f"Every stage well inside budget — {dev.get('machine', '?')}, "
                 f"{dev.get('cores', '?')} cores",
                 fontsize=10, loc="left", weight="bold")
    ax.text(0, -0.62, "Grey is the plan §7.2 allowance for that stage. "
                      "Measured values are p95, except the alert which is worst case.",
            fontsize=7.4, color=BUDGET, transform=ax.get_xaxis_transform())
    save(fig, "fig2_latency")


# ---------------------------------------------------------------- fig 3

def fig_personalization() -> None:
    """The calibration argument, computed live rather than hard-coded."""
    from awas.driver.calibration import (
        POPULATION_CLOSED_EAR, POPULATION_OPEN_EAR, DriverProfile,
        OpennessNormalizer,
    )

    drivers = [("Wide-set eyes", 0.38, 0.025), ("Narrow aperture", 0.20, 0.015)]
    droops = np.linspace(0, 0.99, 400)
    population = OpennessNormalizer(None)

    def first_closed(norm, o, c):
        """The true droop at which this scale first calls the eye shut."""
        span = o - c
        for d in droops:
            if norm.is_closed(o - d * span):
                return d
        return None

    pop_points, per_points = [], []
    for name, o, c in drivers:
        personal = OpennessNormalizer(DriverProfile(name, o, c, n_samples=9000))
        pop_points.append(first_closed(population, o, c))
        per_points.append(first_closed(personal, o, c))

    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    cols = [DARK, MID]

    for row, (points, label) in enumerate(
            [(per_points, "Personal scale"), (pop_points, "Population threshold")]):
        valid = [p for p in points if p is not None]
        if len(valid) == 2:
            spread = abs(valid[0] - valid[1]) * 100
            ax.plot(valid, [row, row], color=INK, lw=1.2, alpha=0.35,
                    zorder=1, solid_capstyle="butt")
            mid = sum(valid) / 2
            note = ("identical" if spread < 1
                    else f"{spread:.0f} points apart")
            ax.text(mid, row + 0.28, note, ha="center", fontsize=8.4,
                    color=INK if spread < 1 else WARN,
                    weight="bold" if spread < 1 else "normal")
        # Draw the first marker larger so that when the two coincide it reads
        # as two rings rather than one dot with a driver mysteriously missing.
        for i, (p, col) in enumerate(zip(points, cols)):
            if p is not None:
                ax.plot([p], [row], marker="o", ms=15 if i == 0 else 9,
                        color=col, markeredgecolor="white",
                        markeredgewidth=1.6, zorder=3 + i)

    ax.set_yticks([0, 1], ["Personal\nscale", "Population\nthreshold"], fontsize=8.8)
    ax.set_ylim(-0.55, 1.65)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("True eyelid droop, as a fraction of that driver's own range")
    ax.set_title("When each scale declares the eye closed",
                 fontsize=10, loc="left", weight="bold")
    ax.grid(axis="y", visible=False)

    handles = [plt.Line2D([], [], marker="o", ls="none", ms=8, color=c, label=n)
               for (n, _, _), c in zip(drivers, cols)]
    ax.legend(handles=handles, frameon=False, fontsize=8,
              loc="lower left", bbox_to_anchor=(0, -0.02), ncol=2)
    save(fig, "fig3_personalization")


# ---------------------------------------------------------------- fig 4

def fig_layer2(res: dict) -> None:
    """The road-awareness contribution: occlusion survival and the fusion trade."""
    e2 = res.get("e2_occlusion", [])
    e3 = res.get("e3_false_alarms", [])
    if not e2 or not e3:
        print("  skipped fig6: no layer2 results")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.1),
                                   gridspec_kw={"wspace": 0.34})

    labels = [r["config"] for r in e2]
    detect = [r["detection_rate"] * 100 for r in e2]
    cover = [r["mean_coverage"] * 100 for r in e2]
    x = np.arange(len(labels))
    w = 0.34

    ax1.bar(x - w / 2, cover, width=w, color=LIGHT, label="channel available")
    ax1.bar(x + w / 2, detect, width=w, color=DARK, label="onset detected")
    for xi, (c, d) in enumerate(zip(cover, detect)):
        ax1.text(xi - w / 2, c + 3, f"{c:.0f}%", ha="center", fontsize=8, color=BUDGET)
        # A zero bar draws nothing, so mark it explicitly rather than leaving a
        # floating label over empty axis.
        if d < 1:
            ax1.plot([xi + w / 2], [0], marker="_", ms=18, color=WARN, mew=2.5)
            ax1.text(xi + w / 2, 6, "0%", ha="center", fontsize=9,
                     weight="bold", color=WARN)
        else:
            ax1.text(xi + w / 2, d + 3, f"{d:.0f}%", ha="center", fontsize=9,
                     weight="bold", color=DARK)

    ax1.set_xticks(x, labels, fontsize=8.6)
    ax1.set_ylim(0, 122)
    ax1.set_ylabel("percent")
    ax1.legend(frameon=False, fontsize=7.6, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax1.grid(axis="x", visible=False)
    ax1.set_title("Face occluded: the road channel carries it",
                  fontsize=9.5, loc="left", weight="bold")

    labels3 = [r["config"].replace("+ vehicle, ", "") for r in e3]
    fires = [r["fires_at_l3"] * 100 for r in e3]
    cols = [WARN if f > 50 else DARK for f in fires]
    ax2.bar(range(len(labels3)), fires, color=cols, width=0.6)
    for i, f in enumerate(fires):
        ax2.text(i, f + 3, f"{f:.0f}%", ha="center", fontsize=8.5, weight="bold")
    ax2.set_xticks(range(len(labels3)), labels3, fontsize=8)
    ax2.set_ylim(0, 122)
    ax2.set_ylabel("false alarms raised (%)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title("Facial misread, lanes steady",
                  fontsize=9.5, loc="left", weight="bold")

    save(fig, "fig6_layer2")


def fig_sensitivity(sweeps: dict) -> None:
    """Hysteresis latency cost and workload-gate selectivity, side by side."""
    hyst = sweeps.get("hysteresis", [])
    wl = sweeps.get("workload_ceiling", [])
    if not hyst or not wl:
        print("  skipped fig5: no sweep data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.7),
                                   gridspec_kw={"wspace": 0.32})

    ticks = [r["escalate_ticks"] for r in hyst]
    lat = [r["median_latency_sec"] for r in hyst]
    ax1.plot(ticks, lat, marker="o", color=DARK, lw=1.6, ms=6)
    deployed = ticks.index(3) if 3 in ticks else None
    if deployed is not None:
        ax1.plot([ticks[deployed]], [lat[deployed]], marker="o", ms=12,
                 mfc="none", mec=WARN, mew=2)
        ax1.annotate("deployed", (ticks[deployed], lat[deployed]),
                     textcoords="offset points", xytext=(8, -14),
                     fontsize=8, color=WARN)
    ax1.set_xlabel("escalation ticks required")
    ax1.set_ylabel("median detection latency (s)")
    ax1.set_title("Hysteresis costs ~1.1 s per tick",
                  fontsize=9.5, loc="left", weight="bold")

    ceil = [r["ceiling"] for r in wl]
    share = [r["fraction"] * 100 for r in wl]
    ax2.step(ceil, share, where="post", color=DARK, lw=1.6)
    ax2.fill_between(ceil, 0, share, step="post", color=DARK, alpha=0.10)
    if 0.4 in ceil:
        i = ceil.index(0.4)
        ax2.plot([0.4], [share[i]], marker="o", ms=12, mfc="none",
                 mec=WARN, mew=2)
        ax2.annotate("deployed", (0.4, share[i]), textcoords="offset points",
                     xytext=(8, -14), fontsize=8, color=WARN)
    ax2.set_xlabel("workload ceiling")
    ax2.set_ylabel("situations permitting speech (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("The gate is selective and tunable",
                  fontsize=9.5, loc="left", weight="bold")

    save(fig, "fig5_sensitivity")


def fig_session(path: Path) -> None:
    """One session: alertness over time, with what the system did."""
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    samples = [r for r in records if r["type"] == "sample" and r.get("level") is not None]
    actions = [r for r in records if r["type"] == "action" and r["kind"] in ("alert", "speak")]
    kss = [r for r in records if r.get("kind") == "kss"]
    if len(samples) < 5:
        print(f"  skipped fig4: {path.name} has too few samples")
        return

    t = np.array([s["t"] for s in samples]) / 60.0
    lvl = np.array([s["level"] for s in samples])

    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    ax.step(t, lvl, where="post", color=DARK, lw=1.4)
    ax.fill_between(t, 0, lvl, step="post", color=DARK, alpha=0.10)

    for a in actions:
        col = WARN if a["kind"] == "alert" else MID
        ax.axvline(a["t"] / 60.0, color=col, lw=0.8, alpha=0.75)
    for k in kss:
        ax.plot(k["t"] / 60.0, 4.6, marker="v", ms=6, color=INK, ls="none")
        ax.text(k["t"] / 60.0, 4.75, str(k.get("value", "")), fontsize=7,
                ha="center", color=INK)

    ax.set_ylim(-0.2, 5.1)
    ax.set_yticks(range(5), ["alert", "tired", "drowsy", "unsafe", "critical"],
                  fontsize=8)
    ax.set_xlabel("minutes")
    ax.set_xlim(0, max(t))
    ax.grid(axis="y", visible=False)

    start = next((r for r in records if r["type"] == "session_start"), {})
    cond = start.get("config", {}).get("condition", "?")
    ax.set_title(f"Session timeline — condition {cond}   "
                 f"(orange = reflex alert, blue = dialogue, ▾ = KSS)",
                 fontsize=9.5, loc="left", weight="bold")
    save(fig, "fig4_session")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Build publication figures.")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--sessions", default="data/sessions")
    args = ap.parse_args()

    style()
    art = Path(args.artifacts)
    print("building figures")

    metrics = load(art / "eyestate_metrics.json")
    if metrics and not metrics.get("smoke"):
        fig_detection(metrics)
    else:
        print("  skipped fig1: no real training metrics "
              "(run training/train_eyestate.py without --smoke)")

    profile = load(art / "host_profile.json") or load(art / "pi_profile.json")
    if profile:
        fig_latency(profile)
    else:
        print("  skipped fig2: no device profile "
              "(run scripts/profile_device.py --out artifacts/host_profile.json)")

    fig_personalization()

    layer2 = load(art / "layer2.json")
    if layer2:
        fig_layer2(layer2)
    else:
        print("  skipped fig6: no layer2.json "
              "(run scripts/layer2_experiments.py)")

    sweeps = load(art / "sweeps.json")
    if sweeps:
        fig_sensitivity(sweeps)
    else:
        print("  skipped fig5: no sweeps.json "
              "(run scripts/parameter_sweep.py)")

    fig_interface()

    logs = sorted(Path(args.sessions).glob("*.jsonl"))
    if logs:
        fig_session(logs[-1])
    else:
        print("  skipped fig4: no session logs yet")

    print("\nPNG for slides and posters, PDF for a paper.")



# ---------------------------------------------------------------- fig 7

# The in-cabin display, rendered rather than screenshotted so it regenerates
# with the rest of the figures and stays consistent with the palette.
PI_W, PI_H = 480, 320
PI_BG, PI_INK, PI_DIM, PI_FAINT = "#0A0C0D", "#E8EDEF", "#7C8A91", "#4A565B"
PI_TRACK = "#1D2225"
PI_LEVEL = ["#38D39B", "#E0A93C", "#F2B23E", "#F27A3E", "#F04438"]


def _pi_fonts():
    from matplotlib import font_manager
    from PIL import ImageFont

    def find(*names):
        for n in names:
            try:
                return font_manager.findfont(font_manager.FontProperties(family=n),
                                             fallback_to_default=False)
            except Exception:
                continue
        return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))

    sans = find("IBM Plex Sans", "DejaVu Sans")
    mono = find("IBM Plex Mono", "DejaVu Sans Mono")
    L = ImageFont.truetype
    return {"huge": L(sans, 46), "big": L(sans, 33), "mid": L(sans, 15),
            "small": L(sans, 13), "mono": L(mono, 13), "tiny": L(mono, 11)}


def _pi_screen(level, word, subtitle, *, channels, stats, band=None,
               flood=False, note=None, capped=False):
    """One 480x320 cabin screen."""
    from PIL import Image, ImageDraw

    f = _pi_fonts()
    colour = PI_LEVEL[min(level, 4)]
    img = Image.new("RGB", (PI_W, PI_H), colour if flood else PI_BG)
    d = ImageDraw.Draw(img)

    if flood:
        d.text((PI_W / 2, 118), word, font=f["huge"], fill="#FFFFFF", anchor="mm")
        d.text((PI_W / 2, 168), subtitle, font=f["mid"], fill="#FFD9D6", anchor="mm")
        d.rectangle([0, 262, PI_W, PI_H], fill="#C7362C")
        d.text((26, 288), channels, font=f["tiny"], fill="#FFD9D6", anchor="lm")
        d.text((PI_W - 26, 288), stats, font=f["tiny"], fill="#FFD9D6", anchor="rm")
        return img

    d.rectangle([0, 0, PI_W, 6], fill=band or colour)

    # Channel provenance — new since the original mockup, and the point of it:
    # the driver can see which evidence the estimate is standing on.
    d.ellipse([26, 25, 33, 32], fill=PI_LEVEL[0] if "FACE" in channels else "#5A6A72")
    d.text((44, 29), channels, font=f["tiny"], fill=PI_FAINT, anchor="lm")
    if capped:
        d.text((PI_W - 26, 29), "CAPPED L3", font=f["tiny"], fill="#5A6A72", anchor="rm")

    seg_w, gap = 76, 8
    for i in range(5):
        x = 26 + i * (seg_w + gap)
        if i < level:
            fill = "#2A3135"
        elif i == level:
            fill = colour
        else:
            fill = PI_TRACK
        d.rectangle([x, 96, x + seg_w, 106], fill=fill)

    d.text((26, 146), word, font=f["big"], fill=colour, anchor="lm")
    d.text((26, 176), subtitle, font=f["small"], fill=PI_DIM, anchor="lm")

    if note:
        d.rectangle([26, 200, PI_W - 26, 244], fill="#14181A")
        d.rectangle([26, 200, 29, 244], fill=colour)
        d.text((42, 222), note, font=f["small"], fill=PI_INK, anchor="lm")

    d.line([26, 268, PI_W - 26, 268], fill=PI_TRACK)
    d.text((26, 292), stats, font=f["mono"], fill=PI_DIM, anchor="lm")
    return img


def fig_interface() -> None:
    """The in-cabin display across the states the results describe."""
    try:
        from PIL import Image
    except ImportError:
        print("  skipped fig7: Pillow not available")
        return

    screens = [
        ("(a) monitoring", _pi_screen(
            0, "AWAKE", "Nothing to read. Eyes up.",
            channels="FACE + LANE", capped=False,
            stats="62 km/h   lane ok   workload 0.31")),
        ("(b) engaging", _pi_screen(
            2, "DROWSY", "Waiting for your answer",
            channels="FACE + LANE",
            stats="58 km/h   lane ok   workload 0.28")),
        ("(c) gate holding", _pi_screen(
            3, "UNSAFE", "Held — heavy traffic",
            channels="FACE + LANE", band="#F27A3E",
            note="Speaking as soon as the road eases",
            stats="41 km/h   stop-go   workload 0.86")),
        ("(d) face occluded", _pi_screen(
            2, "LANE ONLY", "Camera blocked. Still watching.",
            channels="LANE", band="#5A6A72", capped=True,
            note="SDLP 0.21 m — baseline +8 cm",
            stats="54 km/h   lane ok   workload 0.33")),
        ("(e) critical", _pi_screen(
            4, "PULL OVER", "Microsleep detected",
            channels="FACE + LANE", flood=True,
            stats="61 km/h")),
    ]

    cols, rows = 3, 2
    pad, label_h = 18, 26
    W = cols * PI_W + (cols + 1) * pad
    H = rows * (PI_H + label_h) + (rows + 1) * pad

    fig, axes = plt.subplots(rows, cols, figsize=(7.4, 7.4 * H / W * 1.16),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.06})
    for ax, (label, img) in zip(axes.flat, screens):
        ax.imshow(np.asarray(img))
        ax.set_title(label, fontsize=8.6, loc="left", color=INK, pad=5)
        ax.axis("off")
    for ax in list(axes.flat)[len(screens):]:
        ax.axis("off")

    fig.suptitle("In-cabin display: what the driver sees at each state",
                 fontsize=10.5, weight="bold", x=0.02, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    save(fig, "fig7_interface")

if __name__ == "__main__":
    main()
