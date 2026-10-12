#!/usr/bin/env bash
# Rebuild the Section IV .docx from the markdown source and current figures.
#
#     ./scripts/build_paper.sh
#
# Regenerate the inputs first if any measurement has changed:
#     python -m scripts.parameter_sweep
#     python -m scripts.layer2_experiments
#     python -m scripts.make_figures
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v pandoc >/dev/null 2>&1 || {
  echo "pandoc not found — install it, then re-run." >&2; exit 1; }

mkdir -p docs/build
"$ROOT/venv/bin/python" - <<'PY'
import pathlib
src = pathlib.Path("docs/04-results-and-discussion.md").read_text()
figs = [
 ("Maximum error is 0.020 m over a ±0.8 m range, at 2.98 ms per frame.",
  "![**Fig. 1.** Lane calibration diagnostic. Green outlines the sampled trapezoid; the right panel is the warped binary the detector operates on.](../artifacts/lane_calibration.png)"),
 ("occlusion result in §D-2 is precisely the response to it.",
  "![**Fig. 2.** Eye-state classification on seven held-out subjects, and accuracy by capture condition.](../figures/fig1_detection.png)"),
 ("differs between a private vehicle and a commercial fleet.",
  "![**Fig. 3.** What road awareness adds: occlusion survival, and false-positive suppression.](../figures/fig6_layer2.png)"),
 ("that point the display is no longer supporting a driving task.",
  "![**Fig. 4.** The in-cabin display across system states. Channel provenance, gate transparency and personal-baseline SDLP are all products of the road-awareness layer.](../figures/fig7_interface.png)"),
 ("No code path exists from a generator to the loudspeaker.",
  "![**Fig. 5.** Measured latency against the design budget for each stage.](../figures/fig2_latency.png)"),
 ("while following closely, and immediately after a lane departure.",
  "![**Fig. 6.** Hysteresis depth against latency, and workload-gate selectivity.](../figures/fig5_sensitivity.png)"),
]
missing = []
for anchor, fig in figs:
    if anchor in src:
        src = src.replace(anchor, f"{anchor}\n\n{fig}\n", 1)
    else:
        missing.append(anchor[:44])
pathlib.Path("docs/build/section4.md").write_text(src)
if missing:
    print("  WARNING: figure anchors not found (text edited?):")
    for m in missing:
        print(f"    {m}...")
PY

cd docs/build
pandoc section4.md -o "AWAS-Section-IV-Results-and-Discussion.docx" \
  --resource-path=.:..:../..

echo "  built docs/build/AWAS-Section-IV-Results-and-Discussion.docx"
echo "  tables: $(pandoc -t plain 'AWAS-Section-IV-Results-and-Discussion.docx' | grep -c 'TABLE ')"
echo "  images: $(unzip -l 'AWAS-Section-IV-Results-and-Discussion.docx' | grep -c 'media/')"
