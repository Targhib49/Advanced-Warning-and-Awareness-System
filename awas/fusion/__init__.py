"""Fusion core — the research contribution (plan §4.3).

Combines facial evidence (Layer 1) with vehicle-behaviour evidence (Layer 2)
into `drowsiness_level`, `confidence`, and `workload`.

Vehicle behaviour degrades earlier than facial signals and survives occlusion,
so fusing the two should detect onset earlier at a matched false-alarm rate
(hypothesis H2). Start with a transparent weighted-evidence model, not a
learned black box: explainable, debuggable, needs no paired training data we
do not have, and gives an honest baseline for a learned model later.
"""
