"""Layer 2 — road and vehicle context (plan §4.2).

Deliberately narrow: speed, stopped duration, lane position + SDLP, following
distance, traffic light state, road class. Each one exists because it changes
an intervention decision.

Out of scope, by decision: pedestrian detection, general obstacle detection,
collision prediction, sign recognition. AWAS is not an ADAS replacement.
"""
