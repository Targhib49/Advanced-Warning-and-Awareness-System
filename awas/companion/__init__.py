"""Layer 3 — the intelligent companion (plan §4.4).

Two paths, and the split is an invariant (I1):

  L3a reflex   pre-rendered audio selected by ID, <300 ms, never generated
  L3b dialogue intent -> template -> optional LLM variation, <3000 ms,
               template ships if the LLM misses its budget or fails the filter

The LLM is never the only thing between a level-4 driver and a warning.
"""
