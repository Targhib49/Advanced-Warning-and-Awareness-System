"""Experimental conditions (plan §6.3).

The comparison that matters is not "does AWAS detect drowsiness" — the MRL
result already answers that — but "does talking to the driver keep them alert
longer than a beep does". Answering it needs the same perception feeding
different responses, so that any difference in outcome is attributable to the
intervention and nothing else.

That is what these are. The state machine is untouched: it still perceives and
decides identically in every condition. The condition filters only what the
executor is permitted to DO with its decisions.

Condition B is the conventional-ADAS baseline and it is implemented FAIRLY —
same detector, same thresholds, same timing, just a tone instead of a
conversation. A straw-man baseline would make H1 trivially true and worthless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Condition:
    id: str
    name: str
    description: str

    reflex: bool = True      # may play reflex alerts at all
    speech: bool = True      # alerts may use words rather than a bare tone
    dialogue: bool = True    # may hold conversational interventions
    gating: bool = True      # context/workload gating active

    @property
    def silent(self) -> bool:
        return not (self.reflex or self.dialogue)


CONDITIONS: Dict[str, Condition] = {
    "A": Condition(
        id="A", name="no intervention",
        description="Detects and logs, never responds. The floor: how does "
                    "alertness behave when nothing acts on it?",
        reflex=False, speech=False, dialogue=False, gating=True,
    ),
    "B": Condition(
        id="B", name="tone alarm",
        description="Conventional ADAS baseline. Same detector and thresholds, "
                    "but the response is a tone with no words and no dialogue.",
        reflex=True, speech=False, dialogue=False, gating=True,
    ),
    "C": Condition(
        id="C", name="AWAS",
        description="The full system: reflex alerts plus adaptive conversation, "
                    "context-gated.",
        reflex=True, speech=True, dialogue=True, gating=True,
    ),
    "D": Condition(
        id="D", name="AWAS, ungated",
        description="Ablation for H3. Everything of C, but interventions are "
                    "not suppressed by workload — does gating actually help?",
        reflex=True, speech=True, dialogue=True, gating=False,
    ),
}

DEFAULT = "C"


def get(condition_id: str) -> Condition:
    key = (condition_id or DEFAULT).strip().upper()
    if key not in CONDITIONS:
        raise SystemExit(
            f"Unknown condition {condition_id!r}. Choose one of: "
            + ", ".join(f"{k} ({v.name})" for k, v in CONDITIONS.items())
        )
    return CONDITIONS[key]


def describe_all() -> str:
    return "\n".join(
        f"  {c.id}  {c.name:16} {c.description}" for c in CONDITIONS.values()
    )
