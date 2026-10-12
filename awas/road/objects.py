"""Traffic lights and the vehicle ahead, from a forward camera (plan §4.2).

These are the two signals the state machine already reads and nothing produced:
`traffic_light_color` gates the STOPPED_AT_LIGHT state and the green-light
alert, and `vehicle_ahead_distance_m` drives the traffic-jam nudge and feeds
workload. Without them, three of the machine's behaviours are unreachable
outside a mock.

Both come from one COCO-trained detector — `car`, `truck`, `bus` and
`traffic light` are all classes it already knows — so this costs a 13 MB model
and no training data at all.

**It is throttled on purpose.** The detector costs about 38 ms a frame, against
8 ms for the face mesh; running it every frame would blow the 100 ms perception
budget on a Pi several times over. It does not need to run every frame: a
traffic light does not change colour in 300 ms and a following distance does
not change meaningfully either. Driver monitoring stays at full rate, road
objects run at ~3 Hz, and the loop stays affordable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# COCO categories this cares about. Everything else the detector finds is
# ignored rather than filtered later.
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
LIGHT_CLASS = "traffic light"
PERSON_CLASS = "person"

# Typical vehicle width, for the monocular distance estimate. A car is about
# 1.8 m across; trucks and buses are wider, which is why the class matters.
VEHICLE_WIDTH_M: Dict[str, float] = {
    "car": 1.8, "motorcycle": 0.8, "truck": 2.5, "bus": 2.5,
}


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str
    score: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def bottom(self) -> float:
        return self.y + self.h


@dataclass
class RoadObjects:
    lead_vehicle_distance_m: Optional[float] = None
    lead_vehicle_class: str = ""
    traffic_light_color: Optional[str] = None
    traffic_light_score: float = 0.0
    vehicle_count: int = 0
    person_count: int = 0
    latency_ms: float = 0.0
    stale: bool = False          # returned from cache rather than measured now
    boxes: List[Box] = field(default_factory=list)


# ---------------- geometry ----------------

def distance_from_width(pixel_width: float, image_width: int,
                        real_width_m: float) -> Optional[float]:
    """Monocular distance from apparent width, via the pinhole relation.

        distance = real_width * focal_px / pixel_width

    with focal length approximated by the image width, as in `landmarks.py`.
    This is a ROUGH estimate — good enough to tell "right behind someone" from
    "open road", which is all the workload term and the jam nudge need. It is
    not good enough to publish a following-distance figure, and the plan does
    not ask it to be.
    """
    if pixel_width <= 1.0:
        return None
    focal_px = float(image_width)
    d = real_width_m * focal_px / pixel_width
    return d if 1.0 <= d <= 200.0 else None


def pick_lead_vehicle(boxes: List[Box], image_width: int,
                      lane_fraction: float = 0.4) -> Optional[Box]:
    """The vehicle we are following: in our lane, and nearest.

    "In our lane" is approximated by horizontal position near the image centre,
    and "nearest" by how low it sits in the frame — both hold for a
    forward-facing camera on a straight road and both fail on a sharp bend,
    which is a limitation worth stating rather than hiding.
    """
    half = image_width * lane_fraction / 2.0
    centre = image_width / 2.0
    candidates = [b for b in boxes
                  if b.label in VEHICLE_CLASSES and abs(b.cx - centre) <= half]
    return max(candidates, key=lambda b: b.bottom) if candidates else None


# ---------------- traffic light colour ----------------

def classify_light(crop: np.ndarray) -> Tuple[Optional[str], float]:
    """Red, amber or green from a cropped traffic-light box.

    Two independent cues, because either alone is unreliable: which hue
    dominates the lit region, and WHERE in the box the bright pixels are —
    lights are stacked red on top, green at the bottom. When the two agree the
    answer is confident; when they disagree the hue wins but confidence drops,
    since a light seen at an angle still shows its colour.
    """
    import cv2

    if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 3:
        return None, 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Only lit pixels vote: saturated and bright.
    lit = (s > 90) & (v > 110)
    if lit.sum() < max(6, crop.size * 0.01):
        return None, 0.0

    hue = h[lit]
    masks = {
        "red": ((hue <= 10) | (hue >= 170)),
        "yellow": ((hue > 10) & (hue <= 33)),
        "green": ((hue > 38) & (hue <= 92)),
    }
    counts = {k: int(m.sum()) for k, m in masks.items()}
    by_hue = max(counts, key=counts.get)
    if counts[by_hue] == 0:
        return None, 0.0
    hue_share = counts[by_hue] / len(hue)

    # Vertical position of the lit region within the box.
    rows = np.where(lit.any(axis=1))[0]
    by_position = None
    if len(rows):
        centre = float(rows.mean()) / crop.shape[0]
        by_position = "red" if centre < 0.4 else ("yellow" if centre < 0.68 else "green")

    agree = by_position == by_hue
    confidence = min(hue_share * (1.25 if agree else 0.7), 1.0)
    return by_hue, float(confidence)


# ---------------- detector ----------------

class RoadObjectDetector:
    """COCO detector, throttled, producing the two missing road signals."""

    def __init__(
        self,
        model_path: Path = Path("models/efficientdet_lite0.tflite"),
        interval_sec: float = 0.33,          # ~3 Hz; see the module docstring
        score_threshold: float = 0.35,
        min_light_score: float = 0.30,
    ):
        self.model_path = Path(model_path)
        self.interval_sec = interval_sec
        self.min_light_score = min_light_score

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Object detector model not found at {self.model_path}.\n"
                f"Fetch it with:  ./scripts/fetch_models.sh"
            )

        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            ObjectDetector, ObjectDetectorOptions, RunningMode,
        )

        self._detector = ObjectDetector.create_from_options(
            ObjectDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(self.model_path),
                                         delegate=BaseOptions.Delegate.CPU),
                running_mode=RunningMode.VIDEO,
                score_threshold=score_threshold,
                max_results=25,
            )
        )
        self._last: RoadObjects = RoadObjects()
        self._last_t: Optional[float] = None

    def due(self, t: float) -> bool:
        return self._last_t is None or (t - self._last_t) >= self.interval_sec

    def detect(self, frame: np.ndarray, t: float) -> RoadObjects:
        """Detect, or return the previous answer if it is not time yet."""
        if not self.due(t):
            cached = RoadObjects(**{**self._last.__dict__, "stale": True})
            return cached

        import mediapipe as mp

        started = time.perf_counter()
        h, w = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._detector.detect_for_video(image, int(t * 1000))

        boxes: List[Box] = []
        for d in result.detections:
            if not d.categories:
                continue
            bb = d.bounding_box
            boxes.append(Box(
                x=float(bb.origin_x), y=float(bb.origin_y),
                w=float(bb.width), h=float(bb.height),
                label=d.categories[0].category_name or "",
                score=float(d.categories[0].score),
            ))

        out = RoadObjects(
            boxes=boxes,
            vehicle_count=sum(1 for b in boxes if b.label in VEHICLE_CLASSES),
            person_count=sum(1 for b in boxes if b.label == PERSON_CLASS),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

        lead = pick_lead_vehicle(boxes, w)
        if lead is not None:
            out.lead_vehicle_class = lead.label
            out.lead_vehicle_distance_m = distance_from_width(
                lead.w, w, VEHICLE_WIDTH_M.get(lead.label, 1.8))

        lights = [b for b in boxes
                  if b.label == LIGHT_CLASS and b.score >= self.min_light_score]
        if lights:
            # The one that matters is the largest — nearest, and ours.
            box = max(lights, key=lambda b: b.w * b.h)
            crop = frame[int(box.y):int(box.bottom), int(box.x):int(box.x + box.w)]
            colour, conf = classify_light(crop)
            out.traffic_light_color = colour
            out.traffic_light_score = conf

        self._last, self._last_t = out, t
        return out

    def close(self) -> None:
        if getattr(self, "_detector", None) is not None:
            self._detector.close()
            self._detector = None
