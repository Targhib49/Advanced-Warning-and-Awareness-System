"""MRL Eye Dataset loader.

The dataset is ~85k infrared eye crops from 37 subjects, annotated in the
FILENAME rather than a manifest. Documented layout:

    <subject>/<subject>_<imgno>_<gender>_<glasses>_<state>_<refl>_<light>_<sensor>.png
    s0001/   s0001_00001_0_0_0_0_0_01.png

    field      values
    subject    s0001..s0037
    imgno      running counter
    gender     0 man, 1 woman
    glasses    0 no, 1 yes
    state      0 closed, 1 open          <- the label
    refl       0 none, 1 small, 2 big
    light      0 bad, 1 good
    sensor     01 RealSense | 02 IDS | 03 Aptina

The field ORDER and value meanings above are confirmed against the official
dataset page (mrl.cs.vsb.cz/eyedataset.html). What is not independently
confirmed is that the archive encodes them in the filename in exactly this
underscore-separated form, so the parser stays strict: a filename that does not
match raises rather than being silently mislabelled. A flipped label here would
poison every number downstream and would never surface as an error.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# subject _ imgno _ gender _ glasses _ state _ refl _ light _ sensor
_NAME_RE = re.compile(
    r"^(?P<subject>s\d+)"
    r"_(?P<imgno>\d+)"
    r"_(?P<gender>\d)"
    r"_(?P<glasses>\d)"
    r"_(?P<state>\d)"
    r"_(?P<refl>\d)"
    r"_(?P<light>\d)"
    r"_(?P<sensor>\d+)$"
)


SENSORS = {"01": "realsense", "02": "ids", "03": "aptina"}


class MRLFilenameError(ValueError):
    """A filename did not match the documented MRL convention."""


@dataclass(frozen=True)
class EyeSample:
    path: Path
    subject: str
    label: int        # 1 = open, 0 = closed
    glasses: bool
    lighting_good: bool
    reflections: int  # 0 none, 1 small, 2 big
    sensor: str       # "01" RealSense, "02" IDS, "03" Aptina

    @property
    def sensor_name(self) -> str:
        return SENSORS.get(self.sensor, f"sensor_{self.sensor}")

    @property
    def strata(self) -> Dict[str, str]:
        """Slices we report accuracy over separately (plan §6.5).

        Sensor is in here because the three cameras differ in resolution and
        noise, and a model that only works on one of them is not a model that
        survives contact with whatever camera ends up in the enclosure.
        """
        return {
            "glasses": "glasses" if self.glasses else "no_glasses",
            "lighting": "good_light" if self.lighting_good else "bad_light",
            "reflections": ("none", "small", "big")[min(self.reflections, 2)],
            "sensor": self.sensor_name,
        }


def parse_filename(path: Path) -> EyeSample:
    """Parse one MRL filename into a labelled sample. Raises on mismatch."""
    m = _NAME_RE.match(path.stem)
    if m is None:
        raise MRLFilenameError(
            f"{path.name!r} does not match the MRL naming convention "
            f"'<subject>_<imgno>_<gender>_<glasses>_<state>_<refl>_<light>_<sensor>'. "
            f"If your download uses a different layout, fix _NAME_RE in mrl.py "
            f"rather than relaxing it — a wrong label here is invisible later."
        )
    return EyeSample(
        path=path,
        subject=m["subject"],
        label=int(m["state"]),
        glasses=m["glasses"] == "1",
        lighting_good=m["light"] == "1",
        reflections=int(m["refl"]),
        sensor=m["sensor"],
    )


def scan(root: Path, limit: Optional[int] = None) -> List[EyeSample]:
    """Walk `root` for eye images and parse every filename.

    Raises if the directory is missing or yields nothing, with a message that
    says what to do about it.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"MRL dataset not found at {root}.\n"
            f"Download it, unpack it, and either place it there or set "
            f"AWAS_MRL_DIR to point at it.\n"
            f"To exercise the pipeline before the data arrives, run the "
            f"training script with --smoke."
        )

    paths = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if limit is not None:
        paths = paths[:limit]

    if not paths:
        raise FileNotFoundError(f"No images found under {root}.")

    return [parse_filename(p) for p in paths]


def split_by_subject(
    samples: Sequence[EyeSample],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[List[EyeSample], List[EyeSample], List[EyeSample]]:
    """Split BY SUBJECT, never by image.

    This is the single most important line in the file. MRL has ~37 people and
    hundreds of near-identical frames each. A random per-image split puts the
    same eyes in train and test and reports an accuracy that has nothing to do
    with performance on a driver the model has never seen — which is the only
    number that matters for a system that meets a new driver every time it is
    installed.
    """
    subjects = sorted({s.subject for s in samples})
    if len(subjects) < 3:
        raise ValueError(
            f"Only {len(subjects)} subject(s) found; a subject-wise split needs "
            f"at least 3. Check that the dataset unpacked completely."
        )

    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_frac))
    n_val = max(1, round(len(shuffled) * val_frac))

    test_subjects = set(shuffled[:n_test])
    val_subjects = set(shuffled[n_test:n_test + n_val])

    train, val, test = [], [], []
    for s in samples:
        if s.subject in test_subjects:
            test.append(s)
        elif s.subject in val_subjects:
            val.append(s)
        else:
            train.append(s)

    assert not ({s.subject for s in train} & {s.subject for s in test})
    assert not ({s.subject for s in train} & {s.subject for s in val})
    return train, val, test


def describe(samples: Sequence[EyeSample], name: str) -> str:
    """One-line summary for the training log."""
    if not samples:
        return f"{name}: empty"
    n = len(samples)
    open_n = sum(s.label for s in samples)
    subs = len({s.subject for s in samples})
    glasses = sum(s.glasses for s in samples)
    return (
        f"{name}: {n} images, {subs} subjects, "
        f"{open_n / n:.1%} open, {glasses / n:.1%} glasses"
    )
