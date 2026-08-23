"""Training configuration — paths from the environment so one script runs
unchanged locally and on Kaggle/Colab (plan v3 §11, Phase 1).

    local    AWAS_DATA_ROOT=./data          (default)
    Kaggle   AWAS_DATA_ROOT=/kaggle/input   AWAS_OUT_DIR=/kaggle/working
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


DATA_ROOT: Path = _env_path("AWAS_DATA_ROOT", "./data")
OUT_DIR: Path = _env_path("AWAS_OUT_DIR", "./artifacts")

# Where the MRL Eye Dataset was unpacked. Override with AWAS_MRL_DIR when the
# hosting platform puts it somewhere else (Kaggle input directories are named
# after the dataset slug).
MRL_DIR: Path = _env_path("AWAS_MRL_DIR", str(DATA_ROOT / "mrlEyes_2018_01"))


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for the T1 eye-state classifier."""

    crop_px: int = 32          # model input is crop_px x crop_px, grayscale
    batch_size: int = 256
    epochs: int = 30
    lr: float = 3e-3
    weight_decay: float = 1e-4
    seed: int = 1337

    # Subject-wise split (see mrl.py). Fractions of the SUBJECT pool, not of
    # the image pool.
    val_frac: float = 0.15
    test_frac: float = 0.20

    # Light augmentation only. These crops are already tightly framed; heavy
    # geometric augmentation destroys the eyelid signal we are trying to learn.
    aug_shift_px: int = 2
    aug_brightness: float = 0.15

    # "cpu" | "mps" | "cuda" | "auto". For a model this small, MPS transfer
    # overhead can exceed the speedup — time both before committing.
    device: str = "auto"

    stratify_by: tuple = field(default=("glasses", "lighting"))
