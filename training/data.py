"""Torch dataset over parsed MRL samples, plus a synthetic smoke-test set.

`mrl.py` stays torch-free on purpose so the filename parsing and the
subject-wise split can be unit-tested without importing torch.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .mrl import EyeSample


class EyeCropDataset(Dataset):
    """Grayscale eye crops resized to `crop_px`, scaled to [0, 1].

    Decoded images are cached in memory as uint8. Without this, 30 epochs over
    ~70k images means ~2 million PNG opens and decodes, and the run is bound by
    the filesystem rather than by arithmetic. The whole cache is
    `n * crop_px^2` bytes — about 70 MB for the full MRL set at 32x32, which is
    nothing next to spending an afternoon waiting.
    """

    def __init__(
        self,
        samples: Sequence[EyeSample],
        crop_px: int = 32,
        train: bool = False,
        shift_px: int = 2,
        brightness: float = 0.15,
        seed: int = 0,
        cache: bool = True,
        verbose: bool = False,
    ):
        self.samples = list(samples)
        self.crop_px = crop_px
        self.train = train
        self.shift_px = shift_px
        self.brightness = brightness
        self._rng = random.Random(seed)
        self._cache = self._preload(verbose) if cache else None

    def _preload(self, verbose: bool) -> np.ndarray:
        n, p = len(self.samples), self.crop_px
        out = np.empty((n, p, p), dtype=np.uint8)
        for i, s in enumerate(self.samples):
            img = Image.open(s.path).convert("L").resize((p, p), Image.BILINEAR)
            out[i] = np.asarray(img, dtype=np.uint8)
            if verbose and n > 2000 and i % 10000 == 0 and i:
                print(f"    cached {i}/{n}", flush=True)
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        # Light augmentation only. These crops are already tightly framed;
        # aggressive geometric warping destroys the eyelid signal itself.
        if self.shift_px > 0:
            dx = self._rng.randint(-self.shift_px, self.shift_px)
            dy = self._rng.randint(-self.shift_px, self.shift_px)
            arr = np.roll(arr, (dy, dx), axis=(0, 1))
        if self.brightness > 0:
            f = 1.0 + self._rng.uniform(-self.brightness, self.brightness)
            arr = np.clip(arr * f, 0.0, 1.0)
        if self._rng.random() < 0.5:
            arr = arr[:, ::-1].copy()  # left/right eye symmetry
        return arr

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[i]
        if self._cache is not None:
            arr = self._cache[i].astype(np.float32) / 255.0
        else:
            img = Image.open(s.path).convert("L").resize(
                (self.crop_px, self.crop_px), Image.BILINEAR
            )
            arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.train:
            arr = self._augment(arr)
        return torch.from_numpy(arr).unsqueeze(0), s.label


class SyntheticEyeDataset(Dataset):
    """Fake eyes, so the whole pipeline can be verified before the data lands.

    Not a benchmark and not a substitute for MRL — it exists so that training,
    checkpointing, ONNX export and quantization can all be exercised in week 1
    while the real download is still in progress. A closed eye is drawn as a
    dark horizontal band; an open one as a bright iris blob.
    """

    def __init__(self, n: int = 2000, crop_px: int = 32, seed: int = 0):
        self.n = n
        self.crop_px = crop_px
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        g = np.random.default_rng(i)
        p = self.crop_px
        label = int(g.integers(0, 2))
        arr = g.normal(0.45, 0.08, size=(p, p)).astype(np.float32)
        mid = p // 2
        if label == 0:
            arr[mid - 2:mid + 2, :] -= 0.28           # lid line
        else:
            yy, xx = np.mgrid[0:p, 0:p]
            disc = ((yy - mid) ** 2 + (xx - mid) ** 2) < (p * 0.18) ** 2
            arr[disc] += 0.32                          # iris highlight
        return torch.from_numpy(np.clip(arr, 0, 1)).unsqueeze(0), label
