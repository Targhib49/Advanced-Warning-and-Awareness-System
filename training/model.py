"""T1 — the learned eye-state classifier (plan v3 §3, tier T1).

Deliberately tiny. This replaces the EAR threshold, which is a 2D geometric
ratio that breaks under head yaw/pitch and varies two- to threefold across
individuals. A classifier over pixels learns eyelid appearance directly.

The useful output is not the argmax but `P(closed)` as a CONTINUOUS value:
PERCLOS asks what fraction of a window the eyes were more than 80% closed, and
a calibrated probability is a better soft closure measure than a hard flip.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EyeStateNet(nn.Module):
    """~24k parameters over a 32x32 grayscale eye crop.

    Class 0 = closed, class 1 = open (matching the MRL `state` field).
    """

    def __init__(self, in_px: int = 32, width: int = 16):
        super().__init__()
        if in_px % 8 != 0:
            raise ValueError(f"in_px must be divisible by 8, got {in_px}")

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, width),           # 32 -> 16
            block(width, width * 2),   # 16 -> 8
            block(width * 2, width * 4),  # 8 -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width * 4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    @torch.no_grad()
    def p_closed(self, x: torch.Tensor) -> torch.Tensor:
        """Probability the eye is closed — the value PERCLOS actually wants."""
        return self.forward(x).softmax(dim=1)[:, 0]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
