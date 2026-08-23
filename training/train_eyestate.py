"""Train the T1 eye-state classifier.

Runs unchanged locally and on Kaggle/Colab — paths come from the environment
(see config.py). Nothing here writes outside OUT_DIR.

    python -m training.train_eyestate --smoke          # no dataset needed
    python -m training.train_eyestate                  # real MRL run
    python -m training.train_eyestate --epochs 40 --device cpu

`--smoke` trains on synthetic eyes so the whole pipeline — training,
checkpointing, metrics, export — can be verified in week 1 while the real
download is still in progress.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import mrl
from .baselines import logreg_baseline, majority_baseline, stratified_accuracy
from .config import MRL_DIR, OUT_DIR, TrainConfig
from .data import EyeCropDataset, SyntheticEyeDataset
from .model import EyeStateNet


# ---------------- reproducibility ----------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------- eval ----------------

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds: List[int] = []
    probs: List[float] = []
    labels: List[int] = []
    for xb, yb in loader:
        logits = model(xb.to(device))
        p = logits.softmax(dim=1)
        preds.extend(p.argmax(dim=1).cpu().tolist())
        probs.extend(p[:, 0].cpu().tolist())   # P(closed)
        labels.extend(yb.tolist())
    return np.array(labels), np.array(preds), np.array(probs)


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float((y == p).mean())


# ---------------- training ----------------

def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict]]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.CrossEntropyLoss()

    history: List[Dict] = []
    best_acc, best_state = -1.0, None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            seen += len(yb)
        sched.step()

        y, p, _ = predict(model, val_loader, device)
        val_acc = accuracy(y, p)
        history.append({
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "val_acc": val_acc,
            "sec": round(time.time() - t0, 2),
        })
        print(f"  epoch {epoch:3d}  loss {total / max(seen, 1):.4f}  "
              f"val_acc {val_acc:.4f}  ({time.time() - t0:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------- pixel matrices for the baselines ----------------

def to_matrix(ds: Dataset, stride: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize a dataset as (X, y), downsampled, for the sklearn baselines."""
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x.numpy()[0, ::stride, ::stride])
        ys.append(y)
    return np.stack(xs), np.array(ys)


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train the T1 eye-state classifier.")
    ap.add_argument("--smoke", action="store_true",
                    help="train on synthetic eyes — verifies the pipeline without the dataset")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, help="cpu | mps | cuda | auto")
    ap.add_argument("--limit", type=int, default=None, help="cap images, for a quick pass")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg = TrainConfig(**{**asdict(cfg), "epochs": args.epochs})
    if args.device is not None:
        cfg = TrainConfig(**{**asdict(cfg), "device": args.device})

    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device}   out: {out_dir}")

    # ---- data ----
    strata: List[Dict[str, str]] = []
    if args.smoke:
        print("SMOKE MODE — synthetic eyes. Not a benchmark; pipeline check only.")
        n = args.limit or 3000
        train_ds = SyntheticEyeDataset(n=n, crop_px=cfg.crop_px, seed=1)
        val_ds = SyntheticEyeDataset(n=n // 5, crop_px=cfg.crop_px, seed=2)
        test_ds = SyntheticEyeDataset(n=n // 5, crop_px=cfg.crop_px, seed=3)
        cfg = TrainConfig(**{**asdict(cfg), "epochs": min(cfg.epochs, 5)})
    else:
        samples = mrl.scan(MRL_DIR, limit=args.limit)
        tr, va, te = mrl.split_by_subject(
            samples, cfg.val_frac, cfg.test_frac, cfg.seed
        )
        for part, nm in ((tr, "train"), (va, "val"), (te, "test")):
            print("  " + mrl.describe(part, nm))
        strata = [s.strata for s in te]
        print("  caching decoded crops in memory…")
        train_ds = EyeCropDataset(tr, cfg.crop_px, train=True,
                                  shift_px=cfg.aug_shift_px,
                                  brightness=cfg.aug_brightness, seed=cfg.seed,
                                  verbose=True)
        val_ds = EyeCropDataset(va, cfg.crop_px, verbose=True)
        test_ds = EyeCropDataset(te, cfg.crop_px, verbose=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    # ---- baselines first, so the CNN has a bar to clear ----
    print("\nbaselines")
    xb_tr, yb_tr = to_matrix(train_ds)
    xb_te, yb_te = to_matrix(test_ds)
    maj = majority_baseline(yb_tr, yb_te)
    print(f"  majority     {maj:.4f}")
    lr_acc, _ = logreg_baseline(xb_tr, yb_tr, xb_te, yb_te)
    print(f"  logreg       {lr_acc:.4f}")

    # ---- train ----
    model = EyeStateNet(in_px=cfg.crop_px).to(device)
    print(f"\ntraining EyeStateNet ({model.n_params():,} params)")
    model, history = train_one(model, train_loader, val_loader, cfg, device)

    # ---- test ----
    y, p, _ = predict(model, test_loader, device)
    test_acc = accuracy(y, p)
    print(f"\ntest accuracy  cnn {test_acc:.4f}   logreg {lr_acc:.4f}   majority {maj:.4f}")

    verdict = "CNN wins" if test_acc > lr_acc + 0.005 else "CNN does NOT beat logreg — ship the baseline"
    print(f"verdict: {verdict}")

    strat = {}
    if strata:
        strat = stratified_accuracy(y, p, strata)
        print("\nby slice")
        for key, vals in strat.items():
            for v, d in vals.items():
                print(f"  {key:12} {v:12} n={d['n']:6d}  acc {d['acc']:.4f}")

    # ---- save ----
    ckpt = out_dir / "eyestate.pt"
    torch.save({"state_dict": model.state_dict(),
                "crop_px": cfg.crop_px,
                "classes": ["closed", "open"]}, ckpt)

    metrics = {
        "config": asdict(cfg),
        "device": str(device),
        "smoke": bool(args.smoke),
        "n_params": model.n_params(),
        "test_acc_cnn": test_acc,
        "test_acc_logreg": lr_acc,
        "test_acc_majority": maj,
        "verdict": verdict,
        "by_slice": strat,
        "history": history,
    }
    (out_dir / "eyestate_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved {ckpt}")
    print(f"saved {out_dir / 'eyestate_metrics.json'}")


if __name__ == "__main__":
    main()
