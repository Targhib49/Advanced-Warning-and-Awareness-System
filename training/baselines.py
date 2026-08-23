"""Baselines the CNN has to beat before it earns its place on the device.

Plan v3 §3 states the rule: deep learning that does not beat something simpler
is a liability on an embedded target, not an achievement. Two baselines:

  majority   predict the commoner class — the floor any claim must clear
  logreg     logistic regression on downsampled raw pixels — the cheapest
             thing that can possibly work, and a surprisingly strong bar

Note what is NOT here: an EAR baseline. EAR needs facial landmarks, and MRL
ships pre-cropped eyes with no face around them. The EAR-versus-learned
comparison belongs downstream on full-face video (NTHU-DDD, DROZY), not here.
Saying so is better than fabricating a comparison this dataset cannot support.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def majority_baseline(y_train: Sequence[int], y_test: Sequence[int]) -> float:
    counts = np.bincount(np.asarray(y_train, dtype=int), minlength=2)
    pred = int(counts.argmax())
    return float((np.asarray(y_test) == pred).mean())


def logreg_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int = 200,
) -> Tuple[float, np.ndarray]:
    """Logistic regression on flattened pixels. Returns (accuracy, p_closed)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1)
    clf.fit(x_train.reshape(len(x_train), -1), y_train)
    xt = x_test.reshape(len(x_test), -1)
    acc = float(clf.score(xt, y_test))
    p_closed = clf.predict_proba(xt)[:, 0]
    return acc, p_closed


def stratified_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    strata: Sequence[Dict[str, str]],
) -> Dict[str, Dict[str, float]]:
    """Accuracy per slice (glasses, lighting, reflections).

    Aggregate accuracy hides exactly the failures that matter in a cabin:
    glasses and bad light are where a driver-monitoring system actually breaks
    (plan §6.5), so they are reported separately rather than averaged away.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict[str, Dict[str, float]] = {}
    keys = sorted({k for s in strata for k in s})
    for key in keys:
        values = sorted({s[key] for s in strata if key in s})
        out[key] = {}
        for v in values:
            mask = np.array([s.get(key) == v for s in strata])
            if mask.sum() == 0:
                continue
            out[key][v] = {
                "n": int(mask.sum()),
                "acc": float((y_true[mask] == y_pred[mask]).mean()),
            }
    return out
