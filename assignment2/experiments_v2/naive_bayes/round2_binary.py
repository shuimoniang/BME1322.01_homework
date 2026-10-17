"""Round 2: binary thresholding + alpha sweep for Bernoulli NB.

The classic MNIST result (a Bernoulli NB binarized at ~0.5 of max) is
around 84% with the naive per-pixel scheme on raw features, but a
well-chosen threshold plus small alpha typically pushes single-pixel
Bernoulli NB much higher than multi-level binning on raw gray values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit_categorical_nb, load_train, predict_categorical_nb, stratified_split


def binarize(X: np.ndarray, threshold: int) -> np.ndarray:
    return (X > threshold).astype(np.uint8)


def main() -> None:
    X, y = load_train()
    tr, va = stratified_split(y)
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]

    results = []
    for threshold in (16, 32, 48, 64, 96, 128, 160):
        states = binarize(X_tr, threshold)
        states_va = binarize(X_va, threshold)
        for alpha in (0.01, 0.1, 1.0, 8.0):
            log_prior, log_prob = fit_categorical_nb(states, y_tr, 2, alpha)
            preds = predict_categorical_nb(states_va, log_prior, log_prob)
            acc = float(np.mean(preds == y_va))
            results.append((threshold, alpha, acc))
            print(f"threshold={threshold:>3}  alpha={alpha:<5}  val_acc={acc:.4f}")

    best = max(results, key=lambda r: r[2])
    print(f"\nBEST: threshold={best[0]}, alpha={best[1]}, val_acc={best[2]:.4f}")


if __name__ == "__main__":
    main()
