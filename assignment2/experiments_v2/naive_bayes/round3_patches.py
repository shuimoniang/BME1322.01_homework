"""Round 3: 2x2 patch gray-level pattern features.

Classic result (Kaykvilmaz & Aykac etc.): Bernoulli NB on 2x2 blocks of
MNIST reaches ~95%.  Each overlapping 2x2 patch is quantized to a
gray-level pattern id, then a categorical NB is fit over the 27x27=729
patch features.

Variants tested:
  A. 2-level binarization of the patch (any ink > threshold) -> 16 patterns
  B. 4-level gray quantization per pixel (0, low, mid, high) -> 256 patterns
  C. non-overlapping 2x2 blocks (14x14=196 features), same encodings
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import fit_categorical_nb, load_train, predict_categorical_nb, stratified_split


def patch_ids(X: np.ndarray, stride: int = 1, size: int = 2) -> np.ndarray:
    """Extract sliding (or strided) patches and encode each as an integer id.

    X: (n, 784) uint8.  Returns (n, n_patches) uint16.
    """
    n = X.shape[0]
    img = X.reshape(n, 28, 28)
    positions = [(r, c) for r in range(0, 28 - size + 1, stride)
                 for c in range(0, 28 - size + 1, stride)]
    out = np.empty((n, len(positions)), dtype=np.int64)
    for i, (r, c) in enumerate(positions):
        block = img[:, r:r + size, c:c + size]
        # encode 4 pixels, each 2 bits after binarization at 128
        out[:, i] = ((block[:, 0, 0] > 128).astype(np.int64) << 3 |
                     (block[:, 0, 1] > 128).astype(np.int64) << 2 |
                     (block[:, 1, 0] > 128).astype(np.int64) << 1 |
                     (block[:, 1, 1] > 128).astype(np.int64))
    return out


def patch_ids_gray(X: np.ndarray, stride: int = 1, size: int = 2,
                   levels: int = 4) -> np.ndarray:
    """Gray-level patch encoding: each pixel quantized to `levels` states."""
    n = X.shape[0]
    img = X.reshape(n, 28, 28)
    positions = [(r, c) for r in range(0, 28 - size + 1, stride)
                 for c in range(0, 28 - size + 1, stride)]
    out = np.empty((n, len(positions)), dtype=np.int64)
    # thresholds dividing 0..255 into `levels` bands
    edges = [int(255 * k / levels) for k in range(1, levels)]
    for i, (r, c) in enumerate(positions):
        block = img[:, r:r + size, c:c + size].reshape(n, 4)
        q = np.searchsorted(np.array(edges), block, side="right")
        id_ = q[:, 0] * levels ** 3 + q[:, 1] * levels ** 2 + q[:, 2] * levels + q[:, 3]
        out[:, i] = id_
    return out


def run(name, X, y, states_tr, states_va, y_tr, y_va, alpha):
    log_prior, log_prob = fit_categorical_nb(states_tr, y_tr, states_tr.max() + 1, alpha)
    preds = predict_categorical_nb(states_va, log_prior, log_prob)
    acc = float(np.mean(preds == y_va))
    print(f"{name:<42} alpha={alpha:<6} val_acc={acc:.4f}", flush=True)
    return acc


def main() -> None:
    X, y = load_train()
    tr, va = stratified_split(y)
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]

    best = None
    # binary overlapping patches
    for stride in (1, 2):
        s_tr = patch_ids(X_tr, stride)
        s_va = patch_ids(X_va, stride)
        for alpha in (0.1, 1.0, 4.0):
            acc = run(f"2x2 binary stride={stride}", X, y, s_tr, s_va, y_tr, y_va, alpha)
            if best is None or acc > best[0]:
                best = (acc, f"2x2 binary stride={stride} alpha={alpha}")

    # gray overlapping patches
    for levels in (3, 4):
        s_tr = patch_ids_gray(X_tr, 1, levels=levels)
        s_va = patch_ids_gray(X_va, 1, levels=levels)
        for alpha in (1.0, 4.0):
            acc = run(f"2x2 gray L={levels} stride=1", X, y, s_tr, s_va, y_tr, y_va, alpha)
            if acc > best[0]:
                best = (acc, f"2x2 gray L={levels} stride=1 alpha={alpha}")

    print(f"\nBEST: {best[1]}  val_acc={best[0]:.4f}")


if __name__ == "__main__":
    main()
