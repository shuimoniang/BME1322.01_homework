"""Round 4: refine the patch view and combine multiple views.

Refinements:
  1. gray levels 5 and 6 for the 2x2 patch encoding
  2. multi-view combination: patch view + pixel view + 7x7 block view
     (all views are separate NB feature groups; the joint log-likelihood
     is the sum of each group's log-likelihood plus a shared prior --
     still a generative NB with conditionally independent features)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import bincount_2d, load_train, stratified_split
from round3_patches import patch_ids, patch_ids_gray


def fit_group(states: np.ndarray, y: np.ndarray, n_states: int, alpha: float):
    """Return (class_counts, log_prob) for one feature group."""
    class_count = np.bincount(y, minlength=10).astype(np.float64)
    log_prob = np.empty((10, states.shape[1], n_states), dtype=np.float64)
    for c in range(10):
        counts = bincount_2d(states[y == c], n_states)
        log_prob[c] = np.log((counts + alpha) / (class_count[c] + n_states * alpha))
    return class_count, log_prob


def group_scores(states: np.ndarray, log_prob: np.ndarray) -> np.ndarray:
    """(n, 10) summed log-likelihood for one group."""
    n = states.shape[0]
    d, _ = log_prob.shape[1], log_prob.shape[2]
    out = np.empty((n, 10), dtype=np.float64)
    for start in range(0, n, 2048):
        stop = min(start + 2048, n)
        batch = states[start:stop].astype(np.int64)
        lp = log_prob[:, np.arange(d)[None, :], batch].transpose(1, 0, 2)
        out[start:stop] = lp.sum(axis=2)
    return out


def block_view(X: np.ndarray, block: int = 7, levels: int = 4) -> np.ndarray:
    """Aggregate block-mean intensity quantized to `levels` states."""
    n = X.shape[0]
    img = X.reshape(n, 28, 28).astype(np.float32) / 255.0
    edges = np.linspace(0, 1, levels + 1)[1:-1]
    states = []
    for r in range(0, 28, block):
        for c in range(0, 28, block):
            mean = img[:, r:r + block, c:c + block].mean(axis=(1, 2))
            states.append(np.searchsorted(edges, mean, side="right"))
    return np.stack(states, axis=1).astype(np.int64)


def pixel_view(X: np.ndarray, levels: int = 4) -> np.ndarray:
    edges = [int(255 * k / levels) for k in range(1, levels)]
    return np.searchsorted(np.array(edges), X, side="right").astype(np.int64)


def score_single(states_tr, states_va, y_tr, y_va, n_states, alpha, name):
    _, log_prob = fit_group(states_tr, y_tr, n_states, alpha)
    sc = group_scores(states_va, log_prob)
    acc = float(np.mean(np.argmax(sc, axis=1) == y_va))
    print(f"{name:<44} val_acc={acc:.4f}", flush=True)
    return acc


def score_multi(views, y_tr, y_va, name):
    """views: list of (states_tr, states_va, n_states, alpha)."""
    log_prior = None
    total = None
    for s_tr, s_va, n_states, alpha in views:
        class_count, log_prob = fit_group(s_tr, y_tr, n_states, alpha)
        if log_prior is None:
            log_prior = np.log(class_count / class_count.sum())
        sc = group_scores(s_va, log_prob)
        total = sc if total is None else total + sc
    preds = np.argmax(total + log_prior, axis=1)
    acc = float(np.mean(preds == y_va))
    print(f"{name:<44} val_acc={acc:.4f}", flush=True)
    return acc


def main() -> None:
    X, y = load_train()
    tr, va = stratified_split(y)
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]

    # 1. gray level refinement, single patch view
    for levels in (5, 6):
        s_tr = patch_ids_gray(X_tr, 1, levels=levels)
        s_va = patch_ids_gray(X_va, 1, levels=levels)
        for alpha in (0.5, 1.0, 2.0):
            score_single(s_tr, s_va, y_tr, y_va, levels ** 4, alpha,
                         f"patch gray L={levels} alpha={alpha}")

    # 2. multi-view combinations
    p4_tr, p4_va = patch_ids_gray(X_tr, 1, levels=4), patch_ids_gray(X_va, 1, levels=4)
    p5_tr, p5_va = patch_ids_gray(X_tr, 1, levels=5), patch_ids_gray(X_va, 1, levels=5)
    pb_tr, pb_va = patch_ids(X_tr, 1), patch_ids(X_va, 1)
    pix_tr, pix_va = pixel_view(X_tr), pixel_view(X_va)
    blk_tr, blk_va = block_view(X_tr), block_view(X_va)

    score_multi([(p4_tr, p4_va, 256, 1.0), (pb_tr, pb_va, 16, 1.0)],
                y_tr, y_va, "patch L4 + patch binary")
    score_multi([(p4_tr, p4_va, 256, 1.0), (pix_tr, pix_va, 16, 0.1)],
                y_tr, y_va, "patch L4 + pixels L4")
    score_multi([(p4_tr, p4_va, 256, 1.0), (blk_tr, blk_va, 16, 0.5)],
                y_tr, y_va, "patch L4 + 7x7 blocks")
    score_multi([(p5_tr, p5_va, 625, 1.0), (pb_tr, pb_va, 16, 1.0)],
                y_tr, y_va, "patch L5 + patch binary")
    score_multi([(p5_tr, p5_va, 625, 1.0), (pix_tr, pix_va, 16, 0.1)],
                y_tr, y_va, "patch L5 + pixels L4")
    score_multi([(p4_tr, p4_va, 256, 1.0), (p5_tr, p5_va, 625, 1.0)],
                y_tr, y_va, "patch L4 + patch L5")


if __name__ == "__main__":
    main()
