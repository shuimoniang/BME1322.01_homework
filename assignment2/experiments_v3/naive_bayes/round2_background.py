"""Round 2 (v3, clean): background-aware patch factorization.

Split each patch feature z_j into two conditionally independent parts:
  - m_j: Bernoulli occupancy (patch has any ink), P(m_j|y)
  - q_j: pattern of the patch given occupied (16 binary-band states),
    P(q_j|y) fitted only on occupied patches with its own alpha.
Prediction: sum of background log-lik + occupancy deltas + pattern terms.
This removes the all-zero patch mass that flattens class differences and
lets each sub-model have its own smoothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from common import accuracy, load_train, stratified_split

D = 729
N_PAT = 16  # 2 bands ^ 4 pixels


def masks(Xa: np.ndarray) -> np.ndarray:
    n = Xa.shape[0]
    img = Xa.reshape(n, 28, 28)
    out = np.empty((n, D), dtype=bool)
    col = 0
    for r in range(27):
        for c in range(27):
            out[:, col] = img[:, r:r + 2, c:c + 2].any(axis=(1, 2))
            col += 1
    return out


def pattern_states(Xa: np.ndarray) -> np.ndarray:
    n = Xa.shape[0]
    img = Xa.reshape(n, 28, 28)
    bands = np.searchsorted(np.array([64]), img, side="right").astype(np.int64)
    states = np.empty((n, D), dtype=np.int64)
    col = 0
    for r in range(27):
        for c in range(27):
            states[:, col] = (
                (bands[:, r, c] * 2 + bands[:, r, c + 1]) * 2
                + bands[:, r + 1, c]
            ) * 2 + bands[:, r + 1, c + 1]
            col += 1
    return states


def fit(mask_tr, pat_tr, y_tr, alpha_bern, alpha_pat):
    class_count = np.bincount(y_tr, minlength=10).astype(np.float64)
    log_prior = np.log(class_count / class_count.sum())

    ones = np.zeros((10, D), dtype=np.float64)
    for c in range(10):
        ones[c] = mask_tr[y_tr == c].sum(axis=0)
    log_occ = np.log((ones + alpha_bern) / (class_count[:, None] + 2 * alpha_bern))
    log_non = np.log((class_count[:, None] - ones + alpha_bern)
                     / (class_count[:, None] + 2 * alpha_bern))

    # pattern counts: only occupied entries
    counts = np.zeros((D, N_PAT), dtype=np.float64)
    for c in range(10):
        counts_c = np.zeros((D, N_PAT), dtype=np.float64)
        rows = np.flatnonzero(y_tr == c)
        d_idx, s_idx = np.nonzero(mask_tr[rows])
        vals = pat_tr[rows][d_idx, s_idx]
        np.add.at(counts_c, (s_idx, vals), 1.0)
        # per (class, patch) normalization; empty patches -> uniform
        tot = counts_c.sum(axis=1, keepdims=True)
        lp = np.log(counts_c + alpha_pat) - np.log(tot + N_PAT * alpha_pat)
        lp[tot.ravel() == 0] = -np.log(N_PAT)
        # store into log_pat[c] as (D, N_PAT)
        if c == 0:
            log_pat = np.empty((10, D, N_PAT), dtype=np.float64)
        log_pat[c] = lp
    return log_prior, log_occ, log_non, log_pat


def predict(mask_va, pat_va, params, alpha_bern, alpha_pat):
    log_prior, log_occ, log_non, log_pat = params
    n = mask_va.shape[0]
    scores = np.tile(log_prior, (n, 1))
    scores += log_non.sum(axis=1)[None, :]
    d_idx, s_idx = np.nonzero(mask_va)
    vals = pat_va[d_idx, s_idx]
    delta_occ = (log_occ[:, s_idx] - log_non[:, s_idx]).T  # (nnz, 10)
    pat = log_pat[:, s_idx, vals].T  # (nnz, 10)
    scores[d_idx] += delta_occ + pat
    return np.argmax(scores, axis=1)


def main() -> None:
    X, y = load_train()
    tr, va = stratified_split(y)
    X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]

    mask_tr, mask_va = masks(X_tr), masks(X_va)
    pat_tr, pat_va = pattern_states(X_tr), pattern_states(X_va)

    for alpha_bern in (0.5, 1.0):
        for alpha_pat in (0.05, 0.2, 0.5, 1.0):
            params = fit(mask_tr, pat_tr, y_tr, alpha_bern, alpha_pat)
            preds = predict(mask_va, pat_va, params, alpha_bern, alpha_pat)
            acc = accuracy(y_va, preds)
            print(f"[bg NB bern={alpha_bern} pat={alpha_pat}] "
                  f"val_acc={acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
