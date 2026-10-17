"""Round 5 (fixed): TAN on 2x2 patch features, memory-safe implementation.

Structure learning:
  - Compute pairwise MI between the 729 patch features using their
    BINARY states (16 states) via chunked one-hot Gram matrices on a
    20k subsample (fast, low memory).
  - Maximum spanning tree (Prim) over the MI matrix.

Parameter estimation:
  - TAN CPTs P(x_i | y, x_parent) on the GRAY L5 patch states (625),
    with per-parent-state fallback to the marginal when the parent
    state has fewer than `min_parent_count` training samples.
  - Also a full-binary TAN variant (16 states) for comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_train, stratified_split
from round3_patches import patch_ids, patch_ids_gray


def pairwise_mi(states: np.ndarray, n_states: int, n_sub: int = 20000,
                chunk: int = 128, seed: int = 0) -> np.ndarray:
    """Pairwise MI between columns, via chunked one-hot Gram matrices."""
    rng = np.random.default_rng(seed)
    n, d = states.shape
    idx = rng.permutation(n)[:n_sub] if n > n_sub else np.arange(n)
    S = states[idx].astype(np.int64)
    onehot_rows = np.arange(n_sub)[:, None]
    mi = np.zeros((d, d), dtype=np.float64)
    for a0 in range(0, d, chunk):
        a1 = min(a0 + chunk, d)
        wa = a1 - a0
        Za = np.zeros((n_sub, wa * n_states), dtype=np.float32)
        for k in range(wa):
            Za[onehot_rows, S[:, a0 + k] + k * n_states] = 1.0
        for b0 in range(a0, d, chunk):
            b1 = min(b0 + chunk, d)
            wb = b1 - b0
            Zb = np.zeros((n_sub, wb * n_states), dtype=np.float32)
            for k in range(wb):
                Zb[onehot_rows, S[:, b0 + k] + k * n_states] = 1.0
            joint = (Za.T @ Zb).reshape(wa, n_states, wb, n_states) / n_sub
            pa = joint.sum(axis=(1, 3), keepdims=True)  # (wa,1,wb,1)? no
            pa = joint.sum(axis=3, keepdims=True).sum(axis=1, keepdims=True)  # (wa,1,1,1)
            pb = joint.sum(axis=(0, 2), keepdims=True)  # (1,1,wb,1)
            denom = pa * pb
            with np.errstate(divide="ignore", invalid="ignore"):
                contrib = joint * np.log(joint / denom)
            contrib[~np.isfinite(contrib)] = 0.0
            block = contrib.sum(axis=(1, 3))
            ii, jj = np.meshgrid(np.arange(a0, a1), np.arange(b0, b1), indexing="ij")
            sel = ii != jj
            mi[ii[sel], jj[sel]] = block[sel]
    mi = np.maximum(mi, mi.T)
    np.fill_diagonal(mi, 0.0)
    return mi


def max_spanning_tree(mi: np.ndarray) -> np.ndarray:
    """Prim's algorithm; returns parent array (root 0 has parent -1)."""
    d = mi.shape[0]
    parent = np.full(d, -1, dtype=np.int64)
    in_tree = np.zeros(d, dtype=bool)
    in_tree[0] = True
    best_weight = mi[0].copy()
    best_parent = np.zeros(d, dtype=np.int64)
    for _ in range(d - 1):
        cand = np.flatnonzero(~in_tree)
        j = cand[np.argmax(best_weight[cand])]
        parent[j] = best_parent[j]
        in_tree[j] = True
        update = (~in_tree) & (mi[j] > best_weight)
        best_weight[update] = mi[j][update]
        best_parent[update] = j
    return parent


def marginal_log_prob(states: np.ndarray, y: np.ndarray, n_states: int, alpha: float):
    class_count = np.bincount(y, minlength=10).astype(np.float64)
    d = states.shape[1]
    log_prob = np.empty((10, d, n_states), dtype=np.float64)
    for c in range(10):
        sub = states[y == c].astype(np.int64)
        offsets = sub + np.arange(d) * n_states
        counts = np.bincount(offsets.ravel(), minlength=d * n_states).reshape(d, n_states)
        log_prob[c] = np.log((counts + alpha) / (class_count[c] + n_states * alpha))
    return class_count, log_prob


def tan_scores(states_tr, y_tr, states_va, n_states, parent, alpha, min_parent_count):
    """Memory-safe TAN scoring: accumulate per feature, per class."""
    class_count, marg = marginal_log_prob(states_tr, y_tr, n_states, alpha)
    log_prior = np.log(class_count / class_count.sum())
    n_va = states_va.shape[0]
    scores = np.repeat(log_prior[None, :], n_va, axis=0)
    va = states_va.astype(np.int64)
    d = states_tr.shape[1]
    for c in range(10):
        sub = states_tr[y_tr == c].astype(np.int64)
        col = np.zeros(n_va, dtype=np.float64)
        for i in range(d):
            p = int(parent[i])
            if p < 0:
                col += marg[c, i][va[:, i]]
                continue
            counts = np.bincount(sub[:, p] * n_states + sub[:, i],
                                 minlength=n_states ** 2).reshape(n_states, n_states)
            row_totals = counts.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                log_cpt = np.log((counts + alpha) / (row_totals + n_states * alpha))
            log_cpt[~np.isfinite(log_cpt)] = -np.log(n_states)
            # fallback rows: parent states with too few samples
            sparse_rows = np.flatnonzero(row_totals[:, 0] < min_parent_count)
            if sparse_rows.size:
                log_cpt[sparse_rows] = marg[c, i][sparse_rows, :]
            col += log_cpt[va[:, p], va[:, i]]
        scores[:, c] += col
    return scores


def main() -> None:
    X, y = load_train()
    tr, va = stratified_split(y)
    X_tr, X_va = X[tr], X[va]
    y_tr, y_va = y[tr], y[va]

    b_tr, b_va = patch_ids(X_tr, 1), patch_ids(X_va, 1)
    g_tr, g_va = patch_ids_gray(X_tr, 1, levels=5), patch_ids_gray(X_va, 1, levels=5)

    # reference baselines
    cc, marg = marginal_log_prob(g_tr, y_tr, 625, alpha=0.5)
    d = g_va.shape[1]
    gv = g_va.astype(np.int64)
    sc = np.zeros((gv.shape[0], 10))
    for c in range(10):
        sc[:, c] = np.log(cc[c]) + marg[c][np.arange(d)[None, :], gv].sum(axis=1)
    print(f"[marginal patch L5 alpha=0.5] val_acc={np.mean(np.argmax(sc, 1) == y_va):.4f}",
          flush=True)

    # tree from binary-patch MI (cached to disk; MI takes ~35 min)
    cache = Path(__file__).resolve().parent / "mi_cache.npy"
    if cache.is_file():
        mi = np.load(cache)
        print("loaded cached MI matrix.", flush=True)
    else:
        print("computing pairwise MI (binary patches, 20k subsample)...", flush=True)
        mi = pairwise_mi(b_tr, 16)
        np.save(cache, mi)
        print("MI done, cached.", flush=True)
    parent = max_spanning_tree(mi)
    print("tree done.", flush=True)

    for alpha in (0.1, 0.5):
        for min_pc in (50, 200):
            scores = tan_scores(b_tr, y_tr, b_va, 16, parent, alpha, min_pc)
            acc = float(np.mean(np.argmax(scores, axis=1) == y_va))
            print(f"TAN binary patches alpha={alpha} min_pc={min_pc:<4} "
                  f"val_acc={acc:.4f}", flush=True)

    for alpha in (0.1, 0.5):
        for min_pc in (50, 200):
            scores = tan_scores(g_tr, y_tr, g_va, 625, parent, alpha, min_pc)
            acc = float(np.mean(np.argmax(scores, axis=1) == y_va))
            print(f"TAN gray L5 patches  alpha={alpha} min_pc={min_pc:<4} "
                  f"val_acc={acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
