"""Shared helpers for the bayes_net iteration experiments.

Provides data loading (reusing main.py), the standard 49,999/10,001
stratified split, generic CPT fitting for a hand-designed DAG, and
evaluation utilities.  Continuous feature statistics are cached as .npy
so each experiment round only pays for discretization + fitting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from main import find_data_dir, load_mnist, stratified_split  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def load_split():
    """Return (X_tr, y_tr, X_val, y_val) using the course-standard split."""
    data_dir = find_data_dir()
    X_train, y_train, _X_test, _y_test = load_mnist(data_dir)
    tr_idx, va_idx = stratified_split(y_train)
    return X_train[tr_idx], y_train[tr_idx], X_train[va_idx], y_train[va_idx]


def continuous_stats(X: np.ndarray, tag: str) -> dict[str, np.ndarray]:
    """Compute a dictionary of continuous image statistics, cached on disk."""
    cached = CACHE_DIR / f"stats_{tag}.npz"
    if cached.is_file():
        with np.load(cached) as data:
            return {k: data[k] for k in data.files}

    imgs = X.reshape(-1, 28, 28).astype(np.float64)
    norm = imgs / 255.0
    binary = imgs > 64
    padded = np.pad(binary, ((0, 0), (1, 1), (1, 1)))
    ul = padded[:, :-1, :-1].astype(np.uint8)
    ll = padded[:, 1:, :-1].astype(np.uint8)
    ur = padded[:, :-1, 1:].astype(np.uint8)
    lr = padded[:, 1:, 1:].astype(np.uint8)
    block_sum = ul + ll + ur + lr
    diag_pairs = ((ul & lr & ~ll & ~ur).astype(np.uint8)) | (
        (ll & ur & ~ul & ~lr).astype(np.uint8)
    )
    q1 = np.count_nonzero(block_sum == 1, axis=(1, 2))
    q3 = np.count_nonzero(block_sum == 3, axis=(1, 2))
    qd = np.count_nonzero(diag_pairs, axis=(1, 2))
    euler = np.rint((q1 - q3 + 2 * qd) / 4.0)

    rows = np.any(binary, axis=2)
    cols = np.any(binary, axis=1)
    row_idx = np.arange(28)[None, :]
    col_idx = np.arange(28)[None, :]
    row_any = np.any(binary, axis=2)  # (n, 28) ink present per row
    col_any = np.any(binary, axis=1)  # (n, 28) ink present per col
    first_row = np.where(row_any, row_idx * row_any, 0).max(axis=1)
    last_row = np.where(row_any, (27 - row_idx) * row_any, 0).max(axis=1)
    first_col = np.where(col_any, col_idx * col_any, 0).max(axis=1)
    last_col = np.where(col_any, (27 - col_idx) * col_any, 0).max(axis=1)
    height = np.clip(last_row - first_row + 1, 1, 28)
    width = np.clip(last_col - first_col + 1, 1, 28)

    top_density = norm[:, :14, :].mean(axis=(1, 2))
    bottom_density = norm[:, 14:, :].mean(axis=(1, 2))
    left_density = norm[:, :, :14].mean(axis=(1, 2))
    right_density = norm[:, :, 14:].mean(axis=(1, 2))
    center_density = norm[:, 7:21, 7:21].mean(axis=(1, 2))
    ring_density = norm.mean(axis=(1, 2)) - center_density

    total = norm.sum(axis=(1, 2))
    safe_total = np.where(total > 0, total, 1.0)
    row_coord = np.arange(28)[None, :, None]
    col_coord = np.arange(28)[None, None, :]
    centroid_row = (norm * row_coord).sum(axis=(1, 2)) / safe_total
    centroid_col = (norm * col_coord).sum(axis=(1, 2)) / safe_total

    hsym = np.abs(norm - norm[:, :, ::-1]).mean(axis=(1, 2))
    vsym = np.abs(norm - norm[:, ::-1, :]).mean(axis=(1, 2))
    rot_sym = np.abs(norm - norm[:, ::-1, ::-1]).mean(axis=(1, 2))

    run_starts = np.count_nonzero(binary[:, 0, :], axis=1)
    run_starts += np.count_nonzero((~binary[:, :-1, :]) & binary[:, 1:, :], axis=(1, 2))
    occupied_cols = np.maximum(np.count_nonzero(cols, axis=1), 1)
    vert_runs = run_starts / occupied_cols

    horiz_runs = np.count_nonzero(binary[:, :, 0], axis=1)
    horiz_runs += np.count_nonzero((~binary[:, :, :-1]) & binary[:, :, 1:], axis=(1, 2))
    occupied_rows = np.maximum(np.count_nonzero(rows, axis=1), 1)
    horiz_runs = horiz_runs / occupied_rows

    # gradient-orientation stroke statistics on the grayscale image
    gx = np.zeros_like(norm)
    gy = np.zeros_like(norm)
    gx[:, 1:-1, :] = norm[:, 2:, :] - norm[:, :-2, :]
    gy[:, :, 1:-1] = norm[:, :, 2:] - norm[:, :, :-2]
    gx_abs = np.abs(gx).sum(axis=(1, 2))
    gy_abs = np.abs(gy).sum(axis=(1, 2))
    vert_dominance = gy_abs / np.maximum(gx_abs + gy_abs, 1e-9)

    stats = {
        "ink": norm.mean(axis=(1, 2)),
        "aspect": width / height,
        "width": width / 28.0,
        "height": height / 28.0,
        "top_heavy": top_density - bottom_density,
        "side_heavy": left_density - right_density,
        "centroid_row": centroid_row / 27.0,
        "centroid_col": centroid_col / 27.0,
        "hsym": hsym,
        "vsym": vsym,
        "rot_sym": rot_sym,
        "euler": euler,
        "vert_runs": vert_runs,
        "horiz_runs": horiz_runs,
        "vert_dominance": vert_dominance,
        "center_density": center_density,
        "ring_density": ring_density,
        "q3_junctions": q3.astype(np.float64),
    }
    np.savez_compressed(cached, **stats)
    return stats


def discretize(values: np.ndarray, n_states: int, quantiles=None):
    """Quantile discretization; quantiles must come from training data only."""
    if quantiles is None:
        quantiles = np.quantile(values, np.linspace(0, 1, n_states + 1)[1:-1])
    return np.searchsorted(quantiles, values, side="right"), quantiles


def grid_densities(X: np.ndarray) -> np.ndarray:
    """The 10 baseline density statistics: global ink + 3x3 grid cells."""
    imgs = X.reshape(-1, 28, 28).astype(np.float64) / 255.0
    bounds = (7, 12, 17, 22)
    outs = [imgs.mean(axis=(1, 2))]
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        for c0, c1 in zip(bounds[:-1], bounds[1:]):
            outs.append(imgs[:, r0:r1, c0:c1].mean(axis=(1, 2)))
    return np.stack(outs, axis=1)


def disc_zero_sep(v_tr: np.ndarray, v_va: np.ndarray, n_states: int):
    """State 0 = exactly zero; states 1..n-1 = quantiles of positive values.

    Returns (discrete_tr, discrete_va, thresholds) where thresholds are the
    training-derived cut points (for reuse at final-test time).
    """
    pos = v_tr[v_tr > 0]
    qs = np.quantile(pos, np.linspace(0, 1, n_states)[1:-1])
    out_tr = np.searchsorted(qs, v_tr, side="right") + (v_tr > 0)
    out_va = np.searchsorted(qs, v_va, side="right") + (v_va > 0)
    return out_tr, out_va, qs


def fit_eval(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    F_va: np.ndarray,
    y_va: np.ndarray,
    cards: tuple[int, ...],
    parents: tuple[int, ...],
    alpha: float = 1.0,
    backoff: bool = False,
    return_pred: bool = False,
):
    """Fit the hand-designed DAG (Y root, each feature <= 2 parents) and
    evaluate on the validation split.

    parents[i] is the optional feature parent of feature i (-1 = none).
    backoff=True smooths each CPT toward the class-conditional marginal
    instead of a uniform distribution.
    """
    n_feat = F_tr.shape[1]
    n_classes = 10
    class_count = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    log_prior = np.log((class_count + alpha) / (len(y_tr) + n_classes * alpha))

    cpts = []
    for node in range(n_feat):
        card = cards[node]
        parent = parents[node]
        if parent < 0:
            counts = np.zeros((n_classes, card), dtype=np.float64)
            np.add.at(counts, (y_tr, F_tr[:, node]), 1)
            denom = counts.sum(axis=-1, keepdims=True) + card * alpha
            cpts.append(np.log((counts + alpha) / denom))
        else:
            pcard = cards[parent]
            counts = np.zeros((n_classes, pcard, card), dtype=np.float64)
            np.add.at(counts, (y_tr, F_tr[:, parent], F_tr[:, node]), 1)
            if backoff:
                # class-conditional marginal as Dirichlet prior mean
                marg = np.zeros((n_classes, card), dtype=np.float64)
                np.add.at(marg, (y_tr, F_tr[:, node]), 1)
                marg = (marg + alpha) / (class_count[:, None] + card * alpha)
                n_cp = counts.sum(axis=-1, keepdims=True)
                cpts.append(np.log((counts + alpha * marg[:, None, :]) / (n_cp + alpha)))
            else:
                denom = counts.sum(axis=-1, keepdims=True) + card * alpha
                cpts.append(np.log((counts + alpha) / denom))

    scores = np.broadcast_to(log_prior, (F_va.shape[0], n_classes)).copy()
    for node in range(n_feat):
        parent = parents[node]
        if parent < 0:
            scores += cpts[node][:, F_va[:, node]].T
        else:
            scores += cpts[node][:, F_va[:, parent], F_va[:, node]].T
    pred = np.argmax(scores, axis=1)
    acc = float(np.mean(pred == y_va))
    if return_pred:
        return acc, pred
    return acc


def eval_features(
    stats_tr: dict,
    y_tr: np.ndarray,
    stats_va: dict,
    y_va: np.ndarray,
    spec: dict[str, int],
    parents: tuple[int, ...],
    alpha: float = 1.0,
    backoff: bool = False,
    custom_cards: dict[str, int] | None = None,
):
    """Discretize the named stats (quantiles from train only) and run fit_eval."""
    names = list(spec)
    F_tr_cols = []
    F_va_cols = []
    cards = []
    for name in names:
        n_states = (custom_cards or {}).get(name, spec[name])
        v_tr, bins = discretize(stats_tr[name], n_states)
        v_va, _ = discretize(stats_va[name], n_states, quantiles=bins)
        F_tr_cols.append(v_tr)
        F_va_cols.append(v_va)
        cards.append(n_states)
    F_tr = np.stack(F_tr_cols, axis=1)
    F_va = np.stack(F_va_cols, axis=1)
    return fit_eval(
        F_tr, y_tr, F_va, y_va, tuple(cards), parents, alpha=alpha, backoff=backoff
    )
