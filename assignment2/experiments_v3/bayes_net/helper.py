"""Shared helpers for the v3 bayes_net iteration experiments.

Reuses the data loading protocol from experiments_v2/bayes_net/common.py
(main.py stratified_split, seed 1322).  New directional / structural
continuous statistics are computed once and cached on disk so each round
only pays for discretization + CPT fitting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2_DIR = ROOT / "experiments_v2" / "bayes_net"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V2_DIR))

from common import (  # noqa: E402,F401  (re-export v2 helpers)
    disc_zero_sep,
    discretize,
    fit_eval,
    grid_densities,
    load_split,
)

CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def directional_stats(X: np.ndarray, tag: str) -> dict[str, np.ndarray]:
    """New directional / structural continuous statistics, cached on disk.

    All statistics operate on the binary image (> 64) and the normalized
    grayscale image, matching the v2 feature conventions.
    """
    cached = CACHE_DIR / f"dir_{tag}.npz"
    if cached.is_file():
        with np.load(cached) as data:
            return {k: data[k] for k in data.files}

    imgs = X.reshape(-1, 28, 28).astype(np.float64) / 255.0
    binary = imgs > 64 / 255.0
    n = imgs.shape[0]

    # ------------------------------------------------------------------
    # 1. gradient-orientation histogram over the central region
    #    (quantized into up / down / left / right weights)
    # ------------------------------------------------------------------
    gx = np.zeros_like(imgs)
    gy = np.zeros_like(imgs)
    gx[:, 1:-1, :] = imgs[:, 2:, :] - imgs[:, :-2, :]
    gy[:, :, 1:-1] = imgs[:, :, 2:] - imgs[:, :, :-2]
    mag = np.sqrt(gx * gx + gy * gy)
    # central 15x15 region mask
    mask = np.zeros((28, 28), dtype=bool)
    mask[7:22, 7:22] = True
    m = mag * mask
    tot = m.sum(axis=(1, 2))
    safe_tot = np.where(tot > 1e-9, tot, 1.0)
    # orientation weights: vertical vs horizontal, and up vs down sign
    vert_mag = (np.abs(gy) * mask).sum(axis=(1, 2))
    horiz_mag = (np.abs(gx) * mask).sum(axis=(1, 2))
    grad_vert_frac = vert_mag / np.maximum(vert_mag + horiz_mag, 1e-9)
    # signed: downward gradient (ink getting darker downwards -> top edge of stroke)
    down_mag = (np.clip(gy, 0, None) * mask).sum(axis=(1, 2))
    up_mag = (np.clip(-gy, 0, None) * mask).sum(axis=(1, 2))
    grad_up_frac = up_mag / np.maximum(up_mag + down_mag, 1e-9)

    # ------------------------------------------------------------------
    # 2. opening features: ink presence on left edge of upper/lower half
    #    distinguishing 3 (no left ink in middle) from 5 / 2
    # ------------------------------------------------------------------
    left_ink = binary[:, :, 7:12]  # left band of central region
    upper_left_ink = left_ink[:, 7:14, :].mean(axis=(1, 2))   # upper-left band
    lower_left_ink = left_ink[:, 14:22, :].mean(axis=(1, 2))  # lower-left band
    right_ink = binary[:, :, 17:22]
    upper_right_ink = right_ink[:, 7:14, :].mean(axis=(1, 2))
    lower_right_ink = right_ink[:, 14:22, :].mean(axis=(1, 2))
    # "openness" = side of the middle band being empty
    mid_left_ink = binary[:, 13:17, 7:11].mean(axis=(1, 2))
    mid_right_ink = binary[:, 13:17, 17:21].mean(axis=(1, 2))

    # ------------------------------------------------------------------
    # 3. horizontal runs (dual of vertical_runs): average ink segments
    #    per occupied row
    # ------------------------------------------------------------------
    run_starts_h = np.count_nonzero(binary[:, :, 0], axis=1)
    run_starts_h += np.count_nonzero(
        (~binary[:, :, :-1]) & binary[:, :, 1:], axis=(1, 2)
    )
    occupied_rows = np.maximum(np.count_nonzero(binary.any(axis=1), axis=1), 1)
    horiz_runs = run_starts_h / occupied_rows

    # ------------------------------------------------------------------
    # 4. endpoint count (pixels with exactly 1 binary neighbour)
    # ------------------------------------------------------------------
    padded = np.pad(binary.astype(np.uint8), ((0, 0), (1, 1), (1, 1)))
    nb = (
        padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1]
        + padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:]
        + padded[:, :-2, :-2] + padded[:, :-2, 2:]
        + padded[:, 2:, :-2] + padded[:, 2:, 2:]
    )
    core = padded[:, 1:-1, 1:-1]
    endpoints = np.count_nonzero((core == 1) & (nb == 1), axis=(1, 2)).astype(
        np.float64
    )

    # ------------------------------------------------------------------
    # 5. per-band densities (horizontal thirds of the central region)
    #    and stroke-width proxy (ink / perimeter)
    # ------------------------------------------------------------------
    band_top = imgs[:, 7:12, 7:22].mean(axis=(1, 2))
    band_mid = imgs[:, 12:17, 7:22].mean(axis=(1, 2))
    band_bot = imgs[:, 17:22, 7:22].mean(axis=(1, 2))
    ink_total = imgs.sum(axis=(1, 2))
    perimeter = np.count_nonzero(
        core.astype(bool)
        & (nb < 8)
        & (nb > 0),
        axis=(1, 2),
    ).astype(np.float64)
    stroke_width = ink_total / np.maximum(perimeter, 1.0)

    stats = {
        "grad_vert_frac": grad_vert_frac,
        "grad_up_frac": grad_up_frac,
        "upper_left_ink": upper_left_ink,
        "lower_left_ink": lower_left_ink,
        "upper_right_ink": upper_right_ink,
        "lower_right_ink": lower_right_ink,
        "mid_left_ink": mid_left_ink,
        "mid_right_ink": mid_right_ink,
        "horiz_runs": horiz_runs,
        "endpoints": endpoints,
        "band_top": band_top,
        "band_mid": band_mid,
        "band_bot": band_bot,
        "stroke_width": stroke_width,
    }
    np.savez_compressed(cached, **stats)
    return stats


def baseline_columns(X_tr, X_va):
    """The 12 v2 features as (F_tr, F_va, cards) with train-only thresholds.

    IMPORTANT: the v2 module's density cells are (in FEATURE_NAMES order)
    tl, tc, tr, ml, mc, mr, bl, br over the central 15x15 grid — i.e. the
    KEPT cells are grid indices 0,1,2,3,4,5,7,8 of the 3x3 grid, where
    grid index 6 (bottom_center, row 17-22, col 12-17) is omitted.
    grid_densities() returns global ink + all 9 cells, so we must drop
    grid index 6 (column 7), NOT the module's bottom_right.
    """
    G_tr = grid_densities(X_tr)
    G_va = grid_densities(X_va)
    cols_tr, cols_va, cards = [], [], []
    # grid_densities column j: 0=global ink, 1..9 = 3x3 cells in row-major
    # order (1=tl 2=tc 3=tr 4=ml 5=mc 6=mr 7=bl 8=bc 9=br).  The v2 module
    # drops bottom_center (j=8) and keeps bottom_left (j=7).
    for j in (1, 2, 3, 4, 5, 6, 7, 9):
        a, b, _ = disc_zero_sep(G_tr[:, j], G_va[:, j], 5)
        cols_tr.append(a)
        cols_va.append(b)
        cards.append(5)
    from common import continuous_stats

    s_tr = continuous_stats(X_tr, "train")
    s_va = continuous_stats(X_va, "val")
    e_tr = (np.clip(s_tr["euler"], -1, 2) + 1).astype(np.int64)
    e_va = (np.clip(s_va["euler"], -1, 2) + 1).astype(np.int64)
    cols_tr.append(e_tr)
    cols_va.append(e_va)
    cards.append(4)
    for name, card in (("vert_runs", 5), ("aspect", 4), ("vert_dominance", 4)):
        a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
        cols_tr.append(a.astype(np.int64))
        cols_va.append(b.astype(np.int64))
        cards.append(card)
    return (
        np.stack(cols_tr, axis=1).astype(np.int64),
        np.stack(cols_va, axis=1).astype(np.int64),
        tuple(cards),
    )


# v2 FEATURE_PARENTS, in baseline_columns order:
# 8 grid (tl,tc,tr,ml,mc,mr,bl,br) + euler + vruns + aspect + vdom
BASELINE_PARENTS = (-1, -1, -1, 0, 2, 1, 4, 5, -1, 8, 11, -1)
