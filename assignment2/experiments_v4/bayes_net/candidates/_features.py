"""Continuous-feature extraction for the builder-C feature-set redesign.

All continuous statistics are computed ONCE over the 60,000 training images
and cached to `cache/continuous_all.npz`.  Splits (seed 1322 and the extra
falsification seeds) are then obtained by indexing the cached columns with
`stratified_split`, so a new seed costs nothing but discretisation + CPT fit.

Only training data is ever touched.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parents[1]
ROOT = V4.parent
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v4_common import load_train, stratified_split  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
CACHE_FILE = CACHE / "continuous_all.npz"

BINARY_THRESHOLD = 64  # same as the incumbent module


# ----------------------------------------------------------------------
# connected components / holes  (pure NumPy + run-based union-find)
# ----------------------------------------------------------------------
def _row_runs(row: np.ndarray):
    """(start, stop) half-open runs of True in a 1-D boolean row."""
    padded = np.concatenate(([False], row, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return starts, stops


def _count_components_runs(image: np.ndarray, diagonal: bool) -> int:
    """Number of components of a 28x28 mask via row-run union-find.

    `diagonal=False` -> 4-connectivity, `True` -> 8-connectivity.
    """
    parent = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    runs_per_row = []
    for r in range(image.shape[0]):
        row = image[r]
        if not row.any():
            runs_per_row.append((np.empty(0, np.int64), np.empty(0, np.int64), []))
            continue
        starts, stops = _row_runs(row)
        roots = []
        for _ in starts:
            parent.append(len(parent))
            roots.append(len(parent) - 1)
        runs_per_row.append((starts, stops, roots))

    for r in range(image.shape[0] - 1):
        s0, e0, r0 = runs_per_row[r]
        s1, e1, r1 = runs_per_row[r + 1]
        if s0.size == 0 or s1.size == 0:
            continue
        for a in range(s0.size):
            for b in range(s1.size):
                if diagonal:
                    # 8-adjacency: runs merge when expanded by one the intervals touch
                    overlap = (s0[a] <= e1[b]) and (s1[b] <= e0[a])
                else:
                    overlap = (s0[a] < e1[b]) and (s1[b] < e0[a])
                if overlap:
                    ra, rb = find(r0[a]), find(r1[b])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

    return len({find(i) for i in range(len(parent))})


def _border_reachable(background: np.ndarray) -> np.ndarray:
    """4-connected background pixels reachable from the image border (batch)."""
    reached = np.zeros_like(background)
    reached[:, 0, :] = background[:, 0, :]
    reached[:, -1, :] = background[:, -1, :]
    reached[:, :, 0] = background[:, :, 0]
    reached[:, :, -1] = background[:, :, -1]
    while True:
        grown = reached.copy()
        grown[:, 1:, :] |= reached[:, :-1, :]
        grown[:, :-1, :] |= reached[:, 1:, :]
        grown[:, :, 1:] |= reached[:, :, :-1]
        grown[:, :, :-1] |= reached[:, :, 1:]
        grown &= background
        if grown.sum() == reached.sum():
            return reached
        reached = grown


def _components_and_holes(binary: np.ndarray):
    """(ncomp4, ncomp8, nholes4) for a batch of boolean images."""
    n = binary.shape[0]
    ncomp4 = np.zeros(n, dtype=np.int64)
    ncomp8 = np.zeros(n, dtype=np.int64)
    nholes = np.zeros(n, dtype=np.int64)

    reached = _border_reachable(~binary)
    hole_mask = (~binary) & (~reached)

    for i in range(n):
        img = binary[i]
        if not img.any():
            continue
        ncomp4[i] = _count_components_runs(img, diagonal=False)
        ncomp8[i] = _count_components_runs(img, diagonal=True)
        holes = hole_mask[i]
        if holes.any():
            nholes[i] = _count_components_runs(holes, diagonal=False)
    return ncomp4, ncomp8, nholes


def _endpoints_junctions(binary: np.ndarray):
    """Endpoint / junction counts using 8-neighbour topology."""
    padded = np.pad(binary.astype(np.uint8), ((0, 0), (1, 1), (1, 1)))
    core = padded[:, 1:-1, 1:-1].astype(bool)
    nb = (
        padded[:, :-2, :-2] + padded[:, :-2, 1:-1] + padded[:, :-2, 2:]
        + padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:]
        + padded[:, 2:, :-2] + padded[:, 2:, 1:-1] + padded[:, 2:, 2:]
    )
    # cross-count (number of 0->1 transitions around the 8-ring) for junctions
    ring = np.stack([
        padded[:, :-2, 1:-1], padded[:, :-2, 2:], padded[:, 1:-1, 2:],
        padded[:, 2:, 2:], padded[:, 2:, 1:-1], padded[:, 2:, :-2],
        padded[:, 1:-1, :-2], padded[:, :-2, :-2],
    ], axis=0).astype(np.uint8)
    transitions = np.count_nonzero(
        (ring == 0) & (np.roll(ring, -1, axis=0) == 1), axis=0
    ).astype(np.int64)
    endpoints = np.count_nonzero(core & (nb == 1), axis=(1, 2)).astype(np.float64)
    junctions = np.count_nonzero(core & (transitions >= 3), axis=(1, 2)).astype(np.float64)
    return endpoints, junctions


# ----------------------------------------------------------------------
# symmetry
# ----------------------------------------------------------------------
def _symmetry(binary: np.ndarray):
    """Bounding-box-cropped symmetry (reference implementation / slow path)."""
    n = binary.shape[0]
    vsym = np.zeros(n)
    hsym = np.zeros(n)
    for i in range(n):
        img = binary[i]
        rows = np.flatnonzero(img.any(axis=1))
        cols = np.flatnonzero(img.any(axis=0))
        if rows.size == 0:
            continue
        bbox = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        left_right = bbox[:, ::-1]
        top_bottom = bbox[::-1, :]
        inter = np.count_nonzero(bbox & left_right)
        union = np.count_nonzero(bbox | left_right)
        vsym[i] = inter / union if union else 1.0
        inter = np.count_nonzero(bbox & top_bottom)
        union = np.count_nonzero(bbox | top_bottom)
        hsym[i] = inter / union if union else 1.0
    return vsym, hsym


def _reflect_roll(binary: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """binary mirrored along the last axis then rolled per-image by `shift`.

    Reflection about an axis at column c maps j -> 2c - j, which composes with
    the full-image mirror (j -> 27 - j) into a pure integer roll of
    d = round(2c - 27).  Rolled-in (wrapped) columns are zeroed.
    """
    n, h, w = binary.shape
    mirrored = binary[:, :, ::-1]
    idx = np.arange(w)[None, :] - shift[:, None]
    valid = (idx >= 0) & (idx < w)
    idx_c = np.clip(idx, 0, w - 1)
    gathered = np.take_along_axis(
        mirrored, np.broadcast_to(idx_c[:, None, :], (n, h, w)), axis=2
    )
    return gathered & valid[:, None, :]


def _symmetry_aligned(binary: np.ndarray, center_col: np.ndarray, center_row: np.ndarray):
    """Left-right and top-bottom symmetry about the ink bounding-box centre.

    Jaccard overlap between the ink mask and its reflection about the box
    centre: 1.0 = perfectly symmetric (1, 8, 0), ~0.4 = clearly asymmetric (2, 4).
    """
    shift_col = np.rint(2.0 * center_col - 27.0).astype(np.int64)
    shift_row = np.rint(2.0 * center_row - 27.0).astype(np.int64)
    refl_lr = _reflect_roll(binary, shift_col)
    refl_ud = _reflect_roll(binary.transpose(0, 2, 1), shift_row).transpose(0, 2, 1)
    inter_lr = np.count_nonzero(binary & refl_lr, axis=(1, 2)).astype(np.float64)
    union_lr = np.count_nonzero(binary | refl_lr, axis=(1, 2)).astype(np.float64)
    inter_ud = np.count_nonzero(binary & refl_ud, axis=(1, 2)).astype(np.float64)
    union_ud = np.count_nonzero(binary | refl_ud, axis=(1, 2)).astype(np.float64)
    vsym = np.divide(inter_lr, union_lr, out=np.zeros_like(inter_lr), where=union_lr > 0)
    hsym = np.divide(inter_ud, union_ud, out=np.zeros_like(inter_ud), where=union_ud > 0)
    return vsym, hsym


def _symmetry_fast(binary: np.ndarray):
    """Vectorised symmetry: full-image mirror Jaccard (no bbox alignment)."""
    flip_lr = binary[:, :, ::-1]
    flip_ud = binary[:, ::-1, :]
    inter_lr = np.count_nonzero(binary & flip_lr, axis=(1, 2)).astype(np.float64)
    union_lr = np.count_nonzero(binary | flip_lr, axis=(1, 2)).astype(np.float64)
    inter_ud = np.count_nonzero(binary & flip_ud, axis=(1, 2)).astype(np.float64)
    union_ud = np.count_nonzero(binary | flip_ud, axis=(1, 2)).astype(np.float64)
    vsym = np.divide(inter_lr, union_lr, out=np.zeros_like(inter_lr), where=union_lr > 0)
    hsym = np.divide(inter_ud, union_ud, out=np.zeros_like(inter_ud), where=union_ud > 0)
    return vsym, hsym


# ----------------------------------------------------------------------
# full continuous-feature table
# ----------------------------------------------------------------------
def continuous_features(X: np.ndarray, with_components: bool = True) -> dict:
    images = X.reshape(-1, 28, 28).astype(np.float64) / 255.0
    binary = images > (BINARY_THRESHOLD / 255.0)
    n = images.shape[0]

    out = {}

    # --- central 15x15 3x3 grid (reproduces the incumbent's cells) -----
    grid_names = ("tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br")
    for k, (r, c) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):
        out[f"dens_{grid_names[k]}"] = images[:, 7 + 5 * r:12 + 5 * r, 7 + 5 * c:12 + 5 * c].mean(axis=(1, 2))

    # --- 2x2 quadrants of the whole image -----------------------------
    out["dens_q_tl"] = images[:, :14, :14].mean(axis=(1, 2))
    out["dens_q_tr"] = images[:, :14, 14:].mean(axis=(1, 2))
    out["dens_q_bl"] = images[:, 14:, :14].mean(axis=(1, 2))
    out["dens_q_br"] = images[:, 14:, 14:].mean(axis=(1, 2))

    # --- row / column thirds (PDF "top/middle/bottom, left/middle/right") --
    out["row_band_top"] = images[:, 0:9, :].mean(axis=(1, 2))
    out["row_band_mid"] = images[:, 9:19, :].mean(axis=(1, 2))
    out["row_band_bot"] = images[:, 19:28, :].mean(axis=(1, 2))
    out["col_band_left"] = images[:, :, 0:9].mean(axis=(1, 2))
    out["col_band_mid"] = images[:, :, 9:19].mean(axis=(1, 2))
    out["col_band_right"] = images[:, :, 19:28].mean(axis=(1, 2))
    out["global_ink"] = images.mean(axis=(1, 2))

    # --- geometry (correct ink bounding box) --------------------------
    rows_any = np.any(binary, axis=2)
    cols_any = np.any(binary, axis=1)
    row_idx = np.arange(28)[None, :]
    col_idx = np.arange(28)[None, :]
    top = np.where(rows_any, row_idx, 28).min(axis=1)          # first ink row
    bottom = np.where(rows_any, row_idx, -1).max(axis=1)       # last ink row
    left = np.where(cols_any, col_idx, 28).min(axis=1)         # first ink col
    right = np.where(cols_any, col_idx, -1).max(axis=1)        # last ink col
    height = np.clip(bottom - top + 1, 1, 28)
    width = np.clip(right - left + 1, 1, 28)
    out["aspect"] = width / height
    out["bbox_fill"] = np.count_nonzero(binary, axis=(1, 2)) / (height * width).astype(np.float64)
    out["height"] = height.astype(np.float64)
    out["width"] = width.astype(np.float64)
    out["center_row"] = ((top + bottom) / 2.0).astype(np.float64)
    out["center_col"] = ((left + right) / 2.0).astype(np.float64)
    out["bbox_top"] = top.astype(np.float64)
    out["bbox_bottom"] = bottom.astype(np.float64)
    out["bbox_left"] = left.astype(np.float64)
    out["bbox_right"] = right.astype(np.float64)

    # --- the incumbent module's aspect formula (kept for exact reproduction)
    # bayes_net.py computes  height = (27 - first_ink_row) - last_ink_row + 1
    # and width likewise; that is NOT the true bbox extent, but reproducing it
    # is required to match the incumbent's 0.8614 exactly.
    inc_height = np.clip((27 - top) - bottom + 1, 1, 28)
    inc_width = np.clip((27 - left) - right + 1, 1, 28)
    out["aspect_incumbent"] = inc_width.astype(np.float64) / inc_height.astype(np.float64)

    # --- gradient dominance ------------------------------------------
    gx = np.zeros_like(images)
    gy = np.zeros_like(images)
    gx[:, 1:-1, :] = images[:, 2:, :] - images[:, :-2, :]
    gy[:, :, 1:-1] = images[:, :, 2:] - images[:, :, :-2]
    gx_abs = np.abs(gx).sum(axis=(1, 2))
    gy_abs = np.abs(gy).sum(axis=(1, 2))
    out["vertical_dominance"] = gy_abs / np.maximum(gx_abs + gy_abs, 1e-9)

    # --- runs ---------------------------------------------------------
    run_starts_v = np.count_nonzero(binary[:, 0, :], axis=1) + np.count_nonzero(
        (~binary[:, :-1, :]) & binary[:, 1:, :], axis=(1, 2)
    )
    occupied_columns = np.count_nonzero(binary.any(axis=1), axis=1)
    out["vertical_runs"] = run_starts_v / np.maximum(occupied_columns, 1)

    run_starts_h = np.count_nonzero(binary[:, :, 0], axis=1) + np.count_nonzero(
        (~binary[:, :, :-1]) & binary[:, :, 1:], axis=(1, 2)
    )
    occupied_rows = np.count_nonzero(binary.any(axis=2), axis=1)
    out["horiz_runs"] = run_starts_h / np.maximum(occupied_rows, 1)

    # --- euler number (incumbent's formula) ---------------------------
    padded = np.pad(binary, ((0, 0), (1, 1), (1, 1)))
    ul, ll = padded[:, :-1, :-1], padded[:, 1:, :-1]
    ur, lr = padded[:, :-1, 1:], padded[:, 1:, 1:]
    block_sum = ul.astype(np.uint8) + ll + ur + lr
    diag_pairs = (ul & lr & ~ll & ~ur) | (ll & ur & ~ul & ~lr)
    q1 = np.count_nonzero(block_sum == 1, axis=(1, 2))
    q3 = np.count_nonzero(block_sum == 3, axis=(1, 2))
    qd = np.count_nonzero(diag_pairs, axis=(1, 2))
    out["euler"] = np.rint((q1 - q3 + 2 * qd) / 4.0)

    # --- symmetry -----------------------------------------------------
    vsym, hsym = _symmetry_fast(binary)
    out["vsym_jaccard"] = vsym
    out["hsym_jaccard"] = hsym
    center_col = ((left + right) / 2.0).astype(np.float64)
    center_row = ((top + bottom) / 2.0).astype(np.float64)
    vsym_a, hsym_a = _symmetry_aligned(binary, center_col, center_row)
    out["vsym_aligned"] = vsym_a
    out["hsym_aligned"] = hsym_a

    # --- endpoints / junctions ---------------------------------------
    endpoints, junctions = _endpoints_junctions(binary)
    out["endpoints"] = endpoints
    out["junctions"] = junctions

    # --- stroke width proxy ------------------------------------------
    out["stroke_width"] = np.count_nonzero(binary, axis=(1, 2)).astype(np.float64) / np.maximum(
        _perimeter(binary), 1.0
    )

    # --- topology (expensive) ----------------------------------------
    if with_components:
        ncomp4, ncomp8, nholes = _components_and_holes(binary)
        out["ncomp4"] = ncomp4.astype(np.float64)
        out["ncomp8"] = ncomp8.astype(np.float64)
        out["nholes"] = nholes.astype(np.float64)

    return out


def _perimeter(binary: np.ndarray) -> np.ndarray:
    padded = np.pad(binary.astype(np.uint8), ((0, 0), (1, 1), (1, 1)))
    core = padded[:, 1:-1, 1:-1]
    nb = (
        padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1]
        + padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:]
        + padded[:, :-2, :-2] + padded[:, :-2, 2:]
        + padded[:, 2:, :-2] + padded[:, 2:, 2:]
    )
    return np.count_nonzero((core > 0) & (nb < 8), axis=(1, 2)).astype(np.float64)


def build_cache(force: bool = False) -> dict:
    if CACHE_FILE.is_file() and not force:
        with np.load(CACHE_FILE) as data:
            return {k: data[k] for k in data.files}
    X, y = load_train()
    start = time.perf_counter()
    feats = continuous_features(X)
    print(f"continuous features for {X.shape[0]} images in {time.perf_counter()-start:.1f}s")
    feats["label"] = y
    np.savez_compressed(CACHE_FILE, **feats)
    return feats


def split_columns(feats: dict, seed: int, validation_fraction: float = 1 / 6):
    """Return (dict_of_train_columns, ytr, dict_of_val_columns, yva) for a seed."""
    y = np.asarray(feats["label"])
    tr_idx, va_idx = stratified_split(y, validation_fraction, seed)
    keys = [k for k in feats if k != "label"]
    cols_tr = {k: np.asarray(feats[k])[tr_idx] for k in keys}
    cols_va = {k: np.asarray(feats[k])[va_idx] for k in keys}
    return cols_tr, y[tr_idx], cols_va, y[va_idx]


if __name__ == "__main__":
    feats = build_cache(force=True)
    print("columns:", sorted(k for k in feats if k != "label"))
    for k in sorted(feats):
        if k == "label":
            continue
        col = np.asarray(feats[k])
        print(f"  {k:20s} min={col.min():8.4f} med={np.median(col):8.4f} "
              f"max={col.max():8.4f} zeros={np.mean(col == 0)*100:5.1f}%")
