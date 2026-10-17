"""Round-2 idea (A): CONTENT-RELATIVE (bounding-box normalised) features.

The incumbent's 8 grid densities are ABSOLUTE region means: fixed rows/cols
7..22 of the 28x28 image.  A digit written small and centred, or large and
off-centre, lands its ink in different fixed windows, so the absolute windows
mix "shape" with "position and scale".  The features here are measured in the
image's OWN ink bounding box, i.e. *after* normalising position and scale:

  bb*   : density of ink in a normalised band / cell of the ink bbox
          (top / middle / bottom horizontal thirds, left / middle / right
          vertical thirds, and the four 2x2 bbox quadrants)
  bbCR  : vertical centroid of the ink inside the bbox, normalised (0 top .. 1 bottom)
  bbCC  : horizontal centroid of the ink inside the bbox, normalised (0 left .. 1 right)
  bbPR  : normalised position of the fullest ink ROW inside the bbox
  bbPC  : normalised position of the fullest ink COLUMN inside the bbox

Every one is a single continuous statistic with an unambiguous semantic that a
reader can state in one sentence, and is later discretised into 2..5 states.
All extraction is vectorised NumPy; only the 60,000 training images are read.
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
CACHE_FILE = CACHE / "round2_content.npz"

BINARY_THRESHOLD = 64


def _bands(counts, rows_per, cols_per, nb):
    """density per normalised (row, col) band, robust to empty bands."""
    area = rows_per[:, :, None] * cols_per[:, None, :]
    return counts / np.maximum(area, 1.0)


def content_features(X: np.ndarray) -> dict:
    binary = X.reshape(-1, 28, 28) > BINARY_THRESHOLD
    n = binary.shape[0]
    r1 = np.arange(28, dtype=np.float64)[None, :]
    idx1 = np.arange(28)[None, :]

    rows_any = binary.any(axis=2)
    cols_any = binary.any(axis=1)
    top = np.where(rows_any, idx1, 28).min(axis=1)
    bottom = np.where(rows_any, idx1, -1).max(axis=1)
    left = np.where(cols_any, idx1, 28).min(axis=1)
    right = np.where(cols_any, idx1, -1).max(axis=1)
    height = np.clip(bottom - top + 1, 1, 28).astype(np.float64)
    width = np.clip(right - left + 1, 1, 28).astype(np.float64)

    out = {}

    def grid(nb, tag):
        r = np.arange(28)[None, :, None]
        c = np.arange(28)[None, None, :]
        rpos = np.clip(r - top[:, None, None], 0, 27)
        cpos = np.clip(c - left[:, None, None], 0, 27)
        rb = np.clip((rpos * nb) // height[:, None, None], 0, nb - 1).astype(np.int64)
        cb = np.clip((cpos * nb) // width[:, None, None], 0, nb - 1).astype(np.int64)
        cell = rb * nb + cb
        flat = np.arange(n)[:, None, None] * (nb * nb) + cell
        counts = np.bincount(
            flat[binary].ravel(), minlength=n * nb * nb
        ).astype(np.float64).reshape(n, nb, nb)
        row_in = (r1 >= top[:, None]) & (r1 <= bottom[:, None])
        rbidx = np.arange(n)[:, None] * nb + np.clip(
            ((r1 - top[:, None]) * nb) // height[:, None], 0, nb - 1
        ).astype(np.int64)
        rows_per = np.bincount(
            rbidx.ravel(), weights=row_in.ravel().astype(np.float64), minlength=n * nb
        ).reshape(n, nb)
        col_in = (r1 >= left[:, None]) & (r1 <= right[:, None])
        cbidx = np.arange(n)[:, None] * nb + np.clip(
            ((r1 - left[:, None]) * nb) // width[:, None], 0, nb - 1
        ).astype(np.int64)
        cols_per = np.bincount(
            cbidx.ravel(), weights=col_in.ravel().astype(np.float64), minlength=n * nb
        ).reshape(n, nb)
        return counts, rows_per, cols_per, _bands(counts, rows_per, cols_per, nb)

    # --- 3x3 normalised bbox grid -> 3 row bands + 3 column bands + 9 cells
    counts3, rows3, cols3, dens3 = grid(3, "3")
    for a, rn in enumerate(("top", "mid", "bot")):
        out[f"bb{rn.capitalize()}"] = counts3[:, a, :].sum(axis=1) / np.maximum(
            rows3[:, a] * width, 1.0
        )
    for b, cn in enumerate(("left", "mid", "right")):
        out[f"bbC{cn.capitalize()}"] = counts3[:, :, b].sum(axis=1) / np.maximum(
            cols3[:, b] * height, 1.0
        )
    for a, rn in enumerate(("t", "m", "b")):
        for b, cn in enumerate(("l", "c", "r")):
            out[f"bb{rn.upper()}{cn.upper()}"] = dens3[:, a, b]

    # --- 2x2 normalised bbox quadrants
    counts2, rows2, cols2, dens2 = grid(2, "2")
    for a, rn in enumerate(("t", "b")):
        for b, cn in enumerate(("l", "r")):
            out[f"bbQ{rn.upper()}{cn.upper()}"] = dens2[:, a, b]

    # --- normalised centroids inside the bbox
    total = np.maximum(binary.sum(axis=(1, 2)).astype(np.float64), 1.0)
    cent_row = (binary * np.arange(28)[None, :, None]).sum(axis=(1, 2)) / total
    cent_col = (binary * np.arange(28)[None, None, :]).sum(axis=(1, 2)) / total
    out["bbCR"] = (cent_row - top) / np.maximum(height - 1.0, 1.0)
    out["bbCC"] = (cent_col - left) / np.maximum(width - 1.0, 1.0)

    # --- normalised position of the fullest ink row / column inside the bbox
    prof_r = binary.sum(axis=2).astype(np.float64)
    prof_r[(r1 < top[:, None]) | (r1 > bottom[:, None])] = -1.0
    prof_c = binary.sum(axis=1).astype(np.float64)
    prof_c[(r1 < left[:, None]) | (r1 > right[:, None])] = -1.0
    peak_r = prof_r.argmax(axis=1).astype(np.float64)
    peak_c = prof_c.argmax(axis=1).astype(np.float64)
    out["bbPR"] = (peak_r - top) / np.maximum(height - 1.0, 1.0)
    out["bbPC"] = (peak_c - left) / np.maximum(width - 1.0, 1.0)

    return out


def build_cache(force: bool = False) -> dict:
    if CACHE_FILE.is_file() and not force:
        with np.load(CACHE_FILE) as data:
            return {k: data[k] for k in data.files}
    X, y = load_train()
    start = time.perf_counter()
    feats = content_features(X)
    print(f"content features for {X.shape[0]} images in {time.perf_counter()-start:.1f}s")
    feats["label"] = y
    np.savez_compressed(CACHE_FILE, **feats)
    return feats


def split_columns(feats: dict, seed: int, validation_fraction: float = 1 / 6):
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
        print(f"  {k:8s} min={col.min():7.4f} med={np.median(col):7.4f} "
              f"max={col.max():7.4f} zeros={np.mean(col == 0)*100:5.1f}%")


# ----------------------------------------------------------------------
# CONTROL features: isolate POSITION normalisation from SCALE normalisation
# ----------------------------------------------------------------------
def control_features(X: np.ndarray) -> dict:
    """Same 3x3 grid, but with only ONE of the two normalisations applied.

    trans3_*  : grid over a fixed 15x15 window that is merely SHIFTED per image
                so the ink bbox centre lands on the window centre (position
                normalised, scale NOT).  If idea A only "recentres", this
                control should recover most of the gain.
    scale3_*  : fixed window at the image centre, cell size scaled by the
                image's bbox extent (scale normalised, position NOT).
    """
    binary = X.reshape(-1, 28, 28) > BINARY_THRESHOLD
    n = binary.shape[0]
    idx1 = np.arange(28)[None, :]
    rows_any = binary.any(axis=2)
    cols_any = binary.any(axis=1)
    top = np.where(rows_any, idx1, 28).min(axis=1)
    bottom = np.where(rows_any, idx1, -1).max(axis=1)
    left = np.where(cols_any, idx1, 28).min(axis=1)
    right = np.where(cols_any, idx1, -1).max(axis=1)
    height = np.clip(bottom - top + 1, 1, 28).astype(np.float64)
    width = np.clip(right - left + 1, 1, 28).astype(np.float64)
    cent_r = (top + bottom) / 2.0
    cent_c = (left + right) / 2.0

    r = np.arange(28)[None, :, None]
    c = np.arange(28)[None, None, :]
    out = {}

    # --- translation-only: shift so bbox centre -> window centre, fixed 15px window
    sh_r = np.rint(cent_r - 13.5).astype(np.int64)
    sh_c = np.rint(cent_c - 13.5).astype(np.int64)
    rp = r - sh_r[:, None, None]
    cp = c - sh_c[:, None, None]
    in_win = (rp >= 7) & (rp < 22) & (cp >= 7) & (cp < 22)
    rb = np.clip(((rp - 7) * 3) // 15, 0, 2)
    cb = np.clip(((cp - 7) * 3) // 15, 0, 2)
    cell = rb * 3 + cb
    flat = np.arange(n)[:, None, None] * 9 + cell
    cnt = np.bincount(flat[in_win & binary].ravel(), minlength=n * 9).astype(np.float64)
    den = np.bincount(flat[in_win].ravel(), minlength=n * 9).astype(np.float64)
    dens = (cnt / np.maximum(den, 1.0)).reshape(n, 3, 3)
    for a, rn in enumerate(("t", "m", "b")):
        for b, cn in enumerate(("l", "c", "r")):
            out[f"tr{rn.upper()}{cn.upper()}"] = dens[:, a, b]

    # --- scale-only: 3x3 window centred on the IMAGE centre (position NOT
    # normalised) whose cells are sized so the 3 cells span the bbox extent.
    scale = np.maximum(np.maximum(height, width), 6.0) / 3.0  # 3 cells ~ extent
    lo_r = 13.5 - 1.5 * scale
    lo_c = 13.5 - 1.5 * scale
    rb2 = np.clip(((r - lo_r[:, None, None]) / scale[:, None, None]).astype(np.int64), 0, 2)
    cb2 = np.clip(((c - lo_c[:, None, None]) / scale[:, None, None]).astype(np.int64), 0, 2)
    cell2 = rb2 * 3 + cb2
    flat2 = np.arange(n)[:, None, None] * 9 + cell2
    cnt2 = np.bincount(flat2[binary].ravel(), minlength=n * 9).astype(np.float64)
    den2 = np.bincount(flat2.ravel(), minlength=n * 9).astype(np.float64)
    dens2 = (cnt2 / np.maximum(den2, 1.0)).reshape(n, 3, 3)
    for a, rn in enumerate(("t", "m", "b")):
        for b, cn in enumerate(("l", "c", "r")):
            out[f"sc{rn.upper()}{cn.upper()}"] = dens2[:, a, b]

    # --- translation-only / scale-only position statistics of the bbox
    out["trCR"] = np.clip((cent_r - 13.5) / 13.5, -1, 1)
    out["trCC"] = np.clip((cent_c - 13.5) / 13.5, -1, 1)
    out["trH"] = height
    out["trW"] = width
    return out


CONTROL_FILE = CACHE / "round2_controls.npz"


def build_controls(force: bool = False) -> dict:
    if CONTROL_FILE.is_file() and not force:
        with np.load(CONTROL_FILE) as data:
            return {k: data[k] for k in data.files}
    X, y = load_train()
    start = time.perf_counter()
    feats = control_features(X)
    print(f"control features for {X.shape[0]} images in {time.perf_counter()-start:.1f}s")
    feats["label"] = y
    np.savez_compressed(CONTROL_FILE, **feats)
    return feats
