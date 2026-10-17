"""Exploration G: grid geometry (cell window size / position) for the 8 slots.

The incumbent fixes the 3x3 grid to the central 15x15 region and drops the
bottom-centre cell.  The PDF only asks for "top/middle/bottom + left/middle/right
region density", so the window is a free design choice.  Here we vary the grid
window (12x12, 15x15, 18x18, 21x21, 24x24, 28x28 whole image) and re-run the
incumbent's 8-cell + row-chain + 4-semantic design.  Densities are recomputed
and cached per window.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V4 = HERE.parents[1]
sys.path.insert(0, str(V4))

from v4_common import load_train, stratified_split  # noqa: E402
from _features import build_cache, split_columns, CACHE  # noqa: E402
from _harness import evaluate  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP, template_incumbent  # noqa: E402

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
SEM4 = [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
NAMES = ("tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br")


def grid_window(window: int):
    """8 kept cells of a 3x3 grid laid over a `window`x`window` centred square."""
    cache_file = CACHE / f"grid_win{window}.npz"
    if cache_file.is_file():
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}
    X, y = load_train()
    images = X.reshape(-1, 28, 28).astype(np.float64) / 255.0
    start = (28 - window) // 2
    step = window / 3.0
    out = {}
    for k in range(9):
        r, c = k // 3, k % 3
        r0 = int(round(start + r * step))
        r1 = int(round(start + (r + 1) * step))
        c0 = int(round(start + c * step))
        c1 = int(round(start + (c + 1) * step))
        out[f"dens_{NAMES[k]}"] = images[:, r0:r1, c0:c1].mean(axis=(1, 2))
    out["label"] = y
    np.savez_compressed(cache_file, **out)
    return out


def main():
    feats = build_cache()
    ctr0, ytr, cva0, yva = split_columns(feats, 1322)
    base, _, _, _, _ = evaluate(
        [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + SEM4,
        INC_PARENTS, ctr0, cva0, ytr, yva)
    print(f"incumbent (15x15 central) = {base:.4f}\n")

    results = []
    for window in (12, 15, 18, 21, 24, 28):
        g = grid_window(window)
        tr_idx, va_idx = stratified_split(np.asarray(g["label"]), 1 / 6, 1322)
        ctr, cva = dict(ctr0), dict(cva0)
        for k in g:
            if k == "label":
                continue
            ctr[k] = np.asarray(g[k])[tr_idx]
            cva[k] = np.asarray(g[k])[va_idx]
        design = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + SEM4
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr, cva, ytr, yva)
        results.append((acc, window))
        print(f"  window {window:2d}x{window:2d}  {acc:.4f}  ({acc-base:+.4f})")

    print("\n=== also: 8 slots from the 9-cell grid WITHOUT dropping bc, at 24/28 ===")
    for window in (21, 24, 28):
        for drop in ("bc", "mc", "br"):
            g = grid_window(window)
            tr_idx, va_idx = stratified_split(np.asarray(g["label"]), 1 / 6, 1322)
            ctr, cva = dict(ctr0), dict(cva0)
            for k in g:
                if k == "label":
                    continue
                ctr[k] = np.asarray(g[k])[tr_idx]
                cva[k] = np.asarray(g[k])[va_idx]
            keep = [c for c in ("dens_tl", "dens_tc", "dens_tr", "dens_ml", "dens_mc",
                                "dens_mr", "dens_bl", "dens_bc", "dens_br")
                    if c != f"dens_{drop}"][:8]
            # row chain over these cells
            from _sweep2_cells import full_parents
            parents = full_parents(tuple(keep), "incumbent")
            design = [(c, 5, "zero_quantile") for c in keep] + SEM4
            acc, _, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
            print(f"  window {window} drop {drop}: {acc:.4f}  ({acc-base:+.4f})")

    best = max(results)
    print(f"\nbest geometry: window {best[1]} -> {best[0]:.4f}")


if __name__ == "__main__":
    main()
