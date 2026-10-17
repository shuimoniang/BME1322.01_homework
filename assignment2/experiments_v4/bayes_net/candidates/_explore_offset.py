"""Exploration H: grid OFFSET and cell step for the 8 density slots.

`_explore_geometry` showed the incumbent's window (rows/cols 7..22) beats the
centred 15x15 window (6..21) by 2.6pt, so the exact placement is a real design
axis.  Here we sweep the grid origin and cell step with the incumbent's 8-cell
layout + row chain + 4 semantic features, then re-verify the winner on the
extra seeds with the full harness before making any claim.
"""
from __future__ import annotations

import itertools
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
from _sweep2_cells import INCUMBENT_KEEP  # noqa: E402

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
SEM4 = [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
NAMES = ("tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br")
KEPT = (0, 1, 2, 3, 4, 5, 6, 8)  # drop bc (index 7)


def grid_offset(origin: int, step: int):
    cache_file = CACHE / f"grid_o{origin}_s{step}.npz"
    if cache_file.is_file():
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}
    X, y = load_train()
    images = X.reshape(-1, 28, 28).astype(np.float64) / 255.0
    out = {}
    for k in range(9):
        r, c = k // 3, k % 3
        r0, r1 = origin + r * step, origin + (r + 1) * step
        c0, c1 = origin + c * step, origin + (c + 1) * step
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
    print(f"incumbent (origin 7, step 5) = {base:.4f}\n")

    results = []
    for origin, step in itertools.product(range(4, 11), (4, 5, 6, 7)):
        if origin + 3 * step > 28:
            continue
        g = grid_offset(origin, step)
        tr_idx, va_idx = stratified_split(np.asarray(g["label"]), 1 / 6, 1322)
        ctr, cva = dict(ctr0), dict(cva0)
        for k in g:
            if k == "label":
                continue
            ctr[k] = np.asarray(g[k])[tr_idx]
            cva[k] = np.asarray(g[k])[va_idx]
        design = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + SEM4
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr, cva, ytr, yva)
        results.append((acc, origin, step))

    results.sort(reverse=True)
    print(f"{'origin':>6s} {'step':>4s} {'val_acc':>8s} {'delta':>8s}")
    for acc, origin, step in results[:18]:
        print(f"{origin:6d} {step:4d} {acc:8.4f} {acc-base:+8.4f}")

    print("\nbest 5:")
    for acc, origin, step in results[:5]:
        print(f"  origin={origin} step={step}: {acc:.4f}")


if __name__ == "__main__":
    main()
