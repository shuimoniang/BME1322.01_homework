"""Exploration J: hole-POSITION and half-split topology (finer than euler).

`euler` and `nholes` say HOW MANY loops a digit has but not WHERE.  Upper/lower
half hole counts separate 8 (hole top AND bottom) from 9 (top hole) and 6
(bottom hole) -- directly the 8/9 confusion the incumbent's worst grid.
Also tests per-half euler, ink-row counts and diagonal-dominance.
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
from _features import _components_and_holes, build_cache, split_columns, CACHE  # noqa: E402
from _harness import evaluate, mutual_information, DISCRETISERS  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP  # noqa: E402

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
GRID8_DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP]
SEM4 = [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]


def hole_positions():
    cache_file = CACHE / "hole_pos.npz"
    if cache_file.is_file():
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}
    X, y = load_train()
    images = X.reshape(-1, 28, 28).astype(np.float64) / 255.0
    binary = images > (64 / 255.0)
    n = binary.shape[0]
    top_holes = np.zeros(n)
    bot_holes = np.zeros(n)
    from _features import _border_reachable, _count_components_runs
    reached = _border_reachable(~binary)
    hole_mask = (~binary) & (~reached)
    for i in range(n):
        h = hole_mask[i]
        if not h.any():
            continue
        rows = np.flatnonzero(h.any(axis=1))
        lo, hi = rows[0], rows[-1]
        mid = 14
        # a hole is "upper" if its top edge is in the upper half
        upper = h[:mid + 1]
        lower = h[mid:]
        top_holes[i] = _count_components_runs(upper, False) if upper.any() else 0
        bot_holes[i] = _count_components_runs(lower, False) if lower.any() else 0
    out = {"hole_top": top_holes, "hole_bot": bot_holes,
           "label": y}
    np.savez_compressed(cache_file, **out)
    return out


def extra_features():
    cache_file = CACHE / "extra_feats.npz"
    if cache_file.is_file():
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}
    X, y = load_train()
    images = X.reshape(-1, 28, 28).astype(np.float64) / 255.0
    binary = images > (64 / 255.0)
    gx = np.zeros_like(images); gy = np.zeros_like(images)
    gx[:, 1:-1, :] = images[:, 2:, :] - images[:, :-2, :]
    gy[:, :, 1:-1] = images[:, :, 2:] - images[:, :, :-2]
    diag_abs = np.abs(gx + gy).sum(axis=(1, 2))
    anti_abs = np.abs(gx - gy).sum(axis=(1, 2))
    diagonal_dominance = diag_abs / np.maximum(diag_abs + anti_abs, 1e-9)
    rows_any = binary.any(axis=2)
    ink_rows = np.count_nonzero(rows_any, axis=1).astype(np.float64)
    # vertical centroid of ink per image (normalised)
    r = np.arange(28)[None, :]
    row_mass = (binary.sum(axis=2) * r).sum(axis=1) / np.maximum(binary.sum(axis=(1, 2)), 1)
    out = {"diagonal_dominance": diagonal_dominance, "ink_rows": ink_rows,
           "centroid_row": row_mass, "label": y}
    np.savez_compressed(cache_file, **out)
    return out


def main():
    feats = build_cache()
    ctr0, ytr, cva0, yva = split_columns(feats, 1322)
    base, _, _, _, _ = evaluate(GRID8_DESIGN + SEM4, INC_PARENTS, ctr0, cva0, ytr, yva)
    print(f"incumbent = {base:.4f}\n")

    hp = hole_positions()
    ex = extra_features()
    for src in (hp, ex):
        tr_idx, va_idx = stratified_split(np.asarray(src["label"]), 1 / 6, 1322)
        for k in src:
            if k == "label":
                continue
            ctr0[f"x_{k}"] = np.asarray(src[k])[tr_idx]
            cva0[f"x_{k}"] = np.asarray(src[k])[va_idx]

    print("=== I(F;Y) of the new columns ===")
    for key in ("x_hole_top", "x_hole_bot", "x_diagonal_dominance", "x_ink_rows", "x_centroid_row"):
        for k in (3, 4, 5):
            a, _, _ = DISCRETISERS["zero_counts" if "hole" in key else "quantile"](
                ctr0[key], cva0[key], k)
            print(f"  {key:22s} K={k} I(F;Y)={mutual_information(a, ytr, k):.4f}")

    print("\n=== swap into each semantic slot ===")
    rows = []
    for key in ("x_hole_top", "x_hole_bot", "x_diagonal_dominance", "x_ink_rows", "x_centroid_row"):
        disc = "zero_counts" if "hole" in key else "quantile"
        for k in (3, 4, 5):
            for slot in (8, 9, 10, 11):
                design = GRID8_DESIGN + [list(SEM4[i]) for i in range(4)]
                design[slot] = (key, k, disc)
                design = [tuple(d) for d in design]
                acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr0, cva0, ytr, yva)
                rows.append((acc, f"slot{slot} <- {key} K={k}"))
    for acc, label in sorted(rows, reverse=True)[:12]:
        print(f"  {acc:.4f}  {label}  ({acc-base:+.4f})")

    print("\n=== 12-feature: grid8 + euler + hole_top + hole_bot + vruns ===")
    for sem in [
        [("euler", 4, "euler_incumbent"), ("x_hole_top", 4, "zero_counts"),
         ("x_hole_bot", 4, "zero_counts"), ("vertical_runs", 5, "zero_quantile")],
        [("euler", 4, "euler_incumbent"), ("x_hole_top", 3, "zero_counts"),
         ("x_hole_bot", 3, "zero_counts"), ("vertical_runs", 5, "zero_quantile")],
        [("x_hole_top", 4, "zero_counts"), ("x_hole_bot", 4, "zero_counts"),
         ("vertical_runs", 5, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")],
    ]:
        design = GRID8_DESIGN + sem
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr0, cva0, ytr, yva)
        print(f"  {acc:.4f}  {[s[0] for s in sem]}  ({acc-base:+.4f})")


if __name__ == "__main__":
    main()
