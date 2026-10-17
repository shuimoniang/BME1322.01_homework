"""Exploration F: genuinely different budgets and the grid threshold.

V1  coarse 2x2 whole-image quadrant densities + 8 semantic features
V2  6 PDF bands (rows + columns) + 6 semantic features
V3  incumbent 8-grid with the grid densities computed on a dark-core threshold
V4  incumbents grid + 4 semantic, but with horizontal-runs replacing vdom-style
Also re-tests the one positive signal (vertical_dominance K=5).
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
from _harness import evaluate, validate_structure  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP, template_incumbent, GRID_RC  # noqa: E402

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
GRID_DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP]
SEM4 = [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]


def chain_parents(keys, band_rc):
    """Reading-order chain (rsqrt by band), acyclic: each node points to its
    raster predecessor; semantic nodes chain off the last grid node."""
    order = sorted(range(len(keys)), key=lambda i: band_rc[keys[i]])
    parents = [-1] * len(keys)
    for k in range(1, len(order)):
        parents[order[k]] = order[k - 1]
    return tuple(parents)


def grid_dens_threshold(threshold):
    cache_file = CACHE / f"grid_thr{threshold}.npz"
    if cache_file.is_file():
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}
    X, y = load_train()
    images = X.reshape(-1, 28, 28).astype(np.float64) / 255.0
    binary = images > (threshold / 255.0)
    names = ("tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br")
    out = {}
    for k, (r, c) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]):
        out[f"dens_{names[k]}"] = binary[:, 7 + 5 * r:12 + 5 * r,
                                       7 + 5 * c:12 + 5 * c].mean(axis=(1, 2))
    out["label"] = y
    np.savez_compressed(cache_file, **out)
    return out


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    base, _, _, _, _ = evaluate(GRID_DESIGN + SEM4, INC_PARENTS, ctr, cva, ytr, yva)
    print(f"incumbent = {base:.4f}\n")
    results = []

    # --- V1: coarse 2x2 quadrants + 8 semantic ------------------------
    print("=== V1: 2x2 whole-image quadrant densities + 8 semantic ===")
    quad = ["dens_q_tl", "dens_q_tr", "dens_q_bl", "dens_q_br"]
    q_rc = {"dens_q_tl": (0, 0), "dens_q_tr": (0, 1), "dens_q_bl": (1, 0), "dens_q_br": (1, 1)}
    sem8 = [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
            ("vsym_aligned", 5, "quantile"), ("vertical_dominance", 4, "zero_quantile"),
            ("row_band_mid", 5, "zero_quantile"), ("horiz_runs", 5, "zero_quantile"),
            ("aspect_incumbent", 4, "zero_quantile"), ("nholes", 5, "zero_counts")]
    for sem, label in [(sem8[:4], "4 semantic"), (sem8, "8 semantic")]:
        keys = quad + [s[0] for s in sem]
        rc = dict(q_rc)
        for i, s in enumerate(sem):
            rc[s[0]] = (2 + i // 3, i % 3)  # synthetic raster order for the chain
        design = [(k, 5 if k in quad else dict((s[0], s[1]) for s in sem)[k],
                   "zero_quantile" if k in quad else dict((s[0], s[2]) for s in sem)[k])
                  for k in keys]
        parents = chain_parents(keys, rc)
        try:
            validate_structure(keys, [d[1] for d in design], parents)
            acc, _, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
            results.append((acc, f"V1 quad + {label}"))
            print(f"  {acc:.4f}  V1 quad + {label}  ({acc-base:+.4f})")
        except AssertionError as exc:
            print(f"  V1 quad + {label} invalid: {exc}")

    # --- V2: 6 bands + 6 semantic ------------------------------------
    print("\n=== V2: 6 PDF bands (rows + cols) + 6 semantic ===")
    bands = ["row_band_top", "row_band_mid", "row_band_bot",
             "col_band_left", "col_band_mid", "col_band_right"]
    b_rc = {b: (i if i < 3 else 3 + (i - 3), i % 3) for i, b in enumerate(bands)}
    sem6 = [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
            ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile"),
            ("nholes", 5, "zero_counts"), ("vsym_aligned", 5, "quantile")]
    keys = bands + [s[0] for s in sem6]
    rc = dict(b_rc)
    for i, s in enumerate(sem6):
        rc[s[0]] = (6 + i // 3, i % 3)
    design = [(b, 5, "zero_quantile") for b in bands] + sem6
    parents = chain_parents(keys, rc)
    try:
        validate_structure(keys, [d[1] for d in design], parents)
        acc, _, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
        results.append((acc, "V2 six bands + 6 semantic"))
        print(f"  {acc:.4f}  V2 six bands + 6 semantic  ({acc-base:+.4f})")
    except AssertionError as exc:
        print(f"  V2 invalid: {exc}")

    # --- V3: grid densities at other thresholds ----------------------
    print("\n=== V3: incumbent 8-grid at a dark-core threshold ===")
    for thr in (96, 128, 160, 190):
        g = grid_dens_threshold(thr)
        tr_idx, va_idx = stratified_split(np.asarray(g["label"]), 1 / 6, 1322)
        ctr2, cva2 = dict(ctr), dict(cva)
        for k in g:
            if k == "label":
                continue
            ctr2[k] = np.asarray(g[k])[tr_idx]
            cva2[k] = np.asarray(g[k])[va_idx]
        design = GRID_DESIGN + SEM4
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr2, cva2, ytr, yva)
        results.append((acc, f"V3 grid thr={thr}"))
        print(f"  {acc:.4f}  V3 grid thr={thr}  ({acc-base:+.4f})")

    # --- V4: the one positive signal ---------------------------------
    print("\n=== V4: positive-signal candidates ===")
    for label, sem in [
        ("vdom K=5", [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                      ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 5, "zero_quantile")]),
        ("vdom K=5 + aspect K=5", [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                                   ("aspect_incumbent", 5, "zero_quantile"), ("vertical_dominance", 5, "zero_quantile")]),
        ("horiz_runs K=5 replaces vdom", [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                                          ("aspect_incumbent", 4, "zero_quantile"), ("horiz_runs", 5, "zero_quantile")]),
        ("vdom K=5, vruns K=4", [("euler", 4, "euler_incumbent"), ("vertical_runs", 4, "zero_quantile"),
                                 ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 5, "zero_quantile")]),
    ]:
        design = GRID_DESIGN + sem
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr, cva, ytr, yva)
        results.append((acc, f"V4 {label}"))
        print(f"  {acc:.4f}  V4 {label}  ({acc-base:+.4f})")

    print("\n=== SUMMARY ===")
    for acc, label in sorted(results, reverse=True):
        print(f"  {acc:.4f}  {label}  ({acc-base:+.4f})")


if __name__ == "__main__":
    main()
