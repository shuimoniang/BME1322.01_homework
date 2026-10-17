"""Exploration E: threshold sensitivity of the PDF-named topology features.

At threshold 64 the ink mask is so fat that 94% of digits collapse to a single
4-connected component and 95% have no endpoints, so those counts carry almost
no information.  A *dark-core* mask (only clearly-inked pixels, >128) is the
classical skeleton-ish view where broken strokes (2, 4, 5) and enclosed loops
(0, 6, 8, 9) differ.  This recomputes the component / hole / endpoint /
junction counts at several thresholds and screens them again.
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
from _features import (_components_and_holes, _endpoints_junctions,  # noqa: E402
                       _symmetry_aligned, CACHE)
from _harness import (mutual_information, pairwise_mi, evaluate,  # noqa: E402
                      DISCRETISERS, validate_structure)
from _sweep2_cells import INCUMBENT_KEEP, template_incumbent  # noqa: E402

THRESHOLDS = (64, 128, 190)
INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
GRID_DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP]


def compute_topology(threshold: int) -> dict:
    cache_file = CACHE / f"topo_thr{threshold}.npz"
    if cache_file.is_file():
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}
    X, y = load_train()
    images = X.reshape(-1, 28, 28).astype(np.float64) / 255.0
    binary = images > (threshold / 255.0)
    ncomp4, ncomp8, nholes = _components_and_holes(binary)
    endpoints, junctions = _endpoints_junctions(binary)
    ink = np.count_nonzero(binary, axis=(1, 2)).astype(np.float64)
    out = {"ncomp4": ncomp4.astype(np.float64), "ncomp8": ncomp8.astype(np.float64),
           "nholes": nholes.astype(np.float64), "endpoints": endpoints,
           "junctions": junctions, "ink": ink, "label": y}
    np.savez_compressed(cache_file, **out)
    return out


def main():
    # incumbent columns for the grid part
    from _features import build_cache, split_columns
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)

    for thr in THRESHOLDS:
        topo = compute_topology(thr)
        tr_idx, va_idx = stratified_split(np.asarray(topo["label"]), 1 / 6, 1322)
        print(f"\n=== threshold {thr} ===")
        for key in ("ncomp4", "ncomp8", "nholes", "endpoints", "junctions", "ink"):
            col_tr = np.asarray(topo[key])[tr_idx]
            col_va = np.asarray(topo[key])[va_idx]
            a, _, _ = DISCRETISERS["zero_counts"](col_tr, col_va, 5)
            iy = mutual_information(a, ytr, 5)
            mean = col_tr.mean()
            frac_multi = float(np.mean(col_tr > 1))
            print(f"  {key:10s} mean={mean:6.3f} frac>1={frac_multi:5.1%}  I(F;Y)@K5={iy:.4f}")

    # Screen: swap each thr-128 topology feature into the incumbent slots
    print("\n=== swap screen at threshold 128 ===")
    topo = compute_topology(128)
    tr_idx, va_idx = stratified_split(np.asarray(topo["label"]), 1 / 6, 1322)
    base, _, _, _, _ = evaluate(
        GRID_DESIGN + [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                       ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")],
        INC_PARENTS, ctr, cva, ytr, yva)
    print(f"incumbent = {base:.4f}")

    # build a local column dict with the thr-128 topology injected
    ctr2 = dict(ctr)
    cva2 = dict(cva)
    for key in ("ncomp4", "ncomp8", "nholes", "endpoints", "junctions", "ink"):
        ctr2[f"t{key}"] = np.asarray(topo[key])[tr_idx]
        cva2[f"t{key}"] = np.asarray(topo[key])[va_idx]

    rows = []
    for key in ("ncomp4", "ncomp8", "nholes", "endpoints", "junctions", "ink"):
        for k in (3, 4, 5):
            for slot in (8, 10, 11):
                design = GRID_DESIGN + [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                                        ("aspect_incumbent", 4, "zero_quantile"),
                                        ("vertical_dominance", 4, "zero_quantile")]
                design[slot] = (f"t{key}", k, "zero_counts")
                acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr2, cva2, ytr, yva)
                rows.append((acc, f"slot{slot} <- t{key} K={k}"))
    for acc, label in sorted(rows, reverse=True)[:12]:
        print(f"  {acc:.4f}  {label}  ({acc-base:+.4f})")

    # Multi-slot: replace BOTH weak slots (euler, aspect) with thr-128 counts
    print("\n=== multi-slot thr-128 designs ===")
    combos = [
        [("tnholes", 5, "zero_counts"), ("vertical_runs", 5, "zero_quantile"),
         ("tendpoints", 5, "zero_counts"), ("vertical_dominance", 4, "zero_quantile")],
        [("tncomp4", 5, "zero_counts"), ("vertical_runs", 5, "zero_quantile"),
         ("tendpoints", 5, "zero_counts"), ("vertical_dominance", 4, "zero_quantile")],
        [("tnholes", 5, "zero_counts"), ("vertical_runs", 5, "zero_quantile"),
         ("tink", 5, "zero_counts"), ("vertical_dominance", 4, "zero_quantile")],
    ]
    for i, sem in enumerate(combos):
        try:
            design = GRID_DESIGN + sem
            validate_structure([d[0] for d in design], [d[1] for d in design], INC_PARENTS)
            acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr2, cva2, ytr, yva)
            print(f"  {acc:.4f}  combo{i} {[s[0] for s in sem]}  ({acc-base:+.4f})")
        except AssertionError as exc:
            print(f"  combo{i} invalid: {exc}")


if __name__ == "__main__":
    main()
