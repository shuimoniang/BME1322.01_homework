"""Training-only mutual-information screen to GUIDE (not automate) design.

For every cached candidate column we compute I(F;Y) under a few discretisation
schemes, plus I(F; dens_*) against the incumbent grid cells.  The numbers are
used as *human-readable evidence* for a manual choice of the 12 slots --
there is no search over structures here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import build_cache, split_columns  # noqa: E402
from _harness import DISCRETISERS, mutual_information, pairwise_mi  # noqa: E402

GRID = ["dens_tl", "dens_tc", "dens_tr", "dens_ml", "dens_mc", "dens_mr",
        "dens_bl", "dens_br", "dens_bc"]

CANDIDATES = [
    # incumbent slots, for reference
    ("dens_tl", 5, "zero_quantile"), ("dens_tc", 5, "zero_quantile"),
    ("dens_tr", 5, "zero_quantile"), ("dens_ml", 5, "zero_quantile"),
    ("dens_mc", 5, "zero_quantile"), ("dens_mr", 5, "zero_quantile"),
    ("dens_bl", 5, "zero_quantile"), ("dens_br", 5, "zero_quantile"),
    ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile"),
    # PDF-named unexplored family
    ("vsym_aligned", 4, "quantile"), ("vsym_aligned", 5, "quantile"),
    ("hsym_aligned", 4, "quantile"), ("hsym_aligned", 5, "quantile"),
    ("vsym_jaccard", 4, "quantile"),
    ("hsym_jaccard", 4, "quantile"),
    ("ncomp4", 4, "zero_counts"), ("ncomp4", 5, "zero_counts"),
    ("ncomp8", 4, "zero_counts"), ("ncomp8", 5, "zero_counts"),
    ("nholes", 4, "zero_counts"), ("nholes", 5, "zero_counts"),
    ("endpoints", 4, "zero_counts"), ("endpoints", 5, "zero_counts"),
    ("junctions", 4, "zero_counts"), ("junctions", 5, "zero_counts"),
    # band/geometry alternatives (PDF top/mid/bottom, left/mid/right, aspect)
    ("row_band_top", 5, "zero_quantile"), ("row_band_mid", 5, "zero_quantile"),
    ("row_band_bot", 5, "zero_quantile"),
    ("col_band_left", 5, "zero_quantile"), ("col_band_mid", 5, "zero_quantile"),
    ("col_band_right", 5, "zero_quantile"),
    ("aspect", 4, "zero_quantile"), ("aspect", 5, "zero_quantile"),
    ("bbox_fill", 4, "quantile"), ("bbox_fill", 5, "quantile"),
    ("height", 4, "quantile"), ("width", 4, "quantile"),
    ("dens_q_tl", 5, "zero_quantile"),
    # other incumbent-adjacent
    ("horiz_runs", 5, "zero_quantile"), ("stroke_width", 4, "quantile"),
    ("global_ink", 5, "zero_quantile"),
]


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)

    # discretise grid cells once for the redundancy column
    grid_states = {}
    for g in GRID:
        a, _, _ = DISCRETISERS["zero_quantile"](np.asarray(ctr[g]), np.asarray(cva[g]), 5)
        grid_states[g] = a

    print(f"{'feature':22s} {'K':>2s} {'disc':15s} {'I(F;Y)':>8s}  "
          f"{'max I(F;grid)':>13s} {'argmax':>16s}")
    print("-" * 84)
    rows = []
    for key, k, disc in CANDIDATES:
        a, _, _ = DISCRETISERS[disc](np.asarray(ctr[key], np.float64),
                                     np.asarray(cva[key], np.float64), k)
        iy = mutual_information(a, ytr, k)
        best_g, best_mi = None, -1.0
        for g in GRID:
            mb = pairwise_mi(a, grid_states[g], k, 5)
            if mb > best_mi:
                best_mi, best_g = mb, g
        rows.append((key, k, disc, iy, best_mi, best_g))
        print(f"{key:22s} {k:2d} {disc:15s} {iy:8.4f}  {best_mi:13.4f} {best_g:>16s}")

    print("\n--- highest I(F;Y) ---")
    for key, k, disc, iy, bm, bg in sorted(rows, key=lambda r: -r[3])[:20]:
        print(f"  {key:22s} K={k} {disc:15s} I={iy:.4f}  maxI_grid={bm:.4f} ({bg})")


if __name__ == "__main__":
    main()
