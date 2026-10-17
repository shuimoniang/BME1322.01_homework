"""Redundancy (pairwise MI) among the shortlisted features, train only.

A feature earns a slot only if it adds information given the others; this
matrix is the manual evidence for the budget rebalance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import build_cache, split_columns  # noqa: E402
from _harness import DISCRETISERS, mutual_information, pairwise_mi  # noqa: E402

SHORTLIST = [
    ("dens_tl", 5, "zero_quantile"),
    ("dens_tc", 5, "zero_quantile"),
    ("dens_tr", 5, "zero_quantile"),
    ("dens_ml", 5, "zero_quantile"),
    ("dens_mc", 5, "zero_quantile"),
    ("dens_mr", 5, "zero_quantile"),
    ("dens_bl", 5, "zero_quantile"),
    ("dens_br", 5, "zero_quantile"),
    ("euler", 4, "euler_incumbent"),
    ("nholes", 5, "zero_counts"),
    ("vertical_runs", 5, "zero_quantile"),
    ("horiz_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"),
    ("vertical_dominance", 4, "zero_quantile"),
    ("hsym_aligned", 5, "quantile"),
    ("vsym_aligned", 5, "quantile"),
    ("row_band_top", 5, "zero_quantile"),
    ("row_band_bot", 5, "zero_quantile"),
    ("col_band_left", 5, "zero_quantile"),
    ("col_band_right", 5, "zero_quantile"),
    ("ncomp4", 5, "zero_counts"),
]


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    states = {}
    labels = []
    for key, k, disc in SHORTLIST:
        a, _, _ = DISCRETISERS[disc](np.asarray(ctr[key], np.float64),
                                     np.asarray(cva[key], np.float64), k)
        states[key] = (a, k)
        labels.append(key)
    print("I(F;Y):")
    for key in labels:
        a, k = states[key]
        print(f"  {key:20s} {mutual_information(a, ytr, k):.4f}")
    print("\nPairwise I(Fi;Fj) -- high values mean redundancy:")
    header = "".join(f"{l[:7]:>8s}" for l in labels)
    print(f"{'':20s}{header}")
    for li in labels:
        a, ka = states[li]
        row = []
        for lj in labels:
            b, kb = states[lj]
            row.append(pairwise_mi(a, b, ka, kb))
        print(f"{li:20s}" + "".join(f"{v:8.3f}" for v in row))


if __name__ == "__main__":
    main()
