"""Conditional MI given Y + per-class profiles for the shortlist (train only).

For a naive-Bayes-like model the redundancy that HURTS is conditional on the
class:  I(Fi;Fj | Y).  Zero means the two features are conditionally
independent given the label (the "naive" assumption actually holds); large
means they double-count the same evidence.
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
    ("dens_tl", 5, "zero_quantile"), ("dens_tc", 5, "zero_quantile"),
    ("dens_tr", 5, "zero_quantile"), ("dens_ml", 5, "zero_quantile"),
    ("dens_mc", 5, "zero_quantile"), ("dens_mr", 5, "zero_quantile"),
    ("dens_bl", 5, "zero_quantile"), ("dens_br", 5, "zero_quantile"),
    ("euler", 4, "euler_incumbent"), ("nholes", 5, "zero_counts"),
    ("vertical_runs", 5, "zero_quantile"), ("horiz_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile"),
    ("hsym_aligned", 5, "quantile"), ("vsym_aligned", 5, "quantile"),
    ("row_band_top", 5, "zero_quantile"), ("row_band_bot", 5, "zero_quantile"),
    ("col_band_left", 5, "zero_quantile"), ("col_band_right", 5, "zero_quantile"),
    ("ncomp4", 5, "zero_counts"),
]


def cond_mi_given_y(a, b, y, na, nb, n_classes=10):
    """I(A;B|Y) = sum_y P(y) I(A;B|y)."""
    total = 0.0
    for c in range(n_classes):
        mask = y == c
        if mask.sum() < 2:
            continue
        weight = mask.mean()
        joint = np.zeros((na, nb), dtype=np.float64)
        np.add.at(joint, (a[mask], b[mask]), 1)
        joint /= joint.sum()
        pa = joint.sum(axis=1, keepdims=True)
        pb = joint.sum(axis=0, keepdims=True)
        inner = joint > 0
        total += weight * float(np.sum(joint[inner] * np.log(joint[inner] / (pa * pb)[inner])))
    return total


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    states, labels = {}, []
    for key, k, disc in SHORTLIST:
        a, _, _ = DISCRETISERS[disc](np.asarray(ctr[key], np.float64),
                                     np.asarray(cva[key], np.float64), k)
        states[key] = (a, k)
        labels.append(key)

    print("=== per-class mean of each continuous feature (train) ===")
    per_class = {}
    for key, _, _ in SHORTLIST:
        col = np.asarray(ctr[key], np.float64)
        means = [col[ytr == c].mean() for c in range(10)]
        per_class[key] = means
        print(f"{key:20s} " + " ".join(f"{m:7.3f}" for m in means))

    print("\n=== I(Fi;Fj|Y) conditional redundancy (high = double-counts) ===")
    print(f"{'':20s}" + "".join(f"{l[:7]:>8s}" for l in labels))
    cmi = {}
    for li in labels:
        a, ka = states[li]
        row = []
        for lj in labels:
            b, kb = states[lj]
            v = cond_mi_given_y(a, b, ytr, ka, kb)
            cmi[(li, lj)] = v
            row.append(v)
        print(f"{li:20s}" + "".join(f"{v:8.3f}" for v in row))

    print("\n=== redundancy score: I(Fi;Y) vs sum of I(Fi;Fj|Y) ===")
    for key in labels:
        iy = mutual_information(states[key][0], ytr, states[key][1])
        others = sum(v for (li, lj), v in cmi.items() if li == key and lj != key)
        print(f"{key:20s} I(F;Y)={iy:.4f}  sum_I(F;.|Y)={others:.3f}  ratio={iy/max(others,1e-9):.3f}")


if __name__ == "__main__":
    main()
