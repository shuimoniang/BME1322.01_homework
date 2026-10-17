"""Sweep 1: single-slot replacement screen over the incumbent DAG skeleton.

For every cached column (and a few discretisations) we substitute it into each
of the 12 incumbent slots, keeping the incumbent DAG indices as structural
placeholders.  All fits use training-only thresholds and are scored on the
seed-1322 validation split.  Output is the manual evidence for the redesign.

Runtime note: each fit is ~1-3 s, so the whole screen is a couple of minutes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate  # noqa: E402

INCUMBENT_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
INCUMBENT_SLOT = {
    0: ("dens_tl", 5, "zero_quantile"),
    1: ("dens_tc", 5, "zero_quantile"),
    2: ("dens_tr", 5, "zero_quantile"),
    3: ("dens_ml", 5, "zero_quantile"),
    4: ("dens_mc", 5, "zero_quantile"),
    5: ("dens_mr", 5, "zero_quantile"),
    6: ("dens_bl", 5, "zero_quantile"),
    7: ("dens_br", 5, "zero_quantile"),
    8: ("euler", 4, "euler_incumbent"),
    9: ("vertical_runs", 5, "zero_quantile"),
    10: ("aspect_incumbent", 4, "zero_quantile"),
    11: ("vertical_dominance", 4, "zero_quantile"),
}

SWAP_CANDIDATES = [
    ("nholes", 5, "zero_counts"), ("nholes", 4, "zero_counts"), ("nholes", 3, "zero_counts"),
    ("ncomp4", 5, "zero_counts"), ("ncomp8", 5, "zero_counts"),
    ("endpoints", 5, "zero_counts"), ("junctions", 5, "zero_counts"),
    ("horiz_runs", 5, "zero_quantile"), ("horiz_runs", 4, "zero_quantile"),
    ("hsym_aligned", 4, "quantile"), ("hsym_aligned", 5, "quantile"),
    ("vsym_aligned", 4, "quantile"), ("vsym_aligned", 5, "quantile"),
    ("row_band_top", 5, "zero_quantile"), ("row_band_bot", 5, "zero_quantile"),
    ("col_band_left", 5, "zero_quantile"), ("col_band_right", 5, "zero_quantile"),
    ("aspect", 5, "zero_quantile"), ("aspect", 4, "zero_quantile"),
    ("bbox_fill", 5, "quantile"), ("bbox_fill", 4, "quantile"),
    ("stroke_width", 4, "quantile"), ("stroke_width", 5, "quantile"),
    ("global_ink", 5, "zero_quantile"),
    ("dens_q_tl", 5, "zero_quantile"), ("dens_q_br", 5, "zero_quantile"),
]


def design_from(slot):
    return [INCUMBENT_SLOT[i] for i in range(12)]


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)

    base_acc, _, _, _, _ = evaluate(design_from(None), INCUMBENT_PARENTS, ctr, cva, ytr, yva)
    print(f"incumbent baseline = {base_acc:.4f}\n")

    rows = []
    for cand_key, k, disc in SWAP_CANDIDATES:
        for slot in range(12):
            design = design_from(None)
            design[slot] = (cand_key, k, disc)
            acc, _, _, _, _ = evaluate(design, INCUMBENT_PARENTS, ctr, cva, ytr, yva)
            rows.append((acc, cand_key, k, disc, slot))

    rows.sort(reverse=True)
    print(f"{'feature':16s} {'K':>2s} {'disc':14s} {'slot':>4s} {'val_acc':>8s} {'delta':>8s}")
    print("-" * 60)
    for acc, key, k, disc, slot in rows[:40]:
        slot_name = INCUMBENT_SLOT[slot][0]
        print(f"{key:16s} {k:2d} {disc:14s} {slot:4d}({slot_name[:6]:6s}) "
              f"{acc:8.4f} {acc-base_acc:+8.4f}")

    print("\n--- best replacement PER SLOT ---")
    for slot in range(12):
        slot_rows = [r for r in rows if r[4] == slot]
        acc, key, k, disc, s = max(slot_rows)
        slot_name = INCUMBENT_SLOT[slot][0]
        print(f"slot {slot:2d} ({slot_name:18s}) -> {key:16s} K={k} {disc:14s} "
              f"{acc:.4f} ({acc-base_acc:+.4f})")


if __name__ == "__main__":
    main()
