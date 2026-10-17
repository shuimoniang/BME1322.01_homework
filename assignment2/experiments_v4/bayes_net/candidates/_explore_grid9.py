"""Exploration I: the complementary budget -- MORE grid cells, FEWER semantic.

Sweeps 2 fixed 4 semantic features and shrank the grid.  Here we do the
opposite: keep all 9 central cells (including bottom-centre, which the
incumbent drops) and spend only 3 slots on semantic features.  Also tests
9-grid + 3 semantic with different semantic triples and wirings.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate, validate_structure  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP, full_parents, GRID_RC  # noqa: E402

GRID9 = ["dens_tl", "dens_tc", "dens_tr", "dens_ml", "dens_mc", "dens_mr",
         "dens_bl", "dens_bc", "dens_br"]
INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
GRID8_DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP]


def trinity_parents(cells, sem_parents):
    """Grid part via the incumbent template + the custom 3-node semantic chain."""
    grid_part = full_parents(tuple(cells), "incumbent")
    assert len(grid_part) == len(cells) + 4, "full_parents must be sliced to grid+4"
    # keep only the grid part (the first len(cells) entries)
    return tuple(grid_part[:len(cells)]) + tuple(sem_parents)


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    base, _, _, _, _ = evaluate(
        GRID8_DESIGN + [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                        ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")],
        INC_PARENTS, ctr, cva, ytr, yva)
    print(f"incumbent = {base:.4f}\n")

    by_key = {
        "euler": ("euler", 4, "euler_incumbent"),
        "vertical_runs": ("vertical_runs", 5, "zero_quantile"),
        "horiz_runs": ("horiz_runs", 5, "zero_quantile"),
        "aspect_incumbent": ("aspect_incumbent", 4, "zero_quantile"),
        "vertical_dominance": ("vertical_dominance", 4, "zero_quantile"),
        "nholes": ("nholes", 5, "zero_counts"),
        "vsym_aligned": ("vsym_aligned", 5, "quantile"),
        "hsym_aligned": ("hsym_aligned", 5, "quantile"),
        "row_band_top": ("row_band_top", 5, "zero_quantile"),
        "row_band_bot": ("row_band_bot", 5, "zero_quantile"),
        "col_band_left": ("col_band_left", 5, "zero_quantile"),
        "col_band_right": ("col_band_right", 5, "zero_quantile"),
    }
    # a semantic chain wiring: first -> second -> {third, ...}  where the first
    # two are the strongest topology/stroke features and the rest hang off them
    def wire(keys):
        # order semantics by a hand-authored priority: euler/nholes, then runs,
        # then geometry/symmetry
        prio = {"euler": 0, "nholes": 1, "vertical_runs": 2, "horiz_runs": 3,
                "vertical_dominance": 4, "aspect_incumbent": 5, "vsym_aligned": 6,
                "hsym_aligned": 7, "row_band_top": 8, "row_band_bot": 9,
                "col_band_left": 10, "col_band_right": 11}
        order = sorted(keys, key=lambda k: prio[k])
        parents = [-1] * len(keys)
        for a, b in zip(order, order[1:]):
            parents[keys.index(b)] = 9 + keys.index(a)
        return tuple(parents)

    print("=== 9-grid (all cells) + 3 semantic ===")
    results = []
    keys_all = list(by_key)
    for keys in itertools.combinations(keys_all, 3):
        design = [(c, 5, "zero_quantile") for c in GRID9] + [by_key[k] for k in keys]
        parents = trinity_parents(GRID9, wire(keys))
        try:
            validate_structure([d[0] for d in design], [d[1] for d in design], parents)
        except AssertionError:
            continue
        acc, _, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
        results.append((acc, keys, parents))

    results.sort(reverse=True)
    print(f"{len(results)} valid. top 15:")
    for acc, keys, parents in results[:15]:
        print(f"  {acc:.4f}  {keys}")

    print("\n=== 11-grid impossible (9 max) -- test 9-grid + 3 with incumbent triple ===")
    for keys in [("euler", "vertical_runs", "vertical_dominance"),
                 ("euler", "vertical_runs", "aspect_incumbent"),
                 ("euler", "vertical_runs", "nholes"),
                 ("euler", "vertical_runs", "horiz_runs")]:
        design = [(c, 5, "zero_quantile") for c in GRID9] + [by_key[k] for k in keys]
        parents = trinity_parents(GRID9, wire(keys))
        acc, _, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
        print(f"  {acc:.4f}  {keys}  ({acc-base:+.4f})")


if __name__ == "__main__":
    main()
