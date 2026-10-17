"""Sweep 3: which 4 semantic features fill the 4 non-grid slots?

The 8-cell grid + row-chain DAG is a proven local optimum (sweep 2), so the
only free budget is the 4 semantic slots.  This sweep keeps the 8-grid bundle
fixed and searches WHICH 4 semantic features go into slots 8..11, under two
hand-authored wirings, then refines K.  Fixed-K first pass, then K refinement.
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
from _sweep2_cells import INCUMBENT_KEEP, template_incumbent  # noqa: E402

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5) + (-1, 8, 9, 9)
GRID_DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP]

# Candidate semantic features: (key, K, discretiser, priority, semantic role)
# Priority is a HAND-AUTHORED "topology -> stroke dynamics -> global shape"
# order used only to build acyclic chains with a defensible story.
SEMANTIC = [
    ("euler", 4, "euler_incumbent", 0, "holes - components (digital topology)"),
    ("nholes", 5, "zero_counts", 1, "number of enclosed holes"),
    ("ncomp4", 5, "zero_counts", 2, "4-connected ink components"),
    ("ncomp8", 5, "zero_counts", 3, "8-connected ink components"),
    ("vertical_runs", 5, "zero_quantile", 4, "vertical stroke fragmentation"),
    ("horiz_runs", 5, "zero_quantile", 5, "horizontal stroke fragmentation"),
    ("vsym_aligned", 5, "quantile", 6, "left-right symmetry"),
    ("hsym_aligned", 5, "quantile", 7, "top-bottom symmetry"),
    ("aspect_incumbent", 4, "zero_quantile", 8, "width/height proportion"),
    ("vertical_dominance", 4, "zero_quantile", 9, "vertical gradient dominance"),
    ("row_band_top", 5, "zero_quantile", 10, "upper-third ink density"),
    ("row_band_bot", 5, "zero_quantile", 11, "lower-third ink density"),
    ("bbox_fill", 5, "quantile", 12, "ink fill of the bounding box"),
    ("stroke_width", 5, "quantile", 13, "mean stroke thickness"),
]
INCUMBENT_SEM_KEY = ("euler", "vertical_runs", "aspect_incumbent", "vertical_dominance")


def wire_priority_chain(keys, meta, offset=8):
    """Each semantic node's parent = the previous semantic in priority order."""
    order = sorted(keys, key=lambda k: meta[k][0])
    parents = [-1] * len(keys)
    for a, b in zip(order, order[1:]):
        parents[keys.index(b)] = offset + keys.index(a)
    return tuple(parents)


def wire_hub(keys, meta, offset=8):
    """The highest-priority (most topological) feature parents all the rest."""
    hub = min(keys, key=lambda k: meta[k][0])
    return tuple(-1 if k == hub else offset + keys.index(hub) for k in keys)


WIRINGS = {"chain": wire_priority_chain, "hub": wire_hub}


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    meta = {k: (prio, role) for k, kk, dd, prio, role in SEMANTIC}
    by_key = {k: (k, kk, dd) for k, kk, dd, _, _ in SEMANTIC}

    base, _, _, _, _ = evaluate(GRID_DESIGN + [by_key[k] for k in INCUMBENT_SEM_KEY],
                                INC_PARENTS, ctr, cva, ytr, yva)
    print(f"incumbent = {base:.4f}\n")

    results = []
    keys_all = [k for k, *_ in SEMANTIC]
    for keys in itertools.combinations(keys_all, 4):
        design = GRID_DESIGN + [by_key[k] for k in keys]
        for wname, wiring in WIRINGS.items():
            sem_parents = wiring(list(keys), meta, offset=8)
            parents = template_incumbent(INCUMBENT_KEEP) + sem_parents
            names = [d[0] for d in design]
            cards = [d[1] for d in design]
            try:
                validate_structure(names, cards, parents)
            except AssertionError:
                continue
            acc, _, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
            results.append((acc, keys, wname, parents))

    results.sort(reverse=True)
    print(f"{len(results)} valid designs. Top 25:")
    for acc, keys, wname, parents in results[:25]:
        print(f"  {acc:.4f} {wname:6s} {keys}")

    print("\n--- best per wiring ---")
    for wname in WIRINGS:
        rows = [r for r in results if r[2] == wname]
        acc, keys, w, parents = max(rows)
        print(f"  {wname:6s} {acc:.4f} {keys}")
        print(f"          parents={parents}")

    np.save(HERE / "cache" / "sweep3_results.npy",
            np.array([(a, k, w) for a, k, w, _ in results], dtype=object),
            allow_pickle=True)


if __name__ == "__main__":
    main()
