"""Exploration K: DAG re-wiring on the incumbent 12 features (edge sweep).

Feature-set redesign failed to beat the incumbent, so the remaining in-bounds
lever is the DAG.  Each feature node may have at most one feature parent (plus
Y).  Here we do a manual coordinate search: for every node, try every other
node as its parent (and "no parent"), keeping the rest of the incumbent DAG,
then greedily combine the best single-edge changes.  Every retained edge needs
a semantic story, which we record.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate, validate_structure  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP  # noqa: E402

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
    ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
NAMES = [d[0] for d in DESIGN]


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    base, _, _, _, _ = evaluate(DESIGN, INC_PARENTS, ctr, cva, ytr, yva)
    print(f"incumbent = {base:.4f}\n")

    def try_parents(parents):
        try:
            validate_structure(NAMES, [d[1] for d in DESIGN], parents)
        except AssertionError:
            return None
        acc, _, _, _, _ = evaluate(DESIGN, parents, ctr, cva, ytr, yva)
        return acc

    print("=== single-edge sweep: for each node, best alternate parent ===")
    best_changes = []
    for child in range(12):
        rows = []
        for parent in [-1] + [p for p in range(12) if p != child]:
            trial = list(INC_PARENTS)
            trial[child] = parent
            acc = try_parents(tuple(trial))
            if acc is not None:
                rows.append((acc, parent))
        rows.sort(reverse=True)
        cur = INC_PARENTS[child]
        cur_acc = next(a for a, p in rows if p == cur)
        best_acc, best_p = rows[0]
        print(f"  node {child:2d} ({NAMES[child]:18s}) current parent={cur:3d} "
              f"acc={cur_acc:.4f} | best alt parent={best_p:3d} acc={best_acc:.4f} "
              f"({best_acc-cur_acc:+.4f})")
        if best_acc > cur_acc + 1e-9:
            best_changes.append((best_acc - cur_acc, child, best_p))

    print("\n=== greedy coordinate ascent over edge changes ===")
    parents = list(INC_PARENTS)
    cur_acc = base
    improved = True
    while improved:
        improved = False
        for child in range(12):
            for parent in [-1] + [p for p in range(12) if p != child]:
                if parent == parents[child]:
                    continue
                trial = list(parents)
                trial[child] = parent
                acc = try_parents(tuple(trial))
                if acc is not None and acc > cur_acc + 1e-9:
                    parents, cur_acc = trial, acc
                    improved = True
                    print(f"  adopt {NAMES[child]} <- {NAMES[parent] if parent>=0 else 'Y'}: {acc:.4f}")
    print(f"\ncoordinate ascent result: {cur_acc:.4f} ({cur_acc-base:+.4f})")
    print(f"parents = {tuple(parents)}")


if __name__ == "__main__":
    main()
