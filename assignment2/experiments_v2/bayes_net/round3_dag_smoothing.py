"""Round 3: DAG redesign + adaptive smoothing on the round-2 best feature set.

Feature set R (12 features):
  0 ink, 1..9 grid densities, 10 euler, 11 vert_runs
Baseline DAG: chain per column (-1,4,5,6,0,0,0,4,5,6,-1,10)

Experiments:
  A  baseline DAG alpha sweep + backoff smoothing
  B  alternative edge designs (row-major chain, hub-and-spoke, mirrored pairs)
  C  adaptive (parent-count-dependent) smoothing
  D  extra semantic feature variants (aspect, hsym placed differently)
"""

from common import (
    load_split, continuous_stats, grid_densities, disc_zero_sep, fit_eval,
)
import numpy as np


def build_columns(s_tr, s_va, G_tr, G_va, extra=()):
    """Return list of (col_tr, col_va, card) for the 12 base features."""
    cols_tr, cols_va, cards = [], [], []
    for j in range(10):
        a, b, _ = disc_zero_sep(G_tr[:, j], G_va[:, j], 5)
        cols_tr.append(a); cols_va.append(b); cards.append(5)
    e_tr = np.clip(s_tr["euler"], -1, 2) + 1
    e_va = np.clip(s_va["euler"], -1, 2) + 1
    cols_tr.append(e_tr); cols_va.append(e_va); cards.append(4)
    a, b, _ = disc_zero_sep(s_tr["vert_runs"], s_va["vert_runs"], 5)
    cols_tr.append(a); cols_va.append(b); cards.append(5)
    return cols_tr, cols_va, cards


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    s_tr = continuous_stats(X_tr, "train")
    s_va = continuous_stats(X_va, "val")
    G_tr = grid_densities(X_tr)
    G_va = grid_densities(X_va)
    cols_tr, cols_va, cards = build_columns(s_tr, s_va, G_tr, G_va)
    F_tr = np.stack(cols_tr, axis=1).astype(np.int64)
    F_va = np.stack(cols_va, axis=1).astype(np.int64)

    results = []

    def run(tag, parents, alpha=1.0, backoff=False):
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(cards), parents,
                       alpha=alpha, backoff=backoff)
        results.append((tag, acc))
        print(f"{tag:58s} val acc = {acc:.4f}")

    BASE = (-1, 4, 5, 6, 0, 0, 0, 4, 5, 6, -1, 10)

    # A: alpha sweep + backoff on baseline DAG
    for alpha in (0.1, 0.5, 1.0, 2.0, 4.0):
        run(f"A baseline DAG alpha={alpha}", BASE, alpha)
    for alpha in (0.5, 1.0, 2.0):
        run(f"A baseline DAG backoff alpha={alpha}", BASE, alpha, backoff=True)

    # B: alternative DAGs
    # B1 row-major snake chain: ink->each row start, within-row chains
    b1 = (-1, 4, 1, 2, 0, 3, 4, 5, 1, 7, -1, 10)
    run("B1 snake chain", b1)
    # B2 mirrored-pair design: left cell depends on right cell (symmetry priors)
    b2 = (-1, 2, -1, 2, 5, -1, 5, 8, -1, 8, -1, 10)
    run("B2 mirror pairs", b2)
    # B3 diagonal chain (ink -> tl -> center -> br etc.)
    b3 = (-1, 4, 5, -1, 0, 2, -1, 1, 5, -1, -1, 10)
    run("B3 diagonal deps", b3)
    # B4 euler -> vert_runs kept, plus ink -> euler
    b4 = (-1, 4, 5, 6, 0, 0, 0, 4, 5, 6, 0, 10)
    run("B4 +ink->euler", b4)
    # B5 center (5) as hub: tl,tr,bl,br depend on center
    b5 = (-1, 5, 6, 5, 0, 0, 0, 5, 6, 5, -1, 10)
    run("B5 center hub", b5)

    # C: adaptive smoothing handled via backoff already; try per-cardinality alpha

    # D: feature swaps — replace vert_runs with aspect; hsym with vdom
    def extra_feature(tag, name, card, parent):
        a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
        F2_tr = np.column_stack([F_tr, a.astype(np.int64)])
        F2_va = np.column_stack([F_va, b.astype(np.int64)])
        acc = fit_eval(F2_tr, y_tr, F2_va, y_va, tuple(cards + [card]),
                       BASE + (parent,), alpha=1.0)
        results.append((tag, acc))
        print(f"{tag:58s} val acc = {acc:.4f}")

    extra_feature("D +aspect(4) parent=ink", "aspect", 4, 0)
    extra_feature("D +aspect(4) parent=-1", "aspect", 4, -1)
    extra_feature("D +hsym(4) parent=ink", "hsym", 4, 0)
    extra_feature("D +vert_dominance(4) parent=-1", "vert_dominance", 4, -1)
    extra_feature("D +horiz_runs(5) parent=-1", "horiz_runs", 5, -1)
    extra_feature("D +top_heavy(4) parent=ink", "top_heavy", 4, 0)


if __name__ == "__main__":
    main()
