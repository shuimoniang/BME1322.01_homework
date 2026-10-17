"""Round 4: combine the winning features and sweep the combined feature sets.

Round 3 winners: +vert_dominance(4) -> 84.02%, +aspect(4) -> 83.97%,
+horiz_runs(5) -> 83.97%.  Here we try combinations (up to 12 features
total) and a few state-count variants, plus backoff smoothing on the best.
"""

from common import (
    load_split, continuous_stats, grid_densities, disc_zero_sep, fit_eval,
)
import numpy as np

BASE_PARENTS = (-1, 4, 5, 6, 0, 0, 0, 4, 5, 6, -1, 10)


def base_matrix(s_tr, s_va, G_tr, G_va):
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
    cols_tr, cols_va, cards = base_matrix(s_tr, s_va, G_tr, G_va)

    extras = {}
    for name, card in (("aspect", 4), ("vert_dominance", 4), ("horiz_runs", 5),
                       ("hsym", 4), ("top_heavy", 4)):
        a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
        extras[name] = (a.astype(np.int64), b.astype(np.int64), card)

    results = []

    def run(tag, extra_names, extra_parents, parents=BASE_PARENTS,
            alpha=1.0, backoff=False):
        c_tr, c_va, c_cards = list(cols_tr), list(cols_va), list(cards)
        full_parents = list(parents)
        for name, par in zip(extra_names, extra_parents):
            a, b, card = extras[name]
            c_tr.append(a); c_va.append(b); c_cards.append(card)
            full_parents.append(par)
        F_tr = np.stack(c_tr, axis=1).astype(np.int64)
        F_va = np.stack(c_va, axis=1).astype(np.int64)
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(c_cards),
                       tuple(full_parents), alpha=alpha, backoff=backoff)
        results.append((tag, acc))
        print(f"{tag:60s} val acc = {acc:.4f}")

    # single winners reproduced
    run("R3 +vdom(4) -1", ["vert_dominance"], [-1])
    # pairs
    run("R4 +vdom+aspect", ["vert_dominance", "aspect"], [-1, -1])
    run("R4 +vdom+horiz_runs", ["vert_dominance", "horiz_runs"], [-1, -1])
    run("R4 +aspect+horiz_runs", ["aspect", "horiz_runs"], [-1, -1])
    run("R4 +vdom+top_heavy", ["vert_dominance", "top_heavy"], [-1, 0])
    run("R4 +vdom+hsym", ["vert_dominance", "hsym"], [-1, 0])
    # triples would exceed 12 features? base 12 + 1 = 13 already. Constraint:
    # 6-12 features. Base is 12. So we can only ADD by REMOVING others.
    # Try replacing weaker grid cells with new features.
    # vdom replacing vert_runs' slot? Try: drop middle-left density (weak) etc.

    # Replace vert_runs (idx 11) with vdom:
    c_tr = cols_tr[:11] + [extras["vert_dominance"][0]]
    c_va = cols_va[:11] + [extras["vert_dominance"][1]]
    c_cards = cards[:11] + [4]
    F_tr = np.stack(c_tr, axis=1).astype(np.int64)
    F_va = np.stack(c_va, axis=1).astype(np.int64)
    acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(c_cards),
                   BASE_PARENTS[:11] + (-1,), alpha=1.0)
    results.append(("R4 vdom replaces vert_runs", acc))
    print(f"{'R4 vdom replaces vert_runs':60s} val acc = {acc:.4f}")

    # 12-feature set: drop one grid cell, add vdom + aspect
    # drop middle-left (4) density; parents adjusted
    def drop_and_add(drop_idx, add_names, add_parents, tag, parents_mod=None):
        keep = [i for i in range(10) if i != drop_idx]
        c_tr = [cols_tr[i] for i in keep] + [cols_tr[10], cols_tr[11]]
        c_va = [cols_va[i] for i in keep] + [cols_va[10], cols_va[11]]
        c_cards = [cards[i] for i in keep] + [cards[10], cards[11]]
        # rebuild parents for kept grid: original parent indices mapped
        pmap = {orig: new for new, orig in enumerate(keep + [10, 11])}
        # easier: recompute parent pointer per kept node from BASE_PARENTS
        # BASE_PARENTS indexes original 12; for kept originals only.
        new_parents = []
        for new_i, orig_i in enumerate(keep + [10, 11]):
            p = BASE_PARENTS[orig_i]
            new_parents.append(pmap.get(p, -1) if p >= 0 else -1)
        for name, par in zip(add_names, add_parents):
            a, b, card = extras[name]
            c_tr.append(a); c_va.append(b); c_cards.append(card)
            # parent refers to position in new indexing
            new_parents.append(par)
        F2_tr = np.stack(c_tr, axis=1).astype(np.int64)
        F2_va = np.stack(c_va, axis=1).astype(np.int64)
        acc = fit_eval(F2_tr, y_tr, F2_va, y_va, tuple(c_cards),
                       tuple(new_parents), alpha=1.0)
        results.append((tag, acc))
        print(f"{tag:60s} val acc = {acc:.4f}")

    # parents for added features: -1 = class only; 0 in new index = first kept
    # grid cell (global ink) if we keep idx 0. keep list starts with 0 unless
    # dropped.
    # drop middle-left (orig 4): new order = [0,1,2,3,5,6,7,8,9,10,11] then adds
    #   new idx: 0=ink,1..8=grid(tl,tr,tc,ml? no) ... let's just use class parent
    drop_and_add(4, ["vert_dominance", "aspect"], [-1, -1],
                 "R4 drop ml-density + vdom+aspect")
    drop_and_add(5, ["vert_dominance", "aspect"], [-1, 0],
                 "R4 drop center-density + vdom+aspect")
    drop_and_add(9, ["vert_dominance", "aspect"], [-1, 0],
                 "R4 drop br-density + vdom+aspect")
    drop_and_add(8, ["vert_dominance", "aspect"], [-1, 0],
                 "R4 drop bc-density + vdom+aspect")

    # best single so far + backoff smoothing
    run("R4 +vdom backoff a=1", ["vert_dominance"], [-1], backoff=True)
    run("R4 +vdom alpha=0.5", ["vert_dominance"], [-1], alpha=0.5)
    run("R4 +vdom alpha=2.0", ["vert_dominance"], [-1], alpha=2.0)


if __name__ == "__main__":
    main()
