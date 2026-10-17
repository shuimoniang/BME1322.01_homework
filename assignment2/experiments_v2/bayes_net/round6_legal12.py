"""Round 6: 12-feature legal sets built around the grid chain skeleton.

Round 5 lesson: euler(4 states, tied to topology) is very informative; pure
aspect/vdom swaps lose it.  The 13-feature 84.7% set was:
grid10(5) + euler(4) + vert_runs(5) + vdom(4) + aspect(4).
To get to 12 features legally, replace the two weakest grid cells with
vdom+aspect, or drop euler/vruns for aspect+vdom.  Also try multi-state
euler variants and better parent wiring for the semantic features.
"""

from common import (
    load_split, continuous_stats, grid_densities, disc_zero_sep, fit_eval,
)
import numpy as np

BASE_PARENTS = (-1, 4, 5, 6, 0, 0, 0, 4, 5, 6, -1, 10)


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    s_tr = continuous_stats(X_tr, "train")
    s_va = continuous_stats(X_va, "val")
    G_tr = grid_densities(X_tr)
    G_va = grid_densities(X_va)

    cols_tr, cols_va, cards = [], [], []
    for j in range(10):
        a, b, _ = disc_zero_sep(G_tr[:, j], G_va[:, j], 5)
        cols_tr.append(a); cols_va.append(b); cards.append(5)
    e_tr = np.clip(s_tr["euler"], -1, 2) + 1
    e_va = np.clip(s_va["euler"], -1, 2) + 1
    extra = {"euler": (e_tr.astype(np.int64), e_va.astype(np.int64), 4)}
    for name, card in (("vert_runs", 5), ("aspect", 4), ("vert_dominance", 4),
                       ("hsym", 4), ("rot_sym", 4), ("top_heavy", 4),
                       ("horiz_runs", 5)):
        a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
        extra[name] = (a.astype(np.int64), b.astype(np.int64), card)

    results = []

    def report(tag, F_tr, F_va, c_cards, parents, alpha=1.0, backoff=False):
        acc = fit_eval(F_tr.astype(np.int64), y_tr, F_va.astype(np.int64),
                       y_va, tuple(c_cards), tuple(parents), alpha=alpha,
                       backoff=backoff)
        results.append((tag, acc))
        print(f"{tag:64s} val acc = {acc:.4f}")
        return acc

    def set12(drop, names, sem_parents, grid_parents=BASE_PARENTS[:10],
              alpha=1.0, backoff=False):
        """10-grid minus `drop` indices + named extras (total 12)."""
        keep = [i for i in range(10) if i not in drop]
        pmap = {orig: new for new, orig in enumerate(keep)}
        parents = []
        for orig_i in keep:
            p = grid_parents[orig_i]
            parents.append(pmap.get(p, -1) if p >= 0 else -1)
        c_tr = [cols_tr[i] for i in keep]
        c_va = [cols_va[i] for i in keep]
        c_cards = [cards[i] for i in keep]
        for name in names:
            a, b, card = extra[name]
            c_tr.append(a); c_va.append(b); c_cards.append(card)
            parents.append(-1)
        for i, name in enumerate(names):
            if sem_parents[i] is not None:
                parents[len(keep) + i] = sem_parents[i]
        F_tr = np.stack(c_tr, axis=1)
        F_va = np.stack(c_va, axis=1)
        return report(
            f"drop{drop}+{'+'.join(names)}" + (f" a={alpha}" if alpha != 1 else "")
            + (" bk" if backoff else ""),
            F_tr, F_va, c_cards, parents, alpha, backoff)

    # 6.1 reference: full baseline 12
    set12([], ["euler", "vert_runs"], [None, 10])

    # 6.2 drop two weak grid cells, add aspect+vdom, keep euler+vruns -> 12
    # (that's 8 grid + 4 extras = 12)
    set12([2, 4], ["euler", "vert_runs", "aspect", "vert_dominance"],
          [None, None, None, None])
    set12([1, 3], ["euler", "vert_runs", "aspect", "vert_dominance"],
          [None, None, None, None])
    set12([7, 9], ["euler", "vert_runs", "aspect", "vert_dominance"],
          [None, None, None, None])
    set12([3, 9], ["euler", "vert_runs", "aspect", "vert_dominance"],
          [None, None, None, None])

    # 6.3 keep 9 grid + euler + aspect + vdom (drop vruns)
    set12([9], ["euler", "aspect", "vert_dominance"], [None, None, None])
    set12([4], ["euler", "aspect", "vert_dominance"], [None, None, None])
    set12([2], ["euler", "aspect", "vert_dominance"], [None, None, None])

    # 6.4 semantic parent wiring inside best 12
    # drop [2,4]: new idx 0..7 = grid(0,1,3,5,6,7,8,9), 8=euler, 9=vruns,
    # 10=aspect, 11=vdom
    set12([2, 4], ["euler", "vert_runs", "aspect", "vert_dominance"],
          [None, None, 0, 10])  # aspect<-ink, vdom<-aspect
    set12([2, 4], ["euler", "vert_runs", "aspect", "vert_dominance"],
          [None, None, None, 0])  # vdom<-ink
    set12([2, 4], ["euler", "vert_runs", "aspect", "vert_dominance"],
          [None, None, 0, 0])  # both <-ink

    # 6.5 alpha and backoff on the best config (placeholder: rerun best below)
    set12([2, 4], ["euler", "vert_runs", "aspect", "vert_dominance"],
          [None, None, None, None], alpha=0.5)
    set12([2, 4], ["euler", "vert_runs", "aspect", "vert_dominance"],
          [None, None, None, None], backoff=True)


if __name__ == "__main__":
    main()
