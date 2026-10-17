"""Round 7: the winning 13-feature configuration was 84.69%.  The 12-feature
cap means we must drop ONE base feature.  Round 6 shows dropping grid cells
hurts a lot, so drop the weakest base semantic: vert_runs or euler, or try
weaker base = grid cell with the most redundancy (bc).  From round 4:
  drop bc(8) + vdom + aspect with euler+vruns kept = 84.40!  Wait that was
  'R4 drop bc-density' = drop[8] + euler+vruns kept + vdom + aspect -> 84.40.
Reproduce that precisely, wire parents, tune alpha, and compare with
drop[9](br) variant (83.82).
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
                       ("hsym", 4), ("rot_sym", 4)):
        a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
        extra[name] = (a.astype(np.int64), b.astype(np.int64), card)

    results = []

    def set12(drop, names, sem_spec, alpha=1.0, backoff=False, tag=None):
        """drop: grid indices to drop; names: extras; sem_spec: dict
        name->parent index (into final ordering, None=class only)."""
        keep = [i for i in range(10) if i not in drop]
        pmap = {orig: new for new, orig in enumerate(keep)}
        parents = []
        for orig_i in keep:
            p = BASE_PARENTS[orig_i]
            parents.append(pmap.get(p, -1) if p >= 0 else -1)
        c_tr = [cols_tr[i] for i in keep]
        c_va = [cols_va[i] for i in keep]
        c_cards = [cards[i] for i in keep]
        for name in names:
            a, b, card = extra[name]
            c_tr.append(a); c_va.append(b); c_cards.append(card)
            parents.append(-1)
        for name, par in sem_spec.items():
            if par is not None:
                parents[len(keep) + names.index(name)] = par
        F_tr = np.stack(c_tr, axis=1).astype(np.int64)
        F_va = np.stack(c_va, axis=1).astype(np.int64)
        label = tag or f"drop{drop}+{'+'.join(names)}"
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(c_cards), tuple(parents),
                       alpha=alpha, backoff=backoff)
        results.append((label, acc))
        print(f"{label:66s} val acc = {acc:.4f}")
        return acc, parents, c_cards, F_tr, F_va

    # reproduce R4 winners as legal 12-feature sets (drop 1 grid cell)
    set12([8], ["euler", "vert_runs", "vert_dominance", "aspect"], {})
    set12([9], ["euler", "vert_runs", "vert_dominance", "aspect"], {})
    set12([7], ["euler", "vert_runs", "vert_dominance", "aspect"], {})
    set12([3], ["euler", "vert_runs", "vert_dominance", "aspect"], {})
    set12([6], ["euler", "vert_runs", "vert_dominance", "aspect"], {})
    set12([1], ["euler", "vert_runs", "vert_dominance", "aspect"], {})
    set12([5], ["euler", "vert_runs", "vert_dominance", "aspect"], {})
    set12([0], ["euler", "vert_runs", "vert_dominance", "aspect"], {})


if __name__ == "__main__":
    main()
