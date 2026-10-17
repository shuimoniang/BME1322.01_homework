"""Round 5: the 12-feature constraint requires dropping base features to fit
more semantic ones.  Winners so far:
  - +vdom+aspect (13 features, exceeds constraint) -> 84.69
  - +aspect+horiz_runs -> 84.66
Round 5: build a legal 12-feature set around the strongest new features.

Also probe: hsym/vdom state counts, interaction edges for the new features,
and whether replacing vert_runs or euler helps when aspect is present.
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
    extra = {
        "euler": (e_tr.astype(np.int64), e_va.astype(np.int64), 4),
    }
    for name, card in (("vert_runs", 5), ("aspect", 4), ("vert_dominance", 4),
                       ("horiz_runs", 5), ("hsym", 4), ("top_heavy", 4),
                       ("rot_sym", 4)):
        a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
        extra[name] = (a.astype(np.int64), b.astype(np.int64), card)

    results = []

    def run(tag, names, parents, cards_override=None, alpha=1.0, backoff=False):
        # parents: full parent list for the 10 grid features followed by names
        c_tr = [cols_tr[i] for i in range(10)]
        c_va = [cols_va[i] for i in range(10)]
        c_cards = list(cards[:10])
        grid_parents = parents[:10]
        sem_parents = parents[10:10 + len(names)]
        for name in names:
            a, b, card = extra[name]
            c_tr.append(a); c_va.append(b); c_cards.append(card)
        F_tr = np.stack(c_tr, axis=1).astype(np.int64)
        F_va = np.stack(c_va, axis=1).astype(np.int64)
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(c_cards),
                       tuple(grid_parents) + tuple(sem_parents),
                       alpha=alpha, backoff=backoff)
        results.append((tag, acc))
        print(f"{tag:62s} val acc = {acc:.4f}")
        return acc

    # 12-feature candidates (10 grid + 2 semantics); parents = 10 grid + 2 sem
    GRID10 = BASE_PARENTS[:10]
    run("5.1 grid10+vdom+aspect [-1,-1]",
        ["vert_dominance", "aspect"], GRID10 + (-1, -1))
    run("5.2 grid10+aspect+horiz_runs [-1,-1]",
        ["aspect", "horiz_runs"], GRID10 + (-1, -1))
    run("5.3 grid10+vdom+horiz_runs [-1,-1]",
        ["vert_dominance", "horiz_runs"], GRID10 + (-1, -1))
    run("5.4 grid10+aspect+vruns(orig)",
        ["aspect", "vert_runs"], GRID10 + (-1, -1))
    run("5.5 grid10+aspect+euler",
        ["aspect", "euler"], GRID10 + (-1, -1))
    # edges among semantics: vdom<-aspect (tall strokes -> vertical dominance)
    run("5.6 grid10+aspect->vdom",
        ["vert_dominance", "aspect"], GRID10 + (11, -1))
    run("5.7 grid10+aspect->horiz_runs",
        ["horiz_runs", "aspect"], GRID10 + (11, -1))
    # grid parent for semantics: ink(0) -> aspect
    run("5.8 grid10+aspect(parent=ink)+vdom(-1)",
        ["vert_dominance", "aspect"], GRID10 + (-1, 0))

    # drop weakest grid cells to allow 3 semantics (still 12):
    # keep 8 grid cells (drop tc(2) & ml(4)), add vdom, aspect, horiz_runs
    def grid_kit(drop_indices, names, parents):
        keep = [i for i in range(10) if i not in drop_indices]
        pmap = {orig: new for new, orig in enumerate(keep)}
        new_parents = []
        for orig_i in keep:
            p = BASE_PARENTS[orig_i]
            new_parents.append(pmap.get(p, -1) if p >= 0 else -1)
        c_tr = [cols_tr[i] for i in keep]
        c_va = [cols_va[i] for i in keep]
        c_cards = [cards[i] for i in keep]
        for name in names:
            a, b, card = extra[name]
            c_tr.append(a); c_va.append(b); c_cards.append(card)
        new_parents = new_parents + list(parents)
        F_tr = np.stack(c_tr, axis=1).astype(np.int64)
        F_va = np.stack(c_va, axis=1).astype(np.int64)
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(c_cards),
                       tuple(new_parents), alpha=1.0)
        results.append((f"grid8+{'+'.join(names)}", acc))
        print(f"{'grid8-drop' + str(drop_indices) + '+' + '+'.join(names):62s}"
              f" val acc = {acc:.4f}")

    grid_kit([2, 4], ["vert_dominance", "aspect", "horiz_runs"], [-1, -1, -1])
    grid_kit([1, 3], ["vert_dominance", "aspect", "horiz_runs"], [-1, -1, -1])
    grid_kit([2, 4], ["vert_dominance", "aspect"], [-1, -1])
    grid_kit([1, 3], ["vert_dominance", "aspect"], [-1, -1])

    # state-count variants for the best pair
    for vd_card in (3, 5):
        a, b, _ = disc_zero_sep(s_tr["vert_dominance"], s_va["vert_dominance"], vd_card)
        extra2 = dict(extra)
        extra2["vert_dominance"] = (a.astype(np.int64), b.astype(np.int64), vd_card)
        c_tr = [cols_tr[i] for i in range(10)] + [
            extra2["vert_dominance"][0], extra["aspect"][0]]
        c_va = [cols_va[i] for i in range(10)] + [
            extra2["vert_dominance"][1], extra["aspect"][1]]
        c_cards = cards[:10] + [vd_card, 4]
        F_tr = np.stack(c_tr, axis=1).astype(np.int64)
        F_va = np.stack(c_va, axis=1).astype(np.int64)
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(c_cards),
                       (-1, 4, 5, 6, 0, 0, 0, 4, 5, 6, -1, -1), alpha=1.0)
        results.append((f"5.9 vdom card={vd_card}+aspect", acc))
        print(f"{'5.9 grid10+vdom(card' + str(vd_card) + ')+aspect':62s}"
              f" val acc = {acc:.4f}")


if __name__ == "__main__":
    main()
