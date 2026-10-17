"""Round 1: feature engineering experiments.

Test several hand-designed semantic feature sets on the same DAG skeleton
(a chain per 3x3 grid), comparing against the baseline density grid.
All discretization uses training-split quantiles only.
"""

from common import load_split, continuous_stats, eval_features, discretize, fit_eval
import numpy as np


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    print("computing cached continuous stats...")
    s_tr = continuous_stats(X_tr, "train")
    s_va = continuous_stats(X_va, "val")

    results = []

    # A: baseline reconstruction from stats (ink + coarse densities not in cache;
    #    use grid densities computed directly)
    def grid_densities(X):
        imgs = X.reshape(-1, 28, 28) / 255.0
        bounds = (7, 12, 17, 22)
        outs = [imgs.mean(axis=(1, 2))]
        for r0, r1 in zip(bounds[:-1], bounds[1:]):
            for c0, c1 in zip(bounds[:-1], bounds[1:]):
                outs.append(imgs[:, r0:r1, c0:c1].mean(axis=(1, 2)))
        return np.stack(outs, axis=1)

    G_tr = grid_densities(X_tr)
    G_va = grid_densities(X_va)
    for alpha in (0.5, 1.0):
        # parent chain per column like baseline: (-1,4,5,6,0,0,0,4,5,6)
        parents = (-1, 4, 5, 6, 0, 0, 0, 4, 5, 6)
        cards = (5,) * 10
        # discretize per-column with train quantiles, zero-separated not needed here
        F_tr = np.empty_like(G_tr, dtype=np.int64)
        F_va = np.empty_like(G_va, dtype=np.int64)
        for j in range(10):
            qs = np.quantile(G_tr[:, j], np.linspace(0, 1, 6)[1:-1])
            F_tr[:, j] = np.searchsorted(qs, G_tr[:, j], side="right")
            F_va[:, j] = np.searchsorted(qs, G_va[:, j], side="right")
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(cards), parents, alpha=alpha)
        results.append((f"A density-grid 3x3 chain alpha={alpha}", acc))

    # B: semantic global features, naive (no feature edges) as sanity check
    sem_names = ["ink", "aspect", "width", "height", "top_heavy", "side_heavy",
                 "centroid_row", "centroid_col", "hsym", "vsym", "rot_sym",
                 "vert_dominance"]
    spec = {n: 4 for n in sem_names}
    acc = eval_features(s_tr, y_tr, s_va, y_va, spec, tuple([-1] * 12), alpha=1.0)
    results.append((f"B semantic-global naive alpha=1.0", acc))

    # C: semantic global features with a designed chain of dependencies
    # ink -> center_density?  Use: ink -> hsym -> vsym -> rot_sym chain etc.
    # Simple design: ink is root; aspect depends on ink; width<-aspect,
    # height<-aspect; top_heavy<-centroid_row; side_heavy<-centroid_col;
    # others class-only.
    parents_c = (-1, 2, -1, 2, 6, 7, -1, -1, 0, 8, 9, -1)  # by index in sem_names
    acc = eval_features(s_tr, y_tr, s_va, y_va, spec, parents_c, alpha=1.0)
    results.append((f"C semantic-global designed-DAG alpha=1.0", acc))

    # D: mixed — density grid (6 features, 2x3 layout) + key semantics
    # features: ink, top_heavy, side_heavy, aspect, hsym, vert_dominance
    names_d = ["ink", "top_heavy", "side_heavy", "aspect", "hsym", "vert_dominance"]
    # DAG: ink->top_heavy? no: top_heavy<-ink; side_heavy<-ink; aspect<-ink;
    # hsym<-ink; vert_dominance<-hsym
    parents_d = (-1, 0, 0, 0, 0, 4)
    for nstates in (3, 4, 5):
        spec_d = {n: nstates for n in names_d}
        acc = eval_features(s_tr, y_tr, s_va, y_va, spec_d, parents_d, alpha=1.0)
        results.append((f"D mixed6 {nstates}-state alpha=1.0", acc))

    # E: 10-feature mixed: density grid 3x3 chain + aspect + euler
    G_tr2 = np.column_stack([G_tr, s_tr["aspect"], s_tr["euler"], s_tr["hsym"]])
    G_va2 = np.column_stack([G_va, s_va["aspect"], s_va["euler"], s_va["hsym"]])
    parents_e = (-1, 4, 5, 6, 0, 0, 0, 4, 5, 6, -1, 10, -1)  # 13 features
    for card_grid, card_extra in ((4, 3), (5, 4)):
        cards = tuple([card_grid] * 10 + [card_extra, 3, card_extra])
        F_tr = np.empty((G_tr2.shape[0], 13), dtype=np.int64)
        F_va = np.empty((G_va2.shape[0], 13), dtype=np.int64)
        for j in range(13):
            qs = np.quantile(G_tr2[:, j], np.linspace(0, 1, cards[j] + 1)[1:-1])
            F_tr[:, j] = np.searchsorted(qs, G_tr2[:, j], side="right")
            F_va[:, j] = np.searchsorted(qs, G_va2[:, j], side="right")
        acc = fit_eval(F_tr, y_tr, F_va, y_va, cards, parents_e, alpha=1.0)
        results.append((f"E grid13 {card_grid}/{card_extra} alpha=1.0", acc))

    for name, acc in results:
        print(f"{name:45s} val acc = {acc:.4f}")


if __name__ == "__main__":
    main()
