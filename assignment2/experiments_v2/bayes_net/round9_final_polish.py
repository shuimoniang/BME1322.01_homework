"""Round 9: final polish on the leading config (84.97-84.98%).

Combos: col-reversed chain + semantic wiring + backoff + alpha; also test
cardinality of vdom, edge tc<-mc vs mc<-tc etc.  Also verify a couple of
seeds of the split for the top-2 configs to confirm robustness (still
training-only selection).
"""

from common import (
    load_split, continuous_stats, grid_densities, disc_zero_sep, fit_eval,
)
import numpy as np

NAMES = ("euler", "vert_runs", "vert_dominance", "aspect")
# grid order: tl(0),tr(1),tc(2),ml(3),mc(4),mr(5),bl(6),bc(7),br(8)
GRID_A = (-1, 4, 5, -1, -1, -1, 3, 4, 5)   # original column chain, ink dropped
GRID_B = (-1, -1, -1, 0, 2, 1, 3, 4, 5)    # col-reversed: cell <- above
GRID_C = (-1, 4, 5, 0, 2, 1, 3, 4, 5)      # mixed


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    s_tr = continuous_stats(X_tr, "train")
    s_va = continuous_stats(X_va, "val")
    G_tr = grid_densities(X_tr)
    G_va = grid_densities(X_va)

    cols_tr, cols_va, cards = [], [], []
    for j in range(1, 10):
        a, b, _ = disc_zero_sep(G_tr[:, j], G_va[:, j], 5)
        cols_tr.append(a); cols_va.append(b); cards.append(5)
    e_tr = np.clip(s_tr["euler"], -1, 2) + 1
    e_va = np.clip(s_va["euler"], -1, 2) + 1
    extra = {"euler": (e_tr.astype(np.int64), e_va.astype(np.int64), 4)}
    for name, card in (("vert_runs", 5), ("aspect", 4), ("vert_dominance", 4)):
        a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
        extra[name] = (a.astype(np.int64), b.astype(np.int64), card)

    def build(grid_parents, sem_parents, alpha=1.0, backoff=False,
              use_split=None):
        c_tr, c_va, c_cards = list(cols_tr), list(cols_va), list(cards)
        for name in NAMES:
            a, b, card = extra[name]
            c_tr.append(a); c_va.append(b); c_cards.append(card)
        parents = list(grid_parents) + list(sem_parents)
        F_tr = np.stack(c_tr, axis=1).astype(np.int64)
        F_va = np.stack(c_va, axis=1).astype(np.int64)
        if use_split is None:
            return fit_eval(F_tr, y_tr, F_va, y_va, tuple(c_cards),
                            tuple(parents), alpha=alpha, backoff=backoff)
        tr_idx, va_idx, X, y = use_split
        return fit_eval(F_tr[tr_idx], y[tr_idx], F_va[va_idx], y[va_idx],
                        tuple(c_cards), tuple(parents), alpha=alpha,
                        backoff=backoff)

    results = []

    def go(tag, *args, **kwargs):
        acc = build(*args, **kwargs)
        results.append((tag, acc))
        print(f"{tag:66s} val acc = {acc:.4f}")
        return acc

    # grid A (col chain, ink dropped) with best wiring from round 8
    go("9.1 A + aspect<-vruns, vdom<-aspect", GRID_A, (-1, -1, 12, 10))
    go("9.2 A + aspect<-vruns, vdom<-aspect bk", GRID_A, (-1, -1, 12, 10),
       backoff=True)
    go("9.3 A + aspect<-vruns, vdom<-aspect bk a=0.5", GRID_A,
       (-1, -1, 12, 10), alpha=0.5, backoff=True)
    # grid B with same wiring
    go("9.4 B + aspect<-vruns, vdom<-aspect", GRID_B, (-1, -1, 12, 10))
    go("9.5 B class-only semantics", GRID_B, (-1, -1, -1, -1))
    go("9.6 B + vruns<-euler, vdom<-aspect", GRID_B, (-1, 9, 12, -1))
    go("9.7 B + vruns<-euler, vdom<-aspect bk", GRID_B, (-1, 9, 12, -1),
       backoff=True)
    go("9.8 B bk a=0.5", GRID_B, (-1, 9, 12, -1), alpha=0.5, backoff=True)
    go("9.9 C + vruns<-euler, vdom<-aspect", GRID_C, (-1, 9, 12, -1))

    # robustness check on the top-2 with a second split seed (1322 vs 7)
    from main import stratified_split
    data_dir_x = None
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    from main import find_data_dir, load_mnist
    X_all, y_all, _, _ = load_mnist(find_data_dir())
    imgs = X_all
    s_all = continuous_stats(X_all, "all")
    G_all = grid_densities(X_all)
    cols_all = []
    for j in range(1, 10):
        _, b, _ = disc_zero_sep(G_all[:, j], G_all[:, j], 5)  # dummy
        cols_all.append(G_all[:, j])
    e_all = np.clip(s_all["euler"], -1, 2) + 1
    extra_all = {"euler": e_all}
    for name in ("vert_runs", "aspect", "vert_dominance"):
        extra_all[name] = s_all[name]

    def eval_all_seed(seed, grid_parents, sem_parents, alpha=1.0,
                      backoff=False):
        tr_idx, va_idx = stratified_split(y_all, seed=seed)
        F_all = []
        cards_l = []
        for j in range(1, 10):
            v_tr, _, qs = disc_zero_sep(G_all[tr_idx, j], G_all[tr_idx, j], 5)
            v = np.searchsorted(qs, G_all[:, j], side="right") + (G_all[:, j] > 0)
            F_all.append(v); cards_l.append(5)
        v = np.clip(s_all["euler"], -1, 2) + 1
        F_all.append(v); cards_l.append(4)
        for name, card in (("vert_runs", 5), ("aspect", 4),
                           ("vert_dominance", 4)):
            pos = s_all[name][tr_idx]
            pos = pos[pos > 0]
            qs = np.quantile(pos, np.linspace(0, 1, card)[1:-1])
            v = np.searchsorted(qs, s_all[name], side="right") + (
                s_all[name] > 0)
            F_all.append(v); cards_l.append(card)
        F_all = np.stack(F_all, axis=1).astype(np.int64)
        parents = list(grid_parents) + list(sem_parents)
        return fit_eval(F_all[tr_idx], y_all[tr_idx], F_all[va_idx],
                        y_all[va_idx], tuple(cards_l), tuple(parents),
                        alpha=alpha, backoff=backoff)

    for seed in (1322, 7, 2024):
        a1 = eval_all_seed(seed, GRID_A, (-1, -1, 12, 10))
        a2 = eval_all_seed(seed, GRID_B, (-1, 9, 12, -1), backoff=True)
        print(f"seed {seed}: A+wiring={a1:.4f}  B+wiring+bk={a2:.4f}")


if __name__ == "__main__":
    main()
