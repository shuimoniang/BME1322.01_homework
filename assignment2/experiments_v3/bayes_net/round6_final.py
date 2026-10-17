"""Round 6 (v3): final combination, alpha sweep, and cross-seed robustness.

Leading candidates (round 5):
  C: row chain (tc<-tl, mc<-ml, tr<-tc, mr<-mc, br<-mr) + bl<-br
     + vruns<-euler, vdom<-aspect                    -> 85.96%
  A: same chain without bl<-br                        -> 85.87%
  rowchain + vruns<-euler, vdom<-euler, aspect<-vruns -> 85.95%
This round: combine C with the best semantic wiring, sweep alpha, and check
seeds 7 / 2024 to confirm robustness before finalizing.
"""

from helper import baseline_columns, fit_eval, load_split
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from main import find_data_dir, load_mnist, stratified_split  # noqa: E402

# candidate parent tuples (tl,tc,tr,ml,mc,mr,bl,br,euler,vruns,aspect,vdom)
P_C = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 11, -1)     # C: chain + bl<-br
P_A = (-1, 0, 1, -1, 3, 4, -1, 5, -1, 8, 11, -1)    # A: chain only
P_C2 = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)      # C + vdom<-vruns, aspect<-vruns


def eval_full(seed, parents, alpha, backoff=True):
    X_all, y_all, _, _ = load_mnist(find_data_dir())
    tr_idx, va_idx = stratified_split(y_all, seed=seed)
    # compute features on the whole set but with train-only thresholds
    F_all, cards = _features_all(X_all, tr_idx)
    return fit_eval(F_all[tr_idx], y_all[tr_idx], F_all[va_idx],
                    y_all[va_idx], cards, parents, alpha=alpha, backoff=backoff)


def _features_all(X_all, tr_idx):
    """Recompute the v2 12 features with zero-sep thresholds from tr_idx."""
    from common import grid_densities, continuous_stats
    G = grid_densities(X_all)
    s = continuous_stats(X_all, "all")
    cols, cards = [], []
    for j in (1, 2, 3, 4, 5, 6, 7, 9):
        pos = G[tr_idx, j][G[tr_idx, j] > 0]
        qs = np.quantile(pos, np.linspace(0, 1, 5)[1:-1])
        v = np.searchsorted(qs, G[:, j], side="right") + (G[:, j] > 0)
        cols.append(v); cards.append(5)
    e = np.clip(s["euler"], -1, 2) + 1
    cols.append(e.astype(np.int64)); cards.append(4)
    for name, card in (("vert_runs", 5), ("aspect", 4), ("vert_dominance", 4)):
        x = s[name]
        pos = x[tr_idx][x[tr_idx] > 0]
        qs = np.quantile(pos, np.linspace(0, 1, card)[1:-1])
        v = np.searchsorted(qs, x, side="right") + (x > 0)
        cols.append(v.astype(np.int64)); cards.append(card)
    return np.stack(cols, axis=1), tuple(cards)


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    B_tr, B_va, cards = baseline_columns(X_tr, X_va)

    def run(tag, parents, alpha=1.0, backoff=True):
        acc = fit_eval(B_tr, y_tr, B_va, y_va, cards, parents,
                       alpha=alpha, backoff=backoff)
        print(f"{tag:68s} val acc = {acc:.4f}")
        return acc

    print("--- semantic wiring on chain C ---")
    run("C + vruns<-euler, vdom<-aspect (P_C)", P_C)
    run("C + vruns<-euler, vdom<-vruns, aspect<-vruns (P_C2)", P_C2)
    run("C + vruns<-euler, vdom<-euler, aspect<-vruns",
        (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 11, 9))
    run("C + vruns<-euler, vdom<-vruns, aspect<-vdom",
        (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 11))

    print("\n--- alpha sweep on P_C and P_C2 ---")
    for p, nm in ((P_C, "P_C"), (P_C2, "P_C2")):
        for a in (0.3, 0.5, 1.0, 2.0):
            run(f"{nm} alpha={a}", p, alpha=a)
        run(f"{nm} alpha=1.0 no backoff", p, backoff=False)

    print("\n--- cross-seed robustness (seeds 1322 / 7 / 2024) ---")
    for nm, p in (("P_C", P_C), ("P_C2", P_C2), ("v2-baseline", None)):
        for seed in (1322, 7, 2024):
            if p is None:
                from helper import BASELINE_PARENTS
                p_use = BASELINE_PARENTS
            else:
                p_use = p
            acc = eval_full(seed, p_use, 1.0)
            print(f"{nm:12s} seed {seed:<5d} val acc = {acc:.4f}")


if __name__ == "__main__":
    main()
