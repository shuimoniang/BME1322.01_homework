"""Round 2: close the gap to the 83.1% baseline and push past it.

Key finding of round 1: plain quantile discretization of grid densities
underperforms the baseline's calibrated thresholds (74% vs 83%).  The
baseline reserves a dedicated state for "exactly zero ink", which matters.
Here we test zero-separated quantile bins, add euler/vert_runs back, and
try new features (horizontal symmetry, diagonal dominance, aspect).
"""

from common import load_split, continuous_stats, fit_eval
import numpy as np


def grid_densities(X):
    imgs = X.reshape(-1, 28, 28) / 255.0
    bounds = (7, 12, 17, 22)
    outs = [imgs.mean(axis=(1, 2))]
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        for c0, c1 in zip(bounds[:-1], bounds[1:]):
            outs.append(imgs[:, r0:r1, c0:c1].mean(axis=(1, 2)))
    return np.stack(outs, axis=1)


def disc_zero_sep(v_tr, v_va, n_states):
    """State 0 = exactly zero; states 1..n-1 = quantiles of positive values."""
    pos = v_tr[v_tr > 0]
    qs = np.quantile(pos, np.linspace(0, 1, n_states)[1:-1])
    out_tr = np.searchsorted(qs, v_tr, side="right") + (v_tr > 0)
    out_va = np.searchsorted(qs, v_va, side="right") + (v_va > 0)
    return out_tr, out_va


def disc_plain(v_tr, v_va, n_states):
    qs = np.quantile(v_tr, np.linspace(0, 1, n_states + 1)[1:-1])
    return (
        np.searchsorted(qs, v_tr, side="right"),
        np.searchsorted(qs, v_va, side="right"),
    )


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    s_tr = continuous_stats(X_tr, "train")
    s_va = continuous_stats(X_va, "val")
    G_tr = grid_densities(X_tr)
    G_va = grid_densities(X_va)

    results = []

    def run(tag, cols_tr, cols_va, cards, parents, alpha=1.0):
        F_tr = np.stack(cols_tr, axis=1)
        F_va = np.stack(cols_va, axis=1)
        acc = fit_eval(
            F_tr.astype(np.int64), y_tr, F_va.astype(np.int64), y_va,
            tuple(cards), parents, alpha=alpha,
        )
        results.append((tag, acc))
        print(f"{tag:55s} val acc = {acc:.4f}")

    BASE_PARENTS = (-1, 4, 5, 6, 0, 0, 0, 4, 5, 6, -1, 10)

    # 2.1 baseline features with zero-separated quantile bins (grid 10, 5 states)
    for alpha in (0.5, 1.0):
        cols_tr, cols_va, cards = [], [], []
        for j in range(10):
            a, b = disc_zero_sep(G_tr[:, j], G_va[:, j], 5)
            cols_tr.append(a); cols_va.append(b); cards.append(5)
        run(f"2.1 grid10 zero-sep q5 alpha={alpha}", cols_tr, cols_va, cards,
            BASE_PARENTS[:10], alpha)

    # 2.2 baseline replication: grid + euler(4) + vert_runs(5)
    for alpha in (0.5, 1.0):
        cols_tr, cols_va, cards = [], [], []
        for j in range(10):
            a, b = disc_zero_sep(G_tr[:, j], G_va[:, j], 5)
            cols_tr.append(a); cols_va.append(b); cards.append(5)
        e_tr = np.clip(s_tr["euler"], -1, 2) + 1
        e_va = np.clip(s_va["euler"], -1, 2) + 1
        cols_tr.append(e_tr); cols_va.append(e_va); cards.append(4)
        a, b = disc_zero_sep(s_tr["vert_runs"], s_va["vert_runs"], 5)
        cols_tr.append(a); cols_va.append(b); cards.append(5)
        run(f"2.2 grid10+euler+vruns zero-sep alpha={alpha}", cols_tr, cols_va,
            cards, BASE_PARENTS, alpha)

    # 2.3 add horizontal-symmetry (4 states) as 13th feature, class-only parent
    cols_tr, cols_va, cards = [], [], []
    for j in range(10):
        a, b = disc_zero_sep(G_tr[:, j], G_va[:, j], 5)
        cols_tr.append(a); cols_va.append(b); cards.append(5)
    e_tr = np.clip(s_tr["euler"], -1, 2) + 1
    e_va = np.clip(s_va["euler"], -1, 2) + 1
    cols_tr.append(e_tr); cols_va.append(e_va); cards.append(4)
    a, b = disc_zero_sep(s_tr["vert_runs"], s_va["vert_runs"], 5)
    cols_tr.append(a); cols_va.append(b); cards.append(5)
    a, b = disc_zero_sep(s_tr["hsym"], s_va["hsym"], 4)
    cols_tr.append(a); cols_va.append(b); cards.append(4)
    run("2.3 +hsym(4) parent=-1 alpha=1.0", cols_tr, cols_va, cards,
        BASE_PARENTS + (-1,), 1.0)

    # 2.4 hsym with parent = global ink (feature 0): symmetry given ink level
    run("2.4 +hsym(4) parent=ink alpha=1.0", cols_tr, cols_va, cards,
        BASE_PARENTS + (0,), 1.0)

    # 2.5 swap vert_runs parent from euler(10) to hsym(12)? try edge hsym->vert_runs
    run("2.5 +hsym(4) parent=ink, vruns parent=hsym alpha=1.0", cols_tr, cols_va,
        cards, BASE_PARENTS[:11] + (12, 0), 1.0)

    # 2.6 more states on grid (try 4-state and 6-state... max 5 per constraints)
    for nst in (4, 5):
        cols_tr, cols_va, cards = [], [], []
        for j in range(10):
            a, b = disc_zero_sep(G_tr[:, j], G_va[:, j], nst)
            cols_tr.append(a); cols_va.append(b); cards.append(nst)
        e_tr = np.clip(s_tr["euler"], -1, 2) + 1
        e_va = np.clip(s_va["euler"], -1, 2) + 1
        cols_tr.append(e_tr); cols_va.append(e_va); cards.append(4)
        a, b = disc_zero_sep(s_tr["vert_runs"], s_va["vert_runs"], nst)
        cols_tr.append(a); cols_va.append(b); cards.append(nst)
        run(f"2.6 grid+euler+vruns {nst}-state alpha=1.0", cols_tr, cols_va,
            cards, BASE_PARENTS, 1.0)


if __name__ == "__main__":
    main()
