"""Round 2 (v3): combine the best slot replacements from round 1 and refine.

Round 1 findings (single-slot replacements, all below the 84.92% baseline):
  best grid-slot swaps:  grid_br <- lower_right_ink (84.42%)
                         grid_bl <- band_bot        (84.18%)
  best semantic swaps:   vdom <- horiz_runs         (84.42%)
                         aspect <- upper_left_ink   (83.64%) / endpoints (83.83%)
                         aspect <- grad_vert_frac   (84.02%)

No single replacement beat the baseline.  This round tests whether PAIRS of
the top swaps combine positively, and whether re-wiring the DAG for the new
features helps.  Also test new candidate edges for 2/3/5 discrimination
(lower_right_ink / mid_right_ink parents).
"""

from helper import (
    BASELINE_PARENTS, baseline_columns, directional_stats, fit_eval,
    load_split, disc_zero_sep,
)
import numpy as np


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    B_tr, B_va, cards_base = baseline_columns(X_tr, X_va)
    d_tr = directional_stats(X_tr, "train")
    d_va = directional_stats(X_va, "val")

    NEW = ("lower_right_ink", "band_bot", "horiz_runs", "endpoints",
           "grad_vert_frac", "upper_left_ink", "mid_right_ink")
    cand = {}
    for name in NEW:
        card = {"mid_right_ink": 3}.get(name, 4)
        a, b, _ = disc_zero_sep(d_tr[name], d_va[name], card)
        cand[name] = (a.astype(np.int64), b.astype(np.int64), card)

    def run(tag, overrides, parents=BASELINE_PARENTS, alpha=1.0, backoff=True):
        c_tr, c_va, cards = [], [], list(cards_base)
        for j in range(12):
            if j in overrides:
                nm = overrides[j]
                a, b, card = cand[nm]
                c_tr.append(a); c_va.append(b); cards[j] = card
            else:
                c_tr.append(B_tr[:, j]); c_va.append(B_va[:, j])
        F_tr = np.stack(c_tr, axis=1)
        F_va = np.stack(c_va, axis=1)
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(cards), parents,
                       alpha=alpha, backoff=backoff)
        print(f"{tag:64s} val acc = {acc:.4f}")
        return acc

    # ---- pairs of top swaps
    print("--- pair swaps ---")
    run("br<-lower_right_ink + vdom<-horiz_runs",
        {7: "lower_right_ink", 11: "horiz_runs"})
    run("br<-lower_right_ink + bl<-band_bot",
        {6: "band_bot", 7: "lower_right_ink"})
    run("br<-lower_right_ink + aspect<-upper_left_ink",
        {7: "lower_right_ink", 10: "upper_left_ink"})
    run("vdom<-horiz_runs + aspect<-endpoints",
        {10: "endpoints", 11: "horiz_runs"})
    run("vdom<-horiz_runs + aspect<-grad_vert_frac",
        {10: "grad_vert_frac", 11: "horiz_runs"})
    run("bl<-band_bot + vdom<-horiz_runs",
        {6: "band_bot", 11: "horiz_runs"})

    # ---- re-wiring for the best single swap (br <- lower_right_ink):
    # parent of lower_right_ink could be middle_right (grid col 5) instead of
    # following the old bottom_right chain; and mid_right_ink as parent.
    print("--- re-wiring with br<-lower_right_ink ---")
    # indices: 0 tl,1 tc,2 tr,3 ml,4 mc,5 mr,6 bl,7 new(lower_right_ink),
    #          8 euler,9 vruns,10 aspect,11 vdom
    run("br_new, parent=mr(5)  [v2 wire else]",
        {7: "lower_right_ink"}, parents=(-1, -1, -1, 0, 2, 1, 4, 5, -1, 8, 11, -1))
    run("br_new, parent=aspect(10)",
        {7: "lower_right_ink"}, parents=(-1, -1, -1, 0, 2, 1, 4, 10, -1, 8, 11, -1))
    run("br_new, parent=vruns(9)",
        {7: "lower_right_ink"}, parents=(-1, -1, -1, 0, 2, 1, 4, 9, -1, 8, 11, -1))
    run("br_new, parent=mc(4)",
        {7: "lower_right_ink"}, parents=(-1, -1, -1, 0, 2, 1, 4, 4, -1, 8, 11, -1))
    run("br_new, parent=mr(5), drop bl chain (bl<-tl removed)",
        {7: "lower_right_ink"}, parents=(-1, -1, -1, 0, 2, 1, -1, 5, -1, 8, 11, -1))
    run("br_new, parent=mr(5), bl<-mr too",
        {7: "lower_right_ink"}, parents=(-1, -1, -1, 0, 2, 1, 5, 5, -1, 8, 11, -1))

    # ---- mid_right_ink (opening detector) as an EXTRA would be 13 features;
    # here test it replacing vdom while horiz_runs replaces aspect.
    print("--- opening feature mid_right_ink in semantic slots ---")
    run("aspect<-mid_right_ink, vdom<-horiz_runs",
        {10: "mid_right_ink", 11: "horiz_runs"})
    run("vdom<-mid_right_ink, aspect<-horiz_runs",
        {10: "horiz_runs", 11: "mid_right_ink"})
    run("bl<-mid_right_ink", {6: "mid_right_ink"})
    run("br<-mid_right_ink", {7: "mid_right_ink"})


if __name__ == "__main__":
    main()
