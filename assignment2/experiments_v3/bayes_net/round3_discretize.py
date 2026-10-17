"""Round 3 (v3): discretization refinements on the strongest v3 direction.

Round 2's best v3 variant was "br <- lower_right_ink, bl <- mr too"
(84.67%) — still 0.25pt below the 84.92% baseline.  Before abandoning
directional features, this round tries finer discretization knobs that
round 1/2 could not express:

1. lower_right_ink with 3/4/5 states (zero-separated and plain quantile).
2. Same for band_bot and horiz_runs.
3. The baseline grid densities at 3/4 states (v2 only tuned 5 vs 4 on an
   earlier config) — a cheaper-CPT baseline might itself improve.
4. Combination: best card for the swapped slots + best grid card.
"""

from helper import (
    BASELINE_PARENTS, baseline_columns, directional_stats, fit_eval,
    load_split, disc_zero_sep, discretize,
)
import numpy as np


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    B_tr, B_va, cards_base = baseline_columns(X_tr, X_va)
    d_tr = directional_stats(X_tr, "train")
    d_va = directional_stats(X_va, "val")

    def run(tag, overrides=None, grid_cards=None, parents=BASELINE_PARENTS,
            alpha=1.0, backoff=True):
        """overrides: {slot: (vals_tr, vals_va, card)};
        grid_cards: card for each of the 8 grid slots (recompute)."""
        G_tr_cols, G_va_cols, cards = [], [], []
        # recompute grid densities with per-slot cards if requested
        from common import grid_densities
        G_tr_raw = grid_densities(X_tr)
        G_va_raw = grid_densities(X_va)
        grid_js = (1, 2, 3, 4, 5, 6, 7, 9)
        for k, j in enumerate(grid_js):
            card = (grid_cards or (5,) * 8)[k]
            a, b, _ = disc_zero_sep(G_tr_raw[:, j], G_va_raw[:, j], card)
            G_tr_cols.append(a); G_va_cols.append(b); cards.append(card)
        # semantic base columns: euler, vruns, aspect, vdom
        from common import continuous_stats
        s_tr = continuous_stats(X_tr, "train")
        s_va = continuous_stats(X_va, "val")
        base_sem = (
            ((np.clip(s_tr["euler"], -1, 2) + 1).astype(np.int64),
             (np.clip(s_va["euler"], -1, 2) + 1).astype(np.int64), 4),
        )
        G_tr_cols.append(base_sem[0][0]); G_va_cols.append(base_sem[0][1]); cards.append(4)
        for name, card in (("vert_runs", 5), ("aspect", 4), ("vert_dominance", 4)):
            a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
            G_tr_cols.append(a.astype(np.int64)); G_va_cols.append(b.astype(np.int64)); cards.append(card)
        F_tr = np.stack(G_tr_cols, axis=1)
        F_va = np.stack(G_va_cols, axis=1)
        cards = list(cards)
        for j, (v_tr, v_va, card) in (overrides or {}).items():
            F_tr[:, j] = v_tr
            F_va[:, j] = v_va
            cards[j] = card
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(cards), parents,
                       alpha=alpha, backoff=backoff)
        print(f"{tag:64s} val acc = {acc:.4f}")
        return acc

    def cand(name, card, zero_sep=True):
        if zero_sep:
            a, b, _ = disc_zero_sep(d_tr[name], d_va[name], card)
        else:
            # plain quantile cut points from train positives, zero separated
            pos = d_tr[name][d_tr[name] > 0]
            qs = np.quantile(pos, np.linspace(0, 1, card)[1:-1])
            a = np.searchsorted(qs, d_tr[name], side="right") + (d_tr[name] > 0)
            b = np.searchsorted(qs, d_va[name], side="right") + (d_va[name] > 0)
        return a.astype(np.int64), b.astype(np.int64), card

    print("--- baseline card sweep (all grid slots same card) ---")
    for card in (3, 4, 5, 6):
        run(f"baseline grid card={card}", grid_cards=(card,) * 8)

    print("--- lower_right_ink cards (slot 7, wire bl<-mr) ---")
    P = (-1, -1, -1, 0, 2, 1, 5, 5, -1, 8, 11, -1)
    for card in (3, 4, 5):
        for zs in (True, False):
            tag = f"br<-lower_right_ink card={card} zero_sep={zs}"
            run(tag, {7: cand("lower_right_ink", card, zs)}, parents=P)

    print("--- band_bot cards (slot 6, wire bl<-mr) ---")
    for card in (3, 4, 5):
        run(f"bl<-band_bot card={card} + br new parent=mr",
            {6: cand("band_bot", card), 7: cand("lower_right_ink", 4)}, parents=P)

    print("--- horiz_runs as vdom (slot 11) cards ---")
    for card in (3, 4, 5):
        run(f"vdom<-horiz_runs card={card} + br/bl swap",
            {6: cand("band_bot", 4), 7: cand("lower_right_ink", 4),
             11: cand("horiz_runs", card)}, parents=P)

    print("--- combined best candidates ---")
    run("bl<-band_bot4 + br<-lri4 + vdom<-horiz4, wire bl<-mr",
        {6: cand("band_bot", 4), 7: cand("lower_right_ink", 4),
         11: cand("horiz_runs", 4)}, parents=P)
    run("br<-lri4 + vdom<-horiz4, wire bl<-mr",
        {7: cand("lower_right_ink", 4), 11: cand("horiz_runs", 4)}, parents=P)
    run("br<-lri4 + vdom<-horiz4 (v2 wire)",
        {7: cand("lower_right_ink", 4), 11: cand("horiz_runs", 4)})
    run("bl<-band_bot4 + br<-lri4 (wire bl<-mr)",
        {6: cand("band_bot", 4), 7: cand("lower_right_ink", 4)}, parents=P)


if __name__ == "__main__":
    main()
