"""Round 1 (v3): directional-feature replacement candidates vs 2/3/5 confusion.

Step 1: reproduce the v2 baseline (84.92%) in this pipeline.
Step 2: for each candidate new feature, replace ONE grid-density slot in
the v2 12-feature config (keeping the v2 column-chain DAG skeleton) and
measure validation accuracy.  Also try replacing semantic slots.
The grid slot replaced is the one judged most redundant (bottom_right or
middle_center), plus a sweep over all 8 grid slots for the top candidate.
"""

from helper import (
    BASELINE_PARENTS, baseline_columns, directional_stats, fit_eval,
    load_split, disc_zero_sep,
)
import numpy as np

# v2 order: 8 grid + euler + vruns + aspect + vdom
V2_NAMES = (
    "grid_tl", "grid_tc", "grid_tr", "grid_ml", "grid_mc",
    "grid_mr", "grid_bl", "grid_br",
    "euler", "vruns", "aspect", "vdom",
)


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    B_tr, B_va, cards_base = baseline_columns(X_tr, X_va)

    # --- step 1: baseline reproduction
    acc = fit_eval(B_tr, y_tr, B_va, y_va, cards_base, BASELINE_PARENTS,
                   alpha=1.0, backoff=True)
    print(f"baseline (v2 repro)          val acc = {acc:.4f}")

    d_tr = directional_stats(X_tr, "train")
    d_va = directional_stats(X_va, "val")

    # candidate new features: (name, card, is_zero_separated)
    CANDS = (
        ("grad_vert_frac", 4, False),
        ("grad_up_frac", 4, False),
        ("upper_left_ink", 4, True),
        ("lower_left_ink", 4, True),
        ("upper_right_ink", 4, True),
        ("lower_right_ink", 4, True),
        ("mid_left_ink", 3, True),
        ("mid_right_ink", 3, True),
        ("horiz_runs", 5, False),
        ("endpoints", 5, False),
        ("band_top", 4, False),
        ("band_mid", 4, False),
        ("band_bot", 4, False),
        ("stroke_width", 4, False),
    )
    cand_cols = {}
    for name, card, zero_sep in CANDS:
        if zero_sep:
            a, b, _ = disc_zero_sep(d_tr[name], d_va[name], card)
        else:
            a, b, _ = disc_zero_sep(d_tr[name], d_va[name], card)
        cand_cols[name] = (
            a.astype(np.int64), b.astype(np.int64), card,
        )

    def run(tag, col_overrides, parents=BASELINE_PARENTS, alpha=1.0,
            backoff=True, sem_overrides=None):
        """col_overrides: {slot_index: cand_name} replaces that column."""
        c_tr, c_va, cards = [], [], list(cards_base)
        for j in range(12):
            if j in col_overrides:
                nm = col_overrides[j]
                a, b, card = cand_cols[nm]
                c_tr.append(a); c_va.append(b); cards[j] = card
            else:
                c_tr.append(B_tr[:, j]); c_va.append(B_va[:, j])
        F_tr = np.stack(c_tr, axis=1)
        F_va = np.stack(c_va, axis=1)
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(cards), parents,
                       alpha=alpha, backoff=backoff)
        print(f"{tag:52s} val acc = {acc:.4f}")
        return acc

    print("\n--- replace one grid slot with each candidate (slot = bottom_right) ---")
    for name, card, _ in CANDS:
        run(f"replace grid_br(7) <- {name}({card})", {7: name})

    print("\n--- replace grid_mc(4) or grid_bl(6) with top-3 candidates ---")
    for name in ("lower_left_ink", "horiz_runs", "endpoints", "band_bot"):
        run(f"replace grid_mc(4) <- {name}", {4: name})
        run(f"replace grid_bl(6) <- {name}", {6: name})

    print("\n--- replace a semantic slot (aspect/vdom) with candidates ---")
    for name in ("lower_left_ink", "upper_left_ink", "horiz_runs", "endpoints",
                 "grad_vert_frac", "grad_up_frac"):
        run(f"replace aspect(10) <- {name}", {10: name})
        run(f"replace vdom(11) <- {name}", {11: name})


if __name__ == "__main__":
    main()
