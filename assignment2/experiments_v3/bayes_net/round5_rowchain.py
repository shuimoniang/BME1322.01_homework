"""Round 5 (v3): refine the row-chain discovery from round 4.

Round 4: row chains (tc<-tl, mc<-ml, tr<-tc, mr<-mc, br<-mr) jumped to
85.87% (+0.95 over the 84.92% column-chain baseline).  This round:
1. explores row-chain variants (direction, which cells chain, extra/removed
   edges) combined with the semantic wiring;
2. cross-seed validation of the top configs.
Feature order: 0 tl, 1 tc, 2 tr, 3 ml, 4 mc, 5 mr, 6 bl, 7 br,
               8 euler, 9 vruns, 10 aspect, 11 vdom
"""

from helper import (
    BASELINE_PARENTS, baseline_columns, fit_eval, load_split,
)
import numpy as np

ROW_CHAIN = (-1, 0, 1, -1, 3, 4, -1, 5, -1, 8, 11, -1)


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    B_tr, B_va, cards = baseline_columns(X_tr, X_va)

    def run(tag, parents, alpha=1.0, backoff=True):
        acc = fit_eval(B_tr, y_tr, B_va, y_va, cards, parents,
                       alpha=alpha, backoff=backoff)
        print(f"{tag:68s} val acc = {acc:.4f}")
        return acc

    print("--- row-chain base variants (v2 semantic wiring) ---")
    run("A row chain l->r + vruns<-euler, vdom<-aspect", ROW_CHAIN)
    # direction reversed within rows
    run("B row chain r->l (tl<-tc, ml<-mc, bl? ...)",
        (1, -1, -1, 4, -1, -1, -1, -1, -1, 8, 11, -1))
    # row chain but bottom row chained too: bl<-? there is no bottom_center
    # (dropped); bl<-mr diagonal? try br<-mr and bl<-br
    run("C l->r + bl<-br (close bottom row)",
        (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 11, -1))
    run("D l->r + bl<-mc (diag)",
        (-1, 0, 1, -1, 3, 4, 4, 5, -1, 8, 11, -1))
    run("E l->r, drop tr<-tc (top row split)",
        (-1, 0, -1, -1, 3, 4, -1, 5, -1, 8, 11, -1))
    run("F l->r, drop br<-mr",
        (-1, 0, 1, -1, 3, 4, -1, -1, -1, 8, 11, -1))
    run("G only vertical center chain + row chain bottom?",
        (-1, 0, 1, -1, 3, 4, 4, 5, -1, 8, 11, -1))

    print("\n--- semantic wiring on top of row chains ---")
    for tag, sem in (
        ("vruns<-euler, vdom<-aspect", (-1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 11, -1)),
        ("vruns<-euler, vdom<-vruns, aspect<-vruns",
         (-1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 9)),
        ("vruns<-euler, vdom<-euler, aspect<-vdom",
         (-1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 11, 8)),
        ("vruns<-euler, vdom<-vruns, aspect<-vdom",
         (-1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 11)),
        ("vruns<-euler, vdom<-euler, aspect<-vruns",
         (-1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 11, 9)),
        ("vruns<-euler, vdom<-aspect, aspect<-vruns",
         (-1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 11)),
    ):
        # sem parents index features 8..11 only; splice into ROW_CHAIN
        parents = list(ROW_CHAIN)
        for k in range(8, 12):
            parents[k] = sem[k]
        run(f"rowchain + {tag}", tuple(parents))

    print("\n--- alpha + backoff on the leading config ---")


if __name__ == "__main__":
    main()
