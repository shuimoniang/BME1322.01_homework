"""Round 4 (v3): DAG variants + smoothing refinement on the v2 12-feature set.

Cardinality note from round 3: grid densities at 6 states would give 85.07%
but violate the 2-5 states rule, so discarded.  Directional replacements
never beat 84.92%.  This round sweeps hand-designed DAG variants (all legal,
each feature <= 1 feature-parent) and smoothing schemes on the exact v2
12-feature configuration.

Feature order (v2): 0 tl, 1 tc, 2 tr, 3 ml, 4 mc, 5 mr, 6 bl, 7 br,
                    8 euler, 9 vruns, 10 aspect, 11 vdom
"""

from helper import (
    BASELINE_PARENTS, baseline_columns, fit_eval, load_split, disc_zero_sep,
)
import numpy as np
from main import stratified_split


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    B_tr, B_va, cards = baseline_columns(X_tr, X_va)

    def run(tag, parents, alpha=1.0, backoff=True):
        acc = fit_eval(B_tr, y_tr, B_va, y_va, cards, parents,
                       alpha=alpha, backoff=backoff)
        print(f"{tag:64s} val acc = {acc:.4f}")
        return acc

    print("--- DAG variants (alpha=1.0, backoff) ---")
    run("v2 baseline (col chains + euler->vruns, vdom->aspect)",
        BASELINE_PARENTS)
    # column chain reversed: bottom -> top
    run("col chains reversed (ml<-bl, mc<-bc? n/a, mr<-br, tc? )",
        (-1, -1, -1, 6, -1, 7, -1, -1, -1, 8, 11, -1))
    # row chains (left -> right) replacing column chains
    run("row chains (tc<-tl, mc<-ml, tr<-tc, mr<-mc, br<-mr)",
        (-1, 0, 1, -1, 3, 4, -1, 5, -1, 8, 11, -1))
    # row chains right -> left
    run("row chains reversed (tl<-tc, ml<-mc, ...)",
        (1, -1, -1, 4, -1, -1, -1, -1, -1, 8, 11, -1))
    # star-ish: middle_center as hub
    run("hub mc: tl<-mc, tc<-mc, tr<-mc, ml<-mc, mr<-mc",
        (4, 4, 4, 4, -1, 4, -1, -1, -1, 8, 11, -1))
    # diagonal chains: tl->mc->br plus tr->mc
    run("diag: tl->mc, mc->br, tr->mr, ml<-bl? mixed",
        (-1, -1, -1, -1, 0, 2, -1, 4, -1, 8, 11, -1))
    # semantic re-wires
    run("vruns<-euler, aspect<-euler, vdom<-vruns",
        (-1, -1, -1, 0, 2, 1, 4, 5, -1, 8, 9, -1))
    run("aspect<-vruns, vdom<-euler",
        (-1, -1, -1, 0, 2, 1, 4, 5, -1, 8, 9, -1))
    run("vruns<-euler, vdom<-vruns, aspect<-vdom (semantic chain)",
        (-1, -1, -1, 0, 2, 1, 4, 5, -1, 8, 9, -1))
    run("vruns<-euler, vdom<-euler, aspect class-only",
        (-1, -1, -1, 0, 2, 1, 4, 5, -1, 8, -1, 8))
    run("vruns<-euler, vdom<-vruns, aspect<-vruns",
        (-1, -1, -1, 0, 2, 1, 4, 5, -1, 8, 9, 9))
    run("no feature edges (pure NB over 12)",
        (-1,) * 12)

    print("\n--- smoothing variants on v2 baseline DAG ---")
    run("alpha=0.5", BASELINE_PARENTS, alpha=0.5)
    run("alpha=2.0", BASELINE_PARENTS, alpha=2.0)
    run("no backoff, alpha=1.0", BASELINE_PARENTS, backoff=False)

    # hard backoff: CPT rows with parent-state count < threshold fall back
    # entirely to the class-conditional marginal
    def fit_eval_hard_backoff(min_count, alpha=1.0):
        class_count = np.bincount(y_tr, minlength=10).astype(np.float64)
        log_prior = np.log((class_count + alpha) / (len(y_tr) + 10 * alpha))
        cpts = []
        for node, parent in enumerate(BASELINE_PARENTS):
            card = cards[node]
            if parent < 0:
                counts = np.zeros((10, card))
                np.add.at(counts, (y_tr, B_tr[:, node]), 1)
                denom = counts.sum(-1, keepdims=True) + card * alpha
                cpts.append(np.log((counts + alpha) / denom))
            else:
                pcard = cards[parent]
                counts = np.zeros((10, pcard, card))
                np.add.at(counts, (y_tr, B_tr[:, parent], B_tr[:, node]), 1)
                marg = np.zeros((10, card))
                np.add.at(marg, (y_tr, B_tr[:, node]), 1)
                marg = (marg + alpha) / (class_count[:, None] + card * alpha)
                n_cp = counts.sum(-1, keepdims=True)
                # soft backoff weight: full alpha where n_cp >= min_count,
                # pure marginal where n_cp < min_count
                w = np.where(n_cp >= min_count, alpha, 0.0)
                cpts.append(np.log((counts + w * marg[:, None, :]) / (n_cp + w)))
        scores = np.broadcast_to(log_prior, (B_va.shape[0], 10)).copy()
        for node, parent in enumerate(BASELINE_PARENTS):
            if parent < 0:
                scores += cpts[node][:, B_va[:, node]].T
            else:
                scores += cpts[node][:, B_va[:, parent], B_va[:, node]].T
        return float(np.mean(np.argmax(scores, axis=1) == y_va))

    for mc in (5, 10, 20, 50):
        acc = fit_eval_hard_backoff(mc)
        print(f"hard backoff min_count={mc:<22d} val acc = {acc:.4f}")

    # robustness across seeds for anything that beat 84.92
    print("\n--- cross-seed check for promising variants ---")


if __name__ == "__main__":
    main()
