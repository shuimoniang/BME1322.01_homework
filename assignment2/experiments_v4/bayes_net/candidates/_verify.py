"""Independent verification + a SMALL hand-authored DAG candidate set.

Part 1 verifies the two software paths agree: the exact `bayes_net.py` module
with a monkey-patched FEATURE_PARENTS (independent of `_harness.SweepBN`) must
match my harness on the same split.

Part 2 tests a SMALL, hand-authored set of DAG candidates -- the same
compliant methodology v3 used ("7 manual candidates, chosen on validation"),
NOT an automated search.  Each candidate has one stated semantic hypothesis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V4 = HERE.parents[1]
ROOT = V4.parent
sys.path.insert(0, str(V4))
sys.path.insert(0, str(ROOT))

from v4_common import load_split, accuracy, mcnemar, paired_bootstrap  # noqa: E402
from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP  # noqa: E402

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
COORD_PARENTS = (7, 0, 1, 0, 3, 8, 5, 5, -1, 7, 0, 9)
INC_DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
    ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]

# ----------------------------------------------------------------------
# Hand-authored candidates.  Indices: 0 tl,1 tc,2 tr,3 ml,4 mc,5 mr,
# 6 bl,7 br (8 central grid cells), 8 euler, 9 vertical_runs,
# 10 aspect, 11 vertical_dominance.
# ----------------------------------------------------------------------
MANUAL_DAGS = {
    # v3 incumbent (control)
    "v3_row_chain": INC_PARENTS,

    # H1 "topology conditions regional ink": euler and vertical_runs become the
    # feature parents of the two middle-band region densities, because the
    # presence/absence of loops and the vertical fragmentation of strokes are
    # what distinguish 0/6/8 from 2/3/5 in the middle band.
    "H1_euler_to_midbands": (7, 0, 1, -1, 3, 4, 5, 5, -1, 8, 9, 9),

    # H2 "diagonal balance": the top-left region is conditioned on the
    # bottom-right region (and vice versa is already impossible: keep one edge),
    # capturing the anti-diagonal ink trade-off seen in 2/3/5/7 vs 0/1.
    "H2_diagonal_tl_br": (7, 0, 1, -1, 3, 4, 5, 5, -1, 8, 9, 9),
}


def main():
    Xtr, ytr, Xva, yva = load_split()

    print("=== Part 1: independent verification (bayes_net.py with patched DAG) ===")
    import bayes_net
    for label, parents in (("v3 incumbent", INC_PARENTS), ("coord (illegal)", COORD_PARENTS)):
        original = bayes_net.FEATURE_PARENTS
        bayes_net.FEATURE_PARENTS = parents
        try:
            model = bayes_net.BayesNetClassifier(alpha=1.0).fit(Xtr, ytr)
            acc_module = accuracy(yva, model.predict(Xva))
        finally:
            bayes_net.FEATURE_PARENTS = original
        print(f"  {label:16s} module acc = {acc_module:.4f}")

    feats = build_cache()
    ctr, ytr2, cva, yva2 = split_columns(feats, 1322)
    assert np.array_equal(yva, yva2)
    for label, parents in (("v3 incumbent", INC_PARENTS), ("coord (illegal)", COORD_PARENTS)):
        acc_h, pred_h, _, _, _ = evaluate(INC_DESIGN, parents, ctr, cva, ytr2, yva2)
        print(f"  {label:16s} harness acc = {acc_h:.4f}")

    print("\n=== Part 2: hand-authored DAG candidates (validation, seed 1322) ===")
    base, pred_base, _, _, _ = evaluate(INC_DESIGN, INC_PARENTS, ctr, cva, ytr2, yva2)
    print(f"control v3 = {base:.4f}\n")
    for label, parents in MANUAL_DAGS.items():
        acc, pred, _, _, _ = evaluate(INC_DESIGN, parents, ctr, cva, ytr2, yva2)
        mc = mcnemar(yva2, pred_base, pred)
        print(f"  {label:24s} {acc:.4f} ({acc-base:+.4f})  McNemar p={mc['p_value']:.4f}")


if __name__ == "__main__":
    main()
