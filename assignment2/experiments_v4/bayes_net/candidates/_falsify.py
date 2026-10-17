"""Falsification: cross-seed test of every non-trivial candidate.

Two candidates are on the table:
  A) 'vdom5'  -- legitimate: vertical_dominance 5 states instead of 4 (in-bounds)
  B) 'coord'  -- the greedy coordinate-ascent DAG.  This is HILL-CLIMBING
                 structure search, which the PDF FORBIDS; we evaluate it only
                 to show it does not generalise, i.e. that it is validation
                 overfitting rather than a real modelling gain.

Also re-runs the incumbent through the harness on all three seeds as control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V4 = HERE.parents[1]
sys.path.insert(0, str(V4))

from v4_common import load_split, mcnemar, paired_bootstrap, accuracy  # noqa: E402
from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate, validate_structure  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP  # noqa: E402

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
COORD_PARENTS = (7, 0, 1, 0, 3, 8, 5, 5, -1, 7, 0, 9)

INC_DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
    ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
VDOM5_DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
    ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 5, "zero_quantile")]

SEEDS = (1322, 7, 2024)


def main():
    feats = build_cache()
    print(f"{'seed':>5s} {'incumbent':>10s} {'vdom5':>10s} {'coord(illegal)':>15s}")
    preds = {1322: {}, 7: {}, 2024: {}}
    yvals = {}
    for seed in SEEDS:
        ctr, ytr, cva, yva = split_columns(feats, seed)
        row = {}
        for label, design, parents in (
            ("inc", INC_DESIGN, INC_PARENTS),
            ("vdom5", VDOM5_DESIGN, INC_PARENTS),
            ("coord", INC_DESIGN, COORD_PARENTS),
        ):
            acc, pred, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
            row[label] = (acc, pred)
        yvals[seed] = yva
        preds[seed] = row
        print(f"{seed:5d} {row['inc'][0]:10.4f} {row['vdom5'][0]:10.4f} {row['coord'][0]:15.4f}")

    print("\n=== McNemar + bootstrap vs incumbent (challenger = incumbent here means"
          " 'is the candidate a real gain') ===")
    for seed in SEEDS:
        yva = yvals[seed]
        pi = preds[seed]["inc"][1]
        for label in ("vdom5", "coord"):
            pc = preds[seed][label][1]
            mc = mcnemar(yva, pi, pc)
            bs = paired_bootstrap(yva, pi, pc)
            print(f"  seed {seed:5d} {label:6s} delta={bs['delta']:+.4f} "
                  f"CI=[{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}] "
                  f"McNemar b={mc['b']} c={mc['c']} p={mc['p_value']:.4f} "
                  f"real_gain={bool(bs['delta']>0 and mc['p_value']<0.05)}")

    # sanity: validate the coord structure is at least well-formed (it is
    # structurally legal even though it was found illegally)
    validate_structure([d[0] for d in INC_DESIGN], [d[1] for d in INC_DESIGN], COORD_PARENTS)


if __name__ == "__main__":
    main()
