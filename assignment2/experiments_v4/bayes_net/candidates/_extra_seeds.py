"""Extra-seed stress test for the finalists M1 / M6 / M8.

The brief mandates seeds 7 and 2024; because the gain is under 1pt we add four
more independent splits (1, 13, 99, 31337) to make sure M6 is not split luck.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V4 = HERE.parents[1]
sys.path.insert(0, str(V4))

from v4_common import mcnemar, paired_bootstrap  # noqa: E402
from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate, validate_structure  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP  # noqa: E402

DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
    ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
NAMES = [d[0] for d in DESIGN]
CARDS = [d[1] for d in DESIGN]

INC = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
M1 = (-1, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)
M6 = (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)
M8 = (-1, 0, 1, 8, 3, 8, 5, 5, -1, 8, 9, 9)
SEEDS = (1322, 7, 2024, 1, 13, 99, 31337)


def main():
    feats = build_cache()
    print(f"{'seed':>6s} {'inc':>8s} {'M1':>8s} {'M6':>8s} {'M8':>8s}   "
          f"{'M6 delta':>9s} {'M6 p':>8s} {'M8 delta':>9s} {'M8 p':>8s}")
    wins6 = wins8 = 0
    for seed in SEEDS:
        ctr, ytr, cva, yva = split_columns(feats, seed)
        out = {}
        for key, parents in (("inc", INC), ("M1", M1), ("M6", M6), ("M8", M8)):
            validate_structure(NAMES, CARDS, parents)
            acc, pred, _, _, _ = evaluate(DESIGN, parents, ctr, cva, ytr, yva)
            out[key] = (acc, pred)
        m6 = mcnemar(yva, out["inc"][1], out["M6"][1])
        b6 = paired_bootstrap(yva, out["inc"][1], out["M6"][1])
        m8 = mcnemar(yva, out["inc"][1], out["M8"][1])
        b8 = paired_bootstrap(yva, out["inc"][1], out["M8"][1])
        wins6 += b6["delta"] > 0 and m6["p_value"] < 0.05
        wins8 += b8["delta"] > 0 and m8["p_value"] < 0.05
        print(f"{seed:6d} {out['inc'][0]:8.4f} {out['M1'][0]:8.4f} {out['M6'][0]:8.4f} "
              f"{out['M8'][0]:8.4f}   {b6['delta']:+9.4f} {m6['p_value']:8.4f} "
              f"{b8['delta']:+9.4f} {m8['p_value']:8.4f}")
    print(f"\nM6 real gain on {wins6}/{len(SEEDS)} seeds; M8 real gain on {wins8}/{len(SEEDS)} seeds")


if __name__ == "__main__":
    main()
