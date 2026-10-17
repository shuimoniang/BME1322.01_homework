"""Cross-seed falsification of the hand-authored DAG candidates.

Candidates carried forward from `_manual_dags`:
  M1  euler -> {ml, mr}
  M6  M1 + tl <- br          (best legitimate candidate, val 0.8686)
  M8  M1 + bl <- mr
Plus, for context only, the FORBIDDEN coordinate-ascent DAG (to show that an
illegal search buys more than the compliant method -- and to document why we
reject it even though it "works").

For every candidate on every seed we report accuracy, the paired bootstrap CI
of the delta vs the incumbent on that same split, and the McNemar p-value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V4 = HERE.parents[1]
sys.path.insert(0, str(V4))

from v4_common import mcnemar, paired_bootstrap, top_confusions, macro_f1  # noqa: E402
from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate, validate_structure  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP  # noqa: E402

DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
    ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
NAMES = [d[0] for d in DESIGN]
CARDS = [d[1] for d in DESIGN]

CANDIDATES = {
    "incumbent": (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9),
    "M1_euler_midband": (-1, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9),
    "M6_euler_midband_diag": (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9),
    "M8_euler_midband_blbr": (-1, 0, 1, 8, 3, 8, 5, 5, -1, 8, 9, 9),
    "ILLEGAL_coord_ascent": (7, 0, 1, 0, 3, 8, 5, 5, -1, 7, 0, 9),
}
SEEDS = (1322, 7, 2024)


def main():
    feats = build_cache()
    store = {}
    for seed in SEEDS:
        ctr, ytr, cva, yva = split_columns(feats, seed)
        store[seed] = {"y": yva, "preds": {}}
        for key, parents in CANDIDATES.items():
            validate_structure(NAMES, CARDS, parents)
            acc, pred, _, _, _ = evaluate(DESIGN, parents, ctr, cva, ytr, yva)
            store[seed]["preds"][key] = (acc, pred)

    print("=== accuracies ===")
    header = f"{'seed':>5s}" + "".join(f"{k[:22]:>24s}" for k in CANDIDATES)
    print(header)
    for seed in SEEDS:
        print(f"{seed:5d}" + "".join(
            f"{store[seed]['preds'][k][0]:24.4f}" for k in CANDIDATES))

    print("\n=== delta vs incumbent, McNemar, bootstrap CI (per seed) ===")
    for key in CANDIDATES:
        if key == "incumbent":
            continue
        print(f"\n  {key}:")
        for seed in SEEDS:
            yva = store[seed]["y"]
            pi = store[seed]["preds"]["incumbent"][1]
            pc = store[seed]["preds"][key][1]
            mc = mcnemar(yva, pi, pc)
            bs = paired_bootstrap(yva, pi, pc)
            real = bool(bs["delta"] > 0 and mc["p_value"] < 0.05)
            print(f"    seed {seed:5d} delta={bs['delta']:+.4f} "
                  f"CI=[{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}] "
                  f"b={mc['b']:4d} c={mc['c']:4d} p={mc['p_value']:.4f} real_gain={real}")

    print("\n=== top confusions, seed 1322 ===")
    for key in CANDIDATES:
        acc, pred = store[1322]["preds"][key]
        f1, matrix = macro_f1(store[1322]["y"], pred)
        print(f"\n  {key}: acc={acc:.4f} macroF1={f1:.4f}")
        print("    ", top_confusions(np.array(matrix), 6))


if __name__ == "__main__":
    main()
