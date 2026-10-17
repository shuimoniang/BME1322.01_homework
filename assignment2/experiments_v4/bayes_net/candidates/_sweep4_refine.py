"""Sweep 4 -- refinement around the incumbent local optimum.

T1  joint K search over the 4 semantic slots (K in 2..5 each, 256 configs)
T2  alternative discretisations for those slots (physical bands / equal width)
T3  physically-meaningful symmetry bands (near-perfect / partial / asymmetric)
T4  true-bbox aspect vs the incumbent's aspect formula
T5  low-conditional-redundancy swap: vertical_runs + horiz_runs together
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate, validate_structure, DISCRETISERS  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP, template_incumbent  # noqa: E402

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
GRID_DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP]


# --- T3: physical symmetry bands ------------------------------------
def disc_sym_bands(col_tr, col_va, n_states):
    """Fixed, interpretable bands for a Jaccard symmetry score.

    K=3: [0, .55) mostly asymmetric | [.55, .8) partial | [.8, 1] near-perfect
    K=4: splits the asymmetric band once more.
    K=5: splits the partial band as well.
    """
    if n_states == 3:
        edges = np.array([0.55, 0.80])
    elif n_states == 4:
        edges = np.array([0.40, 0.55, 0.80])
    elif n_states == 5:
        edges = np.array([0.40, 0.55, 0.70, 0.80])
    else:
        edges = np.array([0.55])
    f = lambda c: np.searchsorted(edges, c, side="right").astype(np.int64)
    return f(col_tr), f(col_va), edges


DISCRETISERS["sym_bands"] = disc_sym_bands


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    base, _, _, _, _ = evaluate(
        GRID_DESIGN + [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                       ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")],
        INC_PARENTS, ctr, cva, ytr, yva)
    print(f"incumbent = {base:.4f}\n")
    results = []

    print("=== T1: joint K search over the 4 semantic slots (incumbent keys/wiring) ===")
    # euler uses the fixed 4-state digital-topology encoding, so only the other
    # three semantic slots have a free K.
    for k_e, k_v, k_a, k_d in itertools.product((4,), (3, 4, 5), (3, 4, 5), (3, 4, 5)):
        design = GRID_DESIGN + [("euler", k_e, "euler_incumbent"),
                                ("vertical_runs", k_v, "zero_quantile"),
                                ("aspect_incumbent", k_a, "zero_quantile"),
                                ("vertical_dominance", k_d, "zero_quantile")]
        try:
            validate_structure([d[0] for d in design], [d[1] for d in design], INC_PARENTS)
        except AssertionError:
            continue
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr, cva, ytr, yva)
        results.append((acc, f"T1 euler{k_e} vruns{k_v} aspect{k_a} vdom{k_d}"))
    for acc, label in sorted(results, reverse=True)[:8]:
        print(f"  {acc:.4f}  {label}")
    results.clear()

    print("\n=== T2: discretiser alternatives for the semantic slots ===")
    for key, k, disc in [
        ("vertical_runs", 5, "equal_width"), ("vertical_runs", 5, "quantile"),
        ("vertical_dominance", 4, "quantile"), ("vertical_dominance", 5, "quantile"),
        ("vertical_dominance", 5, "equal_width"),
        ("aspect_incumbent", 4, "quantile"), ("aspect_incumbent", 5, "quantile"),
    ]:
        design = GRID_DESIGN + [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                                ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
        slot = {"vertical_runs": 9, "vertical_dominance": 11, "aspect_incumbent": 10}[key]
        design[slot] = (key, k, disc)
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr, cva, ytr, yva)
        results.append((acc, f"T2 {key} K={k} {disc}"))

    print("\n=== T3: physical symmetry bands in the aspect slot ===")
    for key, k in [("hsym_aligned", 3), ("hloc", 0), ("vsym_aligned", 3),
                   ("hsym_aligned", 4), ("vsym_aligned", 4),
                   ("hsym_aligned", 5), ("vsym_aligned", 5)]:
        if key == "hloc":
            continue
        design = GRID_DESIGN + [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                                (key, k, "sym_bands"), ("vertical_dominance", 4, "zero_quantile")]
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr, cva, ytr, yva)
        results.append((acc, f"T3 {key} K={k} sym_bands (aspect slot)"))

    print("\n=== T4: true-bbox aspect vs incumbent aspect formula ===")
    for key, k, disc in [("aspect", 4, "zero_quantile"), ("aspect", 5, "zero_quantile"),
                         ("aspect", 4, "quantile"), ("aspect", 5, "quantile"),
                         ("width", 4, "quantile"), ("width", 5, "quantile"),
                         ("bbox_fill", 4, "quantile"), ("bbox_fill", 5, "quantile"),
                         ("aspect_incumbent", 5, "zero_quantile")]:
        design = GRID_DESIGN + [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
                                (key, k, disc), ("vertical_dominance", 4, "zero_quantile")]
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr, cva, ytr, yva)
        results.append((acc, f"T4 {key} K={k} {disc} (aspect slot)"))

    print("\n=== T5: low-conditional-redundancy pairs in the 4 semantic slots ===")
    combos = [
        [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
         ("horiz_runs", 5, "zero_quantile"), ("aspect_incumbent", 4, "zero_quantile")],
        [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
         ("horiz_runs", 4, "zero_quantile"), ("aspect_incumbent", 4, "zero_quantile")],
        [("nholes", 5, "zero_counts"), ("vertical_runs", 5, "zero_quantile"),
         ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")],
        [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
         ("aspect_incumbent", 5, "zero_quantile"), ("vertical_dominance", 5, "zero_quantile")],
        [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
         ("aspect_incumbent", 4, "zero_quantile"), ("hset", 0, "zero_quantile")],
    ]
    for i, sem in enumerate(combos):
        sem = [s for s in sem if s[0] != "hset"]
        if len(sem) != 4:
            continue
        design = GRID_DESIGN + sem
        acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr, cva, ytr, yva)
        results.append((acc, f"T5 {'+'.join(s[0] for s in sem)}"))

    print("\n=== T2/T3/T4/T5 SUMMARY ===")
    for acc, label in sorted(results, reverse=True):
        print(f"  {acc:.4f}  {label}  ({acc-base:+.4f})")


if __name__ == "__main__":
    main()
