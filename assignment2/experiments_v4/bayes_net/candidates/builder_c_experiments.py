"""builder_c_experiments.py -- reproduces every number reported in results/builder_c.md.

Run from this directory:

    python builder_c_experiments.py            # full reproduction (a few minutes)
    python builder_c_experiments.py --quick    # seed 1322 only

Only the 60,000 training images are ever read.  Continuous features are cached
in `cache/` on first run (~100 s); afterwards every experiment is fast.

Sections
  0  protocol + incumbent reproduction (must print 0.8614)
  1  builder-C candidate module vs incumbent, seed 1322 (McNemar + bootstrap)
  2  cross-seed falsification (seeds 7, 2024; and 4 extra seeds)
  3  top confusions for both models
  4  PDF compliance audit of the candidate structure
  5  the feature-redesign screens that motivated keeping the incumbent features
     (MI screen, single-slot swaps, cell subsets, semantic 4-tuples, budgets)
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parents[1]
ROOT = V4.parent
for path in (str(HERE), str(V4), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from v4_common import (load_split, accuracy, macro_f1, mcnemar,  # noqa: E402
                       paired_bootstrap, top_confusions, check_bn_bounds)
from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate, validate_structure  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP  # noqa: E402

from builder_c_dag_m6 import (BayesNetClassifier as CandidateBN,  # noqa: E402
                              FEATURE_NAMES as C_NAMES,
                              FEATURE_CARDINALITIES as C_CARDS,
                              FEATURE_PARENTS as C_PARENTS)

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
M1_PARENTS = (-1, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)
M8_PARENTS = (-1, 0, 1, 8, 3, 8, 5, 5, -1, 8, 9, 9)
ILLEGAL_PARENTS = (7, 0, 1, 0, 3, 8, 5, 5, -1, 7, 0, 9)

DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
    ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
D_NAMES = [d[0] for d in DESIGN]
D_CARDS = [d[1] for d in DESIGN]

SEEDS_MAIN = (1322, 7, 2024)
SEEDS_EXTRA = (1, 13, 99, 31337)


# ----------------------------------------------------------------------
def section(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def sec0_protocol():
    section("0. Protocol + incumbent reproduction")
    Xtr, ytr, Xva, yva = load_split()
    print(f"train={Xtr.shape} val={Xva.shape}  (seed 1322, stratified 1/6)")
    import bayes_net
    for alpha in (0.5, 1.0, 2.0):
        model = bayes_net.BayesNetClassifier(alpha=alpha).fit(Xtr, ytr)
        print(f"  incumbent alpha={alpha:<4} val_acc={accuracy(yva, model.predict(Xva)):.4f}")
    return Xtr, ytr, Xva, yva


def sec1_candidate(Xtr, ytr, Xva, yva):
    section("1. Builder-C candidate (M6 DAG) vs incumbent, seed 1322")
    inc = __import__("bayes_net").BayesNetClassifier(alpha=1.0).fit(Xtr, ytr)
    cand = CandidateBN(alpha=1.0).fit(Xtr, ytr)
    pi, pc = inc.predict(Xva), cand.predict(Xva)
    ai, ac = accuracy(yva, pi), accuracy(yva, pc)
    mc = mcnemar(yva, pi, pc)
    bs = paired_bootstrap(yva, pi, pc)
    f1i, _ = macro_f1(yva, pi)
    f1c, _ = macro_f1(yva, pc)
    print(f"  incumbent : acc={ai:.4f} macroF1={f1i:.4f}")
    print(f"  candidate : acc={ac:.4f} macroF1={f1c:.4f}")
    print(f"  delta={bs['delta']:+.4f} 95% CI=[{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}]")
    print(f"  McNemar b(inc wrong, cand right)={mc['b']} c={mc['c']} p={mc['p_value']:.4f}")
    print(f"  REAL GAIN = {bool(ac > ai and mc['p_value'] < 0.05)}")
    return pi, pc


def sec2_cross_seed(feats):
    section("2. Cross-seed falsification")
    print(f"{'seed':>6s} {'incumbent':>10s} {'M1':>9s} {'M6':>9s} {'M8':>9s} "
          f"{'illegal':>9s} {'M6 delta':>9s} {'M6 p':>8s}")
    wins = 0
    rows = []
    for seed in SEEDS_MAIN + SEEDS_EXTRA:
        ctr, ytr, cva, yva = split_columns(feats, seed)
        out = {}
        for key, parents in (("inc", INC_PARENTS), ("M1", M1_PARENTS),
                             ("M6", C_PARENTS), ("M8", M8_PARENTS),
                             ("ill", ILLEGAL_PARENTS)):
            validate_structure(D_NAMES, D_CARDS, parents)
            acc, pred, _, _, _ = evaluate(DESIGN, parents, ctr, cva, ytr, yva)
            out[key] = (acc, pred)
        mc = mcnemar(yva, out["inc"][1], out["M6"][1])
        bs = paired_bootstrap(yva, out["inc"][1], out["M6"][1])
        real = bs["delta"] > 0 and mc["p_value"] < 0.05
        wins += real
        rows.append((seed, out["inc"][0], out["M6"][0], bs["delta"], mc["p_value"], real))
        print(f"{seed:6d} {out['inc'][0]:10.4f} {out['M1'][0]:9.4f} {out['M6'][0]:9.4f} "
              f"{out['M8'][0]:9.4f} {out['ill'][0]:9.4f} {bs['delta']:+9.4f} "
              f"{mc['p_value']:8.4f}")
    print(f"\n  M6 real gain on {wins}/{len(SEEDS_MAIN) + len(SEEDS_EXTRA)} seeds")
    mean_delta = float(np.mean([r[3] for r in rows]))
    print(f"  M6 mean delta = {mean_delta:+.4f}")
    return rows


def sec3_confusions():
    section("3. Top confusions (seed 1322)")
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    for label, parents in (("incumbent", INC_PARENTS), ("candidate M6", C_PARENTS)):
        acc, pred, _, _, _ = evaluate(DESIGN, parents, ctr, cva, ytr, yva)
        f1, matrix = macro_f1(yva, pred)
        print(f"\n  {label}: acc={acc:.4f} macroF1={f1:.4f}")
        for t, p, n in top_confusions(np.array(matrix), 8):
            print(f"    {t} -> {p}: {n}")


def sec4_compliance():
    section("4. PDF compliance audit of the candidate")
    check_bn_bounds(C_NAMES, C_CARDS, C_PARENTS)
    print(f"  features        : {len(C_NAMES)}  (rule 6..12) OK")
    print(f"  cardinalities   : {C_CARDS}  all in 2..5 OK")
    edges = [(C_NAMES[p], C_NAMES[c]) for c, p in enumerate(C_PARENTS) if p >= 0]
    print(f"  feature edges   : {len(edges)} (rule >=2) OK")
    for parent, child in edges:
        print(f"    {parent:22s} -> {child}")
    print("  parents/feature : Y + at most 1 feature parent (rule <=2) OK")
    print("  acyclic         : verified by check_bn_bounds (Kahn) OK")
    print("  Y is root of every feature; no hidden variables; no structure search")
    print("  every CPT smoothed (alpha=1.0); predictions by log-joint sum")


def sec5_feature_screens(feats):
    section("5. Feature-redesign screens (why the incumbent feature set stays)")
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    base, _, _, _, _ = evaluate(DESIGN, INC_PARENTS, ctr, cva, ytr, yva)
    print(f"  incumbent reference = {base:.4f}")

    # 5a single-slot swaps (best per slot)
    from _harness import DISCRETISERS, mutual_information
    print("\n  5a. best single-slot replacement per slot (incumbent DAG):")
    swaps = [("nholes", 5, "zero_counts"), ("horiz_runs", 5, "zero_quantile"),
             ("hsym_aligned", 5, "quantile"), ("vsym_aligned", 5, "quantile"),
             ("row_band_bot", 5, "zero_quantile"), ("row_band_top", 5, "zero_quantile"),
             ("ncomp4", 5, "zero_counts"), ("ncomp8", 5, "zero_counts"),
             ("endpoints", 5, "zero_counts"), ("junctions", 5, "zero_counts"),
             ("aspect", 5, "zero_quantile"), ("bbox_fill", 5, "quantile")]
    best_per_slot = {}
    for key, k, disc in swaps:
        for slot in range(12):
            design = [tuple(d) for d in DESIGN]
            design[slot] = (key, k, disc)
            acc, _, _, _, _ = evaluate(design, INC_PARENTS, ctr, cva, ytr, yva)
            if slot not in best_per_slot or acc > best_per_slot[slot][0]:
                best_per_slot[slot] = (acc, key, k, disc)
    for slot in range(12):
        acc, key, k, disc = best_per_slot[slot]
        print(f"    slot {slot:2d} ({DESIGN[slot][0]:18s}) -> {key:16s} "
              f"{acc:.4f} ({acc-base:+.4f})")

    # 5b MI of the PDF-named features
    print("\n  5b. I(F;Y) of the PDF-named feature families (train only):")
    rows = []
    for key, k, disc in [("euler", 4, "euler_incumbent"), ("nholes", 5, "zero_counts"),
                         ("ncomp4", 5, "zero_counts"), ("ncomp8", 5, "zero_counts"),
                         ("endpoints", 5, "zero_counts"), ("junctions", 5, "zero_counts"),
                         ("hsym_aligned", 5, "quantile"), ("vsym_aligned", 5, "quantile"),
                         ("hsym_jaccard", 5, "quantile"), ("vsym_jaccard", 5, "quantile"),
                         ("horiz_runs", 5, "zero_quantile"), ("vertical_runs", 5, "zero_quantile")]:
        a, _, _ = DISCRETISERS[disc](np.asarray(ctr[key], np.float64),
                                     np.asarray(cva[key], np.float64), k)
        rows.append((mutual_information(a, ytr, k), key))
    for v, key in sorted(rows, reverse=True):
        print(f"    {key:18s} I(F;Y)={v:.4f}")

    # 5c cell subsets / semantic tuples / budgets -- reprint the headline results
    print("\n  5c. budget reallocations (seed 1322):")
    print("    (full tables in _sweep2_cells.py / _sweep3_semantic.py /")
    print("     _explore_budget.py / _explore_grid9.py / _explore_geometry.py)")
    for label, cells, parents in (
        ("8-cell incumbent set (control)", INCUMBENT_KEEP, INC_PARENTS),
    ):
        design = [(c, 5, "zero_quantile") for c in cells] + list(DESIGN[8:])
        acc, _, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
        print(f"    {label:42s} {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="seed 1322 only")
    args = parser.parse_args()

    Xtr, ytr, Xva, yva = sec0_protocol()
    sec1_candidate(Xtr, ytr, Xva, yva)
    feats = build_cache()
    if args.quick:
        section("(--quick: cross-seed and screens skipped)")
        return
    sec2_cross_seed(feats)
    sec3_confusions()
    sec4_compliance()
    sec5_feature_screens(feats)
    section("DONE")


if __name__ == "__main__":
    main()
