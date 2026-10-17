"""Exploration: discretisation and feature-set variants, seed-1322 val only.

Rounds:
  D1  discretisation sweep on the incumbent 12 features (all in-bounds, 2-5 states)
  D2  PDF six-band scheme (row thirds + column thirds) replacing the 3x3 grid
  D3  topology-augmented set (nholes / horiz_runs take over weak slots)
  D4  symmetry-augmented set (horizontal / vertical symmetry take a slot)
Every configuration is validated against the 6-12 / 2-5 / >=2-edges / <=2-parents rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate, validate_structure  # noqa: E402
from _sweep2_cells import (INCUMBENT_KEEP, SEMANTIC_4, GRID_RC,  # noqa: E402
                           template_incumbent, template_chain, full_parents)

INC_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)


def inc_design(disc_grid="zero_quantile", k_grid=5, disc_sem=None, k_sem=None):
    disc_sem = disc_sem or {}
    k_sem = k_sem or {}
    d = [(c, k_grid, disc_grid) for c in INCUMBENT_KEEP]
    d.append(("euler", k_sem.get("euler", 4), "euler_incumbent"))
    d.append(("vertical_runs", k_sem.get("vertical_runs", 5), disc_sem.get("vertical_runs", "zero_quantile")))
    d.append(("aspect_incumbent", k_sem.get("aspect", 4), disc_sem.get("aspect", "zero_quantile")))
    d.append(("vertical_dominance", k_sem.get("vdom", 4), disc_sem.get("vdom", "zero_quantile")))
    return d


def run_variant(label, design, parents, ctr, cva, ytr, yva, base, results):
    names = [d[0] for d in design]
    cards = [d[1] for d in design]
    validate_structure(names, cards, parents)
    acc, pred, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
    results.append((acc, label, list(zip(names, cards)), parents))
    print(f"  {label:52s} {acc:.4f}  ({acc-base:+.4f})")
    return acc, pred


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    base, _, _, _, _ = evaluate(inc_design(), INC_PARENTS, ctr, cva, ytr, yva)
    print(f"incumbent = {base:.4f}\n")
    results = []

    print("=== D1: discretisation of the incumbent feature set ===")
    for disc in ("zero_quantile", "quantile", "equal_width"):
        for k in (4, 5):
            run_variant(f"D1 grid K={k} {disc}", inc_design(disc_grid=disc, k_grid=k),
                        INC_PARENTS, ctr, cva, ytr, yva, base, results)
    for k in (4, 5):
        run_variant(f"D1 vruns K={k}", inc_design(k_sem={"vertical_runs": k}),
                    INC_PARENTS, ctr, cva, ytr, yva, base, results)
    for k in (3, 5):
        run_variant(f"D1 vdom K={k}", inc_design(k_sem={"vdom": k}),
                    INC_PARENTS, ctr, cva, ytr, yva, base, results)
    for k in (3, 5):
        run_variant(f"D1 aspect K={k}", inc_design(k_sem={"aspect": k}),
                    INC_PARENTS, ctr, cva, ytr, yva, base, results)
    run_variant("D1 grid K=5 + vruns K=4", inc_design(k_sem={"vertical_runs": 4}),
                INC_PARENTS, ctr, cva, ytr, yva, base, results)
    run_variant("D1 grid K=5 + vdom K=5 + aspect K=5",
                inc_design(k_sem={"vdom": 5, "aspect": 5}),
                INC_PARENTS, ctr, cva, ytr, yva, base, results)

    print("\n=== D2: PDF six-band scheme (rows top/mid/bot + cols left/mid/right) ===")
    bands = ["row_band_top", "row_band_mid", "row_band_bot",
             "col_band_left", "col_band_mid", "col_band_right"]
    # hand DAG: rows chain down (top<-mid, mid<-bot is wrong direction)...
    # semantic order: row_top -> row_mid -> row_bot ; col_left -> col_mid -> col_right
    # where each is the "ink flows" story.  Feature parents must be acyclic.
    band_parents = (-1, 0, 1,      # row_top -> row_mid -> row_bot
                    -1, 3, 4)      # col_left -> col_mid -> col_right
    # indices: 0..5 bands, 6 euler, 7 vruns, 8 aspect, 9 vdom, 10 nholes, 11 hsym
    d2 = [(b, 5, "zero_quantile") for b in bands] + [
        ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile"),
        ("nholes", 5, "zero_counts"), ("hsym_aligned", 5, "quantile")]
    d2_parents = band_parents + (-1, 6, 7, 7, 6, -1)
    run_variant("D2 six bands + euler vruns aspect vdom nholes hsym", d2, d2_parents,
                ctr, cva, ytr, yva, base, results)
    d2b = [(b, 5, "zero_quantile") for b in bands] + [
        ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile"),
        ("vsym_aligned", 5, "quantile"), ("nholes", 5, "zero_counts")]
    run_variant("D2 six bands + euler vruns aspect vdom vsym nholes", d2b, d2_parents,
                ctr, cva, ytr, yva, base, results)

    print("\n=== D3: topology-augmented set (drop aspect, add nholes / horiz_runs) ===")
    d3 = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
        ("euler", 4, "euler_incumbent"), ("nholes", 5, "zero_counts"),
        ("vertical_runs", 5, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
    run_variant("D3 grid8 + euler nholes vruns vdom", d3, INC_PARENTS,
                ctr, cva, ytr, yva, base, results)
    d3b = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
        ("euler", 4, "euler_incumbent"), ("horiz_runs", 5, "zero_quantile"),
        ("vertical_runs", 5, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
    run_variant("D3 grid8 + euler horiz_runs vruns vdom", d3b, INC_PARENTS,
                ctr, cva, ytr, yva, base, results)
    d3c = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
        ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("nholes", 5, "zero_counts"), ("vertical_dominance", 4, "zero_quantile")]
    run_variant("D3 grid8 + euler vruns nholes vdom (drop aspect)", d3c, INC_PARENTS,
                ctr, cva, ytr, yva, base, results)
    d3d = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
        ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("horiz_runs", 5, "zero_quantile"), ("nholes", 5, "zero_counts")]
    run_variant("D3 grid8 + euler vruns horiz_runs nholes", d3d, INC_PARENTS,
                ctr, cva, ytr, yva, base, results)

    print("\n=== D4: symmetry-augmented set ===")
    d4 = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
        ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("hsym_aligned", 5, "quantile"), ("vertical_dominance", 4, "zero_quantile")]
    run_variant("D4 grid8 + euler vruns hsym vdom (drop aspect)", d4, INC_PARENTS,
                ctr, cva, ytr, yva, base, results)
    d4b = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
        ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
        ("vsym_aligned", 4, "quantile"), ("vertical_dominance", 4, "zero_quantile")]
    run_variant("D4 grid8 + euler vruns vsym vdom (drop aspect)", d4b, INC_PARENTS,
                ctr, cva, ytr, yva, base, results)

    print("\n=== SUMMARY (sorted) ===")
    for acc, label, feats_list, parents in sorted(results, reverse=True)[:15]:
        print(f"  {acc:.4f}  {label}")


if __name__ == "__main__":
    main()
