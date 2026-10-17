"""A SMALL, hand-authored DAG candidate family (the v3-compliant methodology).

HYPOTHESIS
----------
The incumbent's worst confusions are 5->3, 3->5, 3->2, 5->2, 2->3 (see
BASELINE/`_repro`).  Those digits all live in the *middle band* and differ in
how the middle band is connected to the rest of the glyph -- 3 has a closed
right side, 5 has a closed top and open bottom, 2 is open everywhere.  The v3
chain only conditions each middle-band cell on its left neighbour, so it can
never express "the middle band is isolated / has a loop / is a broken stroke".
The clean modelling fix: give the middle-band cells a feature parent that
CARRIES the global structure -- the digital-topology euler number (loops minus
components) or the vertical-stroke fragmentation -- while keeping every other
incumbent edge and keeping <=1 feature parent per node.

Each candidate changes only the edges this hypothesis requires.  This is a
curated design list (8 hand-written structures with a stated reason each), the
same methodology v3 used; it is NOT an automatic structure search.

Feature indices (incumbent names):
  0 tl, 1 tc, 2 tr, 3 ml, 4 mc, 5 mr, 6 bl, 7 br  (central 3x3 minus bottom-centre)
  8 euler, 9 vertical_runs, 10 aspect, 11 vertical_dominance
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

# (key, parents, one-line hypothesis)
CANDIDATES = {
    # ---------------- control ----------------
    "M0_control_v3": (
        (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9),
        "v3 incumbent row chain (control)"),

    # ---------------- single-hypothesis candidates ----------------
    "M1_euler_midband": (
        (-1, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9),
        "euler conditions ml and mr: loop/component structure tells whether the "
        "middle band is isolated (2/3), looped (3), or a crossing stroke (5)"),

    "M2_vruns_midband": (
        (-1, 0, 1, 9, 3, 9, 7, 5, -1, 8, 9, 9),
        "vertical-stroke fragmentation conditions ml and mr: one tall stroke "
        "(1/7) vs a broken up-down stroke (3/5/2)"),

    "M3_aspect_midband": (
        (-1, 0, 1, 10, 3, 10, 7, 5, -1, 8, 9, 9),
        "global width/height proportion conditions the middle band (wide vs "
        "narrow glyphs)" ),

    "M4_euler_center": (
        (-1, 0, 1, -1, 8, 8, 7, 5, -1, 8, 9, 9),
        "euler conditions mc and mr: the right/centre of the middle band is "
        "where 3's loop and 5's top bar are decided"),

    "M5_euler_all_middle": (
        (-1, 0, 1, 8, 8, 8, 7, 5, -1, 8, 9, 9),
        "euler conditions the whole middle band (ml, mc, mr) -- the strongest "
        "form of the hypothesis"),

    "M6_euler_middle_and_diag": (
        (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9),
        "M1 plus a diagonal ink-balance edge tl<-br (open-top vs closed-bottom)"),

    "M7_topology_hub_semantic": (
        (-1, 0, 1, 8, 3, 8, 7, 5, -1, 8, 8, 8),
        "M1 plus aspect and vdom also conditioned on euler (single topology hub "
        "for all four semantic features)"),

    "M8_euler_middle_bl_br": (
        (-1, 0, 1, 8, 3, 8, 5, 5, -1, 8, 9, 9),
        "M1 plus bl<-mr (bottom-left hangs off the middle band, matching the "
        "incumbent's br<-mr reading)"),
}


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)

    preds, rows = {}, []
    for key, (parents, story) in CANDIDATES.items():
        validate_structure(NAMES, CARDS, parents)
        acc, pred, _, _, _ = evaluate(DESIGN, parents, ctr, cva, ytr, yva)
        preds[key] = pred
        rows.append((acc, key, parents, story))

    base = preds["M0_control_v3"]
    print("=== hand-authored DAG family (validation, seed 1322) ===")
    for acc, key, parents, story in sorted(rows, reverse=True):
        mc = mcnemar(yva, base, preds[key])
        bs = paired_bootstrap(yva, base, preds[key])
        print(f"  {acc:.4f}  {key:28s} delta={bs['delta']:+.4f} "
              f"CI=[{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}] McNemar p={mc['p_value']:.4f}")
        print(f"            {story}")

    print("\n=== legality (<=12 features, 2-5 states, >=2 edges, <=2 parents, acyclic) ===")
    for acc, key, parents, story in rows:
        print(f"  {key:28s} edges={sum(1 for p in parents if p>=0)} parents={parents}")

    np.save(HERE / "cache" / "manual_preds.npy", preds, allow_pickle=True)


if __name__ == "__main__":
    main()
