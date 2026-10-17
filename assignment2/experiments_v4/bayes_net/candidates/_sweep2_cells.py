"""Sweep 2: which grid cells earn a slot (subset search + matching row chain).

The incumbent spends 8 slots on 8 of the 9 central cells.  Here we ask, on the
validation split, which SUBSET of the 9 cells is best when the freed slots are
re-filled with the PDF-named topology features.  Every configuration is a
complete, valid 12-feature design with a hand-designed reading-order row chain
(the same pattern the incumbent uses: within each image row the kept cells
chain left -> right, and the bottom row closes through its rightmost cell).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import build_cache, split_columns  # noqa: E402
from _harness import evaluate, validate_structure  # noqa: E402

# 3x3 central grid, row-major
GRID = ["dens_tl", "dens_tc", "dens_tr",
        "dens_ml", "dens_mc", "dens_mr",
        "dens_bl", "dens_bc", "dens_br"]
GRID_RC = {g: (i // 3, i % 3) for i, g in enumerate(GRID)}
INCUMBENT_KEEP = ("dens_tl", "dens_tc", "dens_tr", "dens_ml", "dens_mc",
                  "dens_mr", "dens_bl", "dens_br")

SEMANTIC_4 = [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
              ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]


def template_chain(cells):
    """TEMPLATE A -- raster-order predecessor chain.

    Kept cells are ordered by (row band, column).  Each cell's parent is the
    immediately preceding kept cell in that raster order; the first kept cell
    is a chain root.  Semantic story: a glyph is written top-to-bottom and
    left-to-right, so the local ink amount at a cell is most predictive of the
    cell written just before it.  Acyclic by construction (parent always has a
    smaller raster index).
    """
    pos = {c: GRID_RC[c] for c in cells}
    order = sorted(range(len(cells)), key=lambda i: pos[cells[i]])
    parents = [-1] * len(cells)
    for k in range(1, len(order)):
        parents[order[k]] = order[k - 1]
    return tuple(parents)


def template_incumbent(cells):
    """TEMPLATE B -- the incumbent's band reading order + bottom-row closure.

    Rows of the central grid are handled as bands.  For every band except the
    bottom one, the kept cells chain left -> right (writing order) and the band's
    leftmost cell is a chain root.  For the bottom band the chain runs
    right -> left and its rightmost cell hangs off the nearest kept cell in the
    band above; this is exactly the incumbent's `br <- mr`, `bl <- br` closure
    (bottom strokes curve back toward the pen-up point).  On the incumbent's
    8-cell set this reproduces FEATURE_PARENTS = (-1,0,1,-1,3,4,7,5,...).
    """
    pos = {c: GRID_RC[c] for c in cells}
    rows = {}
    for i, c in enumerate(cells):
        rows.setdefault(pos[c][0], []).append(i)
    for r in rows:
        rows[r].sort(key=lambda i: pos[cells[i]][1])
    bottom = max(rows)
    parents = [-1] * len(cells)

    def nearest_above(i, r_ref):
        above = [j for rr in rows if rr < r_ref for j in rows[rr]]
        if not above:
            return -1
        rmax = max(pos[cells[j]][0] for j in above)
        tier = [j for j in above if pos[cells[j]][0] == rmax]
        return min(tier, key=lambda j: abs(pos[cells[j]][1] - pos[cells[i]][1]))

    for r, members in rows.items():
        if r == bottom:
            # right -> left chain; rightmost anchors upward
            for a, b in zip(members[::-1], members[::-1][1:]):
                parents[b] = a
            parents[members[-1]] = nearest_above(members[-1], r)
        else:
            # left -> right chain; leftmost stays a root
            for a, b in zip(members, members[1:]):
                parents[b] = a
    return tuple(parents)


TEMPLATES = {"chain": template_chain, "incumbent": template_incumbent}


def _acyclic(n, parents):
    indeg = np.zeros(n, int)
    kids = {i: [] for i in range(n)}
    for child, parent in enumerate(parents):
        if parent >= 0:
            kids[parent].append(child)
            indeg[child] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while ready:
        u = ready.pop()
        seen += 1
        for v in kids[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return seen == n


def full_parents(cells, template):
    grid_part = TEMPLATES[template](cells)
    n_grid = len(cells)
    # semantic chain: euler -> vertical_runs -> {aspect, vertical_dominance}
    semantic_part = (-1,
                     n_grid + 0,          # vertical_runs <- euler
                     n_grid + 1,          # aspect        <- vertical_runs
                     n_grid + 1)          # vdom          <- vertical_runs
    return tuple(grid_part) + semantic_part


def evaluate_subset(cells, ctr, cva, ytr, yva, semantic=SEMANTIC_4, template="incumbent"):
    design = [(c, 5, "zero_quantile") for c in cells] + list(semantic)
    parents = full_parents(cells, template)
    validate_structure([d[0] for d in design], [d[1] for d in design], parents)
    acc, _, _, _, _ = evaluate(design, parents, ctr, cva, ytr, yva)
    return acc


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)

    base = evaluate_subset(INCUMBENT_KEEP, ctr, cva, ytr, yva)
    print(f"incumbent 8-cell set (template B) = {base:.4f}\n")

    results = []
    for n_keep in (8, 7, 6):
        combos = list(itertools.combinations(GRID, n_keep))
        for cells in combos:
            for template in ("incumbent", "chain"):
                acc = evaluate_subset(cells, ctr, cva, ytr, yva, template=template)
                results.append((acc, n_keep, cells, template))
        best = max(r for r in results if r[1] == n_keep)
        print(f"n_keep={n_keep}: {len(combos)} subsets x 2 templates, "
              f"best={best[0]:.4f} {best[2]} ({best[3]})")

    results.sort(reverse=True)
    print("\n--- top 20 subsets (euler+vruns+aspect+vdom fixed) ---")
    for acc, n, cells, template in results[:20]:
        print(f"  {acc:.4f}  n={n} {template:9s}  {cells}")


if __name__ == "__main__":
    main()
