"""Round-2 idea (A) screen: content-relative features (bbox-normalised).

Stage A1: I(F;Y) of every content-relative column at K=5 (train only).
Stage A2: single-slot substitution of each content feature into each GRID slot,
          scored under BOTH the incumbent DAG and the M6 DAG (seed 1322).
Stage A3: best two-at-a-time substitutions (replace 2 grid cells), seed 1322.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V4 = HERE.parents[1]
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

from v4_common import mcnemar, accuracy  # noqa: E402
from _features import build_cache, split_columns  # noqa: E402
from _harness import mutual_information, DISCRETISERS, SweepBN  # noqa: E402
from _r2_content import build_cache as build_content, split_columns as split_content  # noqa: E402

INC = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)
M6 = (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)

ABS_GRID = ["dens_tl", "dens_tc", "dens_tr", "dens_ml", "dens_mc", "dens_mr", "dens_bl", "dens_br"]
SEMANTIC = [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
            ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]

CONTENT_DENSITY = ["bbTL", "bbTC", "bbTR", "bbML", "bbMC", "bbMR", "bbBL", "bbBC", "bbBR",
                   "bbTop", "bbMid", "bbBot", "bbCLeft", "bbCMid", "bbCRight",
                   "bbQTL", "bbQTR", "bbQBL", "bbQBR"]
CONTENT_POS = ["bbCR", "bbCC", "bbPR", "bbPC"]


class World:
    """Holds the two caches sliced to a seed; discretises and fits on demand."""

    def __init__(self, seed, with_content=True):
        a = build_cache()
        _, _, ava, _ = split_columns(a, seed)
        self.abs_tr, self.y_tr, self.abs_va, self.y_va = split_columns(a, seed)
        if with_content:
            c = build_content()
            self.con_tr, cy, self.con_va, cyva = split_content(c, seed)
            assert np.array_equal(cy, self.y_tr) and np.array_equal(cyva, self.y_va)
        else:
            self.con_tr = self.con_va = {}

    def disc(self, key, k, disc, src):
        cols_tr = self.con_tr if src == "cont" else self.abs_tr
        cols_va = self.con_va if src == "cont" else self.abs_va
        f = DISCRETISERS[disc]
        a, b, _ = f(np.asarray(cols_tr[key], np.float64),
                    np.asarray(cols_va[key], np.float64), k)
        return a, b

    def run(self, spec, parents):
        """spec: [(key, K, disc, src)] -> (acc, predictions)."""
        tr, va, cards, names = [], [], [], []
        for key, k, disc, src in spec:
            a, b = self.disc(key, k, disc, src)
            tr.append(a); va.append(b); cards.append(k); names.append(key)
        Ftr = np.stack(tr, 1); Fva = np.stack(va, 1)
        m = SweepBN(names, cards, parents).fit(Ftr, self.y_tr)
        return accuracy(self.y_va, m.predict(Fva)), m.predict(Fva)


BASE_SPEC = [(k, 5, "zero_quantile", "abs") for k in ABS_GRID] + \
            [(k, K, d, "abs") for k, K, d in SEMANTIC]


def main():
    w = World(1322)

    print("=== Stage A1: I(F;Y) (K=5) ===")
    mi_rows = []
    for k in CONTENT_DENSITY:
        a, _ = w.disc(k, 5, "zero_quantile", "cont")
        mi_rows.append((mutual_information(a, w.y_tr, 5), k))
    for k in CONTENT_POS:
        a, _ = w.disc(k, 5, "quantile", "cont")
        mi_rows.append((mutual_information(a, w.y_tr, 5), k))
    for k in ABS_GRID:
        a, _ = w.disc(k, 5, "zero_quantile", "abs")
        mi_rows.append((mutual_information(a, w.y_tr, 5), "ABS_" + k))
    for name, k in [("euler", ("euler", 4, "euler_incumbent", "abs")),
                    ("vertical_runs", ("vertical_runs", 5, "zero_quantile", "abs")),
                    ("aspect", ("aspect_incumbent", 4, "zero_quantile", "abs")),
                    ("vdom", ("vertical_dominance", 4, "zero_quantile", "abs"))]:
        a, _ = w.disc(k[0], k[1], k[2], k[3])
        mi_rows.append((mutual_information(a, w.y_tr, k[1]), "SEM_" + name))
    for mi, k in sorted(mi_rows, reverse=True):
        print(f"  I(F;Y)={mi:.4f}  {k}")

    print("\n=== Stage A2: single GRID-slot substitution (seed 1322) ===")
    a_inc, p_inc = w.run(BASE_SPEC, INC)
    a_m6, p_m6 = w.run(BASE_SPEC, M6)
    print(f"  baseline: incumbent DAG {a_inc:.4f}   M6 DAG {a_m6:.4f}")

    cands = [(k, 5, "zero_quantile", "cont") for k in CONTENT_DENSITY] + \
            [(k, 5, "quantile", "cont") for k in CONTENT_POS]
    rows = []
    for slot in range(8):
        for key, k, disc, src in cands:
            spec = list(BASE_SPEC)
            spec[slot] = (key, k, disc, src)
            am6, pm = w.run(spec, M6)
            ainc, _ = w.run(spec, INC)
            rows.append((am6, ainc, key, slot, ABS_GRID[slot], disc, pm))
    rows.sort(key=lambda r: -r[0])
    print(f"  {'content key':12s} {'slot':>4s} {'replaces':18s} {'M6 DAG':>8s} {'inc DAG':>8s}")
    for am6, ainc, key, slot, replaced, disc, pm in rows[:20]:
        print(f"  {key:12s} {slot:4d} {replaced:18s} {am6:8.4f} {ainc:8.4f}")

    print("\n  --- best content feature PER SLOT ---")
    for slot in range(8):
        sub = [r for r in rows if r[3] == slot]
        am6, ainc, key, s, replaced, disc, pm = max(sub, key=lambda r: r[0])
        print(f"  slot {slot} ({replaced:14s}) -> {key:10s} M6={am6:.4f} inc={ainc:.4f}")

    print("\n=== Stage A3: two-grid-slot substitution (top-4 content per slot, M6) ===")
    top_per_slot = {s: sorted([r for r in rows if r[3] == s], key=lambda r: -r[0])[:4]
                    for s in range(8)}
    pairs = []
    for s1, s2 in itertools.combinations(range(8), 2):
        for r1 in top_per_slot[s1]:
            for r2 in top_per_slot[s2]:
                spec = list(BASE_SPEC)
                spec[s1] = (r1[2], 5, r1[5], "cont")
                spec[s2] = (r2[2], 5, r2[5], "cont")
                am6, pm = w.run(spec, M6)
                pairs.append((am6, r1[2], s1, r2[2], s2, pm))
    pairs.sort(key=lambda x: -x[0])
    print(f"  {'pair':46s} {'M6 DAG':>8s} {'McNemar p vs M6':>16s}")
    for am6, k1, s1, k2, s2, pm in pairs[:15]:
        mc = mcnemar(w.y_va, p_m6, pm)
        print(f"  {k1:10s}({ABS_GRID[s1][5:8]}) + {k2:10s}({ABS_GRID[s2][5:8]}) "
              f"{am6:8.4f} {mc['p_value']:16.4f}")
    print(f"  (M6 reference {a_m6:.4f})")


if __name__ == "__main__":
    main()
