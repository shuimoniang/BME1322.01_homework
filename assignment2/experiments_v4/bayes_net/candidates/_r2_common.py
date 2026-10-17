"""Round-2 confirmatory harness: 8-seed adjudication with pooled McNemar + Bonferroni.

All candidate designs are NAMED members of small explicit families (no search
selects a structure); the pooled McNemar uses only the 7 held-out seeds so that
the seed-1322 search split is not counted as evidence.  Multiplicity is
corrected with Bonferroni over the size of the declared family.
"""
from __future__ import annotations

import itertools
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V4 = HERE.parents[1]
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

from v4_common import mcnemar, paired_bootstrap  # noqa: E402

SEEDS = (1322, 7, 2024, 1, 42, 99, 123, 555)
SEARCH_SEED = 1322
HELDOUT = tuple(s for s in SEEDS if s != SEARCH_SEED)

M6 = (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)
INC = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)

# incumbent grid slot order -> (absolute key, semantically matched bbox key)
GRID = [("dens_tl", "bbTL"), ("dens_tc", "bbTC"), ("dens_tr", "bbTR"),
        ("dens_ml", "bbML"), ("dens_mc", "bbMC"), ("dens_mr", "bbMR"),
        ("dens_bl", "bbBL"), ("dens_br", "bbBR")]
GRID_SLOT_NAMES = [g[0].replace("dens_", "") for g in GRID]

SEMANTIC = [("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
            ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]

# extra content keys that are not a bbox cell but a content-relative statistic
CONTENT_EXTRA = [("bbCR", "quantile"), ("bbCC", "quantile"), ("bbPR", "quantile"), ("bbPC", "quantile"),
                 ("bbTop", "zero_quantile"), ("bbMid", "zero_quantile"), ("bbBot", "zero_quantile"),
                 ("bbCLeft", "zero_quantile"), ("bbCMid", "zero_quantile"), ("bbCRight", "zero_quantile"),
                 ("bbQTL", "zero_quantile"), ("bbQTR", "zero_quantile"),
                 ("bbQBL", "zero_quantile"), ("bbQBR", "zero_quantile"),
                 ("bbBC", "zero_quantile")]


def pooled_mcnemar(pred_a_by_seed, pred_b_by_seed, y_by_seed, seeds):
    b_tot = c_tot = 0
    per_seed = {}
    for s in seeds:
        mc = mcnemar(y_by_seed[s], pred_a_by_seed[s], pred_b_by_seed[s])
        per_seed[s] = mc
        b_tot += mc["b"]
        c_tot += mc["c"]
    n = b_tot + c_tot
    if n == 0:
        p_one = 1.0
    else:
        k = min(b_tot, c_tot)
        p_one = min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b": b_tot, "c": c_tot, "n": n, "p_one": p_one,
            "per_seed": per_seed}


class Runner:
    """8-seed evaluation over the two caches."""

    def __init__(self):
        from _features import build_cache as abuild, split_columns as asplit
        from _r2_content import build_cache as cbuild, split_columns as csplit
        self.abs = abuild()
        self.con = cbuild()
        self._asplit, self._csplit = asplit, csplit
        self.cache = {}

    def world(self, seed):
        if seed in self.cache:
            return self.cache[seed]
        atr, ytr, ava, yva = self._asplit(self.abs, seed)
        ctr, cy, cva, cyva = self._csplit(self.con, seed)
        assert np.array_equal(cy, ytr) and np.array_equal(cyva, yva)
        w = (atr, ytr, ava, yva, ctr, cva)
        self.cache[seed] = w
        return w

    def run(self, spec, parents, seed):
        """spec: 12 entries (key, K, disc, src). Returns (acc, pred)."""
        from _harness import DISCRETISERS, SweepBN
        from v4_common import accuracy
        atr, ytr, ava, yva, ctr, cva = self.world(seed)
        tr, va, cards, names = [], [], [], []
        for key, k, disc, src in spec:
            cols_tr = ctr if src == "cont" else atr
            cols_va = cva if src == "cont" else ava
            a, b, _ = DISCRETISERS[disc](np.asarray(cols_tr[key], np.float64),
                                         np.asarray(cols_va[key], np.float64), k)
            tr.append(a); va.append(b); cards.append(k); names.append(key)
        Ftr = np.stack(tr, 1); Fva = np.stack(va, 1)
        m = SweepBN(names, cards, parents).fit(Ftr, ytr)
        pred = m.predict(Fva)
        return accuracy(yva, pred), pred

    def spec(self, content_repl, semantic=SEMANTIC):
        """content_repl: {slot_index: (key, K, disc)} over 0..7 grid slots."""
        s = []
        for i, (akey, _bbkey) in enumerate(GRID):
            if i in content_repl:
                key, k, disc = content_repl[i]
                s.append((key, k, disc, "cont"))
            else:
                s.append((akey, 5, "zero_quantile", "abs"))
        s += [(k, K, d, "abs") for k, K, d in semantic]
        return s

    def evaluate_family(self, designs, parents, seeds=SEEDS):
        """designs: {name: content_repl}. Returns {name: {'acc':{seed:..}, 'pred':{seed:..}, 'y':{seed:..}}}"""
        y_by, ref_pred, ref_acc = {}, {}, {}
        for s in seeds:
            _, ytr, _, yva, _, _ = self.world(s)
            y_by[s] = yva
            acc, pred = self.run(self.spec({}), M6, s)
            ref_pred[s] = pred
            ref_acc[s] = acc
        out = {"M6_reference": {"acc": ref_acc, "pred": ref_pred, "y": y_by}}
        for name, repl in designs.items():
            accs, preds = {}, {}
            for s in seeds:
                accs[s], preds[s] = self.run(self.spec(repl), parents, s)
            out[name] = {"acc": accs, "pred": preds, "y": y_by}
        return out

    @staticmethod
    def summarise(res, names, m=1, ref="M6_reference"):
        rows = []
        for name in names:
            accs = [res[name]["acc"][s] for s in SEEDS]
            ho = [res[name]["acc"][s] for s in HELDOUT]
            refs = [res[ref]["acc"][s] for s in SEEDS]
            pm = pooled_mcnemar(res[ref]["pred"], res[name]["pred"], res[name]["y"], HELDOUT)
            p_bonf = min(1.0, pm["p_one"] * m)
            beats = sum(1 for a, b in zip(accs, refs) if a > b)
            rows.append({
                "name": name,
                "mean": float(np.mean(accs)), "min": float(np.min(accs)),
                "mean_ho": float(np.mean(ho)), "min_ho": float(np.min(ho)),
                "mean_delta": float(np.mean([a - b for a, b in zip(accs, refs)])),
                "beats": beats,
                "pooled_b": pm["b"], "pooled_c": pm["c"],
                "p_one_heldout": pm["p_one"], "p_bonferroni": p_bonf,
                "acc": accs,
            })
        return rows
