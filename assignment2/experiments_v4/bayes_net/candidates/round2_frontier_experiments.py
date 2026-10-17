"""Reproduce every number in results/round2_frontier.md.

Round-2 frontier push on the Small Bayesian Network, strictly inside the PDF's
section 4.2 rules.  Usage:

    python _r2_round2_frontier.py            # all sections
    python _r2_round2_frontier.py --quick    # seed 1322 only

Sections:
  0  protocol / reproduction of incumbent (0.8614) and M6 (0.8686)
  A  content-relative (bbox-normalised) features: family screen + 8-seed gate
     A-control: translate-only vs scale-only decomposition
     A-robust : threshold / discretiser / cardinality robustness
  B  named structural refinements on M6 (single hand-authored edits)
  B-estimator: class prior empirical vs uniform; backoff vs uniform Lidstone
  1  end-to-end head-to-head of the final candidate vs M6 (real modules)
  2  check_bn_bounds audit of the final candidate

Only the 60,000 training images are ever read (v4_common.load_split); the test
set is never touched.  All structures are named members of small hand-authored
families -- no automatic structure learning.
"""
from __future__ import annotations

import itertools
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
V4 = HERE.parents[1]
if str(V4) not in sys.path:
    sys.path.insert(0, str(V4))

from v4_common import accuracy, mcnemar, check_bn_bounds, load_split  # noqa: E402
from _r2_common import Runner, M6, INC, GRID, SEEDS, HELDOUT, SEMANTIC, pooled_mcnemar  # noqa: E402

QUICK = "--quick" in sys.argv
RUN_SEEDS = (1322,) if QUICK else SEEDS
RUN_HELDOUT = tuple(s for s in RUN_SEEDS if s != 1322)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(rows, m, limit=None):
    rows = sorted(rows, key=lambda r: -r["mean_delta"])
    print(f"  {'variant':34s} {'mean':>8s} {'min':>8s} {'d':>9s} {'beats':>5s} "
          f"{'b/c':>12s} {'p_one':>10s} {'p_bonf':>9s}")
    for r in rows[:limit]:
        print(f"  {r['name']:34s} {r['mean']:8.4f} {r['min']:8.4f} {r['mean_delta']:+9.4f} "
              f"{r['beats']:5d} {str(r['pooled_b'])+'/'+str(r['pooled_c']):>12s} "
              f"{r['p_one_heldout']:10.3e} {r['p_bonferroni']:9.4f}")


def main():
    R = Runner()
    report = {}

    # ------------------------------------------------------------------
    section("0. Protocol reproduction (incumbent vs M6, all seeds)")
    inc_acc, m6_acc = {}, {}
    for s in RUN_SEEDS:
        inc_acc[s], _ = R.run(R.spec({}), INC, s)
        m6_acc[s], _ = R.run(R.spec({}), M6, s)
        print(f"  seed {s}: incumbent={inc_acc[s]:.4f}  M6={m6_acc[s]:.4f}")
    report["incumbent"] = inc_acc
    report["m6"] = m6_acc

    # ------------------------------------------------------------------
    section("A. Idea A: content-relative (bbox-normalised) grid densities")
    def fam(k):
        out = {}
        for combo in itertools.combinations(range(8), k):
            name = "bbox_" + "+".join(GRID[i][0].replace("dens_", "") for i in combo)
            out[name] = {i: (GRID[i][1], 5, "zero_quantile") for i in combo}
        return out

    res = {}
    for k in (1, 2, 3, 4, 5, 6, 7, 8):
        res.update(R.evaluate_family(fam(k), M6, seeds=RUN_SEEDS))
    report["ideaA"] = {n: [res[n]["acc"][s] for s in RUN_SEEDS] for n in res if n != "M6_reference"}

    a_names = [n for n in res if n != "M6_reference"]
    a_rows = Runner.summarise(res, a_names, m=92)
    print(f"  (whole family m=92; showing best per k and the uniform k=8)")
    for k in (1, 2, 3, 4, 5, 6, 7, 8):
        sub = [r for r in a_rows if r["name"].count("+") == k - 1]
        show(sub, 92, limit=1)
    print("\n  -- full k=2 (m=28) --")
    show([r for r in a_rows if r["name"].count("+") == 1], 28, limit=5)
    print("\n  -- full k=5 (m=56) --")
    show([r for r in a_rows if r["name"].count("+") == 4], 56, limit=5)
    best_a = max(a_rows, key=lambda r: r["mean"])
    print(f"\n  best single-family member: {best_a['name']}  mean={best_a['mean']:.4f} "
          f"p_bonf(m=92)={best_a['p_bonferroni']:.3e}")

    # ------------------------------------------------------------------
    section("A-control: translate-only vs scale-only decomposition")
    from _r2_content import build_controls
    from _harness import DISCRETISERS, SweepBN
    from v4_common import stratified_split
    CF = build_controls()
    ctrl = {}
    for s in RUN_SEEDS:
        yy = np.asarray(CF["label"]); tr, va = stratified_split(yy, 1 / 6, s)
        ctrl[s] = ({k: np.asarray(CF[k])[tr] for k in CF if k != "label"},
                   {k: np.asarray(CF[k])[va] for k in CF if k != "label"})

    def run_ctrl(grid8, parents, s):
        atr, ytr, ava, yva, ctr, cva = R.world(s)
        ctrc, cvac = ctrl[s]
        tr, va, cards, names = [], [], [], []
        for key, k, disc, src in grid8 + [(k, K, d, "abs") for k, K, d in SEMANTIC]:
            CTr = {"abs": atr, "cont": ctr, "ctrl": ctrc}[src]
            CVa = {"abs": ava, "cont": cva, "ctrl": cvac}[src]
            a, b, _ = DISCRETISERS[disc](np.asarray(CTr[key], float), np.asarray(CVa[key], float), k)
            tr.append(a); va.append(b); cards.append(k); names.append(key)
        m = SweepBN(names, cards, parents).fit(np.stack(tr, 1), ytr)
        return accuracy(yva, m.predict(np.stack(va, 1)))

    for label, repl in [("k2_mr+br", {5, 7}), ("k5", {2, 3, 5, 6, 7})]:
        print(f"\n  -- {label}: absolute vs bbox vs translate-only vs scale-only --")
        for tag, mk in [("ABS    ", lambda i: (GRID[i][0], 5, "zero_quantile", "abs")),
                        ("BBOX   ", lambda i: (GRID[i][1], 5, "zero_quantile", "cont")),
                        ("TRANSL ", lambda i: ("tr" + GRID[i][0].replace("dens_", "").upper(), 5, "zero_quantile", "ctrl")),
                        ("SCALE  ", lambda i: ("sc" + GRID[i][0].replace("dens_", "").upper(), 5, "zero_quantile", "ctrl"))]:
            accs = [run_ctrl([mk(i) if i in repl else (GRID[i][0], 5, "zero_quantile", "abs") for i in range(8)], M6, s) for s in RUN_SEEDS]
            print(f"    {tag} mean={np.mean(accs):.4f} min={np.min(accs):.4f}")

    # ------------------------------------------------------------------
    section("A-robust: threshold / discretiser / cardinality robustness (k=2 and k=5)")
    from v4_common import load_train
    import _r2_content as RC
    X_all, _ = load_train()
    yy = np.asarray(RC.build_cache()["label"])
    print("  threshold sweep (k=2 mr+br, bbox extraction):")
    for thr in (48, 64, 80, 96, 128):
        RC.BINARY_THRESHOLD = thr
        feats = RC.content_features(X_all)
        for s in RUN_SEEDS[:2]:
            tr, va = stratified_split(yy, 1 / 6, s)
            atr, ytr, ava, yva, ctr, cva = R.world(s)
            ctr2 = {k: feats[k][tr] for k in feats}; cva2 = {k: feats[k][va] for k in feats}
            spec = []
            for i in range(8):
                if i in (5, 7):
                    spec.append((GRID[i][1], 5, "zero_quantile", "cont"))
                else:
                    spec.append((GRID[i][0], 5, "zero_quantile", "abs"))
            spec += [(k, K, d, "abs") for k, K, d in SEMANTIC]
            trl, val, cards, names = [], [], [], []
            for key, k, disc, src in spec:
                CTr = ctr2 if src == "cont" else atr
                CVa = cva2 if src == "cont" else ava
                a, b, _ = DISCRETISERS[disc](np.asarray(CTr[key], float), np.asarray(CVa[key], float), k)
                trl.append(a); val.append(b); cards.append(k); names.append(key)
            m = SweepBN(names, cards, M6).fit(np.stack(trl, 1), ytr)
            print(f"    thr={thr:3d} seed {s}: bbox_mr+br={accuracy(yva, m.predict(np.stack(val, 1))):.4f}")
    RC.BINARY_THRESHOLD = 64

    print("  discretiser x cardinality (8-seed mean):")
    for label, repl in [("k2_mr+br", {5, 7}), ("k5", {2, 3, 5, 6, 7})]:
        for disc in ("zero_quantile", "quantile"):
            for kk in (4, 5):
                accs = []
                for s in RUN_SEEDS:
                    atr, ytr, ava, yva, ctr, cva = R.world(s)
                    spec = [(GRID[i][1], kk, disc, "cont") if i in repl else (GRID[i][0], 5, "zero_quantile", "abs") for i in range(8)]
                    spec += [(k, K, d, "abs") for k, K, d in SEMANTIC]
                    trl, val, cards, names = [], [], [], []
                    for key, k, d, src in spec:
                        CTr = ctr if src == "cont" else atr; CVa = cva if src == "cont" else ava
                        a, b, _ = DISCRETISERS[d](np.asarray(CTr[key], float), np.asarray(CVa[key], float), k)
                        trl.append(a); val.append(b); cards.append(k); names.append(key)
                    m = SweepBN(names, cards, M6).fit(np.stack(trl, 1), ytr)
                    accs.append(accuracy(yva, m.predict(np.stack(val, 1))))
                print(f"    {label:10s} {disc:14s} K={kk}: mean={np.mean(accs):.4f} min={np.min(accs):.4f}")

    # ------------------------------------------------------------------
    section("B. Idea B: named single-edit structural refinements on M6")
    def cards_spec(aspect_k, vdom_k):
        s = list(R.spec({}))
        s[10] = ("aspect_incumbent", aspect_k, "zero_quantile", "abs")
        s[11] = ("vertical_dominance", vdom_k, "zero_quantile", "abs")
        return s

    def repoint(child, new_parent):
        p = list(M6); p[child] = new_parent; return tuple(p)

    chain = list(M6); chain[10] = 11; chain[11] = 8
    fam_b = {
        "B1_aspect5": (cards_spec(5, 4), M6),
        "B2_vdom5": (cards_spec(4, 5), M6),
        "B3_both5": (cards_spec(5, 5), M6),
        "B4_mc_vruns": (R.spec({}), repoint(4, 9)),
        "B5_mr_vruns": (R.spec({}), repoint(5, 9)),
        "B6_tc_euler": (R.spec({}), repoint(1, 8)),
        "B7_bl_euler": (R.spec({}), repoint(6, 8)),
        "B8_orientation_chain": (R.spec({}), tuple(chain)),
    }
    bres = {"M6_reference": None}
    ref_acc, ref_pred, yby = {}, {}, {}
    for s in RUN_SEEDS:
        _, ytr, _, yva, _, _ = R.world(s); yby[s] = yva
        ref_acc[s], ref_pred[s] = R.run(R.spec({}), M6, s)
    bres["M6_reference"] = {"acc": ref_acc, "pred": ref_pred, "y": yby}
    for name, (spec, parents) in fam_b.items():
        accs, preds = {}, {}
        for s in RUN_SEEDS:
            accs[s], preds[s] = R.run(spec, parents, s)
        bres[name] = {"acc": accs, "pred": preds, "y": yby}
    b_rows = Runner.summarise(bres, list(fam_b), m=len(fam_b))
    show(b_rows, len(fam_b))
    report["ideaB_structure"] = {n: bres[n]["acc"] for n in fam_b}
    survivor = [r["name"] for r in b_rows if r["mean_delta"] > 0 and r["p_bonferroni"] < 0.05]
    print(f"  --> clears corrected gate: {survivor or 'NONE'}")

    # ------------------------------------------------------------------
    section("B-estimator: class prior x CPT scheme under M6")
    from _r2_fitters import SweepBN2
    est = {}
    for s in RUN_SEEDS:
        atr, ytr, ava, yva, ctr, cva = R.world(s)
        tr, va, cards = [], [], []
        for key, k, disc, src in R.spec({}):
            CTr = ctr if src == "cont" else atr
            CVa = cva if src == "cont" else ava
            a, b, _ = DISCRETISERS[disc](np.asarray(CTr[key], float), np.asarray(CVa[key], float), k)
            tr.append(a); va.append(b); cards.append(k)
        Ftr = np.stack(tr, 1); Fva = np.stack(va, 1)
        for prior in ("empirical", "uniform"):
            for cpt in ("backoff", "uniform"):
                nm = f"{prior}_prior_{'backoff' if cpt == 'backoff' else 'lidstone'}"
                m = SweepBN2(cards, cards, M6, alpha=1.0, prior=prior, cpt=cpt).fit(Ftr, ytr)
                est.setdefault(nm, {})[s] = (accuracy(yva, m.predict(Fva)), m.predict(Fva), yva)
    ref = est["empirical_prior_backoff"]
    print(f"  {'variant':26s} {'mean':>8s} {'min':>8s} {'d':>9s} {'beats':>5s} {'p_one':>10s}")
    for nm in ("empirical_prior_backoff", "uniform_prior_backoff",
               "empirical_prior_lidstone", "uniform_prior_lidstone"):
        accs = [est[nm][s][0] for s in RUN_SEEDS]
        refs = [ref[s][0] for s in RUN_SEEDS]
        pm = pooled_mcnemar({s: ref[s][1] for s in RUN_SEEDS},
                            {s: est[nm][s][1] for s in RUN_SEEDS},
                            {s: est[nm][s][2] for s in RUN_SEEDS}, RUN_HELDOUT)
        beats = sum(1 for a, b in zip(accs, refs) if a > b)
        print(f"  {nm:26s} {np.mean(accs):8.4f} {np.min(accs):8.4f} "
              f"{np.mean([a-b for a,b in zip(accs,refs)]):+9.4f} {beats:5d} {pm['p_one']:10.3e}")
    report["ideaB_estimator"] = {nm: [est[nm][s][0] for s in RUN_SEEDS] for nm in est}

    # ------------------------------------------------------------------
    section("1. End-to-end head-to-head of the final candidate vs M6 (real modules)")
    import round2_best as RB
    import builder_c_dag_m6 as M6M
    e2e_new, e2e_m6 = [], []
    for s in RUN_SEEDS:
        Xtr, ytr, Xva, yva = load_split(seed=s)
        pn = RB.BayesNetClassifier(alpha=1.0).fit(Xtr, ytr).predict(Xva)
        p6 = M6M.BayesNetClassifier(alpha=1.0).fit(Xtr, ytr).predict(Xva)
        e2e_new.append(accuracy(yva, pn)); e2e_m6.append(accuracy(yva, p6))
        print(f"  seed {s}: M6={e2e_m6[-1]:.4f}  round2_best={e2e_new[-1]:.4f}  "
              f"d={e2e_new[-1]-e2e_m6[-1]:+.4f}")
    print(f"  mean M6={np.mean(e2e_m6):.4f}  mean round2_best={np.mean(e2e_new):.4f} "
          f" min round2_best={np.min(e2e_new):.4f}")
    report["final_8seed"] = e2e_new
    report["final_m6_8seed"] = e2e_m6

    section("2. check_bn_bounds audit of the final candidate")
    print(f"  FEATURE_NAMES      = {RB.FEATURE_NAMES}")
    print(f"  FEATURE_CARDINALITIES = {RB.FEATURE_CARDINALITIES}")
    print(f"  FEATURE_PARENTS    = {RB.FEATURE_PARENTS}")
    check_bn_bounds(RB.FEATURE_NAMES, RB.FEATURE_CARDINALITIES, RB.FEATURE_PARENTS)

    out = HERE / "cache" / "round2_frontier_raw.json"
    json.dump(report, open(out, "w"), indent=1, default=str)
    print(f"\nraw numbers -> {out}")


if __name__ == "__main__":
    main()
