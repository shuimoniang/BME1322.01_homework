"""Builder A: falsification + final verdict for the best candidate.

Everything a claim needs (per the brief):
  (a) cross_seed() on seeds 7 and 2024 plus the seed-1322 split;
  (b) compare_candidates() vs the incumbent factory -> McNemar p-value and
      paired-bootstrap 95% CI;
  (c) top_confusions() and which class pairs the gain/failure attaches to.

Only the 60,000 training images are ever loaded (via v4_common).  The test set
is never read.

Run:  python builder_a_validate.py
Writes results/builder_a_validation.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parent.parent
ROOT = V4.parent
for path in (str(HERE), str(V4), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import v4_common as V4C
from naive_bayes import NaiveBayesClassifier
import builder_a_candidate as candidate_module
from builder_a_candidate import BuilderANaiveBayes, BEST_ALPHA

RESULTS = HERE.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "builder_a_validation.json"

ALPHAS = [0.0001, 0.001, 0.005, 0.02, 0.05]
PRIMARY_ALPHA = BEST_ALPHA


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def incumbent_factory(alpha=0.05):
    return NaiveBayesClassifier(alpha=alpha)


def main():
    report = {"primary_alpha": PRIMARY_ALPHA, "alphas_tested": ALPHAS}
    global_start = time.perf_counter()

    # ---------------- seed 1322: head-to-head ----------------
    Xtr, ytr, Xva, yva = V4C.load_split()
    print(f"protocol: train={Xtr.shape[0]} val={Xva.shape[0]} seed=1322")

    inc = incumbent_factory().fit(Xtr, ytr)
    pred_inc = np.asarray(inc.predict(Xva))
    acc_inc = V4C.accuracy(yva, pred_inc)
    f1_inc, cm_inc = V4C.macro_f1(yva, pred_inc)
    print(f"incumbent alpha=0.05        val={acc_inc:.4f}  macroF1={f1_inc:.4f}")
    np.save(RESULTS / "builder_a_incumbent_pred.npy", pred_inc)

    head_to_head = {}
    for alpha in ALPHAS:
        model = BuilderANaiveBayes(alpha=alpha).fit(Xtr, ytr)
        pred = np.asarray(model.predict(Xva))
        acc = V4C.accuracy(yva, pred)
        f1, cm = V4C.macro_f1(yva, pred)
        mc = V4C.mcnemar(yva, pred_inc, pred)
        bs = V4C.paired_bootstrap(yva, pred_inc, pred)
        record = {
            "alpha": alpha,
            "val_accuracy": acc,
            "macro_f1": f1,
            "delta_vs_incumbent": acc - acc_inc,
            "mcnemar": mc,
            "bootstrap": bs,
            "real_gain": bool(acc > acc_inc and mc["p_value"] < 0.05),
            "top_confusions": V4C.top_confusions(np.asarray(cm), 6),
            "n_features": int(candidate_module.N_PATCH_FEATURES),
            "n_state_slots": int(candidate_module.N_PATCH_FEATURES
                                 * candidate_module.N_STATES),
            "n_observed_state_slots": int(model.n_observed_states_),
        }
        head_to_head[str(alpha)] = record
        print(f"candidate  alpha={alpha:<6} val={acc:.4f}  macroF1={f1:.4f}  "
              f"delta={acc-acc_inc:+.4f}  McNemar b={mc['b']} c={mc['c']} "
              f"p={mc['p_value']:.3g}  CI=[{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}]")
        if alpha == PRIMARY_ALPHA:
            np.save(RESULTS / "builder_a_candidate_pred.npy", pred)
    report["head_to_head_seed1322"] = head_to_head
    report["incumbent_seed1322"] = {
        "val_accuracy": acc_inc, "macro_f1": f1_inc,
        "top_confusions": V4C.top_confusions(np.asarray(cm_inc), 6),
    }

    # ---------------- incumbent alpha sensitivity (context) ----------------
    print("\nincumbent alpha sweep (context):")
    incumbent_alphas = {}
    for alpha in (0.01, 0.02, 0.05, 0.1):
        model = incumbent_factory(alpha).fit(Xtr, ytr)
        acc = V4C.accuracy(yva, model.predict(Xva))
        incumbent_alphas[str(alpha)] = acc
        print(f"  incumbent alpha={alpha:<6} val={acc:.4f}")
    report["incumbent_alpha_sweep_seed1322"] = incumbent_alphas

    # ---------------- cross-seed falsification ----------------
    print("\ncross-seed (extra splits) on the primary candidate:")
    cross = V4C.cross_seed(lambda: BuilderANaiveBayes(alpha=PRIMARY_ALPHA))
    cross_inc = V4C.cross_seed(lambda: incumbent_factory(0.05))
    # also the seed-1322 split for a clean 3-seed table
    cross_all = {"1322": head_to_head[str(PRIMARY_ALPHA)]["val_accuracy"], **{str(k): v for k, v in cross.items()}}
    cross_inc_all = {"1322": acc_inc, **{str(k): v for k, v in cross_inc.items()}}
    report["cross_seed"] = {
        "candidate": cross_all,
        "incumbent": cross_inc_all,
        "delta_per_seed": {k: cross_all[k] - cross_inc_all[k] for k in cross_all},
    }
    for key in cross_all:
        print(f"  seed {key:<5} incumbent={cross_inc_all[key]:.4f} "
              f"candidate={cross_all[key]:.4f} delta={cross_all[key]-cross_inc_all[key]:+.4f}")

    # ---------------- McNemar on both extra seeds ----------------
    print("\nMcNemar on the extra seeds:")
    extra = {}
    for seed in V4C.EXTRA_SEEDS:
        Xtr_s, ytr_s, Xva_s, yva_s = V4C.load_split(seed=seed)
        inc_s = incumbent_factory(0.05).fit(Xtr_s, ytr_s)
        cand_s = BuilderANaiveBayes(alpha=PRIMARY_ALPHA).fit(Xtr_s, ytr_s)
        pred_i = np.asarray(inc_s.predict(Xva_s))
        pred_c = np.asarray(cand_s.predict(Xva_s))
        mc = V4C.mcnemar(yva_s, pred_i, pred_c)
        bs = V4C.paired_bootstrap(yva_s, pred_i, pred_c)
        extra[str(seed)] = {"mcnemar": mc, "bootstrap": bs,
                            "incumbent_accuracy": V4C.accuracy(yva_s, pred_i),
                            "candidate_accuracy": V4C.accuracy(yva_s, pred_c)}
        print(f"  seed {seed:<5} b={mc['b']} c={mc['c']} p={mc['p_value']:.3g} "
              f"delta={bs['delta']:+.4f} CI=[{bs['ci_low']:+.4f},{bs['ci_high']:+.4f}]")
    report["extra_seed_mcnemar"] = extra

    # ---------------- decomposition: geometry vs smoothing ----------------
    # The incumbent also improves when its smoothing is driven to ~0, so the
    # honest attribution of the total gain matters.  Here the candidate is
    # compared against the incumbent *at its own best alpha*, isolating the
    # feature-geometry effect from the smoothing effect.
    print("\ndecomposition vs strongest smoothing-tuned incumbent (alpha=0.0001):")
    decomposition = {}
    for seed in (1322,) + tuple(V4C.EXTRA_SEEDS):
        Xtr_s, ytr_s, Xva_s, yva_s = V4C.load_split(seed=seed)
        inc_best = incumbent_factory(0.0001).fit(Xtr_s, ytr_s)
        cand = BuilderANaiveBayes(alpha=PRIMARY_ALPHA).fit(Xtr_s, ytr_s)
        pred_i = np.asarray(inc_best.predict(Xva_s))
        pred_c = np.asarray(cand.predict(Xva_s))
        mc = V4C.mcnemar(yva_s, pred_i, pred_c)
        bs = V4C.paired_bootstrap(yva_s, pred_i, pred_c)
        decomposition[str(seed)] = {
            "incumbent_alpha_1e-4": V4C.accuracy(yva_s, pred_i),
            "candidate": V4C.accuracy(yva_s, pred_c),
            "geometry_delta": bs["delta"],
            "mcnemar": mc,
            "bootstrap": bs,
        }
        print(f"  seed {seed:<5} inc_a1e-4={decomposition[str(seed)]['incumbent_alpha_1e-4']:.4f} "
              f"cand={decomposition[str(seed)]['candidate']:.4f} "
              f"geometry_delta={bs['delta']:+.4f} p={mc['p_value']:.3g}")
    report["decomposition_geometry_vs_smoothing"] = decomposition

    # error overlap at matched alpha, seed 1322
    inc_matched = incumbent_factory(0.001).fit(Xtr, ytr)
    pred_matched = np.asarray(inc_matched.predict(Xva))
    pred_cand = np.asarray(BuilderANaiveBayes(alpha=0.001).fit(Xtr, ytr).predict(Xva))
    e_inc = pred_matched != yva
    e_cand = pred_cand != yva
    report["error_overlap_matched_alpha_0.001"] = {
        "incumbent_errors": int(e_inc.sum()),
        "candidate_errors": int(e_cand.sum()),
        "both_wrong": int((e_inc & e_cand).sum()),
        "incumbent_only_wrong": int((e_inc & ~e_cand).sum()),
        "candidate_only_wrong": int((~e_inc & e_cand).sum()),
    }
    print(f"\nerror overlap @alpha=0.001: incumbent={int(e_inc.sum())} "
          f"candidate={int(e_cand.sum())} both={int((e_inc&e_cand).sum())} "
          f"inc-only={int((e_inc&~e_cand).sum())} cand-only={int((~e_inc&e_cand).sum())}")

    report["total_seconds"] = round(time.perf_counter() - global_start, 1)
    OUT.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nvalidation -> {OUT}  ({report['total_seconds']}s)")


if __name__ == "__main__":
    main()
