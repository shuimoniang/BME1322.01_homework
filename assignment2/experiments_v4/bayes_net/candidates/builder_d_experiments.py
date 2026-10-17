"""Builder D experiments: DAG structure + CPT smoothing / calibration for the
Small Bayesian Network, with the incumbent's 12 features FIXED.

All feature extraction reuses the incumbent's own private pipeline
(``BayesNetClassifier._continuous_features`` / ``_discretize``), so the
discretized features are bit-identical to ``assignment2/bayes_net.py``.
Continuous features are cached per split under ``candidates/cache/`` so a DAG /
smoothing variant only re-fits CPTs.

Protocol: ``v4_common.load_split`` -> 49,999 train / 10,001 validation, seed
1322, only the 60,000 training images are ever touched.

Run:  python builder_d_experiments.py --stages grid,smoothing,regress
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BAYES_NET_DIR = HERE.parent               # experiments_v4/bayes_net
V4 = BAYES_NET_DIR.parent                 # experiments_v4
ROOT = V4.parent                          # assignment2
CACHE = HERE / "cache"
RESULTS = BAYES_NET_DIR / "results"
for path in (str(V4), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from v4_common import (  # noqa: E402
    EXTRA_SEEDS,
    accuracy,
    compare_candidates,
    cross_seed,
    load_split,
    macro_f1,
    mcnemar,
    paired_bootstrap,
    top_confusions,
)
from bayes_net import (  # noqa: E402
    FEATURE_CARDINALITIES,
    FEATURE_NAMES,
    BayesNetClassifier,
    _discretize,
)

N_FEATURES = len(FEATURE_NAMES)
N_CLASSES = 10
TL, TC, TR, ML, MC, MR, BL, BR, EUL, VR, ASP, VDOM = range(12)

# ----------------------------------------------------------------------
# feature caching (identical feature pipeline to the incumbent)
# ----------------------------------------------------------------------
def _continuous_features(seed: int):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"continuous_seed{seed}.npz"
    if path.exists():
        blob = np.load(path)
        return {k: blob[k] for k in blob.files}
    Xtr, ytr, Xva, yva = load_split(seed=seed)
    extractor = BayesNetClassifier()  # used only for its feature pipeline
    ftr = extractor._continuous_features(Xtr)
    fva = extractor._continuous_features(Xva)
    payload = {"ytr": ytr, "yva": yva}
    for key in ftr:
        payload[f"tr_{key}"] = ftr[key]
        payload[f"va_{key}"] = fva[key]
    np.savez_compressed(path, **payload)
    return payload


def discretize_pair(blob):
    """Discretize train/val with thresholds fitted on the train part only."""
    ntr = blob["tr_densities"].shape[0]
    nva = blob["va_densities"].shape[0]
    F_train = np.empty((ntr, N_FEATURES), dtype=np.int64)
    F_val = np.empty((nva, N_FEATURES), dtype=np.int64)
    for j in range(8):
        thresholds = _discretize(blob["tr_densities"][:, j], 5)[1]
        F_train[:, j] = _discretize(blob["tr_densities"][:, j], 5, thresholds)[0]
        F_val[:, j] = _discretize(blob["va_densities"][:, j], 5, thresholds)[0]
    F_train[:, EUL] = blob["tr_euler"]
    F_val[:, EUL] = blob["va_euler"]
    for col, key, card in ((VR, "vertical_runs", 5), (ASP, "aspect", 4),
                           (VDOM, "vertical_dominance", 4)):
        thresholds = _discretize(blob[f"tr_{key}"], card)[1]
        F_train[:, col] = _discretize(blob[f"tr_{key}"], card, thresholds)[0]
        F_val[:, col] = _discretize(blob[f"va_{key}"], card, thresholds)[0]
    return F_train, blob["ytr"], F_val, blob["yva"]


# sanity: our discretization must equal the incumbent's own fitted pipeline
def verify_features(F_train, F_val, X_train, X_val):
    """Assert the cached discretization equals the incumbent's own pipeline
    (thresholds fitted on train, applied to both splits)."""
    extractor = BayesNetClassifier().fit(X_train, np.zeros(X_train.shape[0], dtype=np.int64))
    ref_train = extractor._discretize_features(extractor._continuous_features(X_train))
    ref_val = extractor._discretize_features(extractor._continuous_features(X_val))
    assert np.array_equal(F_train, ref_train), "train feature pipeline diverged"
    assert np.array_equal(F_val, ref_val), "val feature pipeline diverged"
    return True


# ----------------------------------------------------------------------
# CPT engine: arbitrary DAG + pluggable smoothing / backoff scheme
# ----------------------------------------------------------------------
def joint_counts(F, y, node, parent):
    card = FEATURE_CARDINALITIES[node]
    if parent < 0:
        flat = y * card + F[:, node]
        return np.bincount(flat, minlength=N_CLASSES * card).reshape(
            N_CLASSES, card).astype(np.float64)
    pcard = FEATURE_CARDINALITIES[parent]
    flat = (y * pcard + F[:, parent]) * card + F[:, node]
    return np.bincount(flat, minlength=N_CLASSES * pcard * card).reshape(
        N_CLASSES, pcard, card).astype(np.float64)


def cpt_table(counts, class_count, alpha, scheme, *, beta=1.0, alpha2=1.0):
    """Raw counts -> conditional probability table (last axis = feature states).

    Schemes for child CPTs that have a feature parent:
      uniform          : P(F|y,Pa) = (n + a) / (n_. + a*K)          [no backoff]
      grand            : Dirichlet prior = grand-mean state profile (ignores y)
      parent_marginal  : Dirichlet prior = P(F|Pa) marginal          (ignores y)
      marginal         : Dirichlet prior = P(F|y) marginal           [incumbent]
      hier             : like marginal, but the y-marginal is itself
                         smoothed toward the grand mean with strength alpha2
      jm               : Jelinek-Mercer interpolation
                         lambda*MLE + (1-lambda)*P(F|y),  lambda = beta
    """
    card = counts.shape[-1]
    n_cp = counts.sum(axis=-1, keepdims=True)
    if counts.ndim == 2 or scheme == "uniform":
        return (counts + alpha) / (n_cp + alpha * card)
    if scheme == "grand":
        grand = counts.sum(axis=(0, 1))                    # (card,) ignores y and Pa
        target = (grand + alpha) / (grand.sum() + alpha * card)
        return (counts + alpha * target) / (n_cp + alpha)
    if scheme == "parent_marginal":
        pooled = counts.sum(axis=0)                        # (pcard, card) ignores y
        target = (pooled + alpha) / (pooled.sum(axis=-1, keepdims=True) + alpha * card)
        return (counts + alpha * target[None, :, :]) / (n_cp + alpha)
    if scheme == "marginal":
        marginal = counts.sum(axis=1)
        target = (marginal + alpha) / (class_count[:, None] + alpha * card)
        return (counts + alpha * target[:, None, :]) / (n_cp + alpha)
    if scheme == "hier":
        marginal = counts.sum(axis=1)
        grand = marginal.sum(axis=0)
        grand = (grand + alpha) / (grand.sum() + alpha * card)
        smoothed = (marginal + alpha2 * grand[None, :]) / (class_count[:, None] + alpha2)
        return (counts + alpha * smoothed[:, None, :]) / (n_cp + alpha)
    if scheme == "jm":
        marginal = counts.sum(axis=1)
        target = (marginal + 1.0) / (class_count[:, None] + card * 1.0)
        mle = counts / np.maximum(n_cp, 1e-12)
        return beta * mle + (1.0 - beta) * target[:, None, :]
    raise ValueError(f"unknown scheme {scheme}")


class DagModel:
    """A fixed 12-feature DAG with a pluggable CPT estimator."""

    def __init__(self, parents, *, scheme="marginal", alpha=1.0,
                 node_alphas=None, beta=1.0, alpha2=1.0,
                 class_prior="empirical", prior_alpha=None,
                 node_powers=None, tempering=None):
        self.parents = tuple(parents)
        self.scheme = scheme
        self.alpha = float(alpha)
        self.node_alphas = node_alphas           # per-node smoothing strength
        self.beta = float(beta)
        self.alpha2 = float(alpha2)
        self.class_prior = class_prior
        self.prior_alpha = float(alpha) if prior_alpha is None else float(prior_alpha)
        self.node_powers = node_powers           # renormalized CPT powers
        self.tempering = tempering               # direct log-likelihood powers

    def _alpha(self, node):
        return self.alpha if self.node_alphas is None else float(self.node_alphas[node])

    def fit_features(self, F, y):
        y = np.asarray(y, dtype=np.int64)
        class_count = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        if self.class_prior == "uniform":
            self.class_log_prior_ = np.full(N_CLASSES, -np.log(N_CLASSES))
        else:
            a = self.prior_alpha
            self.class_log_prior_ = np.log((class_count + a) / (y.size + N_CLASSES * a))
        logs, probs = [], []
        for node, parent in enumerate(self.parents):
            counts = joint_counts(F, y, node, parent)
            table = cpt_table(counts, class_count, self._alpha(node), self.scheme,
                              beta=self.beta, alpha2=self.alpha2)
            if self.node_powers is not None and parent >= 0:
                w = float(self.node_powers[node])
                if w != 1.0:
                    powered = np.power(np.maximum(table, 1e-300), w)
                    table = powered / powered.sum(axis=-1, keepdims=True)
            probs.append(table)
            logs.append(np.log(np.maximum(table, 1e-300)))
        self.cpt_prob_ = tuple(probs)
        self.cpt_log_prob_ = tuple(logs)
        return self

    def joint_log_likelihood(self, F):
        scores = np.broadcast_to(self.class_log_prior_, (F.shape[0], N_CLASSES)).copy()
        for node, parent in enumerate(self.parents):
            log_cpt = self.cpt_log_prob_[node]
            w = 1.0 if self.tempering is None else float(self.tempering[node])
            if parent < 0:
                scores += w * log_cpt[:, F[:, node]].T
            else:
                scores += w * log_cpt[:, F[:, parent], F[:, node]].T
        return scores

    def predict(self, F):
        return np.argmax(self.joint_log_likelihood(F), axis=1)

    def predict_proba(self, F):
        log_scores = self.joint_log_likelihood(F)
        shifted = log_scores - log_scores.max(axis=1, keepdims=True)
        p = np.exp(shifted)
        return p / p.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------
# empirical Bayes: per-node smoothing strength on an internal split
# ----------------------------------------------------------------------
ALPHA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0,
              1000.0, 3000.0, 10000.0)
INTERNAL_SEED = 20240101


def internal_split(y, fraction=0.25, seed=INTERNAL_SEED):
    """Stratified train-internal fit/hold-out split (never touches validation)."""
    rng = np.random.default_rng(seed)
    fit_idx, hold_idx = [], []
    for c in range(N_CLASSES):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = max(1, round(idx.size * fraction))
        hold_idx.append(idx[:k])
        fit_idx.append(idx[k:])
    return np.concatenate(fit_idx), np.concatenate(hold_idx)


def empirical_bayes_alphas(F, y, parents, *, scheme="marginal", grid=ALPHA_GRID,
                           fraction=0.25, pooling=False):
    """Per-node alpha by type-II maximum likelihood on an internal split.

    The DAG joint log-likelihood factorizes per node given the observed
    (y, parent) columns, so each node's smoothing strength can be selected
    independently.  With ``pooling=True`` a single shared alpha is selected by
    the sum of all nodes' held-out log-likelihoods (global empirical Bayes).
    """
    fit_idx, hold_idx = internal_split(y, fraction)
    Ff, yf = F[fit_idx], y[fit_idx]
    Fh, yh = F[hold_idx], y[hold_idx]
    class_count = np.bincount(yf, minlength=N_CLASSES).astype(np.float64)
    counts = [joint_counts(Ff, yf, n, p) for n, p in enumerate(parents)]
    held = [joint_counts(Fh, yh, n, p) for n, p in enumerate(parents)]

    def node_score(node, alpha):
        table = cpt_table(counts[node], class_count, alpha, scheme)
        return float(np.sum(held[node] * np.log(np.maximum(table, 1e-300))))

    if pooling:
        best_alpha, best = grid[0], -np.inf
        for alpha in grid:
            score = sum(node_score(n, alpha) for n in range(len(parents)))
            if score > best:
                best, best_alpha = score, alpha
        return tuple(best_alpha for _ in parents)

    chosen = []
    for node in range(len(parents)):
        best_alpha, best = grid[0], -np.inf
        for alpha in grid:
            score = node_score(node, alpha)
            if score > best:
                best, best_alpha = score, alpha
        chosen.append(best_alpha)
    return tuple(chosen)


# ----------------------------------------------------------------------
# per-feature power calibration on an internal split
# ----------------------------------------------------------------------
POWER_GRID = (0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)


def fit_powers(F, y, parents, *, scheme="marginal", alpha=1.0, node_alphas=None,
               grid=POWER_GRID, fraction=0.25, objective="logloss",
               renormalized=True, passes=3):
    """Coordinate-wise fit of per-feature powers on an internal split.

    renormalized=True : P(F|Pa)^w renormalized over F states.  The table is a
                        valid conditional distribution, so the model keeps the
                        exact log P(y) + sum log P(F|Pa) form (calibrated CPT).
    renormalized=False: w * log P(F|Pa) directly (declared extension: the
                        factors are no longer a normalized CPT, so the quantity
                        is a tempered likelihood, not a joint probability).
    """
    fit_idx, hold_idx = internal_split(y, fraction)
    Ff, yf = F[fit_idx], y[fit_idx]
    Fh, yh = F[hold_idx], y[hold_idx]
    class_count = np.bincount(yf, minlength=N_CLASSES).astype(np.float64)
    if node_alphas is None:
        node_alphas = tuple(alpha for _ in parents)
    tables = [
        cpt_table(joint_counts(Ff, yf, n, p), class_count, node_alphas[n], scheme)
        for n, p in enumerate(parents)
    ]
    log_prior = np.log((class_count + 1.0) / (yf.size + N_CLASSES))

    def node_logfactor(node, w):
        table = tables[node]
        parent = parents[node]
        if renormalized and w != 1.0:
            powered = np.power(np.maximum(table, 1e-300), w)
            powered = powered / powered.sum(axis=-1, keepdims=True)
            table = powered
            w = 1.0
        if parent < 0:
            base = table[:, Fh[:, node]].T
        else:
            base = table[:, Fh[:, parent], Fh[:, node]].T
        return w * np.log(np.maximum(base, 1e-300))

    weights = [1.0] * len(parents)
    for _ in range(passes):
        changed = False
        for node in range(len(parents)):
            others = np.broadcast_to(log_prior, (Fh.shape[0], N_CLASSES)).copy()
            for other in range(len(parents)):
                if other != node:
                    others += node_logfactor(other, weights[other])
            best_w, best = weights[node], -np.inf
            for w in grid:
                scores = others + node_logfactor(node, w)
                if objective == "logloss":
                    shifted = scores - scores.max(axis=1, keepdims=True)
                    value = float(np.mean(
                        shifted[np.arange(yh.size), yh]
                        - np.log(np.exp(shifted).sum(axis=1))))
                else:
                    value = float(np.mean(np.argmax(scores, axis=1) == yh))
                if value > best:
                    best, best_w = value, w
            if best_w != weights[node]:
                weights[node] = best_w
                changed = True
        if not changed:
            break
    return tuple(weights)


# ----------------------------------------------------------------------
# DAG candidates (hand-designed, semantically motivated)
# ----------------------------------------------------------------------
INCUMBENT_SEM = {EUL: -1, VR: EUL, ASP: VR, VDOM: VR}

# Every grid candidate is a forest of left->right / scan-order chains, so each
# feature node keeps at most one feature parent (plus Y) as the rules require.
GRID_VARIANTS = {
    # incumbent: three separate runs, right column closed by br->mr, bl<-br
    "G_incumbent": {TC: TL, TR: TC, MC: ML, MR: MC, BL: BR, BR: MR},
    # reading order (raster) over the 8 kept cells
    "G_raster": {TC: TL, TR: TC, ML: TR, MC: ML, MR: MC, BL: MR, BR: BL},
    # boustrophedon: top L->R, middle R->L, bottom L->R (bottom-right closes)
    "G_boustro": {TC: TL, TR: TC, MR: TR, MC: MR, ML: MC, BL: ML, BR: BL},
    # boustrophedon, bottom row R->L
    "G_boustro_rev": {TC: TL, TR: TC, MR: TR, MC: MR, ML: MC, BR: ML, BL: BR},
    # single perimeter/serpentine chain around the kept cells
    "G_serp": {TC: TL, TR: TC, MR: TR, BR: MR, BL: BR, ML: BL, MC: ML},
    # left column closed downward from the top (tl->ml) + incumbent rows
    "G_ml_tl": {TC: TL, TR: TC, ML: TL, MC: ML, MR: MC, BL: BR, BR: MR},
    # left column closed upward from the bottom (bl->ml)
    "G_ml_bl": {TC: TL, TR: TC, ML: BL, MC: ML, MR: MC, BL: BR, BR: MR},
    # middle row fed by the bottom-right corner (cross-row link br->ml)
    "G_ml_br": {TC: TL, TR: TC, ML: BR, MC: ML, MR: MC, BL: BR, BR: MR},
    # diagonal leak: top-left also explains the middle-centre cell
    "G_diag_tl_mc": {TC: TL, TR: TC, MC: TL, MR: MC, ML: BR, BL: BR, BR: MR},
    # central hub: middle-centre drives both right cells
    "G_hub_mc": {TC: TL, TR: TC, MC: ML, MR: MC, BR: MC, BL: BR, ML: BL},
}

SEMANTIC_VARIANTS = {
    # incumbent chain: euler -> vruns -> {aspect, vdom}
    "S_inc": {EUL: -1, VR: EUL, ASP: VR, VDOM: VR},
    # topology -> orientation -> runs -> shape
    "S_evra": {EUL: -1, VDOM: EUL, VR: VDOM, ASP: VR},
    # topology -> shape -> runs -> orientation
    "S_eavr": {EUL: -1, ASP: EUL, VR: ASP, VDOM: VR},
    # euler -> vruns -> aspect -> vdom (fully serial)
    "S_serial": {EUL: -1, VR: EUL, ASP: VR, VDOM: ASP},
    # vruns as the hub with euler as its child
    "S_vr_hub": {VR: -1, EUL: VR, ASP: VR, VDOM: VR},
    # orientation drives runs
    "S_vr_vdom": {EUL: -1, VR: VDOM, ASP: VR, VDOM: -1},
    # v2 style: two independent semantic edges
    "S_v2": {VR: EUL, ASP: VDOM, VDOM: -1, EUL: -1},
}


def build(grid_map, semantic_map):
    parents = [-1] * N_FEATURES
    for node, parent in {**grid_map, **semantic_map}.items():
        parents[node] = parent
    return tuple(parents)


def describe(parents):
    return "  ".join(
        f"{FEATURE_NAMES[p]}->{FEATURE_NAMES[c]}"
        for c, p in enumerate(parents) if p >= 0
    )


def score(parents, Ftr, ytr, Fva, yva, **kwargs):
    model = DagModel(parents, **kwargs).fit_features(Ftr, ytr)
    return accuracy(yva, model.predict(Fva)), model


# ----------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------
def stage_grid(Ftr, ytr, Fva, yva):
    print("\n== stage 1: grid DAG variants (scheme=marginal, alpha=1) ==")
    rows = []
    for name, grid in GRID_VARIANTS.items():
        parents = build(grid, INCUMBENT_SEM)
        acc, _ = score(parents, Ftr, ytr, Fva, yva)
        rows.append({"variant": name, "accuracy": acc, "parents": parents,
                     "edges": describe(parents)})
        print(f"  {name:<15} acc={acc:.4f}  ({sum(p >= 0 for p in parents)} edges)")
    return sorted(rows, key=lambda r: -r["accuracy"])


def stage_semantic(Ftr, ytr, Fva, yva, grid_map):
    print("\n== stage 2: semantic wirings (best grid map) ==")
    rows = []
    for name, sem in SEMANTIC_VARIANTS.items():
        parents = build(grid_map, sem)
        acc, _ = score(parents, Ftr, ytr, Fva, yva)
        rows.append({"variant": name, "accuracy": acc, "parents": parents,
                     "edges": describe(parents)})
        print(f"  {name:<12} acc={acc:.4f}  {describe(parents)}")
    return sorted(rows, key=lambda r: -r["accuracy"])


def stage_cross(Ftr, ytr, Fva, yva):
    """Cross the strongest grid maps with every semantic wiring."""
    print("\n== stage 2b: grid x semantic cross ==")
    rows = []
    for gname, grid in GRID_VARIANTS.items():
        for sname, sem in SEMANTIC_VARIANTS.items():
            parents = build(grid, sem)
            acc, _ = score(parents, Ftr, ytr, Fva, yva)
            rows.append({"variant": f"{gname}|{sname}", "accuracy": acc,
                         "parents": parents, "edges": describe(parents)})
    rows.sort(key=lambda r: -r["accuracy"])
    for row in rows[:12]:
        print(f"  {row['variant']:<24} acc={row['accuracy']:.4f}")
    print(f"  ... ({len(rows)} combos total)")
    return rows


def stage_cross_seed_finalists():
    """Score DAG/smoothing finalists on seeds 1322 / 7 / 2024 with the cached
    feature pipeline, so a 'gain' that is split luck is exposed early."""
    print("\n== stage 4: cross-seed finalist screen ==")
    finalists = {
        "incumbent": (build(GRID_VARIANTS["G_incumbent"], INCUMBENT_SEM),
                      dict(scheme="marginal", alpha=1.0)),
        "G_serp|S_evra marginal": (build(GRID_VARIANTS["G_serp"], SEMANTIC_VARIANTS["S_evra"]),
                                   dict(scheme="marginal", alpha=1.0)),
        "G_serp|S_evra parent_marginal": (build(GRID_VARIANTS["G_serp"], SEMANTIC_VARIANTS["S_evra"]),
                                          dict(scheme="parent_marginal", alpha=1.0)),
        "G_serp|S_eavr marginal": (build(GRID_VARIANTS["G_serp"], SEMANTIC_VARIANTS["S_eavr"]),
                                   dict(scheme="marginal", alpha=1.0)),
        "G_serp|S_vr_vdom marginal": (build(GRID_VARIANTS["G_serp"], SEMANTIC_VARIANTS["S_vr_vdom"]),
                                      dict(scheme="marginal", alpha=1.0)),
        "G_incumbent|S_evra marginal": (build(GRID_VARIANTS["G_incumbent"], SEMANTIC_VARIANTS["S_evra"]),
                                        dict(scheme="marginal", alpha=1.0)),
    }
    seeds = (1322,) + tuple(EXTRA_SEEDS)
    caches = {}
    for seed in seeds:
        caches[seed] = discretize_pair(_continuous_features(seed))
    results = {}
    for name, (parents, kwargs) in finalists.items():
        accs = {}
        for seed in seeds:
            Ftr, ytr, Fva, yva = caches[seed]
            acc, _ = score(parents, Ftr, ytr, Fva, yva, **kwargs)
            accs[seed] = acc
        mean = float(np.mean(list(accs.values())))
        results[name] = {"accuracies": accs, "mean": mean}
        print(f"  {name:<32} " + "  ".join(f"s{s}={a:.4f}" for s, a in accs.items())
              + f"   mean={mean:.4f}")
    return results


def stage_refine(Ftr, ytr, Fva, yva):
    """Refine around the winning contour serpentine: direction and entry point."""
    print("\n== stage 2c: serpentine refinements ==")
    extra_grid = {
        # clockwise starting top-left, entering interior at top (winner)
        "R_serp_tl_cw": {TC: TL, TR: TC, MR: TR, BR: MR, BL: BR, ML: BL, MC: ML},
        # counter-clockwise from top-left: down the left, across the bottom,
        # up the right, then inward
        "R_serp_tl_ccw": {ML: TL, BL: ML, BR: BL, MR: BR, TR: MR, TC: TR, MC: TC},
        # clockwise from top-right
        "R_serp_tr_cw": {TC: TR, TL: TC, ML: TL, BL: ML, BR: BL, MR: BR, MC: MR},
        # serpentine that enters the interior from the bottom-right cell
        "R_serp_mr_in": {TC: TL, TR: TC, BR: TR, BL: BR, ML: BL, MC: ML, MR: MC},
        # interior spine: centre fed by middle-right (one edge shorter loop)
        "R_serp_short": {TC: TL, TR: TC, MR: TR, BR: MR, BL: BR, ML: BL, MC: ML},
    }
    sem_combo = [(g, s) for g in extra_grid for s in ("S_evra", "S_eavr", "S_inc")]
    rows = []
    for gname, grid in extra_grid.items():
        for sname in ("S_evra", "S_eavr", "S_inc"):
            parents = build(grid, SEMANTIC_VARIANTS[sname])
            acc, _ = score(parents, Ftr, ytr, Fva, yva)
            rows.append({"variant": f"{gname}|{sname}", "accuracy": acc,
                         "parents": parents, "edges": describe(parents)})
    rows.sort(key=lambda r: -r["accuracy"])
    for row in rows:
        print(f"  {row['variant']:<22} acc={row['accuracy']:.4f}")
    return rows


def stage_smoothing(Ftr, ytr, Fva, yva, parents):
    print(f"\n== stage 3: smoothing / calibration ==")
    print(f"  DAG: {describe(parents)}")
    rows = []

    def add(name, acc, **extra):
        rows.append({"variant": name, "accuracy": acc, **extra})
        print(f"  {name:<38} acc={acc:.4f}")

    acc, _ = score(parents, Ftr, ytr, Fva, yva, alpha=1.0, scheme="marginal")
    add("A_marginal alpha=1 (incumbent scheme)", acc)

    for scheme in ("uniform", "grand", "parent_marginal"):
        for alpha in (1.0, 30.0, 300.0, 3000.0):
            acc, _ = score(parents, Ftr, ytr, Fva, yva, alpha=alpha, scheme=scheme)
            add(f"B_{scheme} alpha={alpha:g}", acc)

    for beta in (0.4, 0.6, 0.8, 0.9, 0.97):
        acc, _ = score(parents, Ftr, ytr, Fva, yva, scheme="jm", beta=beta)
        add(f"C_jm lambda={beta:g}", acc)

    acc, _ = score(parents, Ftr, ytr, Fva, yva, class_prior="uniform")
    add("D_uniform class prior", acc)
    for pa in (0.01, 0.1, 10.0):
        acc, _ = score(parents, Ftr, ytr, Fva, yva, prior_alpha=pa)
        add(f"D_prior alpha={pa:g}", acc)

    per_node = empirical_bayes_alphas(Ftr, ytr, parents)
    acc, _ = score(parents, Ftr, ytr, Fva, yva, node_alphas=per_node)
    add("E_EB per-node alpha", acc, node_alphas=list(per_node))
    print(f"      alphas: {per_node}")
    pooled = empirical_bayes_alphas(Ftr, ytr, parents, pooling=True)
    acc, _ = score(parents, Ftr, ytr, Fva, yva, node_alphas=pooled)
    add("E_EB pooled alpha", acc, node_alphas=list(pooled))

    for w in (0.15, 0.3, 0.5):
        powers = fit_powers(Ftr, ytr, parents, renormalized=True,
                            objective="logloss")
        powers = mix(powers, w)
        acc, _ = score(parents, Ftr, ytr, Fva, yva, node_powers=powers)
        add(f"F_pow renorm logloss mix={w:g}", acc, powers=list(powers))

    powers = fit_powers(Ftr, ytr, parents, renormalized=True, objective="logloss")
    acc, _ = score(parents, Ftr, ytr, Fva, yva, node_powers=powers)
    add("F_pow renorm logloss full", acc, powers=list(powers))
    print(f"      powers: {powers}")

    powers = fit_powers(Ftr, ytr, parents, renormalized=True, objective="accuracy")
    acc, _ = score(parents, Ftr, ytr, Fva, yva, node_powers=powers)
    add("F_pow renorm accuracy full", acc, powers=list(powers))
    print(f"      powers: {powers}")

    powers = fit_powers(Ftr, ytr, parents, renormalized=False, objective="logloss")
    acc, _ = score(parents, Ftr, ytr, Fva, yva, tempering=powers)
    add("G_temper raw logloss [EXT]", acc, powers=list(powers))
    print(f"      tempering: {powers}")
    return rows


def mix(powers, fraction):
    """Shrink fitted powers toward 1.0 (defensive half-step)."""
    return tuple(1.0 + fraction * (w - 1.0) for w in powers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="grid,semantic,smoothing")
    parser.add_argument("--seed", type=int, default=1322)
    args = parser.parse_args()

    blob = _continuous_features(args.seed)
    Ftr, ytr, Fva, yva = discretize_pair(blob)
    print(f"features: train={Ftr.shape} val={Fva.shape}")

    if "verify" in args.stages or True:
        Xtr, _, Xva, _ = load_split(seed=args.seed)
        print("feature-pipeline identity vs incumbent:",
              verify_features(Ftr, Fva, Xtr, Xva))

    out_dir = RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    summary = {}

    grid_rows = None
    if "grid" in stages:
        grid_rows = stage_grid(Ftr, ytr, Fva, yva)
        summary["grid"] = grid_rows
    if "semantic" in stages:
        best_grid_name = grid_rows[0]["variant"] if grid_rows else "G_serp"
        best_grid_map = GRID_VARIANTS[best_grid_name]
        print(f"\n(best grid from stage 1: {best_grid_name})")
        sem_rows = stage_semantic(Ftr, ytr, Fva, yva, best_grid_map)
        summary["semantic"] = sem_rows
    if "cross" in stages:
        summary["cross"] = stage_cross(Ftr, ytr, Fva, yva)
    if "refine" in stages:
        summary["refine"] = stage_refine(Ftr, ytr, Fva, yva)
    if "smoothing" in stages:
        if "cross" in summary:
            parents = summary["cross"][0]["parents"]
        elif "semantic" in summary:
            parents = summary["semantic"][0]["parents"]
        else:
            parents = build(GRID_VARIANTS["G_incumbent"], INCUMBENT_SEM)
        par = tuple(parents) if isinstance(parents, (list, tuple)) else parents
        summary["smoothing"] = stage_smoothing(Ftr, ytr, Fva, yva, par)
    if "regress" in stages:
        summary["regression"] = stage_regression()
    if "crossseed" in stages:
        summary["cross_seed"] = stage_cross_seed_finalists()

    path = out_dir / f"builder_d_stage_raw_seed{args.seed}.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {path}")


def stage_regression():
    """Adversarial check: does the cached-feature DagModel reproduce the
    incumbent's own predictions exactly?"""
    import numpy as np
    Xtr, ytr, Xva, yva = load_split()
    incumbent = BayesNetClassifier(alpha=1.0).fit(Xtr, ytr)
    ref = incumbent.predict(Xva)
    blob = _continuous_features(1322)
    Ftr, ytr2, Fva, yva2 = discretize_pair(blob)
    mine, _ = score(INCUMBENT_PARENTS_ALIAS(), Ftr, ytr2, Fva, yva2, alpha=1.0)
    print("\n== stage 0: regression vs incumbent ==")
    print(f"  incumbent acc = {accuracy(yva, ref):.4f}")
    print(f"  DagModel  acc = {mine:.4f}")
    return {"incumbent": float(accuracy(yva, ref)), "dag_model": float(mine)}


def INCUMBENT_PARENTS_ALIAS():
    from bayes_net import FEATURE_PARENTS
    return FEATURE_PARENTS


if __name__ == "__main__":
    main()
