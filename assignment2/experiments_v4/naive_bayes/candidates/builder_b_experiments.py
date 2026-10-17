"""Builder B experiments: parameter estimation / smoothing / priors for the
729-patch Naive Bayes classifier (BME1322.01 Assignment 2, experiments_v4).

Everything here uses ONLY the 60,000 training images via ``v4_common.load_split``.
The 10,000-image test set is never opened.

Run everything:

    python builder_b_experiments.py --only all

Individual sections (results are journalled into builder_b_results.json and
re-used on the next run, so re-running is cheap):

    baseline alpha_curve zero_stats backoff mml temper prior falsify

The reported numbers land in ../results/builder_b.md (written by hand from the
journal) and in ./builder_b_results.json (machine readable).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parent.parent
ROOT = V4.parent
CACHE = HERE / "cache"
JOURNAL = HERE / "builder_b_results.json"
CACHE.mkdir(parents=True, exist_ok=True)
for _p in (ROOT, V4):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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

# ----------------------------------------------------------------------------
# Model geometry -- identical to the incumbent naive_bayes.py
# ----------------------------------------------------------------------------
N_CLASSES = 10
N_LEVELS = 5
N_STATES = N_LEVELS ** 4          # 625 patch states
N_PATCH = 27 * 27                 # 729 overlapping 2x2 patches
PIXEL_EDGES = np.array([51, 102, 153, 204], dtype=np.int64)
N_CLASSES_GRID = tuple(range(N_CLASSES))


# ----------------------------------------------------------------------------
# Vectorised log-gamma (Lanczos), pure NumPy -- no scipy, no stdlib loops.
# Needed for the Dirichlet-multinomial marginal likelihood (MML) criterion.
# ----------------------------------------------------------------------------
_LANCZOS_G = 7.0
_LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-06,
    1.5056327351493116e-07,
])


def lgamma(x):
    """Vectorised log Gamma for x > 0 (Lanczos g=7, n=9)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    small = x < 0.5
    if np.any(small):
        xs = x[small]
        # reflection: Gamma(z)Gamma(1-z) = pi / sin(pi z)
        out[small] = np.log(np.pi / np.sin(np.pi * xs)) - lgamma(1.0 - xs)
    big = ~small
    if np.any(big):
        xb = x[big] - 1.0
        acc = np.full(xb.shape, _LANCZOS_C[0])
        for i in range(1, _LANCZOS_C.size):
            acc = acc + _LANCZOS_C[i] / (xb + i)
        t = xb + _LANCZOS_G + 0.5
        out[big] = (0.5 * np.log(2.0 * np.pi) + (xb + 0.5) * np.log(t) - t
                    + np.log(acc))
    return out


def _selftest_lgamma() -> float:
    probe = np.array([1e-8, 1e-4, 0.1, 0.5, 1.0, 2.5, 7.0, 123.7, 5000.3])
    worst = max(abs(float(lgamma(np.array(v))) - math.lgamma(v)) for v in probe)
    return worst


# ----------------------------------------------------------------------------
# Feature extraction: 729 overlapping 2x2 patches -> base-5 state in [0, 625)
# Bit-for-bit the encoding used by the incumbent (row-major patch order).
# ----------------------------------------------------------------------------
def patch_states(X: np.ndarray) -> np.ndarray:
    """(n, 784) uint8 pixels -> (n, 729) int16 patch states."""
    n = X.shape[0]
    img = np.asarray(X).reshape(n, 28, 28)
    bands = np.searchsorted(PIXEL_EDGES, img, side="right").astype(np.int16)
    states = (
        bands[:, :-1, :-1] * (N_LEVELS ** 3)
        + bands[:, :-1, 1:] * (N_LEVELS ** 2)
        + bands[:, 1:, :-1] * N_LEVELS
        + bands[:, 1:, 1:]
    ).reshape(n, N_PATCH)
    return states.astype(np.int16)


def pixel_bands(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    img = np.asarray(X).reshape(n, 28, 28)
    return np.searchsorted(PIXEL_EDGES, img, side="right").astype(np.int16)


# ----------------------------------------------------------------------------
# Split / state cache (so alpha sweeps never re-encode 50k images)
# ----------------------------------------------------------------------------
def cached_split(seed: int = 1322):
    """(states_tr, y_tr, states_va, y_va) with the patch states cached on disk."""
    path = CACHE / f"patch_states_seed{seed}.npz"
    if path.exists():
        blob = np.load(path)
        return blob["str"], blob["ytr"], blob["sva"], blob["yva"]
    Xtr, ytr, Xva, yva = load_split(seed=seed)
    t = time.perf_counter()
    str_ = patch_states(Xtr)
    sva = patch_states(Xva)
    np.savez_compressed(path, str=str_, ytr=ytr, sva=sva, yva=yva)
    print(f"  [cache] seed={seed} states built in {time.perf_counter()-t:.1f}s -> {path.name}")
    return str_, ytr, sva, yva


def raw_split(seed: int = 1322):
    Xtr, ytr, Xva, yva = load_split(seed=seed)
    return Xtr, ytr, Xva, yva


# ----------------------------------------------------------------------------
# Count statistics
# ----------------------------------------------------------------------------
def train_counts(states_tr: np.ndarray, y_tr: np.ndarray):
    """(counts (10,729,625), class_count (10,)) from the training part only."""
    counts = np.zeros((N_CLASSES, N_PATCH, N_STATES), dtype=np.int32)
    class_count = np.bincount(y_tr, minlength=N_CLASSES).astype(np.int64)
    offsets = (np.arange(N_PATCH, dtype=np.int64) * N_STATES)[None, :]
    for c in range(N_CLASSES):
        sub = states_tr[y_tr == c].astype(np.int64) + offsets
        counts[c] = np.bincount(sub.ravel(),
                                minlength=N_PATCH * N_STATES).reshape(N_PATCH, N_STATES)
    return counts, class_count


# ----------------------------------------------------------------------------
# Prior-mean ("parent") distributions used by the Dirichlet priors
# ----------------------------------------------------------------------------
def parent_uniform(counts, class_count) -> np.ndarray:
    return np.full((N_PATCH, N_STATES), 1.0 / N_STATES)


def parent_pixel_product_from_bands(bands_tr, y_tr, alpha_pix: float = 1.0):
    """Within-patch-independence parent: product of the 4 pixel-band marginals.

    This is exactly what a raw-pixel NB (784 single-pixel features) assigns to
    the patch state, i.e. a coarser, much denser estimate that the joint 2x2 CPT
    can be shrunk towards.  Shape (10, 729, 625).
    """
    n = bands_tr.shape[0]
    class_count = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    pix = np.zeros((N_CLASSES, 28 * 28, N_LEVELS), dtype=np.float64)
    offsets = (np.arange(28 * 28, dtype=np.int64) * N_LEVELS)[None, :]
    flat = bands_tr.reshape(n, 28 * 28).astype(np.int64)
    for c in range(N_CLASSES):
        sub = flat[y_tr == c] + offsets
        pix[c] = np.bincount(sub.ravel(),
                             minlength=28 * 28 * N_LEVELS).reshape(28 * 28, N_LEVELS)
    pix += alpha_pix
    pix /= (class_count[:, None, None] + N_LEVELS * alpha_pix)
    pix = pix.reshape(N_CLASSES, 28, 28, N_LEVELS)

    s = np.arange(N_STATES, dtype=np.int64)
    b00, b01, b10, b11 = s // 125, (s // 25) % 5, (s // 5) % 5, s % 5
    rows = np.arange(N_PATCH) // 27
    cols = np.arange(N_PATCH) % 27
    par = np.empty((N_CLASSES, N_PATCH, N_STATES), dtype=np.float64)
    for c in range(N_CLASSES):
        p = pix[c]
        a = p[rows, cols][:, b00]
        b = p[rows, cols + 1][:, b01]
        d = p[rows + 1, cols][:, b10]
        e = p[rows + 1, cols + 1][:, b11]
        par[c] = a * b * d * e
    return par


def parent_global_frequency(counts, alpha_g: float = 1.0) -> np.ndarray:
    """Class-agnostic per-patch state frequency, shape (729, 625)."""
    total = counts.sum(axis=0).astype(np.float64)          # (729, 625)
    n = total.sum(axis=1, keepdims=True)                    # (729, 1)
    return (total + alpha_g) / (n + N_STATES * alpha_g)


def parent_global_pooled(counts, alpha_g: float = 1.0, group: int = 3) -> np.ndarray:
    """Position-pooled class-agnostic frequency (row/col decimated into blocks).

    States at nearby patch positions have very similar occupation patterns; a
    coarser parent pools them into a `group x group` grid of row/col cells.
    """
    total = counts.sum(axis=0).astype(np.float64).reshape(27, 27, N_STATES)
    pooled = np.empty((group, group, N_STATES), dtype=np.float64)
    for gr in range(group):
        rs = slice(gr * 27 // group, (gr + 1) * 27 // group)
        for gc in range(group):
            cs = slice(gc * 27 // group, (gc + 1) * 27 // group)
            pooled[gr, gc] = total[rs, cs].sum(axis=(0, 1))
    pooled = (pooled + alpha_g) / (pooled.sum(axis=2, keepdims=True) + N_STATES * alpha_g)
    rows = np.arange(N_PATCH) // 27
    cols = np.arange(N_PATCH) % 27
    return pooled[rows * group // 27, cols * group // 27]


# ----------------------------------------------------------------------------
# Log-probability tables and scoring
# ----------------------------------------------------------------------------
def log_prior(class_count: np.ndarray, alpha: float) -> np.ndarray:
    total = class_count.sum()
    return np.log((class_count + alpha) / (total + N_CLASSES * alpha))


def logtab_uniform(counts: np.ndarray, class_count: np.ndarray, alpha: float) -> np.ndarray:
    denom = class_count[:, None, None] + N_STATES * alpha
    return np.log((counts.astype(np.float64) + alpha) / denom)


def logtab_backoff(counts, class_count, par, kappa) -> np.ndarray:
    """Dirichlet prior with mean `par` and total concentration `kappa`.

    ``par`` is (10,729,625) or (729,625).  ``kappa`` may be:
      * a scalar                    -- one concentration for every CPT,
      * shape (10,)                 -- per class,
      * shape (10, 729)             -- per (class, patch) CPT.
    The effective additive pseudo-count per state is ``kappa * par``.
    Reduces to Lidstone smoothing with alpha = kappa / N_STATES when
    par is uniform.  NOTE the units: `kappa` is the *total* Dirichlet
    concentration, not the per-state Lidstone alpha used elsewhere.
    """
    c = counts.astype(np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    par2 = par if par.ndim == 3 else par[None, :, :]        # (1 or 10, 729, 625)
    if kappa.ndim == 0:
        a = float(kappa) * par2
        denom = class_count[:, None, None] + float(kappa)
    elif kappa.ndim == 1:                                   # (10,)
        a = kappa[:, None, None] * par2
        denom = class_count[:, None, None] + kappa[:, None, None]
    else:                                                   # (10, 729)
        a = kappa[:, :, None] * par2
        denom = class_count[:, None, None] + kappa[:, :, None]
    return np.log((c + a) / denom.astype(np.float64))


def joint_scores(states: np.ndarray, logtab: np.ndarray, logpri: np.ndarray,
                 weight: float = 1.0) -> np.ndarray:
    """log P(y) + weight * sum_j log P(z_j | y) for every class."""
    n = states.shape[0]
    idx = states.astype(np.int64) + (np.arange(N_PATCH, dtype=np.int64) * N_STATES)[None, :]
    out = np.empty((n, N_CLASSES), dtype=np.float64)
    for c in range(N_CLASSES):
        out[:, c] = logtab[c].ravel()[idx].sum(axis=1)
    return weight * out + logpri[None, :]


def predict_from(states, logtab, logpri, weight: float = 1.0) -> np.ndarray:
    return np.argmax(joint_scores(states, logtab, logpri, weight), axis=1)


def evaluate(states_va, y_va, logtab, logpri, cutoff=None, weight: float = 1.0):
    if cutoff is not None:
        states_va, y_va = states_va[:cutoff], y_va[:cutoff]
    pred = predict_from(states_va, logtab, logpri, weight)
    return accuracy(y_va, pred), pred


# ----------------------------------------------------------------------------
# MML (Dirichlet-multinomial marginal likelihood) criterion for kappa
# ----------------------------------------------------------------------------
MML_GRID = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0,
                     10.0, 30.0, 100.0, 300.0, 1000.0])


def mml_scores(counts, class_count, par, grid=MML_GRID) -> np.ndarray:
    """log P(counts_{c,j} | kappa) for each kappa in grid -> (10, 729, G).

    Exact Dirichlet-multinomial marginal likelihood of a per-CPT symmetric-mean
    Dirichlet prior with mean `par` and total concentration kappa.  Uses only
    training counts.
    """
    c = counts.astype(np.float64)
    n_c = class_count.astype(np.float64)
    out = np.empty((N_CLASSES, N_PATCH, grid.size), dtype=np.float64)
    for gi, kappa in enumerate(grid):
        if par.ndim == 2:
            a = kappa * par[None, :, :]
            prior_const = lgamma(np.array(kappa))
        else:
            a = kappa * par
            prior_const = lgamma(np.array(kappa))
        out[..., gi] = (
            lgamma(c + a) - lgamma(a)
        ).sum(axis=2) + prior_const - lgamma(n_c[:, None] + kappa)
    return out


def aggregate_mml(counts, class_count, par, grid=MML_GRID):
    """Summed MML over all CPTs (global kappa) -> {grid value: total log lik}."""
    sc = mml_scores(counts, class_count, par, grid)
    return sc.sum(axis=(0, 1)), sc


# ----------------------------------------------------------------------------
# Journal
# ----------------------------------------------------------------------------
def load_journal() -> dict:
    if JOURNAL.exists():
        return json.loads(JOURNAL.read_text(encoding="utf-8"))
    return {}


def save_journal(data: dict) -> None:
    JOURNAL.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def note(journal, key, value):
    journal[key] = value
    save_journal(journal)
    print(f"  [journal] {key} saved")


# ============================================================================
# Sections
# ============================================================================
def section_baseline(journal, quick=False):
    """Reproduce the incumbent numbers with the fast harness."""
    str_, ytr, sva, yva = cached_split(1322)
    counts, class_count = train_counts(str_, ytr)
    out = {}
    for a in (0.05, 0.1, 0.2, 0.5, 1.0):
        t = time.perf_counter()
        acc, _ = evaluate(sva, yva, logtab_uniform(counts, class_count, a),
                          log_prior(class_count, a))
        out[str(a)] = acc
        print(f"  alpha={a:<5} val={acc:.4f} ({time.perf_counter()-t:.1f}s)")
    note(journal, "harness_baseline", out)
    return counts, class_count, sva, yva


def section_alpha_curve(journal, quick=False):
    """Full small-alpha asymptotics: where does validation genuinely peak?"""
    str_, ytr, sva, yva = cached_split(1322)
    counts, class_count = train_counts(str_, ytr)
    alphas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 5e-3, 2e-3, 1e-3,
              5e-4, 2e-4, 1e-4, 1e-5, 1e-6]
    curve, diag = {}, {}
    for a in alphas:
        tab = logtab_uniform(counts, class_count, a)
        acc, pred = evaluate(sva, yva, tab, log_prior(class_count, a))
        curve[str(a)] = acc
        # mechanism diagnostics: how often does the true class hit a zero count?
        idx = sva.astype(np.int64) + (np.arange(N_PATCH, dtype=np.int64) * N_STATES)[None, :]
        true_zero = np.zeros(sva.shape[0], dtype=np.int64)
        for c in range(N_CLASSES):
            mask = yva == c
            z = counts[c].ravel()[idx[mask]] == 0
            true_zero[mask] = z.sum(axis=1)
        diag[str(a)] = {
            "val_acc": acc,
            "mean_zero_states_true_class": float(true_zero.mean()),
            "frac_samples_true_class_has_zero_state": float(np.mean(true_zero > 0)),
            "log_alpha": float(math.log(a)),
        }
        print(f"  alpha={a:<9g} val={acc:.4f}  mean zero-count states (true cls)="
              f"{true_zero.mean():7.2f}  frac={(true_zero>0).mean():.3f}")
    note(journal, "alpha_curve", curve)
    note(journal, "alpha_curve_diagnostics", diag)
    return counts, class_count, sva, yva


def section_zero_stats(journal, quick=False):
    """How sparse are the 625-state CPTs?  Explains the alpha asymptotics."""
    str_, ytr, sva, yva = cached_split(1322)
    counts, class_count = train_counts(str_, ytr)
    zero_frac_all = float(np.mean(counts == 0))
    zero_frac_by_class = [float(np.mean(counts[c] == 0)) for c in range(N_CLASSES)]
    # per-state mean count
    mean_count = float(counts.sum() / (N_CLASSES * N_PATCH * N_STATES))
    # population of states actually used by validation samples, by class
    occ = {}
    idx = sva.astype(np.int64) + (np.arange(N_PATCH, dtype=np.int64) * N_STATES)[None, :]
    for c in range(N_CLASSES):
        sub = idx[yva == c]
        used = np.unique(sub)
        seen = counts[c].ravel()[used] > 0
        occ[c] = {
            "val_patch_states": int(used.size),
            "val_patch_states_unseen_in_train_class": int((~seen).sum()),
            "frac_unseen": float((~seen).mean()),
        }
    note(journal, "zero_stats", {
        "zero_fraction_all_cpt_cells": zero_frac_all,
        "zero_fraction_per_class": zero_frac_by_class,
        "mean_count_per_state": mean_count,
        "train_samples_per_class": class_count.tolist(),
        "val_state_occupancy": occ,
    })
    print(f"  zero cells overall: {zero_frac_all:.3%}; mean count/state {mean_count:.2f}")
    for c in range(N_CLASSES):
        print(f"    class {c}: unseen val states {occ[c]['frac_unseen']:.2%}")


def _parents(counts, class_count, str_, ytr, bands_tr):
    par_uni = parent_uniform(counts, class_count)
    par_pix = parent_pixel_product_from_bands(bands_tr, ytr, alpha_pix=1.0)
    par_freq = parent_global_frequency(counts, alpha_g=1.0)
    par_pool = parent_global_pooled(counts, alpha_g=1.0, group=3)
    return {
        "uniform": par_uni,
        "pixel_product": par_pix,
        "global_freq": par_freq,
        "pooled3x3_freq": par_pool,
    }


def section_backoff(journal, quick=False):
    """Hierarchical / backoff smoothing: shrink each CPT toward a coarser parent."""
    Xtr, ytr, Xva, yva = raw_split(1322)
    str_, _, sva, _ = cached_split(1322)
    bands_tr = pixel_bands(Xtr)
    counts, class_count = train_counts(str_, ytr)
    parents = _parents(counts, class_count, str_, ytr, bands_tr)

    kappas = [1e-3, 3e-3, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    table, curves = {}, {}
    for name, par in parents.items():
        curves[name] = {}
        for k in kappas:
            tab = logtab_backoff(counts, class_count, par, k)
            acc, _ = evaluate(sva, yva, tab, log_prior(class_count, 0.05))
            curves[name][str(k)] = acc
        best_k = max(curves[name], key=lambda kk: curves[name][kk])
        table[name] = {"best_kappa": float(best_k),
                       "best_val_acc": curves[name][best_k],
                       "curve": curves[name]}
        print(f"  parent={name:<16} best kappa={best_k:<8} val={curves[name][best_k]:.4f}")
    note(journal, "backoff", table)
    return counts, class_count, sva, yva, parents


def section_mml(journal, quick=False):
    """Estimate the smoothing strength from TRAINING data only (MML / empirical Bayes)."""
    t0 = time.perf_counter()
    Xtr, ytr, Xva, yva = raw_split(1322)
    str_, _, sva, _ = cached_split(1322)
    bands_tr = pixel_bands(Xtr)
    counts, class_count = train_counts(str_, ytr)
    par_uni = parent_uniform(counts, class_count)
    par_pix = parent_pixel_product_from_bands(bands_tr, ytr, alpha_pix=1.0)

    report = {}
    for name, par in (("uniform", par_uni), ("pixel_product", par_pix)):
        total, sc = aggregate_mml(counts, class_count, par)
        best_global = float(MML_GRID[int(np.argmax(total))])
        report[name] = {
            "mml_total_by_kappa": {str(g): float(v) for g, v in zip(MML_GRID, total)},
            "mml_argmax_kappa": best_global,
        }
        # validation accuracy at the MML-selected global kappa
        tab = logtab_backoff(counts, class_count, par, best_global)
        acc, pred = evaluate(sva, yva, tab, log_prior(class_count, 0.05))
        report[name]["val_acc_at_mml_global"] = acc
        print(f"  parent={name:<16} MML global kappa={best_global:<8g} val={acc:.4f}")

        # per-(class, patch) kappa
        best_idx = np.argmax(sc, axis=2)
        kappa_cj = MML_GRID[best_idx]                      # (10, 729)
        tab = logtab_backoff(counts, class_count, par, kappa_cj)
        acc_cj, _ = evaluate(sva, yva, tab, log_prior(class_count, 0.05))
        # per-class kappa (MML summed over patches within a class)
        sc_cls = sc.sum(axis=1)                            # (10, G)
        kappa_c = MML_GRID[np.argmax(sc_cls, axis=1)]      # (10,)
        tab = logtab_backoff(counts, class_count, par, kappa_c)
        acc_c, _ = evaluate(sva, yva, tab, log_prior(class_count, 0.05))
        report[name].update({
            "kappa_cj_quantiles": {q: float(np.quantile(kappa_cj, q))
                                   for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "val_acc_at_mml_per_class": acc_c,
            "val_acc_at_mml_per_cpt": acc_cj,
            "kappa_per_class": [float(v) for v in kappa_c],
        })
        print(f"    per-class kappa={list(kappa_c)} -> val={acc_c:.4f}")
        print(f"    per-CPT kappa (median {np.median(kappa_cj):g}) -> val={acc_cj:.4f}")
    report["seconds"] = time.perf_counter() - t0
    note(journal, "mml", report)


def section_temper(journal, quick=False):
    """DIAGNOSTIC (modelling extension, not strict NB): tempered likelihood."""
    str_, ytr, sva, yva = cached_split(1322)
    counts, class_count = train_counts(str_, ytr)
    out = {"note": "P(y) * prod P(z_j|y)^w -- strict NB requires w=1; diagnostic only"}
    for name, alpha in (("uniform_a0.05", 0.05), ("uniform_a0.01", 0.01)):
        tab = logtab_uniform(counts, class_count, alpha)
        pri = log_prior(class_count, alpha)
        out[name] = {}
        for w in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
            acc, _ = evaluate(sva, yva, tab, pri, weight=w)
            out[name][str(w)] = acc
        bk = max(out[name], key=lambda kk: out[name][kk])
        out[name + "_best"] = {"w": float(bk), "val_acc": out[name][bk]}
        print(f"  {name}: best w={bk} val={out[name][bk]:.4f}")
    note(journal, "temper_diagnostic", out)


def section_nested(journal, quick=False):
    """Pick alpha by a nested TRAIN-internal split (no validation peeking at all).

    The primary val split is used only to *report*, never to choose: alpha is
    selected on a sub-split carved out of the 49,999 training rows, so the
    procedure is honest "use training data to set the hyper-parameter".
    """
    from v4_common import stratified_split

    str_, ytr, sva, yva = cached_split(1322)
    counts, class_count = train_counts(str_, ytr)
    alphas = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 3e-3, 1e-3, 3e-4,
                       1e-4, 1e-5, 1e-6])
    out = {"protocol": "alpha chosen on a 1/6 sub-split of the 49,999 training rows",
           "by_frac": {}}
    for frac in (1 / 6, 1 / 4, 1 / 2):
        inner_tr, inner_va = stratified_split(ytr, frac, seed=1322)
        c_inner, cc_inner = train_counts(str_[inner_tr], ytr[inner_tr])
        row = {}
        for a in alphas:
            acc, _ = evaluate(str_[inner_va], ytr[inner_va],
                              logtab_uniform(c_inner, cc_inner, a),
                              log_prior(cc_inner, a))
            row[str(a)] = acc
        best = max(row, key=lambda k: row[k])
        # now apply the chosen alpha to the FULL training set, report on val
        tab = logtab_uniform(counts, class_count, float(best))
        full_acc, _ = evaluate(sva, yva, tab, log_prior(class_count, float(best)))
        out["by_frac"][str(frac)] = {
            "inner_curve": row, "chosen_alpha": float(best),
            "val_acc_from_choice": full_acc,
        }
        print(f"  inner_frac={frac:.3f}: chose alpha={best} -> val={full_acc:.4f}")
    note(journal, "nested_alpha_selection", out)


def section_witten_bell(journal, quick=False):
    """Witten-Bell / absolute-discount style adaptive smoothing.

    Additive pseudo-count alpha_j = T_j / (N_j + T_j) * scale, where T_j is the
    number of distinct states observed for CPT j and N_j the total count.  This
    is the classic "smooth in proportion to how much unseen mass remains"
    heuristic: dense CPTs (centre patches) get little smoothing, sparse ones
    (border patches, digit 1) get more.  Estimated from training counts only.
    """
    str_, ytr, sva, yva = cached_split(1322)
    counts, class_count = train_counts(str_, ytr)
    seen = counts > 0
    T = seen.sum(axis=2).astype(np.float64)          # (10, 729) distinct states
    N = counts.sum(axis=2).astype(np.float64)        # (10, 729) total
    out = {}
    for scale in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
        # alpha_cj in Lidstone units; kappa_cj = alpha_cj * N_STATES
        alpha_cj = (T / (N + T)) * scale
        kappa_cj = alpha_cj * N_STATES
        tab = logtab_backoff(counts, class_count, parent_uniform(counts, class_count),
                             kappa_cj)
        acc, _ = evaluate(sva, yva, tab, log_prior(class_count, 0.05))
        out[str(scale)] = acc
        print(f"  WB scale={scale:<6g} alpha_cj median={np.median(alpha_cj):.2e} "
              f"val={acc:.4f}")
    best = max(out, key=lambda k: out[k])
    # per-class Witten-Bell (single alpha per class)
    Tc = T.sum(axis=1)
    Nc = N.sum(axis=1)
    out2 = {}
    for scale in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0):
        alpha_c = (Tc / (Nc + Tc)) * scale
        kappa_c = alpha_c * N_STATES
        tab = logtab_backoff(counts, class_count, parent_uniform(counts, class_count),
                             kappa_c)
        acc, _ = evaluate(sva, yva, tab, log_prior(class_count, 0.05))
        out2[str(scale)] = acc
        print(f"  WB-per-class scale={scale:<6g} val={acc:.4f}")
    note(journal, "witten_bell", {
        "per_cpt": out, "per_class": out2, "best": float(best),
        "alpha_cj_median": float(np.median((T / (N + T)) * 1.0)),
    })


def section_fine_alpha(journal, quick=False):
    """High-resolution scan of the small-alpha optimum + its zero-count regime."""
    str_, ytr, sva, yva = cached_split(1322)
    counts, class_count = train_counts(str_, ytr)
    alphas = np.array([3e-4, 4e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3,
                       5e-3, 7e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2])
    curve = {}
    for a in alphas:
        acc, _ = evaluate(sva, yva, logtab_uniform(counts, class_count, a),
                          log_prior(class_count, a))
        curve[str(a)] = acc
        print(f"  alpha={a:<8g} val={acc:.4f}")
    best = max(curve, key=lambda k: curve[k])
    note(journal, "fine_alpha", {"curve": curve, "best_alpha": float(best),
                                 "best_val": curve[best]})
    print(f"  -> best alpha={best} val={curve[best]:.4f}")


def section_prior(journal, quick=False):
    """Class-marginal prior variants (uniform vs empirical vs smoothed)."""
    str_, ytr, sva, yva = cached_split(1322)
    counts, class_count = train_counts(str_, ytr)
    out = {}
    for alpha_feat in (0.05, 0.01):
        tab = logtab_uniform(counts, class_count, alpha_feat)
        for name, pri in (
            ("empirical_alpha0.05", log_prior(class_count, 0.05)),
            ("empirical_alpha1", log_prior(class_count, 1.0)),
            ("uniform", np.full(N_CLASSES, -math.log(N_CLASSES))),
        ):
            acc, _ = evaluate(sva, yva, tab, pri)
            out[f"feat{alpha_feat}_{name}"] = acc
            print(f"  feat_alpha={alpha_feat} prior={name:<22} val={acc:.4f}")
    note(journal, "class_prior", out)


def section_falsify(journal, quick=False):
    """Adversarial adjudication of the tiny-alpha candidate vs the incumbent.

    Uses the real ``naive_bayes.NaiveBayesClassifier`` as the incumbent and the
    self-contained candidate module as the challenger: compare_candidates
    (McNemar + paired bootstrap), cross_seed on seeds 7 / 2024, and top
    confusions.  This is the falsification test the brief mandates.
    """
    import importlib

    inc_mod = importlib.import_module("naive_bayes")
    cand_mod = importlib.import_module("builder_b_nb_tinyalpha")

    Xtr, ytr, Xva, yva = raw_split(1322)
    inc_factory = lambda: inc_mod.NaiveBayesClassifier(alpha=0.05)
    cand_factory = lambda: cand_mod.NaiveBayesClassifier(alpha=1e-4)

    out = {}
    res = compare_candidates(Xtr, ytr, Xva, yva, inc_factory, cand_factory)
    out["main_split"] = res
    print(f"  main split : incumbent={res['incumbent_accuracy']:.4f} "
          f"candidate={res['challenger_accuracy']:.4f} delta={res['delta']:+.4f} "
          f"mcnemar_p={res['mcnemar']['p_value']:.3e} "
          f"boot_CI=[{res['bootstrap']['ci_low']:+.4f},{res['bootstrap']['ci_high']:+.4f}] "
          f"real_gain={res['real_gain']}")

    pred = cand_factory().fit(Xtr, ytr).predict(Xva)
    f1, cm = macro_f1(yva, pred)
    out["candidate_macro_f1"] = f1
    out["candidate_top_confusions"] = top_confusions(cm)
    print(f"  candidate macro_f1={f1:.4f} top_confusions={out['candidate_top_confusions']}")

    cs_inc = cross_seed(inc_factory)
    cs_cand = cross_seed(cand_factory)
    out["cross_seed"] = {str(s): {"incumbent": cs_inc[s], "candidate": cs_cand[s],
                                  "delta": cs_cand[s] - cs_inc[s]} for s in cs_inc}
    for s in cs_inc:
        print(f"  seed {s}: incumbent={cs_inc[s]:.4f} candidate={cs_cand[s]:.4f} "
              f"delta={cs_cand[s]-cs_inc[s]:+.4f}")

    # per-seed McNemar (candidate vs incumbent) for extra rigour
    per_seed = {}
    for s in EXTRA_SEEDS:
        Xt, yt, Xv, yv = raw_split(s)
        pi = inc_factory().fit(Xt, yt).predict(Xv)
        pc = cand_factory().fit(Xt, yt).predict(Xv)
        per_seed[str(s)] = mcnemar(yv, pi, pc)
        print(f"  seed {s} McNemar: b={per_seed[str(s)]['b']} c={per_seed[str(s)]['c']} "
              f"p={per_seed[str(s)]['p_value']:.3e}")
    out["cross_seed_mcnemar"] = per_seed
    note(journal, "falsify", out)


SECTIONS = {
    "baseline": section_baseline,
    "alpha_curve": section_alpha_curve,
    "zero_stats": section_zero_stats,
    "backoff": section_backoff,
    "mml": section_mml,
    "temper": section_temper,
    "prior": section_prior,
    "nested": section_nested,
    "witten_bell": section_witten_bell,
    "fine_alpha": section_fine_alpha,
    "falsify": section_falsify,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=["all"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    names = list(SECTIONS) if "all" in args.only else args.only
    journal = load_journal()
    print(f"lgamma self-test (max abs err vs math.lgamma): {_selftest_lgamma():.3e}")
    for name in names:
        t = time.perf_counter()
        print(f"== {name} ==")
        SECTIONS[name](journal, quick=args.quick)
        print(f"   ({time.perf_counter()-t:.1f}s)")


if __name__ == "__main__":
    main()
