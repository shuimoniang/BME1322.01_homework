"""CRITIC alpha -- adversarial falsification of Builder A's dilated-binary patch NB.

This script re-tests every number Builder A claims, then pushes the frontier.
It is deliberately a *separate* implementation from ``builder_a_candidate.py``:
a single generic sparse-CPT engine parameterised by feature geometry, so that

  * the incumbent 2x2 / 5-gray-level model,
  * Builder A's 4x4 dilated binary model,
  * every frontier neighbour tested in attack 6

all run through byte-identical CPT / log-likelihood code.  If a claimed gain
survives here it is not a quirk of one particular implementation.

Design note -- username alpha is *not* a fit-time parameter.  The sparse CPT
stores, for every observed (feature, state) pair, its per-class integer count.
The Lidstone log-probabilities are derived from those counts for any alpha on
demand, so one fit + one validation encoding serves a whole alpha sweep.  That
keeps the sweep cheap and, more importantly, guarantees all alphas see exactly
the same counts.

Only the 60,000 training images are ever loaded (through v4_common).  The test
set is never read.

Usage:
    python critic_alpha_experiments.py --only repro decomposition alpha_curves
    python critic_alpha_experiments.py --only all
Sections: repro decomposition alpha_curves nested cross_seed8 memtime frontier
          best8seed geometry_only

Writes results/critic_alpha_raw.json (merged across runs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V4 = HERE.parent.parent
ROOT = V4.parent
for _path in (str(HERE), str(V4), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import v4_common as V4C  # noqa: E402

RESULTS = HERE.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
OUT = RESULTS / "critic_alpha_raw.json"

SEEDS8 = [1322, 7, 2024, 1, 42, 99, 123, 555]


# ======================================================================
# 1. generic geometry + sparse-CPT naive Bayes
# ======================================================================
def grid_shape(ph, pw, ih, iw, stride):
    extent_h = (ph - 1) * ih + 1
    extent_w = (pw - 1) * iw + 1
    return (28 - extent_h) // stride + 1, (28 - extent_w) // stride + 1


def _uniform_edges(levels):
    return np.linspace(0.0, 255.0, levels + 1)[1:-1]


def box_blur(img, k):
    """k x k mean filter, stride 1, reflect-padded, same output size."""
    if k <= 1:
        return img
    pad = k - 1
    padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    total = np.zeros_like(img, dtype=np.float64)
    for di in range(k):
        for dj in range(k):
            total += padded[:, di:di + 28, dj:dj + 28]
    return total / float(k * k)


def encode_states(X, geom):
    """Encode a geometry's patch patterns as one integer state per feature.

    geom keys: ph, pw, ih, iw, stride, mode ('binary'|'levels'), thr, levels,
    edges, pool.  Returns (n_samples, n_features) int64 states.
    """
    ph, pw = geom["ph"], geom["pw"]
    ih, iw = geom.get("ih", geom.get("inner", 1)), geom.get("iw", geom.get("inner", 1))
    stride = geom.get("stride", 1)
    gh, gw = grid_shape(ph, pw, ih, iw, stride)
    img = X.reshape(-1, 28, 28).astype(np.float64)
    pool = geom.get("pool", 0)
    if pool:
        img = box_blur(img, pool)

    if geom["mode"] == "binary":
        bands = (img >= geom.get("thr", 72.0)).astype(np.int64)
        base = 2
    else:
        levels = geom["levels"]
        edges = np.asarray(geom.get("edges", _uniform_edges(levels)), dtype=np.float64)
        bands = np.zeros(img.shape, dtype=np.int64)
        for edge in edges:
            bands += img >= edge
        base = levels

    n_pixels = ph * pw
    out = np.zeros((img.shape[0], gh, gw), dtype=np.int64)
    order = 0
    for i in range(ph):
        for j in range(pw):
            rows = slice(i * ih, i * ih + (gh - 1) * stride + 1, stride)
            cols = slice(j * iw, j * iw + (gw - 1) * stride + 1, stride)
            sub = bands[:, rows, cols]
            if geom["mode"] == "binary":
                out |= sub << order
            else:
                out += (base ** (n_pixels - 1 - order)) * sub
            order += 1
    return out.reshape(img.shape[0], gh * gw)


def geom_nstates(geom):
    if geom["mode"] == "binary":
        return 1 << (geom["ph"] * geom["pw"])
    return geom["levels"] ** (geom["ph"] * geom["pw"])


class SparseGeometryNB:
    """Naive Bayes over geometry-dependent patch features, sparse exact CPTs.

    ``fit`` stores integer per-class counts of observed (feature, state) pairs.
    ``predict_at(X, alpha)`` evaluates the exact naive Bayes log-likelihood
    ``log P(y) + sum_j log P(z_j | y)`` for any alpha without refitting.
    """

    def __init__(self, geom, alpha=1.0, count_dtype=np.uint32):
        self.geom = dict(geom)
        self.alpha = float(alpha)
        self.count_dtype = count_dtype
        self.nstates = geom_nstates(self.geom)

    # ---------------- fit ----------------
    def fit(self, X, y):
        y = np.asarray(y).astype(np.int64)
        states = encode_states(X, self.geom)
        n, n_features = states.shape
        self.n_features = n_features
        offsets = (np.arange(n_features, dtype=np.int64) * self.nstates)[None, :]
        flat = states + offsets                       # (n, F) int64
        self.keys = np.unique(flat)                   # observed feature*state
        self.class_count = np.bincount(y, minlength=10).astype(np.float64)
        counts = np.empty((10, self.keys.size), dtype=self.count_dtype)
        for c in range(10):
            sub = flat[y == c].ravel()
            pos = np.searchsorted(self.keys, sub)
            counts[c] = np.bincount(pos, minlength=self.keys.size).astype(self.count_dtype)
        self.counts = counts
        self.n_train = n
        self.n_observed = int(self.keys.size)
        self.is_fitted_ = True
        return self

    # ---------------- alpha-independent per-sample hit map ----------------
    def _hits(self, X):
        states = encode_states(X, self.geom)
        n = states.shape[0]
        offsets = (np.arange(self.n_features, dtype=np.int64) * self.nstates)[None, :]
        flat = states + offsets
        pos = np.searchsorted(self.keys, flat)
        np.clip(pos, 0, self.keys.size - 1, out=pos)
        found = self.keys[pos] == flat
        si, fi = np.nonzero(found)
        return si, pos[si, fi], n

    def score_at(self, hits, alpha):
        si, ki, n = hits
        alpha = float(alpha)
        nc = self.class_count
        denom = nc + self.nstates * alpha
        default = np.log(alpha / denom)                        # (C,)
        if ki.size:
            obs = np.log((self.counts[:, ki].astype(np.float64) + alpha) / denom[:, None])
            corr = obs - default[:, None]                      # (C, H)
        scores = np.tile(self.n_features * default, (n, 1))
        if ki.size:
            for c in range(10):
                scores[:, c] += np.bincount(si, weights=corr[c], minlength=n)
        scores += np.log((nc + alpha) / (self.n_train + 10.0 * alpha))[None, :]
        return scores

    def predict_hits(self, hits, alpha):
        return np.argmax(self.score_at(hits, alpha), axis=1).astype(np.int64)

    def accuracy_at(self, hits, y_val, alpha):
        return float(np.mean(self.predict_hits(hits, alpha) == np.asarray(y_val)))

    # ---------------- drop-in protocol ----------------
    def predict(self, X):
        return self.predict_hits(self._hits(X), self.alpha)

    def predict_proba(self, X):
        log_scores = self.score_at(self._hits(X), self.alpha)
        shifted = log_scores - np.max(log_scores, axis=1, keepdims=True)
        proba = np.exp(shifted)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba


# ---- reference geometries ----
GEOM_INCUMBENT = {"ph": 2, "pw": 2, "inner": 1, "stride": 1, "mode": "levels",
                  "levels": 5, "edges": _uniform_edges(5)}
GEOM_BUILDER_A = {"ph": 4, "pw": 4, "inner": 2, "stride": 1, "mode": "binary",
                  "thr": 72.0}


def make_model(geom, alpha=1.0):
    return SparseGeometryNB(geom, alpha=alpha)


# ======================================================================
# helpers
# ======================================================================
def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def load_raw():
    if OUT.exists():
        try:
            return json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_raw(report):
    OUT.write_text(json.dumps(_jsonable(report), indent=2, ensure_ascii=False),
                   encoding="utf-8")


def fit_split(geom, seed, alpha=1.0):
    Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
    model = SparseGeometryNB(geom, alpha=alpha).fit(Xtr, ytr)
    return model, Xva, yva


# ======================================================================
# attack 2 -- decomposition
# ======================================================================
def section_repro(report):
    print("\n=== repro: incumbent and Builder A on seed 1322 ===")
    Xtr, ytr, Xva, yva = V4C.load_split()
    out = {}
    for name, geom in (("incumbent_2x2_L5", GEOM_INCUMBENT),
                       ("builder_a_4x4s2", GEOM_BUILDER_A)):
        model = SparseGeometryNB(geom, alpha=1.0).fit(Xtr, ytr)
        hits = model._hits(Xva)
        row = {"n_features": int(model.n_features), "nstates": int(model.nstates),
               "n_observed": int(model.n_observed), "accuracy": {}}
        for alpha in (0.05, 0.001, 0.0001):
            row["accuracy"][str(alpha)] = model.accuracy_at(hits, yva, alpha)
        out[name] = row
        print(f"  {name}: F={row['n_features']} S={row['nstates']} "
              f"obs={row['n_observed']} acc={row['accuracy']}")
    report.setdefault("repro", {})["seed1322"] = out
    return out


# ======================================================================
# attack 2 -- geometry vs smoothing
# ======================================================================
ALPHA_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
# extended grid reaching the regime the curves are still rising into
EXT_GRID = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8]


def _alpha_curve(model, hits, y_val, alphas=ALPHA_GRID):
    return {f"{a:.0e}": model.accuracy_at(hits, y_val, a) for a in alphas}


def section_decomposition(report, seeds=(1322, 7, 2024)):
    print("\n=== decomposition: geometry vs smoothing ===")
    table = {}
    for seed in seeds:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        inc = SparseGeometryNB(GEOM_INCUMBENT, alpha=1.0).fit(Xtr, ytr)
        cand = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
        hi, hc = inc._hits(Xva), cand._hits(Xva)
        ic = _alpha_curve(inc, hi, yva)
        cc = _alpha_curve(cand, hc, yva)
        inc_best_a = max(ic, key=ic.get)
        cand_best_a = max(cc, key=cc.get)
        row = {
            "incumbent_curve": ic, "candidate_curve": cc,
            "incumbent_best_alpha": inc_best_a,
            "candidate_best_alpha": cand_best_a,
            "incumbent_best_acc": ic[inc_best_a],
            "candidate_best_acc": cc[cand_best_a],
            "geometry_only_both_best": cc[cand_best_a] - ic[inc_best_a],
            "geometry_only_matched_1e-4": cc["1e-04"] - ic["1e-04"],
            "geometry_only_matched_1e-3": cc["1e-03"] - ic["1e-03"],
            "geometry_only_matched_1e-2": cc["1e-02"] - ic["1e-02"],
            "smoothing_only_incumbent": ic[inc_best_a] - ic["5e-02"] if "5e-02" in ic else None,
        }
        # predictions at matched alpha for McNemar / bootstrap
        for a, label in ((0.0001, "1e-04"), (0.001, "1e-03")):
            pi = inc.predict_hits(hi, a)
            pc = cand.predict_hits(hc, a)
            row[f"mcnemar_{label}"] = V4C.mcnemar(yva, pi, pc)
            row[f"bootstrap_{label}"] = V4C.paired_bootstrap(yva, pi, pc)
        table[str(seed)] = row
        print(f"  seed {seed}: inc_best={row['incumbent_best_acc']:.4f}@{inc_best_a} "
              f"cand_best={row['candidate_best_acc']:.4f}@{cand_best_a} "
              f"geom_only(both best)={row['geometry_only_both_best']:+.4f} "
              f"geom_only@1e-4={row['geometry_only_matched_1e-4']:+.4f}")
    report.setdefault("decomposition", {}).update(table)
    return table


# ======================================================================
# attack 3 -- alpha curves per seed + nested selection
# ======================================================================
def _stratified_split_inner(y, frac, seed):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = max(1, round(idx.size * frac))
        va.append(idx[:k]); tr.append(idx[k:])
    return np.concatenate(tr), np.concatenate(va)


def section_alpha_curves(report, seeds=(1322, 7, 2024)):
    print("\n=== attack 3: alpha curves per seed ===")
    out = {}
    fine = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    for seed in seeds:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        m = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
        hits = m._hits(Xva)
        curve = _alpha_curve(m, hits, yva, fine)
        best = max(curve, key=curve.get)
        worst = min(curve, key=curve.get)
        out[str(seed)] = {"curve": curve, "argmax_alpha": best, "argmax_acc": curve[best],
                          "argmin_alpha": worst, "argmin_acc": curve[worst]}
        print(f"  seed {seed}: argmax alpha={best} acc={curve[best]:.4f}; "
              f"curve={ {k: round(v,4) for k,v in curve.items()} }")
    report.setdefault("alpha_curves", {}).update(out)
    return out


def section_nested(report, seeds=(1322, 7, 2024)):
    print("\n=== attack 3: training-internal nested alpha selection ===")
    fine = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
    out = {}
    for seed in seeds:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        inner_tr, inner_va = _stratified_split_inner(ytr, 1 / 6, seed + 1000)
        m_in = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr[inner_tr], ytr[inner_tr])
        h_in = m_in._hits(Xtr[inner_va])
        inner_curve = _alpha_curve(m_in, h_in, ytr[inner_va], fine)
        chosen = max(inner_curve, key=inner_curve.get)
        m_full = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
        h_full = m_full._hits(Xva)
        outer_acc = m_full.accuracy_at(h_full, yva, float(chosen))
        outer_at_1e3 = m_full.accuracy_at(h_full, yva, 1e-3)
        outer_at_1e4 = m_full.accuracy_at(h_full, yva, 1e-4)
        out[str(seed)] = {"inner_curve": inner_curve, "chosen_alpha": chosen,
                          "outer_acc_at_chosen": outer_acc,
                          "outer_acc_at_1e-3": outer_at_1e3,
                          "outer_acc_at_1e-4": outer_at_1e4,
                          "outer_best_alpha": max(_alpha_curve(m_full, h_full, yva, fine),
                                                  key=_alpha_curve(m_full, h_full, yva, fine).get)}
        print(f"  seed {seed}: nested picks alpha={chosen} (inner {inner_curve[chosen]:.4f}) "
              f"-> outer {outer_acc:.4f} (1e-3 {outer_at_1e3:.4f}, 1e-4 {outer_at_1e4:.4f})")
    report.setdefault("nested", {}).update(out)
    return out


# ======================================================================
# attack 4 -- 8-seed falsification
# ======================================================================
def section_cross_seed8(report, alpha_a=1e-3, inc_alpha=0.05, inc_best=1e-4):
    print(f"\n=== attack 4: 8-seed falsification (A alpha={alpha_a}) ===")
    rows = {}
    for seed in SEEDS8:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        inc = SparseGeometryNB(GEOM_INCUMBENT, alpha=1.0).fit(Xtr, ytr)
        cand = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
        hi, hc = inc._hits(Xva), cand._hits(Xva)
        a_inc = inc.accuracy_at(hi, yva, inc_alpha)
        a_inc_best = inc.accuracy_at(hi, yva, inc_best)
        a_cand = cand.accuracy_at(hc, yva, alpha_a)
        pi = inc.predict_hits(hi, inc_best)
        pc = cand.predict_hits(hc, alpha_a)
        rows[str(seed)] = {
            "incumbent_0.05": a_inc, "incumbent_1e-4": a_inc_best,
            "candidate": a_cand,
            "delta_vs_inc0.05": a_cand - a_inc,
            "delta_vs_inc_best": a_cand - a_inc_best,
            "mcnemar_vs_inc_best": V4C.mcnemar(yva, pi, pc),
        }
        print(f"  seed {seed:>5}: inc0.05={a_inc:.4f} inc1e-4={a_inc_best:.4f} "
              f"cand={a_cand:.4f} delta_best={a_cand-a_inc_best:+.4f}")
    deltas = [rows[s]["delta_vs_inc_best"] for s in rows]
    report.setdefault("cross_seed8", {})["builder_a_alpha_1e-3"] = {
        "per_seed": rows,
        "mean_candidate": float(np.mean([rows[s]["candidate"] for s in rows])),
        "min_candidate": float(np.min([rows[s]["candidate"] for s in rows])),
        "mean_delta_vs_inc_best": float(np.mean(deltas)),
        "min_delta_vs_inc_best": float(np.min(deltas)),
        "positive_on_all_seeds": bool(all(d > 0 for d in deltas)),
    }
    summary = report["cross_seed8"]["builder_a_alpha_1e-3"]
    print(f"  -> mean cand={summary['mean_candidate']:.4f} min={summary['min_candidate']:.4f} "
          f"mean delta={summary['mean_delta_vs_inc_best']:+.4f} "
          f"positive_all={summary['positive_on_all_seeds']}")
    return rows


# ======================================================================
# attack 5 -- memory / time
# ======================================================================
def _peak_working_set_mb():
    """Windows peak working set (real RSS proxy) via psapi, else tracemalloc."""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                                               wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                   ctypes.byref(counters), counters.cb)
        return counters.PeakWorkingSetSize / 1e6
    except Exception:
        return None


def section_memtime(report):
    print("\n=== attack 5: memory / wall-clock feasibility (subprocess-isolated) ===")
    import subprocess
    out = {}
    for label, geom_name in (("incumbent_2x2_L5", "GEOM_INCUMBENT"),
                             ("builder_a_4x4s2", "GEOM_BUILDER_A")):
        code = f"""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, '..'); sys.path.insert(0, '../../..')
import time, tracemalloc, numpy as np
import critic_alpha_experiments as C
import v4_common as V4C
geom = C.{geom_name}
Xtr, ytr, Xva, yva = V4C.load_split()
m = C.SparseGeometryNB(geom, alpha=1.0)
tracemalloc.start()
t0 = time.perf_counter(); m.fit(Xtr, ytr); t_fit = time.perf_counter() - t0
cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
t1 = time.perf_counter(); h = m._hits(Xva); acc = m.accuracy_at(h, yva, 1e-3)
t_pred = time.perf_counter() - t1
print("RESULT", t_fit, t_pred, peak/1e6, m.n_observed, m.keys.nbytes/1e6,
      m.counts.nbytes/1e6, acc, C._peak_working_set_mb())
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=str(HERE))
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
        if not line:
            out[label] = {"error": proc.stderr[-800:]}
            print(f"  {label}: FAILED {proc.stderr[-200:]}")
            continue
        parts = line[0].split()[1:]
        t_fit, t_pred, peak_mb, nobs, kbytes, cbytes, acc, ws = parts
        out[label] = {
            "fit_seconds_49999": round(float(t_fit), 2),
            "predict_seconds_10001": round(float(t_pred), 2),
            "tracemalloc_peak_mb": round(float(peak_mb), 1),
            "proc_peak_ws_mb": round(float(ws), 1),
            "n_observed": int(nobs),
            "keys_mb": round(float(kbytes), 1),
            "counts_mb": round(float(cbytes), 1),
            "acc_1e-3": float(acc),
        }
        print(f"  {label}: fit={float(t_fit):.2f}s pred={float(t_pred):.2f}s "
              f"tracemalloc_peak={float(peak_mb):.0f}MB proc_peak_ws={float(ws):.0f}MB "
              f"keys+counts={float(kbytes)+float(cbytes):.0f}MB acc={float(acc):.4f}")
    a = out.get("builder_a_4x4s2", {})
    if "fit_seconds_49999" in a:
        out["extrapolation_60k_train"] = {
            "fit_seconds_est": round(a["fit_seconds_49999"] * 60000 / 49999, 2),
            "pred_seconds_est_10000": round(a["predict_seconds_10001"] * 10000 / 10001, 2),
            "tracemalloc_peak_mb_est": round(a["tracemalloc_peak_mb"] * 60000 / 49999, 1),
        }
    report.setdefault("memtime", {})["seed1322"] = out
    return out


def section_memtime60(report):
    """The course harness path: fit on the full 60,000, predict on 10,000."""
    print("\n=== attack 5b: production 60k-train path (isolated subprocesses) ===")
    import subprocess
    out = {}
    for label, modname, alpha in (("builder_a_candidate", "builder_a_candidate", 1e-3),
                                  ("critic_alpha_best", "critic_alpha_best", 1e-6)):
        code = f"""
import sys, time; sys.path.insert(0, '.'); sys.path.insert(0, '../..')
import numpy as np, v4_common as V4C, ctypes
from ctypes import wintypes
class PMC(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('p1',wintypes.DWORD),('PWD',ctypes.c_size_t)]+[('x',ctypes.c_size_t)]*7
k=ctypes.windll.kernel32; p=ctypes.windll.psapi
k.GetCurrentProcess.restype=wintypes.HANDLE
p.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]
def peak():
    c=PMC(); c.cb=ctypes.sizeof(c); p.GetProcessMemoryInfo(k.GetCurrentProcess(),ctypes.byref(c),c.cb); return c.PWD/1e6
import {modname} as M
X, y = V4C.load_train()
m = M.NaiveBayesClassifier(alpha={alpha})
t0=time.perf_counter(); m.fit(X, y); tfit=time.perf_counter()-t0
t1=time.perf_counter(); _ = m.predict(X[:10000]); tpred=time.perf_counter()-t1
print('RESULT', tfit, tpred, peak())
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              cwd=str(HERE))
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
        if not line:
            out[label] = {"error": proc.stderr[-600:]}
            print(f"  {label}: FAILED {proc.stderr[-200:]}")
            continue
        tfit, tpred, ws = line[0].split()[1:]
        out[label] = {"alpha": alpha, "fit_seconds_60000": round(float(tfit), 2),
                      "predict_seconds_10000": round(float(tpred), 2),
                      "proc_peak_ws_mb": round(float(ws), 1)}
        print(f"  {label:<22} alpha={alpha:g} fit60000={float(tfit):.2f}s "
              f"predict10000={float(tpred):.2f}s peak_ws={float(ws):.0f}MB")
    report.setdefault("memtime60", {}).update(out)
    return out


# ======================================================================
# attack 6 -- frontier
# ======================================================================
def frontier_geometries():
    variants = {}

    def g(**kw):
        base = {"stride": 1, "mode": "binary", "thr": 72.0}
        base.update(kw)
        return base

    variants["A_4x4_s2_bin"] = g(ph=4, pw=4, inner=2)
    variants["A_4x4_s2_L3"] = g(ph=4, pw=4, inner=2, mode="levels", levels=3)
    variants["5x5_s2_bin"] = g(ph=5, pw=5, inner=2)
    variants["3x5_s2_bin"] = g(ph=3, pw=5, inner=2)
    variants["5x3_s2_bin"] = g(ph=5, pw=3, inner=2)
    variants["4x4_s3_bin"] = g(ph=4, pw=4, inner=3)
    variants["4x4_s23_bin"] = g(ph=4, pw=4, ih=2, iw=3)
    variants["4x4_s32_bin"] = g(ph=4, pw=4, ih=3, iw=2)
    variants["A_pool2"] = g(ph=4, pw=4, inner=2, pool=2)
    variants["A_pool3"] = g(ph=4, pw=4, inner=2, pool=3)
    variants["4x4_s2_bin_thr48"] = g(ph=4, pw=4, inner=2, thr=48.0)
    variants["4x4_s2_bin_thr96"] = g(ph=4, pw=4, inner=2, thr=96.0)
    variants["3x3_s3_L2"] = g(ph=3, pw=3, inner=3)
    variants["4x4_s2_s4_bin"] = g(ph=4, pw=4, inner=2, stride=2)
    variants["6x6_s2_bin"] = g(ph=6, pw=6, inner=2)
    variants["4x4_s2_L3_ink"] = g(ph=4, pw=4, inner=2, mode="levels", levels=3,
                                  edges=np.array([0.5, 128.0]))
    return variants


def frontier2_geometries():
    """Refined neighbourhood around the 3x5 / 4x4 winners found by frontier()."""
    variants = {}

    def g(**kw):
        base = {"stride": 1, "mode": "binary", "thr": 72.0}
        base.update(kw)
        return base

    variants["3x5_s2_bin"] = g(ph=3, pw=5, inner=2)
    variants["5x3_s2_bin"] = g(ph=5, pw=3, inner=2)
    variants["3x4_s2_bin"] = g(ph=3, pw=4, inner=2)
    variants["4x3_s2_bin"] = g(ph=4, pw=3, inner=2)
    variants["3x6_s2_bin"] = g(ph=3, pw=6, inner=2)
    variants["6x3_s2_bin"] = g(ph=6, pw=3, inner=2)
    variants["2x5_s2_bin"] = g(ph=2, pw=5, inner=2)
    variants["5x2_s2_bin"] = g(ph=5, pw=2, inner=2)
    variants["2x6_s2_bin"] = g(ph=2, pw=6, inner=2)
    variants["6x2_s2_bin"] = g(ph=6, pw=2, inner=2)
    variants["3x5_s2_bin_thr48"] = g(ph=3, pw=5, inner=2, thr=48.0)
    variants["3x5_s2_bin_thr96"] = g(ph=3, pw=5, inner=2, thr=96.0)
    variants["3x5_s2_bin_thr32"] = g(ph=3, pw=5, inner=2, thr=32.0)
    variants["3x5_s2_bin_thr128"] = g(ph=3, pw=5, inner=2, thr=128.0)
    variants["3x5_s2_L3"] = g(ph=3, pw=5, inner=2, mode="levels", levels=3)
    variants["4x4_s2_bin"] = g(ph=4, pw=4, inner=2)
    variants["3x5_s2_bin_str2"] = g(ph=3, pw=5, inner=2, stride=2)
    variants["2x8_s2_bin"] = g(ph=2, pw=8, inner=2)
    variants["8x2_s2_bin"] = g(ph=8, pw=2, inner=2)
    return variants


def _screen(geom_variants, seed=1322, grid=EXT_GRID, label="screen"):
    Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
    rows = {}
    for name, geom in geom_variants.items():
        try:
            t0 = time.perf_counter()
            m = SparseGeometryNB(geom, alpha=1.0).fit(Xtr, ytr)
            hits = m._hits(Xva)
            curve = _alpha_curve(m, hits, yva, grid)
            best = max(curve, key=curve.get)
            rows[name] = {"n_features": int(m.n_features), "nstates": int(m.nstates),
                          "n_observed": int(m.n_observed), "best_alpha": best,
                          "best_acc": curve[best], "curve": curve,
                          "seconds": round(time.perf_counter() - t0, 1)}
            print(f"  {name:<24} F={m.n_features:<4} S={m.nstates:<12} "
                  f"obs={m.n_observed:>9} best={curve[best]:.4f}@{best} "
                  f"a1e-6={curve.get('1e-06', float('nan')):.4f} "
                  f"({rows[name]['seconds']}s)")
        except MemoryError as exc:
            rows[name] = {"error": f"MemoryError: {exc}"}
            print(f"  {name:<24} MEMORYERROR")
        except Exception as exc:  # noqa: BLE001
            rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name:<24} ERROR {type(exc).__name__}: {exc}")
    return rows


def section_frontier2(report):
    print("\n=== attack 6b: refined frontier screen (extended alpha grid) ===")
    rows = _screen(frontier2_geometries(), label="frontier2")
    report.setdefault("frontier2", {})["1322"] = rows
    return rows


def section_deep_alpha(report):
    """Full alpha curve into the 1e-8 regime for A and the top frontier rivals."""
    print("\n=== attack 6c: deep alpha curves (down to 1e-8) ===")
    grid = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    cands = {
        "A_4x4_s2_bin": GEOM_BUILDER_A,
        "3x5_s2_bin": {"ph": 3, "pw": 5, "inner": 2, "stride": 1, "mode": "binary", "thr": 72.0},
        "5x3_s2_bin": {"ph": 5, "pw": 3, "inner": 2, "stride": 1, "mode": "binary", "thr": 72.0},
        "3x4_s2_bin": {"ph": 3, "pw": 4, "inner": 2, "stride": 1, "mode": "binary", "thr": 72.0},
    }
    out = {}
    for seed in (1322, 7, 2024):
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        rows = {}
        for name, geom in cands.items():
            m = SparseGeometryNB(geom, alpha=1.0).fit(Xtr, ytr)
            hits = m._hits(Xva)
            curve = _alpha_curve(m, hits, yva, grid)
            best = max(curve, key=curve.get)
            rows[name] = {"curve": curve, "best_alpha": best, "best_acc": curve[best]}
            print(f"  seed {seed} {name:<14} best={curve[best]:.4f}@{best} "
                  f"curve={ {k: round(v,4) for k,v in curve.items()} }")
        out[str(seed)] = rows
    report.setdefault("deep_alpha", {}).update(out)
    return out


def section_frontier_deep(report):
    """Validate the top-2 frontier geometries on all 8 seeds at their best alpha,
    against Builder A and the incumbent (best alpha)."""
    print("\n=== attack 6d: 8-seed refinement of frontier winners ===")
    cands = {
        "A_4x4_s2_bin": (GEOM_BUILDER_A, 1e-3),
        "3x5_s2_bin_1e-6": ({"ph": 3, "pw": 5, "inner": 2, "stride": 1, "mode": "binary",
                             "thr": 72.0}, 1e-6),
        "3x5_s2_bin_1e-4": ({"ph": 3, "pw": 5, "inner": 2, "stride": 1, "mode": "binary",
                             "thr": 72.0}, 1e-4),
    }
    compiled = {name: {"raw": [], "inc": []} for name in cands}
    for seed in SEEDS8:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        inc = SparseGeometryNB(GEOM_INCUMBENT, alpha=1.0).fit(Xtr, ytr)
        hi = inc._hits(Xva)
        a_inc = inc.accuracy_at(hi, yva, 1e-4)
        line = f"  seed {seed:>5}: inc1e-4={a_inc:.4f}"
        for name, (geom, alpha) in cands.items():
            m = SparseGeometryNB(geom, alpha=1.0).fit(Xtr, ytr)
            h = m._hits(Xva)
            acc = m.accuracy_at(h, yva, alpha)
            compiled[name]["raw"].append(acc)
            compiled[name]["inc"].append(a_inc)
            line += f" | {name.split('_1e')[0]}@{alpha:g}={acc:.4f}"
        print(line)
    out = {}
    for name, (geom, alpha) in cands.items():
        raws = np.array(compiled[name]["raw"])
        incs = np.array(compiled[name]["inc"])
        out[name] = {"alpha": alpha, "per_seed": raws.tolist(),
                     "mean": float(raws.mean()), "min": float(raws.min()),
                     "mean_delta_vs_inc_best": float((raws - incs).mean()),
                     "positive_all": bool((raws > incs).all())}
    report.setdefault("frontier_deep", {}).update(out)
    for name, row in out.items():
        print(f"  -> {name}: mean={row['mean']:.4f} min={row['min']:.4f} "
              f"delta={row['mean_delta_vs_inc_best']:+.4f} pos_all={row['positive_all']}")
    return out


def section_head2head(report):
    """3x5 vs A head-to-head: is the extra +0.05pt real or noise? Paired McNemar."""
    print("\n=== attack 6e: 3x5 vs 4x4(A) head-to-head ===")
    g35 = {"ph": 3, "pw": 5, "inner": 2, "stride": 1, "mode": "binary", "thr": 72.0}
    out_all = {}
    for alpha in (1e-6, 1e-8):
        pooled_b = pooled_c = 0
        per_seed = {}
        for seed in SEEDS8:
            Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
            m35 = SparseGeometryNB(g35, alpha=1.0).fit(Xtr, ytr)
            ma = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
            p35 = m35.predict_hits(m35._hits(Xva), alpha)
            pa = ma.predict_hits(ma._hits(Xva), alpha)
            mc = V4C.mcnemar(yva, pa, p35)   # A is 'a', 3x5 is 'b'
            pooled_b += mc["b"]; pooled_c += mc["c"]
            per_seed[str(seed)] = {"acc_3x5": float(np.mean(p35 == yva)),
                                   "acc_A": float(np.mean(pa == yva)),
                                   "mcnemar_A_vs_3x5": mc}
        from math import comb
        n = pooled_b + pooled_c
        k = min(pooled_b, pooled_c)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) if n else 1.0
        out_all[f"{alpha:.0e}"] = {"per_seed": per_seed, "pooled_b": pooled_b,
                                   "pooled_c": pooled_c,
                                   "pooled_p": float(min(1.0, 2 * tail))}
        print(f"  alpha={alpha:.0e}: pooled b={pooled_b} c={pooled_c} "
              f"p={min(1.0, 2*tail):.3g}")
    report.setdefault("head2head", {})["3x5_vs_A"] = out_all
    return out_all


def section_collapse(report):
    """Does the tiny-alpha curve eventually turn over / collapse?

    Pushing alpha to 1e-12 approximates the unregularised MAP.  If accuracy
    survives there, the alpha -> 0 regime is genuinely safe for this geometry
    (unseen patch states receive a class-dependent constant that cancels).
    """
    print("\n=== attack 3c: collapse test, alpha down to 1e-12 ===")
    grid = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    out = {}
    for seed in (1322, 7, 2024):
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        m = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
        hits = m._hits(Xva)
        curve = _alpha_curve(m, hits, yva, grid)
        out[str(seed)] = curve
        print(f"  seed {seed}: { {k: round(v,4) for k,v in curve.items()} }")
    report.setdefault("collapse", {}).update(out)
    return out


def section_alpha_mcnemar(report):
    """Is the alpha=1e-3 -> 1e-8 improvement itself statistically real?

    Paired McNemar on A's geometry at two operating points, per seed and pooled.
    """
    print("\n=== attack 3d: McNemar between alpha operating points (A geometry) ===")
    pairs = [("1e-3", 1e-3, "1e-6", 1e-6), ("1e-6", 1e-6, "1e-8", 1e-8),
             ("1e-3", 1e-3, "1e-8", 1e-8)]
    out = {}
    for name, a_lo, b_hi in [(f"{na}_vs_{nb}", av, bv) for na, av, nb, bv in pairs]:
        pooled_b = pooled_c = 0
        per_seed = {}
        for seed in SEEDS8:
            Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
            m = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
            hits = m._hits(Xva)
            p_lo = m.predict_hits(hits, a_lo)   # larger alpha
            p_hi = m.predict_hits(hits, b_hi)   # smaller alpha
            mc = V4C.mcnemar(yva, p_lo, p_hi)   # is p_hi better than p_lo?
            pooled_b += mc["b"]; pooled_c += mc["c"]
            per_seed[str(seed)] = mc
        from math import comb
        n = pooled_b + pooled_c
        k = min(pooled_b, pooled_c)
        tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) if n else 1.0
        out[name] = {"per_seed": per_seed, "pooled_b": pooled_b, "pooled_c": pooled_c,
                     "pooled_p": float(min(1.0, 2 * tail))}
        print(f"  {name}: pooled b={pooled_b} c={pooled_c} p={min(1.0, 2*tail):.3g}")
    report.setdefault("alpha_mcnemar", {}).update(out)
    return out


def section_alpha8(report):
    """Alpha sweep of A's exact geometry on all 8 seeds: pick a defensible alpha."""
    print("\n=== attack 6f: 8-seed alpha curve for A's geometry ===")
    grid = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    table = {f"{a:.0e}": {} for a in grid}
    for seed in SEEDS8:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        m = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
        hits = m._hits(Xva)
        for a in grid:
            table[f"{a:.0e}"][str(seed)] = m.accuracy_at(hits, yva, a)
    summary = {}
    for a in grid:
        vals = np.array(list(table[f"{a:.0e}"].values()))
        summary[f"{a:.0e}"] = {"mean": float(vals.mean()), "min": float(vals.min()),
                               "max": float(vals.max())}
        print(f"  alpha={a:.0e}: mean={vals.mean():.4f} min={vals.min():.4f} "
              f"max={vals.max():.4f}")
    best_a = max(summary, key=lambda k: summary[k]["mean"])
    report.setdefault("alpha8", {})["builder_a_geometry"] = {
        "per_seed_per_alpha": table, "summary": summary,
        "best_mean_alpha": best_a, "best_mean_acc": summary[best_a]["mean"]}
    print(f"  -> best mean alpha = {best_a} (mean {summary[best_a]['mean']:.4f})")
    return summary


def section_nested_ext(report, seeds=(1322, 7, 2024)):
    """Nested alpha selection with the grid extended down to 1e-8."""
    print("\n=== attack 3b: nested alpha selection, extended grid ===")
    grid = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12]
    out = {}
    for seed in seeds:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        inner_tr, inner_va = _stratified_split_inner(ytr, 1 / 6, seed + 1000)
        m_in = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr[inner_tr], ytr[inner_tr])
        h_in = m_in._hits(Xtr[inner_va])
        inner_curve = _alpha_curve(m_in, h_in, ytr[inner_va], grid)
        chosen = max(inner_curve, key=inner_curve.get)
        m_full = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
        h_full = m_full._hits(Xva)
        outer_curve = _alpha_curve(m_full, h_full, yva, grid)
        out[str(seed)] = {"inner_curve": inner_curve, "chosen_alpha": chosen,
                          "outer_acc_at_chosen": outer_curve[chosen],
                          "outer_best_alpha": max(outer_curve, key=outer_curve.get),
                          "outer_best_acc": max(outer_curve.values()),
                          "outer_curve": outer_curve}
        print(f"  seed {seed}: inner picks {chosen} -> outer {outer_curve[chosen]:.4f} "
              f"(outer best {out[str(seed)]['outer_best_acc']:.4f}@"
              f"{out[str(seed)]['outer_best_alpha']})")
    report.setdefault("nested_ext", {}).update(out)
    return out


def section_final8(report):
    """Best-vs-best 8-seed table across the surviving configurations."""
    print("\n=== attack 6g: final 8-seed config comparison ===")
    g35 = {"ph": 3, "pw": 5, "inner": 2, "stride": 1, "mode": "binary", "thr": 72.0}
    g34 = {"ph": 3, "pw": 4, "inner": 2, "stride": 1, "mode": "binary", "thr": 72.0}
    configs = {
        "incumbent_1e-4": (GEOM_INCUMBENT, 1e-4),
        "A_4x4_s2_1e-3": (GEOM_BUILDER_A, 1e-3),
        "A_4x4_s2_1e-6": (GEOM_BUILDER_A, 1e-6),
        "A_4x4_s2_1e-8": (GEOM_BUILDER_A, 1e-8),
        "3x5_s2_1e-6": (g35, 1e-6),
        "3x5_s2_1e-8": (g35, 1e-8),
        "3x4_s2_1e-8": (g34, 1e-8),
    }
    acc = {name: [] for name in configs}
    for seed in SEEDS8:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        for name, (geom, alpha) in configs.items():
            m = SparseGeometryNB(geom, alpha=1.0).fit(Xtr, ytr)
            acc[name].append(m.accuracy_at(m._hits(Xva), yva, alpha))
    out = {}
    for name in configs:
        vals = np.array(acc[name])
        out[name] = {"per_seed": vals.tolist(), "mean": float(vals.mean()),
                     "min": float(vals.min())}
        print(f"  {name:<18} mean={vals.mean():.4f} min={vals.min():.4f}")
    report.setdefault("final8", {}).update(out)
    return out


def section_frontier(report, seeds=(1322,)):
    print("\n=== attack 6: frontier geometry screen ===")
    out = {}
    for seed in seeds:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        rows = {}
        for name, geom in frontier_geometries().items():
            try:
                t0 = time.perf_counter()
                m = SparseGeometryNB(geom, alpha=1.0).fit(Xtr, ytr)
                hits = m._hits(Xva)
                curve = _alpha_curve(m, hits, yva)
                best = max(curve, key=curve.get)
                rows[name] = {
                    "n_features": int(m.n_features), "nstates": int(m.nstates),
                    "n_observed": int(m.n_observed), "best_alpha": best,
                    "best_acc": curve[best], "curve": curve,
                    "seconds": round(time.perf_counter() - t0, 1),
                }
                print(f"  {name:<22} F={m.n_features:<4} S={m.nstates:<10} "
                      f"obs={m.n_observed:>9} best={curve[best]:.4f}@{best} "
                      f"({rows[name]['seconds']}s)")
            except MemoryError as exc:
                rows[name] = {"error": f"MemoryError: {exc}"}
                print(f"  {name:<22} MEMORYERROR")
            except Exception as exc:  # noqa: BLE001
                rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {name:<22} ERROR {type(exc).__name__}: {exc}")
        out[str(seed)] = rows
    merged = report.setdefault("frontier", {})
    for seed in seeds:
        merged[str(seed)] = out[str(seed)]
    return out


def section_geometry_only(report):
    """Third, strictest decomposition: swap ONLY the feature geometry,
    holding alpha fixed at one shared value chosen without looking at val."""
    print("\n=== attack 2b: strict geometry-only sweep at fixed alpha ===")
    Xtr, ytr, Xva, yva = V4C.load_split(seed=1322)
    inc = SparseGeometryNB(GEOM_INCUMBENT, alpha=1.0).fit(Xtr, ytr)
    cand = SparseGeometryNB(GEOM_BUILDER_A, alpha=1.0).fit(Xtr, ytr)
    hi, hc = inc._hits(Xva), cand._hits(Xva)
    out = {}
    for a in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
        out[f"{a:.0e}"] = {"incumbent": inc.accuracy_at(hi, yva, a),
                           "candidate": cand.accuracy_at(hc, yva, a),
                           "delta": cand.accuracy_at(hc, yva, a) - inc.accuracy_at(hi, yva, a)}
        print(f"  alpha={a:.0e}: inc={out[f'{a:.0e}']['incumbent']:.4f} "
              f"cand={out[f'{a:.0e}']['candidate']:.4f} delta={out[f'{a:.0e}']['delta']:+.4f}")
    report.setdefault("geometry_only_fixed_alpha", {})["seed1322"] = out
    return out


# ======================================================================
# best-candidate 8-seed
# ======================================================================
def section_best8seed(report, geom_name, geom, alpha, incumbent_best=1e-4):
    print(f"\n=== 8-seed validation of frontier winner {geom_name} alpha={alpha} ===")
    rows = {}
    for seed in SEEDS8:
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        inc = SparseGeometryNB(GEOM_INCUMBENT, alpha=1.0).fit(Xtr, ytr)
        cand = SparseGeometryNB(geom, alpha=1.0).fit(Xtr, ytr)
        hi, hc = inc._hits(Xva), cand._hits(Xva)
        a_inc = inc.accuracy_at(hi, yva, incumbent_best)
        a_cand = cand.accuracy_at(hc, yva, alpha)
        pi = inc.predict_hits(hi, incumbent_best)
        pc = cand.predict_hits(hc, alpha)
        rows[str(seed)] = {"incumbent_best": a_inc, "candidate": a_cand,
                           "delta": a_cand - a_inc,
                           "mcnemar": V4C.mcnemar(yva, pi, pc)}
        print(f"  seed {seed:>5}: inc={a_inc:.4f} cand={a_cand:.4f} delta={a_cand-a_inc:+.4f} "
              f"p={rows[str(seed)]['mcnemar']['p_value']:.3g}")
    deltas = [rows[s]["delta"] for s in rows]
    summary = {"per_seed": rows, "mean_candidate": float(np.mean([rows[s]["candidate"] for s in rows])),
               "min_candidate": float(np.min([rows[s]["candidate"] for s in rows])),
               "mean_delta": float(np.mean(deltas)), "min_delta": float(np.min(deltas)),
               "positive_on_all_seeds": bool(all(d > 0 for d in deltas))}
    report.setdefault("best8seed", {})
    report["best8seed"][f"{geom_name}_alpha_{alpha:.0e}"] = summary
    print(f"  -> mean={summary['mean_candidate']:.4f} min={summary['min_candidate']:.4f} "
          f"mean_delta={summary['mean_delta']:+.4f} positive_all={summary['positive_on_all_seeds']}")
    return summary


def section_threshold(report):
    """Binarization-threshold robustness on A's geometry, incl. the PDF's own
    named thresholds (>0 and >128) plus the plateau Builder A claimed."""
    print("\n=== attack 6h: threshold sweep on A's geometry ===")
    Xtr, ytr, Xva, yva = V4C.load_split()
    out = {}
    for thr in (0.5, 1.0, 32.0, 48.0, 72.0, 96.0, 128.0, 160.0):
        geom = dict(GEOM_BUILDER_A)
        geom["thr"] = thr
        m = SparseGeometryNB(geom, alpha=1.0).fit(Xtr, ytr)
        hits = m._hits(Xva)
        out[str(thr)] = {"acc_1e-6": m.accuracy_at(hits, yva, 1e-6),
                         "acc_1e-3": m.accuracy_at(hits, yva, 1e-3)}
        print(f"  thr={thr:<6} acc@1e-6={out[str(thr)]['acc_1e-6']:.4f} "
              f"acc@1e-3={out[str(thr)]['acc_1e-3']:.4f}")
    report.setdefault("threshold", {})["builder_a_geometry"] = out
    return out


SECTIONS = {
    "repro": section_repro,
    "threshold": section_threshold,
    "decomposition": section_decomposition,
    "geometry_only": section_geometry_only,
    "alpha_curves": section_alpha_curves,
    "nested": section_nested,
    "cross_seed8": section_cross_seed8,
    "memtime": section_memtime,
    "memtime60": section_memtime60,
    "frontier": section_frontier,
    "frontier2": section_frontier2,
    "frontier_deep": section_frontier_deep,
    "deep_alpha": section_deep_alpha,
    "head2head": section_head2head,
    "alpha8": section_alpha8,
    "nested_ext": section_nested_ext,
    "final8": section_final8,
    "collapse": section_collapse,
    "alpha_mcnemar": section_alpha_mcnemar,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=["repro"])
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    report = {} if args.fresh else load_raw()
    names = list(SECTIONS) if "all" in args.only else args.only
    for name in names:
        if name not in SECTIONS:
            print(f"unknown section {name}")
            continue
        SECTIONS[name](report)
        save_raw(report)
    print(f"\nraw -> {OUT}")


if __name__ == "__main__":
    main()
