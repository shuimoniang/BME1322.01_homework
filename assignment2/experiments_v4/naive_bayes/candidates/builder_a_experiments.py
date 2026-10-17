"""Builder A: FEATURE GEOMETRY / STATE-SPACE sweep.

Run:  python builder_a_experiments.py [round]

Writes raw numbers to results/builder_a_raw.json.  Only the 60,000 training
images are ever loaded (through v4_common.load_split); the test set is never
touched.
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
from builder_a_core import PatchNaiveBayes, config_key, uniform_edges, ink_edges
from naive_bayes import NaiveBayesClassifier

RESULTS = HERE.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
RAW = RESULTS / "builder_a_raw.json"

INCUMBENT_ACC = 0.9298


def load_raw():
    if RAW.exists():
        return json.loads(RAW.read_text(encoding="utf-8"))
    return {}


def _jsonable(value):
    """Make view dicts (which may carry numpy edges) JSON-safe."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save_raw(data):
    RAW.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def incumbent_factory(alpha=0.05):
    return NaiveBayesClassifier(alpha=alpha)


def evaluate(views, merge_min_count=0, alphas=(0.05,), tag=None):
    """Fit PatchNaiveBayes for each alpha on the seed-1322 split; return records."""
    Xtr, ytr, Xva, yva = SPLIT
    key = config_key(views, merge_min_count)
    out = []
    for alpha in alphas:
        start = time.perf_counter()
        model = PatchNaiveBayes(alpha=alpha, views=views, merge_min_count=merge_min_count).fit(Xtr, ytr)
        fit_s = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(Xva)
        pred_s = time.perf_counter() - start
        acc = float(np.mean(pred == yva))
        n_features = sum(block[0] for block in model.blocks_)
        n_states_total = sum(block[0] * block[1] for block in model.blocks_)
        record = {
            "tag": tag or key,
            "config": key,
            "views": views,
            "merge_min_count": merge_min_count,
            "alpha": alpha,
            "val_accuracy": acc,
            "n_features": int(n_features),
            "n_state_slots": int(n_states_total),
            "fit_seconds": round(fit_s, 2),
            "predict_seconds": round(pred_s, 2),
        }
        out.append(record)
        print(f"  {record['tag'][:58]:<58} a={alpha:<5} acc={acc:.4f} "
              f"F={n_features:<5} slots={n_states_total:<9} {fit_s:5.1f}s", flush=True)
    return out


SPLIT = None


# ----------------------------------------------------------------------
# Round A: patch size / shape at comparable state cardinality
# ----------------------------------------------------------------------
def round_a(alphas=(0.05,)):
    print("\n== Round A: patch size / shape (single joint state per patch) ==")
    configs = [
        ("A2x2-L5 (incumbent geometry)", [{"ph": 2, "pw": 2, "levels": 5, "edges": "uniform"}]),
        ("A2x2-L6", [{"ph": 2, "pw": 2, "levels": 6, "edges": "uniform"}]),
        ("A2x2-L7", [{"ph": 2, "pw": 2, "levels": 7, "edges": "uniform"}]),
        ("A3x3-L2 (512 st)", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform"}]),
        ("A3x3-L2-ink", [{"ph": 3, "pw": 3, "levels": 2, "edges": "ink"}]),
        ("A1x2-L5 (25 st)", [{"ph": 1, "pw": 2, "levels": 5, "edges": "uniform"}]),
        ("A1x2-L7 (49 st)", [{"ph": 1, "pw": 2, "levels": 7, "edges": "uniform"}]),
        ("A2x1-L5 (25 st)", [{"ph": 2, "pw": 1, "levels": 5, "edges": "uniform"}]),
        ("A2x3-L3 (729 st)", [{"ph": 2, "pw": 3, "levels": 3, "edges": "uniform"}]),
        ("A3x2-L3 (729 st)", [{"ph": 3, "pw": 2, "levels": 3, "edges": "uniform"}]),
        ("A2x3-L4 (4096 st)", [{"ph": 2, "pw": 3, "levels": 4, "edges": "uniform"}]),
        ("A3x3-L2 inner2 (dilated)", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 2}]),
        ("A2x2-L5 inner2 (dilated)", [{"ph": 2, "pw": 2, "levels": 5, "edges": "uniform", "inner": 2}]),
    ]
    records = []
    for tag, views in configs:
        records.extend(evaluate(views, alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round B: quantization scheme (how the gray levels are placed)
# ----------------------------------------------------------------------
def round_b(alphas=(0.05,)):
    print("\n== Round B: quantization / gray-level placement ==")
    configs = [
        ("B2x2-L5-ink (band0={0})", [{"ph": 2, "pw": 2, "levels": 5, "edges": "ink"}]),
        ("B2x2-L6-ink", [{"ph": 2, "pw": 2, "levels": 6, "edges": "ink"}]),
        ("B2x2-L4-ink", [{"ph": 2, "pw": 2, "levels": 4, "edges": "ink"}]),
        ("B3x3-L2-ink", [{"ph": 3, "pw": 3, "levels": 2, "edges": "ink"}]),
        ("B2x2-L5-posq (per-pixel quantiles)", [{"ph": 2, "pw": 2, "levels": 5, "edges": "posq"}]),
        ("B2x2-L4-posq", [{"ph": 2, "pw": 2, "levels": 4, "edges": "posq"}]),
        ("B2x2-L3-posq", [{"ph": 2, "pw": 2, "levels": 3, "edges": "posq"}]),
        ("B3x3-L2-posq", [{"ph": 3, "pw": 3, "levels": 2, "edges": "posq"}]),
    ]
    records = []
    for tag, views in configs:
        records.extend(evaluate(views, alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round C: stride / coarser patch grids
# ----------------------------------------------------------------------
def round_c(alphas=(0.05,)):
    print("\n== Round C: stride / patch-grid density ==")
    configs = [
        ("C2x2-L5 st1 (incumbent)", [{"ph": 2, "pw": 2, "levels": 5, "edges": "uniform", "stride": 1}]),
        ("C2x2-L5 st2", [{"ph": 2, "pw": 2, "levels": 5, "edges": "uniform", "stride": 2}]),
        ("C3x3-L2 st1", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "stride": 1}]),
        ("C3x3-L2 st2", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "stride": 2}]),
        ("C3x3-L3 st1", [{"ph": 3, "pw": 3, "levels": 3, "edges": "uniform", "stride": 1}]),
        ("C4x4-L1 st1 (binary 65536)", [{"ph": 4, "pw": 4, "levels": 2, "edges": "uniform", "stride": 1}]),
        ("C2x2-L5 st1 + merge1", [{"ph": 2, "pw": 2, "levels": 5, "edges": "uniform"}], 1),
    ]
    records = []
    for entry in configs:
        if len(entry) == 2:
            tag, views = entry
            records.extend(evaluate(views, alphas=alphas, tag=tag))
        else:
            tag, views, merge = entry
            records.extend(evaluate(views, merge_min_count=merge, alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round D: state-space compression (rare-state merging) on the incumbent
# ----------------------------------------------------------------------
def round_d(alphas=(0.05, 0.1)):
    print("\n== Round D: rare-state merging on the incumbent geometry ==")
    base = [{"ph": 2, "pw": 2, "levels": 5, "edges": "uniform"}]
    records = []
    for merge in (0, 1, 2, 3, 5, 8, 16):
        records.extend(evaluate(base, merge_min_count=merge, alphas=alphas,
                                tag=f"D2x2-L5 merge>={merge}"))
    return records


# ----------------------------------------------------------------------
# Round E: heterogeneous / multi-scale feature sets
# ----------------------------------------------------------------------
def round_e(alphas=(0.05,)):
    print("\n== Round E: heterogeneous multi-scale views (stronger independence claim) ==")
    p2u = {"ph": 2, "pw": 2, "levels": 5, "edges": "uniform"}
    configs = [
        ("E2x2L5 + 3x3L2 (multi-scale)", [p2u, {"ph": 3, "pw": 3, "levels": 2, "edges": "uniform"}]),
        ("E2x2L5 + 4x4L2 (multi-scale)", [p2u, {"ph": 4, "pw": 4, "levels": 2, "edges": "uniform", "stride": 3}]),
        ("E2x2L5 + 1x2L5 (orientation)", [p2u, {"ph": 1, "pw": 2, "levels": 5, "edges": "uniform"}]),
        ("E2x2L5 + 2x3L3", [p2u, {"ph": 2, "pw": 3, "levels": 3, "edges": "uniform"}]),
        ("E3x3L2 + 1x2L5", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform"},
                            {"ph": 1, "pw": 2, "levels": 5, "edges": "uniform"}]),
    ]
    records = []
    for tag, views in configs:
        records.extend(evaluate(views, alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round F: factorized patch encoding (pattern x ink level, independent blocks)
# ----------------------------------------------------------------------
def round_f(alphas=(0.05,)):
    print("\n== Round F: factorized patch encoding (pattern || mean) ==")
    configs = [
        ("F3x3 binary || mean3", [{"ph": 3, "pw": 3, "factorize": True, "binary_edges": [0.5], "mean_levels": 3}]),
        ("F3x3 binary || mean4", [{"ph": 3, "pw": 3, "factorize": True, "binary_edges": [0.5], "mean_levels": 4}]),
        ("F3x3 bin2 || mean3", [{"ph": 3, "pw": 3, "factorize": True, "binary_edges": [64.0], "mean_levels": 3}]),
        ("F2x2 binary || mean4", [{"ph": 2, "pw": 2, "factorize": True, "binary_edges": [0.5], "mean_levels": 4}]),
        ("F2x2 binary || mean8", [{"ph": 2, "pw": 2, "factorize": True, "binary_edges": [0.5], "mean_levels": 8}]),
        ("F2x2-L5 || 3x3bin || 3x3mean3",
         [{"ph": 2, "pw": 2, "levels": 5, "edges": "uniform"},
          {"ph": 3, "pw": 3, "factorize": True, "binary_edges": [0.5], "mean_levels": 3}]),
    ]
    records = []
    for tag, views in configs:
        records.extend(evaluate(views, alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round G: dilation / receptive-field scale (the strongest lead from A & C)
# ----------------------------------------------------------------------
def round_g(alphas=(0.05,)):
    print("\n== Round G: receptive-field size (binary p x p with pixel spacing s) ==")
    configs = [
        ("G2x2 bin s1 (extent 2)", [{"ph": 2, "pw": 2, "levels": 2, "edges": "uniform", "inner": 1, "stride": 1}]),
        ("G3x3 bin s1 (extent 3)", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 1, "stride": 1}]),
        ("G3x3 bin s2 (extent 5)", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 2, "stride": 1}]),
        ("G3x3 bin s3 (extent 7)", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 3, "stride": 1}]),
        ("G3x3 bin s4 (extent 9)", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 4, "stride": 1}]),
        ("G3x3 bin s2 st2", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 2, "stride": 2}]),
        ("G3x3 bin s2 st3", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 2, "stride": 3}]),
        ("G3x3 bin s3 st2", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 3, "stride": 2}]),
        ("G4x4 bin s1 (extent 4, 65536)", [{"ph": 4, "pw": 4, "levels": 2, "edges": "uniform", "inner": 1, "stride": 1}]),
        ("G2x2 bin s2 (extent 3)", [{"ph": 2, "pw": 2, "levels": 2, "edges": "uniform", "inner": 2, "stride": 1}]),
        ("G2x2 bin s3 (extent 4)", [{"ph": 2, "pw": 2, "levels": 2, "edges": "uniform", "inner": 3, "stride": 1}]),
        ("G2x2 bin s4 (extent 5)", [{"ph": 2, "pw": 2, "levels": 2, "edges": "uniform", "inner": 4, "stride": 1}]),
        ("G2x2 bin s6 (extent 7)", [{"ph": 2, "pw": 2, "levels": 2, "edges": "uniform", "inner": 6, "stride": 1}]),
    ]
    records = []
    for tag, views in configs:
        records.extend(evaluate(views, alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round H: transposed / asymmetric receptive fields at the winning scale
# ----------------------------------------------------------------------
def round_h(alphas=(0.05,)):
    print("\n== Round H: patch aspect at the winning spatial scale ==")
    configs = [
        ("H3x3 bin s2 st1", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 2, "stride": 1}]),
        ("H3x3 bin s3 st1", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 3, "stride": 1}]),
        ("H3x3 bin s2 st2", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 2, "stride": 2}]),
        ("H3x3 bin s3 st2", [{"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 3, "stride": 2}]),
        ("H2x2 bin s2 st1", [{"ph": 2, "pw": 2, "levels": 2, "edges": "uniform", "inner": 2, "stride": 1}]),
        ("H2x2 bin s2 st2", [{"ph": 2, "pw": 2, "levels": 2, "edges": "uniform", "inner": 2, "stride": 2}]),
    ]
    records = []
    for tag, views in configs:
        records.extend(evaluate(views, alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round I: tune the winning geometry (alpha, binarization threshold)
# ----------------------------------------------------------------------
G3x3_S2 = {"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 2, "stride": 1}


def round_i(alphas=(0.01, 0.02, 0.05, 0.1, 0.2)):
    print("\n== Round I: alpha sweep on G3x3-bin-s2 ==")
    records = evaluate([G3x3_S2], alphas=alphas, tag="I3x3bins2 alpha-sweep")
    print("\n-- Round I2: binarization threshold (3x3, s=2) --")
    thresholds = [
        ("thr0.5 (pixel>=1)", 0.5),
        ("thr16", 16.0),
        ("thr32", 32.0),
        ("thr64", 64.0),
        ("thr96", 96.0),
        ("thr128 (incumbent)", 127.5),
        ("thr160", 160.0),
    ]
    for tag, thr in thresholds:
        view = dict(G3x3_S2, edges=np.array([thr], dtype=np.float64))
        records.extend(evaluate([view], alphas=(0.05,), tag=f"I3x3bins2 {tag}"))
    return records


# ----------------------------------------------------------------------
# Round J: heterogeneous combinations of the winning geometry
# ----------------------------------------------------------------------
def round_j(alphas=(0.05,)):
    print("\n== Round J: multi-scale combinations with the winner ==")
    p2u = {"ph": 2, "pw": 2, "levels": 5, "edges": "uniform"}
    g1 = {"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 1, "stride": 1}
    g2 = dict(G3x3_S2)
    g3 = {"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 3, "stride": 1}
    g4s1 = {"ph": 4, "pw": 4, "levels": 2, "edges": "uniform", "inner": 1, "stride": 1}
    h2 = {"ph": 2, "pw": 2, "levels": 2, "edges": "uniform", "inner": 2, "stride": 1}
    v2 = {"ph": 2, "pw": 2, "levels": 5, "edges": "uniform", "inner": 2, "stride": 1}
    configs = [
        ("J 3x3s2 + 2x2L5 (multi-scale)", [g2, p2u]),
        ("J 3x3s2 + 3x3s1", [g2, g1]),
        ("J 3x3s2 + 3x3s3", [g2, g3]),
        ("J 3x3s2 + 4x4s1", [g2, g4s1]),
        ("J 3x3s2 + 2x2bins2", [g2, h2]),
        ("J 3x3s2 + 2x2L5-inners2", [g2, v2]),
        ("J 3x3s2 + 4x2bins2", [g2, {"ph": 4, "pw": 2, "levels": 2, "edges": "uniform", "inner": 2, "stride": 1}]),
        ("J 4x4s2 + 2x2L5", [{"ph": 4, "pw": 4, "levels": 2, "edges": "uniform", "inner": 2, "stride": 1}, p2u]),
    ]
    records = []
    for tag, views in configs:
        records.extend(evaluate(views, alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round K: state-space compression on the winning geometry
# ----------------------------------------------------------------------
def round_k(alphas=(0.05,)):
    print("\n== Round K: rare-state merging on G3x3-bin-s2 (and stride-3 variant) ==")
    records = []
    for merge in (0, 1, 2, 4, 8, 16, 32):
        records.extend(evaluate([G3x3_S2], merge_min_count=merge, alphas=alphas,
                                tag=f"K3x3bins2 merge>={merge}"))
    g2s3 = {"ph": 3, "pw": 3, "levels": 2, "edges": "uniform", "inner": 2, "stride": 3}
    for merge in (0, 2, 8):
        records.extend(evaluate([g2s3], merge_min_count=merge, alphas=alphas,
                                tag=f"K3x3bins2st3 merge>={merge}"))
    return records


# ----------------------------------------------------------------------
# Round L: joint refinement of the binarization threshold and alpha
# ----------------------------------------------------------------------
def round_l(alphas=(0.02, 0.05, 0.1)):
    print("\n== Round L: threshold x alpha refinement on 3x3, s=2 ==")
    records = []
    for thr in (40.0, 48.0, 56.0, 64.0, 72.0, 80.0, 88.0):
        view = dict(G3x3_S2, edges=np.array([thr], dtype=np.float64))
        records.extend(evaluate([view], alphas=alphas, tag=f"L3x3s2 thr{thr:.0f}"))
    return records


# ----------------------------------------------------------------------
# Round M: does the winning insight transfer to other geometries?
# ----------------------------------------------------------------------
def round_m(alphas=(0.05,)):
    print("\n== Round M: dilated-binary across shapes (spacing 2) ==")
    records = []
    configs = [
        ("M2x2bins2", {"ph": 2, "pw": 2, "levels": 2, "inner": 2, "stride": 1}),
        ("M3x3bins2", {"ph": 3, "pw": 3, "levels": 2, "inner": 2, "stride": 1}),
        ("M4x4bins2", {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 1}),
        ("M2x2bins3", {"ph": 2, "pw": 2, "levels": 2, "inner": 3, "stride": 1}),
        ("M2x4bins2", {"ph": 2, "pw": 4, "levels": 2, "inner": 2, "stride": 1}),
        ("M4x2bins2", {"ph": 4, "pw": 2, "levels": 2, "inner": 2, "stride": 1}),
        ("M3x3bins2-ho", {"ph": 3, "pw": 3, "levels": 2, "inner": 2, "stride": 1,
                          "edges": np.array([64.0])}),
    ]
    for tag, view in configs:
        records.extend(evaluate([view], alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round N: refine 4x4 binary at spacing 2 (the best single view so far)
# ----------------------------------------------------------------------
def round_n(alphas=(0.02, 0.05, 0.1)):
    print("\n== Round N: threshold x alpha on 4x4 binary, spacing 2 ==")
    records = []
    for thr in (48.0, 56.0, 64.0, 72.0, 80.0):
        view = {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 1,
                "edges": np.array([thr], dtype=np.float64)}
        records.extend(evaluate([view], alphas=alphas, tag=f"N4x4s2 thr{thr:.0f}"))
    return records


# ----------------------------------------------------------------------
# Round O: spacing / stride variants around the extent-7 winner
# ----------------------------------------------------------------------
def round_o(alphas=(0.05,)):
    print("\n== Round O: receptive-field variants near the winner ==")
    records = []
    configs = [
        ("O4x4s2 st1 (extent 7)", {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 1}),
        ("O4x4s2 st2", {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 2}),
        ("O4x4s2 st3", {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 3}),
        ("O3x3s3 st1 (extent 7)", {"ph": 3, "pw": 3, "levels": 2, "inner": 3, "stride": 1}),
        ("O4x4s1 st1 (extent 4)", {"ph": 4, "pw": 4, "levels": 2, "inner": 1, "stride": 1}),
        ("O3x4s2 st1", {"ph": 3, "pw": 4, "levels": 2, "inner": 2, "stride": 1}),
        ("O4x3s2 st1", {"ph": 4, "pw": 3, "levels": 2, "inner": 2, "stride": 1}),
        ("O4x4s2 st1 thr72", {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 1,
                              "edges": np.array([72.0])}),
        ("O4x4s2 st2 thr72", {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 2,
                              "edges": np.array([72.0])}),
        ("O3x3s2 st1 (extent 5) thr72", {"ph": 3, "pw": 3, "levels": 2, "inner": 2, "stride": 1,
                                         "edges": np.array([72.0])}),
    ]
    for tag, view in configs:
        records.extend(evaluate([view], alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round P: larger receptive fields via exact sparse CPT storage
# ----------------------------------------------------------------------
def round_p(alphas=(0.02, 0.05)):
    print("\n== Round P: large receptive fields (sparse CPT) ==")
    records = []
    configs = [
        ("P4x4s2 thr72 (2^16)", {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 1,
                                 "edges": np.array([72.0])}),
        ("P5x5s2 thr72 (2^25)", {"ph": 5, "pw": 5, "levels": 2, "inner": 2, "stride": 1,
                                 "edges": np.array([72.0])}),
        ("P4x4s2 st2 thr72", {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 2,
                              "edges": np.array([72.0])}),
        ("P5x5s2 st2 thr72", {"ph": 5, "pw": 5, "levels": 2, "inner": 2, "stride": 2,
                              "edges": np.array([72.0])}),
        ("P3x3s2 thr72 (2^9, ref)", {"ph": 3, "pw": 3, "levels": 2, "inner": 2, "stride": 1,
                                     "edges": np.array([72.0])}),
    ]
    for tag, view in configs:
        records.extend(evaluate([view], alphas=alphas, tag=tag))
    return records


# ----------------------------------------------------------------------
# Round Q: final refinement around 4x4 binary, spacing 2, thr~72
# ----------------------------------------------------------------------
WIN = {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 1, "edges": np.array([72.0])}


def round_q(alphas=(0.005, 0.01, 0.02, 0.03, 0.04)):
    records = evaluate([WIN], alphas=alphas, tag="Q4x4s2thr72 alpha")
    for tag, levels in (("inkq-L2", 2), ("inkq-L3", 3), ("inkq-L4", 4)):
        view = {"ph": 4, "pw": 4, "levels": levels, "inner": 2, "stride": 1, "edges": "inkq"}
        records.extend(evaluate([view], alphas=(0.02,), tag=f"Q4x4s2 {tag}"))
    print("\n== Round Q3: aspect ratio at extent 7/8 ==")
    aspects = [
        ("Q4x3s2", {"ph": 4, "pw": 3, "levels": 2, "inner": 2, "stride": 1}),
        ("Q3x4s2", {"ph": 3, "pw": 4, "levels": 2, "inner": 2, "stride": 1}),
        ("Q5x3s2", {"ph": 5, "pw": 3, "levels": 2, "inner": 2, "stride": 1}),
        ("Q3x5s2", {"ph": 3, "pw": 5, "levels": 2, "inner": 2, "stride": 1}),
        ("Q4x5s2", {"ph": 4, "pw": 5, "levels": 2, "inner": 2, "stride": 1}),
        ("Q5x4s2", {"ph": 5, "pw": 4, "levels": 2, "inner": 2, "stride": 1}),
    ]
    for tag, view in aspects:
        records.extend(evaluate([view], alphas=(0.02,), tag=tag))
    print("\n== Round Q4: rare-state merging on the winner ==")
    for merge in (2, 4, 8, 16, 32):
        records.extend(evaluate([WIN], merge_min_count=merge, alphas=(0.02,),
                                tag=f"Q4x4s2thr72 merge>={merge}"))
    print("\n== Round Q5: multi-scale combinations with the winner ==")
    combos = [
        ("Q 4x4s2 + 4x4s1", [WIN, {"ph": 4, "pw": 4, "levels": 2, "inner": 1, "stride": 1}]),
        ("Q 4x4s2 + 3x3s2thr72", [WIN, {"ph": 3, "pw": 3, "levels": 2, "inner": 2,
                                        "stride": 1, "edges": np.array([72.0])}]),
        ("Q 4x4s2 + 2x2L5", [WIN, {"ph": 2, "pw": 2, "levels": 5, "edges": "uniform"}]),
        ("Q 4x4s2 + 3x3s3", [WIN, {"ph": 3, "pw": 3, "levels": 2, "inner": 3, "stride": 1}]),
    ]
    for tag, views in combos:
        records.extend(evaluate(views, alphas=(0.02,), tag=tag))
    return records


# ----------------------------------------------------------------------
# Round R: fine alpha sweep at the frozen winning geometry
# ----------------------------------------------------------------------
def round_r():
    print("\n== Round R: fine alpha sweep, 4x4s2 thr72 ==")
    alphas = (0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05)
    return evaluate([WIN], alphas=alphas, tag="R4x4s2thr72 alpha-fine")


# ----------------------------------------------------------------------
# Round S: robustness of the frozen winner (threshold, alpha extremes,
#          anisotropic spacing, shape at the winning receptive field)
# ----------------------------------------------------------------------
def round_s():
    print("\n== Round S1: binarization threshold, 4x4 s=2, alpha=0.02 ==")
    records = []
    for thr in (0.5, 16.0, 32.0, 64.0, 72.0, 96.0, 127.5, 160.0, 192.0):
        view = {"ph": 4, "pw": 4, "levels": 2, "inner": 2, "stride": 1,
                "edges": np.array([thr], dtype=np.float64)}
        records.extend(evaluate([view], alphas=(0.02,), tag=f"S4x4s2 thr{thr:.0f}"))
    print("\n== Round S2: very small alpha on the frozen winner ==")
    for alpha in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        records.extend(evaluate([WIN], alphas=(alpha,), tag="S4x4s2thr72 alpha-small"))
    print("\n== Round S3: anisotropic and shape variants at the winning scale ==")
    shapes = [
        ("S4x4 (2,1)", {"ph": 4, "pw": 4, "levels": 2, "inner": (2, 1), "stride": 1}),
        ("S4x4 (1,2)", {"ph": 4, "pw": 4, "levels": 2, "inner": (1, 2), "stride": 1}),
        ("S4x4 (2,3)", {"ph": 4, "pw": 4, "levels": 2, "inner": (2, 3), "stride": 1}),
        ("S4x4 (3,2)", {"ph": 4, "pw": 4, "levels": 2, "inner": (3, 2), "stride": 1}),
        ("S4x5 s2", {"ph": 4, "pw": 5, "levels": 2, "inner": 2, "stride": 1}),
        ("S5x4 s2", {"ph": 5, "pw": 4, "levels": 2, "inner": 2, "stride": 1}),
    ]
    for tag, view in shapes:
        view.setdefault("edges", np.array([72.0], dtype=np.float64))
        records.extend(evaluate([view], alphas=(0.02,), tag=tag))
    return records


# ----------------------------------------------------------------------
# Round T: the "strongest smoothing-tuned incumbent" frontier
# ----------------------------------------------------------------------
def round_t():
    print("\n== Round T: incumbent lower-alpha frontier (smoothing control) ==")
    Xtr, ytr, Xva, yva = SPLIT
    records = []
    for alpha in (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 1e-1):
        model = NaiveBayesClassifier(alpha=alpha).fit(Xtr, ytr)
        acc = float(np.mean(model.predict(Xva) == yva))
        records.append({"tag": "Tincumbent alpha-frontier", "config": "incumbent 2x2-L5",
                        "alpha": alpha, "val_accuracy": acc})
        print(f"  incumbent alpha={alpha:<8} acc={acc:.4f}", flush=True)
    return records


# ----------------------------------------------------------------------
# Round U: tiny-alpha sweep on the winning geometry, across all three seeds
# ----------------------------------------------------------------------
def round_u():
    print("\n== Round U: tiny-alpha x seed grid, 4x4 s2 thr72 ==")
    records = []
    for seed in (1322, 7, 2024):
        Xtr, ytr, Xva, yva = V4C.load_split(seed=seed)
        for alpha in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            model = PatchNaiveBayes(alpha=alpha, views=[WIN]).fit(Xtr, ytr)
            acc = float(np.mean(model.predict(Xva) == yva))
            records.append({"tag": "U4x4s2thr72 alpha x seed", "config": config_key([WIN]),
                            "seed": seed, "alpha": alpha, "val_accuracy": acc})
            print(f"  seed {seed:<5} alpha={alpha:<8} acc={acc:.4f}", flush=True)
    return records


ROUNDS = {"a": round_a, "b": round_b, "c": round_c, "d": round_d, "e": round_e,
          "f": round_f, "g": round_g, "h": round_h, "i": round_i, "j": round_j,
          "k": round_k, "l": round_l, "m": round_m, "n": round_n, "o": round_o,
          "p": round_p, "q": round_q, "r": round_r, "s": round_s, "t": round_t,
          "u": round_u}


def main():
    global SPLIT
    rounds = sys.argv[1:] or ["a"]
    SPLIT = V4C.load_split()
    Xtr, ytr, Xva, yva = SPLIT
    print(f"protocol: train={Xtr.shape[0]} val={Xva.shape[0]}")
    store = load_raw()
    if "protocol" not in store:
        store["protocol"] = {"seed": 1322, "validation_fraction": 1 / 6,
                             "n_train": int(Xtr.shape[0]), "n_val": int(Xva.shape[0])}
        store["incumbent_reference"] = {
            "description": "naive_bayes.NaiveBayesClassifier, 729 2x2 patches, 5 levels, alpha=0.05",
            "val_accuracy": INCUMBENT_ACC,
        }
        start = time.perf_counter()
        pred = incumbent_factory().fit(Xtr, ytr).predict(Xva)
        store["incumbent_reference"]["reproduced_val_accuracy"] = float(np.mean(pred == yva))
        store["incumbent_reference"]["reproduce_seconds"] = round(time.perf_counter() - start, 2)
        np.save(RESULTS / "builder_a_incumbent_pred.npy", np.asarray(pred))
    for name in rounds:
        records = ROUNDS[name]()
        key = f"round_{name}"
        store.setdefault(key, [])
        store[key].extend(records)
        save_raw(store)
    print(f"\nraw -> {RAW}")


if __name__ == "__main__":
    main()
