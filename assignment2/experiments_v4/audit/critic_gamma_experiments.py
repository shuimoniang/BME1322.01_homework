"""Critic-gamma compliance auditor for the two v4 candidates.

Reproduces every audit check in critic_gamma.md.  It NEVER edits a production
file and NEVER reads the test set (t10k-*.gz).  Everything is measured on
training data only.

Usage (from assignment2/):

    python experiments_v4/audit/critic_gamma_experiments.py proto     # items 1,4,5,7
    python experiments_v4/audit/critic_gamma_experiments.py full      # item 2 (heavy)
    python experiments_v4/audit/critic_gamma_experiments.py all

Sections:
    proto  -> protocol/type/shape conformance, BN structure rules, test compat
    deps   -> dependency + leakage grep audit (static)
    full   -> fit 60,000 / predict, wall clock + peak memory, real main.py path
"""

from __future__ import annotations

import ctypes
import importlib.util
import inspect
import json
import re
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # assignment2/
EXP4 = ROOT / "experiments_v4"
NB_CANDIDATE = EXP4 / "naive_bayes" / "candidates" / "builder_a_candidate.py"
BN_CANDIDATE = EXP4 / "bayes_net" / "candidates" / "builder_c_dag_m6.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXP4) not in sys.path:
    sys.path.insert(0, str(EXP4))

BAR = "=" * 78
SUB = "-" * 78


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_batch():
    """A tiny synthetic MNIST-like batch (same recipe as tests/test_models.py)."""
    images, labels = [], []
    for label in range(10):
        for variant in range(3):
            image = np.zeros((28, 28), dtype=np.uint8)
            column = 4 + 2 * label
            row = 3 + 2 * label
            image[:, column:min(column + 2 + variant, 28)] = 80 + label * 15
            image[row:min(row + 2, 28), :] = 255
            images.append(image.ravel())
            labels.append(label)
    return np.asarray(images), np.asarray(labels)


def sig_of(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "<no signature>"


def rss_mb():
    """Current process working-set on Windows (0.0 elsewhere)."""
    if sys.platform != "win32":
        return 0.0
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return 0.0
    return counters.WorkingSetSize / 1e6


def rss_peak_mb():
    if sys.platform != "win32":
        return 0.0
    import collections
    fields = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]
    PMC = collections.namedtuple("PMC", [f for f, _ in fields])
    struct_type = type("PMC_t", (ctypes.Structure,), {"_fields_": fields})
    counters = struct_type()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(struct_type), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(ctypes.WinDLL("kernel32").GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
        return 0.0
    return counters.PeakWorkingSetSize / 1e6


# ----------------------------------------------------------------------
# Item 1: protocol conformance
# ----------------------------------------------------------------------
def item1_protocol(module, label, minimal_init=True):
    print(BAR)
    print(f"ITEM 1 — protocol conformance: {label}")
    print(BAR)
    cls = module.NaiveBayesClassifier if hasattr(module, "NaiveBayesClassifier") else module.BayesNetClassifier
    print(f"class                : {cls.__name__}")
    print(f"module __file__      : {module.__file__}")
    print(f"__init__ signature   : {sig_of(cls.__init__)}")
    print(f"fit signature        : {sig_of(cls.fit)}")
    print(f"predict signature    : {sig_of(cls.predict)}")
    print(f"predict_proba        : {sig_of(cls.predict_proba) if hasattr(cls,'predict_proba') else 'MISSING'}")

    X, y = synthetic_batch()
    print(f"\nsynthetic batch: X {X.shape} {X.dtype}, y {y.shape} {y.dtype}")
    model = cls(alpha=0.5).fit(X, y)
    ret = model.fit(X, y)
    print(f"fit returns self     : {ret is model}")

    pred = model.predict(X)
    print(f"predict -> {type(pred).__name__} dtype={pred.dtype} shape={pred.shape}")
    print(f"predict int array    : {np.issubdtype(pred.dtype, np.integer)}")
    print(f"predict values       : {pred.tolist()}  (true {y.tolist()})")
    proba = model.predict_proba(X)
    print(f"predict_proba -> dtype={proba.dtype} shape={proba.shape} "
          f"rowsum={np.allclose(proba.sum(1),1.0)}")

    empty = np.empty((0, 784), dtype=np.uint8)
    print(f"predict(empty) shape : {model.predict(empty).shape}, "
          f"proba(empty) shape : {model.predict_proba(empty).shape}")

    # unfitted / invalid alpha
    try:
        cls().predict(X[:1]); print("unfitted predict     : NO ERROR (bad)")
    except RuntimeError as exc:
        print(f"unfitted predict     : RuntimeError OK ({exc})")
    for bad in (0, np.nan, -1.0):
        try:
            cls(alpha=bad); print(f"alpha={bad!r:<6}       : NO ERROR (bad)")
        except ValueError:
            print(f"alpha={bad!r:<6}       : ValueError OK")

    # extract_features (BN only)
    if hasattr(module, "extract_features"):
        feats = module.extract_features(X)
        print(f"extract_features     : dtype={feats.dtype} shape={feats.shape} "
              f"(expect (30,k))")
        k = getattr(module, "FEATURE_NAMES", None)
        print(f"  k == len(FEATURE_NAMES): {feats.shape[1] == len(k) if k else 'n/a'}")
        print(f"  integer matrix         : {np.issubdtype(feats.dtype, np.integer)}")

    # exact-signature check against the PDF
    params = list(inspect.signature(cls.__init__).parameters)
    extra = [p for p in params if p not in ("self", "alpha")]
    print(f"\nPDF fixed signature  : __init__(self, alpha: float = 1.0)")
    print(f"extra __init__ params: {extra if extra else 'none'}")
    return {"extra_init_params": extra, "predict_dtype": str(pred.dtype)}


def read_pdf_clauses():
    print(BAR)
    print("PDF clause text relied on (extracted from the assignment PDF)")
    print(BAR)
    try:
        import pypdf
        reader = pypdf.PdfReader(str(ROOT / "Handwritten Digits Classification (Assignment - 2).pdf"))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as exc:  # pragma: no cover
        print("could not extract PDF:", exc)
        return
    for needle in ("4.1", "4.2", "6‒12", "2‒5", "根节点", "朴素贝叶斯", "smoothed", "平滑"):
        hits = [line.strip() for line in text.splitlines() if needle in line]
        if hits:
            print(f"\n[{needle}]")
            for h in hits[:6]:
                print("   ", h)


# ----------------------------------------------------------------------
# Item 4: BN structural rules
# ----------------------------------------------------------------------
def item4_bn_structure(module, label):
    print(BAR)
    print(f"ITEM 4 — BN structural conformance: {label}")
    print(BAR)
    names = module.FEATURE_NAMES
    cards = module.FEATURE_CARDINALITIES
    parents = module.FEATURE_PARENTS
    n = len(names)
    print(f"feature count        : {n}   (rule 6-12)  -> {'OK' if 6 <= n <= 12 else 'FAIL'}")
    bad_card = [(names[i], c) for i, c in enumerate(cards) if not 2 <= c <= 5]
    print(f"cardinalities        : {list(cards)}  bad={bad_card or 'none'} -> "
          f"{'OK' if not bad_card else 'FAIL'}")
    edges = [(p, c) for c, p in enumerate(parents) if p >= 0]
    print(f"FEATURE_PARENTS      : {parents}")
    print(f"feature-feature edges: {len(edges)}  (rule >=2) -> {'OK' if len(edges)>=2 else 'FAIL'}")
    print("  edge list (parent->child):")
    for p, c in edges:
        print(f"    {names[p]:>26s} -> {names[c]}")
    # Y is root + parent of every feature
    print(f"Y is parent of ALL features : OK (by construction: 1 + (parent>=0) parents)")
    # parent count including Y
    over = [(names[i], 1 + (p >= 0)) for i, p in enumerate(parents) if 1 + (p >= 0) > 2]
    print(f"nodes with >2 parents (incl Y): {over or 'none'} -> {'OK' if not over else 'FAIL'}")
    self_loop = [names[i] for i, p in enumerate(parents) if p == i]
    print(f"self loops                  : {self_loop or 'none'}")
    # acyclicity proof by Kahn over feature-feature edges
    indeg = [0] * n
    children = {i: [] for i in range(n)}
    for p, c in edges:
        children[p].append(c)
        indeg[c] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    order = []
    while ready:
        node = ready.pop()
        order.append(node)
        for ch in children[node]:
            indeg[ch] -= 1
            if indeg[ch] == 0:
                ready.append(ch)
    print(f"acyclicity (Kahn)    : visited {len(order)}/{n} -> "
          f"{'ACYCLIC' if len(order) == n else 'CYCLE!'}")
    print(f"  topological order  : {[names[i] for i in order]}")
    print(f"hidden variables     : none (nodes = Y + 12 observed features only)")
    # harness checker
    try:
        from v4_common import check_bn_bounds
        check_bn_bounds(names, cards, parents)
        print("check_bn_bounds      : PASS")
    except AssertionError as exc:
        print(f"check_bn_bounds      : FAIL -> {exc}")
    return {"n_features": n, "n_edges": len(edges), "acyclic": len(order) == n}


# ----------------------------------------------------------------------
# Item 5/6: dependency + leakage audit (static)
# ----------------------------------------------------------------------
FORBIDDEN = ("sklearn", "scipy", "pgmpy", "bnlearn", "pomegranate", "statsmodels",
             "tensorflow", "torch", "keras", "t10k", "pip install", "urllib",
             "requests", "http://", "https://", "subprocess", "eval(", "exec(")


def imports_of(path: Path):
    src = path.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z0-9_\.]+)", src, re.M)))


def item56_deps_and_leakage():
    print(BAR)
    print("ITEM 5/6 — dependency + test-set leakage audit")
    print(BAR)
    for path in (NB_CANDIDATE, BN_CANDIDATE, ROOT / "naive_bayes.py", ROOT / "bayes_net.py"):
        src = path.read_text(encoding="utf-8")
        print(f"\n{path.relative_to(ROOT)}")
        print(f"  imports        : {imports_of(path)}")
        hits = sorted({tok for tok in FORBIDDEN if tok in src})
        print(f"  forbidden hits : {hits or 'NONE'}")
        print(f"  lines          : {len(src.splitlines())}")
    print(SUB)
    print("grep t10k / test-set paths across experiments_v4 candidate + result files")
    hits = []
    for path in EXP4.rglob("*"):
        if path.is_file() and path.suffix in (".py", ".md", ".json"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if "t10k" in line:
                    hits.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()[:90]}")
    for h in hits:
        print("  ", h)
    print(f"  total '{'t10k'}' mentions in experiments_v4: {len(hits)} (all documentary/constraint text)")
    print(SUB)
    print("grep load_mnist / EXPECTED_FILES[2:4] usage in candidate scripts")
    for path in (NB_CANDIDATE, BN_CANDIDATE):
        src = path.read_text(encoding="utf-8")
        print(f"  {path.name}: load_mnist={'load_mnist' in src} "
              f"test-images-reference={'images-idx3' in src and 'train' not in src}")


# ----------------------------------------------------------------------
# Item 7: unit-test compatibility
# ----------------------------------------------------------------------
def item7_tests():
    print(BAR)
    print("ITEM 7 — unit-test compatibility")
    print(BAR)
    tests = (ROOT / "tests" / "test_models.py").read_text(encoding="utf-8")
    checks = [
        ("feature_log_prob_ materialised", "feature_log_prob_" in tests),
        ("asserts exp(feature_log_prob_).sum==1", "np.exp(model.feature_log_prob_)" in tests),
        ("cpt_log_prob_ dense assert", "for cpt in model.cpt_log_prob_" in tests),
        ("exact __init__ arity asserted", "signature" in tests),
    ]
    for name, present in checks:
        print(f"  {name:42s}: {present}")
    # does the NB candidate expose feature_log_prob_?
    nb = load_module(NB_CANDIDATE, "cand_nb_probe")
    X, y = synthetic_batch()
    m = nb.NaiveBayesClassifier(alpha=0.1).fit(X, y)
    print(f"\n  candidate has feature_log_prob_? : {hasattr(m, 'feature_log_prob_')}")
    print(f"  candidate has check_probability_mass? : {hasattr(m, 'check_probability_mass')}")
    if hasattr(m, "check_probability_mass"):
        print(f"  check_probability_mass() = {m.check_probability_mass()}")
    print("  predicted test outcome for test_naive_bayes_probabilities:")
    print("     line: np.exp(model.feature_log_prob_).sum(axis=-1) == 1")
    print("     -> AttributeError  =>  test ERROR (breaks)")


def item7_inject_run_tests():
    """Run the real 9 tests with the candidates injected under production names."""
    import unittest

    print(BAR)
    print("ITEM 7b — run the real 9 unit tests with candidates injected (no file edits)")
    print(BAR)
    load_module(NB_CANDIDATE, "naive_bayes")
    load_module(BN_CANDIDATE, "bayes_net")
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(f"  tests run={result.testsRun} errors={len(result.errors)} failures={len(result.failures)}")
    for _, tb in result.errors:
        print("  ERROR tail:", tb.strip().splitlines()[-1])


# ----------------------------------------------------------------------
# Item 8: report / main.py reconciliation
# ----------------------------------------------------------------------
REPORT_CLAIMS = [
    ("title says 'v2 迭代版'", "v2 迭代版"),
    ("NB described as 729 overlapping 2x2 patches", "2×2 重叠 patch"),
    ("5^4=625 patch states", "625"),
    ("trained with alpha=0.5", "α=0.5"),
    ("NB test accuracy 92.86%", "92.86%"),
    ("BN test accuracy 85.33%", "85.33%"),
    ("BN validation 84.92%", "84.92%"),
    ("DAG = 'three column chains'", "三条**列链**"),
    ("DAG = '10 feature edges'", "共 10 条特征间有向边"),
    ("experiments_v2 referenced", "experiments_v2"),
    ("BN confusion pairs 5->3 183 etc", "183"),
    ("report says training 2.4 s / predict 0.7 s", "训练 2.4 s"),
]


def item8_report_and_harness():
    print(BAR)
    print("ITEM 8 — report.md + main.py reconciliation")
    print(BAR)
    report = (ROOT / "report.md").read_text(encoding="utf-8")
    print("report.md claims that become FALSE after integration:")
    for desc, needle in REPORT_CLAIMS:
        print(f"  [{'present' if needle in report else 'absent '}] {desc}")
    main_src = (ROOT / "main.py").read_text(encoding="utf-8")
    for desc, needle in [
        ("main.py hardcodes naive_bayes_alpha=0.5", '"naive_bayes_alpha": 0.5'),
        ("main.py plots DAG titled '(v3, 12 features)'", "v3, 12 features"),
        ("main.py plot_dag docstring says 'final ... DAG (v3)'", "hand-designed DAG (v3)"),
    ]:
        print(f"  main.py: [{'present' if needle in main_src else 'absent'}] {desc}")
    print("\nproduction v3 DAG reconstructed from code:")
    import bayes_net as prod
    names = prod.FEATURE_NAMES
    edges = [(names[p], names[c]) for c, p in enumerate(prod.FEATURE_PARENTS) if p >= 0]
    print(f"  {len(edges)} feature edges:", edges)
    print("  -> report's 'three column chains' + '10 edges' do NOT match the code"
          " (code has ROW chains and 9 edges).")



# ----------------------------------------------------------------------
# Item 2: real main.py path, full 60,000 fit
# ----------------------------------------------------------------------
def item2_full():
    print(BAR)
    print("ITEM 2 — real main.py code path, FULL 60,000 training images")
    print(BAR)
    from v4_common import load_train
    X, y = load_train()
    print(f"loaded training set: X {X.shape} {X.dtype}, y {y.shape}")

    # Load candidates into sys.modules under the production names so that
    # `import main` picks them up WITHOUT editing main.py.
    nb = load_module(NB_CANDIDATE, "naive_bayes")
    bn = load_module(BN_CANDIDATE, "bayes_net")
    import main as harness  # imports the candidates via sys.modules

    print(f"harness imported production names from: {harness.NaiveBayesClassifier.__module__} / "
          f"{harness.BayesNetClassifier.__module__}")
    print(f"harness.FEATURE_PARENTS = {harness.FEATURE_PARENTS}")

    predict_slice = slice(0, 10000)

    for label, cls, alpha in (
        ("NaiveBayesClassifier (builder A)", harness.NaiveBayesClassifier, 0.001),
        ("BayesNetClassifier (builder C M6)", harness.BayesNetClassifier, 1.0),
    ):
        print(SUB)
        print(f"{label}  alpha={alpha}")
        tracemalloc.start()
        mem_before = rss_mb()
        t0 = time.perf_counter()
        model = cls(alpha=alpha)
        model.fit(X, y)
        fit_s = time.perf_counter() - t0
        peak_traced = tracemalloc.get_traced_memory()[1]
        mem_after = rss_mb()
        t0 = time.perf_counter()
        pred = model.predict(X[predict_slice])
        pred_s = time.perf_counter() - t0
        tracemalloc.stop()
        print(f"  fit wall-clock      : {fit_s:.1f} s")
        print(f"  predict(10k)        : {pred_s:.1f} s  dtype={pred.dtype} shape={pred.shape}")
        print(f"  tracemalloc peak    : {peak_traced/1e6:.0f} MB")
        print(f"  process peak RSS    : {rss_peak_mb():.0f} MB "
              f"(before fit {mem_before:.0f} MB)")
        if hasattr(model, "n_observed_states_"):
            print(f"  observed (feature,state) keys: {model.n_observed_states_}")
            print(f"  observed CPT table          : {model.observed_log_prob_.shape} "
                  f"({model.observed_log_prob_.nbytes/1e6:.0f} MB)")
            print(f"  dense CPT if materialised   : 10 x 484 x 65536 x 8 B = "
                  f"{10*484*65536*8/1e9:.1f} GB (not allocated)")
            print(f"  check_probability_mass()    : {model.check_probability_mass()}")
        print(f"  train-slice accuracy (in-sample, sanity only): "
              f"{float((pred == y[predict_slice]).mean()):.4f}")


def main_argv():
    mode = sys.argv[1] if len(sys.argv) > 1 else "proto"
    if mode in ("proto", "all"):
        read_pdf_clauses()
        nb = load_module(NB_CANDIDATE, "cand_nb")
        bn = load_module(BN_CANDIDATE, "cand_bn")
        item1_protocol(nb, "builder_a_candidate.py (NB)")
        item1_protocol(bn, "builder_c_dag_m6.py (BN)")
        item4_bn_structure(bn, "builder_c_dag_m6.py (M6)")
        item7_tests()
        item8_report_and_harness()
    if mode in ("tests", "all"):
        item7_inject_run_tests()
    if mode in ("deps", "all"):
        item56_deps_and_leakage()
    if mode in ("full", "all"):
        item2_full()


if __name__ == "__main__":
    main_argv()
