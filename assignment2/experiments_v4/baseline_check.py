"""Establish ground-truth validation baselines for the v4 adversarial iteration.

Protocol: main.py stratified_split(y, 1/6, seed=1322) -> 49,999 train / 10,001 val.
Only training data (60,000) is loaded. Test set never touched.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from naive_bayes import NaiveBayesClassifier
from bayes_net import BayesNetClassifier

DATA = ROOT / "assignment2-data" / "assignment2-data" / "mnist"

def read_idx(path):
    import gzip, struct
    with gzip.open(path, "rb") as s:
        magic = s.read(4); dt = magic[2]; nd = magic[3]
        dtype_map = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4"}
        dims = [struct.unpack(">I", s.read(4))[0] for _ in range(nd)]
        raw = s.read()
    return np.frombuffer(raw, dtype=dtype_map[dt]).reshape(dims)

def split(y, frac=1/6, seed=1322):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = max(1, round(idx.size * frac))
        va.append(idx[:k]); tr.append(idx[k:])
    return np.concatenate(tr), np.concatenate(va)

X = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784)
y = read_idx(DATA / "train-labels-idx1-ubyte.gz").astype(np.int64)
tr, va = split(y)
print(f"train={tr.size} val={va.size}")
Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]

for a in (0.05, 0.1, 0.2, 0.5, 1.0):
    t = time.perf_counter()
    nb = NaiveBayesClassifier(alpha=a).fit(Xtr, ytr)
    acc = float(np.mean(nb.predict(Xva) == yva))
    print(f"NB alpha={a:<5} val_acc={acc:.4f}  ({time.perf_counter()-t:.1f}s)")

for a in (0.5, 1.0, 2.0):
    t = time.perf_counter()
    bn = BayesNetClassifier(alpha=a).fit(Xtr, ytr)
    acc = float(np.mean(bn.predict(Xva) == yva))
    print(f"BN alpha={a:<5} val_acc={acc:.4f}  ({time.perf_counter()-t:.1f}s)")
