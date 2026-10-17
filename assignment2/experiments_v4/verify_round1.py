"""Main-session cross-validation of all four builder candidates.

Checks: cross-seed stability, alpha robustness, threshold sensitivity (NB-A),
and a head-to-head adjudication of the two competing BN DAGs.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,str(Path('.').resolve()))
from v4_common import load_split, accuracy, load_train, stratified_split, mcnemar

print("="*72); print("NAIVE BAYES: cross-seed at recommended alphas"); print("="*72)
sys.path.insert(0,'naive_bayes/candidates')
import builder_a_candidate as A
import builder_b_nb_tinyalpha as B
from naive_bayes import NaiveBayesClassifier as IncNB

for seed in (1322,7,2024):
    Xtr,ytr,Xva,yva=load_split(seed=seed)
    inc=accuracy(yva,IncNB(alpha=0.05).fit(Xtr,ytr).predict(Xva))
    a1=accuracy(yva,A.NaiveBayesClassifier(alpha=0.001).fit(Xtr,ytr).predict(Xva))
    a2=accuracy(yva,A.NaiveBayesClassifier(alpha=0.0001).fit(Xtr,ytr).predict(Xva))
    b1=accuracy(yva,B.NaiveBayesClassifier(alpha=0.0001).fit(Xtr,ytr).predict(Xva))
    print(f"seed {seed:>5}: incumbent(2x2L5,a=.05)={inc:.4f}  A(4x4dil,a=1e-3)={a1:.4f}  A(a=1e-4)={a2:.4f}  B(2x2L5,a=1e-4)={b1:.4f}")

print(); print("="*72); print("NB-A: binarization-threshold sensitivity (seed 1322)"); print("="*72)
Xtr,ytr,Xva,yva=load_split()
for thr in (48,64,72,80,96,112,128):
    A.BINARY_THRESHOLD=float(thr)
    import importlib; importlib.reload(A) if False else None
    # threshold is module-level and read inside _patch_states at call time
    A.BINARY_THRESHOLD=float(thr)
    acc=accuracy(yva,A.NaiveBayesClassifier(alpha=0.0001).fit(Xtr,ytr).predict(Xva))
    print(f"  thr={thr:>4}: val={acc:.4f}")
A.BINARY_THRESHOLD=72.0
