"""Main-session DAG adjudication: incumbent v3 vs C-M6 vs D-serpentine vs combos.

Uses the incumbent module with a swapped module-level FEATURE_PARENTS (the CPT
fitter reads that global), so features/discretisation/smoothing are held fixed.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,str(Path('.').resolve()))
from v4_common import load_split, accuracy, mcnemar, check_bn_bounds
import bayes_net as bn

DAGS = {
 "v3 incumbent":        (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9),
 "C_M6":                (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9),
 "D_serpentine":        (-1, 0, 1, 6, 3, 2, 7, 5, -1, 11, 9, 8),
}
# combos: serpentine grid + C's euler->ml/mr targeting, etc.
DAGS["combo_serp+euler_mid"] = (-1, 0, 1, 8, 3, 8, 7, 5, -1, 11, 9, 8)
DAGS["combo_serp+euler_mid+tl<-br"] = (7, 0, 1, 8, 3, 8, 7, 5, -1, 11, 9, 8)
# D grid serpentine with original semantic chain
DAGS["combo_serpgrid+orig_sem"] = (-1, 0, 1, 6, 3, 2, 7, 5, -1, 8, 9, 9)
DAGS["combo_Cgrid+serp_sem"] = (7, 0, 1, 8, 3, 8, 7, 5, -1, 11, 9, 8)
# no feature edges (pure NB) for reference
DAGS["pure_NB(no edges)"] = (-1,)*12

results = {}
for seed in (1322,7,2024):
    Xtr,ytr,Xva,yva=load_split(seed=seed)
    print(f"--- seed {seed} ---")
    for name,par in DAGS.items():
        bn.FEATURE_PARENTS = par
        try:
            check_bn_bounds(bn.FEATURE_NAMES, bn.FEATURE_CARDINALITIES, par)
        except AssertionError as e:
            print(f"  {name:32s} INVALID: {e}"); continue
        m = bn.BayesNetClassifier(alpha=1.0).fit(Xtr,ytr)
        pred = m.predict(Xva)
        acc = accuracy(yva,pred)
        results.setdefault(name,{})[seed]=(acc,pred,yva)
        print(f"  {name:32s} val={acc:.4f}")

print()
print("="*78)
print("seed 1322 head-to-head vs v3 incumbent (McNemar)")
print("="*78)
base_acc,base_pred,base_y = results["v3 incumbent"][1322]
for name in DAGS:
    if name=="v3 incumbent" or 1322 not in results.get(name,{}): continue
    acc,pred,y = results[name][1322]
    mc = mcnemar(base_y, base_pred, pred)
    print(f"  {name:32s} delta={acc-base_acc:+.4f}  b={mc['b']:>4} c={mc['c']:>4}  p={mc['p_value']:.4g}")
