"""Leak-proof NB alpha selection: pick by MEAN accuracy over 8 training-internal
splits (never the test set), then confirm robustness of the chosen value."""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,str(Path('.').resolve()))
from v4_common import load_split, accuracy, mcnemar
sys.path.insert(0,'naive_bayes/candidates')
import builder_a_candidate as A

SEEDS = (1322,7,2024,1,42,99,123,555)
ALPHAS = (1e-2,1e-3,1e-4,1e-5,1e-6,1e-8,1e-10)

table = {a:{} for a in ALPHAS}
for seed in SEEDS:
    Xtr,ytr,Xva,yva = load_split(seed=seed)
    for a in ALPHAS:
        m = A.NaiveBayesClassifier(alpha=a).fit(Xtr,ytr)
        table[a][seed] = accuracy(yva, m.predict(Xva))
    print("seed", seed, {f"{a:g}":round(table[a][seed],4) for a in ALPHAS}, flush=True)

print()
print(f"{'alpha':>8} {'mean':>8} {'min':>8} {'max':>8}")
means={}
for a in ALPHAS:
    v=np.array([table[a][s] for s in SEEDS]); means[a]=v.mean()
    print(f"{a:>8g} {v.mean():>8.4f} {v.min():>8.4f} {v.max():>8.4f}")
best = max(ALPHAS, key=lambda a: means[a])
print(f"\nargmax by 8-split MEAN (leak-proof selection): alpha={best:g}  mean={means[best]:.4f}")

# plateau check: is best statistically distinguishable from neighbours?
Xtr,ytr,Xva,yva = load_split(seed=1322)
pb = A.NaiveBayesClassifier(alpha=best).fit(Xtr,ytr).predict(Xva)
for a in ALPHAS:
    if a==best: continue
    pa = A.NaiveBayesClassifier(alpha=a).fit(Xtr,ytr).predict(Xva)
    mc = mcnemar(yva, pa, pb)
    print(f"  seed1322 {a:g} vs {best:g}: delta={accuracy(yva,pb)-accuracy(yva,pa):+.4f} McNemar p={mc['p_value']:.3g}")
json.dump({"table":{f"{a:g}":{str(s):table[a][s] for s in SEEDS} for a in ALPHAS},
           "means":{f"{a:g}":means[a] for a in ALPHAS}, "best":f"{best:g}"},
          open('naive_bayes/results/final_alpha_selection.json','w'), indent=2)
print("wrote naive_bayes/results/final_alpha_selection.json")
