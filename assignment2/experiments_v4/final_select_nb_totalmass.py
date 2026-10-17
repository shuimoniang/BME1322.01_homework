"""Leak-proof final NB selection under TOTAL-MASS smoothing.

alpha = total Dirichlet mass per patch variable; per-state pseudo-count = alpha/K.
Selected by MEAN accuracy over 8 training-internal splits (test set never read).
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,str(Path('.').resolve()))
from v4_common import load_split, accuracy, mcnemar

THR=72.0; PH=PW=4; INNER=2; K=1<<16
def states(X):
    n=X.shape[0]; ink=X.reshape(n,28,28)>=THR; g=22
    st=np.zeros((n,g,g),dtype=np.int64); bit=0
    for i in range(PH):
        for j in range(PW):
            st|=ink[:,i*INNER:i*INNER+g,j*INNER:j*INNER+g].astype(np.int64)<<bit; bit+=1
    return st.reshape(n,g*g)
def fit(X,y,a):
    st=states(X); F=st.shape[1]; n=X.shape[0]
    cc=np.bincount(y,minlength=10).astype(np.float64)
    prior=np.log((cc+a/K)/(n+10*a/K))
    off=(np.arange(F,dtype=np.int64)*K)[None,:]
    keys=np.unique((st+off).ravel()); lp=np.empty((10,keys.size))
    default=np.log((a/K)/(cc+K*(a/K)))
    for c in range(10):
        sub=(st[y==c]+off).ravel(); pos=np.searchsorted(keys,sub)
        cnt=np.bincount(pos,minlength=keys.size).astype(np.float64)
        lp[c]=np.log((cnt+a/K)/(cc[c]+K*(a/K)))
    return prior,lp,default,keys,F
def pred(X,mdl):
    prior,lp,default,keys,F=mdl
    st=states(X); off=(np.arange(F,dtype=np.int64)*K)[None,:]
    sc=np.tile(prior,(st.shape[0],1))+float(F)*default[None,:]
    for s in range(0,st.shape[0],1024):
        e=min(s+1024,st.shape[0]); flat=st[s:e]+off
        pos=np.searchsorted(keys,flat); np.clip(pos,0,keys.size-1,out=pos)
        found=keys[pos]==flat
        if not found.any(): continue
        si,_=np.nonzero(found); ki=pos[si,np.nonzero(found)[1]]
        corr=lp[:,ki]-default[:,None]
        for c in range(10): sc[s:e,c]+=np.bincount(si,weights=corr[c],minlength=e-s)
    return np.argmax(sc,axis=1)

SEEDS=(1322,7,2024,1,42,99,123,555)
ALPHAS=(10.0,3.0,1.0,0.3,0.1,0.03,0.01)
tab={a:{} for a in ALPHAS}
for seed in SEEDS:
    Xtr,ytr,Xva,yva=load_split(seed=seed)
    for a in ALPHAS:
        tab[a][seed]=accuracy(yva,pred(Xva,fit(Xtr,ytr,a)))
print(f"{'alpha':>8} {'mean':>8} {'min':>8}   (total-mass parametrisation)")
means={}
for a in ALPHAS:
    v=np.array([tab[a][s] for s in SEEDS]); means[a]=v.mean()
    print(f"{a:>8g} {v.mean():>8.4f} {v.min():>8.4f}")
best=max(ALPHAS,key=lambda a:means[a])
print(f"\nleak-proof argmax (8-split mean) = alpha={best:g}, mean={means[best]:.4f}, min={min(tab[best].values()):.4f}")
print("alpha=1.0 (PDF default) =", f"{means[1.0]:.4f}")
json.dump({"means":{f"{a:g}":means[a] for a in ALPHAS},
           "per_seed":{f"{a:g}":{str(s):tab[a][s] for s in SEEDS} for a in ALPHAS},
           "best":f"{best:g}"}, open('naive_bayes/results/final_alpha_totalmass.json','w'), indent=2)
