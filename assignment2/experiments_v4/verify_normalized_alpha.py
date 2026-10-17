"""Test state-count-aware (total-mass) smoothing: per-state pseudo-count = alpha/N_STATES.

Motivation: the 65536-state patch variable needs alpha interpreted as TOTAL mass,
else the PDF's default alpha=1.0 adds 65536 pseudo-counts/class (13x the data).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,str(Path('.').resolve()))
from v4_common import load_split, accuracy, mcnemar

THR=72.0; PH=PW=4; INNER=2; S=1<<16
def states(X):
    n=X.shape[0]; ink=X.reshape(n,28,28)>=THR; g=22
    st=np.zeros((n,g,g),dtype=np.int64); bit=0
    for i in range(PH):
        for j in range(PW):
            st|=ink[:,i*INNER:i*INNER+g,j*INNER:j*INNER+g].astype(np.int64)<<bit; bit+=1
    return st.reshape(n,g*g)

def fit(X,y,per_state_alpha):
    st=states(X); F=st.shape[1]; n=X.shape[0]
    cc=np.bincount(y,minlength=10).astype(np.float64)
    prior=np.log((cc+per_state_alpha)/(n+10*per_state_alpha))
    off=(np.arange(F,dtype=np.int64)*S)[None,:]
    keys=np.unique((st+off).ravel()); lp=np.empty((10,keys.size))
    default=np.log(per_state_alpha/(cc+S*per_state_alpha))
    for c in range(10):
        sub=(st[y==c]+off).ravel(); pos=np.searchsorted(keys,sub)
        cnt=np.bincount(pos,minlength=keys.size).astype(np.float64)
        lp[c]=np.log((cnt+per_state_alpha)/(cc[c]+S*per_state_alpha))
    return prior,lp,default,keys,F
def pred(X,mdl):
    prior,lp,default,keys,F=mdl
    st=states(X); off=(np.arange(F,dtype=np.int64)*S)[None,:]
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

print("TOTAL-MASS smoothing: alpha is per-FEATURE total mass, per-state = alpha/N_STATES")
print("Under the PDF default alpha=1.0 -> per-state pseudo-count = 1.0/65536 = 1.53e-05\n")
for seed in (1322,7,2024):
    Xtr,ytr,Xva,yva=load_split(seed=seed)
    print(f"--- seed {seed} ---")
    for a in (1.0,0.5,0.1,0.01):
        psa = a/S
        acc=accuracy(yva,pred(Xva,fit(Xtr,ytr,psa)))
        print(f"  alpha(total)={a:<6} per_state={psa:.2e}  val={acc:.4f}")
