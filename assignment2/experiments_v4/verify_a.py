"""Independent verification of Builder A's dilated 4x4 binary patch NB.

(1) Run the builder's own module through the shared harness.
(2) Reimplement the idea from scratch (my own code) and compare.
(3) Time + memory.
"""
import sys, time, tracemalloc
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, str(Path('.').resolve()))
from v4_common import load_split, accuracy, check_nb_bounds

Xtr, ytr, Xva, yva = load_split()

# ---------- (1) builder's module ----------
sys.path.insert(0, 'naive_bayes/candidates')
import builder_a_candidate as A
t=time.perf_counter(); m=A.NaiveBayesClassifier(alpha=0.001).fit(Xtr,ytr); ft=time.perf_counter()-t
t=time.perf_counter(); p=m.predict(Xva); pt=time.perf_counter()-t
print(f"[builder A module] alpha=0.001 val={accuracy(yva,p):.4f} fit={ft:.1f}s pred={pt:.1f}s n_observed={m.n_observed_states_}")
print("  prob mass:", m.check_probability_mass())

# ---------- (2) my own independent reimplementation ----------
THR=72.0; PH=PW=4; INNER=2
def my_states(X):
    n=X.shape[0]; img=X.reshape(n,28,28); ink=img>=THR
    g=28-(3*2+1)+1  # 22
    st=np.zeros((n,g,g),dtype=np.int64); bit=0
    for i in range(PH):
        for j in range(PW):
            st |= ink[:, i*INNER:i*INNER+g, j*INNER:j*INNER+g].astype(np.int64)<<bit
            bit+=1
    return st.reshape(n,g*g)

S=1<<16
def fit_my(X,y,alpha):
    st=my_states(X); F=st.shape[1]; n=X.shape[0]
    cc=np.bincount(y,minlength=10).astype(np.float64)
    prior=np.log((cc+alpha)/(n+10*alpha))
    off=(np.arange(F,dtype=np.int64)*S)[None,:]
    keys=np.unique((st+off).ravel())
    lp=np.empty((10,keys.size))
    default=np.log(alpha/(cc+S*alpha))
    for c in range(10):
        sub=(st[y==c]+off).ravel()
        pos=np.searchsorted(keys,sub)
        cnt=np.bincount(pos,minlength=keys.size).astype(np.float64)
        lp[c]=np.log((cnt+alpha)/(cc[c]+S*alpha))
    return prior,lp,default,keys,F
def pred_my(X,model):
    prior,lp,default,keys,F=model
    st=my_states(X); off=(np.arange(F,dtype=np.int64)*S)[None,:]
    sc=np.tile(prior,(st.shape[0],1))+float(F)*default[None,:]
    bs=1024
    for s in range(0,st.shape[0],bs):
        e=min(s+bs,st.shape[0]); flat=st[s:e]+off
        pos=np.searchsorted(keys,flat); np.clip(pos,0,keys.size-1,out=pos)
        found=keys[pos]==flat
        if not found.any(): continue
        si,_=np.nonzero(found); ki=pos[si,np.nonzero(found)[1]]
        corr=lp[:,ki]-default[:,None]
        for c in range(10):
            sc[s:e,c]+=np.bincount(si,weights=corr[c],minlength=e-s)
    return np.argmax(sc,axis=1)

tracemalloc.start()
t=time.perf_counter(); model=fit_my(Xtr,ytr,0.001); ft2=time.perf_counter()-t
cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
pm=pred_my(Xva,model)
print(f"[MY reimpl]       alpha=0.001 val={accuracy(yva,pm):.4f} fit={ft2:.1f}s peak_alloc={peak/1e6:.0f}MB")
print(f"  agreement with builder A predictions: {np.mean(pm==p):.4f}")

# ---------- (3) alpha sensitivity on my reimpl ----------
for a in (1e-4,1e-3,1e-2,0.05,0.5):
    mm=fit_my(Xtr,ytr,a)
    print(f"  [my reimpl] alpha={a:<8} val={accuracy(yva,pred_my(Xva,mm)):.4f}")
