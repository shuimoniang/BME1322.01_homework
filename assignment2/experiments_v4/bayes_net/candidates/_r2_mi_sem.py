import sys, numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parents[1]))
from _r2_common import Runner, M6, GRID
from _harness import DISCRETISERS, mutual_information

def cmi(a,b,y,na,nb):
    j=np.zeros((10,na,nb)); np.add.at(j,(y,a,b),1); j/=j.sum()
    tot=0.0
    for c in range(10):
        m=j[c]; pc=m.sum()
        if pc==0: continue
        pa=m.sum(axis=1,keepdims=True)/pc; pb=m.sum(axis=0,keepdims=True)/pc
        mm=m/pc; mask=mm>0
        tot+=pc*np.sum(mm[mask]*np.log(mm[mask]/(pa*pb)[mask]))
    return float(tot)

R=Runner()
atr,ytr,ava,yva,ctr,cva=R.world(1322)
def disc(key,k,d,src):
    CTr = ctr if src=='cont' else atr
    CVa = cva if src=='cont' else ava
    return DISCRETISERS[d](np.asarray(CTr[key],float),np.asarray(CVa[key],float),k)[0]

print("Marginal informativeness I(F;Y):")
for slot in range(8):
    a_abs=np.asarray(disc(GRID[slot][0],5,'zero_quantile','abs'))
    a_bb =np.asarray(disc(GRID[slot][1],5,'zero_quantile','cont'))
    print(f"  {GRID[slot][0].replace('dens_',''):4s} ABS I={mutual_information(a_abs,ytr,5):.4f}   "
          f"{GRID[slot][1]:5s} BBOX I={mutual_information(a_bb,ytr,5):.4f}")

print("\nConditional redundancy with the other retained ABSOLUTE cells, sum I(F;G|Y):")
others=[0,1,2,3,4,6]
for slot in (5,7):
    absA=np.asarray(disc(GRID[slot][0],5,'zero_quantile','abs'))
    bbA =np.asarray(disc(GRID[slot][1],5,'zero_quantile','cont'))
    tot_abs=sum(cmi(absA,np.asarray(disc(GRID[o][0],5,'zero_quantile','abs')),ytr,5,5) for o in others)
    tot_bb =sum(cmi(bbA, np.asarray(disc(GRID[o][0],5,'zero_quantile','abs')),ytr,5,5) for o in others)
    print(f"  slot {slot} ({GRID[slot][0]}): I(abs;others|Y)={tot_abs:.4f}  I(bbox;others|Y)={tot_bb:.4f}")
# cross-cell: bbMR/bbBR vs each other and vs euler
bbMR=np.asarray(disc('bbMR',5,'zero_quantile','cont')); bbBR=np.asarray(disc('bbBR',5,'zero_quantile','cont'))
absMR=np.asarray(disc('dens_mr',5,'zero_quantile','abs')); absBR=np.asarray(disc('dens_br',5,'zero_quantile','abs'))
eu=np.asarray(atr['euler']); eu=(np.clip(eu,-1,2).astype(np.int64)+1)
print(f"\n  I(bbMR;bbBR|Y)={cmi(bbMR,bbBR,ytr,5,5):.4f}  vs I(absMR;absBR|Y)={cmi(absMR,absBR,ytr,5,5):.4f}")
print(f"  I(bbMR;euler|Y)={cmi(bbMR,eu,ytr,5,4):.4f}  I(bbBR;euler|Y)={cmi(bbBR,eu,ytr,5,4):.4f}")
