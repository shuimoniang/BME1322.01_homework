"""公共协议：课程固定折、折内变换、连续分数及可审计结果。"""
import json, hashlib, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
 roc_auc_score, r2_score, mean_absolute_error, mean_squared_error, confusion_matrix)
from sklearn.exceptions import ConvergenceWarning
ROOT=Path(__file__).resolve().parent
SEED=42
warnings.filterwarnings('error', category=ConvergenceWarning)

def load(data_dir=None):
 p=Path(data_dir) if data_dir else ROOT.parent/'data'/'assignment4-data'
 df=pd.read_csv(p/'heart_cleveland.csv'); folds=json.loads((p/'splits.json').read_text())['cleveland']['folds']
 assert df.shape==(303,14) and df['num'].value_counts().to_dict()=={0:164,1:55,2:36,3:35,4:13}
 assert df.isna().sum().to_dict()['ca']==4 and df.isna().sum().sum()==6
 assert sorted(sum(folds,[]))==list(range(303))
 return p,df,[np.array(f) for f in folds]

def splits(folds, pool=None):
 pool=np.arange(303) if pool is None else np.asarray(pool)
 return [(np.setdiff1d(pool,f),np.intersect1d(pool,f)) for f in folds if len(np.intersect1d(pool,f))]

def pipe(model,scale=True,poly=False):
 steps=[('impute',SimpleImputer(strategy='median'))]
 if scale: steps.append(('scale',StandardScaler()))
 if poly: steps.extend([('poly',PolynomialFeatures(2,include_bias=False)),('poly_scale',StandardScaler())])
 return Pipeline(steps+[('model',model)])

def metrics(y,pred,score=None):
 if score is None:
  return {'r2':float(r2_score(y,pred)), 'mae':float(mean_absolute_error(y,pred)), 'mse':float(mean_squared_error(y,pred)),'rmse':float(np.sqrt(mean_squared_error(y,pred)))}
 out={'accuracy':float(accuracy_score(y,pred)),'auc':float(roc_auc_score(y,score)), 'hard_auc':float(roc_auc_score(y,pred))}
 for c in [0,1]:
  for name,fun in [('precision',precision_score),('recall',recall_score),('f1',f1_score)]:
   out[f'{name}_{c}']=float(fun(y,pred,pos_label=c,zero_division=0))
 out.update(precision=out['precision_1'],recall=out['recall_1'],f1=out['f1_1'],sensitivity=out['recall_1'],specificity=out['recall_0'],auc_0=out['auc'],auc_1=out['auc'],accuracy_0=out['accuracy'],accuracy_1=out['accuracy'])
 return out

def evaluate(model,X,y,cv,cls=False,extra=None):
 records=[]; predictions=[]
 for k,(tr,te) in enumerate(cv):
  m=clone(model)
  if extra: m=extra(m,X[tr],y[tr])
  m.fit(X[tr],y[tr]); p=m.predict(X[te]); s=m.decision_function(X[te]) if cls else None
  a=metrics(y[te],p,s); train=metrics(y[tr],m.predict(X[tr]),m.decision_function(X[tr]) if cls else None)
  a.update({'train_'+key:value for key,value in train.items()})
  est=m[-1]
  if hasattr(est,'tree_'):
   t=est.tree_; counts=t.n_node_samples[t.children_left==-1]
   a.update(depth=int(est.get_depth()),leaves=int(est.get_n_leaves()),leaf_min=int(min(counts)),leaf_median=float(np.median(counts)),singleton_fraction=float(np.mean(counts==1)),ccp_alpha=est.ccp_alpha)
  if hasattr(est,'n_support_'): a['support_vectors']=int(sum(est.n_support_))
  if hasattr(est,'coef_'): a['zero_count']=int(np.sum(est.coef_==0))
  records.append(a)
  for j,i in enumerate(te): predictions.append({'row':int(i),'fold':k,'truth':float(y[i]),'prediction':float(p[j]),'score':float(s[j]) if cls else float(p[j])})
 avg={key:float(np.mean([r[key] for r in records])) for key in records[0]}
 if not cls: avg['rmse']=float(np.sqrt(avg['mse']))
 return {'mean':avg,'folds':records,'predictions':predictions}

def sweep(candidates,X,y,cv,cls=False):
 output=[]
 for name,model,extra in candidates:
  r=evaluate(model,X,y,cv,cls,extra); r['params']=name; output.append(r)
 key='auc' if cls else 'r2'
 best=max(range(len(output)),key=lambda i:round(output[i]['mean'][key],6))
 return {'runs':output,'best':best}

def pruning(q):
 def set_alpha(m,X,y):
  z=m[:-1].fit_transform(X,y)
  path=m[-1].cost_complexity_pruning_path(z,y).ccp_alphas
  m.set_params(model__ccp_alpha=float(path[int(round(q*(len(path)-1)))]))
  return m
 return set_alpha

def nested(candidates,X,y,folds,cls=False):
 out=[]; choices=[]
 for k,test in enumerate(folds):
  train=np.setdiff1d(np.arange(len(y)),test)
  inner=splits([f for j,f in enumerate(folds) if j!=k],train)
  scan=sweep(candidates,X,y,inner,cls); chosen=scan['best']; name,m,extra=candidates[chosen]
  r=evaluate(m,X,y,[(train,test)],cls,extra); out.append(r); choices.append(name)
 avg={key:float(np.mean([r['mean'][key] for r in out])) for key in out[0]['mean']}
 if not cls: avg['rmse']=float(np.sqrt(avg['mse']))
 return {'mean':avg,'choices':choices,'folds':[r['folds'][0] for r in out],'predictions':[dict(p,fold=k) for k,r in enumerate(out) for p in r['predictions']]}
