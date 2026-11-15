"""协议测试：运行 python -m unittest discover -s assignment4 -p test_protocol.py。"""
import unittest,json
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.svm import SVC
import common as c

class AuditRegressor(RegressorMixin,BaseEstimator):
 seen=[]
 def fit(self,X,y):
  self.__class__.seen.append((X.copy(),y.copy()));self.mean_=float(np.mean(y));return self
 def predict(self,X):return np.repeat(self.mean_,len(X))

class ProtocolTests(unittest.TestCase):
 def test_official_fold_integrity(self):
  _,df,folds=c.load();cv=c.splits(folds)
  self.assertEqual(sorted(np.concatenate([te for _,te in cv])),list(range(303)))
  for tr,te in cv:self.assertEqual(len(np.intersect1d(tr,te)),0)
 def test_auc_continuous(self):
  y=np.array([0,0,1,1]);pred=np.zeros(4);score=np.array([.1,.2,.3,.4])
  m=c.metrics(y,pred,score)
  self.assertEqual(m['auc'],1);self.assertEqual(m['hard_auc'],.5)
 def test_imputer_uses_training_only(self):
  X=np.array([[1.],[3.],[np.nan],[1000.],[2000.] ]);y=np.arange(5.)
  AuditRegressor.seen=[]
  c.evaluate(c.pipe(AuditRegressor(),False),X,y,[(np.array([0,1,2]),np.array([3,4]))])
  self.assertEqual(AuditRegressor.seen[0][0][2,0],2.)
 def test_tree_results_serializable(self):
  from sklearn.tree import DecisionTreeRegressor
  _,df,f=c.load();X=df.drop(columns='num').to_numpy(float);y=df.num.to_numpy(float)
  result=c.evaluate(c.pipe(DecisionTreeRegressor(max_depth=3,random_state=42),False),X,y,c.splits(f))
  json.dumps(result,allow_nan=False)
 def test_inner_excludes_outer(self):
  _,_,folds=c.load()
  for outer in range(5):
   pool=np.setdiff1d(np.arange(303),folds[outer])
   for tr,te in c.splits([f for j,f in enumerate(folds) if j!=outer],pool):
    self.assertEqual(len(np.intersect1d(tr,folds[outer])),0)
    self.assertEqual(len(np.intersect1d(te,folds[outer])),0)
 def test_rmse_aggregation(self):
  X=np.arange(6.).reshape(-1,1);y=np.array([0,0,0,1,2,10.])
  a=c.evaluate(c.pipe(AuditRegressor(),False),X,y,[(np.array([0,1,2]),np.array([3,4])),(np.array([0,1,2]),np.array([4,5]))])
  self.assertAlmostEqual(a['mean']['rmse'],np.sqrt((2.5+52)/2))

if __name__=='__main__':unittest.main()
