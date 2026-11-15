"""Task B：预剪枝、训练折路径后剪枝、正则化及 SVR。"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge,Lasso,LinearRegression
from sklearn.svm import SVR
from common import pipe,pruning,SEED

def candidates(name):
 if name=='depth':
  return [({'max_depth':d},pipe(DecisionTreeRegressor(max_depth=d,random_state=SEED),False),None) for d in list(range(1,13))+[None]]
 if name=='prune':
  return [({'path_fraction':q},pipe(DecisionTreeRegressor(random_state=SEED),False),pruning(q)) for q in [1,.95,.9,.75,.5,.25,.1,0]]
 if name in ['ridge','poly']:
  return [({'alpha':a},pipe(Ridge(alpha=a),True,name=='poly'),None) for a in [10000,1000,100,10,1,.1,.01,.001]]
 if name=='lasso':
  return [({'alpha':a},pipe(Lasso(alpha=a,max_iter=100000,tol=1e-6)),None) for a in [10,1,.1,.01,.001,.0001]]
 if name=='ols': return [({},pipe(LinearRegression()),None)]
 if name=='svr':
  return [({'C':c,'epsilon':e,'gamma':g},pipe(SVR(C=c,epsilon=e,gamma=g)),None) for c in [.1,1,10,100] for e in [.01,.1,.5] for g in ['scale',.01,.1]]
 raise ValueError(name)
