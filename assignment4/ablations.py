"""四项必做消融，均复用公共评估协议。"""
import numpy as np
from sklearn.model_selection import train_test_split
from common import evaluate,pipe
from sklearn.tree import DecisionTreeRegressor

def single_splits(X,y):
 rows=[]
 for seed in range(10):
  tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed)
  rows.append(evaluate(pipe(DecisionTreeRegressor(max_depth=3,random_state=42),False),X,y,[(tr,te)])['mean']['r2'])
 return {'values':rows,'min':min(rows),'max':max(rows),'std':float(np.std(rows,ddof=0))}
