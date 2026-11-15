"""Task A：SVM 双核，正式 AUC 始终使用 decision_function。"""
from sklearn.svm import SVC
from common import pipe
C_VALUES=[.001,.01,.1,1,10,100]
GAMMAS=['scale',.001,.01,.1,1,10]
def candidates(kernel,scale=True):
 return [({'C':c,'gamma':g,'scale':scale},pipe(SVC(C=c,kernel=kernel,gamma=g,cache_size=256),scale),None)
  for c in C_VALUES for g in (GAMMAS if kernel=='rbf' else ['scale'])]
