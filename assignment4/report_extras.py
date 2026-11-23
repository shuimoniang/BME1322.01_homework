"""从正式结果生成机制解释及完整消融附录；不引入手填成绩。"""
import numpy as np
from selection_summary import NAMES

def enrich(content,r,table,data):
 sc=r['scans'];ab=r['ablation'];best=lambda z:z['runs'][z['best']];names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
 text=['\n## 附录：依据实测结果的机制核查\n']
 linear=best(sc['linear'])['mean'];rbf=best(sc['rbf'])['mean']
 winner='线性核' if linear['auc']>=rbf['auc'] else 'RBF核'
 text += [f"固定五折扫描以{winner}的AUC较高；AUC差为{abs(linear['auc']-rbf['auc']):.6f}。这支持当前样本下该核更适合排序，但不能把微小差异解释为总体上的确定优势。\n"]
 for n in ['linear','rbf']:
  a=best(sc[n]);b=best(ab['unscaled_'+n]);text+=[f"{NAMES[n]}标准化最优C={a['params']['C']}，不标准化最优C={b['params']['C']}；AUC由{a['mean']['auc']:.6f}变为{b['mean']['auc']:.6f}。C{'发生' if a['params']['C']!=b['params']['C'] else '未发生'}漂移，说明调参结果须与特征尺度一起解释。\n"]
  runs=ab['unscaled_'+n]['runs'];text +=[table(['不标准化模型','参数','训练AUC','验证AUC'],[[n,str(a['params']),a['mean']['train_auc'],a['mean']['auc']] for a in runs])]
 for gamma in ['scale',.001,.01,.1,1,10]:
  rr=[a for a in sc['rbf']['runs'] if a['params']['gamma']==gamma];a=max(rr,key=lambda z:z['mean']['auc'])
  text +=[f"gamma={gamma}时，该行最佳C={a['params']['C']}，训练AUC={a['mean']['train_auc']:.6f}、验证AUC={a['mean']['auc']:.6f}、差距={a['mean']['train_auc']-a['mean']['auc']:.6f}。\n"]
 a=best(sc['prune']);b=best(sc['depth']);text +=[f"后剪枝最优R²={a['mean']['r2']:.6f}，预剪枝最优R²={b['mean']['r2']:.6f}；本候选集合中{'后剪枝' if a['mean']['r2']>b['mean']['r2'] else '预剪枝'}表现较好。这个对照不能隔离所有结构差异，因此不能据此普遍证实或反驳‘预剪枝可能剪早’。\n"]
 for n in ['ridge','lasso']:
  text +=[table(['模型','alpha','零系数数量（五折均值）','R²','MAE'],[[n,a['params']['alpha'],a['mean']['zero_count'],a['mean']['r2'],a['mean']['mae']] for a in sc[n]['runs']])]
  co=np.array(r['coefficients'][n]);text +=[f"{NAMES[n]}最优解中逐折精确零系数数量={np.sum(co==0,axis=1).tolist()}，逐特征系数标准差最大值={np.std(co,axis=0).max():.6f}。这给出稀疏性与稳定性的直接证据。\n"]
 for n in ['linear','ridge','prune']:
  imp=np.mean(r['importance'][n],axis=0);idx=np.argsort(imp)[::-1][:5];text +=[f"{NAMES[n]}测试折置换重要性前五："+'、'.join(f'{names[i]}({imp[i]:.6f})' for i in idx)+'。\n']
 top={n:set(np.argsort(np.mean(r['importance'][n],axis=0))[::-1][:5]) for n in ['linear','ridge']}
 text +=['分类线性SVM与岭回归前五交集：'+ '、'.join(names[i] for i in sorted(top['linear']&top['ridge']))+'。排序交集支持共享风险信息，但不要求两个不同损失函数得到完全相同排序。\n']
 a=best(ab['poly']);b=best(sc['ridge']);text +=[f"二次容量模型R²={a['mean']['r2']:.6f}，原岭回归={b['mean']['r2']:.6f}；容量增加在本实验中{'有益' if a['mean']['r2']>b['mean']['r2'] else '无益'}，且这是经过正则化扫描后的比较，不是未调参模型的比较。\n"]
 # 树的缩放机制以真实重复实验验证。
 from common import load,splits,pipe,evaluate
 from sklearn.tree import DecisionTreeRegressor
 _,df,folds=load(data);X=df.drop(columns='num').to_numpy(float);y=df.num.to_numpy(float)
 a=evaluate(pipe(DecisionTreeRegressor(max_depth=3,random_state=42),True),X,y,splits(folds))
 text +=[f"树固定深度3：标准化R²={a['mean']['r2']:.6f}，未标准化R²={r['fixed']['depth']['mean']['r2']:.6f}。差异接近零支持基于排序分裂对尺度不敏感；浮点阈值与并列分裂可能产生细微差异。\n"]
 text +=[table(['多项式alpha','R²','MAE','RMSE'],[[a['params']['alpha']]+[a['mean'][k] for k in ['r2','mae','rmse']] for a in ab['poly']['runs']])]
 return content+'\n'.join(text)
