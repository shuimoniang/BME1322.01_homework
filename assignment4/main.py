"""一键实验：python assignment4/main.py，输出表格、图和实验数据。"""
import argparse,json,re,hashlib,platform,contextlib,io,importlib.metadata
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.linear_model import Ridge,Lasso,LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
import common as c
import classify,regress,ablations
ROOT=c.ROOT

def best(s):return s['runs'][s['best']]
def run(data, figures):
 p,df,folds=c.load(data); X=df.drop(columns='num').to_numpy(float); y=df.num.to_numpy(float); yc=(y>0).astype(int); cv=c.splits(folds)
 r={'data_hashes':{f:hashlib.sha256((p/f).read_bytes()).hexdigest() for f in ['heart_cleveland.csv','splits.json','baselines.py']},'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','matplotlib']},'python':platform.python_version(),'scans':{},'nested':{},'ablation':{}}
 for name in ['linear','rbf','depth','prune','ridge','lasso','ols','svr']:
  print('主实验/独立选参',name,flush=True)
  cls=name in ['linear','rbf']; candidates=classify.candidates(name) if cls else regress.candidates(name)
  r['scans'][name]=c.sweep(candidates,X,yc if cls else y,cv,cls)
  r['nested'][name]=c.nested(candidates,X,yc if cls else y,folds,cls)
 # 原始基线保持不变，程序化调用并保存基线输出。
 with contextlib.chdir(p),contextlib.redirect_stdout(io.StringIO()) as log:
  import runpy
  b=runpy.run_path(str((p/'baselines.py').resolve())); reference={}; b['heart'](reference)
 r['reference']=reference; (figures/'baseline_output.txt').write_text(log.getvalue(),encoding='utf-8')
 specifications=[('linear','cleveland/cls/SVM (linear)',c.pipe(SVC(kernel='linear')),True),('rbf','cleveland/cls/SVM (RBF)',c.pipe(SVC()),True),('depth','cleveland/reg/Regression Tree (d=3)',c.pipe(DecisionTreeRegressor(max_depth=3,random_state=42),False),False),('ridge','cleveland/reg/Ridge (alpha=1)',c.pipe(Ridge(alpha=1)),False),('lasso','cleveland/reg/Lasso (alpha=0.01)',c.pipe(Lasso(alpha=.01)),False),('ols','cleveland/reg/Linear Regression',c.pipe(LinearRegression(),False),False),('svr','cleveland/reg/SVR (RBF, C=1)',c.pipe(SVR()),False)]
 r['fixed']={};r['verification']=[]
 for name,key,m,cls in specifications:
  a=c.evaluate(m,X,yc if cls else y,cv,cls);r['fixed'][name]=a
  for metric,v in reference[key].items():
   own=a['mean'][metric];delta=own-v
   r['verification'].append({'model':name,'metric':metric,'reference':v,'own':own,'delta':delta})
   assert abs(delta)<=.01,(name,metric,delta)
 # DummyClassifier 不提供 decision_function，手动计算全部分类指标。
 records=[];preds=[]
 for k,(tr,te) in enumerate(cv):
  pred=np.full(len(te),int(np.mean(yc[tr])>.5));records.append(c.metrics(yc[te],pred,pred))
 r['dummy_cls']={'mean':{key:float(np.mean([a[key] for a in records])) for key in records[0]}}
 r['dummy_reg']=c.evaluate(c.pipe(DummyRegressor(),False),X,y,cv)
 for name in ['linear','rbf']:
  print('预处理消融',name,flush=True)
  r['ablation']['unscaled_'+name]=c.sweep(classify.candidates(name,False),X,yc,cv,True)
 keep=[i for i,n in enumerate(df.columns[:-1]) if n not in ['ca','thal']]; small=X[:,keep]
 for name in ['linear','rbf','depth','ridge','lasso','svr']:
  print('特征消融',name,flush=True)
  cls=name in ['linear','rbf'];cc=classify.candidates(name) if cls else regress.candidates(name)
  r['ablation']['drop_'+name]=c.sweep(cc,small,yc if cls else y,cv,cls)
  fixed=next(a for a in specifications if a[0]==name)
  r['ablation']['drop_fixed_'+name]=c.evaluate(fixed[2],small,yc if cls else y,cv,cls)
 print('容量与随机划分',flush=True)
 r['ablation']['poly']=c.sweep(regress.candidates('poly'),X,y,cv)
 r['ablation']['single']=ablations.single_splits(X,y)
 r['coefficients']={};r['importance']={}
 for name in ['linear','ridge','lasso','ols','prune']:
  cc=classify.candidates(name) if name=='linear' else regress.candidates(name)
  _,m,extra=cc[r['scans'][name]['best']]; allcoef=[];imp=[]
  for tr,te in cv:
   mm=c.clone(m)
   if extra:mm=extra(mm,X[tr],y[tr])
   mm.fit(X[tr],yc[tr] if name=='linear' else y[tr])
   if hasattr(mm[-1],'coef_'):allcoef.append(mm[-1].coef_.reshape(-1).tolist())
   imp.append(permutation_importance(mm,X[te],yc[te] if name=='linear' else y[te],scoring='roc_auc' if name=='linear' else 'r2',n_repeats=10,random_state=42).importances_mean.tolist())
  r['coefficients'][name]=allcoef;r['importance'][name]=imp
 cc=regress.candidates('prune');_,m,extra=cc[r['scans']['prune']['best']];m=extra(m,X,y);m.fit(X,y)
 plt.figure(figsize=(24,12));plot_tree(m[-1],feature_names=list(df.columns[:-1]),filled=True,rounded=True,fontsize=7);plt.tight_layout();plt.savefig(figures/'tree.png',dpi=150);plt.close()
 r['tree_nodes']=[{'node':i,'feature':str(df.columns[m[-1].tree_.feature[i]]) if m[-1].tree_.feature[i]>=0 else 'leaf','threshold':float(m[-1].tree_.threshold[i]),'samples':int(m[-1].tree_.n_node_samples[i])} for i in range(min(7,m[-1].tree_.node_count))]
 return r

MODEL_NAMES = {
 'linear': 'SVM（线性核）', 'rbf': 'SVM（RBF核）',
 'depth': '回归树（预剪枝）', 'prune': '回归树（后剪枝）',
 'ridge': '岭回归', 'lasso': 'Lasso回归',
 'ols': '普通最小二乘线性回归', 'svr': '支持向量回归（SVR）',
 'poly': '二次多项式岭回归',
}

def model_name(value):
 return MODEL_NAMES.get(value,value) if isinstance(value,str) else value

def table(headers,rows):
 def fmt(x):return f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(model_name(x))
 return '| '+' | '.join(str('树深' if x=='depth' else model_name(x)) for x in headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in rows)+'\n'

def figure(figures,name,title,x,series,xlabel='参数'):
 plt.figure(figsize=(8,4.8))
 for label,values in series.items():plt.plot(x,values,'o-',label=label)
 plt.title(title);plt.xlabel(xlabel);plt.legend();plt.grid(alpha=.2);plt.tight_layout();plt.savefig(figures/name,dpi=150);plt.close()

from selection_summary import comparisons, reason, small_alpha_note, kernel_conclusion, rbf_gamma_analysis

def render_tables_and_figures(r,data,fig):
 p,df,folds=c.load(data);sc=r['scans'];ab=r['ablation'];names=list(df.columns[:-1])
 plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
 plt.figure();df.num.value_counts().sort_index().plot.bar();plt.xlabel('num');plt.ylabel('人数');plt.tight_layout();plt.savefig(fig/'labels.png',dpi=150);plt.close()
 plt.figure();plt.scatter(df.oldpeak,df.num,alpha=.4);plt.xlabel('oldpeak');plt.ylabel('num');plt.tight_layout();plt.savefig(fig/'feature.png',dpi=150);plt.close()
 for name in sc:
  runs=sc[name]['runs'];cls=name in ['linear','rbf'];metric='auc' if cls else 'r2'
  figure(fig,name+'.png',model_name(name)+'：训练与固定五折',list(range(len(runs))),{'训练':[a['mean']['train_'+metric] for a in runs],'验证':[a['mean'][metric] for a in runs]},'候选编号（对应扫描表）')
 vals=np.array([a['mean']['auc'] for a in sc['rbf']['runs']]).reshape(len(classify.C_VALUES),len(classify.GAMMAS))
 plt.figure();plt.imshow(vals,aspect='auto');plt.xticks(range(6),classify.GAMMAS);plt.yticks(range(6),classify.C_VALUES);plt.xlabel('gamma');plt.ylabel('C');plt.colorbar(label='AUC');plt.tight_layout();plt.savefig(fig/'heatmap.png',dpi=150);plt.close()
 plt.figure(figsize=(9,7));plt.imshow(df.corr(),vmin=-1,vmax=1,cmap='coolwarm');plt.xticks(range(14),df.columns,rotation=90);plt.yticks(range(14),df.columns);plt.colorbar();plt.tight_layout();plt.savefig(fig/'correlation.png',dpi=150);plt.close()
 classification=comparisons(r,True,table)
 regression=comparisons(r,False,table)
 out=['# 医学数据的分类与回归\n','## 1. 设计与复现\n303例、13特征；分类目标 num>0，回归目标 num（0–4有序分级，不是血管支数）。二分类对应有无显著狭窄；回归保留程度信息，但假设等级距离可比较。五分类会损失顺序结构且最重等级仅13例，估计更不稳定。分类准确率与回归R²不可直接比较。\n', '安装依赖后运行 `python assignment4/main.py`。默认数据目录是仓库根目录 `data/assignment4-data`，可用 `--data-dir` 指定。全部种子42，随机划分种子0–9。版本及哈希见 figures/results.json。公共管线处理缺失和标准化；分类、回归、消融分别由对应脚本提供候选及实验定义。\n','课程固定五折用于扫描，扫描最优不等于无偏泛化估计；独立检查复用五个原始折，外层测试隔离，内层四折选参。分类按AUC、回归按R²选参；六位小数相同保留候选表中先出现的设置。\n','## 2. 数据探索\n',table(['字段','最小值','最大值','缺失数'],[[n,df[n].min(),df[n].max(),int(df[n].isna().sum())] for n in df.columns]),'连续量为age、trestbps、chol、thalach、oldpeak；其他输入沿用课程数值编码，因而类别间数值距离是假设和局限。缺失位于ca四处、thal两处，共6个单元格。处理方式为在Pipeline中使用SimpleImputer(strategy=\"median\")：每折仅根据训练集各列的非缺失值计算中位数，再用这些值填补训练集和验证集；随后才进行标准化（树模型省略标准化）和模型拟合。选择这一方案首先是为了与课程baselines.py的预处理保持一致，使基线核对与后续扫描具有可比性；其次，303例属于小样本，保留含少量缺失的患者可避免删除整行带来的信息损失，中位数也比均值更少受极端值影响。填补器只在训练折拟合，可避免验证集分布参与预处理造成信息泄漏。这是简单、可复现的基线方案，并非对缺失临床值的真实恢复：ca、thal均为离散编码，尤其thal编码间距没有连续数值意义，因此中位数不一定是最合理的类别填补方式；本次沿用课程协议，尚未通过与众数填补等方案的消融证明其最优，也不假定缺失机制已被验证。\n![标签](figures/labels.png)\n![特征](figures/feature.png)\n![相关](figures/correlation.png)\n','## 3. 复现核对\n',table(['方法','指标','基线','我的值','差值'],[[v[k] for k in ['model','metric','reference','own','delta']] for v in r['verification']]),'我从这张表里读出：纳入实验的全部基线项均在0.01容差内，正式扫描以这一已核对的管线为基础。原参考脚本另有不要求的分类器，仅用作背景，没有冒充自行实现。\n','## 4. 分类结果\n',classification,'我从这张表里读出：'+f"线性核扫描AUC={best(sc['linear'])['mean']['auc']:.6f}，RBF={best(sc['rbf'])['mean']['auc']:.6f}；差距需结合独立选参结果判断。\n",'AUC调用decision_function；负类以相反分数和相反标签计算时AUC相同，逐类accuracy按one-vs-rest定义也相同。敏感性等于正类recall，特异性等于负类recall；precision以预测为该类的样本为分母。\n![网格](figures/heatmap.png)\n']
 out.append(kernel_conclusion(r))
 out += ['### 参数遍历所得最优配置的分类指标对照\n',f"在本次参数网格中，分别以五折平均验证AUC最高为准则选取两种核的配置：线性核C={best(sc['linear'])['params']['C']}；RBF核C={best(sc['rbf'])['params']['C']}、gamma={best(sc['rbf'])['params']['gamma']}。下表报告这两组配置的五折指标均值。这里的最优指已遍历网格内的验证AUC最优，不代表所有指标同时最优；独立选参检查结果另见第六节。\n",table(['指标','SVM（线性核，网格最优）','SVM（RBF核，网格最优）'],[[metric,best(sc['linear'])['mean'].get(metric,'—'),best(sc['rbf'])['mean'].get(metric,'—')] for metric in dict.fromkeys(list(best(sc['linear'])['mean'])+list(best(sc['rbf'])['mean']))]),'指标说明：train_前缀表示训练折指标，其余为验证折指标；类别后缀0表示阴性、1表示阳性。敏感性等于阳性召回率，特异性等于阴性召回率。support_vectors为支持向量数量的五折均值；zero_count为线性核权重中精确零系数数量的五折均值，RBF核不适用，以“—”表示。\n']
 for n in ['linear','rbf']:
  b=best(sc[n]);pred=b['predictions']
  cm=confusion_matrix([z['truth'] for z in pred],[z['prediction'] for z in pred],labels=[0,1])
  fig_cm,ax=plt.subplots(figsize=(6,5))
  im=ax.imshow(cm,cmap='Blues',vmin=0)
  labels=['阴性（num=0）','阳性（num>0）']
  ax.set(xticks=[0,1],yticks=[0,1],xticklabels=labels,yticklabels=labels,xlabel='预测类别',ylabel='真实类别',title=f'{model_name(n)}：固定五折折外混淆矩阵')
  for i in range(2):
   for j in range(2):
    ax.text(j,i,f'{cm[i,j]}',ha='center',va='center',fontsize=18,color='white' if cm[i,j]>cm.max()/2 else 'black')
  fig_cm.colorbar(im,ax=ax,label='患者数')
  fig_cm.tight_layout();fig_cm.savefig(fig/f'confusion_{n}.png',dpi=180);plt.close(fig_cm)
  out += [f'### {model_name(n)} 混淆矩阵\n',f'![{model_name(n)} 混淆矩阵](figures/confusion_{n}.png)\n',f'图中汇总扫描最优参数在固定五折上的303例折外预测，行是真实类别，列是预测类别；TN={cm[0,0]}、FP={cm[0,1]}、FN={cm[1,0]}、TP={cm[1,1]}。这是课程五折扫描结果，不是嵌套选参的外层结果。\n']
 for n in ['linear','rbf']:
  s=sc[n]
  cls=n in ['linear','rbf'];keys=['auc','train_auc','accuracy','hard_auc','support_vectors'] if cls else ['r2','train_r2','mae','rmse']+(['depth','leaves','leaf_min','leaf_median','singleton_fraction','ccp_alpha'] if n in ['depth','prune'] else [])
  out += [f'### {model_name(n)} 完整扫描\n选定：`{best(s)["params"]}`。\n',table(['编号','参数']+keys,[[i,str(a['params'])]+[a['mean'][k] for k in keys] for i,a in enumerate(s['runs'])]),f'![{model_name(n)}](figures/{n}.png)\n']
  out.append(reason(r,n,table))
  if n=='rbf': out.append(rbf_gamma_analysis(r,table,fig))
 out+=['## 5. 回归结果\n',regression,'我从这张表里读出：应在同一指标下比较模型；树的深度不能替代验证成绩。后剪枝用训练折路径的位置选候选，各折实际alpha保存在结果中。\n','![回归树后剪枝结构图](figures/tree.png)\n图：代价复杂度后剪枝后的回归树。采用五折扫描选出的最优剪枝路径位置，在全部303例数据上重新计算剪枝路径、确定ccp_alpha并拟合，用于展示模型结构。该图不是预剪枝树，也不是某个测试折的树；报告中的泛化指标均由交叉验证获得，不使用这棵全数据解释树的训练成绩。部分节点规则如下（节点编号采用树数组顺序，非层序）：\n',table(['节点','特征','阈值','样本数'],[[z[k] for k in ['node','feature','threshold','samples']] for z in r['tree_nodes']]),'规则左分支为特征≤阈值。ca、thal相关分裂属于题目所述昂贵检查信息；胸痛编码和ST压低相关分裂表示风险分组，不能解释为因果或诊断建议。\n']
 out += ['### 预剪枝与后剪枝的区别及效果比较\n','预剪枝在树生长过程中施加约束，本实验通过max_depth限制最大深度；达到限制后不再继续分裂。后剪枝先生成完整树，再根据代价复杂度准则，以ccp_alpha权衡拟合误差和叶节点数量，删除收益不足的子树。因此预剪枝可能阻止后续有益分裂，后剪枝则先考察更完整的结构，但二者都需要验证数据选择复杂度。\n',f"按本作业固定五折平均验证R²选参，预剪枝最优为max_depth={best(sc['depth'])['params']['max_depth']}，R²={best(sc['depth'])['mean']['r2']:.6f}；后剪枝最优R²={best(sc['prune'])['mean']['r2']:.6f}。因此在本次候选范围和课程五折扫描口径下，预剪枝更有效，未观察到后剪枝因避免‘剪早’而取得更优验证结果。\n",f"独立选参检查中，预剪枝与后剪枝的外层平均R²分别为{r['nested']['depth']['mean']['r2']:.6f}和{r['nested']['prune']['mean']['r2']:.6f}，差距较小且排序相反，说明扫描优势不应扩大为稳定的总体优劣结论。深树容易形成样本很少甚至只有一个样本的叶节点，记住训练噪声；本实验的叶内样本数和训练／验证差距用于检验这一机制。\n"]
 for n,s in sc.items():
  if n in ['linear','rbf']: continue
  if n=='ols':
   a=best(s);fold_metrics=['train_r2','r2','train_mae','mae','train_rmse','rmse']
   rows=[[i+1]+[z[k] for k in fold_metrics] for i,z in enumerate(a['folds'])]
   rows.append(['五折汇总']+[a['mean'][k] for k in fold_metrics])
   out += ['### 普通最小二乘线性回归：五折评估\n','普通最小二乘线性回归不含正则化强度、核宽度或树深等需要扫描的超参数。本实验采用带截距的LinearRegression，沿用训练折中位数填补和标准化管线，在课程固定五折上评估。该模型作为无正则化的线性参照，用于与岭回归和Lasso比较，不进行参数扫描。\n',table(['验证折','训练R²','验证R²','训练MAE','验证MAE','训练RMSE','验证RMSE'],rows),'汇总行的R²和MAE为五折算术平均，RMSE为各折MSE平均后开方，与主结果表口径一致。\n']
   fig_ols,axes=plt.subplots(1,2,figsize=(11,4.5))
   x=np.arange(1,6)
   axes[0].bar(x-.18,[z['train_r2'] for z in a['folds']],width=.36,label='训练')
   axes[0].bar(x+.18,[z['r2'] for z in a['folds']],width=.36,label='验证')
   axes[0].set(xlabel='验证折编号',ylabel='R²',xticks=x,title='五折训练与验证表现');axes[0].legend()
   predicted=a['predictions'];truth=[z['truth'] for z in predicted];estimate=[z['prediction'] for z in predicted]
   axes[1].scatter(truth,estimate,alpha=.4,s=18)
   low=min(0,min(estimate));high=max(4,max(estimate));axes[1].plot([low,high],[low,high],'--',color='gray',label='预测=真实')
   axes[1].set(xlabel='真实严重程度（num）',ylabel='折外预测值',title='303例折外预测');axes[1].legend()
   fig_ols.suptitle('普通最小二乘线性回归');fig_ols.tight_layout();fig_ols.savefig(fig/'ols.png',dpi=180);plt.close(fig_ols)
   out += ['![普通最小二乘线性回归五折评估](figures/ols.png)\n',f"左图比较各折的训练与验证R²；右图使用全部303例折外预测，虚线为理想预测。五折汇总R²={a['mean']['r2']:.6f}、MAE={a['mean']['mae']:.6f}、RMSE={a['mean']['rmse']:.6f}，预测值保留连续数值，不截断到0–4。\n"]
   continue
  cls=n in ['linear','rbf'];keys=['auc','train_auc','accuracy','hard_auc','support_vectors'] if cls else ['r2','train_r2','mae','rmse']+(['depth','leaves','leaf_min','leaf_median','singleton_fraction','ccp_alpha'] if n in ['depth','prune'] else [])
  display_runs=sorted(s['runs'],key=lambda a:a['params']['alpha']) if n=='ridge' else s['runs']
  if n=='ridge':
   figure(fig,'ridge.png','岭回归：训练与固定五折',list(range(len(display_runs))),{'训练':[a['mean']['train_r2'] for a in display_runs],'验证':[a['mean']['r2'] for a in display_runs]},'候选编号（alpha从小到大，对应扫描表）')
  out += [f'### {model_name(n)} 完整扫描\n选定：`{best(s)["params"]}`。\n',table(['编号','参数']+keys,[[i,str(a['params'])]+[a['mean'][k] for k in keys] for i,a in enumerate(display_runs)]),f'![{model_name(n)}](figures/{n}.png)\n']
  out.append(reason(r,n,table))
  if n=='ridge': out.append(small_alpha_note(r,table))
 unpruned=next(a for a in sc['prune']['runs'] if a['params']['path_fraction']==0)
 out += ['### 回归树后剪枝的参照与剪枝效果\n','课程基线脚本只提供max_depth=3的回归树，没有单独规定后剪枝的固定参数基线。因此以相同固定五折上的完整树（max_depth=None、ccp_alpha=0）作为剪枝前对照，并列出课程深度3基线，区分剪枝自身的收益与相对课程基线的表现。\n',table(['设置','R²','MAE','RMSE','平均树深','平均叶节点数'],[[label]+[a['mean'][k] for k in ['r2','mae','rmse','depth','leaves']] for label,a in [('完整回归树（剪枝前）',unpruned),('课程回归树基线（深度3）',r['fixed']['depth']),('回归树后剪枝（网格最优）',best(sc['prune']))]]),'后剪枝相对完整树提高了验证R²并减少叶节点，但本次网格最优R²仍低于课程深度3基线；因此不能将剪枝有效误写为后剪枝优于预剪枝。表中树深和叶节点数均为五折均值。\n']
 out+=['### 正则化系数与稳定性\n系数基于训练折标准化后的量纲；下表为五折均值±标准差。alpha趋零与OLS对照；Lasso精确零数量见扫描JSON。岭回归在本次拟合中没有稀疏选择功能，但“任何数据下系数绝不等于零”不是数学定理。高度相关输入可能使Lasso选中变量随折变化；相关图和逐折系数是判断依据，不能只凭单次系数排序。\n',table(['特征','OLS','Ridge','Lasso'],[[name]+[f'{np.mean(np.array(r["coefficients"][n])[:,i]):.6f} ± {np.std(np.array(r["coefficients"][n])[:,i]):.6f}' for n in ['ols','ridge','lasso']] for i,name in enumerate(names)]),'## 6. 独立选参检查\n',table(['模型','课程扫描最优','外层独立成绩','外层参数'],[[n,best(sc[n])['mean']['auc' if n in ['linear','rbf'] else 'r2'],r['nested'][n]['mean']['auc' if n in ['linear','rbf'] else 'r2'],str(r['nested'][n]['choices'])] for n in sc]),'我从这张表里读出：选参表现与外层表现的差距体现选择偏差和样本不确定性，不以外层结果再挑参数并宣称无偏。\n','## 7. 四项消融\n']
 for prefix,title,models in [('drop_','去除ca、thal',['linear','rbf','depth','ridge','lasso','svr']),('unscaled_','去除标准化',['linear','rbf'])]:
  rows=[]
  for n in models:
   key='auc' if n in ['linear','rbf'] else 'r2';a=best(ab[prefix+n]);rows.append([n,best(sc[n])['mean'][key],a['mean'][key],str(a['params'])])
  out += [f'### {title}\n',table(['模型','原始扫描最优','消融扫描最优','消融参数'],rows)]
  if prefix=='drop_':out +=[table(['模型','原固定参数','删除特征固定参数'],[[n,r['fixed'][n]['mean']['auc' if n in ['linear','rbf'] else 'r2'],ab['drop_fixed_'+n]['mean']['auc' if n in ['linear','rbf'] else 'r2']] for n in models]),'我从这张表里读出：应同时看固定参数和重新选参的变化。删去两列改变了可用信息和模型方差，不同模型可以反向变化。全特征成绩是课程可比设置，11特征结果更贴近题目提出的低成本初筛场景；即使高分也不能证明模型可直接用于真实临床。\n']
  else:out+=['我从这张表里读出：标准化改变距离和惩罚的有效尺度，最优C可能漂移；RBF的scale还会随总体方差改变，不能把差异全归因于C。树按单变量排序分裂，正线性缩放通常保持候选分组。\n']
 out +=['### 单次划分与模型容量\n',table(['实验','数值'],list(ab['single'].items())),f"固定深度3五折R²={r['fixed']['depth']['mean']['r2']:.6f}。十次单次划分的波动说明一次好成绩不能稳定排序模型。\n",table(['模型','R²','MAE','RMSE','选定参数'],[[n]+[a['mean'][k] for k in ['r2','mae','rmse']]+[str(a['params'])] for n,a in [('ridge',best(sc['ridge'])),('poly',best(ab['poly']))]]),'我从这张表里读出：二次特征增加至104维并扩大估计方差，正则化能缓解；是否有益以本表和验证结果判断，与深树都涉及容量和小样本的权衡。\n']
 # 误差逆序：保留逐样本审计表，不先假定教材示例必然复现。
 runs=sc['depth']['runs'];pairs=[(a,b) for a in runs for b in runs if a['mean']['r2']>b['mean']['r2'] and a['mean']['mae']>b['mean']['mae']];assert pairs
 a,b=pairs[0];errors=[]
 for z in [a,b]:
  pp=z['predictions'];e=np.array([abs(v['truth']-v['prediction']) for v in pp]);mask=e>1
  errors.append([str(z['params']),z['mean']['r2'],z['mean']['mae'],int(mask.sum()),float((e[mask]**2).sum()/(e**2).sum()),float(e[mask].sum()/e.sum()),pp[int(np.argmax(e))]['row'],float(e.max())])
 out+=['## 8. 误差与排名冲突\n',table(['参数','R²','MAE','误差>1人数','平方误差占比','绝对误差占比','最大误差行号','最大误差'],errors),'我从这张表里读出：R²对大误差更敏感；本实验以R²选回归参数，同时给出MAE排序。若改以MAE选参，树的最佳参数为 '+str(min(runs,key=lambda z:z['mean']['mae'])['params'])+'；按R²为 '+str(best(sc['depth'])['params'])+'。这不等于所有模型中的冠军必然变化。\n']
 out+=['## 9. 讨论题\n','## 10. 实验日志\n',make_log(r,fig),'## 11. 局限、材料勘误与AI声明\n']
 from discussion_review import build
 content=build('\n'.join(out),r,table,data,fig)
 import pandas as pd
 pd.DataFrame([dict(model=n,**v) for n,s in sc.items() for v in best(s)['predictions']]).to_csv(fig/'predictions.csv',index=False)
 # 固定折置换重要性比较
 imp={n:np.mean(v,axis=0) for n,v in r['importance'].items()}
 rows=[[name]+[float(imp[n][i]) for n in imp] for i,name in enumerate(names)]
 content+='\n## 附录：两条任务线的重要特征\n'+table(['特征']+list(imp),rows)
 from report_extras import enrich
 from discussion_review import finish_tables
 return finish_tables(enrich(content,r,table,data),r,table,data)

def make_log(r,figures):
 from unexpected_findings import render
 return render(r,table,figures)

def export_tables(content,path):
 tables=['# 实验结果表\n']
 section=''
 for line in content.splitlines():
  if line.startswith('#'):
   section=line
  elif line.startswith('|'):
   if not tables[-1].startswith('|'):
    tables.extend(['', '## '+section.lstrip('# '), ''])
   tables.append(line)
  elif tables[-1].startswith('|'):
   tables.append('')
 path.write_text('\n'.join(tables)+'\n',encoding='utf-8')

def main():
 parser=argparse.ArgumentParser(description='复现固定五折实验、统计表和图')
 parser.add_argument('--data-dir', help='课程数据目录')
 parser.add_argument('--output-dir', type=Path, default=ROOT/'figures', help='图表和实验数据目录')
 args=parser.parse_args()
 figures=args.output_dir.resolve()
 figures.mkdir(parents=True,exist_ok=True)
 plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
 r=run(args.data_dir,figures)
 content=render_tables_and_figures(r,args.data_dir,figures)
 export_tables(content,figures/'tables.md')
 path=figures/'results.json'
 temporary=path.with_suffix('.tmp')
 temporary.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item()),encoding='utf-8')
 temporary.replace(path)
 print(f'实验图表及结果已生成：{figures}',flush=True)

if __name__=='__main__':
 main()
