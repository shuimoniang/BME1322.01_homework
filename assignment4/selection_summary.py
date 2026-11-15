"""依据正式实验记录生成参数选择依据和基线差值。"""
NAMES={'linear':'SVM（线性核）','rbf':'SVM（RBF核）','depth':'回归树（预剪枝）','prune':'回归树（后剪枝）','ridge':'岭回归','lasso':'Lasso回归','ols':'普通最小二乘线性回归','svr':'支持向量回归（SVR）'}
BASE={'linear':'C=1','rbf':"C=1，gamma=scale",'depth':'max_depth=3','ridge':'alpha=1','lasso':'alpha=0.01','ols':'带截距，无待扫描超参数','svr':'C=1，epsilon=0.1，gamma=scale'}
def best(scan):return scan['runs'][scan['best']]
def signed(value):
 return f'{0.0 if abs(value)<5e-9 else value:+.8f}'
def comparisons(r,classification,table):
 from report_presentation import model_summary
 return model_summary(r,classification,table)

def reason(r,name,table):
 scan=r['scans'][name];a=best(scan);key='auc' if name in ['linear','rbf'] else 'r2';metric='AUC' if key=='auc' else 'R²'
 if name=='ols':return ''
 ref=r['fixed'].get(name)
 params=str(a['params']);content=f'参数选择依据：以固定五折平均验证{metric}最大为准则，选取`{params}`，验证{metric}={a["mean"][key]:.8f}。'
 rest=[z for i,z in enumerate(scan['runs']) if i!=scan['best']]
 runner=max(rest,key=lambda z:z['mean'][key]);gap=a['mean'][key]-runner['mean'][key]
 content+=f'次高候选为`{runner["params"]}`，验证{metric}={runner["mean"][key]:.8f}，差值={signed(gap)}；该差距用于判断是否存在明显优势，不能解释为统计显著性。'
 if name=='linear':content+='小C对应较强正则化；本次较强约束取得最高验证AUC，选择依据是验证成绩而非训练拟合程度。'
 if name=='rbf':content+='所选C位于扫描上边界，gamma位于数值网格下边界，因此仅能称为当前网格最优，尚不能证明已定位更广范围内的最优点。'
 if name=='depth':content+='选中的深度与课程固定基线相同，故本次扫描没有带来提升；继续加深的结果和叶内样本数见本表，不能只因树更复杂便优先选择。'
 if name=='prune':
  alphas=[z['ccp_alpha'] for z in a['folds']]
  content+='path_fraction表示训练折剪枝路径的位置，而不是ccp_alpha本身；所选位置对应五个训练折的实际ccp_alpha分别为'+str([round(v,8) for v in alphas])+'。各折均只用自己的训练数据生成路径，完整树与课程深度3树的两种差值已在总表分别报告。'
 if name=='ridge':content+='选择alpha时优先考虑验证误差，alpha更大不保证表现更好；与固定alpha=1的差值用于判断增强收缩是否实际有益。'
 if name=='lasso':content+='所选alpha与课程基线相同，因而没有参数选择带来的提升；数值微差源于扫描中使用了更严格的求解容差和迭代上限，应视为求解精度差异。'
 if name=='svr':content+='通过C、epsilon、gamma的联合网格选择，epsilon控制不敏感带，gamma控制核的局部性；该设置因验证R²最高被选中，不以训练R²或MAE单独选参。'
 if ref:
  content+=f'相对固定基线`{BASE[name]}`，{metric}变化为{signed(a["mean"][key]-ref["mean"][key])}；其余指标的变化见总表。'
 return content+'\n'

def small_alpha_note(r,table):
 small=min(r['scans']['ridge']['runs'],key=lambda z:z['params']['alpha'])
 ols=best(r['scans']['ols']);alpha=small['params']['alpha']
 ridge_pred={z['row']:z['prediction'] for z in small['predictions']}
 ols_pred={z['row']:z['prediction'] for z in ols['predictions']}
 assert sorted(ridge_pred)==sorted(ols_pred)==list(range(303))
 maximum=max(abs(ridge_pred[i]-ols_pred[i]) for i in ridge_pred)
 rows=[[label]+[f'{a["mean"][k]:.8f}' for k in ['r2','mae','rmse']] for label,a in [(f'岭回归（alpha={alpha}）',small),('普通最小二乘线性回归',ols)]]
 return (f'#### 扫描首个取值：小alpha岭回归与普通最小二乘的对应关系\n\n本节按alpha从小到大排列，首个取值为{alpha}。岭回归在残差平方和之外加入alpha加权的系数平方惩罚；当alpha足够小时，惩罚作用很弱，解趋近于无正则化的普通最小二乘解。因此，按照作业要求，可以将扫描首个小alpha的岭回归结果作为近似普通线性回归的参照，而不必单独运行普通线性回归。本实验额外运行普通最小二乘，用相同预处理和固定五折检验这一近似：\n\n'
  +table(['模型','验证R²','验证MAE','验证RMSE'],rows)
  +f'\n两者303例折外预测的最大绝对差仅为{maximum:.8f}个等级，五折汇总指标也几乎一致，说明在本数据及当前标准化尺度下，alpha={alpha}时岭回归已近似普通最小二乘。这里的小alpha取值用于解释线性参照，并不是本次验证R²最高的alpha；最终选定alpha仍依据完整扫描结果。\n')

def kernel_conclusion(r):
 linear=best(r['scans']['linear'])['mean'];rbf=best(r['scans']['rbf'])['mean']
 fixed_l=r['fixed']['linear']['mean'];fixed_r=r['fixed']['rbf']['mean']
 outer_l=r['nested']['linear']['mean'];outer_r=r['nested']['rbf']['mean']
 return ('### 两种核的适用性结论\n\n'
  +f'以AUC为主要评价指标，本数据集当前实验更倾向选择线性核。固定C=1时，线性核AUC为{fixed_l["auc"]:.6f}，高于RBF核的{fixed_r["auc"]:.6f}；分别调参后，两者为{linear["auc"]:.6f}和{rbf["auc"]:.6f}，线性核仍高出{linear["auc"]-rbf["auc"]:.6f}。独立选参检查的外层平均AUC分别为{outer_l["auc"]:.6f}和{outer_r["auc"]:.6f}，方向一致。\n\n'
  +f'最优配置的训练AUC减验证AUC，线性核为{linear["train_auc"]-linear["auc"]:.6f}，RBF核为{rbf["train_auc"]-rbf["auc"]:.6f}。RBF具有更强的局部拟合能力，但本次没有取得更好的验证排序能力，且训练与验证差距更大，支持在这份小样本数据上优先考虑简单线性分界的选择；这不能证明真实关系完全线性。\n\n'
  +f'这一选择并非所有指标上的全面胜出：默认判别阈值下，RBF核召回率为{rbf["recall"]:.6f}，高于线性核的{linear["recall"]:.6f}。若目标改为优先降低漏诊，需要另行明确召回率约束并在训练数据内选择阈值，不能仅凭AUC决定。调参后的AUC差距较小，且RBF最优参数位于网格边界，因此结论限于当前数据、扫描范围和评价口径，不声称统计显著或全局最优。\n')

def rbf_gamma_analysis(r,table):
 import numpy as np
 import matplotlib.pyplot as plt
 from common import ROOT
 scan=r['scans']['rbf'];chosen=best(scan)
 rows=[a for a in scan['runs'] if a['params']['C']==1]
 numeric=sorted([a for a in rows if isinstance(a['params']['gamma'],(int,float))],key=lambda a:a['params']['gamma'])
 at={a['params']['gamma']:a['mean'] for a in numeric}
 fig,ax=plt.subplots(figsize=(8,4.8))
 x=[a['params']['gamma'] for a in numeric]
 ax.semilogx(x,[a['mean']['train_auc'] for a in numeric],'o-',label='训练AUC')
 ax.semilogx(x,[a['mean']['auc'] for a in numeric],'o-',label='五折验证AUC')
 ax.set(xlabel='gamma（对数尺度）',ylabel='AUC',title='RBF核：固定C=1时gamma的影响');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures/rbf_gamma.png',dpi=180);plt.close(fig)
 comparison=table(['gamma','训练AUC','五折验证AUC','差距（训练−验证）','验证accuracy','平均支持向量数'],[[a['params']['gamma'],a['mean']['train_auc'],a['mean']['auc'],a['mean']['train_auc']-a['mean']['auc'],a['mean']['accuracy'],a['mean']['support_vectors']] for a in rows])
 return ('#### RBF核的联合网格及gamma过大的验证\n\n'
  +'C取{0.001, 0.01, 0.1, 1, 10, 100}，gamma取{scale, 0.001, 0.01, 0.1, 1, 10}，形成36组组合，超过题目对两个参数各至少4个取值的要求。第四节AUC热力图横轴为gamma、纵轴为C，完整扫描表对每组组合同时列出训练和五折验证AUC。\n\n'
  +f'以平均验证AUC最大选取C={chosen["params"]["C"]}、gamma={chosen["params"]["gamma"]}，验证AUC={chosen["mean"]["auc"]:.6f}，训练AUC={chosen["mean"]["train_auc"]:.6f}；选择依据是验证排序能力，而不是追求训练AUC=1。该参数处于网格边界，仅代表已遍历范围内的最优。\n\n'
  +'为单独考察gamma而不混入C变化，下面固定C=1，沿用相同数据、折内预处理和五折划分；scale作为自适应宽度参考，数值gamma按从小到大绘制。\n\n'+comparison
  +'\n![固定C时gamma的影响](figures/rbf_gamma.png)\n'
  +f'当gamma从0.01增至0.1、1、10时，训练AUC依次为{at[.01]["train_auc"]:.6f}、{at[.1]["train_auc"]:.6f}、{at[1]["train_auc"]:.6f}、{at[10]["train_auc"]:.6f}，验证AUC却从{at[.01]["auc"]:.6f}降至{at[.1]["auc"]:.6f}、{at[1]["auc"]:.6f}和{at[10]["auc"]:.6f}。gamma=10时训练与验证AUC差距达到{at[10]["train_auc"]-at[10]["auc"]:.6f}，验证accuracy降为{at[10]["accuracy"]:.6f}，已接近多数类参考线。由此直接验证了题目所述“训练AUC达到1.000，而新样本排序表现明显下降”的过拟合现象；该变化不意味着gamma在全部范围上越小越好，例如0.001到0.01时验证AUC仍有提升。\n\n'
  +'RBF核定义为K(x,x′)=exp(−gamma·‖x−x′‖²)，其中距离基于当前训练折拟合的标准化输入。对于不同样本，gamma越大，相似度随距离衰减越快；相同样本始终有K(x,x)=1，而其他样本对它的影响越来越弱。模型于是更容易围绕训练点形成高度局部的响应，记住训练样本，却难以把这种拟合推广到未见患者。支持向量数在gamma=10时接近每折训练样本数，也与多数训练点参与局部拟合的解释一致。\n\n'
  +'题目用“孤岛”和“退化为最近邻”描述极端局部化的直觉。本实验的AUC差距支持这一过拟合机制，但未绘制高维决策边界，不能声称直接观察到其形状，也不能将核SVM严格等同于最近邻分类器；最终决策仍取决于支持向量系数、C和截距。gamma=10时AUC仍高于0.5而accuracy已接近多数类水平，也表明连续分数的排序能力与默认阈值下的分类决策需要分开解释。\n')
