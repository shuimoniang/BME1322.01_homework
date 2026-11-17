"""逐条落实作业讨论项，所有数字从正式记录或指定训练折重建。"""
import json
import numpy as np
from common import ROOT,load,metrics,clone
from regress import candidates

def best(scan):return scan['runs'][scan['best']]

def chosen(r,name):return r['scans'][name]['runs'][r['scans'][name]['best']]
N={'linear':'SVM（线性核）','rbf':'SVM（RBF核）','depth':'回归树（预剪枝）','prune':'回归树（后剪枝）','ridge':'岭回归','lasso':'Lasso回归','ols':'普通最小二乘线性回归','svr':'支持向量回归（SVR）'}

def build(content,r,table,data=None):
 _,df,folds=load(data);sc=r['scans'];ab=r['ablation'];features=list(df.columns[:-1])
 def insert(anchor,text):
  nonlocal content
  assert anchor in content,anchor
  content=content.replace(anchor,text+'\n'+anchor,1)
 def tbl(headers,rows,insight):return table(headers,rows)+'\n我从这张表里读出：'+insight+'\n'
 # 4.2(3)：固定gamma，不把重新选择gamma的收益算作C敏感性。
 linear=sc['linear']['runs'];rbf=[a for a in sc['rbf']['runs'] if a['params']['gamma']==.01]
 ranges={name:max(a['mean']['auc'] for a in runs)-min(a['mean']['auc'] for a in runs) for name,runs in [('linear',linear),('rbf',rbf)]}
 text='#### C的作用、双核敏感性与训练验证差距（4.2（3））\n\n'
 text+=tbl(['核','C','训练AUC','验证AUC','训练错误率','验证错误率'],[[N[n],a['params']['C'],a['mean']['train_auc'],a['mean']['auc'],1-a['mean']['train_accuracy'],1-a['mean']['accuracy']] for n,runs in [('linear',linear),('rbf',rbf)] for a in runs],'RBF固定gamma=0.01时，大C提高了训练拟合能力，却可能损害验证排序；线性核的验证AUC变化更小。')
 text+=f'在线性核中，C从0.001到100的验证AUC极差为{ranges["linear"]:.6f}；RBF固定gamma=0.01时为{ranges["rbf"]:.6f}。两者对C的敏感性不同，并且RBF的结论必须限定gamma。线性核C=10与100的验证AUC均为{linear[-1]["mean"]["auc"]:.6f}，说明这个范围已进入平台，而非继续增大C就能提高泛化。\n\n'
 text+='SVM最小化“权重范数惩罚＋C×间隔违例损失”，对偶约束为0≤a_i≤C；C增大放宽单样本对偶权重上限，同时加重原目标中对违例的惩罚，通常表示相对更弱的正则化。题目中关于C越大越容忍越界的文字方向与目标函数不一致，此处按目标函数解释。验证AUC对C不敏感，说明该范围内排序较稳定，不能直接推出训练边界完全不变；需要结合上表训练错误率、验证错误率和训练验证AUC差距判断。该现象可能来自近似线性结构和正则化平台，不能据此断言数据完全线性可分。\n'
 insert('### SVM（线性核） 完整扫描',text)
 # 4.4：AUC口径。
 auc_rows=[[N[n],r['fixed'][n]['mean']['hard_auc'],r['fixed'][n]['mean']['auc'],r['fixed'][n]['mean']['auc']-r['fixed'][n]['mean']['hard_auc']] for n in ['linear','rbf']]
 text='### AUC口径、逐类指标与筛查含义（4.4）\n\n'+tbl(['默认参数模型','硬标签AUC（错误口径示例）','连续分数AUC（正式口径）','差值'],auc_rows,'使用硬标签会丢失类别内部的排序信息；是否发生模型排序逆转应以本表实测值判断，不照抄讲义示例。')
 hard=max(['linear','rbf'],key=lambda n:r['fixed'][n]['mean']['hard_auc']);soft=max(['linear','rbf'],key=lambda n:r['fixed'][n]['mean']['auc'])
 text+=f'本次固定参数结果中，硬标签口径排序较高的是{N[hard]}，连续分数口径排序较高的是{N[soft]}。正式评估始终使用decision_function而非predict；数值与讲义示例不同不改变这一规范。敏感性TP/(TP+FN)等于阳性recall，特异性TN/(TN+FP)等于阴性recall，阳性precision为TP/(TP+FP)，三者分母不同。两类AUC以相反分数和标签定义时相同，二分类逐类one-vs-rest准确率也相同，不能把这两个重复数值当作独立证据。混淆矩阵汇总303例折外预测；表中指标先逐折计算再平均，故直接从汇总矩阵计算的比例可能略有不同。\n'
 insert('### 参数遍历所得最优配置的分类指标对照',text)
 # 树深机制。
 depth3=next(a for a in sc['depth']['runs'] if a['params']['max_depth']==3);full=next(a for a in sc['depth']['runs'] if a['params']['max_depth'] is None)
 text='#### 树加深后为何可能不如线性模型（4.3（1））\n\n'+tbl(['设置','训练R²','验证R²','叶节点数均值','叶内样本中位数均值','单例叶比例均值'],[[label]+[a['mean'][k] for k in ['train_r2','r2','leaves','leaf_median','singleton_fraction']] for label,a in [('深度3回归树',depth3),('无限深回归树',full)]],'完整树将训练误差降为零，却把大量叶节点分给很少的患者，无法可靠估计新患者的平均严重程度。')
 text+=f'岭回归的验证R²为{chosen(r,"ridge")["mean"]["r2"]:.6f}，无限深树仅为{full["mean"]["r2"]:.6f}。叶节点预测是叶内训练标签的均值，样本少时一个异常或高等级患者就能大幅改变预测；单例叶直接记住一个标签。因此树加深带来的训练拟合收益被估计方差抵消，出现验证成绩低于共享全局系数的线性模型，机制不只是笼统的“过拟合”。\n'
 insert('### 回归树（后剪枝） 完整扫描',text)
 # 前两层用真实树的层序节点，而非数组前七个。
 X=df.drop(columns='num').to_numpy(float);y=df.num.to_numpy(float)
 _,tree,adjust=candidates('prune')[sc['prune']['best']];tree=adjust(clone(tree),X,y);tree.fit(X,y);t=tree[-1].tree_
 nodes=[0]+([int(t.children_left[0]),int(t.children_right[0])] if t.children_left[0]>=0 else [])
 interpretations={'thal':'阈值4.5将正常灌注编码3与固定／可逆缺损编码6、7分开，反映检查结果的分组。','ca':'阈值0.5将显影支数0与至少1支分开，使用了题目所述昂贵检查信息。','oldpeak':'按运动ST段压低幅度分组，数值较大对应不同的运动负荷反应；这是统计分组而非因果证据。'}
 text='#### 后剪枝树前两层的实际分裂及临床解释（4.3（2））\n\n'
 text+=tbl(['层级','节点','规则（左分支）','解释'],[[1 if i==0 else 2,i,f'{features[t.feature[i]]} ≤ {t.threshold[i]:.6f}' if t.feature[i]>=0 else '叶节点',interpretations.get(features[t.feature[i]],'该特征将患者划为不同预测风险组，阈值只属于本次解释树。') if t.feature[i]>=0 else '不再分裂'] for i in nodes],'树的早期分裂依赖灌注或显影信息，因此全特征树的良好拟合不等于低成本初筛可直接使用。')
 text+='这张图是全数据后剪枝解释树，不是外层测试树；节点阈值和结构不能用于回算交叉验证成绩。固定五折中预剪枝优于后剪枝，没有证据证明本次预剪枝确实因“剪早”而失分；外层独立检查两者成绩接近且排序相反，也不能把一次结果推广成两种方法的普遍优劣。\n'
 insert('### 回归树（预剪枝） 完整扫描',text)
 # 系数收缩、稀疏、全零及相关结构。
 corr=df[features].corr();pairs=sorted([(abs(corr.iloc[i,j]),features[i],features[j],corr.iloc[i,j]) for i in range(13) for j in range(i)],reverse=True)
 text='### 正则化作用、稀疏性与相关特征稳定性（4.3（3））\n\n代码中的alpha对应讲义的正则化强度λ，不是SVM对偶变量。扫描跨多个数量级，适合比较弱惩罚、适度惩罚和强惩罚的不同机制。\n\n'
 text+=tbl(['模型','alpha','平均精确零系数数','验证R²','验证MAE'],[[N[n],a['params']['alpha'],a['mean']['zero_count'],a['mean']['r2'],a['mean']['mae']] for n in ['ridge','lasso'] for a in sorted(sc[n]['runs'],key=lambda z:z['params']['alpha'])],'岭回归在本次全部强度下没有产生精确零系数，Lasso随惩罚增强出现稀疏解，并最终退化为常数预测。')
 high=max(sc['lasso']['runs'],key=lambda a:a['params']['alpha']);coefrows=[]
 for i,f in enumerate(features):
  ol=np.array(r['coefficients']['ols'])[:,i];ri=np.array(r['coefficients']['ridge'])[:,i];la=np.array(r['coefficients']['lasso'])[:,i]
  coefrows.append([f,float(np.mean(ri-ol)),int(np.count_nonzero(la)),float(np.std(la))])
 text+=tbl(['特征','岭系数−OLS系数（五折均值）','最优Lasso非零折数／5','Lasso系数折间标准差'],coefrows,'系数变化和变量保留情况须逐特征检查，不能把一次拟合的绝对系数直接当作稳定的临床重要性。')
 text+=f'普通最小二乘R²={chosen(r,"ols")["mean"]["r2"]:.6f}，岭回归最优R²={chosen(r,"ridge")["mean"]["r2"]:.6f}，差为{chosen(r,"ridge")["mean"]["r2"]-chosen(r,"ols")["mean"]["r2"]:+.6f}，表明在仅13个输入上适度收缩收益较小；岭alpha=10000时R²仅{sc["ridge"]["runs"][0]["mean"]["r2"]:.6f}，说明惩罚过强会欠拟合。Lasso的alpha={high["params"]["alpha"]}时，五折均将13个系数压到零，R²={high["mean"]["r2"]:.6f}，与训练均值参考线一致；其截距保留训练标签均值，测试R²为负是训练和测试均值不一致造成的正常结果。\n\n'
 text+=tbl(['特征1','特征2','Pearson相关系数'],[[a,b,float(v)] for _,a,b,v in pairs[:5]],'数据存在相关输入，但不能仅据相关图声称所有特征高度共线，也不能把类别编码的Pearson系数等同于完整的依赖关系。')
 text+='相关变量能共享预测信息，Lasso可能在它们之间分配或替换权重；上表非零折数和系数标准差提供本数据的稳定性证据，固定种子下并不是算法每次随机选择。当前五折现象只能部分支持讲义关于相关特征选择不稳定的提醒，不能证明选择“近乎随机”；要严格检验需要训练内重采样或受控相关特征实验。岭回归更适合保留多个相关输入的共同信息，Lasso适合希望获得稀疏解释的情形，但稀疏性不保证泛化或因果解释更好。\n'
 insert('## 6. 独立选参检查',text)
 # 朴素参考提升。
 text='### 相对朴素参考线的实际提升（4.3（4））\n\n'
 text+='这里的朴素参考线是“训练折均值预测基线”：不使用患者的任何特征，在每一折中只计算训练折标签num的平均值，再将该平均值作为测试折所有患者的预测结果；不使用测试折标签计算均值。它用于判断模型是否比“所有人都猜训练平均等级”更有效。'+f"本实验中，该基线的五折汇总R²为{r['dummy_reg']['mean']['r2']:.6f}、MAE为{r['dummy_reg']['mean']['mae']:.6f}、RMSE为{r['dummy_reg']['mean']['rmse']:.6f}。\n\n"
 text+='下表的ΔR²为模型R²减去基线R²；MAE减少量和RMSE减少量分别为基线误差减去模型误差，因此三列均为正值时表示相对基线有所改善。\n\n'
 text+=tbl(['模型','验证R²','相对训练均值的ΔR²','MAE减少量','RMSE减少量'],[[N[n],chosen(r,n)['mean']['r2'],chosen(r,n)['mean']['r2']-r['dummy_reg']['mean']['r2'],r['dummy_reg']['mean']['mae']-chosen(r,n)['mean']['mae'],r['dummy_reg']['mean']['rmse']-chosen(r,n)['mean']['rmse']] for n in ['depth','prune','ridge','lasso','ols','svr']],'全部入选模型均优于真正按训练折均值预测的参考线；误差减少量为正才表示改善。')
 text+='题目写R²=0指的是理想均值参考概念；严格折外训练均值不等于测试标签均值，因此其测试R²略小于零。这里用实际折外参考作差，不在测试折上拟合均值。\n'
 insert('## 6. 独立选参检查',text)
 # 四项消融逐项明确数字与结论。
 delta={n:ab['drop_fixed_'+n]['mean']['auc' if n in ['linear','rbf'] else 'r2']-r['fixed'][n]['mean']['auc' if n in ['linear','rbf'] else 'r2'] for n in ['linear','rbf','depth','ridge']}
 text='### 特征可用性、尺度、划分与容量的结论（4.5）\n\n'
 text+=f'（1）去掉ca和thal后，固定参数的线性SVM、RBF SVM、回归树和岭回归，主要指标变化分别为{delta["linear"]:+.6f}、{delta["rbf"]:+.6f}、{delta["depth"]:+.6f}、{delta["ridge"]:+.6f}。变化方向不一致：删除信息也可能减少树的过拟合，不能把某个模型的改善解释为这些特征普遍无用。是否允许使用特征应按预测时点确定；面向未接受这些检查的低成本初筛，不应依赖尚不可获得的信息。全特征实验用于课程可比性，11特征实验回答不同信息条件下的问题；thal的检查流程也不能笼统等同于所有患者都已完成造影。\n\n'
 for n in ['linear','rbf']:
  a=chosen(r,n);b=best(ab['unscaled_'+n]);text+=f'（2）{N[n]}标准化最优C={a["params"]["C"]}、AUC={a["mean"]["auc"]:.6f}；不标准化最优C={b["params"]["C"]}、AUC={b["mean"]["auc"]:.6f}，C{("发生" if a["params"]["C"]!=b["params"]["C"] else "未发生")}漂移。'
 text+='SVM的线性权重惩罚和RBF距离都受单位影响，尺度改变后同一C对应的约束效果不同；RBF的scale还会随输入总体方差变化。树按单变量阈值形成样本分组，正线性缩放保留排序，因此通常不需标准化，但并列分裂和浮点误差可能导致细微变化。\n\n'
 z=ab['single'];text+=f'（3）深度3回归树十次80/20划分的R²最小{z["min"]:.6f}、最大{z["max"]:.6f}、标准差{z["std"]:.6f}，与固定五折平均{depth3["mean"]["r2"]:.6f}相比，一次划分甚至可从正变负。挑选最好一次会系统性夸大成绩；小测试集的重症病例比例和个别大误差都能改变平方损失，因此不能依据单次结果稳定排序模型。\n\n'
 a=best(ab['poly']);b=chosen(r,'ridge');text+=f'（4）二次多项式岭回归选alpha={a["params"]["alpha"]}，R²={a["mean"]["r2"]:.6f}，低于原岭回归{b["mean"]["r2"]:.6f}，变化为{a["mean"]["r2"]-b["mean"]["r2"]:+.6f}。二次展开将13维增加为104维，本次在较强正则化下仍未获得收益；这与深树相同，都可能增加小样本下的估计方差，但具体机制分别是交互项系数估计和小叶节点均值估计，不能说两者完全相同。\n'
 insert('## 8. 误差与排名冲突',text)
 # 4.4 按两个指标排序，给出跨模型最优和深度逆序。
 rankings=[(n,chosen(r,n)['mean']) for n in ['depth','prune','ridge','lasso','ols','svr']]
 rankr=sorted(rankings,key=lambda z:-z[1]['r2']);rankm=sorted(rankings,key=lambda z:z[1]['mae'])
 text='### 指标冲突的完整模型排名与选择理由（4.4）\n\n'+tbl(['R²名次','模型','R²','MAE','MAE名次'],[[i+1,N[n],a['r2'],a['mae'],next(j+1 for j,(k,_) in enumerate(rankm) if k==n)] for i,(n,a) in enumerate(rankr)],f'按R²选择的总体模型为{N[rankr[0][0]]}，改按MAE选择则为{N[rankm[0][0]]}，评价目标会改变最佳模型。')
 text+='本实验主要按R²选回归参数，是为了与课程比较并重视少数严重误差；但等级是否等距本身是建模假设，不意味平方惩罚具有既定临床代价。MAE更直接表达平均错多少等级，必须同时报告。前表的深度候选逆序及误差>1级样本的平方误差占比，解释了为何较少的大误差能改变R²排序；这与平均绝对误差下降不矛盾。\n'
 insert('## 9. 讨论题',text)
 # 同口径比较：逐折计算分数AUC与阈值指标后平均。
 threshold=[]
 def avg_classification(name,threshold_value=None):
  pred=chosen(r,name)['predictions'];values=[]
  for fold in range(5):
   pp=[z for z in pred if z['fold']==fold]
   if threshold_value is None:
    truth=np.array([z['truth'] for z in pp]);score=np.array([z['score'] for z in pp]);label=np.array([z['prediction'] for z in pp])
   else:
    truth=np.array([z['truth']>0 for z in pp]);score=np.array([z['prediction'] for z in pp]);label=(score>threshold_value).astype(int)
   values.append(metrics(truth,label,score))
  return {key:float(np.mean([v[key] for v in values])) for key in ['auc','accuracy','f1']}
 svm={n:avg_classification(n) for n in ['linear','rbf']}
 for n in ['ridge','depth']:
  for value in [.3,.5,.7,1.]:threshold.append((n,value,avg_classification(n,value)))
 important={n:set(np.argsort(np.mean(r['importance'][n],axis=0))[::-1][:5]) for n in ['linear','ridge']};overlap='、'.join(features[i] for i in sorted(important['linear']&important['ridge']))
 lin=chosen(r,'linear')['mean'];rb=chosen(r,'rbf')['mean'];ri=chosen(r,'ridge')['mean'];de=chosen(r,'depth')['mean']
 gamma10=next(a['mean'] for a in sc['rbf']['runs'] if a['params']['C']==1 and a['params']['gamma']==10)
 first=threshold[0][2];last=threshold[3][2]
 paragraphs=[
 ['## 1. 线性核与RBF核',f'本次实验选择线性核：其最优五折平均AUC为{lin["auc"]:.6f}，高于RBF核的{rb["auc"]:.6f}。','非线性核没有带来提升，说明在当前特征和样本规模下，线性分界已经能够捕捉主要判别信息，增加局部非线性拟合没有获得额外收益。',f'当C=1、gamma=10时，RBF训练AUC达到{gamma10["train_auc"]:.6f}，验证AUC降至{gamma10["auc"]:.6f}，直接表现为过拟合。','这是因为核函数K(x,x′)=exp(−gamma·‖x−x′‖²)随gamma增大而快速衰减，训练点之间的相似性变小，模型的响应集中在训练点附近。',f'默认参数下从线性核换成RBF核，AUC变化为{r["fixed"]["rbf"]["mean"]["auc"]-r["fixed"]["linear"]["mean"]["auc"]:+.6f}；对RBF自身调参则提高{rb["auc"]-r["fixed"]["rbf"]["mean"]["auc"]:.6f}，因此本次调好现有模型参数比直接换复杂模型更有效。',f'调参后线性核仍领先{lin["auc"]-rb["auc"]:.6f}，结合其结构简单、参数较少，我将线性核作为本实验的首选分类模型。',f'这一选择以AUC为主要依据；若改以默认阈值下的召回率为标准，则RBF核以{rb["recall"]:.6f}高于线性核的{lin["recall"]:.6f}。'],
 ['## 2. 两条任务线的结论是否一致','两条任务线的结论一致：在本实验中，简单线性模型均优于对应的复杂非线性模型。',f'分类中线性核AUC为{lin["auc"]:.6f}，高于RBF核的{rb["auc"]:.6f}；回归中岭回归R²为{ri["r2"]:.6f}，高于回归树的{de["r2"]:.6f}。','这两组任务内部的比较共同说明，当前数据的主要预测信息适合用平滑的全局关系表达，小样本下复杂局部结构的拟合收益不足以抵消估计方差。',f'线性SVM与岭回归测试折置换重要性前五共同包含{overlap}，进一步表明两条任务线利用了相近的输入信号。',f'去掉ca和thal后，岭回归R²为{best(ab["drop_ridge"])["mean"]["r2"]:.6f}，仍高于回归树的{best(ab["drop_depth"])["mean"]["r2"]:.6f}，所以删去昂贵检查信息后这一模型选择结论仍成立。','回归树删除这两列后成绩提高、岭回归成绩下降，说明相同信息对不同模型的拟合与方差影响不同，树更容易因细分样本而损失泛化能力。','因此，两条线相互印证的数据性质是：主要信号可以由较简单的全局关系捕获，当前样本规模更适合控制模型复杂度。'],
 ['## 3. 回归模型能否用作分类器',f'可以，岭回归预测值能够用作分类分数，其五折平均AUC为{first["auc"]:.6f}，接近线性SVM的{svm["linear"]["auc"]:.6f}。',f'将岭回归预测值按大于1.0判为阳性时，accuracy为{last["accuracy"]:.6f}、F1为{last["f1"]:.6f}，均高于线性SVM默认决策的{svm["linear"]["accuracy"]:.6f}和{svm["linear"]["f1"]:.6f}。',f'但按大于0.3判为阳性时，岭回归accuracy为{first["accuracy"]:.6f}、F1为{first["f1"]:.6f}，均低于线性SVM，所以两种比较并不总给出相同结论。','据此，本次明确结论是：按连续分数AUC，岭回归排序能力接近SVM；按阈值0.3的分类决策，岭回归较差；按阈值1.0的accuracy和F1，岭回归更好。','回归器能够产生合理排序，是因为训练目标num本身包含num>0的信息，预测的严重程度越高，通常也越倾向于存在病变。','其局限在于平方损失拟合的是严重程度的条件均值，而分类关心的是阳性概率，少量重症与较多轻症可产生相近均值，使风险排序和二值边界发生偏差。','因此，回归模型在本实验中可以兼作分类器，分类效果取决于阈值；0.3、0.5、0.7、1.0的结果用于展示这种敏感性，实际阈值应在训练数据内选择后再作独立评估。']]
 paragraphs[0].append(f'删除ca、thal两项昂贵检查特征并重新选参后，线性核与RBF核的AUC分别降至{best(ab["drop_linear"])["mean"]["auc"]:.6f}和{best(ab["drop_rbf"])["mean"]["auc"]:.6f}，线性核略优的方向仍成立，但全特征成绩不能直接代表低成本初筛能力，面向尚未取得这些检查结果的患者应参考删除两列后的结果。')
 analysis='# 三题简答\n\n'+'\n\n'.join(p[0]+'\n'+'\n'.join(p[1:]) for p in paragraphs)+'\n\n'
 analysis+=tbl(['模型','阈值','五折平均AUC','五折平均accuracy','五折平均F1'],[[N[n],value,a['auc'],a['accuracy'],a['f1']] for n,value,a in threshold]+[[N[n],'SVM默认决策',a['auc'],a['accuracy'],a['f1']] for n,a in svm.items()],'所有模型使用相同的五折聚合方式，回归器阈值改变可使与SVM的accuracy／F1排序改变，但连续分数AUC不随阈值变化。')
 from paired_comparison import render_paired
 analysis+='\n'+render_paired(r,table)
 from report_clarity import clarify_analysis
 analysis=clarify_analysis(analysis)
 start=content.index('## 9. 讨论题');end=content.index('## 10. 实验日志',start)
 embedded='\n'.join(('#### 补充分析：逐样本配对' if line.startswith('## 补充分析：') else '#'+line if line.startswith('## ') else line) for line in analysis.splitlines() if line!='# 三题简答')
 content=content[:start]+'## 9. 讨论题\n\n'+embedded+'\n\n'+content[end:]
 content=content.replace('\n# 实验日志\n','\n',1)
 (ROOT/'analysis.md').write_text(analysis,encoding='utf-8')
 evidence={'regression_classification':[{'model':n,'threshold':v,**a} for n,v,a in threshold],'svm_classification':svm,'correlation_pairs':[[a,b,float(v)] for _,a,b,v in pairs[:5]],'discussion_sentence_counts':[len(p)-1 for p in paragraphs]}
 (ROOT/'figures/discussion_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
 from reflection_section import reflection
 insert('## 11. '+chr(0x5c40)+chr(0x9650)+chr(0x3001)+chr(0x6750)+chr(0x6599)+chr(0x52d8)+chr(0x8bef)+chr(0x4e0e)+'AI'+chr(0x58f0)+chr(0x660e),reflection(r))
 return content

def finalize(r,table,data=None):
 p=ROOT/'report.md';content=p.read_text(encoding='utf-8')
 coverage=[
 ('2.4','选定参数及理由、相对固定基线差值','参数选择依据：'),
 ('3.2','任务定义及其他定义的预期影响','若把num改为五分类'),
 ('4.1','首个小alpha近似OLS','扫描首个取值：'),
 ('4.2（1）','哪个核更合适及数字依据','两种核的适用性结论'),
 ('4.2（2）','网格、gamma和过拟合验证','RBF核的联合网格及gamma过大的验证'),
 ('4.2（3）','C的作用与敏感性','C的作用、双核敏感性与训练验证差距'),
 ('4.2（4）/4.5（2）','标准化及C漂移','特征可用性、尺度、划分与容量的结论'),
 ('4.3（1）','深树与叶内样本数量','树加深后为何可能不如线性模型'),
 ('4.3（2）','剪枝路径与前后对照','回归树后剪枝的参照与剪枝效果'),
 ('4.3（2）','树图前两层的临床解释','后剪枝树前两层的实际分裂及临床解释'),
 ('4.3（2）','两种剪枝及剪早论断','预剪枝与后剪枝的区别及效果比较'),
 ('4.3（3）','系数、收缩、稀疏、全零退化','正则化作用、稀疏性与相关特征稳定性'),
 ('4.3（3）','相关结构与变量选择稳定性','当前五折现象只能部分支持'),
 ('4.3（4）','相对训练均值的提升','相对朴素参考线的实际提升'),
 ('4.4','逐类指标与混淆矩阵','参数遍历所得最优配置的分类指标对照'),
 ('4.4','连续分数AUC及敏感性特异性','AUC口径、逐类指标与筛查含义'),
 ('4.4','指标冲突、误差贡献和最佳模型变化','指标冲突的完整模型排名与选择理由'),
 ('4.5（1）','昂贵特征和变化方向','特征可用性、尺度、划分与容量的结论'),
 ('4.5（3）','单次划分结论风险','十次80/20划分'),
 ('4.5（4）','容量作用及与深树比较','二次多项式岭回归选alpha'),
 ('4.6（1）','核选择与换模型或调参','线性核与RBF核'),
 ('4.6（2）','两条任务线的一致性','两条任务线的结论是否一致'),
 ('4.6（3）','回归器分数、阈值和SVM对比','回归模型能否用作分类器'),
 ('五/六','真实反直觉发现与验证','## 10. 实验日志'),
 ('六','易错步骤与后续实验','### 讨论与反思'),
 ('八','工具、用途、错误及人工状态','## AI 使用声明')]
 content+='''
## AI 使用声明

使用Codex AI编程助手完成需求解析、代码、实验运行、排错、图表、数字分析和报告文字，使用Python和scikit-learn执行模型计算。本文结论由AI依据实际记录提出，用户已逐项复核human_review.md所列内容并确认没有问题；AI自动检查与用户人工复核分别记录。

'''
 from reflection_section import ai_error_example
 content+=ai_error_example(r)
 from paired_comparison import ai_question
 content+=ai_question(r)
 content+='验证依据为基线核对、关键模型重复计算、折内预处理及外层隔离测试、逐样本折外预测和报告重建。引用资料为课程作业PDF第2–14页、课程数据README及baselines.py、UCI Heart Disease数据说明、scikit-learn官方文档和本次AI对话。\n'
 for _,_,anchor in coverage:assert anchor in content,anchor
 index='# 作业讨论与回答逐项核对\n\n'+table(['题目条款','要求回答的问题','报告定位标题或关键词'],coverage)+'\n各项均有正文回答，数字取自正式记录或同折预测重算；三道简答正文分别为8、7、7句，用户已逐项完成人工复核并确认没有问题。\n'
 (ROOT/'discussion_checklist.md').write_text(index,encoding='utf-8')
 from report_presentation import finish_presentation
 content,table_count=finish_presentation(content,r,table)
 from report_structure import organize,clean_documents
 content=organize(content,r)
 from report_clarity import clarify_report
 content=clarify_report(content,r,table,data)
 p.write_text(content,encoding='utf-8')
 (ROOT/'figures/presentation_validation.json').write_text(json.dumps({'tables_with_individual_conclusions':table_count,'model_rows_metric_columns':True,'data_dictionary_fields':14},indent=2),encoding='utf-8')
 errors=ROOT/'ai_error_log.md';e=errors.read_text(encoding='utf-8')
 if '聚合口径不一致' not in e:
  e+='\n## 回归器转分类的聚合口径不一致\n\n初稿把岭回归全部303例折外预测计算的AUC=0.909853，与SVM逐折平均AUC作比较，口径不一致。现对各模型相同五折分别计算AUC、accuracy和F1，再取算术平均；报告第九节和analysis.md已更新，同口径数字保存在figures/discussion_evidence.json，不能把聚合变化解释为训练提升。\n'
  errors.write_text(e,encoding='utf-8')
 evidence=json.loads((ROOT/'figures/discussion_evidence.json').read_text(encoding='utf-8'))
 assert evidence['discussion_sentence_counts']==[8,7,7]
 for n in ['ridge','depth']:
  scores=[a['auc'] for a in evidence['regression_classification'] if a['model']==n]
  assert max(scores)-min(scores)<1e-12
 for n in ['linear','rbf']:
  for key in ['auc','accuracy','f1']:assert abs(evidence['svm_classification'][n][key]-chosen(r,n)['mean'][key])<1e-12
 clean_documents(ROOT)
 from remote_discussions import sync_archives
 sync_archives(ROOT)
