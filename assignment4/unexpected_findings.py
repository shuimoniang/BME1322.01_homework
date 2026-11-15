"""从完整正式记录整理反直觉结果；不虚构事前预注册或因果验证。"""
import json
import numpy as np
from common import ROOT


def collect(r):
    sc=r['scans'];ab=r['ablation'];items=[]
    def best(s):return s['runs'][s['best']]
    def mean(n):return best(sc[n])['mean']
    def comp(model,metric,a,va,b,vb):
        return dict(model=model,metric=metric,condition_a=a,value_a=float(va),condition_b=b,value_b=float(vb))
    def add(title,expected,comparisons,hypothesis,done,result,pending):
        items.append(dict(id=len(items)+1,title=title,intuitive_expectation=expected,comparisons=comparisons,hypothesis=hypothesis,verification_completed=done,verification_result=result,further_verification=pending))
    lin=mean('linear');rbf=mean('rbf');ridge=mean('ridge');ols=mean('ols');tree=mean('depth');post=mean('prune');lasso=mean('lasso');svr=mean('svr')
    add('复杂核没有取得更好的分类排序','表达能力更强的RBF核可能超过线性核。',[
        comp('SVM','五折AUC','线性核最优',lin['auc'],'RBF核最优',rbf['auc']),
        comp('SVM','外层AUC','线性核',r['nested']['linear']['mean']['auc'],'RBF核',r['nested']['rbf']['mean']['auc'])],
        '在303例小样本中，全局平滑关系可能已足够；RBF增加的局部拟合方差抵消了表达能力收益。',
        '分别完成双核网格扫描，并用同一固定折进行嵌套选参；比较训练与验证差距。',
        '两种评估口径均倾向线性核，但扫描后差距较小，不能解释为显著或普遍优势。',
        '扩大RBF边界网格，并在保持外层隔离的条件下检验；需要独立数据才能判断可推广性。')
    low=next(a['mean'] for a in sc['rbf']['runs'] if a['params']['C']==1 and a['params']['gamma']==.01)
    high=next(a['mean'] for a in sc['rbf']['runs'] if a['params']['C']==1 and a['params']['gamma']==10)
    add('RBF训练排序完美，验证分类却接近多数类','更精细的局部拟合可能同时提高训练和验证表现。',[
        comp('RBF SVM，C=1','训练AUC','gamma=0.01',low['train_auc'],'gamma=10',high['train_auc']),
        comp('RBF SVM，C=1','验证AUC','gamma=0.01',low['auc'],'gamma=10',high['auc']),
        comp('gamma=10与多数类','验证accuracy','RBF SVM',high['accuracy'],'多数类参考线',r['dummy_cls']['mean']['accuracy'])],
        '核相似度随gamma增大快速衰减，模型记住局部训练响应，未见样本的分数可能集中在默认判别阈值一侧。',
        '固定C、折和预处理，只改变gamma，同时检查训练AUC、验证AUC、accuracy及支持向量数。',
        '训练与验证差距扩大；AUC仍高于0.5而accuracy接近多数类，说明排序和默认决策不能混为一谈。',
        '可进一步检查外层测试分数分布和训练内阈值选择，不将当前证据称为高维孤岛形状的直接观察。')
    c1=next(a['mean'] for a in sc['rbf']['runs'] if a['params']['C']==1 and a['params']['gamma']==.01)
    c100=next(a['mean'] for a in sc['rbf']['runs'] if a['params']['C']==100 and a['params']['gamma']==.01)
    add('增大C提高训练AUC，却降低验证AUC','更重视训练违例的惩罚可能提高整体准确性。',[
        comp('RBF SVM，gamma=0.01','训练AUC','C=1',c1['train_auc'],'C=100',c100['train_auc']),
        comp('RBF SVM，gamma=0.01','验证AUC','C=1',c1['auc'],'C=100',c100['auc'])],
        '大C相对减弱正则化，使模型更愿意为训练点形成复杂分界，可能拟合噪声。',
        '保持gamma=0.01和五折划分不变，对照C扫描的训练及验证指标。',
        '在该gamma下观察到拟合与泛化反向变化；不能推广为任意gamma下大C都会更差。',
        '对不同gamma重复同样比较，并在外层测试中验证参数选择的稳定性。')
    fixed=r['fixed']['linear']['mean']
    add('按AUC选出的最优配置，召回率反而下降','调参得到最优模型后，各项指标可能一起改善。',[
        comp('线性SVM','验证AUC','C=1',fixed['auc'],'C=0.01',lin['auc']),
        comp('线性SVM','验证recall','C=1',fixed['recall'],'C=0.01',lin['recall'])],
        '选参目标为排序AUC，未优化默认阈值下的召回率；分数排序和类别决策是不同目标。',
        '对相同测试折同时计算五项分类指标并核对混淆矩阵。',
        'AUC和accuracy提升不代表漏诊减少，最优必须限定评价指标。',
        '若优先召回，应在训练数据内设定召回约束和阈值，再进行独立评估。')
    f1=r['fixed']['linear']['mean'];f2=r['fixed']['rbf']['mean']
    add('AUC计算口径一变，核的排名反转','硬标签和连续分数计算的AUC可能只是数值不同，排序应一致。',[
        comp('固定参数SVM','硬标签AUC（错误口径对照）','线性核',f1['hard_auc'],'RBF核',f2['hard_auc']),
        comp('固定参数SVM','连续分数AUC（正式口径）','线性核',f1['auc'],'RBF核',f2['auc'])],
        '二值化删除了类别内部排序，两个输出值形成的ROC与连续分数ROC含有不同信息。',
        '保留同一模型的predict和decision_function输出，只改变AUC输入。',
        '错误口径偏向RBF，正式口径偏向线性核；这是受控口径对照，不是正式AUC算错。',
        '已通过协议单元测试防止硬标签进入正式AUC；无需再制造错误运行。')
    ul=best(ab['unscaled_linear']);ur=best(ab['unscaled_rbf'])
    add('标准化的收益和最优C漂移并不一致','两个SVM可能对尺度同样敏感，或者最优C都应同方向漂移。',[
        comp('线性SVM最优','验证AUC','标准化',lin['auc'],'不标准化',ul['mean']['auc']),
        comp('RBF SVM最优','验证AUC','标准化',rbf['auc'],'不标准化',ur['mean']['auc']),
        comp('线性SVM','最优C','标准化',best(sc['linear'])['params']['C'],'不标准化',ul['params']['C']),
        comp('RBF SVM','最优C','标准化',best(sc['rbf'])['params']['C'],'不标准化',ur['params']['C'])],
        '线性权重惩罚与RBF距离受尺度影响的方式不同；RBF的scale还随总体方差变化，超参数可部分补偿尺度。',
        '在标准化和不标准化两种折内管线中各自扫描，保留完整候选和最优参数。',
        '线性核最优C漂移而RBF最优C未变，但RBF的AUC下降更大；重新选参比较不等于只改变尺度的单因素机制证明。',
        '可固定C与数值gamma逐项配对比较，分离尺度变化和重新选参的影响。')
    full=next(a['mean'] for a in sc['depth']['runs'] if a['params']['max_depth'] is None)
    add('深树训练完全拟合，验证几乎没有解释力','更多分裂能够利用更多规律，可能提高预测效果。',[
        comp('回归树','训练R²','深度3',tree['train_r2'],'无限深',full['train_r2']),
        comp('回归树','验证R²','深度3',tree['r2'],'无限深',full['r2']),
        comp('回归树','平均叶节点数','深度3',tree['leaves'],'无限深',full['leaves']),
        comp('回归树','单例叶比例均值','深度3',tree['singleton_fraction'],'无限深',full['singleton_fraction'])],
        '小叶节点均值对单个患者标签过于敏感，单例叶直接记忆训练标签，估计方差增大。',
        '完成深度1–12及None扫描，记录训练与验证成绩、叶内样本数和单例叶比例。',
        '训练R²达到1时验证R²仅约0.003，伴随叶节点细碎化，支持小样本叶均值不稳定的解释。',
        '可在训练内重采样比较叶结构及预测方差，进一步隔离样本扰动效应。')
    add('后剪枝没有超过预剪枝，外层评估又改变两者排序','先长完整树再剪，可能避免预剪枝剪早，从而更好。',[
        comp('回归树','扫描验证R²','预剪枝',tree['r2'],'后剪枝',post['r2']),
        comp('回归树','独立外层R²','预剪枝',r['nested']['depth']['mean']['r2'],'后剪枝',r['nested']['prune']['mean']['r2'])],
        '两种结构约束及候选集合不同，后剪枝只扫描路径的若干位置；扫描选择偏差和样本扰动也会影响排序。',
        '对比完整树、最优预剪枝和后剪枝的叶数、树深及成绩，再采用外层隔离评估。',
        '后剪枝优于完整树却未超过深度3扫描成绩；外层差距很小且排序相反，不支持稳定的普遍优劣结论。',
        '可在各训练折细化后剪枝路径候选，并用同一独立外层比较；不能把未细扫的结果当作后剪枝理论上限。')
    add('独立选参不一定让成绩下降，岭回归成绩完全相同','经过内层重新选择参数后，外层成绩可能必然低于扫描最优。',[
        comp('岭回归','R²','课程扫描最优',ridge['r2'],'独立外层',r['nested']['ridge']['mean']['r2']),
        comp('预剪枝回归树','R²','课程扫描最优',tree['r2'],'独立外层',r['nested']['depth']['mean']['r2'])],
        '若每个外层的内层选择都得到相同参数，外层重新拟合与固定五折中对应模型可完全一致；下降不是嵌套评估的必然数学性质。',
        '检查五个外层选择的参数和逐样本预测，岭回归五次均选择alpha=10。',
        '岭回归参数稳定，而树的选择更受训练样本变化影响；相同数字本身不能作为泄漏证据。',
        '外层隔离仍由协议测试检查；需要外部数据才能评估当前数据来源以外的表现。')
    strongest=max(sc['ridge']['runs'],key=lambda a:a['params']['alpha'])
    add('岭回归适度正则化收益很小，惩罚过强反而接近失效','加入或增大正则化可能显著改善线性模型。',[
        comp('线性回归家族','验证R²','普通最小二乘',ols['r2'],'最优岭alpha=10',ridge['r2']),
        comp('岭回归','验证R²','alpha=10',ridge['r2'],'alpha=10000',strongest['mean']['r2'])],
        '13个输入上的普通线性解已较稳定，适度收缩仅带来小收益；过强惩罚压制了有用信号，形成欠拟合。',
        '跨多个数量级扫描alpha，并对照OLS、训练成绩和标准化系数。',
        '适度正则化改善有限；不能把正则化越强与泛化越好画等号。',
        '可用训练内重采样比较系数方差，区分方差降低与偏差增加的贡献。')
    highlasso=max(sc['lasso']['runs'],key=lambda a:a['params']['alpha'])
    add('Lasso全部系数为零，R²为负，且与均值参考一致','严重收缩后可能仍保留少数重要变量，常数参考的R²应为0。',[
        comp('Lasso','平均零系数数','alpha=0.01',lasso['zero_count'],'alpha=10',highlasso['mean']['zero_count']),
        comp('Lasso','验证R²','alpha=0.01',lasso['r2'],'alpha=10',highlasso['mean']['r2']),
        comp('常数预测','验证R²','全零Lasso',highlasso['mean']['r2'],'训练均值参考',r['dummy_reg']['mean']['r2'])],
        '强惩罚使系数全零但截距仍等于训练均值；测试均值与训练均值不同，所以常数模型测试R²可为负。',
        '检查逐折零系数计数，并比较全零Lasso与DummyRegressor的折外预测。',
        '负R²属于正常退化，不是程序报错；R²=0的概念参照不能代替真正的折外均值基线。',
        '已完成预测相等性检查；不需要为了非负R²去改动测试预测。')
    coefficients=np.array(r['coefficients']['lasso']);zeros=np.sum(coefficients==0,axis=1)
    add('相同alpha下Lasso的稀疏程度随训练折变化','选定一个正则化强度后，筛选出的变量可能应保持一致。',[
        comp('Lasso，alpha=0.01','零系数数量','外层训练折1',zeros[0],'外层训练折4',zeros[3])],
        '弱贡献特征的系数接近阈值，训练样本及相关输入的变化可能改变变量是否被压到零。',
        '逐特征检查五折系数、非零频次及标准差，并结合相关结构表。',
        '五折零系数数量为'+str(zeros.tolist())+'，证明存在折间选择差异，但不能仅据此断言由相关性导致或选择近乎随机。',
        '需要训练内bootstrap或受控相关特征实验检验选择频率及原因；本次尚未执行。')
    inverse=[(a,b) for a in sc['depth']['runs'] for b in sc['depth']['runs'] if a['mean']['r2']>b['mean']['r2'] and a['mean']['mae']>b['mean']['mae']]
    a,b=max(inverse,key=lambda pair:(pair[0]['mean']['r2']-pair[1]['mean']['r2'])*(pair[0]['mean']['mae']-pair[1]['mean']['mae']))
    add('平均绝对误差较小，却可能有更差甚至负的R²','误差指标可能应给出相同模型排序。',[
        comp('回归模型','R²','Lasso',lasso['r2'],'SVR',svr['r2']),
        comp('回归模型','MAE','Lasso',lasso['mae'],'SVR',svr['mae']),
        comp('回归树','R²','深度'+str(a['params']['max_depth']),a['mean']['r2'],'深度'+str(b['params']['max_depth']),b['mean']['r2']),
        comp('回归树','MAE','深度'+str(a['params']['max_depth']),a['mean']['mae'],'深度'+str(b['params']['max_depth']),b['mean']['mae'])],
        '绝对损失和平方损失对少量大误差的权重不同，中等误差减少也可能伴随尾部误差增加。',
        '遍历所有树深候选的两种指标排序，检查逐患者误差及误差超过一级的损失贡献。',
        '两个指标确实出现逆序；按R²总体选择Lasso，按MAE选择SVR，不是四舍五入造成。',
        '若要推断临床代价，需要另外定义等级间损失，而不能直接以任一统计指标替代临床效用。')
    td=ab['drop_fixed_depth']['mean'];rd=ab['drop_fixed_ridge']['mean']
    add('删除检查信息使树的R²上升，岭回归下降，树的MAE却略增','删除有用信息可能使所有模型、所有指标同方向变差。',[
        comp('深度3回归树','R²','13特征',r['fixed']['depth']['mean']['r2'],'删除ca、thal',td['r2']),
        comp('岭回归alpha=1','R²','13特征',r['fixed']['ridge']['mean']['r2'],'删除ca、thal',rd['r2']),
        comp('深度3回归树','MAE','13特征',r['fixed']['depth']['mean']['mae'],'删除ca、thal',td['mae'])],
        '删除信息可能减少树的不稳定分裂，却损害岭回归可利用的全局信号；树的平方误差与绝对误差也可能发生权衡。',
        '先固定模型参数与五折，仅删除两列；再对11特征重新选参，区分信息变化与参数失配。',
        '重新选参后方向仍表现为树改善、岭回归下降；这支持模型依赖的偏差方差权衡，不证明两列普遍无用。',
        '可在训练内重采样测量预测方差，并分析误差尾部变化；临床可用性仍应按预测时点判断。')
    poly=best(ab['poly'])
    add('加入二次交互特征后，训练更好但验证明显更差','更丰富的交互关系可能提高模型表达能力和验证成绩。',[
        comp('岭回归家族','训练R²','原始13维',ridge['train_r2'],'二次104维',poly['mean']['train_r2']),
        comp('岭回归家族','验证R²','原始13维',ridge['r2'],'二次104维',poly['mean']['r2'])],
        '104维系数在小样本中可能吸收偶然关系，正则化未完全抵消估计方差增加。',
        '原始及二次模型分别扫描alpha，所有变换折内拟合，同时比较训练与验证成绩；二次模型最优alpha='+str(poly['params']['alpha'])+'。',
        '训练改善与验证下降并存，支持当前数据下容量收益不足的解释。',
        '可做训练集规模学习曲线，检验增加样本后容量劣势是否减轻；本次未作该额外实验。')
    single=ab['single']
    add('只改变随机划分，就能得到正负两种R²结论','固定算法与参数的评价结果可能应比较稳定。',[
        comp('深度3回归树','单次80/20测试R²','十次最小值',single['min'],'十次最大值',single['max']),
        comp('深度3回归树','R²','单次最小值',single['min'],'固定五折平均',tree['r2'])],
        '约61人的测试集对重症比例和个别大误差敏感；训练样本变化也会改变树结构。',
        '固定树参数及树种子，只将划分种子设为0–9，报告全部十次结果；标准差='+f'{single["std"]:.6f}'+'。',
        '挑最好一次会夸大表现；一次负R²也不能证明模型在所有划分上无效。',
        '若需不确定区间，应采用预先指定的重复评估设计，不能只增加有利种子。')
    evidence=json.loads((ROOT/'figures/discussion_evidence.json').read_text(encoding='utf-8'))
    rows=[a for a in evidence['regression_classification'] if a['model']=='ridge'];first=rows[0];last=rows[-1]
    add('程度回归能接近分类器，换阈值又能改变谁更好','没有用二分类目标训练的回归模型可能应明显不如分类器。',[
        comp('回归器与分类器','五折平均AUC','岭回归连续预测',first['auc'],'线性SVM',lin['auc']),
        comp('岭回归','accuracy','阈值0.3',first['accuracy'],'阈值1.0',last['accuracy']),
        comp('阈值决策','F1','岭回归阈值1.0',last['f1'],'线性SVM默认决策',lin['f1'])],
        'num本身包含num>0的信息，条件均值可提供风险排序；阈值改变决策，但不改变连续分数排序。',
        '以相同五折分别计算AUC及四个阈值的accuracy、F1，再平均，与两个SVM同口径比较。',
        '岭回归AUC接近SVM，低阈值下决策较差，阈值1.0时accuracy和F1反而较高；该阈值仅作敏感性分析，未独立验证。',
        '应仅在训练数据选阈值，在外层测试评估；不能挑当前最高阈值成绩宣称可部署。')
    values=[a['prediction'] for a in best(sc['ols'])['predictions']];negative=sum(value<0 for value in values)
    add('目标只取0–4，普通线性回归却预测负等级','训练目标有界且非负，预测可能也应在相同范围。',[
        comp('普通最小二乘','数值范围','真实目标最小值',0,'折外预测最小值',min(values)),
        comp('普通最小二乘','负值预测人数','按标签范围预期',0,'303例实际预测',negative)],
        '无约束线性函数输出不受目标取值范围限制，平方损失也不会自动施加上下界。',
        '遍历全部折外预测，统计最小值、最大值和越界人数；保留原值计算课程指标。',
        '负预测是模型约束与等级任务不一致的表现，不是读错标签；事后截断会改变评价协议。',
        '若比较截断版本，须作为预先声明的单独对照，不能静默修改正式结果。')
    fold=best(sc['ols'])['folds'][0]
    trainvar=fold['train_mse']/(1-fold['train_r2']);testvar=fold['mse']/(1-fold['r2'])
    add('某折验证R²高于训练R²，但验证MSE反而更大','验证R²较高可能意味着新样本的实际预测误差更小。',[
        comp('OLS，第1折','R²','训练',fold['train_r2'],'验证',fold['r2']),
        comp('OLS，第1折','MSE','训练',fold['train_mse'],'验证',fold['mse']),
        comp('OLS，第1折','标签方差（由MSE和R²还原）','训练',trainvar,'验证',testvar)],
        'R²按各自被评估集合的标签方差归一化，训练集与测试集的分母不同，不能直接以R²大小判断绝对误差。',
        '读取同一折训练与验证MSE、R²，按R²=1−MSE/Var(y)核对分母。',
        '测试标签方差较大，足以使更大的MSE得到更高R²；不能据此声称模型在新患者上优于训练患者。',
        '跨集合比较时同时给出MAE、MSE和标签分布，避免只看R²。')
    # 这些验证不增加新训练，仅检查已保存结果的一致性。
    assert len(items)==19
    assert all(len(item['comparisons'])>=1 for item in items)
    for n in ['ridge','lasso']:
        outer=r['nested'][n]['predictions'];ordinary=best(sc[n])['predictions']
        assert {p['row']:p['prediction'] for p in outer}=={p['row']:p['prediction'] for p in ordinary}
    high_predictions={p['row']:p['prediction'] for p in highlasso['predictions']};dummy={p['row']:p['prediction'] for p in r['dummy_reg']['predictions']}
    assert max(abs(high_predictions[i]-dummy[i]) for i in dummy)<1e-12
    return items


# ???????????????????????????????????
from findings_layout import render
