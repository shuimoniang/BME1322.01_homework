"""逐图逐表补充实测结果解读，并修正对照条件的说明。"""
import re
from common import load


def improve_readouts(text, r, data=None):
    _, df, _ = load(data)
    def best(name):
        scan=r['scans'][name]
        return scan['runs'][scan['best']]

    captions={
        'labels.png': '图解（标签分布）：num=0、1、2、3、4分别有164、55、36、35、13例；二分类阴性164例、阳性139例，类别并非完全均衡，而最高等级4只有13例。少量重症患者的误差可能明显影响回归平方损失，因此不能只报告多数等级的拟合情况，也不宜用单次划分代替固定五折。',
        'feature.png': f'图解（oldpeak与严重程度）：横轴为运动相对静息的ST段压低幅度oldpeak，纵轴为真实等级num；两者的全表Pearson相关系数为{df.oldpeak.corr(df.num):.6f}。图中较高oldpeak更多出现在阳性等级，但同一oldpeak范围内多个等级仍明显重叠，且num=0也有较高值，因此单个特征不能准确区分全部患者或确定具体等级；散点重叠还会遮住部分样本，不能直接把可见点数当作人数。',
        'correlation.png': f'图解（相关结构）：颜色表示Pearson相关系数，红色为正、蓝色为负，对角线为变量与自身的相关。oldpeak与slope的相关系数为{df.oldpeak.corr(df.slope):.6f}，age与thalach为{df.age.corr(df.thalach):.6f}，说明输入之间存在共享信息；num与ca的相关系数为{df.num.corr(df.ca):.6f}，与thalach为{df.num.corr(df.thalach):.6f}。这些是全表描述性相关，包含目标num，未用来在折外挑选特征；相关不代表因果，离散编码的Pearson值也不能概括全部依赖关系。',
        'heatmap.png': f'图解（RBF联合网格）：横轴是gamma、纵轴是C，颜色越亮表示固定五折平均验证AUC越高；scale是库根据训练数据计算的规则，不是一个与其他数值等距的gamma。当前网格最高AUC为{best("rbf")["mean"]["auc"]:.6f}，对应C=100、gamma=0.001；gamma=1或10的列整体较低，说明过度局部化损害新样本排序。最高点位于数值网格边界，结论只能限定为当前网格内最优，不能据此声称全局最优。',
    }
    for name, caption in captions.items():
        pattern=r'(!\[[^\]]*\]\(figures/'+re.escape(name)+r'\))'
        if caption.split('：')[0] not in text:
            text=re.sub(pattern,lambda m:m[0]+'\n\n'+caption+'\n',text,count=1)

    # 消融表中混合两项任务时，按每行任务明确指标。
    lines=text.splitlines();out=[];in_table=False
    headers={'| 模型 | 原始扫描最优 | 消融扫描最优 | 消融参数 |',
             '| 模型 | 原固定参数 | 删除特征固定参数 |'}
    for line in lines:
        if line in headers:
            in_table=True
            out.append(line.replace('| 模型 |','| 模型 | 评价指标 |',1))
        elif in_table and line.startswith('|'):
            if re.fullmatch(r'[| :\-]+',line):
                out.append(line+'---|')
            else:
                cells=line.split('|')
                cells.insert(2,' AUC ' if 'SVM' in cells[1] else ' R² ')
                out.append('|'.join(cells))
        else:
            in_table=False
            out.append(line)
    text='\n'.join(out)+'\n'
    old='gamma=scale时，该行最佳C=0.1，训练AUC=0.924762、验证AUC=0.903754、差距=0.021008。'
    text=text.replace(old,'作为标准化条件下的参照，RBF核在gamma=scale这一列的最佳C=0.1，训练AUC=0.924762、验证AUC=0.903754、差距=0.021008；这一行来自标准化实验，不是不标准化表中的候选，也不是整个联合网格的最优配置。')
    text=text.replace('C发生漂移。（2）SVM（RBF核）','C发生漂移。SVM（RBF核）')
    note='本表固定各模型先按R²选出的配置，再分别按R²和MAE排序；MAE名次表示这些已选配置之间的比较，没有把每个模型重新按MAE选参。因此这里的SVR是表内MAE最佳配置，不能把本表数值称为整个参数网格的MAE最优值。'
    anchor=re.search(r'^### 5\.13 .*$',text,re.M)
    if anchor and note not in text:
        text=text[:anchor.end()]+'\n\n'+note+text[anchor.end():]
    text=text.replace('按R²选择的总体模型为Lasso回归，改按MAE选择则为支持向量回归（SVR），评价目标会改变最佳模型。',
                      '在本表已按R²选参的配置中，按R²选择Lasso回归，改按MAE比较则选择支持向量回归（SVR），评价目标会改变表内模型排序。')
    # 七个扫描对照表原来只有方向说明，逐一替换为包含实际变化的解读。
    generic='本对照只比较同一模型的固定配置与当前网格所选配置；AUC、R²等越高越好，MAE和RMSE越低越好，差值方向须按指标解释。'
    for name in ['linear','rbf','depth','prune','ridge','lasso','svr']:
        ref=r['fixed'][name] if name!='prune' else next(z for z in r['scans']['prune']['runs'] if z['params']['path_fraction']==0)
        chosen=best(name);key='auc' if name in ['linear','rbf'] else 'r2'
        label='AUC' if key=='auc' else 'R²'
        delta=chosen['mean'][key]-ref['mean'][key]
        interpretation=f'扫描最优{label}由固定对照的{ref["mean"][key]:.6f}变为{chosen["mean"][key]:.6f}，差值为{delta:+.8f}。'
        if name=='linear': interpretation+='AUC略升但阳性召回率下降，说明按AUC调参并不保证每项分类指标改善。'
        elif name=='rbf': interpretation+='AUC增益较明显，但不能仅凭调参收益断言RBF超过线性核，仍须比较两者最终成绩。'
        elif name=='depth': interpretation+='选中的深度仍为3，与固定基线相同，所以这次扫描没有额外提升。'
        elif name=='prune': interpretation+='这里固定对照是完整未剪枝树；改善证明本次剪枝优于完整树，不代表超过课程深度3基线。'
        elif name=='ridge': interpretation+='适度增加惩罚的收益较小，不能把很小的数值优势解释成显著提升。'
        elif name=='lasso': interpretation+='最优alpha与固定基线相同，微小差异来自求解容差，不是调参带来的性能收益。'
        else: interpretation+='该配置同时减少MAE和RMSE，但其选择依据仍是R²；按MAE重新选参可能得到另一配置。'
        text=text.replace(generic,interpretation,1)
    # 原AUC错误记录已是最后一条，删除指向不存在下文的提示。
    last=text.rfind('其他错误及修正记录见本节下文。')
    if last!=-1 and text.rfind('#### 10.4.4')<last and '### 10.5' in text[last:]:
        text=text[:last]+text[last:].replace('其他错误及修正记录见本节下文。','',1)
    return text.rstrip()+'\n'
