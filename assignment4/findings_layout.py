"""将已验证的发现按重要性与作业条款同时组织。"""
import json
import numpy as np
from common import ROOT

RANKING=[5,14,13,8,17,15,16,7,2,4,1,6,12,11,3,9,10,18,19]
# 原始编号稳定；排名用于组织材料，不是新增实验或统计显著性评分。
DETAILS={
 5:('4.2（1）；4.4；4.6（1）','评估口径对照','评价口径直接影响核的排名，必须先保证流程正确。'),
 14:('4.3（1）；4.4；4.5（1）','本数据上的特征与模型表现','同一特征删除使树与岭回归反向变化，且树的两个误差指标也不同向。'),
 13:('4.3；4.4；4.6（2）','损失函数与指标排序','误差分布与评价目标会改变最佳模型，不能只报告一个冠军。'),
 8:('4.3（2）','复杂度控制与独立评估','剪枝方法的扫描优势没有稳定延续到外层评估，限制了可推广结论。'),
 17:('4.6（3）','任务目标与决策阈值','程度回归能提供分类排序，但决策优劣取决于阈值和指标。'),
 15:('4.3（1）；4.5（4）','本数据上的模型容量','二次展开提高训练成绩却降低验证成绩，支持小样本容量收益不足的解释。'),
 16:('4.5（3）','数据划分敏感性','只改变划分就能改变R²的符号，单次划分无法支持稳定结论。'),
 7:('4.3（1）','本数据上的树容量','叶内样本数量为深树训练拟合与泛化的矛盾提供机制证据。'),
 2:('4.2（2）；4.6（1）','本数据上的核局部化','固定C下增大gamma造成训练与验证分离，直接回应讲义的过拟合论断。'),
 4:('4.2（1）；4.6（1）','排序与筛查决策的权衡','按AUC选参并不保证默认阈值下的召回率改善。'),
 1:('4.2（1）；4.6（1）（2）','本数据上的核选择','简单核在本次评估中略优，但微小差距不足以宣称普遍优势。'),
 6:('4.2（4）；4.5（2）','尺度与参数的共同作用','尺度收益与最优C漂移不一致，不能统一解释两个核的反应。'),
 12:('4.3（3）','稀疏解的折间稳定性','变量选择随折变化是实测现象，但相关性是否为原因还需受控验证。'),
 11:('4.3（3）（4）','模型退化与均值参考','全零系数和负R²反映强收缩及训练测试均值差异，不应误判为程序错误。'),
 3:('4.2（3）','本数据上的惩罚参数','固定gamma改变C，可以区分训练拟合收益与验证损失。'),
 9:('4.3（3）；评估协议','参数稳定性与选参协议','外层成绩不必然下降，参数选择稳定时可与固定五折成绩完全相同。'),
 10:('4.3（3）','本数据上的正则化收益','适度收缩收益有限，而过强惩罚又会损失预测信号。'),
 18:('4.3；4.6（3）','输出约束与等级目标','标签范围不自动约束无约束回归的预测范围，提示模型与任务尺度的差异。'),
 19:('4.4；4.6（2）','指标归一化口径','跨训练和验证集合的R²不能直接替代绝对误差比较。'),
}


def format_entries(entries,heading_level):
    out=[]
    for item in entries:
        out += ['#'*heading_level+f' {item["rank"]:02d}. {item["title"]}\n',
                '对应作业条款：'+item['assignment_sections']+'。现象类别：'+item['phenomenon_category']+'。\n',
                '研究意义：'+item['ranking_reason']+'\n','直觉预期：'+item['intuitive_expectation']+'\n']
        for c in item['comparisons']:
            out.append(f'- 数字对照：{c["model"]}，{c["metric"]}：{c["condition_a"]} **{c["value_a"]:.6f}** → {c["condition_b"]} **{c["value_b"]:.6f}**。')
        out += ['\n可检验的原因假设：'+item['hypothesis']+'\n',
                '已完成的验证方式：'+item['verification_completed']+'\n',
                '验证结果与边界：'+item['verification_result']+'\n',
                '进一步验证：'+item['further_verification']+'\n']
    return '\n'.join(out)


def render(r,table,figures,heading_level=3):
    from unexpected_findings import collect
    raw=collect(r);by_id={item['id']:item for item in raw}
    assert len(RANKING)==19 and set(RANKING)==set(by_id)==set(DETAILS)
    entries=[]
    for i,original_id in enumerate(RANKING):
        clauses,category,reason=DETAILS[original_id]
        entries.append(dict(by_id[original_id],rank=i+1,ranking_reason=reason,
                            assignment_sections=clauses,phenomenon_category=category,selected_for_report=i<5))
    inventory=[]
    grids=[('主实验/'+name,scan) for name,scan in r['scans'].items()]
    grids += [('消融/'+name,scan) for name,scan in r['ablation'].items() if 'runs' in scan]
    for name,scan in grids:
        runs=scan['runs'];objective='auc' if 'auc' in runs[0]['mean'] else 'r2'
        for run in runs:
            assert all(np.isfinite(value) for value in run['mean'].values())
            assert sorted(p['row'] for p in run['predictions'])==list(range(303))
            assert all(np.isfinite(p['prediction']) and np.isfinite(p['score']) for p in run['predictions'])
        inventory.append(dict(group=name,candidates=len(runs),objective=objective,
                              minimum=min(a['mean'][objective] for a in runs),
                              maximum=max(a['mean'][objective] for a in runs),
                              max_training_validation_gap=max(a['mean']['train_'+objective]-a['mean'][objective] for a in runs),
                              r2_mae_inversions=sum(a['mean']['r2']>b['mean']['r2'] and a['mean']['mae']>b['mean']['mae'] for a in runs for b in runs) if objective=='r2' else None))
    scope=dict(main_families=len(r['scans']),main_candidates=sum(len(s['runs']) for s in r['scans'].values()),ablation_candidate_sets=sum(len(s['runs']) for s in r['ablation'].values() if 'runs' in s),nested_outer_fits=sum(len(s['folds']) for s in r['nested'].values()),findings=19,report_findings=5,additional_findings=14)
    (figures/'unexpected_findings.json').write_text(json.dumps({'scope':scope,'ranking_criteria':['流程正确性与课程评分关联','反直觉程度与研究意义','验证证据强度及结论边界'],'candidate_inventory':inventory,'findings':entries},ensure_ascii=False,indent=2),encoding='utf-8')
    convention='除单折、单次划分及全体折外统计外，均按固定五折汇总。\n'
    return ('# 实验日志\n\n'
            '以下五项分别展示评价口径、信息集合、损失函数、选参协议和任务定义如何影响结论。它们与本数据上的反直觉模型表现一并记录，但评估口径造成的现象不解释为数据集固有性质。其余14项见[补充实验发现](additional_findings.md)，完整记录仍保留。\n\n'
            +convention+'\n'+format_entries(entries[:5],heading_level))
