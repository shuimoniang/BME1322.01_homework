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


def write_index(entries,table):
    text=['# 本数据集反直觉现象与预期不符结果：按作业条款索引\n',
          '本索引将同一套19条实测记录按4.2、4.3、4.5、4.6归类，与按重要性排列的实验日志互补。一条发现可回应多个条款，交叉出现只是索引，不增加发现数量。\n',
          '本数据上的模型表现具有数据与候选范围限制；指标定义、错误口径对照、输出约束及选参协议带来的现象，不能全部归因于数据集固有性质。所有条目均附实测数字，完整原因假设、已做验证、验证边界和未做的进一步检验见对应记录。\n',
          '排名1–5见[主要实验日志](experiment_log.md)及报告第十节；排名6–19见[补充实验发现](additional_findings.md)。原始编号、排名、条款、分类和全部数值见figures/unexpected_findings.json。\n']
    descriptions={
      '4.2':'分类：核的选择、C与gamma、训练验证差距及尺度影响',
      '4.3':'回归：树的复杂度、剪枝、收缩、稀疏性与程度预测',
      '4.5':'消融：信息删除、标准化、划分波动与容量增加',
      '4.6':'任务比较：核的优劣、分类与回归联系、分数与阈值',
    }
    for clause,title in descriptions.items():
        selected=[item for item in entries if clause in item['assignment_sections']]
        rows=[]
        for item in selected:
            comparison='<br>'.join(f'{c["model"]}，{c["metric"]}：{c["condition_a"]} {c["value_a"]:.6f} → {c["condition_b"]} {c["value_b"]:.6f}' for c in item['comparisons'][:2])
            location=('experiment_log.md' if item['selected_for_report'] else 'additional_findings.md')+f' 第{item["rank"]:02d}项'
            rows.append([f'{item["rank"]:02d}',item['title'],comparison,item['phenomenon_category'],location])
        text += ['## '+clause+' '+title+'\n',table(['重要性排名','现象','关键实测对照','现象类别','详细记录'],rows),
                 '我从这张表里读出：'+{
                    '4.2':'核更复杂、C或gamma更大，均不保证泛化更好；AUC排序、默认分类决策和尺度效应还需分别解释。',
                    '4.3':'更深的树、更完整的生长路径、更强的收缩，并不必然产生更好的验证结果；负R²、稀疏变化及越界预测也各有不同机制。',
                    '4.5':'增加信息或容量不保证改善，减少信息也不保证变差；同一模型的评价还可能强烈依赖划分。',
                    '4.6':'模型优劣取决于目标、损失、评价口径及阈值，不能把一次排序当作所有任务上的统一结论。',
                 }[clause]+'\n']
    text += ['## 评估口径与模型边界的补充现象\n',
             '第16项（独立选参不一定降分）、第18项（预测可越出等级范围）、第19项（验证R²更高而MSE更大）主要涉及方法或指标性质；应与数据集模型表现一并记录，但不称为本数据集独有规律。\n',
             '这里的第16、18、19项是重要性排名；全部条目都保留原始编号，便于与数值记录核对。\n']
    (ROOT/'phenomena_index.md').write_text('\n'.join(text),encoding='utf-8')


def render(r,table,heading_level=3):
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
    (ROOT/'figures/unexpected_findings.json').write_text(json.dumps({'scope':scope,'ranking_criteria':['流程正确性与课程评分关联','反直觉程度与研究意义','验证证据强度及结论边界'],'candidate_inventory':inventory,'findings':entries},ensure_ascii=False,indent=2),encoding='utf-8')
    write_index(entries,table)
    convention='除明确标为单折、单次划分或全体折外预测统计的项目外，成绩均为课程固定五折平均。“预期”是用于理解结果的直觉假设，不声称事前预注册；已完成检查与待做的进一步验证分开列出。按作业条款浏览全部现象可查阅[反直觉现象索引](phenomena_index.md)。\n'
    supplement=('# 补充实验发现（重要性排序第6–19项）\n\n'
                '按流程正确性与评分关联、反直觉程度和验证证据综合排序。前五项见report.md第十节或experiment_log.md；本文件保存其余14项。完整19项数字、条款、分类和原始编号另存figures/unexpected_findings.json。\n\n'
                +convention+'\n'+format_entries(entries[5:],2))
    (ROOT/'additional_findings.md').write_text(supplement,encoding='utf-8')
    return ('# 实验日志\n\n'
            '以下五项分别展示评价口径、信息集合、损失函数、选参协议和任务定义如何影响结论。它们与本数据上的反直觉模型表现一并记录，但评估口径造成的现象不解释为数据集固有性质。其余14项见[补充实验发现](additional_findings.md)，完整记录仍保留。\n\n'
            +convention+'\n'+format_entries(entries[:5],heading_level))
