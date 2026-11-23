"""从已保存的折外预测生成配对错误分析，不重新训练或选择阈值。"""
import csv
import json


def paired_evidence(r):
    def predictions(name):
        scan = r['scans'][name]
        return {p['row']: p for p in scan['runs'][scan['best']]['predictions']}

    ridge = predictions('ridge')
    summaries, cases = [], []
    for name in ['linear', 'rbf']:
        svm = predictions(name)
        assert set(ridge) == set(svm) == set(range(303))
        for threshold in [.3, .5, .7, 1.]:
            counts = dict(both_correct=0, svm_only_correct=0,
                          ridge_only_correct=0, both_wrong=0,
                          lost_false_negative=0, lost_false_positive=0)
            for row, p in sorted(ridge.items()):
                assert p['fold'] == svm[row]['fold']
                truth = int(p['truth'] > 0)
                svm_label = int(svm[row]['prediction'])
                ridge_label = int(p['prediction'] > threshold)
                sc, rc = svm_label == truth, ridge_label == truth
                category = ('both_correct' if rc else 'svm_only_correct') if sc else ('ridge_only_correct' if rc else 'both_wrong')
                counts[category] += 1
                if sc and not rc:
                    counts['lost_false_negative' if truth else 'lost_false_positive'] += 1
                if svm_label != ridge_label:
                    cases.append(dict(svm=name, threshold=threshold, row=row,
                                      fold=p['fold'], num=p['truth'], truth=truth,
                                      svm_prediction=svm_label,
                                      ridge_prediction=p['prediction'],
                                      ridge_label=ridge_label, category=category))
            correct = counts['both_correct'] + counts['svm_only_correct']
            wrong = counts['both_wrong'] + counts['ridge_only_correct']
            assert correct + wrong == 303
            summaries.append(dict(svm=name, threshold=threshold, **counts,
                                  svm_correct=correct, svm_wrong=wrong,
                                  correct_loss_rate=counts['svm_only_correct']/correct,
                                  error_correction_rate=counts['ridge_only_correct']/wrong,
                                  disagreement_rate=(counts['svm_only_correct']+counts['ridge_only_correct'])/303))
    return {'aggregation': 'pooled out-of-fold counts over 303 patients',
            'summaries': summaries, 'disagreements': cases}


def render_paired(r, table, figures):
    evidence = paired_evidence(r)
    (figures/'paired_comparison.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    with (figures/'paired_disagreements.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(evidence['disagreements'][0]))
        writer.writeheader()
        writer.writerows(evidence['disagreements'])
    rows = []
    for s in evidence['summaries']:
        rows.append(['线性核' if s['svm']=='linear' else 'RBF核', f"{s['threshold']:.1f}",
                     s['svm_only_correct'], s['ridge_only_correct'], s['both_wrong'],
                     f"{100*s['correct_loss_rate']:.2f}%",
                     f"{100*s['error_correction_rate']:.2f}%",
                     f"{100*s['disagreement_rate']:.2f}%"])
    return '''#### 逐样本配对：总分相近是否意味着判断相同

上述四个阈值沿用作业第4.6（3）题的示例，用于敏感性分析，没有通过当前测试结果优化阈值。为检查模型是否在同一批患者上判断正确，将岭回归与两种SVM扫描最优配置的折外预测按原始行号配对；每位患者的预测均来自未用该患者训练的模型。

正确样本损失率＝“SVM正确、岭回归错误”的人数／SVM正确人数；错误样本纠正率＝“SVM错误、岭回归正确”的人数／SVM错误人数。前者描述替换模型时损失了多少原本正确的判断，后者描述纠正了多少原有错误；两者分母不同，不能直接相减作为净收益。另列判定不一致率＝两者预测类别不同的人数／303。这些是以SVM为参照的描述性指标，并不把SVM当作真实标签。

'''+table(['参照SVM','岭回归阈值','SVM对／岭错人数','岭对／SVM错人数','两者都错人数','正确样本损失率','错误样本纠正率','判定不一致率'], rows)+'''
线性SVM共判对256例、判错47例；阈值1.0时，岭回归损失其中4例正确判断（1.56%），同时纠正6例错误（12.77%），两者仍共同判错41例。新增的4例错误包括1例漏诊和3例误报；阈值0.3时损失的61例则全部为误报。因此，即使总体accuracy、F1或AUC接近，也不能保证同一个患者得到相同判断，更不能仅凭总分宣称两个模型可以互换。

accuracy只统计正确总数，F1汇总混淆矩阵，均不保留患者身份；一个模型纠正部分病例、同时判错另一些病例，可以得到相近总分。岭回归拟合严重程度num的条件均值，SVM使用二分类间隔目标，两者对重症、轻症及边界病例的权衡不同；改变阈值只能改变岭回归分数上的切分位置，不能任意重排患者，因而不保证复现SVM的逐样本决策。这些机制解释了为何总体表现接近与个体判断不一致可以同时发生，但尚未对每例不一致的原因作因果验证。

本表汇总303例折外预测的人数和比例，区别于上一表先逐折计算再平均的指标；它仍使用课程扫描选出的配置，属于探索性配对分析，不是新增的独立验证或显著性检验。逐例行号、真实等级、预测值与错误归属见[不一致样本清单](figures/paired_disagreements.csv)，row为原CSV零基数据行号、fold为零基折号；完整配对统计见figures/paired_comparison.json。

'''


def ai_question(r):
    s = next(x for x in paired_evidence(r)['summaries']
             if x['svm']=='linear' and x['threshold']==1.)
    return f'''### 对AI结论的质疑：总分相近能否代表逐样本可替代

此前AI主要依据AUC、accuracy和F1讨论“岭回归不逊于分类器”，虽然限定了指标和阈值，却没有检查两个模型是否在同一批患者上判对，容易让读者把总体指标接近理解为具体患者的判断也可替代。用户明确质疑这种证据是否充分，并提出统计“SVM分对、回归判错”的样本；本次检查由用户提出，配对计算由AI执行。

据此补充正确样本损失率和错误样本纠正率，并同时列出反向纠正、共同错误及判定不一致情况：阈值1.0时，岭回归相对线性SVM新增{s['svm_only_correct']}例错误、纠正{s['ridge_only_correct']}例错误，正确样本损失率为{100*s['correct_loss_rate']:.2f}%、错误样本纠正率为{100*s['error_correction_rate']:.2f}%，仍有{s['both_wrong']}例两者都判错；新增错误中有{s['lost_false_negative']}例漏诊和{s['lost_false_positive']}例误报。

修正后的结论是：总体指标只能支持相应指标上的表现比较，不能证明逐样本决策一致或临床可替代。汇总指标丢失患者身份，不同训练目标也可能改变患者排序；调阈值无法保证消除这种差异。新增分析可由保存的折外预测重算，未重新训练模型，也未利用配对结果再选择最优阈值；本次新增内容尚需用户审阅，不沿用此前人工复核已完成的声明。

'''
