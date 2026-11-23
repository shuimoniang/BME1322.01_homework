"""从正式实验数据生成报告第 8.6–8.8 节的补充对照表。"""
from selection_summary import best


def supplementary_tables(r, table):
    scans = r['scans']
    fixed = r['fixed']
    unpruned = next(run for run in scans['prune']['runs'] if run['params']['path_fraction'] == 0)
    delta_rows = []
    for name, label, metric in [
        ('linear', '线性SVM，相对固定C=1', 'auc'),
        ('rbf', 'RBF SVM，相对固定C=1、gamma=scale', 'auc'),
        ('ridge', '岭回归，相对固定alpha=1', 'r2'),
        ('svr', 'SVR，相对固定配置', 'r2'),
        ('prune', '后剪枝，相对完整树', 'r2'),
    ]:
        baseline = unpruned if name == 'prune' else fixed[name]
        delta = best(scans[name])['mean'][metric] - baseline['mean'][metric]
        delta_rows.append([label, 'AUC' if metric == 'auc' else 'R²', f'{delta:+.6f}'])

    svm_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    svm_rows = []
    for config, runs in [('固定超参数', fixed), ('各自按AUC选参', {n: best(scans[n]) for n in ('linear', 'rbf')})]:
        for name, kernel in [('linear', '线性'), ('rbf', 'RBF')]:
            svm_rows.append([config, kernel, *(runs[name]['mean'][key] for key in svm_metrics)])

    pruned = best(scans['prune'])
    depth3 = fixed['depth']
    after = pruned['mean']

    def comparison(before, metric):
        delta = after[metric] - before[metric]
        signed_improvement = delta if metric == 'r2' else -delta
        outcome = '改善' if signed_improvement > 0 else '变差' if signed_improvement < 0 else '持平'
        return f'{before[metric]:.6f}→{after[metric]:.6f}，{outcome}'

    before = unpruned['mean']
    pre = depth3['mean']
    pruning_rows = []
    for label, baseline, structure in [
        ('未剪枝完整树', before, f'{before["depth"]:g}→{after["depth"]:g}、{before["leaves"]:g}→{after["leaves"]:g}，结构明显缩小'),
        ('深度3预剪枝树', pre, f'深度{pre["depth"]:g}→{after["depth"]:g}，叶数{pre["leaves"]:g}→{after["leaves"]:g}，结构约束不同'),
    ]:
        pruning_rows.append([label, *(comparison(baseline, key) for key in ('r2', 'mae', 'rmse')), structure])

    folds = []
    for index, (baseline, selected) in enumerate(zip(depth3['folds'], pruned['folds']), 1):
        baseline_r2, selected_r2 = baseline['r2'], selected['r2']
        folds.append([index, baseline_r2, selected_r2, f'{selected_r2-baseline_r2:.6f}'])
    fraction = pruned['params']['path_fraction']
    return '\n'.join([
        '### 8.6 固定基线与调参收益',
        table(['对照', '主指标', '扫描配置相对参照的增益'], delta_rows),
        '### 8.7 默认阈值指标与连续分数AUC的差异',
        table(['配置', '核', 'accuracy', 'precision', 'recall', 'F1', 'AUC'], svm_rows),
        '### 8.8 回归树后剪枝的逐折差异',
        table(['后剪枝相对的参照', 'R²（越高越好）', 'MAE（越低越好）', 'RMSE（越低越好）', '树深／叶数的含义'], pruning_rows),
        table(['验证折', '深度3预剪枝R²', f'路径位置{fraction:g}后剪枝R²', '后剪枝−预剪枝'], folds),
    ])
