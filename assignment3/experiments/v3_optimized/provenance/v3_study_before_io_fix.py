"""第三版：只在驱动层调学习率，独立验证、冻结、测试并留档。

不读取环境隐藏状态，不改变提交智能体及原始经验顺序。
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from glucose_env import GlucoseEnv, STATE_NAMES, ACTION_NAMES, collect_random_episodes
from model_based import ModelBasedAgent
from q_learning import QLearningAgent

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    'eps0_fixed': dict(epsilon=0., epsilon_decay=1., epsilon_min=0.),
    'eps03_fixed': dict(epsilon=.3, epsilon_decay=1., epsilon_min=0.),
    'eps03_decay': dict(epsilon=.3, epsilon_decay=.995, epsilon_min=.05),
}
ALPHAS = [.05, .1, .2, .3, .5, .7, 1.]
BASELINE = 'fixed_0.5'


def candidates():
    out = {f'fixed_{a}': {'alpha': a} for a in ALPHAS}
    for k in (10, 50):
        for p in (.6, 1.):
            out[f'decay_k{k}_p{p}'] = {'k': k, 'p': p}
    return out


def learning_rate(cfg, n):
    """n 从 1 开始，是当前 (s,a) 的累计更新次数。"""
    return cfg['alpha'] if 'alpha' in cfg else .5 / (1 + n / cfg['k']) ** cfg['p']


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def policy_of(agent):
    return {s: agent.get_policy(s) for s in STATE_NAMES}


def evaluate(policy, seed_bases, n=500, env_class=GlucoseEnv):
    """逐轨迹保存 [return, B3步数, B1终止指示, 总步数]；reset到B1也先行动。"""
    records = []
    for base in seed_bases:
        for i in range(n):
            env = env_class(seed=base + i)
            s, done = env.reset(), False
            ret, b3, steps = 0., 0, 0
            while not done:
                s, r, done, _ = env.step(policy[s])
                ret += r
                steps += 1
                b3 += int(s == 'B3')
            records.append([ret, b3, int(s == 'B1'), steps])
    return records


def metrics(records):
    a = np.asarray(records, dtype=float).reshape(-1, 4)
    return {'mean_return': float(a[:, 0].mean()),
            'b3_time_ratio': float(a[:, 1].sum() / a[:, 3].sum()),
            'b1_termination_rate': float(a[:, 2].mean()),
            'mean_steps': float(a[:, 3].mean())}


def admissible(candidate, baseline):
    return (candidate['b1_termination_rate'] <= baseline['b1_termination_rate']
            and candidate['b3_time_ratio'] >= baseline['b3_time_ratio'])


def select(rows):
    baseline = rows[BASELINE]['metrics']
    eligible = [key for key, row in rows.items() if admissible(row['metrics'], baseline)]
    # 完全平局优先固定学习率，再优先原始基线，最后使用名称稳定排序。
    winner = min(eligible, key=lambda key: (-rows[key]['metrics']['mean_return'],
                  not key.startswith('fixed_'), key != BASELINE, key))
    reasons = {key: ('selected' if key == winner else
                    'safety_or_B3_gate_failed' if key not in eligible else
                    'lower_return_or_tie_break') for key in rows}
    return {'winner': winner, 'reasons': reasons, 'selection_data': 'validation_only'}


def train_online(cfg, setting, seed, episodes=600):
    agent = QLearningAgent(STATE_NAMES, ACTION_NAMES, seed=seed, **SETTINGS[setting])
    env = GlucoseEnv(seed=seed * 1000 + 777)
    counts, records = {}, []
    for _ in range(episodes):
        s, done = env.reset(), False
        ret, b3, steps = 0., 0, 0
        while not done:
            a = agent.select_action(s)
            ns, r, done, _ = env.step(a)
            key = (s, a)
            counts[key] = counts.get(key, 0) + 1
            agent.alpha = learning_rate(cfg, counts[key])
            agent.update(s, a, r, ns, done)
            ret += r
            steps += 1
            b3 += int(ns == 'B3')
            s = ns
        records.append([ret, b3, int(s == 'B1'), steps])
        agent.decay_epsilon()
    return {'policy': policy_of(agent), 'q_table': agent.get_q_table(),
            'records': records, 'last100_raw': metrics(records[-100:]),
            'training_metrics': metrics(records), 'update_count': agent.update_count,
            'visit_counts': {f'{s}/{a}': n for (s, a), n in counts.items()}}


def paired_comparison(baseline, candidate, n_boot=3000):
    """配对交叉 bootstrap：重采训练种子及共享环境种子，保留两方案配对。

    输入形状 training_seed × evaluation_episode × 4。
    学习曲线比较时第二轴为回合，先聚合每个训练种子，避免假设相邻回合独立。
    """
    b, c = np.asarray(baseline, float), np.asarray(candidate, float)
    if b.shape != c.shape or b.ndim != 3 or b.shape[2] != 4:
        raise ValueError('paired records must have identical [seed, episode, 4] shape')
    bm, cm = metrics(b), metrics(c)
    keys = ['mean_return', 'b3_time_ratio', 'b1_termination_rate']
    rng = np.random.default_rng(1322)
    draws = {key: [] for key in keys}
    for _ in range(n_boot):
        si = rng.integers(b.shape[0], size=b.shape[0])
        ei = rng.integers(b.shape[1], size=b.shape[1])
        mb, mc = metrics(b[si][:, ei]), metrics(c[si][:, ei])
        for key in keys:
            draws[key].append(mc[key] - mb[key])
    delta = {key: {'point': cm[key] - bm[key],
                  'ci95': np.percentile(draws[key], [2.5, 97.5]).tolist()} for key in keys}
    accepted = (delta['mean_return']['ci95'][0] > 0 and admissible(cm, bm))
    uncertain = (delta['b1_termination_rate']['ci95'][1] > 0 or
                 delta['b3_time_ratio']['ci95'][0] < 0)
    return {'baseline': bm, 'candidate': cm, 'delta': delta,
            'passes_predeclared_gate': bool(accepted),
            'safety_evidence_insufficient': bool(uncertain),
            'bootstrap': {'replicates': n_boot, 'seed': 1322,
                          'method': 'paired training-seed and shared evaluation-seed resampling'}}


def training_comparison(baseline, candidate, n_boot):
    # 训练回合自相关；只在五个训练种子间配对重采，回合内部不重采。
    # 每个种子压成一条均值记录：B3/步数的比值仍是该组的池化时间占比。
    b = np.asarray([np.asarray(x[-100:]).mean(axis=0) for x in baseline])
    c = np.asarray([np.asarray(x[-100:]).mean(axis=0) for x in candidate])
    return paired_comparison(b[:, None, :], c[:, None, :], n_boot)


def source_hashes():
    paths = [ROOT/'main.py', ROOT/'model_based.py', ROOT/'q_learning.py', Path(__file__),
             ROOT.parent/'glucose_env.py', ROOT.parent/'sanity_check.py']
    return {p.relative_to(ROOT.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def run(main, quick=False, resume=False):
    out = ROOT/'experiments'/('v3_quick' if quick else 'v3_optimized')
    out.mkdir(parents=True, exist_ok=True)
    config = {'quick': quick, 'data_seed': 2026, 'n_data': 200 if quick else 2000,
              'online_episodes': 60 if quick else 600,
              'validation_training_seeds': [11] if quick else list(range(11, 16)),
              'test_training_seeds': [21] if quick else list(range(21, 26)),
              'validation_seed_bases': [110000],
              'test_seed_bases': [120000] if quick else [120000,121000,122000,123000,124000],
              'n_eval': 20 if quick else 500, 'bootstrap_replicates': 100 if quick else 3000,
              'candidates': candidates(), 'settings': SETTINGS,
              'selection': 'max validation return with B1<=baseline and B3>=baseline; fixed then baseline ties',
              'acceptance': 'paired return CI lower>0; B1 point<=0; B3 point>=0'}
    manifest = {'config': config, 'sources_sha256': source_hashes()}
    mp = out/'manifest.json'
    if mp.exists() and load(mp) != manifest:
        raise RuntimeError('源码或配置与已有实验不同；拒绝混合结果，请保留旧目录后另开版本。')
    if not resume and (out/'checkpoints').exists():
        raise RuntimeError('已有检查点，请使用 --resume；不会覆盖历史实验。')
    save(mp, manifest)

    def cached(name, fn):
        path = out/'checkpoints'/f'{name}.json'
        if path.exists():
            if not resume:
                raise RuntimeError(f'检查点已存在: {name}')
            return load(path)
        result = fn()
        save(path, result)
        return result

    # 只按公开协议运行课程实验，正式/快速输出严格隔离。
    main.FIG_DIR = str(out/'figures')
    required = cached('required', lambda: _required(main, quick))
    episodes = collect_random_episodes(config['n_data'], seed=2026)
    eval_cache = {}

    def ev(policy, bases):
        key = (tuple(policy.items()), tuple(bases))
        if key not in eval_cache:
            eval_cache[key] = evaluate(policy, bases, config['n_eval'])
        return eval_cache[key]

    def offline_row(cfg):
        agent = QLearningAgent(STATE_NAMES, ACTION_NAMES, alpha=cfg['alpha'])
        agent.learn(episodes)
        records = ev(policy_of(agent), config['validation_seed_bases'])
        return {'policy': policy_of(agent), 'q_table': agent.get_q_table(),
                'records': records, 'metrics': metrics(records)}

    offline = {}
    for name, cfg in candidates().items():
        if 'alpha' in cfg:
            offline[name] = cached('offline_'+name, lambda cfg=cfg: offline_row(cfg))
    selection = {'offline': select(offline), 'online': {}}
    online = {}
    for setting in SETTINGS:
        online[setting] = {}
        for name, cfg in candidates().items():
            print(f'[v3 validation] {setting} / {name}', flush=True)
            def validation(cfg=cfg, setting=setting):
                runs = [train_online(cfg, setting, seed, config['online_episodes'])
                        for seed in config['validation_training_seeds']]
                records = [ev(run['policy'], config['validation_seed_bases']) for run in runs]
                return {'runs': runs, 'records': records, 'metrics': metrics(records)}
            online[setting][name] = cached(f'validation_{setting}_{name}', validation)
        selection['online'][setting] = select(online[setting])
    # 在接触测试数据前单独落盘，续跑必须与原先的冻结决定完全一致。
    sp = out/'selection.json'
    if sp.exists() and load(sp) != selection:
        raise RuntimeError('冻结选优决定不一致')
    save(sp, selection)
    print('[v3] selection frozen:', selection, flush=True)

    nb = config['bootstrap_replicates']
    chosen = selection['offline']['winner']
    off_test = {}
    for name in dict.fromkeys([BASELINE, chosen]):
        off_test[name] = cached('offline_test_'+name,
            lambda name=name: ev(offline[name]['policy'], config['test_seed_bases']))
    off_compare = paired_comparison([off_test[BASELINE]], [off_test[chosen]], nb)
    model = ModelBasedAgent(STATE_NAMES, ACTION_NAMES)
    model.learn_model(episodes)
    model.run_value_iteration()
    part_a = cached('part_a_test', lambda: ev(policy_of(model), config['test_seed_bases']))
    online_test = {}
    for setting in SETTINGS:
        chosen = selection['online'][setting]['winner']
        runs = {}
        for name in dict.fromkeys([BASELINE, chosen]):
            print(f'[v3 test] {setting} / {name}', flush=True)
            def test(name=name, setting=setting):
                trained = [train_online(candidates()[name], setting, seed, config['online_episodes'])
                           for seed in config['test_training_seeds']]
                records = [ev(run['policy'], config['test_seed_bases']) for run in trained]
                return {'runs': trained, 'records': records, 'metrics': metrics(records)}
            runs[name] = cached(f'test_{setting}_{name}', test)
        online_test[setting] = {'winner': chosen, 'candidates': runs,
            'frozen_greedy_comparison': paired_comparison(runs[BASELINE]['records'], runs[chosen]['records'], nb),
            'training_last100_comparison': training_comparison(
                [r['records'] for r in runs[BASELINE]['runs']],
                [r['records'] for r in runs[chosen]['runs']], nb)}
    result = dict(required)
    result['v3'] = {'config': config, 'offline_validation': offline, 'online_validation': online,
                    'selection': selection, 'offline_test_records': off_test,
                    'offline_comparison': off_compare, 'part_a_test': {'policy': policy_of(model),
                    'records': part_a, 'metrics': metrics(part_a)}, 'online_test': online_test}
    result['version'] = 'v3_quick' if quick else 'v3_optimized'
    save(out/'results.json', result)
    save(out/'figures'/'results.json', result)
    write_summary(out, result)
    plot(main, out, result)
    snapshot = out/'snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ['main.py','model_based.py','q_learning.py','requirements.txt']:
        shutil.copy2(ROOT/name, snapshot/name)
    shutil.copy2(__file__, snapshot/'v3_study.py')
    if not quick:
        for src in (out/'figures').iterdir():
            if src.is_file():
                shutil.copy2(src, ROOT/'figures'/src.name)
    print(f'[v3] complete: {out}', flush=True)


def _required(main, quick):
    main.run_required(quick)
    return load(Path(main.FIG_DIR)/'results.json')


def write_summary(out, result):
    v3 = result['v3']
    lines = ['# 第三版实验结果', '', '验证集选优后冻结；测试集不用于调参。', '',
             '| 实验 | 候选 | 基线回报 | 候选回报 | 回报差95%区间 | 通过预设门槛 | 安全性证据尚不足 |',
             '|---|---|---:|---:|---|---|---|']
    rows = [('离线', v3['selection']['offline']['winner'], v3['offline_comparison'])]
    for setting, data in v3['online_test'].items():
        rows.append((setting+' 贪心', data['winner'], data['frozen_greedy_comparison']))
        rows.append((setting+' 训练末100回合', data['winner'], data['training_last100_comparison']))
    for label, name, comp in rows:
        lo, hi = comp['delta']['mean_return']['ci95']
        lines.append(f"| {label} | {name} | {comp['baseline']['mean_return']:.4f} | "
                     f"{comp['candidate']['mean_return']:.4f} | [{lo:.4f}, {hi:.4f}] | "
                     f"{comp['passes_predeclared_gate']} | {comp['safety_evidence_insufficient']} |")
    lines += ['', '详细B3/B1差值及区间、逐回合记录和全部失败候选见 results.json。',
              'False 表示本次未证明提高，不等价于证明永远无法提高。',
              '通过门槛仅包含安全点估计条件；区间容许退步时不得声称安全性已获证实。',
              '五个训练种子的统计推断仍有限；三个在线设置分别检验，未作多重比较校正。']
    text = '\n'.join(lines)+'\n'
    (out/'DECISION.md').write_text(text, encoding='utf-8')
    (out/'figures'/'results_summary.md').write_text(text, encoding='utf-8')


def plot(main, out, result):
    plt = main.plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (setting, data) in zip(axes, result['v3']['online_test'].items()):
        for name, run in data['candidates'].items():
            curves = np.array([[r[0] for r in x['records']] for x in run['runs']])
            smoothed = np.array([[np.mean(row[max(0,i-49):i+1]) for i in range(len(row))]
                                 for row in curves])
            mean = smoothed.mean(axis=0)
            ax.plot(np.arange(1,len(mean)+1), mean, label=name)
            ax.fill_between(np.arange(1,len(mean)+1), smoothed.min(axis=0), smoothed.max(axis=0), alpha=.15)
        ax.set(title=setting, xlabel='训练回合', ylabel='回报（窗口50）')
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out/'figures'/'fig6_v3_online_training.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    keys = ['mean_return','b3_time_ratio','b1_termination_rate']
    labels = ['离线'] + list(result['v3']['online_test'])
    comps = [result['v3']['offline_comparison']] + [d['frozen_greedy_comparison']
              for d in result['v3']['online_test'].values()]
    for ax,key in zip(axes,keys):
        points = [c['delta'][key]['point'] for c in comps]
        ci = np.array([c['delta'][key]['ci95'] for c in comps]).T
        # 百分位区间未必含点估计，用vlines直接画区间，避免负误差条。
        ax.vlines(range(4),ci[0],ci[1],color='#2a9d8f')
        ax.scatter(range(4),points,color='#264653')
        ax.axhline(0,color='grey',linestyle='--')
        ax.set_xticks(range(4), labels, rotation=15, fontsize=8)
        ax.set_title(key+' 候选减基线')
    fig.tight_layout()
    fig.savefig(out/'figures'/'fig7_v3_paired_differences.png',dpi=150)
    plt.close(fig)
