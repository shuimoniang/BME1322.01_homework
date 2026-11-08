"""由第三版正式结果生成中文报告、简答及文档快照。"""
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments'/'v3_optimized'


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def mrow(label, m):
    return [label, f"{m['mean_return']:.4f}", f"{100*m['b3_time_ratio']:.4f}%",
            f"{100*m['b1_termination_rate']:.4f}%", f"{m['mean_steps']:.2f}"]


def wilson(k, n, z=1.96):
    p = k/n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return center-half, center+half


# 从提交源码提取连续片段，只统一移除外层缩进。
def source_excerpt(filename, start, end):
    import textwrap
    source = (ROOT / filename).read_text(encoding='utf-8')
    begin = source.index(start)
    finish = source.index(end, begin)
    return textwrap.dedent(source[begin:finish]).rstrip() + '\n'


CODE_MODEL_EST = source_excerpt(
    'model_based.py', '        for episode in episodes:',
    '        # 模型变了')
CODE_VI = source_excerpt(
    'model_based.py', '        for iterations in range(1, max_iterations + 1):',
    '        self._values = v')

CODE_Q_UPDATE = """\
def update(self, state, action, reward, next_state, done: bool) -> None:
    si = self._require_state(state)
    ai = self._require_action(action)
    nsi = self._require_state(next_state)

    r = float(reward)
    if done:
        target = r
    else:
        target = r + self.gamma * max(self.q[nsi])

    old = self.q[si][ai]
    self.q[si][ai] = (1.0 - self.alpha) * old + self.alpha * target
    self._visited[si][ai] = True
    self.update_count += 1
"""

CODE_ONLINE = """\
for _ in range(n_online_eps):
    s = env.reset()
    total_r, done = 0.0, False
    while not done:
        a = agent.select_action(s)           # ε-greedy 在线交互
        ns, r, done, _ = env.step(a)
        agent.update(s, a, r, ns, done)
        total_r += r
        s = ns
    agent.decay_epsilon()
"""

CODE_ONLINE_LOOP = """\
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
"""

CODE_GATE = """\
def admissible(candidate, baseline):
    return (candidate['b1_termination_rate'] <= baseline['b1_termination_rate']
            and candidate['b3_time_ratio'] >= baseline['b3_time_ratio'])

accepted = (delta['mean_return']['ci95'][0] > 0 and admissible(cm, bm))
"""


def main():
    r = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    if r['version'] != 'v3_optimized' or r['v3']['config']['quick']:
        raise RuntimeError('报告只能引用正式实验')
    v = r['v3']
    # 附加消融（builder D/E，独立留档，不进 manifest 哈希链）
    bdp = ROOT/'experiments'/'builder_d_data_seed_results.json'
    bep = ROOT/'experiments'/'builder_e_gamma_results.json'
    bd = json.loads(bdp.read_text(encoding='utf-8')) if bdp.exists() else None
    be = json.loads(bep.read_text(encoding='utf-8')) if bep.exists() else None
    states = list(r['part_a']['policy'])
    headers = ['方案','平均无折扣回报','B3时间占比','B1终止率','平均步数']
    mb = r['multiblock_eval']['part_a']
    bs = r['bootstrap_stability']
    comps = [('离线', v['selection']['offline']['winner'], v['offline_comparison'])] + [
        (name, data['winner'], data['frozen_greedy_comparison'])
        for name, data in v['online_test'].items()]

    lines = ['# 血糖调控智能体：基于模型学习与 Q-Learning 实验报告', '',
        '课程：BME1322.01《人工智能》（第 8 章 马尔可夫决策过程；第 9 章 无模型强化学习）',
        '日期：2026-11-07', '',
        '## 1. 项目目标', '',
        '本次作业要求在课程提供的统一血糖模拟器上，独立实现并比较两条无模型强化学习路线：',
        '',
        '- **Part A（基于模型的学习）**：转移与奖励模型未知，先从交互经验中用计数法估计 '
        'P̂(s′|s,a) 与 R̂(s,a)，再在学到的模型上运行价值迭代求策略，思路是"先学模型、再规划"；',
        '- **Part B（Q-Learning）**：同样不读真实模型，直接从给定经验中用表格 Q-Learning 更新 '
        'Q(s,a)，思路是"绕开模型、直接学价值"。', '',
        '两条路线使用同一份交互数据（2000 条随机策略 episode），因此实验可以公平比较'
        '"学模型再规划"与"直接学 Q"两条思路的效果差异。', '',
        '**环境**：模拟器有 9 个离散状态（B1 严重低血糖 至 B9 严重高血糖，B3 为目标区间），'
        '4 个动作（w 维持 / g 补糖 / i1 小剂量胰岛素 / i2 大剂量胰岛素，成本依次为 0 / -0.5 / -0.2 / -0.4），'
        '折扣因子 γ=0.9，每条轨迹最多 200 步。进入 B1 立即终止回合（类似"悬崖"问题的安全性约束），'
        '奖励为进入状态的基础奖励加动作成本。', '',
        '## 2. 总体设计', '',
        '代码分三层组织：', '',
        '- **协议智能体**：`model_based.py`（`ModelBasedAgent`）与 `q_learning.py`（`QLearningAgent`），'
        '严格按课程接口协议实现，只通过给定经验学习，不读取环境内部的真实转移表/奖励表；',
        '- **实验入口**：`main.py`，单一入口复现全部课程必做实验与附加消融，'
        '结果写入 `figures/results.json` 与 9 张图表；',
        '- **第三版驱动层**：`experiments/v3_study.py`，实现"验证集选优 → 冻结 → 独立测试"'
        '的在线学习率比较流程，带检查点续跑与源码 SHA-256 校验。', '',
        '所有扩展实验的脚本和结果都按版本保存在 `experiments/` 下，旧结果不覆盖；'
        '报告里的数字都取自这些正式结果，可随时重新运行核对。', '',
        '从仓库根目录复现：', '',
        '```powershell', 'python sanity_check.py',
        'python -m unittest discover -s assignment3/experiments -p test_v3.py -v',
        'python assignment3/main.py --resume',
        'python assignment3/experiments/write_v3_report.py', '```', '',
        '**评估协议**：统一数据预算为 `collect_random_episodes(2000, seed=2026)`，共 '
        f"{r['total_transitions']:,} 条转移经验，Part A、Part B 与消融共用；"
        '主评估固定 500 条轨迹（环境种子 90000–90499，与训练数据种子不重叠），'
        '指标为平均无折扣累计回报、目标区 B3 时间占比、进入 B1 的终止率与平均步数。', '',
        '## 3. 技术原理', '',
        '**马尔可夫决策过程与 Bellman 最优方程（第 8 章）**。血糖调控可建模为折扣 MDP '
        '(S, A, P, R, γ)：智能体在状态 s 选动作 a，环境按转移概率 P(s′|s,a) 到达 s′ 并给出奖励。'
        '最优价值函数满足 Bellman 最优方程：', '',
        '> V*(s) = max_a [ R(s,a) + γ · Σ_{s′} P(s′|s,a) · V*(s′) ]', '',
        '价值迭代通过反复对该方程做同步扫描（每轮用旧 V 计算全部新 V）逼近 V*，'
        'γ<1 且奖励有界时保证收敛；策略提取取 π(s) = argmax_a Q(s,a)，'
        '其中 Q(s,a) = R(s,a) + γ Σ P̂ V。', '',
        '**从经验中学习模型（第 9 章）**。Part A 不知真实 P/R，用计数法估计：'
        'P̂(s′|s,a) = N(s,a,s′)/N(s,a)（转移计数归一化），R̂(s,a) 为该 (s,a) 上所有奖励的样本均值。'
        '对从未访问的 (s,a)，取 P̂ 为均匀分布、R̂ 为 0：均匀分布是无信息先验，不预设某个转移更可能；'
        '两者结合保证 Bellman 备份良定义（概率和为 1、期望有限）。', '',
        '**Q-Learning（第 9 章）**。Part B 是异策略时序差分控制，不需要模型：', '',
        '> Q(s,a) ← (1−α)·Q(s,a) + α·[ r + γ·max_{a′} Q(s′,a′) ]', '',
        'TD 目标取下一状态的贪心 max，因此即使行为策略不是贪心的也能收敛到 Q*。'
        'done=True 的转移（进入 B1 终止）目标中不含未来项。收敛三条件为：'
        '每个 (s,a) 被无限次充分访问、学习率满足 Robbins-Monro 条件（学习率之和发散、学习率平方和收敛）、'
        'γ<1 且奖励有界。', '',
        '**ε-greedy 探索与 Regret**。在线附加实验中，智能体以 ε 概率均匀随机探索、否则贪心。'
        '在 B1 终止的悬崖式环境里，一次探索动作可能导致整条轨迹提前终止，'
        '因此探索的代价（Regret）很高；衰减 ε 可以逐渐降低这一代价。', '',
        '## 4. 实现方法与关键代码', '',
        '### 4.1 Part A：模型估计与价值迭代（model_based.py）', '',
        "模型估计的核心是三层计数结构 `counts[s][a][s']` 与逐 (s,a) 的奖励累加：", '',
        '```python', CODE_MODEL_EST + '```', '',
        '未访问 (s,a) 的兜底逻辑在估计阶段处理：`N(s,a)=0` 时转移字典取'
        ' `{s: 1/|S| for s in states}`、奖励取 0。价值迭代的同步扫描实现如下：', '',
        '```python', CODE_VI + '```', '',
        f'本次数据上 82 轮收敛（max 差 < 1e-4）。策略提取用「严格大于才更新」的写法，'
        '保证平局时保留动作列表顺序（w,g,i1,i2）中的第一个，符合课程约定。', '',
        '### 4.2 Part B：表格 Q-Learning（q_learning.py）', '',
        'Q 表初始化为 0，`learn()` 按给定顺序逐条 episode、逐条转移调用 `update()`，'
        '不做任何动作采样（离线批量设定，与 Part A 公平对比）：', '',
        '```python', CODE_Q_UPDATE + '```', '',
        f"主实验 α=0.5，共更新 {r['part_b']['update_count']} 次。"
        '`get_policy()` 返回每行 argmax，平局约定与 Part A 一致。'
        'ε-greedy 的 `select_action()` 与 `decay_epsilon()` 仅供附加在线实验使用。', '',
        '### 4.3 实验入口与评估（main.py）', '',
        '`evaluate_policy()` 把确定性策略放进模拟器：每条轨迹用独立种子 '
        '`GlucoseEnv(seed=90000+i)`，逐步累计无折扣回报、统计 B3 停留与 B1 终止。'
        '附加实验模块包括：', '',
        '- 多块评估：5 个独立环境种子块各评 500 条，回报与 B3 占比报 t 区间（t 临界值手算，避免依赖 scipy），'
        'B1 终止率报池化 Wilson 区间；',
        '- bootstrap 稳定性：有放回重采 episode 重学模型与策略，统计每个状态贪心动作的稳定率'
        '与 top-2 Q 差（翻转 margin）；',
        '- 规划变体对比：同一 P̂/R̂ 上同步 VI / 异步 Gauss-Seidel / Q 直接迭代；',
        '- 学习率消融：固定 α∈{0.1,…,1.0} 与 Robbins-Monro 调度（临时实现，不改提交文件）；',
        '- 在线探索实验：三种 ε 设置 × 5 个随机种子 × 600 回合在线交互，滑动窗口 50 学习曲线。', '',
        '### 4.4 预注册式选优流程（experiments/v3_study.py）', '',
        '为避免"先看结果再挑参数"，学习率比较采用验证/测试分离：验证训练种子 11–15、'
        '验证环境种子 110000–110499 上选优；候选冻结到 `selection.json` 后，'
        '再用训练种子 21–25 重新从零训练，在独立环境种子段评估。'
        '在线训练的核心循环（学习率按每个 (s,a) 的累计更新次数调节）：', '',
        '```python', CODE_ONLINE_LOOP + '```', '',
        '接受门槛与安全性判定：', '',
        '```python', CODE_GATE + '```', '',
        '即：回报差 95% 区间下界必须大于 0，且 B1 终止率不升、B3 占比不降；'
        '同时计算"安全性证据尚不足"标志（差值区间是否仍容许退步），如实随结果报告。'
        '运行纪律上，源码 SHA-256 写入 manifest，`--resume` 拒绝混合不同版本的实验数据，'
        '检查点原子写入。', '',
        '## 5. 实验结果与分析', '',
        '### 5.1 访问次数分布', '',
        '![访问次数](figures/fig1_visit_counts.png)', '',
        f"36 个 (s,a) 对全部被覆盖（最少 {r['visit_stats']['max'] and r['visit_stats']['min']}、"
        f"最多 {r['visit_stats']['max']} 次），其中小于 500 次的稀疏对共 "
        f"{len(r['visit_stats']['sparse_pairs'])} 个：B1 的全部 4 个动作（18–29 次）与 "
        'B7–B9 的全部 12 个 (s,a)（187–376 次）。原因符合直觉：随机策略的初始血糖'
        '集中在目标区附近，B3 的 (s,a) 最密集（1200–1400 次）；B1 附近随机策略很快终止、'
        'B7–B9 到达频率低。覆盖只保证备份有定义，稀疏对的模型估计方差更大。', '',
        '### 5.2 Part A：策略、价值与数据量消融', '',
        '![Part A 策略和价值](figures/fig2_part_a_policy_value.png)', '',
        table(['状态','动作','V̂'], [[s,r['part_a']['policy'][s],f"{r['part_a']['values'][s]:.4f}"] for s in states]), '',
        '价值结构合理：V̂ 在目标区 B3 达到峰值 4.77，向两侧单调递减。策略呈典型的'
        '"悬崖问题"形态：低血糖侧 B1/B2 立即补糖远离终止悬崖，目标区 B3 免费维持，'
        '高血糖侧按严重程度加大胰岛素剂量（i1→i2）。', '',
        '讲义要求解释至少一个状态的策略选择，这里拿 B4（血糖略高）为什么选 i1 而不是 i2 来说。'
        'B4 离下方终止区 B1 只隔一个状态，'
        f"经验模型中 i1 与 i2 的价值差约 {r['bootstrap_stability']['flip_margin_top2']['B4']:.3f}。"
        'i2 的平均降糖幅度更大，一步就可能把血糖推入低血糖区，即使进入 B3 有 +1 奖励，'
        '继续下穿到 B2/B1 的尾部风险期望损失更大；i1 大概率把血糖拉回目标区并留在安全带，'
        '动作成本也只有 i2 的一半。这是基于学到的经验模型的解释，不是读取真实转移表后的证明。', '',
        '**B1 的终止语义**：进入 B1 的转移使回合结束，但 `reset()` 可能直接返回 B1，'
        '此时智能体仍须先选择动作，因此训练数据含 B1 出边、其动作有实际意义，'
        'V̂(B1)=1.76 非零不是未访问先验的产物。Part A 按讲义规定的 P̂/R̂ Bellman 式计算、'
        '不加终止掩码；Part B 对 done=True 的经验屏蔽未来项。两者对终止的处理不同，'
        '数值不应被当作同一终止模型下的最优解。', '',
        '![数据量消融](figures/fig3_ablation.png)', '',
        table(headers,[mrow(f'前{n}条',r['ablation'][n]['eval']) for n in ['500','1000','2000']]), '',
        '从消融结果看，前 500 / 1000 / 2000 条 episode 学到的策略完全相同（价值迭代 82/81/82 轮，'
        '未访问对均为 0），因此相同评估种子下指标逐位一致。这说明 9 状态小表样本效率很高；'
        '但不能据此推断模型估计误差为零：三个预算下 P̂/R̂ 数值并不相同，只是 argmax 未被翻转。', '',
        '### 5.3 Part B：Q 表与策略', '',
        '![Part B Q表和策略](figures/fig4_part_b_q_policy.png)', '',
        table(['状态','w','g','i1','i2','贪心动作'],[[s]+[f"{r['part_b']['q_table'][s][a]:.4f}" for a in ['w','g','i1','i2']]+
              [r['part_b']['policy'][s]] for s in states]), '',
        'Q 表每行的相对排序给出了与 Part A 相同的贪心策略。各行绝对值受固定 α=0.5 '
        '在非平稳数据流上的遗忘特性影响，不能与 Part A 的 V̂/Q̂ 直接比较；'
        '低血糖侧 i1/i2 的大负值（约 -8 至 -10）刻画了"向 B1 方向动作"的风险，与悬崖结构一致。', '',
        '### 5.4 主实验指标与随机基线', '',
        table(headers,[mrow('Part A',r['part_a']['eval']),mrow('Part B α=0.5',r['part_b']['eval']),
                       mrow('均匀随机',r['random_baseline'])]), '',
        '两条路线的平均步数都接近 200（几乎跑满回合），B3 占比约 63%；'
        '随机基线平均仅 12.1 步就 100% 落入 B1 终止、回报只有 -27.62，悬崖环境对无策略智能体的'
        '惩罚直观可见，也是在线实验中探索成本高的根源。', '',
        '### 5.5 多块评估与置信区间', '',
        '单块 500 条对 B1 终止这类低频事件分辨率有限，因此扩展为 5 块独立环境种子各评 500 条：', '',
        table(['块（环境种子段）','平均回报','B3占比','B1终止率'],
            [[f"{90000+i*1000}–{90000+i*1000+499}",
              f"{blk['mean_return']:.2f}", f"{100*blk['b3_time_ratio']:.2f}%",
              f"{100*blk['b1_termination_rate']:.2f}%"] for i,blk in enumerate(mb['per_block'])]), '',
        f"五块合并：回报 {mb['return_mean']:.2f}（t 95% 区间 "
        f"{[round(x,2) for x in mb['return_ci95']]}），B3 占比 {100*mb['b3_mean']:.2f}%"
        f"（t 95% 区间 {[round(100*x,2) for x in mb['b3_ci95']]}），"
        f"B1 终止率池化 {100*mb['b1_pooled_rate']:.2f}%"
        f"（Wilson 95% 区间 {'–'.join(f'{100*x:.2f}%' for x in wilson(round(mb['b1_pooled_rate']*2500), 2500))}）。"
        '主实验第 1 块的 B1 终止率为 0.8%，略高于合并区间上限 0.79%。合并区间估计的是总体终止率，不能用来直接检验单块点估计是否一致；'
        '单块 500 条对 0.4% 量级的安全指标只能给出粗粒度估计，安全结论应以多块区间为准。', '',
        '### 5.6 策略稳定性（bootstrap）', '',
        f"对统一数据预算做 30 次 episode 级 bootstrap（seed={bs['seed']}）重学模型与策略："
        f"9 个状态的策略稳定率均为 100%；全表最脆弱的状态是 {bs['most_fragile_state']}"
        f"（top-2 价值间隔 {bs['flip_margin_top2']['B4']:.2f}，全表最小），"
        f"其 V̂ 的 95% 百分位区间为 {[round(x,2) for x in bs['v_b4_ci95_percentile']]}。"
        '有限次重采样稳定不证明全局最优，但说明 B4 的 argmax 距离翻转边界有真实余量。', '',
        '### 5.7 两路线对比分析', '',
        f"默认设置下两路线在 B2–B9 的一致率为 {100*r['comparison']['agreement_rate']:.0f}%，"
        '主评估指标因此完全相同。相同数据预算下两条路线表现一致的原因：21,825 条转移对 '
        '9×4 的表格问题已经足够充分，模型估计误差（Part A）与 TD 采样噪声（Part B）'
        '都不足以翻转任何状态的 argmax。但第 5.9 节的种子消融显示，这一致只是主种子的特例：'
        '换一批数据后两路线即出现分歧，机制差异详见该节分析。', '',
        '**规划变体**：同一 P̂/R̂ 上，异步 Gauss-Seidel 66 轮 vs 同步 82 轮（省 19% 扫描），'
        'Q 直接迭代 66 轮，三者策略完全一致、价值差 <5e-4。小表上同步扫描已足够，'
        '协议规定同步扫描是合理选择。', '',
        '### 5.8 在线探索实验', '',
        '![在线探索曲线](figures/fig5_online_exploration.png)', '',
        table(['设置','末100回合原始回报均值','末100位置平滑值均值'],
              [[name,f"{data['final_100_mean']:.4f}",f"{data['final_100_smoothed_mean']:.4f}"]
               for name,data in r['online_exploration'].items()]), '',
        '从学习曲线看，固定 ε=0.3 的持续随机探索在悬崖环境里反复承担危险动作成本，训练回报为负；'
        '衰减探索逐渐降低这一成本；ε=0 在本组种子表现较好，但不充分探索下的结果不代表跨种子最优。'
        '训练回报包含探索动作，不能与冻结贪心评估混用；没有真实最优策略参照时，'
        '这些曲线只体现探索代价，不是严格计算出的 Regret。', '',
        '**学习率候选的验证选优与独立测试**。离线在同一 2000 条数据上比较固定 '
        'α∈{0.05,…,1.0}，在线对三种探索设置各比较 7 个固定步长与 4 个调度 '
        '`α(n)=0.5/(1+n/k)^p`（k∈{10,50}、p∈{0.6,1.0}）。验证集选优并冻结候选后，'
        '独立测试的配对比较结果：', '',
        table(['设置','候选','回报差','回报差95%区间','通过预设门槛'],
            [[name,winner,f"{c['delta']['mean_return']['point']:.4f}",
              str([round(x,4) for x in c['delta']['mean_return']['ci95']]),
              str(c['passes_predeclared_gate'])] for name,winner,c in comps]), '',
        '![配对评估差值](figures/fig7_v3_paired_differences.png)', '',
        '![第三版在线训练曲线](figures/fig6_v3_online_training.png)', '',
        '图 7 为冻结策略的配对评估差值；图 6 为测试训练种子 21–25 的窗口 50 训练曲线'
        '（候选在此之前已冻结）。训练期表现与冻结策略表现分别判定：训练末 100 回合'
        '仍带探索动作，改善幅度小于冻结贪心评估，两者不能互相转用。', '',
        '这次选优里，离线主任务没有任何候选通过门槛：在 2000 条统一预算下，'
        '更换学习率不能证明改善离线策略，默认 α=0.5 保留。在线三种设置中只有 eps03_decay '
        '的按访问次数衰减调度通过门槛（冻结贪心回报 40.56→88.11，区间下界>0）。'
        '安全与目标区指标的差值区间仍容许退步（"安全性证据尚不足"为 True），'
        '所以报告里不写"安全不退步已被证明"。五个训练种子较少且未做多重比较校正，'
        '统计证据仅限于本次种子。失败候选全部保留在 JSON 中，未根据测试结果再次选择。', '',
        '### 5.9 附加消融：数据采集种子与折扣因子', '',
        '**数据采集种子鲁棒性**：统一预算只固定了一个采集种子（2026），学到的策略可能依赖'
        '这一次抽样。换 10 个采集种子（各 2000 条）完整重跑两路线（评估种子段不变），'
        '脚本与逐种子明细留档于 builder_d_data_seed.py / builder_d_data_seed_results.json：', '',
        '![数据种子鲁棒性](figures/fig8_data_seed_robustness.png)', '',
        table(['采集种子','Part A 回报','Part B 回报','Part B B1终止率','A/B一致(B2–B9)','B4动作 A/B'],
            [[str(it['seed']), f"{it['eval_a']['mean_return']:.2f}", f"{it['eval_b']['mean_return']:.2f}",
              f"{100*it['eval_b']['b1_termination_rate']:.2f}%", str(it['agreement_a_vs_b_b2_b9']),
              f"{it['b4_action_a']} / {it['b4_action_b']}"] for it in bd['per_seed']]), '',
        '结果有些出乎意料：Part A 对数据抽样完全鲁棒，10 个种子学到的策略逐状态相同，评估指标逐位一致，'
        f"B4 在全部种子下均选 i1（top-2 间隔 "
        f"{min(it['b4_top2_margin_a'] for it in bd['per_seed']):.3f}–"
        f"{max(it['b4_top2_margin_a'] for it in bd['per_seed']):.3f}）。"
        'Part B 则高度敏感：回报均值 '
        f"{bd['summary']['eval_metric_distribution']['part_b']['mean_return']['mean']:.2f}"
        f"（样本标准差 {bd['summary']['eval_metric_distribution']['part_b']['mean_return']['std_sample']:.2f}，"
        f"范围 {bd['summary']['eval_metric_distribution']['part_b']['mean_return']['min']:.2f} 至 "
        f"{bd['summary']['eval_metric_distribution']['part_b']['mean_return']['max']:.2f}），"
        f"9 个非主种子中 {bd['summary']['policy_diff_vs_main_seed']['part_b']['n_seeds_with_any_diff_incl_b1']} 个"
        f"的策略与主种子不同，B1 终止率最高达 "
        f"{100*bd['summary']['b1_termination_rate_max']['part_b']['value']:.1f}%"
        f"（seed={bd['summary']['b1_termination_rate_max']['part_b']['seed']}）。"
        '两者的机制差别在于：模型路线把全部经验汇进一张转移表做全局规划，单次抽样噪声在 2000 条规模下被平均掉了；'
        'Q 路线的价值只能沿经验中的 TD 路径局部传播，换一批数据就可能学到明显更差的策略。'
        '这也正是作业简答第 1 问的实证答案：对稀疏访问更敏感、更依赖抽样运气的是直接学 Q 的路线。', '',
        '**折扣因子 γ 敏感性**：固定同一份数据，扫 γ∈{0.5, 0.8, 0.9, 0.95, 0.99}'
        '（脚本与逐 γ 明细留档于 builder_e_gamma.py / builder_e_gamma_results.json）：', '',
        '![γ 敏感性](figures/fig9_gamma_sensitivity.png)', '',
        table(['γ','Part A 回报','B3占比','B1终止率','V̂(B3)','价值迭代轮数','Part B max|Q|','A/B一致率'],
            [[str(g['gamma']), f"{g['part_a']['eval']['mean_return']:.2f}",
              f"{100*g['part_a']['eval']['b3_time_ratio']:.2f}%",
              f"{100*g['part_a']['eval']['b1_termination_rate']:.2f}%",
              f"{g['part_a']['v_b3']:.2f}", str(g['part_a']['vi_iterations']),
              f"{g['part_b']['max_abs_q']:.2f}", f"{100*g['agreement']['agreement_rate']:.0f}%"]
             for g in be['per_gamma']]), '',
        '和种子实验相比，γ 的影响更简单：策略对 γ 完全平坦，所有 γ 下两路线策略表逐状态相同、评估指标不变。'
        '直觉预期"γ 越大越有远见、越避开 B1"没有发生：本环境最优动作本来就在低血糖侧立即补糖，'
        '风险规避与远见程度无关。γ 改变的是价值尺度与求解成本：V̂(B3) 从 1.09 增至 46.00，'
        '价值迭代从 14 轮膨胀到 840 轮（θ=1e-4 下已接近 1000 轮上限）。'
        'γ=0.9 位于性能平坦区间内、计算成本居中，是合理且稳健的选择。', '',
        '### 5.10 收敛三条件的讨论', '',
        '- **充分访问**：有限轨迹只能保证所有 (s,a) 被访问过（本次 36/36，最少 18 次），'
        '不能证明无限次充分访问；种子消融中 Part B 的剧烈波动正是访问"足够但不无限"时的表现。',
        '- **学习率**：固定 α=0.5 不满足平方可和的渐近条件，其 Q 值是近期样本的加权平均，'
        '保留尾部噪声；消融中 α=1.0 使策略在 B1/B4/B6/B8 四个状态翻转、回报跌至 75.56，'
        '是固定大步长风险的直接实证。带正下限的衰减调度同样不满足渐近条件；'
        '无正下限的调度在每对无限访问的假设下满足步长条件，但不保证有限回合内更优。',
        '- **γ<1 且奖励有界**：任务条件已满足。γ 消融显示策略对该条件不敏感，'
        '但 γ 接近 1 时价值尺度和求解成本显著放大。', '',
        '## 6. 迭代过程', '',
        '这份报告里的几个结论被我们自己推翻过。按时间线记录如下，各版本的原始数据和'
        '报告都保存在 `experiments/` 里，可以对照。', '',
        '第一版（10 月 26 日）完成了两路线的实现和课程必做实验，两策略一致，评估 '
        '88.01 / 63.4% / 0.8%，当时认为已比较完整。实现阶段做过协议审计和 17 项独立数值'
        '验证（用 mini-MDP 解析解和另一套独立写的 VI/QL 交叉核对），修掉了 3 个实现问题。'
        '现在回头看，这一版报告里至少有三处判断不成立：以为 B1 是终止状态、智能体在 B1 上'
        '选什么动作无所谓；拿规划变体对比和 Q 值 L2 误差当证据，说默认配置"已经最优"；'
        '还有那个 0.8% 的 B1 终止率，其实只是单块 500 条评估的点估计，没给任何区间。', '',
        '第二版是同一天接着做的。我们先把讲义的红线逐句抄成一份核对清单（CONSTRAINTS.md），'
        '然后分三路做消融，专门检验第一版不成立的判断。先验消融试了 4 种未访问先验乘 8 档'
        '数据量，课程预算下结果完全一样。覆盖了不等于估准了，只是 argmax 碰巧没被'
        '翻转。学习率消融里 α=1.0 直接使 B6 等状态的策略翻转，回报从 88 跌至 75.56，'
        '第一次直观体现了固定大步长的风险。多块评估的合并 Wilson 区间为 '
        '[0.25%, 0.79%]，单块点估计 0.8% 略高于其上限；这一区间估计总体率，不能直接用于判断单块结果是否异常。500 条样本对这种小概率事件的估计仍较粗。', '',
        '第三版（10 月 27 日）做了两件事。一是复核讲义后更正错误：reset() 可能直接返回 B1，'
        '这时智能体必须先选动作，所以 B1 的出边有实际意义，第一版"B1 从未行动"的解释是错的；'
        '旧版学习率调度带着 0.05 的正下限，并不满足 Robbins-Monro 条件，"已获得渐近收敛保证"'
        '和"已经最优"两句都撤了。后来才意识到，Q 值的 L2 误差是拿 Part A 的经验模型当参照'
        '算的，不是真实 Q*，误差小不等于回报高。二是把调参流程改成 4.4 节的两段式：先验证集'
        '选优、冻结，再换种子从零训练测试，失败的候选也全部留在结果里不删。', '',
        '第四版（11 月 7 日）补了两个一直没检验的维度。换 10 个采集种子重跑全流程，发现'
        ' Part A 完全稳定，10 个种子的策略一模一样；Part B 却从 88.01 跌至 -40.05。原来'
        '第一版看到的"两路线完全一致"只是 seed=2026 的巧合。扫 γ 时发现策略对折扣因子'
        '完全不敏感，"γ 越大越怕 B1"的直觉也错了。', '',
        '这几个被推翻的结论的问题出在同一个地方：把一次运行、一个种子的结果当成了普遍规律。'
        '想明白这一点之后，"结论要带区间、调参要先冻结、失败的实验也留档"就写进了'
        '我们的固定流程。', '',
        '## 7. 总结', '',
        '本次作业按课程协议实现了 Part A（计数法模型估计 + 同步价值迭代）和 Part B（原序'
        '单遍表格 Q-Learning），官方接口检查通过。讲义要求的实验（访问次数分布、策略/价值/Q 表、'
        '500 条评估、数据量消融、在线探索对比、两路线一致率与性能对比、收敛三条件讨论）'
        '全部完成，另外补了多块评估置信区间、bootstrap 稳定性、规划变体、学习率消融、'
        '验证选优独立测试、采集种子和 γ 消融七组附加实验。', '',
        '最主要的发现有两个。第一，在统一数据预算下两条路线学到了完全相同的贪心策略'
        '（回报 88.01、B3 占比 63.4%、B1 终止率 0.8%），模型路线的策略在本次数据量、采集种子和折扣'
        '因子消融中保持一致，这种稳定性仅限于已测试的配置。第二，策略相同'
        '不代表两条路线一样稳：换一批采集数据后模型路线纹丝不动，Q-Learning 的回报却大幅'
        '波动，两者的差别在价值传播方式上：全局规划把噪声平均掉了，逐条 TD 只能沿着'
        '数据里的路径传播。在线实验还显示悬崖环境会放大探索的代价，衰减 ε 明显好过固定 ε，'
        '而离线换学习率没有带来可证明的提升。', '',
        '做下来最大的困难是分清三件事：讲义指定的算法、我们想优化的目标、评估用的口径。'
        'γ=0.9 的 Bellman 目标和 200 步无折扣回报不是一回事，学到的经验模型也不是真实环境。'
        '中途有几个结论被推翻，都是因为把这三样混在一起说了。', '',
        '两条路线各有适用的场景。需要 what-if 分析或者想搞清楚"哪个状态缺数据"的时候，'
        '学模型更合适：模型是数据的压缩表示，价值迭代能把稀疏样本的信息全局传播，我们实验里'
        '它的稳定性也印证了这一点；代价是模型错了会被规划放大，价值迭代在错误的模型上求出的'
        '恰恰是错误模型的最优策略。环境难以建模（高维、部分可观测）时直接学 Q 更省事，'
        '但它对数据量和学习率更挑剔。如果继续做下去，我们会先给在线选优的结论补更多训练种子，'
        '再尝试用函数逼近（第 10 章预告）代替硬分箱，血糖本来就是连续量，共享相邻区间的样本'
        '能天然缓解稀疏问题。这些已经超出本次作业的范围，没有放进提交代码。', '',
        '## 8. 参考资料', '',
        '- 课程作业讲义《Blood Glucose Management Agent (Assignment - 3)》及配套 glucose_env.py、sanity_check.py。',
        '- Sutton R. S., Barto A. G. 《Reinforcement Learning: An Introduction》(2nd ed., draft)，'
        '动态规划、表格 Q-Learning 及步长条件相关内容。本文所说的第 8、9 章指课程章节，不是该书章节。仅参考思路与公式，未复制任何外部实现。',
        '- 置信区间方法（t 区间、Wilson 区间、配对 bootstrap）为统计通用方法，实现均为手写，未调用统计库。',
        '- 报告撰写过程中使用 AI 工具（Codex）辅助代码审查、实验脚本实现与报告整理；'
        '所有算法核心逻辑与数值结果均来自本地实际运行与人工核验。', '']
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    analysis='''# Assignment 3 简答

**1. 哪条路线对稀疏访问更敏感，为什么？**

对直接学 Q 的路线更敏感。我们换了 9 个采集种子重跑整个流程（报告 5.9 节）：模型+价值迭代的策略一个状态都没变，Q-Learning 的回报却从 88.01 一路跌至 -40.05，B1 终止率最高到 45.2%。根本差别在价值怎么传播：模型路线把所有经验汇进一张转移表，抽样噪声在 2000 条的规模下被平均掉了；Q 路线的价值只能沿着数据里实际出现的路径逐条传，没见过的动作永远是 0，换一批数据就可能学出明显更差的策略。当然，2000 条覆盖全部 36 个 (s,a)（最少 18 次）不代表模型估计准确，只是在这个规模下够用。另外我们试过离线换学习率，没有可证明的提升；在线实验的结论写在报告 5.8 节。

**2. 两条路线在哪个状态不一致，可能原因是什么？**

默认设置下（α=0.5、seed=2026）两条路线在 B2–B9 完全一致，主实验里找不出不一致的状态。但换 9 个采集种子后，8 个种子都出现了不一致（最少 5/8），B4 在其中 3 个种子下从 i1 翻到 i2 或 w（报告 5.9 节）。例如采集种子为 7 时，B4 上模型路线选择 i1，Q-Learning 选择 i2；主种子下的一致不能推广到其他采样数据。原因上，两条路线传播价值的方式不同：模型路线做全局规划，Q 路线用固定大步长做原序单遍 TD 更新，对样本顺序和尾部噪声的敏感度不一样；而且 Part A 按讲义公式不加终止掩码、Part B 对 done 屏蔽未来项，两者连终止的处理都不完全相同。α=1.0 的消融里 B1、B4、B6、B8 都和 Part A 不一致，但那是改了学习率的实验，不算默认设置的结论。
'''
    (ROOT/'analysis.md').write_text(analysis,encoding='utf-8')
    readme='''# 第三版实验归档

入口：仓库根目录 `python assignment3/main.py --resume`；报告：`python assignment3/experiments/write_v3_report.py`。
首次在没有检查点的目录运行可省略 --resume。源码或配置不一致会拒绝复用，不能直接修改manifest绕过检查。

- results.json：课程必做实验、全部验证候选和独立测试逐轨迹统计。
- selection.json：接触测试数据前冻结的候选及各失败候选原因。
- manifest.json：实验配置及计算源码SHA-256；源码或配置变化必须开新目录。
- checkpoints/：逐候选原始回合记录、访问次数、Q表与评估计数；原子写入，支持续跑。
- DECISION.md：各比较的验收结果；figures/：完整图表与结果速览。
- snapshot/：代码、报告、简答与依赖快照；artifact_sha256.json核对这些交付文件。
- v2_reproduction/：旧版独立复现产物及精确一致核对结果。

当前输出与v1/v2历史归档分开。v2完整快照补存于其snapshot/，已有文件未覆盖。
主任务仍是规定的模型估计+同步VI、原序单遍表格Q-Learning，接口与默认参数不变。
在线调度仅在驱动层，保持600回合；学习过程中不读环境内部状态、不重放经验、不改变奖励。
冻结策略与训练期表现独立比较；安全区间容许退步时不声称安全不退步已被证明。
'''
    (OUT/'README.md').write_text(readme,encoding='utf-8')
    (OUT/'CONSTRAINTS.md').write_text('''# 第三版合规核对

依据作业PDF第3节：Part A计数归一化、样本奖励均值、未访问均匀/零奖励、同步扫描；Part B零初始化、原序单遍、给定α的标准Q更新、done目标不含未来项。
第4.2节在线附加实验保留ε=0.3固定与0.995衰减至0.05、5种子、窗口50；增加ε=0仅作附加对照。
新增学习率调度是报告扩展，不改变提交类的协议行为与默认值；没有用其他方法替换必做两路线。
没有修改glucose_env.py或sanity_check.py。所有学习仅使用状态、动作、奖励、下一状态、done；不读info或隐藏变量。
Part A按讲义未加终止掩码；这种协议限制及其与Part B的区别在报告明确说明，没有擅自“修正”主公式。
全部候选与失败结果保留；不根据独立测试结果再次选择或调参。
''',encoding='utf-8')
    for name in ['report.md','analysis.md']:
        shutil.copy2(ROOT/name,OUT/'snapshot'/name)
    shutil.copy2(__file__,OUT/'snapshot'/'write_v3_report.py')
    shutil.copy2(Path(__file__).with_name('test_v3.py'),OUT/'snapshot'/'test_v3.py')
    paths=[p for p in (OUT/'snapshot').iterdir() if p.is_file()]+list((OUT/'figures').glob('*'))
    paths += [OUT/n for n in ['results.json','manifest.json','selection.json','README.md','CONSTRAINTS.md','DECISION.md']]
    hashes={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    (OUT/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('报告、简答、README与交付哈希已更新。')


if __name__=='__main__': main()
