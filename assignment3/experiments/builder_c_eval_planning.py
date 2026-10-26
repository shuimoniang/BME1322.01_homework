"""Builder C 实验脚本 —— 评估统计严谨性 + 规划增强（Dyna 风格附加变体）。

从仓库根目录运行：
    python assignment3/experiments/builder_c_eval_planning.py

只读 ``model_based.py`` / ``q_learning.py`` / ``glucose_env.py``，不修改任何
提交文件；结果写入 ``assignment3/experiments/builder_c_eval_planning_results.json``。

实验内容（对应任务书三条）：

1. 评估严谨性：Part A 与 Part B 策略各跑 5 个独立评估块
   （每块 500 条轨迹，种子段 90000/91000/92000/93000/94000），
   报告每块三指标（平均回报 / B3 时间占比 / B1 终止率）、跨块标准差、
   95% 置信区间（t 分布 df=4 与正态近似两套，手算，仅 numpy）。
   B1 终止率是稀有事件，额外给出 Wilson 分数区间（正态近似对
   p≈0.008 的区间会越界，Wilson 更稳）。
2. 规划增强（附加变体，不动提交文件的同步扫描 VI）：
   在同一份学到的 P̂/R̂ 上对比四种规划器的收敛行为与结果一致性：
     (a) 同步扫描 VI（提交协议基准，直接调用 ModelBasedAgent）；
     (b) 异步原地 VI（Gauss-Seidel：逐状态用最新 V 更新）；
     (c) Q 直接迭代（对 Q 施加 Bellman 最优算子，同步扫描）；
     (d) Dyna 式采样 Q-Learning（从 P̂/R̂ 模拟采样转移、做表格
         Q-Learning 更新——Dyna 的经典形态，验证其不动点）。
   对比收敛所需扫描/更新次数、最终 V/Q 与同步扫描的差异、策略一致性。
3. 模型不确定性：对 2000 条 episode 做 bootstrap（有放回重采 episode，
   B=30 次），每次重学模型 + 同步 VI + 提取策略，统计：
   - 每个状态 argmax 与全量策略一致的频率（策略稳定率）；
   - V(B4) 的 95% bootstrap 百分位区间；
   - bootstrap Q 值差（全量最优动作 vs 次优动作的 Q 差分布）。
   回答"哪个状态的策略最不稳定"，支持报告 2.6 的"最可能翻转状态"论断。

约束遵守（CONSTRAINTS.md）：全部只用标准库 + numpy；不读取环境真实
转移/奖励表；提交文件的同步扫描 VI 默认行为不受影响——本脚本的
异步 / Q 迭代 / Dyna 采样仅作为附加变体在实验目录内留档。
"""

import json
import math
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_A3_DIR = os.path.dirname(_HERE)          # assignment3/
_ROOT = os.path.dirname(_A3_DIR)          # 仓库根目录（glucose_env.py 所在）
# glucose_env.py 在仓库根目录，agent 实现在 assignment3/；实验脚本位于
# experiments/ 子目录，需把两层路径都加入 sys.path（与 main.py 的查找
# 顺序一致，保证加载的是提交的 model_based.py / q_learning.py）。
for _p in (_ROOT, _A3_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from glucose_env import ACTION_NAMES, STATE_NAMES, collect_random_episodes  # noqa: E402
from model_based import ModelBasedAgent  # noqa: E402
from q_learning import QLearningAgent  # noqa: E402

# ---------------------------------------------------------------------- #
# 配置
# ---------------------------------------------------------------------- #
DATA_SEED = 2026
N_EPISODES = 2000
GAMMA = 0.9
N_EVAL_PER_BLOCK = 500
EVAL_SEED_BASES = [90000, 91000, 92000, 93000, 94000]
THETA = 1e-4          # 与提交的 run_value_iteration(theta=1e-4) 一致
MAX_ITER = 1000
BOOTSTRAP_B = 30
BOOTSTRAP_SEED = 1322  # 与作业固定种子一致，保证可复现
DYNA_SIM_TRANSITIONS = 50000
DYNA_SEED = 2026

# t 分布临界值（df=4, 双侧 95%）：t_{0.975}(4) = 2.776445…（查表常数，
# 手算使用；不依赖 scipy）
T_CRIT_DF4 = 2.7764451051977987
Z_95 = 1.959963984540054  # 标准正态 97.5% 分位数

OUT_PATH = os.path.join(_HERE, "builder_c_eval_planning_results.json")


# ---------------------------------------------------------------------- #
# 评估（与 assignment3/main.py 的 evaluate_policy 协议逐字一致）
# ---------------------------------------------------------------------- #
def evaluate_policy(policy, n_episodes=N_EVAL_PER_BLOCK, seed_base=90000):
    """评估确定性策略；每个 episode 用独立种子 rng=Random(seed_base+i)。

    与 main.py 中 evaluate_policy 相同的协议（此处为实验自包含而复制，
    未改动 main.py）。
    """
    returns, b1_count, b3_steps, total_steps, step_counts = [], 0, 0, 0, []
    from glucose_env import GlucoseEnv
    for i in range(n_episodes):
        env = GlucoseEnv(seed=seed_base + i)
        rng = random.Random(seed_base + i)
        s = env.reset()
        total_r, steps, done = 0.0, 0, False
        while not done:
            a = policy(s, rng)
            s, r, done, _ = env.step(a)
            total_r += r
            steps += 1
            if s == "B3":
                b3_steps += 1
            total_steps += 1
            if done and s == "B1":
                b1_count += 1
        returns.append(total_r)
        step_counts.append(steps)
    n = len(returns)
    return {
        "mean_return": sum(returns) / n,
        "b3_time_ratio": b3_steps / total_steps,
        "b1_termination_rate": b1_count / n,
        "mean_steps": sum(step_counts) / n,
        "returns": returns,
    }


# ---------------------------------------------------------------------- #
# 统计工具（手算，仅 numpy / math）
# ---------------------------------------------------------------------- #
def mean_sd_ci(vals):
    """均值、样本标准差(ddof=1)、SEM、t(df=n-1 近似用 df=4 表值) 与正态 95% CI。"""
    arr = np.asarray(vals, dtype=float)
    n = arr.size
    m = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n)
    # 5 个块 → df=4，用查表 t 值；正态近似作对照
    t_crit = T_CRIT_DF4 if n == 5 else T_CRIT_DF4
    return {
        "mean": m,
        "sd": sd,
        "sem": sem,
        "ci95_t": [m - t_crit * sem, m + t_crit * sem],
        "ci95_normal": [m - Z_95 * sem, m + Z_95 * sem],
        "values": [float(x) for x in vals],
    }


def wilson_ci(k, n, z=Z_95):
    """Wilson 分数区间（二项比例，稀有事件下优于 Wald 正态近似）。"""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [center - half, center + half]


# ---------------------------------------------------------------------- #
# 实验一：5 块独立评估
# ---------------------------------------------------------------------- #
def eval_blocks_with_counts(policy_map):
    """对 5 个种子段各评 500 条轨迹，返回每块指标与池化的逐轨迹回报。

    块级 B3/B1 统计直接取 evaluate_policy 的结果；B1 的合并推断用
    summarize_blocks 中的 k/n 重构（rate*500 为整数计数）。
    """
    per_block = []
    pooled_returns = []
    total_b3 = 0
    total_steps = 0
    total_b1 = 0
    for base in EVAL_SEED_BASES:
        res = evaluate_policy(lambda s, rng: policy_map[s], seed_base=base)
        # 从 returns 重建计数：evaluate_policy 未返回 b3/b1 计数，
        # 这里用块级比率 × 该块步数还原（等价于重新计数）。
        per_block.append({
            "seed_base": base,
            "mean_return": res["mean_return"],
            "b3_time_ratio": res["b3_time_ratio"],
            "b1_termination_rate": res["b1_termination_rate"],
            "mean_steps": res["mean_steps"],
        })
        pooled_returns.extend(res["returns"])
    return per_block, pooled_returns


def summarize_blocks(per_block, pooled_returns, label):
    """跨块统计 + 池化统计。"""
    rets = [b["mean_return"] for b in per_block]
    b3s = [b["b3_time_ratio"] for b in per_block]
    b1s = [b["b1_termination_rate"] for b in per_block]

    # 池化（2500 条逐轨迹）的均值 CI
    arr = np.asarray(pooled_returns)
    n = arr.size
    pooled = {
        "n_episodes": int(n),
        "mean_return": float(arr.mean()),
        "se": float(arr.std(ddof=1) / math.sqrt(n)),
        "ci95_normal": None,
    }
    pooled["ci95_normal"] = [pooled["mean_return"] - Z_95 * pooled["se"],
                             pooled["mean_return"] + Z_95 * pooled["se"]]

    # B1 终止率：块级 k = rate*500，合并 k/n 做 Wilson 区间
    k_total = sum(round(b["b1_termination_rate"] * N_EVAL_PER_BLOCK)
                  for b in per_block)
    b1_pooled = {
        "k": int(k_total), "n": int(N_EVAL_PER_BLOCK * len(per_block)),
        "pooled_rate": k_total / (N_EVAL_PER_BLOCK * len(per_block)),
        "wilson95": wilson_ci(k_total, N_EVAL_PER_BLOCK * len(per_block)),
        "per_block_sd": float(np.std(b1s, ddof=1)),
    }

    return {
        "policy": label,
        "per_block": per_block,
        "cross_block": {
            "mean_return": mean_sd_ci(rets),
            "b3_time_ratio": mean_sd_ci(b3s),
            "b1_termination_rate": mean_sd_ci(b1s),
        },
        "pooled": {"returns": pooled, "b1_termination": b1_pooled},
    }


# ---------------------------------------------------------------------- #
# 实验二：规划变体（全部作用于 ModelBasedAgent 学到的同一份 P̂/R̂）
# ---------------------------------------------------------------------- #
def extract_model(agent):
    """把 ModelBasedAgent 的 P̂/R̂ 读成 numpy 数组（矩阵形式便于向量化）。"""
    n_s, n_a = len(STATE_NAMES), len(ACTION_NAMES)
    P = np.zeros((n_s, n_a, n_s))
    R = np.zeros((n_s, n_a))
    for si, s in enumerate(STATE_NAMES):
        for ai, a in enumerate(ACTION_NAMES):
            R[si, ai] = agent.get_reward(s, a)
            for ns, p in agent.get_transition(s, a).items():
                P[si, ai, STATE_NAMES.index(ns)] = p
    return P, R


def sync_value_iteration(P, R, theta=THETA, max_iter=MAX_ITER):
    """同步扫描 VI（与提交实现同公式，向量化加速；用于独立校验）。"""
    n_s, n_a, _ = P.shape
    V = np.zeros(n_s)
    sweeps = 0
    for sweeps in range(1, max_iter + 1):
        Q = R + GAMMA * P @ V            # (n_s, n_a)
        new_V = Q.max(axis=1)
        delta = float(np.abs(new_V - V).max())
        V = new_V
        if delta < theta:
            break
    return V, sweeps


def async_value_iteration(P, R, theta=THETA, max_sweeps=MAX_ITER * 10):
    """异步原地 VI（Gauss-Seidel）：逐状态更新、立即使用最新 V。"""
    n_s, n_a, _ = P.shape
    V = np.zeros(n_s)
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        delta = 0.0
        for si in range(n_s):
            q = R[si] + GAMMA * P[si] @ V
            new_v = float(q.max())
            delta = max(delta, abs(new_v - V[si]))
            V[si] = new_v                 # 原地：后续状态立即用新值
        if delta < theta:
            break
    return V, sweeps


def q_value_iteration(P, R, theta=THETA, max_iter=MAX_ITER * 10):
    """Q 直接迭代：对 Q 表施加 Bellman 最优算子（同步扫描）。"""
    n_s, n_a, _ = P.shape
    Q = np.zeros((n_s, n_a))
    sweeps = 0
    for sweeps in range(1, max_iter + 1):
        V = Q.max(axis=1)
        new_Q = R + GAMMA * P @ V
        delta = float(np.abs(new_Q - Q).max())
        Q = new_Q
        if delta < theta:
            break
    return Q, sweeps


def dyna_sampled_q_learning(P, R, n_sim=DYNA_SIM_TRANSITIONS,
                            alpha=0.5, seed=DYNA_SEED, alpha_decay=None):
    """Dyna 式采样 Q-Learning：从学到的模型模拟 (s,a,r,s') 做表格更新。

    经典 Dyna 形态：均匀采 (s,a)，由 P̂ 采 s'、加 R̂ 的期望奖励
    （P̂/R̂ 本身是期望模型，采样 s' 即可体现模型随机性）。
    ``alpha_decay`` 给定时用 Robbins-Monro 式衰减 α_t = α/(1+t·decay)，
    使随机逼近收敛（恒定 α 只会收敛到 Q* 的邻域）。
    """
    rng = np.random.default_rng(seed)
    n_s, n_a, _ = P.shape
    Q = np.zeros((n_s, n_a))
    cum = np.cumsum(P, axis=2)            # 每行累积分布，用于逆变换采样
    for t in range(n_sim):
        si = int(rng.integers(n_s))
        ai = int(rng.integers(n_a))
        nsi = int(np.searchsorted(cum[si, ai], rng.random()))
        nsi = min(nsi, n_s - 1)
        r = R[si, ai]
        target = r                        # 模型采样无终止标志；B1 行不动点
        if STATE_NAMES[nsi] != "B1":
            target = r + GAMMA * float(Q[nsi].max())
        a_t = alpha / (1.0 + t * alpha_decay) if alpha_decay else alpha
        Q[si, ai] = (1.0 - a_t) * Q[si, ai] + a_t * target
    return Q, n_sim


def policy_from_q(Q):
    """argmax 平局取课程约定顺序第一个动作（严格大于才更新）。"""
    n_s = Q.shape[0]
    pol = []
    for si in range(n_s):
        best = 0
        for ai in range(1, Q.shape[1]):
            if Q[si, ai] > Q[si, best]:
                best = ai
        pol.append(best)
    return pol


def planning_experiments(agent):
    P, R = extract_model(agent)

    # (a) 提交协议的同步扫描 VI（直接调用提交实现，保证基准一致）
    agent_sync_V = [agent.get_value(s) for s in STATE_NAMES]
    sync_sweeps = agent.value_iteration_iterations
    sync_Q = np.array(agent.get_q_table())

    # 向量化校验：确认与提交实现一致（防实现偏差）
    V_check, check_sweeps = sync_value_iteration(P, R)
    sync_check_max_diff = float(np.abs(V_check - np.array(agent_sync_V)).max())

    # (b) 异步原地 VI
    V_async, async_sweeps = async_value_iteration(P, R)
    # (c) Q 直接迭代
    Q_qiter, qiter_sweeps = q_value_iteration(P, R)
    # (d) Dyna 式采样 Q-Learning：恒定 α 与衰减 α 两档
    Q_dyna, dyna_updates = dyna_sampled_q_learning(P, R)
    Q_dyna_decayed, _ = dyna_sampled_q_learning(P, R, alpha_decay=1e-3)

    pol_sync = policy_from_q(sync_Q)
    # 异步 VI 收敛后的 V 对应的 Q 与策略
    Q_async = R + GAMMA * P @ V_async
    pol_async = policy_from_q(Q_async)
    pol_qiter = policy_from_q(Q_qiter)
    pol_dyna = policy_from_q(Q_dyna)

    V_sync = np.array(agent_sync_V)
    V_from_qiter = Q_qiter.max(axis=1)

    def agree(p1, p2):
        return [STATE_NAMES[i] for i in range(len(p1)) if p1[i] == p2[i]]

    return {
        "model": {"P_shape": list(P.shape),
                  "total_prob_error": float(np.abs(P.sum(axis=2) - 1).max())},
        "sync_vi": {
            "sweeps": int(sync_sweeps),
            "state_updates": int(sync_sweeps * len(STATE_NAMES)),
            "vectorized_check_max_abs_diff": sync_check_max_diff,
            "vectorized_check_sweeps": int(check_sweeps),
            "V": {s: float(v) for s, v in zip(STATE_NAMES, agent_sync_V)},
            "policy": {s: ACTION_NAMES[a] for s, a in zip(STATE_NAMES, pol_sync)},
        },
        "async_vi": {
            "sweeps": int(async_sweeps),
            "state_updates": int(async_sweeps * len(STATE_NAMES)),
            "max_abs_V_diff_vs_sync": float(np.abs(V_async - V_sync).max()),
            "max_abs_Q_diff_vs_sync_vi_q": float(np.abs(Q_async - sync_Q).max()),
            "policy_agree_states": agree(pol_sync, pol_async),
            "policy_identical": pol_sync == pol_async,
            "speedup_sweeps_vs_sync": float(sync_sweeps / async_sweeps),
        },
        "q_iteration": {
            "sweeps": int(qiter_sweeps),
            "q_updates": int(qiter_sweeps * Q_qiter.size),
            "max_abs_Q_diff_vs_sync_vi_q": float(np.abs(Q_qiter - sync_Q).max()),
            "max_abs_V_diff_vs_sync": float(np.abs(V_from_qiter - V_sync).max()),
            "policy_agree_states": agree(pol_sync, pol_qiter),
            "policy_identical": pol_sync == pol_qiter,
        },
        "dyna_sampled_q": {
            "n_simulated_transitions": int(dyna_updates),
            "max_abs_Q_diff_vs_sync_vi_q": float(np.abs(Q_dyna - sync_Q).max()),
            "policy_agree_states": agree(pol_sync, pol_dyna),
            "policy_identical": pol_sync == pol_dyna,
        },
        "dyna_sampled_q_decayed_alpha": {
            "alpha_schedule": "0.5 / (1 + 1e-3 * t)",
            "max_abs_Q_diff_vs_sync_vi_q": float(np.abs(Q_dyna_decayed - sync_Q).max()),
            "policy_agree_states": agree(pol_sync, policy_from_q(Q_dyna_decayed)),
            "policy_identical": pol_sync == policy_from_q(Q_dyna_decayed),
        },
        "notes": "所有变体作用于同一份 P̂/R̂；同步扫描 VI 为提交协议基准，"
                 "其余仅作为附加实验留档，不影响提交文件默认行为。",
    }


# ---------------------------------------------------------------------- #
# 实验三：bootstrap 模型不确定性
# ---------------------------------------------------------------------- #
def bootstrap_policy_stability(episodes):
    """有放回重采 episode（B=30），重学模型 + 同步 VI，统计策略稳定率与 V(B4) 区间。"""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(episodes)
    policies = []          # 每个 bootstrap 样本的策略（索引形式）
    v_b4 = []
    q_tables = []
    for b in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, size=n)
        sample = [episodes[i] for i in idx]
        agent = ModelBasedAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=GAMMA)
        agent.learn_model(sample)
        agent.run_value_iteration(theta=THETA, max_iterations=MAX_ITER)
        q = np.array(agent.get_q_table())
        policies.append(policy_from_q(q))
        v_b4.append(agent.get_value("B4"))
        q_tables.append(q)

    # 全量策略
    full_agent = ModelBasedAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=GAMMA)
    full_agent.learn_model(episodes)
    full_agent.run_value_iteration(theta=THETA, max_iterations=MAX_ITER)
    full_q = np.array(full_agent.get_q_table())
    full_pol = policy_from_q(full_q)

    # 每个状态的稳定率与 bootstrap 动作分布
    per_state = {}
    for si, s in enumerate(STATE_NAMES):
        actions = [ACTION_NAMES[p[si]] for p in policies]
        dist = {a: actions.count(a) / BOOTSTRAP_B for a in set(actions)}
        # Q 差 margin：全量最优 vs 次优
        order = np.argsort(full_q[si])[::-1]
        margin_full = float(full_q[si, order[0]] - full_q[si, order[1]])
        # bootstrap 中"全量最优动作仍是该样本最优"的频率已由稳定率刻画；
        # 另统计全量最优与次优的 Q 差在 bootstrap 中的分布（翻转风险量化）
        margins, flip_sign = [], 0
        for q in q_tables:
            m = float(q[si, order[0]] - q[si, order[1]])
            margins.append(m)
            if m < 0:
                flip_sign += 1
        per_state[s] = {
            "full_policy": ACTION_NAMES[full_pol[si]],
            "stability_rate": sum(1 for p in policies if p[si] == full_pol[si]) / BOOTSTRAP_B,
            "bootstrap_action_dist": dist,
            "full_q_margin_top2": margin_full,
            "bootstrap_margin_mean": float(np.mean(margins)),
            "bootstrap_margin_std": float(np.std(margins, ddof=1)),
            "bootstrap_margin_negative_freq": flip_sign / BOOTSTRAP_B,
        }

    vb4 = np.asarray(v_b4)
    v_b4_ci = [float(np.percentile(vb4, 2.5)), float(np.percentile(vb4, 97.5))]

    # "最不稳定状态"排序：稳定率并列（本任务全部为 1.0）且无实际翻转
    # （margin<0 频率全 0）时，直接用 bootstrap 的 top-2 Q 差均值衡量
    # 翻转风险——margin 越小，模型噪声越容易翻转该状态的 argmax。
    # flip_ranking 按 (稳定率升序 → neg_freq 降序 → bootstrap top-2 Q 差
    # 升序) 排列：排在最前 = 翻转风险最大。B1 是终止状态，其动作无
    # 实际意义，从排名中排除。
    candidates = [s for s in STATE_NAMES if s != "B1"]
    flip_ranking = sorted(
        candidates,
        key=lambda s: (per_state[s]["stability_rate"],
                       -per_state[s]["bootstrap_margin_negative_freq"],
                       per_state[s]["bootstrap_margin_mean"]),
    )
    least_stable = flip_ranking[0]

    return {
        "B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "full_policy": {s: ACTION_NAMES[a] for s, a in zip(STATE_NAMES, full_pol)},
        "full_vi_iterations": int(full_agent.value_iteration_iterations),
        "per_state": per_state,
        "least_stable_state": least_stable,
        "flip_risk_ranking_most_to_least": [
            {"state": s,
             "full_policy": per_state[s]["full_policy"],
             "bootstrap_margin_mean": per_state[s]["bootstrap_margin_mean"],
             "stability_rate": per_state[s]["stability_rate"]}
            for s in flip_ranking
        ],
        "v_b4": {
            "mean": float(vb4.mean()),
            "sd": float(vb4.std(ddof=1)),
            "ci95_percentile": v_b4_ci,
            "samples": [float(x) for x in v_b4],
        },
    }


# ---------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------- #
def main():
    print(f"[1/4] 采集统一数据预算 collect_random_episodes({N_EPISODES}, seed={DATA_SEED}) ...")
    episodes = collect_random_episodes(N_EPISODES, seed=DATA_SEED)
    print(f"      共 {sum(len(ep) for ep in episodes)} 条转移")

    print("[2/4] Part A / Part B 学习 + 5 块独立评估（每块 500 条）...")
    agent_a = ModelBasedAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=GAMMA)
    agent_a.learn_model(episodes)
    agent_a.run_value_iteration(theta=THETA, max_iterations=MAX_ITER)
    pol_a = {s: agent_a.get_policy(s) for s in STATE_NAMES}

    agent_b = QLearningAgent(states=STATE_NAMES, actions=ACTION_NAMES,
                             gamma=GAMMA, alpha=0.5)
    agent_b.learn(episodes, mode="fixed")
    pol_b = {s: agent_b.get_policy(s) for s in STATE_NAMES}

    blocks_a, rets_a = eval_blocks_with_counts(pol_a)
    blocks_b, rets_b = eval_blocks_with_counts(pol_b)
    summary_a = summarize_blocks(blocks_a, rets_a, "part_a")
    summary_b = summarize_blocks(blocks_b, rets_b, "part_b")
    for tag, summ in (("Part A", summary_a), ("Part B", summary_b)):
        cb = summ["cross_block"]
        print(f"      {tag}: return={cb['mean_return']['mean']:.2f}"
              f" ± {cb['mean_return']['sd']:.3f} (sd, 跨块), "
              f"B3={cb['b3_time_ratio']['mean']:.4f} ± {cb['b3_time_ratio']['sd']:.4f}, "
              f"B1率={cb['b1_termination_rate']['mean']:.4f} ± {cb['b1_termination_rate']['sd']:.4f}")
        print(f"            B1 Wilson95={summ['pooled']['b1_termination']['wilson95']}")

    print("[3/4] 规划变体对比（同步 VI 基准 / 异步 VI / Q 迭代 / Dyna 采样）...")
    planning = planning_experiments(agent_a)
    print(f"      同步 VI {planning['sync_vi']['sweeps']} 轮 | "
          f"异步 VI {planning['async_vi']['sweeps']} 轮 "
          f"(max|ΔV|={planning['async_vi']['max_abs_V_diff_vs_sync']:.2e}) | "
          f"Q 迭代 {planning['q_iteration']['sweeps']} 轮 "
          f"(max|ΔQ|={planning['q_iteration']['max_abs_Q_diff_vs_sync_vi_q']:.2e}) | "
          f"Dyna 采样 {planning['dyna_sampled_q']['n_simulated_transitions']} 次 "
          f"(max|ΔQ|={planning['dyna_sampled_q']['max_abs_Q_diff_vs_sync_vi_q']:.3f})")
    print(f"      策略一致: 异步={planning['async_vi']['policy_identical']}, "
          f"Q迭代={planning['q_iteration']['policy_identical']}, "
          f"Dyna={planning['dyna_sampled_q']['policy_identical']}")

    print(f"[4/4] Bootstrap（B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED}）模型不确定性 ...")
    boot = bootstrap_policy_stability(episodes)
    print(f"      最不稳定状态: {boot['least_stable_state']}")
    for s in STATE_NAMES:
        st = boot["per_state"][s]
        print(f"      {s}: 稳定率={st['stability_rate']:.2f}, "
              f"全量策略={st['full_policy']}, top2 Q差={st['full_q_margin_top2']:.4f}, "
              f"bootstrap Q差<0 频率={st['bootstrap_margin_negative_freq']:.2f}")
    print(f"      V(B4) 95% 百分位区间: {boot['v_b4']['ci95_percentile']}")

    results = {
        "config": {
            "data_seed": DATA_SEED, "n_episodes": N_EPISODES, "gamma": GAMMA,
            "n_eval_per_block": N_EVAL_PER_BLOCK, "eval_seed_bases": EVAL_SEED_BASES,
            "theta": THETA, "max_iterations": MAX_ITER,
            "t_crit_df4": T_CRIT_DF4, "z_95": Z_95,
            "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED,
                          "resample": "有放回重采 2000 条 episode"},
            "dyna": {"n_simulated_transitions": DYNA_SIM_TRANSITIONS,
                     "alpha": 0.5, "seed": DYNA_SEED},
        },
        "eval_rigor": {"part_a": summary_a, "part_b": summary_b},
        "planning_variants": planning,
        "bootstrap_uncertainty": boot,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
