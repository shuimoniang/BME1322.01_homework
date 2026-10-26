"""Builder B 附加实验 —— 离线 Q-Learning 学习率调度探索。

目的：v1 基线（固定 α=0.5 顺序扫一遍统一数据）贪心策略正确，但 Q 绝对值
偏离 Q*（讲义第 9 章收敛三条件之一是学习率满足 Robbins-Monro 条件）。
本实验在**不改提交文件协议**的前提下，用临时变体类探索学习率调度维度：

变体清单
--------
1. 固定 α ∈ {0.1, 0.3, 0.5, 0.7, 1.0}（直接用提交文件 QLearningAgent，
   协议行为不变）。
2. 每对计数衰减（对第 n 次被更新的 (s,a) 用 α(n)）：
   - harmonic:      α(n) = 1/n                （标准 Robbins-Monro）
   - power_0.6:     α(n) = max(0.05, 1/n^0.6) （衰减更慢，兼顾收敛速度）
   - episode_reset: 每 episode 把每对计数清零，episode 内 α(n) = 0.5/n
                    （episode 开始时 α=0.5，对内衰减）
   全部以临时变体类实现，更新式严格保持协议形式 Q ← (1−α)Q + α[r + γ max Q(s')]，
   只是 α 由外部调度器按更新次数给出（等价于让调用方控制传给 update 的 α）。

参照系
------
Q* 参考值用 Part A 学到的模型算出的 Q̂ 表代替（run_model_based 的
get_q_table）。报告 max|Q − Q̂| 与 L2 距离；为公平起见同时给出
"全部 (s,a)" 与 "仅被数据更新过的 (s,a)"（visited）两种口径——
B1 为终止状态，其 (s,a) 在数据中从未出现，Q-Learning 侧恒为 0，
而 Part A 侧按未访问先验（均匀 P̂ + R̂=0）给出非零 Q̂，全部口径会把
这部分与学习率无关的系统性偏差也计入。

数据与评估协议（与主实验完全同口径）
------------------------------------
- 统一数据预算 collect_random_episodes(2000, seed=2026)；另用前 500 条
  episode 重复全部对比（考察小样本下哪个调度更稳健）。
- 评估：500 条轨迹、种子 90000+i；指标为无折扣累计回报平均、
  B3 时间占比、B1 终止率。
- 数值部分只依赖标准库 + numpy。

运行（从仓库根目录）：
    python assignment3/experiments/builder_b_lr_schedule.py
输出：
    assignment3/experiments/builder_b_lr_schedule_results.json
"""

import json
import os
import random
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSIGNMENT_DIR = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_ASSIGNMENT_DIR)
for _p in (_ROOT, _ASSIGNMENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from glucose_env import (  # noqa: E402
    ACTION_NAMES, STATE_NAMES, GlucoseEnv, collect_random_episodes,
)
from model_based import ModelBasedAgent  # noqa: E402
from q_learning import QLearningAgent  # noqa: E402

DATA_SEED = 2026
N_EVAL_EPISODES = 500
EVAL_SEED_BASE = 90000
GAMMA = 0.9
OUT_PATH = os.path.join(_HERE, "builder_b_lr_schedule_results.json")

NON_TERMINAL = [s for s in STATE_NAMES if s != "B1"]


# ---------------------------------------------------------------------- #
# 评估（与 main.py evaluate_policy 完全同口径，标准库实现）
# ---------------------------------------------------------------------- #
def evaluate_policy(policy, n_episodes=N_EVAL_EPISODES, seed_base=EVAL_SEED_BASE):
    """把确定性策略放进模拟器跑 500 条轨迹（种子 90000+i）。"""
    returns, b1_count, b3_steps, total_steps, step_counts = [], 0, 0, 0, []
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
    }


# ---------------------------------------------------------------------- #
# 变体类：每对计数调度 α(n)（临时实验类，不改提交文件）
# ---------------------------------------------------------------------- #
class ScheduledQLearningAgent(QLearningAgent):
    """按"每个 (s,a) 第 n 次被更新时用 α(n)"调度学习率的临时变体。

    更新式严格保持协议形式 Q ← (1−α)Q + α[r + γ max Q(s')]；
    schedule: 可调用对象 n -> alpha（n 从 1 开始计）。
    episode_reset=True 时每个 episode 开始把每对 episode 内计数清零
    （用于"每 episode 重置为 0.5 但对内衰减"变体）。
    """

    def __init__(self, states, actions, schedule, gamma=GAMMA, seed=None,
                 episode_reset=False):
        super().__init__(states, actions, gamma=gamma, alpha=0.5, seed=seed)
        self.schedule = schedule
        self.episode_reset = episode_reset
        n_a = self._n_actions
        # 全局每对更新计数（跨 episode 累计）
        self._pair_counts = [[0] * n_a for _ in range(self._n_states)]
        # episode 内每对计数（episode_reset 变体用）
        self._episode_pair_counts = [[0] * n_a for _ in range(self._n_states)]

    def learn(self, episodes: list, mode: str = "fixed") -> None:
        for episode in episodes:
            if self.episode_reset:
                self._episode_pair_counts = [
                    [0] * self._n_actions for _ in range(self._n_states)
                ]
            for (state, action, reward, next_state, done) in episode:
                self.update(state, action, reward, next_state, done)

    def update(self, state, action, reward, next_state, done: bool) -> None:
        """单步更新：α 由调度器按该 (s,a) 的更新次数给出，形式不变。"""
        si = self._require_state(state)
        ai = self._require_action(action)
        nsi = self._require_state(next_state)

        r = float(reward)
        if done:
            target = r
        else:
            target = r + self.gamma * max(self.q[nsi])

        self._pair_counts[si][ai] += 1
        self._episode_pair_counts[si][ai] += 1
        n = (self._episode_pair_counts[si][ai] if self.episode_reset
             else self._pair_counts[si][ai])
        alpha = float(self.schedule(n))

        old = self.q[si][ai]
        self.q[si][ai] = (1.0 - alpha) * old + alpha * target
        self._visited[si][ai] = True
        self.update_count += 1


# ---------------------------------------------------------------------- #
# 变体定义
# ---------------------------------------------------------------------- #
def make_variants():
    """返回 [(variant_id, description, factory)]，factory(episodes)->agent。"""

    def fixed_factory(alpha):
        def factory(episodes):
            agent = QLearningAgent(states=STATE_NAMES, actions=ACTION_NAMES,
                                   gamma=GAMMA, alpha=alpha)
            agent.learn(episodes, mode="fixed")
            return agent
        return factory

    def scheduled_factory(schedule, episode_reset=False):
        def factory(episodes):
            agent = ScheduledQLearningAgent(STATE_NAMES, ACTION_NAMES,
                                            schedule=schedule,
                                            gamma=GAMMA,
                                            episode_reset=episode_reset)
            agent.learn(episodes, mode="fixed")
            return agent
        return factory

    variants = []
    for alpha in (0.1, 0.3, 0.5, 0.7, 1.0):
        variants.append((
            f"fixed_alpha_{alpha}",
            f"固定 α={alpha}（提交文件 QLearningAgent，协议行为）",
            fixed_factory(alpha),
        ))
    variants.append((
        "harmonic_1_over_n",
        "每对计数衰减 α(n)=1/n（标准 Robbins-Monro）",
        scheduled_factory(lambda n: 1.0 / n),
    ))
    variants.append((
        "power_0.6_floor_0.05",
        "每对计数衰减 α(n)=max(0.05, 1/n^0.6)",
        scheduled_factory(lambda n: max(0.05, 1.0 / n ** 0.6)),
    ))
    variants.append((
        "episode_reset_0.5_over_n",
        "每 episode 重置为 0.5 但对内衰减 α(n)=0.5/n",
        scheduled_factory(lambda n: 0.5 / n, episode_reset=True),
    ))
    return variants


# ---------------------------------------------------------------------- #
# 单次实验：一个数据预算下的全部变体
# ---------------------------------------------------------------------- #
def q_distances(q_table, q_ref, visited):
    """max|Q−Q̂| 与 L2 距离；分别给全部 (s,a) 与 visited (s,a) 口径。"""
    q = np.asarray(q_table, dtype=float)
    ref = np.asarray(q_ref, dtype=float)
    diff = q - ref
    mask = np.asarray(visited, dtype=bool)
    out = {
        "max_abs_all": float(np.max(np.abs(diff))),
        "l2_all": float(np.linalg.norm(diff)),
        "max_abs_visited": float(np.max(np.abs(diff[mask]))),
        "l2_visited": float(np.linalg.norm(diff[mask])),
        "n_visited_pairs": int(mask.sum()),
    }
    return out


def run_variants(episodes, q_ref, part_a_policy):
    """在给定数据上跑全部变体并汇总指标。"""
    results = {}
    for vid, desc, factory in make_variants():
        agent = factory(episodes)
        q_table = agent.get_q_table()
        visited = agent.get_visited()
        # 评估需要全部状态（含 B1，虽然 episode 进入 B1 即终止）；
        # 与 Part A 的一致性比较只看非终止状态。
        policy_map = {s: agent.get_policy(s) for s in STATE_NAMES}
        disagree = [s for s in NON_TERMINAL if policy_map[s] != part_a_policy[s]]
        eval_res = evaluate_policy(lambda s, rng: policy_map[s])
        results[vid] = {
            "description": desc,
            "policy": policy_map,
            "policy_agrees_part_a": len(disagree) == 0,
            "disagree_states": disagree,
            "eval": eval_res,
            "q_distance_vs_qhat": q_distances(q_table, q_ref, visited),
            "q_table": {STATE_NAMES[si]: {ACTION_NAMES[ai]: q_table[si][ai]
                                          for ai in range(len(ACTION_NAMES))}
                        for si in range(len(STATE_NAMES))},
        }
    return results


def run_part_a(episodes):
    """Part A：学模型 + 价值迭代，返回 Q̂ 参考表、策略与评估。"""
    agent = ModelBasedAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=GAMMA)
    agent.learn_model(episodes)
    agent.run_value_iteration(theta=1e-4, max_iterations=1000)
    q_ref = agent.get_q_table()
    policy_map = {s: agent.get_policy(s) for s in STATE_NAMES}
    eval_res = evaluate_policy(lambda s, rng: policy_map[s])
    return {
        "q_ref": q_ref,
        "policy": policy_map,
        "vi_iterations": agent.value_iteration_iterations,
        "eval": eval_res,
    }


# ---------------------------------------------------------------------- #
# 结论汇总
# ---------------------------------------------------------------------- #
def summarize(block, part_a_eval):
    """对单个数据预算的变体结果排出结论字段。"""
    visited_max = {vid: r["q_distance_vs_qhat"]["max_abs_visited"]
                   for vid, r in block.items()}
    visited_l2 = {vid: r["q_distance_vs_qhat"]["l2_visited"]
                  for vid, r in block.items()}
    best_max = min(visited_max, key=visited_max.get)
    best_l2 = min(visited_l2, key=visited_l2.get)
    agreeing = [vid for vid, r in block.items() if r["policy_agrees_part_a"]]
    best_return = max(block, key=lambda v: block[v]["eval"]["mean_return"])
    return {
        "best_q_absolute_by_max_abs_visited": {
            "variant": best_max, "value": visited_max[best_max],
        },
        "best_q_absolute_by_l2_visited": {
            "variant": best_l2, "value": visited_l2[best_l2],
        },
        "variants_policy_agrees_part_a": agreeing,
        "best_mean_return": {
            "variant": best_return,
            "value": block[best_return]["eval"]["mean_return"],
        },
        "part_a_eval_reference": part_a_eval,
    }


# ---------------------------------------------------------------------- #
def main():
    print(f"[1/4] 采集统一数据预算：2000 条随机策略 episode (seed={DATA_SEED}) ...")
    episodes = collect_random_episodes(2000, seed=DATA_SEED)
    total_transitions = sum(len(ep) for ep in episodes)
    print(f"      共 {total_transitions} 条转移经验；另取前 500 条 episode 做小样本对比")

    print("[2/4] Part A 参照（2000 / 500 条数据各跑一次价值迭代）...")
    part_a_full = run_part_a(episodes)
    part_a_500 = run_part_a(episodes[:500])
    for tag, pa in (("2000", part_a_full), ("500", part_a_500)):
        e = pa["eval"]
        print(f"      n={tag}: 策略={pa['policy']}")
        print(f"             评估: return={e['mean_return']:.2f}, "
              f"B3={e['b3_time_ratio']:.3f}, B1率={e['b1_termination_rate']:.3f}")

    print("[3/4] 变体实验：2000 条数据 ...")
    results_full = run_variants(episodes, part_a_full["q_ref"],
                                part_a_full["policy"])
    for vid, r in results_full.items():
        d = r["q_distance_vs_qhat"]
        e = r["eval"]
        print(f"      {vid:28s} 一致={r['policy_agrees_part_a']} "
              f"return={e['mean_return']:7.2f} "
              f"max|dQ|(visited)={d['max_abs_visited']:.4f} "
              f"L2(visited)={d['l2_visited']:.4f}")

    print("[4/4] 变体实验：500 条数据（小样本稳健性）...")
    results_500 = run_variants(episodes[:500], part_a_500["q_ref"],
                               part_a_500["policy"])
    for vid, r in results_500.items():
        d = r["q_distance_vs_qhat"]
        e = r["eval"]
        print(f"      {vid:28s} 一致={r['policy_agrees_part_a']} "
              f"return={e['mean_return']:7.2f} "
              f"max|dQ|(visited)={d['max_abs_visited']:.4f} "
              f"L2(visited)={d['l2_visited']:.4f}")

    summary = {
        "n_episodes_2000": summarize(results_full, part_a_full["eval"]),
        "n_episodes_500": summarize(results_500, part_a_500["eval"]),
    }

    # 跨数据量的策略稳定性：同一变体在 500/2000 下是否都与对应 Part A 一致
    stability = {}
    for vid in results_full:
        stability[vid] = {
            "agrees_at_2000": results_full[vid]["policy_agrees_part_a"],
            "agrees_at_500": results_500[vid]["policy_agrees_part_a"],
            "stable_both": (results_full[vid]["policy_agrees_part_a"]
                            and results_500[vid]["policy_agrees_part_a"]),
        }
    stable_variants = [v for v, s in stability.items() if s["stable_both"]]

    out = {
        "experiment": "builder_b_lr_schedule",
        "config": {
            "data_seed": DATA_SEED, "n_episodes_full": 2000,
            "n_episodes_small": 500,
            "n_eval_episodes": N_EVAL_EPISODES,
            "eval_seed_base": EVAL_SEED_BASE,
            "gamma": GAMMA,
            "q_star_reference": "Part A run_model_based(...).get_q_table()",
            "distance_metrics_note": (
                "visited 口径只统计被数据更新过的 (s,a)；全部口径额外包含 "
                "B1（终止态）等未更新对，其 Q-Learning 侧恒为 0 而 Part A 侧"
                "按未访问先验非零，属与学习率无关的系统性偏差。"
            ),
        },
        "total_transitions": total_transitions,
        "part_a_full": {
            "policy": part_a_full["policy"],
            "vi_iterations": part_a_full["vi_iterations"],
            "eval": part_a_full["eval"],
            "q_hat_table": {STATE_NAMES[si]: {ACTION_NAMES[ai]:
                            part_a_full["q_ref"][si][ai]
                            for ai in range(len(ACTION_NAMES))}
                            for si in range(len(STATE_NAMES))},
        },
        "part_a_500": {
            "policy": part_a_500["policy"],
            "vi_iterations": part_a_500["vi_iterations"],
            "eval": part_a_500["eval"],
        },
        "variants_2000": results_full,
        "variants_500": results_500,
        "summary": summary,
        "policy_stability_across_budgets": stability,
        "stable_variants_both_budgets": stable_variants,
        "conclusions": {
            "q_absolute_closest": {
                "at_2000": summary["n_episodes_2000"]["best_q_absolute_by_l2_visited"],
                "at_500": summary["n_episodes_500"]["best_q_absolute_by_l2_visited"],
            },
            "most_stable_policy": stable_variants,
            "submission_default_alpha_note": (
                "提交文件 q_learning.py 的 update() 协议形式 Q<-(1-a)Q+a[...] "
                "与默认 alpha=0.5 必须保留；学习率调度只能作为报告附加实验"
                "（本脚本中的临时变体类），若引入需改接口才能生效，不改提交文件。"
            ),
        },
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已写入 {OUT_PATH}")

    # 控制台速览表
    print("\n===== 速览（visited 口径 Q 距离 / 评估）=====")
    print(f"{'变体':32s} {'n=2000: 一致 回报 max|dQ| L2':>34s} | "
          f"{'n=500: 一致 回报 max|dQ| L2':>32s}")
    for vid in results_full:
        r1, r0 = results_full[vid], results_500[vid]
        d1, d0 = r1["q_distance_vs_qhat"], r0["q_distance_vs_qhat"]
        e1, e0 = r1["eval"], r0["eval"]
        print(f"{vid:32s} "
              f"{str(r1['policy_agrees_part_a']):>5s} {e1['mean_return']:8.2f} "
              f"{d1['max_abs_visited']:8.4f} {d1['l2_visited']:8.4f} | "
              f"{str(r0['policy_agrees_part_a']):>5s} {e0['mean_return']:8.2f} "
              f"{d0['max_abs_visited']:8.4f} {d0['l2_visited']:8.4f}")


if __name__ == "__main__":
    main()
