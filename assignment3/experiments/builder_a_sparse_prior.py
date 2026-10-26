"""Builder A 消融实验：未访问/稀疏 (s,a) 先验假设对 Part A 的影响。

目的
----
v1 基线（讲义默认）：从未访问过的 (s,a) 取 P̂ = 均匀分布、R̂ = 0。
本实验在**不改动提交文件**的前提下，用临时变体类比较 4 种先验选择：

  1. uniform_r0     —— 未访问 (s,a)：P̂ 均匀 + R̂=0（v1 基线 / 讲义默认）
  2. laplace_a1     —— Dirichlet 平滑：对每个 (s,a,s') 加伪计数 α=1（Laplace），
                       P̂(s'|s,a) = (N(s,a,s')+1)/(N(s,a)+9)；R̂ 仍为样本均值（未访问 R̂=0）
  3. pessimistic_d1 —— 悲观先验：未访问动作的 Q̂ 减 δ=1（作用于价值迭代
                       备份与策略提取两处，均经 _q_from）
  4. optimistic_d1  —— 乐观先验：未访问动作的 Q̂ 加 δ=1（同上两处生效）

协议说明
--------
- 数据：collect_random_episodes(2000, seed=2026) 的前 n 条（n ∈
  {100, 300, 500, 1000, 2000}），与主实验同一数据集、取前缀，保证可比。
- 评估：**每个设置 200 条轨迹、种子 70000+i** —— 注意这与主实验口径
  （500 条、种子 90000+i）不同，本实验为辅助消融，结论只做相对比较。
- 模型估计仍只用经验五元组，不读取环境真实转移/奖励表。
- 数值部分只使用标准库（未用 numpy/scipy/sklearn）。

产出：builder_a_sparse_prior_results.json（本目录下）。
"""

import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_A3 = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_A3)
for _p in (_ROOT, _A3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from glucose_env import (  # noqa: E402
    STATE_NAMES, ACTION_NAMES, GlucoseEnv, collect_random_episodes,
)
from model_based import ModelBasedAgent  # noqa: E402

# ---------------------------------------------------------------------- #
# 实验配置
# ---------------------------------------------------------------------- #
DATA_SEED = 2026
N_DATA_FULL = 2000
DATA_SIZES = [100, 300, 500, 1000, 2000]
# 补充极小数据探测：主规格的 5 档下 36 个 (s,a) 全部被覆盖、先验不激活；
# 再往下探（10/25/50 条）才能观察到未访问对真正出现时各先验的行为差异。
PROBE_SIZES = [10, 25, 50]
GAMMA = 0.9
LAPLACE_ALPHA = 1.0
DELTA = 1.0

# 评估口径（注意：与主实验的 500 条 / 90000+i 不同，仅用于本消融的相对比较）
N_EVAL_EPISODES = 200
EVAL_SEED_BASE = 70000

NON_TERMINAL = [s for s in STATE_NAMES if s != "B1"]


# ---------------------------------------------------------------------- #
# 评估（与 main.py evaluate_policy 同口径，仅种子基数/条数不同）
# ---------------------------------------------------------------------- #
def evaluate_policy(policy, n_episodes=N_EVAL_EPISODES, seed_base=EVAL_SEED_BASE):
    """确定性策略入模拟器跑 n_episodes 条轨迹，返回均值指标（不含逐条回报）。"""
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
    mean_ret = sum(returns) / n
    var = sum((x - mean_ret) ** 2 for x in returns) / (n - 1)
    return {
        "mean_return": mean_ret,
        "std_return": var ** 0.5,
        "sem_return": (var / n) ** 0.5,
        "b3_time_ratio": b3_steps / total_steps,
        "b1_termination_rate": b1_count / n,
        "mean_steps": sum(step_counts) / n,
    }


# ---------------------------------------------------------------------- #
# 临时变体类（不改动提交文件 model_based.py）
# ---------------------------------------------------------------------- #
class LaplaceAgent(ModelBasedAgent):
    """Dirichlet(1,…,1) 平滑：对每个 (s,a,s') 加伪计数 α=1（Laplace 规则）。

    P̂(s'|s,a) = (N(s,a,s') + α) / (N(s,a) + α·|S|)
    R̂ 仍为样本均值，未访问 (s,a) 的 R̂ = 0（与基线一致，只改 P̂ 的平滑方式）。
    对未访问 (s,a) 退化为均匀分布，与基线相同——差异只出现在已访问对上。
    """

    def __init__(self, states, actions, gamma=GAMMA, alpha=LAPLACE_ALPHA, seed=None):
        super().__init__(states, actions, gamma=gamma, seed=seed)
        self.alpha = float(alpha)

    def learn_model(self, episodes):
        super().learn_model(episodes)  # 先按基线得到计数与 R̂
        a = self.alpha
        self._transition = []
        for si in range(self._n_states):
            row = []
            for ai in range(self._n_actions):
                n_sa = self._reward_count[si][ai]
                denom = n_sa + a * self._n_states
                dist = {
                    s_name: (self._counts[si][ai].get(s_name, 0) + a) / denom
                    for s_name in self.states
                }
                row.append(dist)
            self._transition.append(row)
        # 模型已重建，旧价值/策略失效（与基类 learn_model 末尾行为一致）
        self._values = None
        self._policy_idx = None
        self._q_table = None
        self._vi_iterations = 0


class DeltaQAgent(ModelBasedAgent):
    """未访问 (s,a) 的 Q̂ 加 sign·δ 的先验（悲观 sign=-1 / 乐观 sign=+1）。

    Q̂(s,a) = R̂(s,a) + γ Σ P̂ V(s') + sign·δ·1[N(s,a)=0]
    P̂/R̂ 的估计式与讲义完全一致；δ 只作用于"从未见过数据"的动作，
    且同时作用于价值迭代备份与策略提取（都经 _q_from，保持自洽）。
    """

    def __init__(self, states, actions, gamma=GAMMA, delta=DELTA, sign=-1.0, seed=None):
        super().__init__(states, actions, gamma=gamma, seed=seed)
        self.delta = float(delta)
        self.sign = float(sign)

    def _q_from(self, si, ai, v):
        q = super()._q_from(si, ai, v)
        if self._reward_count[si][ai] == 0:
            q += self.sign * self.delta
        return q


VARIANTS = {
    "uniform_r0": (
        "v1 基线：未访问 (s,a) 均匀 P̂ + R̂=0（讲义默认）",
        lambda: ModelBasedAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=GAMMA),
    ),
    "laplace_a1": (
        f"Dirichlet/Laplace 平滑 α={LAPLACE_ALPHA}：P̂ 加伪计数，R̂ 不变",
        lambda: LaplaceAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=GAMMA),
    ),
    "pessimistic_d1": (
        f"悲观先验：未访问动作 Q̂ − δ={DELTA}",
        lambda: DeltaQAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=GAMMA,
                            sign=-1.0),
    ),
    "optimistic_d1": (
        f"乐观先验：未访问动作 Q̂ + δ={DELTA}",
        lambda: DeltaQAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=GAMMA,
                            sign=+1.0),
    ),
}


# ---------------------------------------------------------------------- #
# 主实验
# ---------------------------------------------------------------------- #
def main():
    print(f"采集统一数据预算：{N_DATA_FULL} 条随机策略 episode (seed={DATA_SEED}) ...")
    episodes_full = collect_random_episodes(N_DATA_FULL, seed=DATA_SEED)
    print(f"共 {sum(len(ep) for ep in episodes_full)} 条转移经验")

    results = {
        "config": {
            "data_seed": DATA_SEED,
            "data_sizes": DATA_SIZES,
            "probe_sizes": PROBE_SIZES,
            "gamma": GAMMA,
            "laplace_alpha": LAPLACE_ALPHA,
            "delta": DELTA,
            "n_eval_episodes": N_EVAL_EPISODES,
            "eval_seed_base": EVAL_SEED_BASE,
            "eval_protocol_note": (
                "辅助消融口径：200 条评估轨迹、种子 70000+i；"
                "与主实验（500 条、90000+i）不同，结论仅做变体间相对比较"
            ),
        },
        "variants": {k: v[0] for k, v in VARIANTS.items()},
        "runs": {},
    }

    v1_policy = None  # uniform_r0 @ n=2000，即提交默认行为产出的策略

    for n in DATA_SIZES + PROBE_SIZES:
        eps = episodes_full[:n]
        n_trans = sum(len(ep) for ep in eps)
        print(f"\n=== 数据量 n={n}（{n_trans} 条转移）===")
        for name, (_desc, factory) in VARIANTS.items():
            agent = factory()
            agent.learn_model(eps)
            agent.run_value_iteration(theta=1e-4, max_iterations=1000)
            policy_map = {s: agent.get_policy(s) for s in STATE_NAMES}
            eval_res = evaluate_policy(lambda s, rng: policy_map[s])
            counts = agent.get_visit_counts()
            unvisited = sum(1 for row in counts for c in row if c == 0)
            min_count = min(c for row in counts for c in row)

            if name == "uniform_r0" and n == N_DATA_FULL:
                v1_policy = dict(policy_map)

            results["runs"].setdefault(name, {})[str(n)] = {
                "policy": policy_map,
                "vi_iterations": agent.value_iteration_iterations,
                "unvisited_pairs": unvisited,
                "min_sa_count": min_count,
                "eval": eval_res,
            }
            e = eval_res
            print(f"  {name:15s} return={e['mean_return']:8.2f}±{e['sem_return']:.2f}  "
                  f"B3={e['b3_time_ratio']:.3f}  B1率={e['b1_termination_rate']:.3f}  "
                  f"未访问对={unvisited:2d}  min N(s,a)={min_count}")

    # ------------------------------------------------------------------ #
    # 分析：与 v1 策略的一致性、小数据稳健性、2000 条下的最终比较
    # ------------------------------------------------------------------ #
    assert v1_policy is not None
    policy_diff = {
        name: {
            str(n): [s for s in NON_TERMINAL
                     if results["runs"][name][str(n)]["policy"][s] != v1_policy[s]]
            for n in DATA_SIZES
        }
        for name in VARIANTS
    }
    results["policy_diff_vs_v1"] = policy_diff

    small_sizes = [n for n in DATA_SIZES + PROBE_SIZES if n <= 500]
    robustness = {}
    for name in VARIANTS:
        rets = [results["runs"][name][str(n)]["eval"]["mean_return"] for n in small_sizes]
        robustness[name] = {
            "small_data_returns": {str(n): r for n, r in zip(small_sizes, rets)},
            "min_return_small": min(rets),
            "mean_return_small": sum(rets) / len(rets),
        }
    results["small_data_robustness"] = robustness

    e2000 = {name: results["runs"][name]["2000"]["eval"] for name in VARIANTS}
    best_2000 = max(e2000, key=lambda k: e2000[k]["mean_return"])
    results["n2000_comparison"] = {
        name: {
            "mean_return": e["mean_return"], "sem_return": e["sem_return"],
            "b3_time_ratio": e["b3_time_ratio"],
            "b1_termination_rate": e["b1_termination_rate"],
            "policy": results["runs"][name]["2000"]["policy"],
            "policy_same_as_v1": results["runs"][name]["2000"]["policy"] == v1_policy,
        }
        for name, e in e2000.items()
    }
    results["n2000_best_variant"] = best_2000

    # 综合结论（写入 JSON，供报告引用）
    ret_gap_2000 = (max(e["mean_return"] for e in e2000.values())
                    - min(e["mean_return"] for e in e2000.values()))
    max_sem = max(e["sem_return"] for e in e2000.values())
    small_best = max(robustness, key=lambda k: robustness[k]["min_return_small"])
    results["conclusion"] = {
        "small_data_most_robust": small_best,
        "n2000_best_variant": best_2000,
        "n2000_return_gap": ret_gap_2000,
        "n2000_max_sem": max_sem,
        "n2000_gap_within_noise": ret_gap_2000 < 2 * max_sem,
        "note": (
            "gap_within_noise 指最优与最差变体的回报差 < 2×最大标准误"
            "（200 条评估下的粗略显著性判据）"
        ),
    }

    # ------------------------------------------------------------------ #
    # 控制台摘要表
    # ------------------------------------------------------------------ #
    print("\n================ 摘要表（mean_return / B3占比 / B1终止率 / 与v1策略差异状态数）================")
    header = f"{'variant':15s}" + "".join(f"{'n=' + str(n):>26s}" for n in DATA_SIZES)
    print(header)
    for name in VARIANTS:
        cells = []
        for n in DATA_SIZES:
            r = results["runs"][name][str(n)]
            e = r["eval"]
            nd = len(policy_diff[name][str(n)])
            cells.append(f"{e['mean_return']:7.2f}/{e['b3_time_ratio']:.2f}/"
                         f"{e['b1_termination_rate']:.2f}/d{nd}")
        print(f"{name:15s}" + "".join(f"{c:>26s}" for c in cells))

    print("\n策略差异状态（vs v1 提交策略，非终止状态）：")
    for name in VARIANTS:
        for n in DATA_SIZES:
            d = policy_diff[name][str(n)]
            if d:
                pol = results["runs"][name][str(n)]["policy"]
                detail = ", ".join(f"{s}: v1={v1_policy[s]}→{pol[s]}" for s in d)
                print(f"  {name} @ n={n}: {detail}")
        if not any(policy_diff[name][str(n)] for n in DATA_SIZES):
            print(f"  {name}: 所有数据量下策略与 v1 完全一致")

    print("\nv1 提交策略（uniform_r0 @ n=2000）:",
          {s: v1_policy[s] for s in NON_TERMINAL})
    print("n=2000 各变体回报:",
          {k: round(v['mean_return'], 2) for k, v in e2000.items()})
    print("综合结论:", json.dumps(results["conclusion"], ensure_ascii=False))

    out_path = os.path.join(_HERE, "builder_a_sparse_prior_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已写入 {out_path}")


if __name__ == "__main__":
    main()
