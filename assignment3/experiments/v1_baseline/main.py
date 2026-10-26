"""BME1322 Assignment 3 —— 实验入口：复现报告中全部图表。

从仓库根目录（包含 ``assignment3/`` 的上一级目录）运行：

    python assignment3/main.py            # 全部实验（约 1-2 分钟）
    python assignment3/main.py --quick    # 快速冒烟测试（缩减评估条数）

产出：
    assignment3/figures/           全部图表（PNG）
    assignment3/figures/results.json   所有实验的数值结果（报告数据来源）
    assignment3/figures/results_summary.md  结果速览（撰写报告用）

实验清单（对应报告要求 4.1 / 4.2 节）：
  1. N(s,a) 访问次数分布（直方图），标注稀疏的 (s,a)。
  2. Part A：学到的策略表 + V̂ 表；策略入模拟器 500 条评估轨迹，
     报告平均回报（无折扣累计）、B3 时间占比、B1 终止率。
  3. 数据量消融：前 500 / 1000 / 2000 条 episode 学模型求策略，
     对比三者评估性能。
  4. Part B：最终 Q 表 + 贪心策略表；同样 500 条评估。
  5. 附加在线实验：在线采集数据训练 Q-Learning，对比
     ε=0 固定 / ε=0.3 固定 / ε=0.3 衰减(0.995, min 0.05) 三种探索设置，
     5 个随机种子，滑动窗口 50 的学习曲线。
  6. 对比分析：两路线策略一致率（每个非终止状态）、评估性能差异。

数据协议：统一数据预算 = collect_random_episodes(2000, seed=2026)，
Part A / Part B / 消融共用同一份数据，随机种子固定，结果可复现。
评估模拟器使用独立种子（不与训练数据种子重叠），评估策略本身是
确定性的，因此 500 条评估轨迹的种子固定为 90000+i。
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（Windows: 微软雅黑；缺失时退回系统默认，避免中文变方框）
from matplotlib import font_manager
for _font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
    if any(_font.lower() == f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

# main.py 位于 assignment3/ 内；课程环境文件在仓库根目录（assignment3/ 的上一级）
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from glucose_env import (  # noqa: E402
    ACTION_NAMES, STATE_NAMES, GlucoseEnv, collect_random_episodes,
)
from model_based import ModelBasedAgent  # noqa: E402
from q_learning import QLearningAgent  # noqa: E402

DATA_SEED = 2026
N_EPISODES = 2000
N_EVAL_EPISODES = 500
EVAL_SEED_BASE = 90000
STATE_LABELS = {
    "B1": "B1 严重低血糖", "B2": "B2 轻度低血糖", "B3": "B3 目标区间",
    "B4": "B4 略高", "B5": "B5 轻度高", "B6": "B6 中度高",
    "B7": "B7 较高高", "B8": "B8 高血糖", "B9": "B9 严重高",
}
ACTION_LABELS = {"w": "w 维持", "g": "g 补糖", "i1": "i1 小胰岛素", "i2": "i2 大胰岛素"}

FIG_DIR = os.path.join(_HERE, "figures")


# ---------------------------------------------------------------------- #
# 评估：把确定性策略放进模拟器跑 500 条轨迹
# ---------------------------------------------------------------------- #
def evaluate_policy(policy, n_episodes=N_EVAL_EPISODES, seed_base=EVAL_SEED_BASE):
    """评估策略（可为确定性或随机性）。

    policy 签名为 policy(state, rng)；确定性策略忽略 rng。
    每个 episode 用独立种子 rng = random.Random(seed_base + i)，保证
    随机策略基线跨进程可复现（不能用 hash(str)——字符串 hash 有进程级
    盐化，会导致结果不可复现）。

    返回 dict：mean_return（无折扣累计回报的直接平均）、
    b3_time_ratio（处于目标区间 B3 的时间比例）、
    b1_termination_rate（进入 B1 的终止率）、mean_steps。
    """
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
        "returns": returns,
    }


def random_policy_baseline(n_episodes=N_EVAL_EPISODES):
    """均匀随机策略基线（评估对比用）。"""
    return evaluate_policy(lambda s, rng: rng.choice(ACTION_NAMES), n_episodes)


# ---------------------------------------------------------------------- #
# 实验一：N(s,a) 访问次数分布
# ---------------------------------------------------------------------- #
def exp_visit_counts(episodes):
    agent = ModelBasedAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=0.9)
    agent.learn_model(episodes)
    counts = agent.get_visit_counts()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # 上：按 (s,a) 的计数热图（表格比直方图更能定位稀疏项）
    ax = axes[0]
    im = ax.imshow(counts, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(ACTION_NAMES)), [ACTION_LABELS[a] for a in ACTION_NAMES])
    ax.set_yticks(range(len(STATE_NAMES)), STATE_NAMES)
    for si in range(len(STATE_NAMES)):
        for ai in range(len(ACTION_NAMES)):
            ax.text(ai, si, str(counts[si][ai]), ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_title("N(s,a) 访问次数（2000 条随机策略 episode）")
    fig.colorbar(im, ax=ax, label="次数")

    # 下：传统直方图视角——计数值的分布，标注稀疏 (s,a)
    ax = axes[1]
    flat = [counts[si][ai] for si in range(len(STATE_NAMES))
            for ai in range(len(ACTION_NAMES))]
    bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    ax.hist(flat, bins=bins, edgecolor="black", color="#4C72B0")
    sparse = [(STATE_NAMES[si], ACTION_NAMES[ai], counts[si][ai])
              for si in range(len(STATE_NAMES))
              for ai in range(len(ACTION_NAMES)) if counts[si][ai] < 500]
    for name, act, c in sparse[:6]:
        ax.annotate(f"{name},{act}:{c}", xy=(c, 1), xytext=(c, 2.2),
                    fontsize=8, rotation=45, ha="center")
    ax.set_xlabel("N(s,a)（次）")
    ax.set_ylabel("(s,a) 对个数")
    ax.set_title("N(s,a) 直方图（<500 次为稀疏）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_visit_counts.png"), dpi=150)
    plt.close(fig)

    return {
        "counts_table": {STATE_NAMES[si]: {ACTION_NAMES[ai]: counts[si][ai]
                                           for ai in range(len(ACTION_NAMES))}
                         for si in range(len(STATE_NAMES))},
        "min": min(flat), "max": max(flat),
        "sparse_pairs": [{"state": s, "action": a, "count": c} for s, a, c in sparse],
    }


# ---------------------------------------------------------------------- #
# Part A：模型 + 价值迭代 + 评估
# ---------------------------------------------------------------------- #
def run_model_based(episodes, gamma=0.9):
    agent = ModelBasedAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=gamma)
    agent.learn_model(episodes)
    agent.run_value_iteration(theta=1e-4, max_iterations=1000)
    return agent


def exp_part_a(episodes):
    agent = run_model_based(episodes)
    policy_map = {s: agent.get_policy(s) for s in STATE_NAMES}
    v_map = {s: agent.get_value(s) for s in STATE_NAMES}
    eval_res = evaluate_policy(lambda s, rng: policy_map[s])

    # 策略/价值表图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"w": "#dddddd", "g": "#f4a261", "i1": "#2a9d8f", "i2": "#e76f51"}
    for si, s in enumerate(STATE_NAMES):
        a = policy_map[s]
        ax1.barh(si, 1, color=colors[a], edgecolor="black")
        ax1.text(0.5, si, f"{s} → {a}", ha="center", va="center", fontsize=10)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_title("Part A 学到的策略 π(s)")
    ax1.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax1.legend(handles, [ACTION_LABELS[a] for a in colors], loc="lower right",
               fontsize=8)

    ax2.plot(range(1, len(STATE_NAMES) + 1), [v_map[s] for s in STATE_NAMES],
             marker="o", color="#2a9d8f")
    ax2.set_xticks(range(1, len(STATE_NAMES) + 1), STATE_NAMES)
    ax2.set_xlabel("状态")
    ax2.set_ylabel("V(s)")
    ax2.set_title("Part A 价值迭代收敛后的 V(s)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_part_a_policy_value.png"), dpi=150)
    plt.close(fig)

    return agent, {
        "policy": policy_map, "values": v_map,
        "vi_iterations": agent.value_iteration_iterations,
        "eval": eval_res,
    }


# ---------------------------------------------------------------------- #
# 消融：前 500 / 1000 / 2000 条 episode
# ---------------------------------------------------------------------- #
def exp_ablation(episodes):
    out = {}
    for n in (500, 1000, 2000):
        agent = run_model_based(episodes[:n])
        policy_map = {s: agent.get_policy(s) for s in STATE_NAMES}
        eval_res = evaluate_policy(lambda s, rng: policy_map[s])
        counts = agent.get_visit_counts()
        n_unvisited = sum(1 for row in counts for c in row if c == 0)
        out[str(n)] = {
            "policy": policy_map,
            "eval": eval_res,
            "unvisited_pairs": n_unvisited,
            "vi_iterations": agent.value_iteration_iterations,
        }

    # 消融对比图
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ns = ["500", "1000", "2000"]
    for ax, key in zip(axes, ("mean_return", "b3_time_ratio", "b1_termination_rate")):
        vals = [out[n]["eval"][key] for n in ns]
        ax.bar(range(len(ns)), vals, color=["#e9c46a", "#2a9d8f", "#264653"])
        ax.set_xticks(range(len(ns)), [f"{n} 条" for n in ns])
        ax.set_title(key)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if key == "mean_return" else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("数据量消融：前 500 / 1000 / 2000 条 episode 学模型")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_ablation.png"), dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------- #
# Part B：Q-Learning + 评估
# ---------------------------------------------------------------------- #
def exp_part_b(episodes):
    agent = QLearningAgent(states=STATE_NAMES, actions=ACTION_NAMES,
                           gamma=0.9, alpha=0.5)
    agent.learn(episodes, mode="fixed")
    policy_map = {s: agent.get_policy(s) for s in STATE_NAMES}
    eval_res = evaluate_policy(lambda s, rng: policy_map[s])
    q_table = agent.get_q_table()

    # Q 表热图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    im = ax1.imshow(q_table, cmap="RdYlGn", aspect="auto")
    ax1.set_xticks(range(len(ACTION_NAMES)), [ACTION_LABELS[a] for a in ACTION_NAMES])
    ax1.set_yticks(range(len(STATE_NAMES)), STATE_NAMES)
    for si in range(len(STATE_NAMES)):
        for ai in range(len(ACTION_NAMES)):
            ax1.text(ai, si, f"{q_table[si][ai]:.2f}", ha="center",
                     va="center", fontsize=8)
    ax1.set_title("Part B 最终 Q 表（α=0.5, 离线顺序更新）")
    fig.colorbar(im, ax=ax1, label="Q 值")

    colors = {"w": "#dddddd", "g": "#f4a261", "i1": "#2a9d8f", "i2": "#e76f51"}
    for si, s in enumerate(STATE_NAMES):
        a = policy_map[s]
        ax2.barh(si, 1, color=colors[a], edgecolor="black")
        ax2.text(0.5, si, f"{s} → {a}", ha="center", va="center", fontsize=10)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_title("Part B 贪心策略 π(s)")
    ax2.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax2.legend(handles, [ACTION_LABELS[a] for a in colors], loc="lower right",
               fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_part_b_q_policy.png"), dpi=150)
    plt.close(fig)

    return agent, {
        "policy": policy_map,
        "q_table": {STATE_NAMES[si]: {ACTION_NAMES[ai]: q_table[si][ai]
                                      for ai in range(len(ACTION_NAMES))}
                    for si in range(len(STATE_NAMES))},
        "update_count": agent.update_count,
        "eval": eval_res,
    }


# ---------------------------------------------------------------------- #
# 附加在线实验：三种 ε 设置的学习曲线（5 种子，滑动窗口 50）
# ---------------------------------------------------------------------- #
def exp_online_exploration(seeds=(1, 2, 3, 4, 5), window=50, quick=False):
    settings = {
        "eps0_fixed": {"epsilon": 0.0, "epsilon_decay": 1.0, "epsilon_min": 0.0},
        "eps03_fixed": {"epsilon": 0.3, "epsilon_decay": 1.0, "epsilon_min": 0.0},
        "eps03_decay": {"epsilon": 0.3, "epsilon_decay": 0.995, "epsilon_min": 0.05},
    }
    n_online_eps = 300 if quick else 600
    curves = {k: {s: [] for s in seeds} for k in settings}

    for k, cfg in settings.items():
        for seed in seeds:
            env = GlucoseEnv(seed=seed * 1000 + 777)
            agent = QLearningAgent(states=STATE_NAMES, actions=ACTION_NAMES,
                                   gamma=0.9, alpha=0.5, seed=seed, **cfg)
            episode_returns = []
            for _ in range(n_online_eps):
                s = env.reset()
                total_r, done = 0.0, False
                while not done:
                    a = agent.select_action(s)   # ε-greedy 在线交互
                    ns, r, done, _ = env.step(a)
                    agent.update(s, a, r, ns, done)
                    total_r += r
                    s = ns
                agent.decay_epsilon()
                episode_returns.append(total_r)
            curves[k][seed] = episode_returns

    # 滑动窗口 50 平滑 + 种子平均
    def smooth(xs, w=window):
        out = []
        for i in range(len(xs)):
            lo = max(0, i - w + 1)
            out.append(sum(xs[lo:i + 1]) / (i + 1 - lo))
        return out

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"eps0_fixed": "#e76f51", "eps03_fixed": "#e9c46a",
              "eps03_decay": "#2a9d8f"}
    labels = {"eps0_fixed": "ε=0 固定（无探索）",
              "eps03_fixed": "ε=0.3 固定",
              "eps03_decay": "ε=0.3 衰减 (0.995, min 0.05)"}
    stats = {}
    for k in settings:
        all_smoothed = [smooth(curves[k][s]) for s in seeds]
        mean = [sum(col) / len(col) for col in zip(*all_smoothed)]
        lo = [min(col) for col in zip(*all_smoothed)]
        hi = [max(col) for col in zip(*all_smoothed)]
        ax.plot(range(len(mean)), mean, color=colors[k], label=labels[k])
        ax.fill_between(range(len(mean)), lo, hi, color=colors[k], alpha=0.15)
        stats[k] = {
            "final_100_mean": sum(mean[-100:]) / 100,
            "best_window_mean": max(mean),
            "per_seed_final_returns": {str(s): sum(curves[k][s][-100:]) / 100
                                       for s in seeds},
        }
    ax.set_xlabel("在线训练 episode")
    ax.set_ylabel("每条 episode 回报（滑动窗口 50 平滑）")
    ax.set_title(f"附加在线实验：不同探索设置的学习曲线（{len(seeds)} 个种子）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_online_exploration.png"), dpi=150)
    plt.close(fig)
    return stats


# ---------------------------------------------------------------------- #
# 对比分析：策略一致率 + 性能对比
# ---------------------------------------------------------------------- #
def exp_comparison(res_a, res_b):
    non_terminal = [s for s in STATE_NAMES if s != "B1"]
    agree = [s for s in non_terminal if res_a["policy"][s] == res_b["policy"][s]]
    disagree = [s for s in non_terminal if s not in agree]
    table = {s: {"A": res_a["policy"][s], "B": res_b["policy"][s],
                 "agree": s in agree} for s in non_terminal}
    return {
        "agreement_rate": len(agree) / len(non_terminal),
        "agree_states": agree,
        "disagree_states": disagree,
        "table": table,
        "eval": {
            "part_a": {k: v for k, v in res_a["eval"].items() if k != "returns"},
            "part_b": {k: v for k, v in res_b["eval"].items() if k != "returns"},
        },
    }


def exp_random_baseline():
    res = random_policy_baseline()
    return {k: v for k, v in res.items() if k != "returns"}


# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Assignment 3 experiments")
    parser.add_argument("--quick", action="store_true", help="快速冒烟测试")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    n_episodes = 200 if args.quick else N_EPISODES
    eval_seed_base = EVAL_SEED_BASE

    print(f"[1/6] 采集统一数据预算：{n_episodes} 条随机策略 episode (seed={DATA_SEED}) ...")
    episodes = collect_random_episodes(n_episodes, seed=DATA_SEED)
    total_transitions = sum(len(ep) for ep in episodes)
    print(f"      共 {total_transitions} 条转移经验")

    print("[2/6] 实验一：N(s,a) 访问统计 ...")
    visit_stats = exp_visit_counts(episodes)
    print(f"      N(s,a) 范围 [{visit_stats['min']}, {visit_stats['max']}]，"
          f"稀疏对 {len(visit_stats['sparse_pairs'])} 个")

    print("[3/6] Part A：学模型 + 价值迭代 + 评估 ...")
    _, res_a = exp_part_a(episodes)
    e = res_a["eval"]
    print(f"      策略: {res_a['policy']}")
    print(f"      评估: mean_return={e['mean_return']:.2f}, "
          f"B3占比={e['b3_time_ratio']:.3f}, B1终止率={e['b1_termination_rate']:.3f}")

    print("[4/6] 消融：500 / 1000 / 2000 ...")
    ablation = exp_ablation(episodes)
    for n in ("500", "1000", "2000"):
        e = ablation[n]["eval"]
        print(f"      n={n}: return={e['mean_return']:.2f}, "
              f"B3={e['b3_time_ratio']:.3f}, B1率={e['b1_termination_rate']:.3f}, "
              f"未访问对={ablation[n]['unvisited_pairs']}")

    print("[5/6] Part B：Q-Learning + 评估 ...")
    _, res_b = exp_part_b(episodes)
    e = res_b["eval"]
    print(f"      策略: {res_b['policy']}")
    print(f"      评估: mean_return={e['mean_return']:.2f}, "
          f"B3占比={e['b3_time_ratio']:.3f}, B1终止率={e['b1_termination_rate']:.3f}")

    print("[6/6] 附加在线实验（3 设置 x 5 种子）与随机基线 ...")
    online = exp_online_exploration(quick=args.quick)
    for k, v in online.items():
        print(f"      {k}: 后100条平均回报={v['final_100_mean']:.2f}")
    baseline = exp_random_baseline()
    print(f"      随机策略基线: return={baseline['mean_return']:.2f}, "
          f"B3={baseline['b3_time_ratio']:.3f}, B1率={baseline['b1_termination_rate']:.3f}")

    comparison = exp_comparison(res_a, res_b)
    print(f"策略一致率: {comparison['agreement_rate']:.2f} "
          f"(不一致: {comparison['disagree_states']})")

    results = {
        "config": {
            "data_seed": DATA_SEED, "n_episodes": n_episodes,
            "n_eval_episodes": N_EVAL_EPISODES, "eval_seed_base": eval_seed_base,
            "gamma": 0.9,
        },
        "total_transitions": total_transitions,
        "visit_stats": visit_stats,
        "part_a": res_a,
        "ablation": ablation,
        "part_b": res_b,
        "online_exploration": online,
        "random_baseline": baseline,
        "comparison": comparison,
    }
    out_path = os.path.join(FIG_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n全部结果已写入 {out_path}")

    # 速览 markdown（报告撰写用）
    lines = ["# 实验结果速览\n"]
    lines.append(f"- 总转移数: {total_transitions}")
    lines.append(f"- Part A 评估: {res_a['eval']['mean_return']:.2f} / "
                 f"B3={res_a['eval']['b3_time_ratio']:.3f} / "
                 f"B1率={res_a['eval']['b1_termination_rate']:.3f} "
                 f"(VI 迭代 {res_a['vi_iterations']} 轮)")
    lines.append(f"- Part B 评估: {res_b['eval']['mean_return']:.2f} / "
                 f"B3={res_b['eval']['b3_time_ratio']:.3f} / "
                 f"B1率={res_b['eval']['b1_termination_rate']:.3f}")
    for n in ("500", "1000", "2000"):
        e = ablation[n]["eval"]
        lines.append(f"- 消融 n={n}: return={e['mean_return']:.2f}, "
                     f"B3={e['b3_time_ratio']:.3f}, B1率={e['b1_termination_rate']:.3f}")
    for k, v in online.items():
        lines.append(f"- 在线 {k}: 后100条平均={v['final_100_mean']:.2f}")
    lines.append(f"- 随机基线: return={baseline['mean_return']:.2f}, "
                 f"B3={baseline['b3_time_ratio']:.3f}, B1率={baseline['b1_termination_rate']:.3f}")
    lines.append(f"- 策略一致率: {comparison['agreement_rate']:.2f}，"
                 f"不一致状态: {comparison['disagree_states']}")
    with open(os.path.join(FIG_DIR, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("完成。")


if __name__ == "__main__":
    main()
