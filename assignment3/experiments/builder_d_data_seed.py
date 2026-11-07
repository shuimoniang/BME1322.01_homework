"""Builder D 实验脚本 —— 离线主任务的数据采集种子鲁棒性。

从仓库根目录运行：
    python assignment3/experiments/builder_d_data_seed.py

只读 ``model_based.py`` / ``q_learning.py`` / ``glucose_env.py``，不修改任何
提交文件；结果写入 ``assignment3/experiments/builder_d_data_seed_results.json``，
图写入 ``assignment3/figures/fig8_data_seed_robustness.png``，
运行日志写入 ``assignment3/experiments/builder_d_run.log``。

实验内容：

主实验（builder_a/b/c 与 v3 主结果）固定用 collect_random_episodes(2000, seed=2026)
采集离线数据。本实验回答：换一批采集种子，两路线（Part A 模型+价值迭代、
Part B 离线 Q-Learning）学到的策略和评估表现是否稳定？

对 10 个采集种子 [2026, 1, 7, 42, 100, 314, 520, 1322, 2718, 9999]
（2026 为主实验种子，其余 9 个为鲁棒性检验），每个种子：
  1. collect_random_episodes(2000, seed=s) 采集数据；
  2. 训练 ModelBasedAgent + run_value_iteration(theta=1e-4) 得策略 A；
     训练 QLearningAgent（默认 alpha=0.5, mode="fixed"）得策略 B；
  3. 记录：两路线策略表（9 状态各自动作）、A/B 策略一致率
     （口径一：非终止状态 B2–B9 共 8 个，主口径；口径二：含终止态 B1 共 9 个）、
     评估四指标（与主实验相同的 500 条评估种子段 90000–90499，跨种子可比）、
     B4 状态的动作选择（v2 留档：B4 是全表 margin 最小的状态，重点观察翻转）、
     Part A 上 B4 的 top-2 Q margin、价值迭代轮数。
  4. 汇总：9 个非主种子与主种子（2026）策略的差异状态数、评估指标
     均值/标准差（样本 std, ddof=1）/最小最大值、B1 终止率最大值（安全关注点）、
     两路线一致率是否始终 100%。

约束遵守（CONSTRAINTS.md）：全部只用标准库 + numpy + matplotlib；不读取
环境真实转移/奖励表；提交文件默认行为不受影响。
"""

import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime

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

from glucose_env import (  # noqa: E402
    ACTION_NAMES, STATE_NAMES, GlucoseEnv, collect_random_episodes,
)
from model_based import ModelBasedAgent  # noqa: E402
from q_learning import QLearningAgent  # noqa: E402

# ---------------------------------------------------------------------- #
# 中文字体（沿用 main.py 的设置：Microsoft YaHei 优先）
# ---------------------------------------------------------------------- #
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

for _font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
    if any(_font.lower() == f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------- #
# 配置（与主实验 / builder_a-c 保持一致）
# ---------------------------------------------------------------------- #
SEEDS = [2026, 1, 7, 42, 100, 314, 520, 1322, 2718, 9999]
MAIN_SEED = 2026
N_EPISODES = 2000
GAMMA = 0.9
Q_ALPHA = 0.5                      # 提交文件默认值，协议行为
N_EVAL_EPISODES = 500
EVAL_SEED_BASE = 90000             # 与主实验相同的评估种子段，跨种子可比
THETA = 1e-4
MAX_ITER = 1000

ACTION_COLORS = {"w": "#dddddd", "g": "#f4a261", "i1": "#2a9d8f", "i2": "#e76f51"}
ACTION_LABELS = {"w": "w 维持", "g": "g 补糖", "i1": "i1 小胰岛素", "i2": "i2 大胰岛素"}

SCRIPT_PATH = os.path.abspath(__file__)
OUT_PATH = os.path.join(_HERE, "builder_d_data_seed_results.json")
FIG_PATH = os.path.join(_A3_DIR, "figures", "fig8_data_seed_robustness.png")
LOG_PATH = os.path.join(_HERE, "builder_d_run.log")


# ---------------------------------------------------------------------- #
# 运行日志：stdout 同步写入 log 文件
# ---------------------------------------------------------------------- #
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


# ---------------------------------------------------------------------- #
# 评估（与 assignment3/main.py 的 evaluate_policy 协议逐字一致）
# ---------------------------------------------------------------------- #
def evaluate_policy(policy, n_episodes=N_EVAL_EPISODES, seed_base=EVAL_SEED_BASE):
    """评估确定性策略；每个 episode 用独立种子 rng=Random(seed_base+i)。

    与 main.py 中 evaluate_policy 相同的协议（此处为实验自包含而复制，
    未改动 main.py）。返回四指标 + 逐轨迹 returns。
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


# ---------------------------------------------------------------------- #
# 单种子实验
# ---------------------------------------------------------------------- #
def top2_margin(q_row):
    """一行 Q 值的 top-1 与 top-2 之差（argmax margin）。"""
    s = sorted(q_row, reverse=True)
    return s[0] - s[1]


def run_one_seed(seed):
    t0 = time.time()
    print(f"\n----- 采集种子 seed={seed} -----")
    episodes = collect_random_episodes(N_EPISODES, seed=seed)
    n_trans = sum(len(ep) for ep in episodes)
    print(f"  采集 {len(episodes)} 条 episode，共 {n_trans} 条转移 "
          f"({time.time() - t0:.1f}s)")

    # Part A：模型 + 同步扫描价值迭代（协议默认参数）
    agent_a = ModelBasedAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=GAMMA)
    agent_a.learn_model(episodes)
    agent_a.run_value_iteration(theta=THETA, max_iterations=MAX_ITER)
    policy_a = {s: agent_a.get_policy(s) for s in STATE_NAMES}
    vi_iters = agent_a.value_iteration_iterations

    # Part B：离线 Q-Learning（提交默认 alpha=0.5，fixed 模式）
    agent_b = QLearningAgent(states=STATE_NAMES, actions=ACTION_NAMES,
                             gamma=GAMMA, alpha=Q_ALPHA)
    agent_b.learn(episodes, mode="fixed")
    policy_b = {s: agent_b.get_policy(s) for s in STATE_NAMES}

    # A/B 一致率：口径一 B2–B9（非终止，主口径）；口径二含 B1 共 9 个
    non_terminal = [s for s in STATE_NAMES if s != "B1"]
    agree_nt = [s for s in non_terminal if policy_a[s] == policy_b[s]]
    agree_all = [s for s in STATE_NAMES if policy_a[s] == policy_b[s]]

    # 评估：两路线策略各跑同一 500 条评估种子段（90000–90499）
    eval_a = evaluate_policy(lambda s, rng: policy_a[s])
    eval_b = evaluate_policy(lambda s, rng: policy_b[s])

    # B4 重点观察：argmax 动作与 top-2 Q margin（Part A 口径）
    b4_idx = STATE_NAMES.index("B4")
    q_a = agent_a.get_q_table()
    q_b = agent_b.get_q_table()
    b4_margin_a = top2_margin(q_a[b4_idx])
    b4_margin_b = top2_margin(q_b[b4_idx])
    print(f"  策略A: {policy_a}")
    print(f"  策略B: {policy_b}")
    print(f"  A/B 一致(B2-B9): {len(agree_nt)}/8, 含B1: {len(agree_all)}/9")
    e = eval_a
    print(f"  A 评估: return={e['mean_return']:.2f}, B3={e['b3_time_ratio']:.3f}, "
          f"B1率={e['b1_termination_rate']:.3f}, steps={e['mean_steps']:.1f}")
    e = eval_b
    print(f"  B 评估: return={e['mean_return']:.2f}, B3={e['b3_time_ratio']:.3f}, "
          f"B1率={e['b1_termination_rate']:.3f}, steps={e['mean_steps']:.1f}")
    print(f"  B4: A选 {policy_a['B4']} (margin {b4_margin_a:.4f}), "
          f"B选 {policy_b['B4']} (margin {b4_margin_b:.4f}), VI {vi_iters} 轮 "
          f"({time.time() - t0:.1f}s)")

    return {
        "seed": seed,
        "n_transitions": n_trans,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "agreement_a_vs_b_b2_b9": f"{len(agree_nt)}/8",
        "agreement_a_vs_b_incl_b1": f"{len(agree_all)}/9",
        "disagree_a_vs_b_states": [s for s in STATE_NAMES if policy_a[s] != policy_b[s]],
        "vi_iterations": vi_iters,
        "b4_action_a": policy_a["B4"],
        "b4_action_b": policy_b["B4"],
        "b4_top2_margin_a": b4_margin_a,
        "b4_top2_margin_b": b4_margin_b,
        "eval_a": {k: eval_a[k] for k in
                   ("mean_return", "b3_time_ratio", "b1_termination_rate", "mean_steps")},
        "eval_b": {k: eval_b[k] for k in
                   ("mean_return", "b3_time_ratio", "b1_termination_rate", "mean_steps")},
    }


# ---------------------------------------------------------------------- #
# 汇总统计
# ---------------------------------------------------------------------- #
def _stats(xs):
    """均值 / 样本标准差(ddof=1) / 最小 / 最大。"""
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std_sample": float(arr.std(ddof=1)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def build_summary(per_seed):
    main = next(r for r in per_seed if r["seed"] == MAIN_SEED)
    others = [r for r in per_seed if r["seed"] != MAIN_SEED]

    # 与主种子策略的差异（两口径），Part A 与 Part B 各算一遍
    diff_stats = {}
    for key, pkey in (("part_a", "policy_a"), ("part_b", "policy_b")):
        per_seed_diff = {}
        for r in others:
            dis = [s for s in STATE_NAMES if r[pkey][s] != main[pkey][s]]
            dis_nt = [s for s in non_terminal_states() if r[pkey][s] != main[pkey][s]]
            per_seed_diff[str(r["seed"])] = {
                "diff_states_incl_b1": dis,
                "n_diff_incl_b1": len(dis),
                "diff_states_b2_b9": dis_nt,
                "n_diff_b2_b9": len(dis_nt),
            }
        diff_stats[key] = {
            "reference_seed": MAIN_SEED,
            "per_seed_vs_main": per_seed_diff,
            "n_seeds_with_any_diff_incl_b1": sum(
                1 for v in per_seed_diff.values() if v["n_diff_incl_b1"] > 0),
            "n_seeds_with_any_diff_b2_b9": sum(
                1 for v in per_seed_diff.values() if v["n_diff_b2_b9"] > 0),
        }

    # 评估指标跨 10 种子的分布（Part A / Part B）
    metric_dist = {}
    for key, ekey in (("part_a", "eval_a"), ("part_b", "eval_b")):
        metric_dist[key] = {
            m: _stats([r[ekey][m] for r in per_seed])
            for m in ("mean_return", "b3_time_ratio",
                      "b1_termination_rate", "mean_steps")
        }

    # 安全关注点：B1 终止率最大值（逐种子）
    b1_max = {
        "part_a": max(per_seed, key=lambda r: r["eval_a"]["b1_termination_rate"]),
        "part_b": max(per_seed, key=lambda r: r["eval_b"]["b1_termination_rate"]),
    }

    # 两路线一致率是否始终 100%
    agreement = {
        "agreement_always_100pct_b2_b9": all(
            r["agreement_a_vs_b_b2_b9"] == "8/8" for r in per_seed),
        "agreement_always_100pct_incl_b1": all(
            r["agreement_a_vs_b_incl_b1"] == "9/9" for r in per_seed),
        "min_agreement_b2_b9": min(int(r["agreement_a_vs_b_b2_b9"].split("/")[0])
                                   for r in per_seed),
        "min_agreement_incl_b1": min(int(r["agreement_a_vs_b_incl_b1"].split("/")[0])
                                     for r in per_seed),
    }

    # B4 翻转观察：非主种子中 B4 动作与主种子不同的种子
    b4 = {
        "main_seed_b4_action_a": main["b4_action_a"],
        "main_seed_b4_action_b": main["b4_action_b"],
        "main_seed_b4_top2_margin_a": main["b4_top2_margin_a"],
        "main_seed_b4_top2_margin_b": main["b4_top2_margin_b"],
        "seeds_b4_flipped_a": [r["seed"] for r in others
                               if r["b4_action_a"] != main["b4_action_a"]],
        "seeds_b4_flipped_b": [r["seed"] for r in others
                               if r["b4_action_b"] != main["b4_action_b"]],
        "b4_margin_a_stats": _stats([r["b4_top2_margin_a"] for r in per_seed]),
        "b4_margin_b_stats": _stats([r["b4_top2_margin_b"] for r in per_seed]),
        "vi_iterations": {str(r["seed"]): r["vi_iterations"] for r in per_seed},
    }

    return {
        "n_non_main_seeds": len(others),
        "policy_diff_vs_main_seed": diff_stats,
        "eval_metric_distribution": metric_dist,
        "b1_termination_rate_max": {
            "part_a": {"seed": b1_max["part_a"]["seed"],
                       "value": b1_max["part_a"]["eval_a"]["b1_termination_rate"]},
            "part_b": {"seed": b1_max["part_b"]["seed"],
                       "value": b1_max["part_b"]["eval_b"]["b1_termination_rate"]},
        },
        "agreement_a_vs_b": agreement,
        "b4_observation": b4,
    }


def non_terminal_states():
    return [s for s in STATE_NAMES if s != "B1"]


# ---------------------------------------------------------------------- #
# 图：fig8_data_seed_robustness.png
# ---------------------------------------------------------------------- #
def make_figure(per_seed, main):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # 左面板：10 种子评估回报，Part A / Part B 并列，主种子高亮
    xs = np.arange(len(per_seed))
    w = 0.38
    rets_a = [r["eval_a"]["mean_return"] for r in per_seed]
    rets_b = [r["eval_b"]["mean_return"] for r in per_seed]
    labels = [str(r["seed"]) for r in per_seed]
    main_i = next(i for i, r in enumerate(per_seed) if r["seed"] == MAIN_SEED)

    colors_a = ["#264653" if i != main_i else "#e76f51" for i in xs]
    colors_b = ["#8ab17d" if i != main_i else "#e9c46a" for i in xs]
    ax1.bar(xs - w / 2, rets_a, width=w, color=colors_a, edgecolor="black",
            linewidth=1.2, label="Part A（模型+价值迭代）")
    ax1.bar(xs + w / 2, rets_b, width=w, color=colors_b, edgecolor="black",
            linewidth=0.8, label="Part B（离线 Q-Learning）")

    mean_a = float(np.mean(rets_a))
    mean_b = float(np.mean(rets_b))
    ax1.axhline(mean_a, color="#264653", linestyle="--", linewidth=1.2,
                label=f"Part A 均值 {mean_a:.2f}")
    ax1.axhline(mean_b, color="#8ab17d", linestyle=":", linewidth=1.4,
                label=f"Part B 均值 {mean_b:.2f}")

    for i, (va, vb) in enumerate(zip(rets_a, rets_b)):
        ax1.text(i - w / 2, va, f"{va:.1f}", ha="center", va="bottom", fontsize=7)
        ax1.text(i + w / 2, vb, f"{vb:.1f}", ha="center", va="bottom", fontsize=7)
    ax1.annotate("主种子\n2026", xy=(main_i, max(rets_a[main_i], rets_b[main_i])),
                 xytext=(main_i, max(rets_a) + 4.5), ha="center", fontsize=9,
                 color="#e76f51", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#e76f51"))
    ax1.set_xticks(xs, labels)
    ax1.set_xlabel("数据采集种子（collect_random_episodes(2000, seed=s)）")
    ax1.set_ylabel("评估平均回报（无折扣，500 条，种子段 90000–90499）")
    ax1.set_title("10 个采集种子的评估回报（主种子 2026 高亮）")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(alpha=0.3, axis="y")

    # 右面板：逐状态策略热图（Part A，10 种子 × 9 状态）
    act_to_idx = {a: i for i, a in enumerate(ACTION_NAMES)}
    grid = [[act_to_idx[r["policy_a"][s]] for s in STATE_NAMES] for r in per_seed]
    grid = np.array(grid)
    cmap = ListedColormap([ACTION_COLORS[a] for a in ACTION_NAMES])
    ax2.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(ACTION_NAMES) - 0.5,
               aspect="auto")
    ax2.set_xticks(range(len(STATE_NAMES)), STATE_NAMES)
    ax2.set_yticks(range(len(per_seed)),
                   [f"{r['seed']}{' (主)' if r['seed'] == MAIN_SEED else ''}"
                    for r in per_seed])
    for si in range(grid.shape[0]):
        for ai in range(grid.shape[1]):
            a = ACTION_NAMES[grid[si, ai]]
            diff = per_seed[si]["policy_a"][STATE_NAMES[ai]] != \
                main["policy_a"][STATE_NAMES[ai]]
            ax2.text(ai, si, a, ha="center", va="center", fontsize=9,
                     fontweight="bold" if diff else "normal",
                     color="black")
            if diff:
                # 与主种子不同的格子：黑色加框 + 右上角标记
                ax2.add_patch(Rectangle((ai - 0.5, si - 0.5), 1, 1, fill=False,
                                        edgecolor="black", linewidth=2.2))
                ax2.text(ai + 0.38, si - 0.38, "×", ha="center", va="center",
                         fontsize=10, color="black", fontweight="bold")
    ax2.set_title("Part A 策略热图（黑框×=与主种子 2026 不同）")
    handles = [Patch(facecolor=ACTION_COLORS[a], edgecolor="black",
                     label=ACTION_LABELS[a]) for a in ACTION_NAMES]
    ax2.legend(handles=handles, fontsize=8, loc="upper left",
               bbox_to_anchor=(1.01, 1.0))

    fig.suptitle("附加实验 Builder D：离线主任务的数据采集种子鲁棒性", fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"\n图已写入 {FIG_PATH}")


# ---------------------------------------------------------------------- #
def main():
    t_start = time.time()
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)

    # 运行日志：stdout + log 文件
    log_f = open(LOG_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_f)
    print(f"Builder D 运行开始: {datetime.now().isoformat(timespec='seconds')}")
    print(f"脚本: {SCRIPT_PATH}")

    per_seed = [run_one_seed(s) for s in SEEDS]
    main_res = next(r for r in per_seed if r["seed"] == MAIN_SEED)
    summary = build_summary(per_seed)

    # 控制台速览
    print("\n===== 汇总速览 =====")
    print(f"{'种子':>6s} | {'A 回报':>7s} {'A B3':>6s} {'A B1率':>7s} | "
          f"{'B 回报':>7s} {'B B3':>6s} {'B B1率':>7s} | "
          f"{'A/B一致':>7s} {'与主A差异':>8s} {'B4(A)':>5s}")
    for r in per_seed:
        d = summary["policy_diff_vs_main_seed"]["part_a"]["per_seed_vs_main"][
            str(r["seed"])]["n_diff_incl_b1"] if r["seed"] != MAIN_SEED else 0
        print(f"{r['seed']:>6d} | "
              f"{r['eval_a']['mean_return']:7.2f} "
              f"{r['eval_a']['b3_time_ratio']:6.3f} "
              f"{r['eval_a']['b1_termination_rate']:7.3f} | "
              f"{r['eval_b']['mean_return']:7.2f} "
              f"{r['eval_b']['b3_time_ratio']:6.3f} "
              f"{r['eval_b']['b1_termination_rate']:7.3f} | "
              f"{r['agreement_a_vs_b_b2_b9']:>7s} {d:>8d} {r['b4_action_a']:>5s}")
    em = summary["eval_metric_distribution"]
    print(f"\nPart A 回报: mean={em['part_a']['mean_return']['mean']:.2f} "
          f"std={em['part_a']['mean_return']['std_sample']:.2f} "
          f"[{em['part_a']['mean_return']['min']:.2f}, "
          f"{em['part_a']['mean_return']['max']:.2f}]")
    print(f"Part B 回报: mean={em['part_b']['mean_return']['mean']:.2f} "
          f"std={em['part_b']['mean_return']['std_sample']:.2f} "
          f"[{em['part_b']['mean_return']['min']:.2f}, "
          f"{em['part_b']['mean_return']['max']:.2f}]")
    b1m = summary["b1_termination_rate_max"]
    print(f"B1 终止率最大值: Part A {b1m['part_a']['value']:.3f} (seed={b1m['part_a']['seed']}), "
          f"Part B {b1m['part_b']['value']:.3f} (seed={b1m['part_b']['seed']})")
    ag = summary["agreement_a_vs_b"]
    print(f"A/B 一致率始终 100%: B2-B9 {ag['agreement_always_100pct_b2_b9']}, "
          f"含B1 {ag['agreement_always_100pct_incl_b1']}")
    b4 = summary["b4_observation"]
    print(f"B4 翻转: Part A 翻转种子 {b4['seeds_b4_flipped_a']}, "
          f"Part B 翻转种子 {b4['seeds_b4_flipped_b']} "
          f"(主种子 B4: A={b4['main_seed_b4_action_a']}, B={b4['main_seed_b4_action_b']})")
    pd = summary["policy_diff_vs_main_seed"]
    print(f"与主种子策略不同的种子数: Part A {pd['part_a']['n_seeds_with_any_diff_incl_b1']}/9, "
          f"Part B {pd['part_b']['n_seeds_with_any_diff_incl_b1']}/9 (含B1口径)")

    # 图
    sys.stdout = sys.__stdout__   # 绘图阶段不再刷 log（避免字体告警刷屏）
    make_figure(per_seed, main_res)
    sys.stdout = _Tee(sys.__stdout__, log_f)

    # JSON（含脚本自身 SHA-256 与运行时间戳）
    h = hashlib.sha256()
    with open(SCRIPT_PATH, "rb") as f:
        h.update(f.read())
    out = {
        "experiment": "builder_d_data_seed",
        "description": "离线主任务的数据采集种子鲁棒性：10 个采集种子下 "
                       "Part A / Part B 策略与评估稳定性",
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t_start, 1),
        "script_sha256": h.hexdigest(),
        "config": {
            "seeds": SEEDS,
            "main_seed": MAIN_SEED,
            "n_episodes": N_EPISODES,
            "gamma": GAMMA,
            "q_alpha": Q_ALPHA,
            "vi_theta": THETA,
            "vi_max_iterations": MAX_ITER,
            "n_eval_episodes": N_EVAL_EPISODES,
            "eval_seed_base": EVAL_SEED_BASE,
            "eval_note": "与主实验相同的 500 条评估种子段 90000–90499，"
                         "训练数据种子与评估种子不重叠，跨种子可比",
        },
        "per_seed": per_seed,
        "main_seed_result": main_res,
        "summary": summary,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已写入 {OUT_PATH}")
    print(f"总耗时 {time.time() - t_start:.1f}s")
    print(f"Builder D 运行结束: {datetime.now().isoformat(timespec='seconds')}")
    sys.stdout = sys.__stdout__   # 先还原 stdout 再关日志，避免 flush 已关闭文件
    log_f.close()


if __name__ == "__main__":
    main()
