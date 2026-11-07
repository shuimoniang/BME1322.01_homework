"""Builder E 附加实验 —— 折扣因子 γ 敏感性消融。

目的：主实验固定 γ=0.9（回报 88.01、B3 占比 63.43%、B1 终止率 0.8%）。
本实验扫描 γ ∈ {0.5, 0.8, 0.9, 0.95, 0.99}，考察折扣因子对
"安全（避开 B1）— 收益（留在 B3）"权衡的影响，并检验 γ→1 时
离线 Q-Learning 是否出现大幅值、两路线策略一致率是否仍保持。

协议与公平性
------------
- γ 只通过构造参数传入 ModelBasedAgent / QLearningAgent，
  **提交文件 model_based.py / q_learning.py 一律不改**，
  其默认 gamma=0.9 保持不变（合规红线）。
- 数据预算固定：collect_random_episodes(2000, seed=2026)，**采集一次、
  循环外共享**，全部 γ 共用同一份经验（与主实验同一种子），
  因此逐 γ 差异只能来自折扣因子本身。
- 评估与 main.py 完全同口径：500 条轨迹、每条 GlucoseEnv(seed=90000+i)
  与 random.Random(90000+i)，指标为 mean_return（无折扣累计回报）、
  b3_time_ratio、b1_termination_rate、mean_steps。

记录内容（每个 γ）
------------------
- Part A：策略表、V̂(B3)、价值迭代轮数（theta=1e-4, max 1000）、评估四指标。
- Part B：策略表、训完 Q 表最大绝对值 max|Q|、评估四指标。
- 两路线策略一致率（非终止状态，8 个）。

理论预期（用数据验证或推翻）
----------------------------
γ 越大越"有远见"：更早规避远期风险（B1 终止率应下降或不变），
但也可能更保守地远离目标区（B3 占比可能下降、回报可能下降）；
γ=0.9 是否处于折中由曲线形态判断。

运行（从仓库根目录）：
    python assignment3/experiments/builder_e_gamma.py
输出：
    assignment3/experiments/builder_e_gamma_results.json
    assignment3/figures/fig9_gamma_sensitivity.png
    assignment3/experiments/builder_e_run.log
"""

import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patches as mpatches

# 中文字体（与 main.py 相同：微软雅黑优先，缺失时退回）
for _font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
    if any(_font.lower() == f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

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
N_EPISODES = 2000
N_EVAL_EPISODES = 500
EVAL_SEED_BASE = 90000
GAMMAS = [0.5, 0.8, 0.9, 0.95, 0.99]
GAMMA_MAIN = 0.9
VI_THETA = 1e-4
VI_MAX_ITER = 1000

OUT_JSON = os.path.join(_HERE, "builder_e_gamma_results.json")
OUT_LOG = os.path.join(_HERE, "builder_e_run.log")
OUT_FIG = os.path.join(_ASSIGNMENT_DIR, "figures", "fig9_gamma_sensitivity.png")

NON_TERMINAL = [s for s in STATE_NAMES if s != "B1"]
ACTION_COLORS = {"w": "#dddddd", "g": "#f4a261", "i1": "#2a9d8f", "i2": "#e76f51"}
ACTION_LABELS = {"w": "w 维持", "g": "g 补糖", "i1": "i1 小胰岛素", "i2": "i2 大胰岛素"}


# ---------------------------------------------------------------------- #
# 日志：同时写控制台与 builder_e_run.log
# ---------------------------------------------------------------------- #
class _Tee:
    def __init__(self, path):
        self._file = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------- #
# 评估（与 main.py evaluate_policy 完全同口径，标准库实现）
# ---------------------------------------------------------------------- #
def evaluate_policy(policy, n_episodes=N_EVAL_EPISODES, seed_base=EVAL_SEED_BASE):
    """确定性策略入模拟器跑 500 条轨迹（种子 90000+i）。"""
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


def agreement(policy_a, policy_b):
    """非终止状态上的策略一致率。"""
    agree = [s for s in NON_TERMINAL if policy_a[s] == policy_b[s]]
    disagree = [s for s in NON_TERMINAL if policy_a[s] != policy_b[s]]
    return {
        "agreement_rate": len(agree) / len(NON_TERMINAL),
        "agree_states": agree,
        "disagree_states": disagree,
    }


def max_abs_q(q_table):
    return max(abs(q_table[si][ai])
               for si in range(len(STATE_NAMES))
               for ai in range(len(ACTION_NAMES)))


# ---------------------------------------------------------------------- #
# 逐 γ 实验
# ---------------------------------------------------------------------- #
def run_one_gamma(gamma, episodes):
    # ---- Part A：模型 + 价值迭代 ----
    agent_a = ModelBasedAgent(states=STATE_NAMES, actions=ACTION_NAMES, gamma=gamma)
    agent_a.learn_model(episodes)
    agent_a.run_value_iteration(theta=VI_THETA, max_iterations=VI_MAX_ITER)
    policy_a = {s: agent_a.get_policy(s) for s in STATE_NAMES}
    v_b3 = agent_a.get_value("B3")
    vi_iters = agent_a.value_iteration_iterations
    vi_converged = vi_iters < VI_MAX_ITER
    eval_a = evaluate_policy(lambda s, rng: policy_a[s])

    # ---- Part B：离线 Q-Learning（α=0.5 与主实验一致）----
    agent_b = QLearningAgent(states=STATE_NAMES, actions=ACTION_NAMES,
                             gamma=gamma, alpha=0.5)
    agent_b.learn(episodes, mode="fixed")
    policy_b = {s: agent_b.get_policy(s) for s in STATE_NAMES}
    q_table = agent_b.get_q_table()
    eval_b = evaluate_policy(lambda s, rng: policy_b[s])

    agree = agreement(policy_a, policy_b)
    return {
        "gamma": gamma,
        "part_a": {
            "policy": policy_a,
            "v_b3": v_b3,
            "values": {s: agent_a.get_value(s) for s in STATE_NAMES},
            "vi_iterations": vi_iters,
            "vi_converged": vi_converged,
            "eval": eval_a,
        },
        "part_b": {
            "policy": policy_b,
            "max_abs_q": max_abs_q(q_table),
            "q_table": {STATE_NAMES[si]: {ACTION_NAMES[ai]: q_table[si][ai]
                                          for ai in range(len(ACTION_NAMES))}
                        for si in range(len(STATE_NAMES))},
            "update_count": agent_b.update_count,
            "eval": eval_b,
        },
        "agreement": agree,
    }


# ---------------------------------------------------------------------- #
# 汇总分析
# ---------------------------------------------------------------------- #
def summarize(per_gamma):
    """从逐 γ 结果中提取趋势与结论字段（全部由真实数据推导）。"""
    by_g = {r["gamma"]: r for r in per_gamma}
    ref = by_g[GAMMA_MAIN]

    def series(route, key):
        return {str(g): by_g[g][route]["eval"][key] for g in GAMMAS}

    # 策略相对 γ=0.9 主实验的翻转状态（Part A / Part B 分别看）
    flips = {}
    for route in ("part_a", "part_b"):
        flips[route] = {
            str(g): [s for s in STATE_NAMES
                     if by_g[g][route]["policy"][s] != ref[route]["policy"][s]]
            for g in GAMMAS if g != GAMMA_MAIN
        }

    # γ=0.99 与 γ=0.9 的指标变化方向
    e99a = by_g[0.99]["part_a"]["eval"]
    e9a = ref["part_a"]["eval"]
    b1_trend = ("下降" if e99a["b1_termination_rate"] < e9a["b1_termination_rate"]
                else "上升" if e99a["b1_termination_rate"] > e9a["b1_termination_rate"]
                else "不变")
    b3_trend = ("上升" if e99a["b3_time_ratio"] > e9a["b3_time_ratio"]
                else "下降" if e99a["b3_time_ratio"] < e9a["b3_time_ratio"]
                else "不变")
    ret_trend = ("上升" if e99a["mean_return"] > e9a["mean_return"]
                 else "下降" if e99a["mean_return"] < e9a["mean_return"]
                 else "不变")

    # 一致率是否全程 100%
    rates = {str(g): by_g[g]["agreement"]["agreement_rate"] for g in GAMMAS}
    all_agree = all(r == 1.0 for r in rates.values())

    # γ=0.9 是否为回报意义上的折中/峰值（在各 γ 中比较 Part A 回报）
    rets_a = {g: by_g[g]["part_a"]["eval"]["mean_return"] for g in GAMMAS}
    best_g = max(rets_a, key=rets_a.get)

    return {
        "eval_series_part_a": {k: series("part_a", k) for k in
                               ("mean_return", "b3_time_ratio",
                                "b1_termination_rate", "mean_steps")},
        "eval_series_part_b": {k: series("part_b", k) for k in
                               ("mean_return", "b3_time_ratio",
                                "b1_termination_rate", "mean_steps")},
        "v_b3_series_part_a": {str(g): by_g[g]["part_a"]["v_b3"] for g in GAMMAS},
        "vi_iterations_series_part_a": {str(g): by_g[g]["part_a"]["vi_iterations"]
                                        for g in GAMMAS},
        "max_abs_q_series_part_b": {str(g): by_g[g]["part_b"]["max_abs_q"]
                                    for g in GAMMAS},
        "agreement_rate_series": rates,
        "agreement_all_100pct": all_agree,
        "policy_flips_vs_gamma09": flips,
        "trend_gamma09_to_099_part_a": {
            "b1_termination_rate": b1_trend,
            "b3_time_ratio": b3_trend,
            "mean_return": ret_trend,
        },
        "best_gamma_by_part_a_return": {
            "gamma": best_g, "mean_return": rets_a[best_g],
        },
    }


# ---------------------------------------------------------------------- #
# 图：fig9_gamma_sensitivity.png
# ---------------------------------------------------------------------- #
def make_figure(per_gamma):
    gammas = [r["gamma"] for r in per_gamma]
    by_g = {r["gamma"]: r for r in per_gamma}

    fig, (ax1, ax3, axh) = plt.subplots(
        1, 3, figsize=(17, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.15]})

    # ---- 左面板：评估指标曲线（双 y 轴处理量纲差异）----
    ax2 = ax1.twinx()
    color_ret = "#264653"
    color_b3 = "#2a9d8f"
    color_b1 = "#e76f51"

    ret_a = [by_g[g]["part_a"]["eval"]["mean_return"] for g in gammas]
    ret_b = [by_g[g]["part_b"]["eval"]["mean_return"] for g in gammas]
    b3_a = [by_g[g]["part_a"]["eval"]["b3_time_ratio"] for g in gammas]
    b3_b = [by_g[g]["part_b"]["eval"]["b3_time_ratio"] for g in gammas]
    b1_a = [by_g[g]["part_a"]["eval"]["b1_termination_rate"] for g in gammas]
    b1_b = [by_g[g]["part_b"]["eval"]["b1_termination_rate"] for g in gammas]

    ax1.plot(gammas, ret_a, "-", color=color_ret, marker="o",
             label="mean_return (A)")
    ax1.plot(gammas, ret_b, "--", color=color_ret, marker="s",
             label="mean_return (B)")
    ax2.plot(gammas, b3_a, "-", color=color_b3, marker="o",
             label="B3 占比 (A)")
    ax2.plot(gammas, b3_b, "--", color=color_b3, marker="s",
             label="B3 占比 (B)")
    ax2.plot(gammas, b1_a, "-", color=color_b1, marker="o",
             label="B1 终止率 (A)")
    ax2.plot(gammas, b1_b, "--", color=color_b1, marker="s",
             label="B1 终止率 (B)")

    ax1.axvline(GAMMA_MAIN, color="gray", linestyle=":", linewidth=1.5)
    ax1.text(GAMMA_MAIN, ax1.get_ylim()[1], "  主实验 γ=0.9",
             rotation=90, va="top", ha="left", fontsize=8, color="gray")

    ax1.set_xlabel("折扣因子 γ")
    ax1.set_ylabel("mean_return（无折扣累计回报）", color=color_ret)
    ax2.set_ylabel("B3 占比 / B1 终止率", color="#555555")
    ax1.tick_params(axis="y", labelcolor=color_ret)
    ax1.set_title("γ 敏感性：评估指标（实线 Part A / 虚线 Part B）")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xticks(gammas)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.minorticks_off()

    # ---- 右面板：Part A 逐 γ 策略热图（动作颜色编码）----
    ref_policy = by_g[GAMMA_MAIN]["part_a"]["policy"]
    n_g, n_s = len(gammas), len(STATE_NAMES)
    axh.imshow([[0] * n_s] * n_g, cmap="Greys", vmin=0, vmax=4, aspect="auto")
    for gi, g in enumerate(gammas):
        for si, s in enumerate(STATE_NAMES):
            a = by_g[g]["part_a"]["policy"][s]
            axh.add_patch(plt.Rectangle((si - 0.5, gi - 0.5), 1, 1,
                                        color=ACTION_COLORS[a],
                                        ec="black", linewidth=0.5))
            # 与主实验 γ=0.9 策略不同的格子加粗黑框标出
            if a != ref_policy[s]:
                axh.add_patch(plt.Rectangle((si - 0.5, gi - 0.5), 1, 1,
                                            fill=False, ec="black", linewidth=2.5))
            axh.text(si, gi, a, ha="center", va="center", fontsize=10,
                     fontweight="bold")
    axh.set_xticks(range(n_s), STATE_NAMES)
    axh.set_yticks(range(n_g), [f"γ={g}" for g in gammas])
    axh.invert_yaxis()
    axh.set_title("Part A 策略表（加粗框 = 与 γ=0.9 不同）")
    handles = [mpatches.Patch(color=c, label=ACTION_LABELS[a])
               for a, c in ACTION_COLORS.items()]
    axh.legend(handles=handles, fontsize=7, loc="upper center",
               bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

    # ---- 补充面板：Part B 策略热图（同一编码，便于逐 γ 对照）----
    axh2 = ax3
    ref_policy_b = by_g[GAMMA_MAIN]["part_b"]["policy"]
    axh2.imshow([[0] * n_s] * n_g, cmap="Greys", vmin=0, vmax=4, aspect="auto")
    for gi, g in enumerate(gammas):
        for si, s in enumerate(STATE_NAMES):
            a = by_g[g]["part_b"]["policy"][s]
            axh2.add_patch(plt.Rectangle((si - 0.5, gi - 0.5), 1, 1,
                                         color=ACTION_COLORS[a],
                                         ec="black", linewidth=0.5))
            if a != ref_policy_b[s]:
                axh2.add_patch(plt.Rectangle((si - 0.5, gi - 0.5), 1, 1,
                                             fill=False, ec="black", linewidth=2.5))
            axh2.text(si, gi, a, ha="center", va="center", fontsize=10,
                      fontweight="bold")
    axh2.set_xticks(range(n_s), STATE_NAMES)
    axh2.set_yticks(range(n_g), [f"γ={g}" for g in gammas])
    axh2.invert_yaxis()
    axh2.set_title("Part B 策略表（加粗框 = 与 γ=0.9 不同）")

    fig.suptitle("附加实验：折扣因子 γ 敏感性消融（数据 seed=2026 共享，评估 500 条 seed 90000+i）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- #
def main():
    tee = _Tee(OUT_LOG)
    sys.stdout = tee
    t0 = time.time()
    print(f"==== Builder E：γ 敏感性消融 ====")
    print(f"开始时间: {datetime.now().isoformat(timespec='seconds')}")

    print(f"[1/3] 采集统一数据预算（采集一次、循环外共享）："
          f"{N_EPISODES} 条随机策略 episode (seed={DATA_SEED}) ...")
    episodes = collect_random_episodes(N_EPISODES, seed=DATA_SEED)
    total_transitions = sum(len(ep) for ep in episodes)
    print(f"      共 {total_transitions} 条转移经验，全部 γ 共用此份数据")

    print(f"[2/3] 扫描 γ ∈ {GAMMAS}（Part A 价值迭代 + Part B Q-Learning + 评估）...")
    per_gamma = []
    for gamma in GAMMAS:
        r = run_one_gamma(gamma, episodes)
        per_gamma.append(r)
        ea, eb = r["part_a"]["eval"], r["part_b"]["eval"]
        print(f"      γ={gamma}: A return={ea['mean_return']:7.2f} "
              f"B3={ea['b3_time_ratio']:.4f} B1率={ea['b1_termination_rate']:.4f} "
              f"steps={ea['mean_steps']:.2f} | V̂(B3)={r['part_a']['v_b3']:8.2f} "
              f"VI={r['part_a']['vi_iterations']}轮(收敛={r['part_a']['vi_converged']})")
        print(f"              B return={eb['mean_return']:7.2f} "
              f"B3={eb['b3_time_ratio']:.4f} B1率={eb['b1_termination_rate']:.4f} "
              f"steps={eb['mean_steps']:.2f} | max|Q|={r['part_b']['max_abs_q']:.2f} "
              f"| A/B一致率={r['agreement']['agreement_rate']:.3f} "
              f"(不一致: {r['agreement']['disagree_states']})")
        policy_a_str = " ".join(f"{s}:{r['part_a']['policy'][s]}"
                                for s in STATE_NAMES)
        policy_b_str = " ".join(f"{s}:{r['part_b']['policy'][s]}"
                                for s in STATE_NAMES)
        print(f"              A 策略: {policy_a_str}")
        print(f"              B 策略: {policy_b_str}")

    print("[3/3] 汇总分析 + 绘图 + 写 JSON ...")
    summary = summarize(per_gamma)
    print(f"      γ 0.9→0.99（Part A）: B1率{summary['trend_gamma09_to_099_part_a']['b1_termination_rate']}、"
          f"B3占比{summary['trend_gamma09_to_099_part_a']['b3_time_ratio']}、"
          f"回报{summary['trend_gamma09_to_099_part_a']['mean_return']}")
    print(f"      两路线一致率是否全程 100%: {summary['agreement_all_100pct']}")
    print(f"      Part A 回报最优 γ: {summary['best_gamma_by_part_a_return']['gamma']} "
          f"(return={summary['best_gamma_by_part_a_return']['mean_return']:.2f})")
    print(f"      策略翻转（vs γ=0.9, Part A）: {summary['policy_flips_vs_gamma09']['part_a']}")
    print(f"      策略翻转（vs γ=0.9, Part B）: {summary['policy_flips_vs_gamma09']['part_b']}")
    print(f"      max|Q|（Part B）: {summary['max_abs_q_series_part_b']}")

    make_figure(per_gamma)
    print(f"      图已写入 {OUT_FIG}")

    # 脚本自身 SHA-256（留档自校验）
    with open(os.path.abspath(__file__), "rb") as f:
        script_sha = hashlib.sha256(f.read()).hexdigest()

    out = {
        "experiment": "builder_e_gamma",
        "description": "折扣因子 γ 敏感性消融：γ ∈ {0.5, 0.8, 0.9, 0.95, 0.99}，"
                       "同一份 seed=2026 数据，Part A / Part B 分别训练评估",
        "config": {
            "gammas": GAMMAS,
            "gamma_main_experiment": GAMMA_MAIN,
            "data_seed": DATA_SEED,
            "n_episodes": N_EPISODES,
            "n_eval_episodes": N_EVAL_EPISODES,
            "eval_seed_base": EVAL_SEED_BASE,
            "vi_theta": VI_THETA,
            "vi_max_iterations": VI_MAX_ITER,
            "part_b_alpha": 0.5,
            "data_sharing_note": "episodes 采集一次循环外共享，所有 γ 用同一份数据",
            "submission_default_gamma_note": "提交文件默认 gamma=0.9 不变，γ 仅经构造参数传入",
        },
        "total_transitions": total_transitions,
        "script_sha256": script_sha,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - t0, 2),
        "per_gamma": per_gamma,
        "summary": summary,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"      结果已写入 {OUT_JSON}")

    print(f"\n完成，用时 {time.time() - t0:.1f} s。")
    sys.stdout = tee._stdout
    tee.close()


if __name__ == "__main__":
    main()
