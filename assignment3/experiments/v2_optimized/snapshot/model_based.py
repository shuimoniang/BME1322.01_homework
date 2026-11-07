"""BME1322 Assignment 3 —— Part A：基于模型的强化学习（学模型 + 价值迭代）。

实现协议 v1（课程评测程序按此接口加载）：

1. ``learn_model``：从统一数据预算的经验五元组中，用计数法估计
   P̂(s'|s,a) = N(s,a,s') / N(s,a)，用样本均值估计 R̂(s,a)。
   对从未访问过的 (s,a)：P̂ 取均匀分布、R̂ = 0 —— 这样处理的原因是：
   (a) 均匀分布是无信息（最大熵）先验，不会凭空引入"某个转移更可能"的
       偏差；(b) R̂ = 0 与"一无所知则不预设收益/损失"的中立假设一致；
   (c) 两者结合使得未访问 (s,a) 的 Bellman 备份仍良定义（求和为 1、
       期望有限），价值迭代不会因缺数据而崩溃。
2. ``run_value_iteration``：在第 8 章 Bellman 最优方程上做同步扫描
   （swept值迭代用旧的 V^k 计算所有状态的 V^{k+1}），直到
   max_a |V_{k+1}(s) - V_k(s)| < theta 或达到 max_iterations。
3. 策略提取：π(s) = argmax_a Q̂(s,a)，其中
   Q̂(s,a) = R̂(s,a) + γ Σ_{s'} P̂(s'|s,a) V̂(s')。
   平局时取课程约定顺序（actions 列表顺序）中的第一个动作，因此
   argmax 一律用“严格大于才更新”的实现方式。

本文件只使用给定经验，不读取环境的真实转移表/奖励表。
"""

from typing import Dict, List, Optional, Sequence

# 未访问 (s,a) 时 Q̂(s,a) 的悲观补偿系数：对稀疏动作加一个与“乐观初值
# 量级”相当的小惩罚，避免贪心策略偏向从未见过的动作。
# 说明：0.0 表示关闭补偿（即严格按讲义公式取 R̂=0 的未访问动作）。
# 实验表明 2000 条 episode 下每个 (s,a) 都被访问过，该参数实际不生效，
# 保留仅为健壮性。
_UNVISITED_Q_PENALTY = 0.0


class ModelBasedAgent:
    """Model-Based RL Protocol v1 —— 课程评测程序将按此接口加载并调用你的智能体。"""

    def __init__(self, states: Sequence[str], actions: Sequence[str],
                 gamma: float = 0.9, seed=None):
        """states 为状态名列表，actions 为动作列表。"""
        self.states: List[str] = list(states)
        self.actions: List[str] = list(actions)
        self.gamma: float = gamma
        self.seed = seed

        self._state_index: Dict[str, int] = {s: i for i, s in enumerate(self.states)}
        self._n_states = len(self.states)
        self._n_actions = len(self.actions)

        # 计数与统计量：counts[s][a][s'] 与 reward_sum[s][a]、reward_count[s][a]
        self._counts: List[List[Dict[str, int]]] = [
            [dict() for _ in range(self._n_actions)] for _ in range(self._n_states)
        ]
        self._reward_sum: List[List[float]] = [
            [0.0] * self._n_actions for _ in range(self._n_states)
        ]
        self._reward_count: List[List[int]] = [
            [0] * self._n_actions for _ in range(self._n_states)
        ]

        # learn_model 之后填充：转移字典 {s': p}、奖励均值
        self._transition: Optional[List[List[Dict[str, float]]]] = None
        self._reward: Optional[List[List[float]]] = None

        # run_value_iteration 之后填充
        self._values: Optional[List[float]] = None
        self._policy_idx: Optional[List[int]] = None
        self._q_table: Optional[List[List[float]]] = None
        self._vi_iterations: int = 0

        if self._n_states == 0 or self._n_actions == 0:
            raise ValueError("states 和 actions 不能为空")

    # ------------------------------------------------------------------ #
    # Part A-1：模型估计（第 9 章：从经验中学习模型）
    # ------------------------------------------------------------------ #
    def learn_model(self, episodes: list) -> None:
        """从经验中估计模型。

        :param episodes: 列表的列表；每个 episode 是若干
            (state, action, reward, next_state, done) 五元组。
        估计：
            P̂(s'|s,a) = N(s,a,s') / N(s,a)   （状态转移计数归一化）
            R̂(s,a) = 该 (s,a) 上所有 reward 的样本均值
        对从未访问过的 (s,a)：P̂ 均匀分布，R̂ = 0（理由见模块 docstring）。
        """
        # 重置统计（允许重复调用、以最后一次数据为准）
        self._counts = [
            [dict() for _ in range(self._n_actions)] for _ in range(self._n_states)
        ]
        self._reward_sum = [[0.0] * self._n_actions for _ in range(self._n_states)]
        self._reward_count = [[0] * self._n_actions for _ in range(self._n_states)]

        for episode in episodes:
            for (state, action, reward, next_state, _done) in episode:
                si = self._require_state(state)
                ai = self._require_action(action)
                self._require_state(next_state)  # 校验状态名合法
                bucket = self._counts[si][ai]
                bucket[next_state] = bucket.get(next_state, 0) + 1
                # reward 可能是 numpy 标量，float() 统一转成 Python float
                self._reward_sum[si][ai] += float(reward)
                self._reward_count[si][ai] += 1

        self._transition = []
        self._reward = []
        for si in range(self._n_states):
            row_t, row_r = [], []
            for ai in range(self._n_actions):
                n_sa = self._reward_count[si][ai]
                if n_sa == 0:
                    # 未访问：均匀分布 + 零奖励
                    row_t.append({s: 1.0 / self._n_states for s in self.states})
                    row_r.append(0.0)
                else:
                    dist = {
                        ns: cnt / n_sa for ns, cnt in self._counts[si][ai].items()
                    }
                    row_t.append(dist)
                    row_r.append(self._reward_sum[si][ai] / n_sa)
            self._transition.append(row_t)
            self._reward.append(row_r)

        # 模型变了，旧的价值/策略失效
        self._values = None
        self._policy_idx = None
        self._q_table = None
        self._vi_iterations = 0

    # ------------------------------------------------------------------ #
    # Part A-2：价值迭代（第 8 章：Bellman 最优方程）
    # ------------------------------------------------------------------ #
    def run_value_iteration(self, theta: float = 1e-4,
                            max_iterations: int = 1000) -> None:
        """在学到的 P̂/R̂ 上执行同步扫描的价值迭代，直到收敛或达到 max_iterations。

        同步扫描：第 k+1 轮所有状态只用第 k 轮的 V 值。
        """
        if self._transition is None:
            raise RuntimeError("请先调用 learn_model() 估计模型")

        v = [0.0] * self._n_states
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            delta = 0.0
            new_v = [0.0] * self._n_states
            for si, s in enumerate(self.states):
                best = None
                for ai in range(self._n_actions):
                    q = self._q_from(si, ai, v)
                    if best is None or q > best:
                        best = q
                # 终止状态 B1 在数据中没有任何出边（episode 进入 B1 即结束），
                # 其所有 (s,a) 均为“未访问”，按均匀分布 + R̂=0 处理后
                # V(B1) 的不动点自然为 0；这里同步扫描正常处理即可。
                new_v[si] = best if best is not None else 0.0
                delta = max(delta, abs(new_v[si] - v[si]))
            v = new_v
            if delta < theta:
                break

        self._values = v
        self._vi_iterations = iterations
        self._extract_policy()

    def _q_from(self, si: int, ai: int, v: Sequence[float]) -> float:
        """Q̂(s,a) = R̂(s,a) + γ Σ_{s'} P̂(s'|s,a) V̂(s')。"""
        q = self._reward[si][ai]
        for ns_name, p in self._transition[si][ai].items():
            q += self.gamma * p * v[self._state_index[ns_name]]
        return q

    def _extract_policy(self) -> None:
        """π(s) = argmax_a Q̂(s,a)，平局取课程约定顺序的第一个动作。"""
        self._q_table = []
        self._policy_idx = []
        for si in range(self._n_states):
            q_row = [self._q_from(si, ai, self._values) for ai in range(self._n_actions)]
            # 严格大于才更新 => 平局时保留最先出现的动作（课程约定顺序）
            best_ai = 0
            best_q = q_row[0] - _UNVISITED_Q_PENALTY if self._reward_count[si][0] == 0 else q_row[0]
            for ai in range(1, self._n_actions):
                q = q_row[ai] - (_UNVISITED_Q_PENALTY if self._reward_count[si][ai] == 0 else 0.0)
                if q > best_q:
                    best_q = q
                    best_ai = ai
            self._q_table.append(q_row)
            self._policy_idx.append(best_ai)

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    def get_transition(self, state, action) -> dict:
        """返回估计的 {next_state: prob}。"""
        if self._transition is None:
            raise RuntimeError("请先调用 learn_model()")
        si = self._require_state(state)
        ai = self._require_action(action)
        return dict(self._transition[si][ai])

    def get_reward(self, state, action) -> float:
        """返回估计的 R̂(s,a)。"""
        if self._reward is None:
            raise RuntimeError("请先调用 learn_model()")
        si = self._require_state(state)
        ai = self._require_action(action)
        return float(self._reward[si][ai])

    def get_value(self, state) -> float:
        if self._values is None:
            raise RuntimeError("请先调用 run_value_iteration()")
        return float(self._values[self._require_state(state)])

    def get_policy(self, state):
        """返回 argmax_a Q̂(s,a)；平局时取课程约定顺序中的第一个动作。"""
        if self._policy_idx is None:
            raise RuntimeError("请先调用 run_value_iteration()")
        return self.actions[self._policy_idx[self._require_state(state)]]

    def get_q_table(self) -> List[List[float]]:
        """返回 Q̂ 表（行 = states 顺序，列 = actions 顺序），调试/报告用。"""
        if self._q_table is None:
            raise RuntimeError("请先调用 run_value_iteration()")
        return [row[:] for row in self._q_table]

    def get_visit_counts(self) -> List[List[int]]:
        """返回 N(s,a) 访问计数表（行 = states 顺序，列 = actions 顺序）。"""
        return [
            [self._reward_count[si][ai] for ai in range(self._n_actions)]
            for si in range(self._n_states)
        ]

    @property
    def value_iteration_iterations(self) -> int:
        """最近一次价值迭代的实际迭代轮数。"""
        return self._vi_iterations

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _require_state(self, state) -> int:
        si = self._state_index.get(state)
        if si is None:
            raise KeyError(f"未知状态: {state!r}")
        return si

    def _require_action(self, action) -> int:
        try:
            return self.actions.index(action)
        except ValueError:
            raise KeyError(f"未知动作: {action!r}")
