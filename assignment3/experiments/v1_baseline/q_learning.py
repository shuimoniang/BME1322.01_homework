"""BME1322 Assignment 3 —— Part B：无模型表格 Q-Learning（离线批量设定）。

实现协议 v1（课程评测程序按此接口加载）：

- 主任务是离线批量学习：``learn()`` 只按给定顺序逐条 episode、逐条
  转移调用 ``self.update()``，不做任何动作采样（统一数据预算下与
  Part A 公平对比）。``mode`` 参数仅为接口一致性保留。
- 更新式与第 9 章一致（异策略 off-policy，TD 目标是贪心的）：

      Q(s,a) ← (1−α)·Q(s,a) + α·[ r + γ·max_{a'} Q(s',a') ]

  done=True 时 target 中不含未来项（B1 为终止状态，也无后继价值）。
- Q 表初始化为 0；``epsilon`` 相关参数仅用于报告 4.2 节的附加在线
  探索实验，离线主任务不使用。
- 不读取真实转移表/奖励表，只使用给定的 (s,a,r,s',done) 经验。
"""

import random
from typing import Dict, List, Sequence, Tuple


class QLearningAgent:
    """Q-Learning Protocol v1 —— 课程评测程序将按此接口加载并调用你的智能体。"""

    def __init__(self, states: Sequence[str], actions: Sequence[str],
                 gamma: float = 0.9, alpha: float = 0.5,
                 epsilon: float = 0.3, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05, seed=None):
        """epsilon 相关参数仅用于报告的附加在线探索实验；离线主任务不使用。"""
        self.states: List[str] = list(states)
        self.actions: List[str] = list(actions)
        self.gamma: float = gamma
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.seed = seed
        self._rng = random.Random(seed)

        self._state_index: Dict[str, int] = {s: i for i, s in enumerate(self.states)}
        self._n_states = len(self.states)
        self._n_actions = len(self.actions)

        # Q 表初始化为 0（协议要求）
        self.q: List[List[float]] = [
            [0.0] * self._n_actions for _ in range(self._n_states)
        ]
        # 每行是否被更新过（用于区分“Q=0 因为没见过”和“Q=0 因为学到 0”）
        self._visited: List[List[bool]] = [
            [False] * self._n_actions for _ in range(self._n_states)
        ]
        self.update_count = 0

        if self._n_states == 0 or self._n_actions == 0:
            raise ValueError("states 和 actions 不能为空")

    # ------------------------------------------------------------------ #
    # 离线批量学习
    # ------------------------------------------------------------------ #
    def learn(self, episodes: list, mode: str = "fixed") -> None:
        """从统一数据预算的 episodes 中按顺序做表格 Q-Learning 更新。

        逐条 episode、逐条转移调用 self.update()，不做任何动作采样。
        mode 参数保留接口一致性，本作业离线主任务中不使用探索率。
        """
        for episode in episodes:
            for (state, action, reward, next_state, done) in episode:
                self.update(state, action, reward, next_state, done)

    def update(self, state, action, reward, next_state, done: bool) -> None:
        """单步表格 Q-Learning 更新；done=True 时 target 中不含未来项。"""
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

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #
    def get_q_value(self, state, action) -> float:
        return float(self.q[self._require_state(state)][self._require_action(action)])

    def get_policy(self, state):
        """返回当前贪心动作（Q 表每行 argmax）。

        平局时取课程约定顺序（actions 列表顺序）中的第一个动作，
        与 Part A 的策略提取约定保持一致。
        """
        row = self.q[self._require_state(state)]
        best_ai = 0
        for ai in range(1, self._n_actions):
            if row[ai] > row[best_ai]:
                best_ai = ai
        return self.actions[best_ai]

    def get_q_table(self) -> List[List[float]]:
        """返回 Q 表（行 = states 顺序，列 = actions 顺序），报告用。"""
        return [row[:] for row in self.q]

    def get_visited(self) -> List[List[bool]]:
        """返回每个 (s,a) 是否被更新过，报告稀疏性用。"""
        return [row[:] for row in self._visited]

    # ------------------------------------------------------------------ #
    # 附加在线实验（报告 4.2 节）用的 ε-greedy 辅助方法
    # ------------------------------------------------------------------ #
    def select_action(self, state) -> str:
        """ε-greedy 动作选择：以 ε 概率均匀随机探索，否则贪心。"""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        return self.get_policy(state)

    def decay_epsilon(self) -> None:
        """每个 episode 结束后衰减 ε（附加在线实验用）。"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
