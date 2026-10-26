# assignment3 v2 迭代硬约束（rulebook）

来源：`assignment3/Blood Glucose Management Agent (Assignment - 3).pdf`（7 页）。
任何候选优化必须逐条满足；违反任一条即为无效方案，无论评估指标多高。

## 算法红线

- 不得调用任何现成强化学习/规划库（gymnasium、stable-baselines3 等）；允许 Python 标准库、numpy、matplotlib。
- **不得调用现成统计库的高级接口**（如 scipy.stats、sklearn）做模型估计。
- 不得读取真实转移表/奖励表——只能使用给定 (s,a,r,s',done) 经验（不得 import glucose_env 的 REWARDS/BIN_EDGES/ACTION_COST 等常量进算法文件）。
- Part A：P̂ 用计数归一化、R̂ 用样本均值（讲义指定的估计式）；价值迭代为同步扫描。
- Part B：Q 初始化 0；更新式 Q←(1−α)Q+α[r+γ max Q(s')]，done=True 不含未来项；learn() 按给定顺序逐条更新、不采样动作。

## 接口协议（评测程序按此加载，不可破坏）

- `ModelBasedAgent(states, actions, gamma=0.9, seed=None)` + learn_model / run_value_iteration(theta=1e-4, max_iterations=1000) / get_transition / get_reward / get_value / get_policy。
- `QLearningAgent(states, actions, gamma=0.9, alpha=0.5, epsilon=0.3, epsilon_decay=0.995, epsilon_min=0.05, seed=None)` + learn(episodes, mode="fixed") / update / get_q_value / get_policy。
- get_policy 平局取课程约定顺序（w,g,i1,i2）第一个动作。
- 评测程序从仓库根目录运行，读取 assignment3/model_based.py 与 assignment3/q_learning.py。

## 数据与评估协议

- 统一数据预算：collect_random_episodes(2000, seed=2026)（约 21,825 条转移），Part A/B/消融共用。
- 评估：500 条轨迹（种子 90000+i，与训练数据种子不重叠），指标为无折扣累计回报平均、B3 时间占比、B1 终止率。
- **评分重点不是最高回报**，而是算法正确、实验完整、解释清楚（评分标准原文）。因此任何"优化"不得以破坏协议或牺牲可解释性为代价。
- 提交文件清单固定：model_based.py、q_learning.py、analysis.md、main.py、requirements.txt、figures/、report.md。新增文件只允许实验留档目录（experiments/）。

## 优化候选的合法空间（v2 判定规则）

- 允许：评估协议增强（置信区间、多次评估）、更严格的实验设计、可视化改进、超参敏感性分析、报告/图表完善——只要默认行为仍符合上述协议。
- 允许（需留档消融并在报告说明）：智能体默认参数之外的附加实验（如其他 α、其他先验、Dyna 式规划），但**提交的 model_based.py / q_learning.py 的协议行为必须与讲义公式一致**。
- 不允许：修改算法核心公式（P̂/R̂ 估计式、Q 更新式、同步扫描）、修改课程环境文件、把测试信息泄漏进训练、在 learn() 中采样动作。
