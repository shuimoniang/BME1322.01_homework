# 广义五子棋与对抗搜索实验报告

## 摘要

本项目实现了支持任意 `N >= 3`、`3 <= K <= N` 的广义五子棋，并提供 Baseline Alpha-Beta、Enhanced Alpha-Beta、MCTS + UCT 和 Hybrid Threat Search 四种 AI。课程提交入口 `GomokuAI` 保持原协议不变，静态继承 Hybrid Threat Search。Enhanced、Hybrid 和 MCTS 共用参数化威胁摘要；Hybrid 进一步使用强制着法、VCF/VCT 风格威胁证明、Quiescence Search、PVS、Aspiration Window、killer/history、LMR 和置换表。

本次正式主矩阵只使用当前源码版本，运行 1500 局：Hybrid 对 Random 300 局、Hybrid 自我对战 300 局，以及 Baseline、MCTS、Hybrid 分别对 Enhanced 各 300 局。为完整满足实验评估并横向比较全部实现，又独立运行 2400 局全策略补充矩阵：四种 AI 各自对 Random 及各自自我对战。两套实验均覆盖 `(9,4)`、`(15,5)` 和 `0.5/1.0/5.0s` 三档时限，每组 50 局；合计 3900 局，协议失败为 0。优化前的 1800 局正式数据已归档，不参与本报告结论。

## 1. 系统与规则

棋盘使用 `0=EMPTY, 1=BLACK, 2=WHITE`，黑方先手；落子后沿横、竖和两条对角线检查长度至少为 `K` 的连珠，长连也判胜。棋盘填满且无人获胜时为和棋。AI 接口为：

```python
GomokuAI(player_id, board_size, win_length)
get_move(board, last_opponent_move, time_limit) -> (row, col)
```

所有策略先确定合法保底着法，在棋盘副本上搜索，并在递归、候选生成、评估、哈希和 MCTS 阶段检查软截止时间。课程入口、GUI 和四种策略的公共构造函数保持兼容。

## 2. AI 方法

### 2.1 Baseline Alpha-Beta

Baseline 保留原始对照逻辑：邻域候选、固定浅层 Negamax/Alpha-Beta 和连续段评估，不使用高级威胁摘要或迭代加深。它用于衡量增加搜索和评估结构后的收益，不代表最终提交策略。

### 2.2 泛化威胁摘要

Enhanced、Hybrid 和 MCTS 对每个局面只扫描一次全部长度为 `K` 的窗口，生成双方立即胜点、每个候选的后续胜点集合、单/双威胁、普通进度和着法排序分。威胁等级由 `K` 和完成点数量推导，不写死“活三、冲四、活四”等五子棋模板，因此适用于 `(5,3)`、`(7,4)`、`(9,5)`、`(11,6)`、`(15,7)` 等配置。

多个窗口共享同一完成点时只计算一个胜点；边界和对方棋子阻断的窗口不算开放威胁。普通窗口权重按棋子数量和 `K` 指数增长，并限制在胜负分以下。候选优先级为：己方立即胜、阻挡对方立即胜、己方双胜点、阻止对方双胜点、单一强制威胁或攻防兼备、消除对方单威胁、普通进度与中心性。对方潜在双威胁不会未经搜索直接选一个防守点，而是与进攻候选一起交给搜索判断。

### 2.3 Enhanced Alpha-Beta

Enhanced 使用迭代加深、主变和威胁排序、动态分支上限以及两格邻域候选。置换表以 `(Zobrist hash, player)` 为主键，条目保存深度、分数、`exact/lower/upper` bound 和最佳着法；浅条目只用于排序，足够深时才剪枝，胜负分读写按 ply 归一化。叶节点以“当前行动方威胁减对方威胁”评估，而不是固定乘防守系数。

### 2.4 MCTS + UCT

MCTS 的根节点和内部节点优先扩展强制候选，普通候选按父节点访问次数渐进开放。rollout 依次处理立即胜、立即阻挡、己方双威胁和对方双威胁防守；安静着法使用同一泛化摘要排序，截断 rollout 使用威胁差值评估。UCT 使用根玩家收益反传，并受独立的时间检查保护。

### 2.5 Hybrid Threat Search

Hybrid 在 Enhanced 基础上增加 VCF/VCT 风格强制威胁搜索、三态证明、Quiescence Search、内部节点 PVS、Aspiration Window、killer/history 和受保护的 LMR。威胁搜索只扩展攻击着法和有效防守点；返回值为 `PROVEN`、`DISPROVEN`、`UNKNOWN`，超时或节点上限只能返回 `UNKNOWN`，不能缓存为失败。单一胜点必须模拟唯一阻挡，两个不同胜点才可直接证明。无法证明时安全回退到最近一次完整 Alpha-Beta 迭代或合法保底着法。

## 3. 实验方法

### 3.1 最终矩阵

正式 runner 为 `assignment1/final_experiments.py`，输出目录为 `assignment1/experiments/final/`。manifest 记录 schema、随机种子、矩阵和当前 `gomoku_ai.py` SHA-256：

`fdd5be0d43fe1a34e10411821e8bd4c6cec9a67744fe0f97a3cec2133b618b82`

| 板块 | 对战 | 组数 | 每组 | 总局数 |
| --- | --- | ---: | ---: | ---: |
| Random 基线 | Hybrid vs Random | 6 | 50 | 300 |
| 自我对战 | Hybrid A vs Hybrid B | 6 | 50 | 300 |
| Enhanced 基准 | Baseline vs Enhanced | 6 | 50 | 300 |
| Enhanced 基准 | MCTS vs Enhanced | 6 | 50 | 300 |
| Enhanced 基准 | Hybrid vs Enhanced | 6 | 50 | 300 |

Random 组从空棋盘开始，Hybrid 黑白各 25 局。AI 对 AI 组使用每个配置 25 个合法、未终局的四 ply 中央区域开局，每个开局出现两次并交换双方颜色。每局保存胜负、原因、手数、逐步耗时、搜索统计、开局编号、随机种子、源码哈希和协议错误。写入采用 flush + fsync；manifest 不匹配时拒绝续跑。

### 3.2 其他验证

100 题跨 `N/K` 基础战术基准和 36 题多步/伪威胁基准继续保留；此前 6 题 before/after 搜索效率测试和 96 局小规模先导对战也保留，但不混入 1500 局主矩阵。优化前四个正式文件归档在 `experiments/archive_pre_optimization/`。

### 3.3 全策略补充矩阵

独立 runner `assignment1/comprehensive_evaluation.py` 对 Baseline、Enhanced、MCTS、Hybrid 使用与正式矩阵完全相同的两组配置、三档时限和每组 50 局。对 Random 时从空棋盘开始且 AI 黑白各 25 局；自我对战使用 25 个合法、未终局的四 ply 中央开局，每个开局交换 A/B 颜色。矩阵共 48 组、2400 局，其中对 Random 1200 局、自我对战 1200 局。原始数据、汇总和版本清单位于 `experiments/comprehensive_evaluation/`，不与原 1500 局混合。

## 4. 最终实验结果

以下表格均从 `experiments/final/summary.json` 生成；`W-L-D` 从该行策略视角统计，时间为该策略每步 `mean/median/P95/max` 秒。自我对战的 A/B 结果在总量上各为 25-25-0；逐组胜负会受固定开局和先手影响。

### 4.1 Hybrid 对 Random

| 配置 | 时限 | W-L-D | 胜率 | 平均手数 | 时间 mean/median/P95/max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9x9,K=4 | 0.5 | 50-0-0 | 100% | 7.7 | 0.221/0.425/0.430/0.434 |
| 9x9,K=4 | 1.0 | 50-0-0 | 100% | 7.6 | 0.444/0.850/0.857/0.859 |
| 9x9,K=4 | 5.0 | 50-0-0 | 100% | 8.0 | 2.592/4.750/4.758/4.763 |
| 15x15,K=5 | 0.5 | 50-0-0 | 100% | 10.7 | 0.278/0.426/0.429/0.434 |
| 15x15,K=5 | 1.0 | 50-0-0 | 100% | 9.9 | 0.532/0.851/0.856/0.870 |
| 15x15,K=5 | 5.0 | 50-0-0 | 100% | 11.6 | 3.188/4.756/4.758/4.763 |

Hybrid 在 300 局随机基线中全胜，但 Random 很弱，不能据此断言 Hybrid 一定强于其他搜索。5 秒预算主要增加思考时间，没有改变这一饱和胜率指标。

### 4.2 Hybrid 自我对战

| 配置 | 时限 | A W-L-D | B W-L-D | 黑胜-白胜-和 | 平均手数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9x9,K=4 | 0.5 | 25-25-0 | 25-25-0 | 50-0-0 | 10.2 |
| 9x9,K=4 | 1.0 | 26-24-0 | 24-26-0 | 50-0-0 | 10.3 |
| 9x9,K=4 | 5.0 | 25-25-0 | 25-25-0 | 50-0-0 | 10.0 |
| 15x15,K=5 | 0.5 | 25-25-0 | 25-25-0 | 28-22-0 | 25.0 |
| 15x15,K=5 | 1.0 | 25-25-0 | 25-25-0 | 28-22-0 | 28.3 |
| 15x15,K=5 | 5.0 | 25-25-0 | 25-25-0 | 27-23-0 | 23.4 |

A/B 总成绩完全平衡，但黑方胜局更多，说明先手效应仍明显。自我对战不应被解释为独立样本上的总体棋力估计。

### 4.3 以 Enhanced 为唯一基准

| 对手 | 配置 | 时限 | 对手 W-L-D | 对手胜率 | Enhanced W-L-D |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 9x9,K=4 | 0.5 | 12-38-0 | 24% | 38-12-0 |
| Baseline | 9x9,K=4 | 1.0 | 11-39-0 | 22% | 39-11-0 |
| Baseline | 9x9,K=4 | 5.0 | 14-36-0 | 28% | 36-14-0 |
| Baseline | 15x15,K=5 | 0.5 | 3-47-0 | 6% | 47-3-0 |
| Baseline | 15x15,K=5 | 1.0 | 5-45-0 | 10% | 45-5-0 |
| Baseline | 15x15,K=5 | 5.0 | 4-46-0 | 8% | 46-4-0 |
| MCTS | 9x9,K=4 | 0.5 | 19-31-0 | 38% | 31-19-0 |
| MCTS | 9x9,K=4 | 1.0 | 15-35-0 | 30% | 35-15-0 |
| MCTS | 9x9,K=4 | 5.0 | 22-28-0 | 44% | 28-22-0 |
| MCTS | 15x15,K=5 | 0.5 | 20-26-4 | 40% | 26-20-4 |
| MCTS | 15x15,K=5 | 1.0 | 17-32-1 | 34% | 32-17-1 |
| MCTS | 15x15,K=5 | 5.0 | 5-44-1 | 10% | 44-5-1 |
| Hybrid | 9x9,K=4 | 0.5 | 25-25-0 | 50% | 25-25-0 |
| Hybrid | 9x9,K=4 | 1.0 | 24-26-0 | 48% | 26-24-0 |
| Hybrid | 9x9,K=4 | 5.0 | 25-25-0 | 50% | 25-25-0 |
| Hybrid | 15x15,K=5 | 0.5 | 15-35-0 | 30% | 35-15-0 |
| Hybrid | 15x15,K=5 | 1.0 | 20-30-0 | 40% | 30-20-0 |
| Hybrid | 15x15,K=5 | 5.0 | 28-22-0 | 56% | 22-28-0 |

跨六组配置聚合时，Baseline 对 Enhanced 为 `49-251-0`（16.3%），MCTS 为 `98-196-6`（32.7%，积分率 33.7%），Hybrid 为 `137-163-0`（45.7%）。Enhanced 作为基准方在这些 900 局中的总结果为 `610-284-6`；该数字包含三种对手，不应与单一配对胜率混用。Hybrid 在 0.5/1.0 秒大棋盘落后，但 5 秒大棋盘为 28-22，说明威胁证明需要足够预算才能抵消额外分析开销；这也是“组合更多技术”不等于短时限一定更强的直接证据。

### 4.4 其他基准与合规

历史当前版本基准仍为：100 题一步战术 Enhanced/Hybrid/MCTS 均 100%；36 题多步基准 Enhanced 35/36、Hybrid 36/36、MCTS 170/180；搜索效率测试中 Enhanced 冷评估加速 6.45x、Hybrid 加速 6.08x，均有 4/6 局面提升至少一层且没有下降；96 局先导实验积分为 Enhanced 37、Hybrid 33、MCTS 26。先导实验只用于策略选择背景，不替代本次 1500 局主矩阵。

最终矩阵 1500 行中 0 次 timeout、0 次 exception、0 次 illegal move、0 次 input-board modification。课程 `arena.py` 的 15x15、K=5、5 秒双局验证也保持通过。

### 4.5 四种 AI 对 Random

下表来自补充矩阵的 1200 局。四种 AI 在每个配置和时限下均为 `50-0-0`，因此 Random 胜率已饱和，主要可比较量是平均每步思考时间和平均局长。时间为秒。

| AI | 配置 | 时限 | W-L-D | 平均每步 | P95 | 平均手数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 9x9,K=4 | 0.5 | 50-0-0 | 0.0014 | 0.0030 | 8.58 |
| Baseline | 9x9,K=4 | 1.0 | 50-0-0 | 0.0015 | 0.0034 | 8.78 |
| Baseline | 9x9,K=4 | 5.0 | 50-0-0 | 0.0041 | 0.0128 | 8.22 |
| Baseline | 15x15,K=5 | 0.5 | 50-0-0 | 0.0029 | 0.0059 | 15.58 |
| Baseline | 15x15,K=5 | 1.0 | 50-0-0 | 0.0031 | 0.0063 | 16.38 |
| Baseline | 15x15,K=5 | 5.0 | 50-0-0 | 0.0063 | 0.0150 | 15.94 |
| Enhanced | 9x9,K=4 | 0.5 | 50-0-0 | 0.2239 | 0.4284 | 7.78 |
| Enhanced | 9x9,K=4 | 1.0 | 50-0-0 | 0.4416 | 0.8535 | 7.78 |
| Enhanced | 9x9,K=4 | 5.0 | 50-0-0 | 2.4948 | 4.7575 | 7.74 |
| Enhanced | 15x15,K=5 | 0.5 | 50-0-0 | 0.2671 | 0.4277 | 10.10 |
| Enhanced | 15x15,K=5 | 1.0 | 50-0-0 | 0.5406 | 0.8548 | 10.14 |
| Enhanced | 15x15,K=5 | 5.0 | 50-0-0 | 2.9461 | 4.7637 | 9.98 |
| MCTS | 9x9,K=4 | 0.5 | 50-0-0 | 0.2273 | 0.4288 | 7.66 |
| MCTS | 9x9,K=4 | 1.0 | 50-0-0 | 0.4438 | 0.8539 | 7.74 |
| MCTS | 9x9,K=4 | 5.0 | 50-0-0 | 2.4502 | 4.7575 | 7.58 |
| MCTS | 15x15,K=5 | 0.5 | 50-0-0 | 0.2752 | 0.4279 | 10.10 |
| MCTS | 15x15,K=5 | 1.0 | 50-0-0 | 0.5569 | 0.8566 | 10.34 |
| MCTS | 15x15,K=5 | 5.0 | 50-0-0 | 2.9741 | 4.7649 | 10.14 |
| Hybrid | 9x9,K=4 | 0.5 | 50-0-0 | 0.2199 | 0.4286 | 7.70 |
| Hybrid | 9x9,K=4 | 1.0 | 50-0-0 | 0.4335 | 0.8534 | 7.62 |
| Hybrid | 9x9,K=4 | 5.0 | 50-0-0 | 2.5409 | 4.7572 | 7.74 |
| Hybrid | 15x15,K=5 | 0.5 | 50-0-0 | 0.2658 | 0.4275 | 10.02 |
| Hybrid | 15x15,K=5 | 1.0 | 50-0-0 | 0.5315 | 0.8559 | 10.02 |
| Hybrid | 15x15,K=5 | 5.0 | 50-0-0 | 3.1552 | 4.7643 | 11.34 |

Baseline 不使用完整时间预算，因而平均每步仅为毫秒级；其余三种高级方法在安静局面通常使用接近软截止时间的预算。`15x15,K=5` 的平均局长普遍高于 `9x9,K=4`，但 Random 太弱，四种算法全胜，不能据此排序棋力。

### 4.6 四种 AI 自我对战

自我对战的 A/B 是同一算法的两个独立实例。`黑-白-和` 展示颜色效应，平均手数包括预置的四 ply 开局。

| AI | 配置 | 时限 | A W-L-D | B W-L-D | 黑-白-和 | 平均手数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 9x9,K=4 | 0.5 | 25-25-0 | 25-25-0 | 40-10-0 | 11.04 |
| Baseline | 9x9,K=4 | 1.0 | 25-25-0 | 25-25-0 | 40-10-0 | 11.04 |
| Baseline | 9x9,K=4 | 5.0 | 25-25-0 | 25-25-0 | 40-10-0 | 11.04 |
| Baseline | 15x15,K=5 | 0.5 | 25-25-0 | 25-25-0 | 34-16-0 | 32.76 |
| Baseline | 15x15,K=5 | 1.0 | 25-25-0 | 25-25-0 | 34-16-0 | 32.76 |
| Baseline | 15x15,K=5 | 5.0 | 25-25-0 | 25-25-0 | 34-16-0 | 32.76 |
| Enhanced | 9x9,K=4 | 0.5 | 25-25-0 | 25-25-0 | 46-4-0 | 10.12 |
| Enhanced | 9x9,K=4 | 1.0 | 25-25-0 | 25-25-0 | 44-6-0 | 9.92 |
| Enhanced | 9x9,K=4 | 5.0 | 25-25-0 | 25-25-0 | 46-4-0 | 9.64 |
| Enhanced | 15x15,K=5 | 0.5 | 25-25-0 | 25-25-0 | 34-16-0 | 36.00 |
| Enhanced | 15x15,K=5 | 1.0 | 26-24-0 | 24-26-0 | 35-15-0 | 31.10 |
| Enhanced | 15x15,K=5 | 5.0 | 25-24-1 | 24-25-1 | 31-18-1 | 39.68 |
| MCTS | 9x9,K=4 | 0.5 | 24-26-0 | 26-24-0 | 43-7-0 | 10.86 |
| MCTS | 9x9,K=4 | 1.0 | 24-26-0 | 26-24-0 | 45-5-0 | 11.90 |
| MCTS | 9x9,K=4 | 5.0 | 27-23-0 | 23-27-0 | 42-8-0 | 10.56 |
| MCTS | 15x15,K=5 | 0.5 | 20-15-15 | 15-20-15 | 24-11-15 | 98.22 |
| MCTS | 15x15,K=5 | 1.0 | 21-21-8 | 21-21-8 | 28-14-8 | 67.36 |
| MCTS | 15x15,K=5 | 5.0 | 25-21-4 | 21-25-4 | 26-20-4 | 51.60 |
| Hybrid | 9x9,K=4 | 0.5 | 25-25-0 | 25-25-0 | 44-6-0 | 10.16 |
| Hybrid | 9x9,K=4 | 1.0 | 25-25-0 | 25-25-0 | 44-6-0 | 10.16 |
| Hybrid | 9x9,K=4 | 5.0 | 25-25-0 | 25-25-0 | 46-4-0 | 9.96 |
| Hybrid | 15x15,K=5 | 0.5 | 26-24-0 | 24-26-0 | 27-23-0 | 24.30 |
| Hybrid | 15x15,K=5 | 1.0 | 25-25-0 | 25-25-0 | 34-16-0 | 32.92 |
| Hybrid | 15x15,K=5 | 5.0 | 25-25-0 | 25-25-0 | 38-12-0 | 23.40 |

颜色交换使确定性策略的 A/B 总成绩基本平衡，但黑方在几乎所有组中明显占优。MCTS 在大棋盘产生 `15/8/4` 局和棋，平均局长随预算增加由 98.22 降至 51.60 手，说明更多迭代有助于更早形成或识别有效进攻。Hybrid 在大棋盘的平均局长为 24.30、32.92、23.40，明显短于同档 MCTS；Enhanced 的 5 秒组出现一局满盘和棋，平均为 39.68 手。补充矩阵共 2400 局，0 timeout、0 exception、0 illegal move、0 input-board modification。

## 5. 局限性与讨论

1. Random 基线过弱，Hybrid 全胜只能证明协议和基础战术稳定，不能给出强棋力结论。
2. 确定性自我对战和固定四 ply 开局存在样本相关性；颜色交换能平衡角色，但不能消除先手优势。
3. 威胁分类可能产生伪威胁，VCF/VCT 分支在复杂中局中会膨胀；`UNKNOWN` 回退意味着不能证明所有长强制链。
4. LMR、候选上限和窗口启发式可能漏掉安静关键着法；普通窗口仍可能重叠计分，权重尚未机器学习校准。
5. Python 线程 watchdog 无法强制终止失控外部 AI；进程级裁判会提供更强隔离。
6. 5 秒组揭示了 Python 搜索的成本：MCTS 对 Enhanced 曾出现 225 手、约 900 秒的合法满盘和棋。它没有违反单步时限，但总局时长应在报告中单独说明。

## 6. 复现

```powershell
# 最终 1500 局，支持断点续跑
python assignment1/final_experiments.py --only all --resume

# 从原始 JSONL 重建汇总；manifest 会自动读取 games-per-group
python assignment1/final_experiments.py --summarize

# 四种 AI 对 Random 及各自自我对战，共 2400 局
python assignment1/comprehensive_evaluation.py --only all --resume
python assignment1/comprehensive_evaluation.py --summarize

# 验证
python -m compileall -q assignment1
python -m unittest discover -s assignment1/tests -v
python assignment1/tactical_benchmark.py --time-limit 0.15
python assignment1/advanced_tactical_benchmark.py --time-limit 0.3
python assignment-1-code/arena.py assignment1/gomoku_ai.py assignment-1-code/random_ai.py --board 15 --win 5 --time 5 --games 2 --quiet
```

最终原始数据见 [`experiments/final/results.jsonl`](experiments/final/results.jsonl)，汇总见 [`experiments/final/summary.json`](experiments/final/summary.json)，版本清单见 [`experiments/final/manifest.json`](experiments/final/manifest.json)。优化前正式文件只保留在 `experiments/archive_pre_optimization/`；战术、效率和 96 局先导数据继续保留在 `experiments/` 下。

全策略补充矩阵的原始数据、汇总和版本清单分别见 [`experiments/comprehensive_evaluation/results.jsonl`](experiments/comprehensive_evaluation/results.jsonl)、[`summary.json`](experiments/comprehensive_evaluation/summary.json) 和 [`manifest.json`](experiments/comprehensive_evaluation/manifest.json)。

## 参考资料

课程讲义《广义五子棋游戏与对抗搜索（Assignment-1）》、课程提供的 `arena.py` 和 `random_ai.py`。本项目未调用外部棋力引擎、网络服务或预计算应答表。
