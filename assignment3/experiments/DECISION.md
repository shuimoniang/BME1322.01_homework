# v2 迭代定稿决策记录（主会话）

## 背景

v1 基线（`experiments/v1_baseline/`）：Part A（学模型+VI）与 Part B（离线 Q-Learning α=0.5）
策略完全一致，评估回报 88.01 / B3 占比 63.4% / B1 终止率 0.8%。评分重点是算法正确、
实验完整、解释清楚（PDF §5），不是最高回报；且两个提交文件的行为被讲义公式与协议锁定。

三个 Builder 并行探索（脚本与结果 JSON 留档于本目录）：

| Builder | 维度 | 核心发现 | 结论 |
|---|---|---|---|
| A 稀疏先验 | 未访问 (s,a) 的先验选择 | 2000 条预算下 36/36 覆盖，4 种先验（uniform/Laplace/悲观/乐观）策略与指标逐位相同；差异只在 n<25 的极端稀疏场景出现，悲观先验在 n=10 唯一稳健 | **不改提交默认**，作为消融留档 |
| B 学习率调度 | 离线 Q-Learning 的 α | 固定 α 下 Q 偏差随 α 单调放大（α=0.5 L2≈16.1）；α(n)=max(0.05,1/n^0.6) L2=4.91 最接近 Q*，且双数据量策略稳定；α=1.0 在 2000 条下策略翻转（B6），回报跌至 75.56 | **不改提交默认 α=0.5**，调度作为附加实验进报告 |
| C 评估与规划 | 统计严谨性 + 规划变体 | 5 块×500 条评估：回报 88.81±1.05，t 95% CI [87.51, 90.12]；B1 率池化 2500 条 0.44%，Wilson 95% [0.25%, 0.79%]（v1 的 0.8% 在区间上缘，属抽样波动）；bootstrap(B=30)：9 状态策略稳定率 100%，B4 翻转 margin 0.58 全表最小；异步 VI 仅省 19% 扫描、Dyna 5 万次更新不达同步 VI 精度 | 多块评估+CI、bootstrap、规划变体小表值得进 v2；Dyna 只作反例引用 |

## v2 采纳清单

1. **多块评估 + 置信区间**（Builder C-1）：主评估升级为 5 块 × 500 条（种子段 90000–94000），
   回报与 B3 占比报 t 区间（df=4），B1 终止率报池化 Wilson 区间。第 1 块种子段与 v1 完全一致
   （90000+i），保证 v1 数字可追溯。
2. **Bootstrap 策略稳定性**（Builder C-3，B=30，seed=1322）：量化"最可能翻转状态"——
   B4 margin 0.58 全表最小，30 次重采样 0 翻转；写入报告消融小节。
3. **规划变体消融**（Builder C-2）：同步 VI（提交协议）/ 异步 Gauss-Seidel / Q 直接迭代
   同模型同精度对比，论证小规模下同步扫描已最优；Dyna 采样作为反例一句话引用。
4. **学习率调度消融**（Builder B）：固定 α∈{0.1,0.3,0.5,0.7,1.0} + Robbins-Monro 调度，
   报 Q 与 Q̂（Part A 参照）的距离与策略一致性；α=1.0 翻转案例是"固定 α 不满足收敛三条件"
   的实证，进报告 Part B 小节。
5. **稀疏先验消融**（Builder A）：4 种先验 × 8 档数据量，结论"课程预算下不敏感、极端稀疏下
   悲观先验最稳"，进报告 Part A 未访问处理小节。

## 不采纳清单

- 修改提交文件默认行为（先验/α/规划方式）：三个 Builder 一致结论为无收益或违反协议。
- Dyna 式采样规划进主流程：精度差、无策略收益。
- 单独把 bootstrap/消融写成独立脚本常驻：统一并入 main.py 的 v2 实验段，保持单一入口。

## 复现

- v1：`experiments/v1_baseline/`（代码+报告+figures 快照 + README）。
- Builder 探索：`experiments/builder_*_*.py` + `*_results.json`。
- v2 主实验：`python assignment3/main.py`（根目录运行），结果写 `figures/results.json`
  与 `experiments/v2_optimized/`（含 results.json 快照与 DECISION 引用）。
- 硬约束：`experiments/CONSTRAINTS.md`（候选合规判定依据）。

## 第四版附加消融（2026-11-07）

v3 定稿后，在不重复 Builder A/B/C 与 v3 在线实验的前提下，补两个空白维度的附加消融。
均不改动提交文件（main.py / model_based.py / q_learning.py 保持 manifest 哈希不变），
默认参数（α=0.5、γ=0.9、统一数据协议）不变，结果由独立脚本留档、报告由
write_v3_report.py 重建时读取对应 JSON 生成第 6–7 节。

| Builder | 维度 | 核心发现 | 结论 |
|---|---|---|---|
| D 数据采集种子 | 换 10 个采集种子重跑全管线（评估种子段固定 90000–90499） | Part A 策略 10/10 种子逐位相同、指标恒为 88.01/63.43%/0.8%，B4 从不翻转（margin 0.556–0.681）；Part B 回报 42.74±51.13（-40.05 至 88.01），8/9 非主种子策略与主种子不同，B1 率最高 45.2%，B4 在 3 个种子翻转 | 主种子下两路线一致属种子特例；模型路线对抽样鲁棒、离线单遍 TD 高度敏感——analysis.md 第 1 问的实证。留档进报告第 6 节 |
| E 折扣因子 | γ∈{0.5,0.8,0.9,0.95,0.99}，同一份 seed=2026 数据 | 策略对 γ 完全平坦（所有 γ 下两路线策略逐状态相同、评估指标不变）；γ 只放大价值尺度与求解成本：V̂(B3) 1.09→46.00，VI 14→840 轮（接近 1000 上限），max|Q| 0.99 时 58.62 | “γ 越大越避开 B1”被推翻（最优动作本在低血糖侧立即补糖）；γ=0.9 位于平坦区间、计算成本居中。留档进报告第 7 节 |

## 第四版复现

- Builder D：`python assignment3/experiments/builder_d_data_seed.py`（约 6 s）→ `builder_d_data_seed_results.json`、`figures/fig8_data_seed_robustness.png`、`builder_d_run.log`。
- Builder E：`python assignment3/experiments/builder_e_gamma.py`（约 3 s）→ `builder_e_gamma_results.json`、`figures/fig9_gamma_sensitivity.png`、`builder_e_run.log`。
- 报告：`python assignment3/experiments/write_v3_report.py`（从 v3 正式 JSON 与两个 builder JSON 重建 report.md / analysis.md 全文）。
- 合规：两个脚本均只通过 `collect_random_episodes` 与给定经验交互，不读环境内部常量，不调用现成 RL/统计库；脚本自身 SHA-256 已写入各自 results.json。
