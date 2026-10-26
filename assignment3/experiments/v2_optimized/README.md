# v2 优化实验留档（2026-10-26）

第二版迭代的全部产物。复现入口：仓库根目录 `python assignment3/main.py`
（结果同时写入 `figures/results.json` 与本目录 `results.json`）。

## 迭代流程

1. **v1 基线**：`experiments/v1_baseline/`（代码、报告、figures 快照 + README）。
2. **硬约束**：`CONSTRAINTS.md`（从作业 PDF 逐句抽取的合规判定依据，候选方案逐条核对）。
3. **三 Builder 并行探索**（脚本 + 结果 JSON 均在本目录）：
   - `builder_a_sparse_prior.{py,json}` —— 未访问 (s,a) 先验消融（4 先验 × 8 数据量）。
   - `builder_b_lr_schedule.{py,json}` —— 离线 Q-Learning 学习率调度（固定 α 扫描 + Robbins-Monro）。
   - `builder_c_eval_planning.{py,json}` —— 多块评估/置信区间、bootstrap 稳定性、规划变体。
4. **定稿决策**：`DECISION.md`（采纳/不采纳清单与理由）。
5. **v2 集成**：main.py 新增 4 组附加实验（多块评估+CI、bootstrap、规划变体、学习率调度），
   提交文件 model_based.py / q_learning.py 与 v1 逐字节相同（探索结论：协议内已最优）。

## v2 关键数字（来源 figures/results.json，version=v2_optimized）

- 多块评估（5 块 × 500 条）：回报 88.81，t 95% CI [87.51, 90.12]；B1 率 0.44%，
  Wilson 95% [0.25%, 0.79%]（v1 单块 0.8% 在区间上缘，属抽样波动）。
- Bootstrap（B=30, seed=1322）：9 状态策略稳定率 100%；最脆弱 B4（top-2 margin 0.60）；
  V(B4) 95% 百分位区间 [3.97, 4.84]。
- 规划变体：同步 VI 82 轮 / 异步 Gauss-Seidel 66 轮 / Q 直接迭代 66 轮，策略全部一致。
- 学习率调度：固定 α 偏差单调放大（L2: 5.6→10.7→16.1→22.4→34.3）；
  α=1.0 策略翻转 4 状态、回报 75.56；α(n)=max(0.05,1/n^0.6) L2=4.91 最优且策略稳定。
- 主实验数字与 v1 一致：Part A/B 策略相同，评估 88.01 / 63.4% / 0.8%（第 1 块种子段同 v1）。

## 包含文件

`results.json`（v2 全量结果快照）、`CONSTRAINTS.md`、`DECISION.md`、三个 builder 脚本与结果 JSON、`v1_baseline/`。
