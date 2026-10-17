# 贝叶斯网络迭代日志（v3）

评测协议：训练集 60,000 张内部固定分层划分（seed 1322，49,999 训练 / 10,001 验证），报告验证准确率。测试集（10,000 张）全程未参与选择，仅在最终定稿后评估一次。

基线（v2 定稿，`experiments_v2/bayes_net/`）：验证 84.92%，测试 85.33%。

每轮实验脚本在本目录（`round1_directional.py` … `round6_final.py`，另有 `final_eval.py`），公共数据管线在 `helper.py`（复用 v2 的 `common.py`，新方向性统计缓存在 `cache/`）。

## Round 1 — 方向性特征逐槽位替换（`round1_directional.py`）

**假设**：v2 最大瓶颈是 2/3/5 混淆（5→3 183、2→3 107、3→5 69），区域密度丢失笔画方向信息；把方向性/开口/结构特征换入单个槽位应当能攻此瓶颈。

候选新特征（共 14 个）：梯度竖直占比 grad_vert_frac、梯度向上占比 grad_up_frac、四角开口墨迹（upper/lower × left/right band ink）、中央侧壁开口（mid_left/right_ink）、水平穿越次数 horiz_runs、端点数 endpoints、水平三带密度 band_top/mid/bot、笔画宽度 stroke_width。

结果：单一槽位替换全部低于基线 84.92%。最好的几次：

| 替换 | 验证 Acc |
|:---|---:|
| grid_br ← lower_right_ink(4) | 84.42% |
| grid_bl ← band_bot(4) | 84.18% |
| vdom ← horiz_runs(5) | 84.42% |
| aspect ← endpoints(4) | 83.83% |

**结论：放弃**——方向性特征与网格密度高度冗余，且网格密度 5 档 CPT 信息量更大（与 v2 Round 5 结论一致）。

## Round 2 — 双槽组合与新接线（`round2_combine.py`）

**假设**：两个最优替换单槽组合可能产生互补增益；给新特征换更合理的父节点可能有帮助。

最优组合 84.67%（br←lower_right_ink 且 bl←mr），其余组合 82.8–84.0%，全部低于基线。**继续放弃方向性替换路线**。

## Round 3 — 离散化档数精修（`round3_discretize.py`）

**假设**：方向性特征档数没调好；网格密度档数也许还有余量。

- 方向性特征 3/4/5 档扫描：最高 br←lower_right_ink 5 档 84.80%，仍低于基线。
- 网格密度整体 6 档：**85.07%**（超过基线），但违反作业"每特征 2–5 个取值"约束，**放弃**。

**结论**：特征层面在 2–5 档约束内已无廉价增益，转向 DAG 结构。

## Round 4 — DAG 变体扫描（`round4_dag_smoothing.py`）

**假设**：v2 的列链（自上而下）只是若干人工候选之一；行的方向（自左向右书写顺序）可能更符合墨迹相关性。

在完全相同的 12 特征上比较人工候选 DAG：

| 候选 | 验证 Acc |
|:---|---:|
| v2 列链（基线） | 84.92% |
| 列链反转（下→上） | 84.47% |
| **行链（tc←tl, mc←ml, tr←tc, mr←mc, br←mr）** | **85.87%** |
| 行链反转（右→左） | 84.22% |
| 中心枢纽 mc | 84.35% |
| 对角链 | 83.75% |
| 无特征边（纯 NB） | 83.37% |
| 语义链 vruns←euler、aspect←vruns、vdom←vruns | 85.02% |

平滑细调（alpha 0.5/1/2、硬回退）差异 ≤0.06pt，硬回退实现有缺陷且无增益，放弃。

**发现**：行链（左→右，与书写方向一致）比列链高约 +0.95pt；语义特征接成以 vertical_runs 为中心的链也优于 v2 的两条独立边。

## Round 5 — 行链变体与语义接线（`round5_rowchain.py`）

**假设**：行链底行未闭合（bottom_center 已删），补一条边；语义链接法还有优化空间。

| 变体 | 验证 Acc |
|:---|---:|
| 行链 A + v2 语义边 | 85.87% |
| **行链 C（A + bl←br 闭合底行）+ v2 语义边** | **85.96%** |
| 行链 C + vruns←euler、aspect←vruns、vdom←vruns（语义链） | 86.14% |
| 行链 C + 其他语义接法 | 84.2–86.0% |

## Round 6 — 终局组合、alpha 与跨 seed 验证（`round6_final.py`）

| 配置 | 验证 Acc |
|:---|---:|
| C + 语义链，alpha=0.3 / 0.5 / 1.0 / 2.0 | 86.16 / 86.15 / 86.14 / 86.15% |
| C + 语义链，无 backoff | 86.08% |
| C + v2 语义边（P_C），alpha=1.0 | 85.96% |

alpha 不敏感（与 v2 结论一致），保留协议默认 alpha=1.0 + backoff。

跨 seed 稳健性（同一配置、不同划分）：

| seed | v3（C+语义链） | v2 基线 |
|:---|---:|---:|
| 1322 | 86.14% | 84.92% |
| 7 | 84.55% | 83.51% |
| 2024 | 85.86% | 84.64% |

三个 seed 均领先基线 +0.95~1.22pt，增益稳健，非单次划分偶然。

## 定稿（v3）

- **特征**：与 v2 完全相同（8 网格密度 + euler + vruns + aspect + vdom），未动 `FEATURE_NAMES` / `FEATURE_CARDINALITIES` / `extract_features`。
- **DAG（8 条特征间边）**：行链 `tl→tc→tr`、`ml→mc→mr`、`br→mr`、`br→bl`；语义链 `euler→vruns`、`vruns→aspect`、`vruns→vdom`。即 `FEATURE_PARENTS = (-1, 0, 1, -1, 3, 4, 7, 5, -1, 8, 9, 9)`。
- 每特征父节点 ≤2，Y 为根，无隐藏变量，全人工候选验证集选择，合规。
- `tests/test_models.py` 9 项测试通过（`FEATURE_PARENTS` 改动不影响断言：仍 ≥2 条边、DAG 无环、每节点 ≤2 父）。
- `main.py` plot_dag 仅改注释与标题（布局恰好无需变动：行链边在行内横向、语义链边在右侧向下/横向，均不交叉）。

**最终测试集（60,000 训练重训，alpha=1.0）：Accuracy 86.27%，Macro F1 86.08%**（详见 `final_test_result.md` / `final_test_result.json`；对比 v2 测试 85.33%，**+0.94pt**；对比 v1 基线 83.33%，+2.94pt）。

## 备注：管线修复记录

v3 初版 helper 复现基线只得 83.01%，定位后发现 `grid_densities` 的网格列序为行主序（j=7 是 bottom_left、j=8 是 bottom_center），v2 模块删除的是 bottom_center (j=8) 而非 bottom_left (j=7)。修正 cell 选择后基线精确复现 84.92%（与 `bayes_net.py` 直接评估逐位一致）。Round 1/2 的初次输出使用的是错误 cell，已在修正后全部重跑，本日志只记录修正后的数字。
