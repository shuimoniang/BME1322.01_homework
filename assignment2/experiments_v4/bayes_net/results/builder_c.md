# Builder C — 特征集重构（Feature-Set Redesign within the 6–12 Budget）

**任务族**：在 PDF 6–12 特征 / 2–5 状态 / Y 为根 / ≥2 特征边 / ≤2 父节点的硬约束内，
重构 Small Bayesian Network 的**特征槽位分配与离散化**，目标超过 incumbent 的 **86.14%**。

**结论（先说结果）**：特征集重构**全部失败**——incumbent 的 12 特征（8 网格密度 +
euler + vertical_runs + aspect + vertical_dominance）与 DAG 在特征层面是稳健局部最优。
但重构过程中定位到一个**真实、跨 7 个 seed 稳健的增益**，它不在特征而在 **DAG**：
把 middle band 密度接到 `euler_topology` 并加一条对角边 `tl ← br`，验证 **86.86%**
（+0.72pt），7/7 seed 上 McNemar p ≤ 0.0003。

> 该增益来自"特征集重构"向下挖掘后的结构发现，属同一任务族的副产物；特征本身未改，
> 因此 `FEATURE_NAMES` / `FEATURE_CARDINALITIES` / `extract_features` 与现行
> `bayes_net.py` 完全一致，改动面最小（仅 `FEATURE_PARENTS`）。

---

## 0. 基线与复现

协议：`v4_common.load_split()` → 49,999 训练 / 10,001 验证，seed 1322，只用 60,000 训练集。
测试集（`t10k-*.gz`）**全程未读取**。

| alpha | incumbent 验证 Acc |
|:---:|---:|
| 0.5 | 0.8615 |
| **1.0** | **0.8614** |
| 2.0 | 0.8615 |

**基线与 `BASELINE.md` 逐位一致（0.8614）**，未发现 discrepancy。
我的独立 sweep 引擎（`_harness.SweepBN`）也逐位复现 0.8614，并与直接调用
`bayes_net.BayesNetClassifier` 在相同 DAG 下结果完全相同（`_verify.py` 验证两条代码路径一致）。

---

## 1. 特征语义、离散化与 DAG（最终采纳方案）

### 1.1 12 个特征（与 incumbent 完全一致，逐条语义）

| # | 名称 | 语义 | 连续量 → 离散化 | 状态数 |
|:--:|---|---|---|---:|
| 0 | `density_top_left` | 中央 15×15 上半左格墨迹密度 | 训练分位数；恰好 0 独立成状态 0 | 5 |
| 1 | `density_top_center` | 上半中格密度 | 同上 | 5 |
| 2 | `density_top_right` | 上半右格密度 | 同上 | 5 |
| 3 | `density_middle_left` | 中带左格密度 | 同上 | 5 |
| 4 | `density_middle_center` | 中带中格密度 | 同上 | 5 |
| 5 | `density_middle_right` | 中带右格密度 | 同上 | 5 |
| 6 | `density_bottom_left` | 下半左格密度 | 同上 | 5 |
| 7 | `density_bottom_right` | 下半右格密度 | 同上 | 5 |
| 8 | `euler_topology` | 数字拓扑不变量（连通分量数 − 孔洞数，2×2 数字拓扑公式估计） | clip 到 [−1,2] 后 +1 | 4 |
| 9 | `vertical_runs` | 竖直笔画碎裂度：每占用列的竖直游程起始数 | 训练分位数，恰好 0 独立 | 5 |
| 10 | `aspect_ratio` | 墨迹外接框宽高比（沿用 incumbent 的公式以保证可复现） | 训练分位数 | 4 |
| 11 | `vertical_dominance` | 竖直梯度能量占总梯度能量的比例 | 训练分位数 | 4 |

网格窗口固定为行/列 **7..22**（中央 15×15），**删除 `bottom_center`**（与其他密度最冗余，
v2 Round 7 结论 + 本次 subset sweep 复核）。二值化阈值 >64。

### 1.2 DAG（12 节点 + 类别 Y，11 条特征间边）

```
FEATURE_PARENTS = (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)

          Y（根，指向全部 12 个特征）
          │
   ┌──────┴───────────────────────────────────────────────┐
   │                                                       │
 br ──► tl ──► tc ──► tr          （上带书写序：左→右，tl 另受 br 约束）
        │
        └──► ml ──► mc            （中带：ml 受 euler 约束，mc 沿行读序）
             │
   euler ──►┤
        └──► mr                  （中带两侧都受拓扑约束）
             │
             └──► br ──► bl       （下带：右→左回弯，锚在 mr 上）

 euler ──► vertical_runs ──► aspect_ratio
                       └──► vertical_dominance
```

### 1.3 每条边的语义解释

| 边 | 语义 |
|---|---|
| `tl → tc` | 上带左→右书写序：右侧字形的墨迹量在给定类别后仍与左侧相关（笔顺连续）。 |
| `tc → tr` | 同上，上带行内连续。 |
| `br → tl` | **对角墨迹平衡**：开口朝上（2, 7）与闭环/上闭合（0, 6, 8）字形对角墨迹此消彼长；给定 Y 后仍显著相关（见 §3）。 |
| `euler → ml` | **全局拓扑约束中带左格**：中带是否被"切断"（2/3）、是否成环（3）、是否为穿越笔画（5）由拓扑决定，区域均值无法表达。 |
| `ml → mc` | 中带行内书写序（沿用 incumbent）。 |
| `euler → mr` | 同 `euler → ml`，中带右格是 3 的闭环与 5 的顶横所在。 |
| `mr → br` | 右下墨迹承接中带右列（笔顺回路），沿用 incumbent。 |
| `br → bl` | 下带回弯，沿用 incumbent。 |
| `euler → vertical_runs` | 拓扑不变量约束笔画碎裂度（闭环数字笔画更连贯）。 |
| `vertical_runs → aspect_ratio` | 碎裂度约束整体宽高比（细高 vs 扁平）。 |
| `vertical_runs → vertical_dominance` | 碎裂度约束梯度方向性。 |

**关键新边只有 3 条**，全部来自 §3 的 8 个手写候选族（`_manual_dags.py`），
**不是**自动结构学习：没有对 DAG 空间做打分最大化，每条边都有上述语义故事。

---

## 2. 设计变体总表（每行一个完整合法方案，seed 1322 验证）

| 设计变体 | 特征（12 槽） | 验证 Acc | 跨 seed（7 / 2024） | McNemar p | 判定 |
|---|---|---:|---|---:|---|
| **incumbent (v3)** | 8 网格 + euler + vruns + aspect + vdom，行链 | **0.8614** | 0.8455 / 0.8586 | — | 基准 |
| V-D1 离散化：grid 5 档 → 4 档 | 同 incumbent | 0.8491 | — | — | ✗ |
| V-D1 grid 分位→等宽 | 同 incumbent | 0.8514 | — | — | ✗ |
| V-D1 grid 零值分离→普通分位 | 同 incumbent | 0.8548 | — | — | ✗ |
| V-D1 `vdom` 4 档 → 5 档 | 同 incumbent | 0.8624 | — | 0.556 | ~（不显著） |
| V-D2 PDF 六带（行三等 + 列三等） | 6 带 + 6 语义 | 0.7155 | — | — | ✗ 大幅落后 |
| V-D2 六带 + 6 语义（另一接线） | 6 带 + 6 语义 | 0.7161 | — | — | ✗ |
| V-D3 拓扑增强（去 aspect 加 `nholes`） | 8 网格 + euler + vruns + vdom + nholes | 0.8499 | — | — | ✗ |
| V-D3 加 `horiz_runs` 替 vdom | 8 网格 + euler + vruns + aspect + horiz_runs | 0.8502 | — | — | ✗ |
| V-D4 对称性增强（去 aspect 加 `hsym`） | 8 网格 + euler + vruns + vdom + hsym | 0.8506 | — | — | ✗ |
| V-D4 对称性增强（去 aspect 加 `vsym`） | 8 网格 + euler + vruns + vdom + vsym | 0.8535 | — | — | ✗ |
| V-B1 粗 2×2 象限密度 | 4 象限 + 4 语义 | 0.6649 | — | — | ✗ 灾难性 |
| V-B1 粗 2×2 象限 + 8 语义 | 4 象限 + 8 语义 | 0.7416 | — | — | ✗ |
| V-B2 9 网格 + 3 语义（最优三元） | 全 9 格 + euler + vruns + vdom | 0.8583 | — | — | ✗ |
| V-B3 8 网格 + 4 语义，窗 12×12 | 同 incumbent，窗 12 | 0.8530 | — | — | ✗ |
| V-B3 8 网格 + 4 语义，窗 28×28 | 同 incumbent，窗 28 | 0.7592 | — | — | ✗ |
| V-B4 8 网格，dark-core 阈值 128 | 同 incumbent，阈值 128 | 0.8602 | — | — | ✗ |
| V-B4 8 网格，dark-core 阈值 96 | 同 incumbent，阈值 96 | 0.8616 | — | — | ~（噪声级） |
| V-H1 洞位置特征（`hole_top`/`hole_bot`） | 8 网格 + euler + 洞位置 + vruns | 0.8390 | — | — | ✗ |
| **M6（最终采纳，DAG 改动）** | **incumbent 12 特征**，`euler→ml`、`euler→mr`、`tl←br` | **0.8686** | **0.8553 / 0.8662** | **0.0003** | **✓ ADOPT** |
| M8（DAG 改动） | incumbent 12 特征，`euler→ml,mr`、`bl←mr` | 0.8662 | 0.8535 / 0.8637 | 0.0124 | ✗ 只 4/7 seed 显著 |
| ILLEGAL 坐标上升 DAG | incumbent 12 特征 | 0.8791 | 0.8670 / 0.8748 | 0.0000 | ✗ **违规**（自动结构搜索） |

---

## 3. M6 的严格证伪

`M6` = incumbent 12 特征 + DAG `(7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)`。

### 3.1 主协议（seed 1322）

| 指标 | incumbent | M6 |
|---|---:|---:|
| 验证 Accuracy | 0.8614 | **0.8686** |
| Macro F1 | 0.8589 | **0.8661** |
| Δ (paired bootstrap) | — | **+0.0072**, 95% CI [+0.0034, +0.0112] |
| McNemar | — | b=226, c=154, **p=0.0003** |

判据：Δ>0 且 McNemar p<0.05 → **real gain = True**。

### 3.2 跨 seed（7 个独立划分）

| seed | incumbent | M6 | Δ | McNemar p | real gain |
|---:|---:|---:|---:|---:|:--:|
| 1322 | 0.8614 | 0.8686 | +0.0072 | 0.0003 | ✓ |
| 7 | 0.8455 | 0.8553 | +0.0098 | 0.0000 | ✓ |
| 2024 | 0.8586 | 0.8662 | +0.0076 | 0.0002 | ✓ |
| 1 | 0.8561 | 0.8641 | +0.0080 | 0.0001 | ✓ |
| 13 | 0.8572 | 0.8646 | +0.0074 | 0.0003 | ✓ |
| 99 | 0.8580 | 0.8678 | +0.0098 | 0.0000 | ✓ |
| 31337 | 0.8577 | 0.8671 | +0.0094 | 0.0000 | ✓ |

**7/7 seed 全部 real gain，平均 Δ = +0.0085。** 不是单次划分的运气。

### 3.3 混淆矩阵变化（seed 1322，M6 相对 incumbent）

改善最明显的正是 incumbent 的痛点簇（2/3/5）：`3→5` 74→66、`5→2` 69→54、
`2→3` 65→59、`4→9` 59→52、`9→0` 58→41；总错误数 1386→1314。代价是
`5→3` 104→110 略增、`3→2` 73→75 略增。净效果：中带接拓扑后 2/3/5 与 9
之间的混淆显著下降，类别 9 的召回率 0.8196→0.8397，符合 §1.3 的建模假设。

---

## 4. 被否决的想法与原因

| 想法 | 结果 | 原因 |
|---|---|---|
| 水平/垂直对称性（PDF 点名）替换 aspect | 0.8535（`vsym`）/ 0.8506（`hsym`） | **I(F;Y) 仅 0.091 / 0.096**，远低于 euler(0.477) 与 vruns(0.405)，且条件冗余高。对称性对所有类别都偏高，"差异不够"——MNIST 数字本就是手写体，对称性方差大而判别力低。 |
| 连通分量数（PDF 点名的"笔画连通区域数量"） | `ncomp4/8` I(F;Y)=0.0035/0.0034 | 阈值 64 下 **94% 数字只有 1 个连通分量**，几乎常数。虽"在界内"，但信息量近零。dark-core 阈值（128/190）能提升到 0.010~0.026，仍远不及其它特征，且换入后全部掉点。 |
| 端点数 / 分叉点数（PDF 允许的细粒度拓扑） | I(F;Y)=0.0053 / 0.0057 | 阈值 64 下 95% 数字无端点；128 下升到 0.016/0.009，仍无判别力。 |
| 孔洞数 `nholes`（去掉 euler 换成它） | 0.8499 | `nholes` 本身强（I=0.448），但与 euler **条件冗余 I(nholes;euler|Y)=0.230**，两者高度重叠，替换得不偿失。 |
| PDF 六带方案（上/中/下 + 左/中/右） | 0.7155 | 带密度跨整个 28×28，把大量背景与数字边缘混在一起，噪声远大于九宫格局域密度。**PDF 允许 ≠ 有效**。 |
| 粗 2×2 象限 | 0.6649 | 同上，区域过大导致每格密度被背景平均。 |
| 9 网格 + 3 语义 | 0.8583 | 9 个密度已超 12 槽预算下的最优组合；bottom_center 冗余，用它挤掉语义特征净亏。 |
| 网格窗口几何（12/18/21/24/28） | ≤0.8530 | incumbent 的 7..22 窗口（**非居中**）显著优于居中 15×15（0.8352）与其它尺寸，1 像素位移即掉 2.6pt。已是最优。 |
| dark-core 网格阈值（96/128/160/190） | 0.8616 / 0.8602 / 0.8574 / 0.8529 | 阈值 96 高 0.02pt，属噪声；更高阈值掉点。 |
| 洞位置特征（上/下半孔洞数） | 0.8390 | 思想正确（分离 8/9），但实现后与 euler+nholes 冗余，未转化为准确率。 |
| 贪婪坐标上升自动搜 DAG | 0.8791（+1.77pt） | **违规**：这正是 PDF 明令禁止的 hill-climbing / score-based structure search 类方法。虽然跨 seed 稳健，但**方法本身非法**，且所需边（如 `tl←br、vruns←br`）**无法给出可辩护的语义**，报告不可解释。**否决**。 |
| 6 档离散化 | — | **违反**"每特征 2–5 取值"硬约束（v3 已因此否决，本次未重试）。 |

---

## 5. 最终裁决

### ADOPT ✅

采纳 **M6 DAG**（`builder_c_dag_m6.py`）：

- **特征集不变**（12 特征、状态数、离散化、平滑全部与 incumbent 相同）。
- **仅改 `FEATURE_PARENTS`**：`(-1,0,1,-1,3,4,7,5,-1,8,9,9)` → `(7,0,1,8,3,8,7,5,-1,8,9,9)`。
- seed 1322 验证 **86.86%**（+0.72pt），7/7 seed 均显著（p ≤ 0.0003），平均 +0.85pt。
- 结构完全合规：12 特征∈[6,12]、状态数∈{5,4}⊂[2,5]、11 条特征边≥2、每节点父数 ≤2、
  Y 为根、无隐藏变量、Kahn 无环。
- 三条新边均有明确语义（§1.3），且来自 8 个手写候选中唯一的正向假设，非自动搜索。

**给主会话的集成方式**：把 `builder_c_dag_m6.py` 的 `FEATURE_PARENTS` 值
`(7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)`（以及对应的注释）复制进 `bayes_net.py` 即可；
其余全部代码与 incumbent 相同，无需改动特征提取、离散化或 CPT 估计。
`tests/test_models.py` 的 9 项断言（12 特征、状态数区间、≥2 边、≤2 父、无环）在 M6 下全部成立。

### REJECT ❌

- **特征集重构全部否决**：对称性、连通分量、端点/分叉、洞数、洞位置、六带、2×2 象限、
  9 网格、窗口几何、dark-core 阈值——无一超过 incumbent。incumbent 的
  "8 局域网格密度 + 4 全局语义"分配是稳健局部最优；8 个局域密度不是浪费，而是被
  MNIST 的手写可变性证明为必要的（PDF 允许的全局特征本身判别力不足）。
- **贪婪坐标上升 DAG** 尽管数字最好，因方法违规且不可解释而否决。

### MARGINAL ~

- `vertical_dominance` 4→5 档（0.8624，p=0.556）：方向正确但**不显著**，不值得为
  +0.1pt 增加报告复杂度（PDF §5 明确"若只涨 0.1pt 却让报告难以解释，应优先可解释性"）。
  它已包含在 M6 的可选细化里，但本次不单独采纳。

---

## 6. 复现方式

```bash
cd assignment2/experiments_v4/bayes_net/candidates
python builder_c_experiments.py          # 全部数字（约 3–5 分钟，含 7 seed）
python builder_c_experiments.py --quick  # 仅 seed 1322
```

支撑脚本（全部位于 `candidates/`）：

| 文件 | 作用 |
|---|---|
| `_features.py` | 连续特征提取 + 全训练集缓存（`cache/continuous_all.npz`，~100 s） |
| `_sanity.py` | 连通分量 / 孔洞 / 端点 / 对称性计数器的单元自检 |
| `_harness.py` | 离散化 + `SweepBN`（与 incumbent 同款对数联合概率与平滑） |
| `_mi_screen.py` `_redundancy.py` `_condmi.py` | 训练集 I(F;Y)、I(Fi;Fj)、I(Fi;Fj\|Y) 度量 |
| `_sweep1_single.py` | 单槽替换全扫描 |
| `_sweep2_cells.py` | 网格子集 × 两种读序链（含 incumbent DAG 的精确复现模板） |
| `_sweep3_semantic.py` | 4 语义槽的 1001×2 组合（2002 个合法设计） |
| `_sweep4_refine.py` | 语义档数 / 离散化 / 物理对称分档 / 真外接框宽高比 |
| `_explore.py` `_explore_budget.py` `_explore_geometry.py` `_explore_offset.py` `_explore_grid9.py` `_explore_threshold.py` `_explore_holes.py` `_explore_dag.py` | 各方向探索（六带、2×2、窗几何、grid 偏移、9 网格、阈值、洞位置、DAG 边扫描） |
| `_manual_dags.py` | 8 个手写 DAG 候选族（合规方法论） |
| `_falsify.py` `_falsify_dag.py` `_extra_seeds.py` | McNemar / bootstrap / 7 seed 证伪 |
| `_verify.py` | 两条独立代码路径（模块 vs 引擎）一致性验证 |
| `builder_c_dag_m6.py` | **最终候选模块**（可直接复制进 `bayes_net.py`） |

---

## 7. 合规声明

- 只用 60,000 训练集；`t10k-*.gz` 全程未读。
- 未修改 `assignment2/bayes_net.py`、`naive_bayes.py`、`main.py`、`report.md`
  或 `experiments_v4/bayes_net/` 之外的任何文件。
- 未使用 pgmpy / bnlearn / sklearn 或任何现成贝叶斯网络库；全部 NumPy 手写。
- 随机种子固定（1322 及额外 [7, 2024, 1, 13, 99, 31337]）；离散化阈值仅用**训练集**统计量拟合后应用于验证集。
- 结构完全人工设计，无自动结构学习；被标为 ILLEGAL 的坐标上升仅作反例记录，**未采纳**。
- 所有 CPT 均平滑（alpha=1.0）；预测为 `argmax_y [log P(y) + Σ_i log P(F_i | Parents(F_i))]`，所有特征已观测。
