# 贝叶斯网络 v3 最终测试结果

模型：12 特征手工 DAG（v3 行链结构）。特征与 v2 完全相同（8 个中央 15×15 网格密度，bottom_center 省略 + euler + vertical_runs + aspect_ratio + vertical_dominance），改变的只有 DAG 的特征间边：

- **行链**（每行内自左向右）：`top_left→top_center→top_right`、`middle_left→middle_center→middle_right`、`bottom_right→middle_right`、`bottom_left←bottom_right`（底行闭合）
- **语义链**：`euler_topology→vertical_runs`，`vertical_runs→aspect_ratio`，`vertical_runs→vertical_dominance`

类别 Y 仍是所有节点的根父节点；每个特征至多 2 个父节点；8 条特征间边；无隐藏变量；结构为人工设计（在人工候选间用验证集选择）。CPT 仍用向类条件边缘 backoff 的平滑，alpha=1.0。

选择过程仅使用训练集内部分层划分（seed 1322，49,999/10,001），验证准确率 86.14%。以下为定稿后在完整测试集（10,000 张，唯一一次使用）上的结果：

| 指标 | 数值 |
|:---|---:|
| Accuracy | **86.27%** |
| Macro Precision | 86.20% |
| Macro Recall | 86.05% |
| Macro F1 | **86.08%** |
| fit 耗时 | 3.1 s（60,000 张） |
| predict 耗时 | 0.36 s（10,000 张） |

## 各类 F1

| 类别 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|:--|---|---|---|---|---|---|---|---|---|---|
| F1 | .903 | .968 | .760 | .796 | .900 | .755 | .918 | .898 | .882 | .829 |

## 最易混淆对

| 真实 → 预测 | 数量 |
|:---|---:|
| 5 → 3 | 112 |
| 2 → 3 | 105 |
| 3 → 5 | 76 |
| 5 → 2 | 67 |
| 8 → 9 | 61 |

## DAG 特征间边（语义解释）

- 每行内自左向右的行链刻画"同一水平带内墨迹从起笔区向收笔区转移"的相关性（如 3、5、2 的右侧钩与其左侧中部结构联动）。
- 底行闭合 `bottom_right→bottom_left`：底部左右两格常由同一横向底笔画（如 2 的横）一次性生成。
- `euler→vertical_runs→{aspect, vdom}`：孔洞/连通分量数决定每列分离墨迹段数，并进一步影响宽高比与竖直梯度占优程度——三者都依赖于环结构这一共同原因，因此把它们接成一条链（vruns 作中心）优于 v2 的两条独立边。

## 与历史版本对比

| 版本 | 验证 Acc (seed 1322) | 测试 Acc | 测试 Macro F1 |
|:---|---:|---:|---:|
| v1 基线（archive/baseline_v1） | 83.02% | 83.33% | 82.95% |
| v2 最终（列链 DAG） | 84.92% | 85.33% | 85.05% |
| **v3 最终（行链 + 语义链 DAG）** | **86.14%** | **86.27%** | **86.08%** |

对比 v2 测试 85.33%：**+0.94 个百分点**；对比 v1 基线：+2.94 个百分点。

5→3 混淆从 183 降到 112，2→3 从 107 降到 105，3→5 从 69 升到 76（基本持平），2/3/5 类 F1 分别提升 0.022 / 0.024 / 0.034。

## 复现

```powershell
cd experiments_v3/bayes_net && python final_eval.py   # 写出 final_test_result.json
```
