# 贝叶斯网络迭代日志（v2）

评测协议：训练集 60,000 张内部固定分层划分（seed 1322，49,999 训练 / 10,001 验证），报告验证准确率。测试集（10,000 张）全程未参与选择，仅在最终定稿后评估一次。

基线（v1，`archive/baseline_v1/`）：验证 83.02%（alpha=0.5），测试 83.33%。

每轮实验脚本在本目录（`round0_baseline.py` … `round9_final_polish.py`），公共数据管线在 `common.py`（连续特征统计缓存在 `cache/`）。

## Round 0 — 基线复现
| 配置 | 验证 Accuracy |
|:---|---:|
| v1 协议路径 alpha=0.1 / 0.5 / 1.0 | 83.09% / 83.02% / 83.01% |

## Round 1 — 特征族探索
| 特征族 | 验证 Accuracy |
|:---|---:|
| 纯 3×3 密度网格（链式 DAG） | 74.0% |
| 全局语义特征（简单 DAG） | 54–60% |
| 密度网格 + 拓扑/笔画特征（13 维） | 80.5% |

结论：密度网格是主体，语义特征只能作为补充。

## Round 2 — 离散化
| 方案 | 验证 Accuracy |
|:---|---:|
| 网格 + euler + vruns，零值分离（状态 0 = 恰好零，正值按训练分位分档） | **83.33%** |
| 同上，普通分位切分 | 83.32% |
| 4 档 | 81.7% |

## Round 3 — DAG 与平滑
| 方案 | 验证 Accuracy |
|:---|---:|
| 基线 DAG + 稀疏回退平滑（CPT 向类条件边缘 backoff） | 83.40% |
| 加入 aspect(4)（宽高比） | 83.96% |
| 加入 vert_dominance(4)（竖直梯度占优） | **84.02%** |
| 加入 horiz_runs / top_heavy / hsym | 83.3–84.0% |

## Round 4 — 特征两两组合
| 组合 | 验证 Accuracy |
|:---|---:|
| grid10 + vdom + aspect（class-only 父） | **84.69%** |
| grid10 + aspect + horiz_runs | 84.66% |
| 其他组合 | ≤ 84.4% |

## Round 5 — 用语义特征替换网格密度（失败）
替换后全部掉到 74–79%，确认网格密度不可删。

## Round 6 — 12 特征塞入尝试
在 12 特征硬上限内保留全部候选均劣于 Round 4（79.7–83.3%）。

## Round 7 — 删除单个网格密度（13 特征探索）
| 删除项 | 验证 Accuracy |
|:---|---:|
| 删除 global_ink（9 网格 + 4 语义） | **84.97%** |
| 删除其他密度 | 81.7–84.4% |

全局墨迹与九宫格高度冗余，删除后腾出一个特征位。但 13 特征超出作业 6–12 上限，需在 12 特征内定稿。

## Round 8 — 精修（13 特征）
列反转/列链对比、语义特征接线（aspect←vruns、vdom←aspect）等，84.8–85.0%。

## Round 9 — 终局组合（13 特征）
最优 85.10%（列反转链 + vruns←euler、vdom←aspect + backoff），跨 seed 83.6–85.1%。同样因 13 特征超限不可提交。

## 合法 12 特征定稿（修正 wire 后）

约束：作业 PDF 要求 6–12 个离散特征。在 12 特征内系统比较：

| 方案 | 验证 Accuracy |
|:---|---:|
| 9 网格 + euler + vruns + vdom（去 aspect），vruns←euler | 84.47% |
| 9 网格 + euler + vruns + aspect（去 vdom） | ≤ 83.8% |
| **8 网格（删 bottom_center）+ euler + vruns + aspect + vdom，vruns←euler、vdom←aspect，backoff** | **84.92%** |

bottom_center 是与其他密度最冗余的格子（Round 7 中删除它的代价最小）。

最终 DAG（12 特征，类别 Y 是所有节点的父节点，每特征至多 2 父）：
- 列链（自上而下）：`top_left→middle_left→bottom_left`、`top_center→middle_center`、`top_right→middle_right→bottom_right`
- 语义边：`euler_topology→vertical_runs`、`vertical_dominance→aspect_ratio`

跨 seed 稳健性（同一配置，不同划分）：seed 1322/7/2024 → 84.92% / 83.51% / 84.64%，均显著高于基线。

**最终测试集（alpha=1.0）：Accuracy 85.33%，Macro F1 85.05%**（详见 `final_test_result.md`；对比基线测试 83.33%，+2.0 个百分点）。

## 备注：实验脚本与定稿的一处偏差

Round 8/9 脚本中 `above_of` 有一处索引偏移，导致当时评估的"列反转链"实为接错的 DAG（84.71%）。定稿实现使用语义正确的列链后验证升至 84.92%，因此采纳正确接线。本文档保留全部原始实验输出以供追溯。
