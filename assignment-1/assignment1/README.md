# 广义五子棋

这是一个支持任意 `N >= 3`、`3 <= K <= N` 配置的图形化广义五子棋程序。界面支持人人、人机和机机对战，并可选择 Baseline Alpha-Beta、Enhanced Alpha-Beta、MCTS + UCT、Hybrid Threat Search 或外部协议 AI。

## 启动与协议

```powershell
python assignment1/main.py
```

项目只使用 Python 标准库和 Tkinter。课程提交入口是 `assignment1/gomoku_ai.py` 中的 `GomokuAI`，构造函数与调用协议保持为：

```python
GomokuAI(player_id, board_size, win_length)
get_move(board, last_opponent_move, time_limit) -> (row, col)
```

Baseline 保留原始固定深度行为。Enhanced、Hybrid 和 MCTS 共用一次扫描全部 `K` 窗口的 `PositionThreatSummary`，按不同完成点区分立即胜、单威胁、双威胁和普通进度，并在每次 `get_move` 内使用有界 Zobrist 局面缓存。

Enhanced 使用迭代加深、标准 exact/lower/upper 置换表和杀棋距离归一化。Hybrid 进一步加入三态威胁证明、Quiescence Search、PVS、Aspiration Window、killer/history 和 LMR。MCTS 使用强制着法优先与渐进扩展。

当前 `GomokuAI` 静态继承 `HybridThreatSearchAI`。选择顺序是协议与 100 题基础战术硬门槛、36 题多步战术正确率、96 局成对对战积分、完成深度、P95。最终复跑中 Hybrid 为 36/36，Enhanced 为 35/36，MCTS 为 170/180，因此在进入成对积分比较前已经选择 Hybrid；运行时不会执行基准。

## 基准与实验

```powershell
# 100 个跨 N/K 的一步攻防题
python assignment1/tactical_benchmark.py --time-limit 0.15

# 36 个多步强制胜、唯一防守和伪威胁题
python assignment1/advanced_tactical_benchmark.py --time-limit 0.3

# 6 个中局的 before/after 搜索效率基准
python assignment1/search_efficiency_benchmark.py --time-limit 0.5 --repetitions 3

# 96 局成对开局对战；存在结果时自动断点续跑
python assignment1/advanced_match_benchmark.py
```

当前效率结果中，Enhanced 与 Hybrid 的冷评估中位耗时分别加速 `6.45x` 和 `6.08x`；两者均有 4/6 个局面提升至少一层，且没有深度下降。96 局实验为 3 个高级策略两两对战、2 种棋盘、2 档时限、每组 4 个固定合法随机开局并交换黑白，全部任务键唯一，0 超时、0 异常、0 非法落子。

最终正式矩阵由独立 runner 生成，严格绑定当前 `gomoku_ai.py` 的 SHA-256；每局写入后刷新并同步磁盘，支持安全续跑：

```powershell
# 正式 1500 局：Hybrid 对 Random、自我对战，以及三种策略对 Enhanced
python assignment1/final_experiments.py --only all --resume

# 只运行一个板块，或从已有 JSONL 重建汇总
python assignment1/final_experiments.py --only random --resume
python assignment1/final_experiments.py --summarize
```

正式矩阵固定为 `(9,4)`、`(15,5)`，每组 `50` 局，时限 `0.5/1.0/5.0s`：Hybrid 对 Random 300 局、Hybrid 自我对战 300 局，以及 Baseline/MCTS/Hybrid 分别对 Enhanced 共 900 局。AI 对 AI 组使用 25 个合法四 ply 中央开局并交换颜色。逐局数据、汇总和版本清单分别位于 `experiments/final/results.jsonl`、`summary.json`、`manifest.json`；优化前正式数据只归档在 `experiments/archive_pre_optimization/`，不混入当前统计。

最终记录包含唯一任务键、源码哈希、开局、胜负、逐步耗时、搜索统计和协议错误字段；`--summarize` 会从 JSONL 重新生成汇总，manifest 不匹配时拒绝追加，避免混合不同版本的实验。

当前版本的基础战术、多步战术、before/after 效率和成对对战结果保存在：

- `experiments/tactical_benchmark.json`
- `experiments/advanced_tactical_benchmark.json`
- `experiments/search_benchmark_before.json`
- `experiments/search_benchmark_after.json`
- `experiments/advanced_match_results.jsonl`
- `experiments/advanced_match_benchmark.json`

四种策略的完整 Random 与自我对战补充矩阵使用相同的两组配置、三档时限和每组 50 局，共 2400 局：

```powershell
python assignment1/comprehensive_evaluation.py --only random --resume
python assignment1/comprehensive_evaluation.py --only selfplay --resume
python assignment1/comprehensive_evaluation.py --summarize
```

其逐局数据、汇总和版本清单位于 `experiments/comprehensive_evaluation/`。当前记录为 2400 个唯一任务、48 个完整实验组，协议失败为 0。

优化前 1800 局仅用于追溯，归档位置为 `experiments/archive_pre_optimization/`。

## 测试

```powershell
python -m unittest discover -s assignment1/tests -v
python assignment-1-code/arena.py assignment1/gomoku_ai.py assignment-1-code/random_ai.py --board 15 --win 5 --time 5 --games 2 --quiet
```
