# AGENTS.md

## 仓库概况

BME1322.01《人工智能》课程作业仓库（Python）。每个 assignment 是独立自包含的目录，互不共享代码：

- `assignment0/` — 纯 Markdown 作业（AI 工具使用报告模板），无代码。
- `assignment1/` — 广义五子棋（任意 N≥3、3≤K≤N），纯标准库 + Tkinter，无第三方依赖。
- `assignment2/` — MNIST 手写数字分类（朴素贝叶斯 + 贝叶斯网络），依赖 numpy、matplotlib，全部模型代码手写实现，禁止调用 sklearn/pgmpy 等现成分类器。

报告写在各目录的 `report.md`（另有导出的 PDF）。作业原始要求是对应目录里的 `*.pdf`。

## 常用命令

```powershell
# assignment1：运行 GUI（从仓库根目录）
python assignment1/main.py

# assignment1：单元测试（tests 内部已把 assignment1 加入 sys.path，从根目录运行）
python -m unittest discover -s assignment1/tests -v

# assignment1：课程 arena 对战验收
python assignment1/assignment-1-code/arena.py assignment1/gomoku_ai.py assignment1/assignment-1-code/random_ai.py --board 15 --win 5 --time 5 --games 2 --quiet

# assignment2：完整实验（写 figures/metrics.json 与图）
cd assignment2 && python main.py --validation-study

# assignment2：单元测试
python -m unittest discover -s assignment2/tests -v
```

## assignment1 关键约束

- **提交协议不可破坏**：课程提交入口是 `assignment1/gomoku_ai.py` 中的 `GomokuAI`，必须保持 `GomokuAI(player_id, board_size, win_length)` 和 `get_move(board, last_opponent_move, time_limit) -> (row, col)`。当前 `GomokuAI` 静态继承 `HybridThreatSearchAI`。
- 四种 AI（Baseline / Enhanced / MCTS / Hybrid）都在 `gomoku_ai.py` 中；Enhanced、Hybrid、MCTS 共用 `PositionThreatSummary`（一次扫描全部 K 窗口）和有界 Zobrist 缓存。改搜索逻辑时保持这些共享结构的一致性。
- `main.py` / `gui.py` 是本地 GUI 对战外壳；`assignment-1-code/` 是课程提供的官方框架（含 arena），不要改动其中课程代码。
- 基准脚本：`tactical_benchmark.py`、`advanced_tactical_benchmark.py`、`search_efficiency_benchmark.py`、`advanced_match_benchmark.py`；正式实验用 `final_experiments.py --only all --resume`，汇总重建用 `--summarize`。
- **实验数据有版本绑定**：`final_experiments.py` 会校验 `gomoku_ai.py` 的 SHA-256（manifest），代码改动后旧 JSONL 会被拒绝追加——这是有意设计，防止混合不同版本的实验数据。优化前数据归档在 `experiments/archive_pre_optimization/`，不要与当前统计混用。

## assignment2 关键约束

- 入口 `main.py`，模型在 `naive_bayes.py` / `bayes_net.py`；数据在 `assignment2-data/`（嵌套同名目录，`find_data_dir` 自动查找，四个 MNIST gzip 的 SHA-256 需与课程 manifest 一致，不要改动或重新压缩）。
- 固定随机种子 `1322`；模型选择只用训练集内部分层划分（49,999/10,001），测试集不参与任何选择——修改实验流程时必须保持这一点。
- 结果输出到 `figures/`（metrics.json + 图）。

## 其他注意事项

- Windows 环境（Git Bash / PowerShell），路径含中文，shell 命令中用引号包裹路径。
- 仓库无 lint/typecheck 配置；遵循各文件既有的中文注释与文档风格（README/report 均为中文）。
- `__pycache__/`、`__MACOSX/`、`assignment2-data.zip` 属于杂物/压缩包，无需处理也不应提交。
