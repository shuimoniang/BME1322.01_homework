# AGENTS.md

## 仓库概况

BME1322.01《人工智能》课程作业仓库（Python）。每个 assignment 是独立自包含的目录，互不共享代码：

- `assignment0/` — 纯 Markdown 作业（AI 工具使用报告模板），无代码。
- `assignment1/` — 广义五子棋（任意 N≥3、3≤K≤N），纯标准库 + Tkinter，无第三方依赖。
- `assignment2/` — MNIST 手写数字分类（朴素贝叶斯 + 贝叶斯网络），依赖 numpy、matplotlib，全部模型代码手写实现，禁止调用 sklearn/pgmpy 等现成分类器。
- `assignment3/` — 血糖调控强化学习（Part A 计数法模型估计 + 同步价值迭代；Part B 表格 Q-Learning），依赖 numpy、matplotlib，算法全部手写，禁止调用任何现成强化学习/规划库（gymnasium、stable-baselines3 等）。

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

# assignment3：协议自检与单元测试（从仓库根目录）
python sanity_check.py
python -m unittest discover -s assignment3/experiments -p test_v3.py -v

# assignment3：复现全部实验，再重建报告/简答（检查点续跑）
python assignment3/main.py --resume
python assignment3/experiments/write_v3_report.py
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

## assignment3 关键约束

- 提交协议：`model_based.py`（`ModelBasedAgent`）与 `q_learning.py`（`QLearningAgent`）严格按课程接口实现，只从给定经验（状态、动作、奖励、下一状态、done）学习，不读环境内部真实转移表/奖励表，不读 info。Part A 按讲义公式不加终止掩码；Part B 对 done=True 屏蔽未来项——两者对终止的处理不同是有意保留的协议差异，不要"统一"。
- 平局约定：策略提取「严格大于才更新」，保留动作顺序（w,g,i1,i2）中的第一个。
- **实验数据有版本绑定**：`experiments/v3_optimized/manifest.json` 用 SHA-256 锁定 `main.py`、`model_based.py`、`q_learning.py`、`experiments/v3_study.py`、`glucose_env.py`、`sanity_check.py`，改任何一个都会让 `--resume` 拒绝续跑（有意设计，防止混合不同版本的数据）。`write_v3_report.py` 不在锁定清单里，是报告迭代的安全落点。
- **报告由脚本重建**：`report.md`、`analysis.md`、`v3_optimized/` 下的 README/CONSTRAINTS 及 artifact 哈希全部由 `experiments/write_v3_report.py` 生成，数字取自正式 JSON，不手写——要改报告就改生成器，再从仓库根目录重跑。报告中的代码片段必须逐字摘自提交源码。
- 附加消融（builder_a–e 脚本及其 JSON）独立留档，不进 manifest 哈希链；评估协议统一为 2000 条数据（seed=2026）、500 条评估轨迹（环境种子 90000–90499）。
- 结果输出到 `figures/`（fig1–fig9 + results.json）；报告引用的图表都在这里。

## 提交注意事项（各 assignment 通用）

- 课程数据包在 `data/` 下（含各作业 PDF 与官方框架 zip），按作业要求**不入库**，本地保留即可；课程提供的评测配套文件（assignment3 的 `glucose_env.py`、`sanity_check.py`，位于仓库根目录）同样不提交（评测程序自带）。
- 不要提交：`__pycache__/`、`*.pyc`、`__MACOSX/`、实验运行日志（`*.log`）、assignment3 的检查点缓存（`experiments/v3_optimized/checkpoints/`，仅供 `--resume` 续跑）。正式结果 JSON（`figures/results.json` 与 `experiments/v3_optimized/results.json`，各约 7.9M）**必须提交**：`write_v3_report.py` 以其为输入，他人克隆后可不经重跑直接重建报告；完整实验重算则只需课程配套的 `glucose_env.py`/`sanity_check.py`（已在克隆环境验证可跑通）。
- 提交内容以各作业 PDF 的「提交内容与方式」一节为准，通常只含：算法文件、`main.py`、`requirements.txt`、`figures/`、`report.md`（可另附 PDF）、`analysis.md`（简答）。`experiments/` 存证链属于个人留档，可提交可不提交，但其中的脚本与 JSON 保持可对照。

## 其他注意事项

- Windows 环境（Git Bash / PowerShell），路径含中文，shell 命令中用引号包裹路径。
- 仓库无 lint/typecheck 配置；遵循各文件既有的中文注释与文档风格（README/report 均为中文）。
- `data/assignment2/assignment2-data/` 内的 MNIST gzip 为课程分发文件：SHA-256 需与 manifest 一致，不要改动或重新压缩（assignment2 的 `find_data_dir` 会到仓库根目录的 `data/` 下查找）。
