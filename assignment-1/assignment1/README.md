# 广义五子棋

这是一个支持任意 `N >= 3`、`3 <= K <= N` 配置的图形化广义五子棋程序。

## 启动

```powershell
python assignment1/main.py
```

程序使用 Python 标准库和 Tkinter，不需要安装第三方依赖。界面支持人人、人机和机机对战，并可选择基础 Alpha-Beta、增强 Alpha-Beta、MCTS + UCT 或外部协议 AI。

## 课程协议

提交入口为 `assignment1/gomoku_ai.py` 中的 `GomokuAI`。默认策略是带迭代加深、着法排序、置换表和强化棋型评估的 Alpha-Beta。

基础 Alpha-Beta 与 MCTS 的裁判适配入口分别为 `baseline_ai.py` 和 `mcts_ai.py`。

## 测试

```powershell
python -m unittest discover -s assignment1/tests -v
python assignment-1-code/arena.py assignment1/gomoku_ai.py assignment-1-code/random_ai.py --board 15 --win 5
```
