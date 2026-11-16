# Assignment 4 复现说明

## 环境

已验证 Python 3.14.7；直接依赖版本见 requirements.txt。推荐使用独立虚拟环境：

```powershell
python -m venv assignment4/.venv
assignment4/.venv/Scripts/python.exe -m pip install -r assignment4/requirements.txt
assignment4/.venv/Scripts/python.exe assignment4/main.py
assignment4/.venv/Scripts/python.exe -m unittest discover -s assignment4 -p test_protocol.py -v
assignment4/.venv/Scripts/python.exe assignment4/verify_artifacts.py
```

本地已有虚拟环境沿用系统提供的 numpy/matplotlib，其他依赖安装于作业虚拟环境；版本已锁定，以上普通独立环境安装方式同样可复现。

课程数据默认位于仓库根目录 `data/assignment4-data/`，也可用 `--data-dir` 指定。CSV、固定折和课程参考脚本保持原样，不随作业提交。

`main.py` 一键重新训练、扫描、独立选参、消融并生成结果与全部文稿。`main.py --report-only` 从 `figures/results.json` 重建图表与报告；树结构和标准化机制验证会重新拟合相应解释模型。完整运行包含嵌套选参与未标准化SVM扫描，在本机约数分钟，明显长于课程的最小基线。

## 阅读顺序

- report.md：完整实验报告，主表与详细扫描、消融、独立选参、机制核查。
- analysis.md：三题简答，各七句及阈值敏感性表。
- human_review.md：本人已逐项复核并确认没有问题的记录。
- experiment_log.md：反直觉发现及验证；错误纠正记录统一见报告末尾声明。
- completion_checklist.md：题目要求到实现与证据的映射。
- figures/results.json：原始逐折结果、候选、预测、系数、置换重要性、版本及数据哈希。
- figures/predictions.csv：各主模型扫描最优的逐行折外预测，row为原CSV零基数据行号。

结果可能与题目示例数字略有差异，应以相同环境下原始课程基线的复现核对为准。独立选参检查与课程五折扫描分别报告，不能相互替代。
