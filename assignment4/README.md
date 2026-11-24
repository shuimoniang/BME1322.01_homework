# Assignment 4 复现说明

安装依赖后，从仓库根目录运行：

```powershell
python -m venv assignment4/.venv
assignment4/.venv/Scripts/python.exe -m pip install -r assignment4/requirements.txt
assignment4/.venv/Scripts/python.exe assignment4/main.py
assignment4/.venv/Scripts/python.exe -m unittest discover -s assignment4 -p test_protocol.py -v
assignment4/.venv/Scripts/python.exe assignment4/verify_artifacts.py
```

正式结果使用 Python 3.14.7，依赖版本列于 `requirements.txt`。其他依赖版本可能改变少数训练指标的末位数。课程数据默认从 `data/assignment4-data/` 读取，支持解压后多一层同名目录，也可用 `--data-dir` 指定。CSV、固定划分和课程基线脚本在本地保留，不随作业提交。

`main.py` 执行固定五折扫描、外层选参及四项消融，生成结果表 `figures/tables.md`、图、逐折结果和折外预测。指定 `--output-dir` 可将重跑产物写到独立目录。`tables.md` 为运行产物，首次核验前须先运行 `main.py`。报告、简答、补充发现和 AI 记录作为提交文稿保留。

- [实验报告](report.md)涵盖结果、消融及实验日志。
- [讨论题](analysis.md)含三题简答和阈值对照。
- [补充实验发现](additional_findings.md)提供补充证据；[AI 错误记录索引](ai_error_log.md)指向报告中的纠错经过。
- `figures/results.json` 保存正式结果、版本和课程数据哈希。`figures/predictions.csv` 保存逐行折外预测，`row` 为原 CSV 的零基数据行号。
