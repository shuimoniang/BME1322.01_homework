# Baseline v1 留档（2026-10-14 完成）

迭代优化前的原始版本，完整测试集结果（10,000 张，未参与任何选择）：

- Naive Bayes（7 档灰度，alpha=0.1）：Accuracy 84.97%，Macro F1 84.79%
- Bayesian Network（12 特征九宫格 DAG，alpha=0.5）：Accuracy 83.33%，Macro F1 82.95%

验证集（训练集内部 49,999/10,001，seed 1322）：
- NB alpha 0.1/0.5/1.0 → 84.09% / 84.02% / 84.00%
- BN alpha 0.1/0.5/1.0 → 83.09% / 83.02% / 83.01%

包含：report.md、main.py、两个模型源码、figures/（混淆矩阵、DAG 图、误分类样例、metrics.json）。
