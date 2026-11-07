# 第三版实验归档

入口：仓库根目录 `python assignment3/main.py --resume`；报告：`python assignment3/experiments/write_v3_report.py`。
首次在没有检查点的目录运行可省略 --resume。源码或配置不一致会拒绝复用，不能直接修改manifest绕过检查。

- results.json：课程必做实验、全部验证候选和独立测试逐轨迹统计。
- selection.json：接触测试数据前冻结的候选及各失败候选原因。
- manifest.json：实验配置及计算源码SHA-256；源码或配置变化必须开新目录。
- checkpoints/：逐候选原始回合记录、访问次数、Q表与评估计数；原子写入，支持续跑。
- DECISION.md：各比较的验收结果；figures/：完整图表与结果速览。
- snapshot/：代码、报告、简答与依赖快照；artifact_sha256.json核对这些交付文件。
- v2_reproduction/：旧版独立复现产物及精确一致核对结果。

当前输出与v1/v2历史归档分开。v2完整快照补存于其snapshot/，已有文件未覆盖。
主任务仍是规定的模型估计+同步VI、原序单遍表格Q-Learning，接口与默认参数不变。
在线调度仅在驱动层，保持600回合；学习过程中不读环境内部状态、不重放经验、不改变奖励。
冻结策略与训练期表现独立比较；安全区间容许退步时不声称安全不退步已被证明。
