# Critic γ — 合规审计报告（assignment2 v4 集成前置审查）

**角色**：对抗性合规审计员。目标不是提升精度，而是找出严格评分者会扣分的每一条、
以及任何“提交文件加载失败 / 崩溃 / 违反学术诚信”的风险。
**规则来源**：`experiments_v4/CONSTRAINTS.md` + 作业 PDF
《Handwritten Digits Classification (Assignment - 2)》（7 页，原文逐句抽取核对）。
**被审对象**（两个待集成候选）：

| 候选 | 文件 | 家族 | 验证 Acc（seed 1322，仅训练集内部划分） |
|---|---|---|---|
| NB 候选（Builder A） | `naive_bayes/candidates/builder_a_candidate.py` | 特征几何：484 个膨胀二值 4×4 patch | 0.9737（α=0.001） |
| BN 候选（Builder C，M6） | `bayes_net/candidates/builder_c_dag_m6.py` | M6 DAG（仅改 `FEATURE_PARENTS`） | 0.8686（α=1.0） |

**复现脚本**：`experiments_v4/audit/critic_gamma_experiments.py`
（`proto` = 项 1/4/7/8，`deps` = 项 5/6，`tests` = 项 7b，`full` = 项 2，`all` = 全部）。
所有测量只用 60,000 训练图，**从未读取 `t10k-*.gz`**。

审计员未修改任何生产文件；仅写入 `experiments_v4/audit/`。

---

## 项 1 — 协议一致性（PDF §2.1/§2.2、§2.3）

**PDF 固定接口**（原句）：
> “每位同学提交一个名为 `naive_bayes.py` 的文件，其中必须包含一个名为 `NaiveBayesClassifier` 的类，
> 并实现以下接口：`def __init__(self, alpha: float = 1.0)` … `def fit(self, X, y)` … `def predict(self, X)` …
> 此外，文件中可以（可选）实现 `def predict_proba(self, X)`。”
> “提交后，助教将直接从你课程 GitHub 仓库的 `assignment2/naive_bayes.py` 和 `assignment2/bayes_net.py`
> 加载文件…因此**文件名、类名与接口签名不可更改**。”
> “`bayes_net.py` 中还应实现一个可复用的特征提取函数：`def extract_features(X)`”

**实测**（合成小批量 X=(30,784) uint8、y=(30,)）：

| 检查 | NB 候选 | BN 候选 |
|---|---|---|
| 类名 | `NaiveBayesClassifier` ✔ | `BayesNetClassifier` ✔ |
| `fit` 签名 | `(self, X, y)` ✔ | `(self, X, y)` ✔ |
| `fit` 返回 self | `True` ✔ | `True` ✔ |
| `predict` 签名 / 返回 | `(self, X)` / `ndarray int64 (30,)` ✔ | `(self, X)` / `ndarray int64 (30,)` ✔ |
| `predict_proba` | 存在，`float64 (30,10)`，行和=1 ✔ | 存在，`float64 (30,10)`，行和=1 ✔ |
| 空批量 | `predict→(0,)`、`proba→(0,10)` ✔ | 同 ✔ |
| 未 fit 预测 | `RuntimeError` ✔ | `RuntimeError` ✔ |
| 非法 alpha（0/NaN/−1） | `ValueError` ✔ | `ValueError` ✔ |
| 输入校验（错列数 / 像素>255） | 抛 `ValueError` ✔ | 抛 `ValueError` ✔ |
| `extract_features(X)` | n/a（NB 无此要求） | 存在，`int64 (30,12)`，k=12=len(FEATURE_NAMES) ✔ |
| `__init__` 精确签名 | ⚠ `(self, alpha=1.0, dense_slot_limit=20000000)` | ✔ `(self, alpha=1.0)` |

**唯一偏差**：NB 候选多出一个带默认值的 kwarg `dense_slot_limit`。
它在第 157 行被 `self.dense_slot_limit = int(...)` 存下，但**全文再未被读取**
（`grep` 确认：仅第 85、153、157 行出现），是死参数——稀疏存储是唯一使用的路径，
没有切换回密集存储的分支。评分者按文档签名 `NaiveBayesClassifier(alpha=…)` 调用**不会出错**
（额外参数有默认值），但若评分者用 `inspect.signature` 严格比对“接口签名不可更改”，会看到记号不一致。

**判定：RISK（轻微）** — 集成时必须删除 `dense_slot_limit` 以逐字匹配文档签名。
BN 候选：**PASS**。

---

## 项 2 — 评分者调用路径的鲁棒性（不修改 main.py）

**方法**：把两个候选分别以 `sys.modules["naive_bayes"]` / `sys.modules["bayes_net"]` 注入，
再 `import main` —— 验证 `main.py` 无需任何编辑即可拾取候选
（`harness.NaiveBayesClassifier.__module__ == "naive_bayes"`，`harness.FEATURE_PARENTS` = M6 DAG）。
然后在**完整 60,000 张**训练图上 `fit`，在 10,000 行切片上 `predict`。

实测（Windows / NumPy 2.5.4）：

| 指标 | NB 候选（α=0.001） | BN 候选（α=1.0） |
|---|---|---|
| fit wall-clock（60,000） | **7–9 s** | **4.6–5.4 s** |
| predict（10,000） | **2.4 s** | **0.4 s** |
| `tracemalloc` 峰值 | **744 MB** | 1565 MB |
| 进程峰值 RSS | **~870–940 MB**（两次运行） | **~1.6–1.7 GB**（两次运行） |
| 返回 dtype/shape | `int64 (10000,)` ✔ | `int64 (10000,)` ✔ |
| 稀疏 CPT（NB） | 观测键 1,470,735；(10,1470735) 表 = 118 MB | — |
| 若物化密集 CPT（NB） | 10×484×65536×8 B = **2.5 GB（未分配）** | — |
| `check_probability_mass()` | `[484.×10]` ✔（每 patch 条件分布和为 1） | — |
| 崩溃 / 警告 | 无 | 无 |

结论：
- **NB 候选**：稀疏 CPT 方案在内存上完全可行（871 MB 峰值，远低于物化的 2.5 GB），
  `predict` 分批（batch_size=1024）处理 10,000 行无问题；无崩溃、无警告、耗时合理。
  额外用**独立从零重写**的暴力实现对 300 个训练样本逐样本逐特征累加，预测与候选
  **100% 一致**（`agreement = 1.0`）——证明其稀疏/默认值逻辑是精确 NB 求和，不是近似。
- **BN 候选**：无崩溃，时间合理，但峰值 RSS ~1.6 GB 偏高。根因是 `fit` 中
  `self._continuous_features(X_array)` 被**调用两次**（`bayes_net.py:191-192`，M6 同），
  且内部用 float64 构造多份 (60000,28,28) 中间量。这**不是规则违反**，但可诚实指出为工程开销。

**判定：PASS**（两者均可被评分者直接调用；BN 的内存占用为已知工程注记）。

---

## 项 3 — 朴素贝叶斯的合法性（PDF §1.1、§4.1）

**PDF §1.1 原文**：
> “朴素贝叶斯模型使用如下近似：`P(y|X) ∝ P(y) ∏_{i=1}^{784} P(x_i|y)`；
> 预测类别为 `argmax_y P(y) ∏_{i=1}^{784} P(x_i|y)`。”
> **§2.3**：“评测程序会把 28×28 的图像展平成 (n_samples, 784) 的二维数组后传入模型。…
> **你可以在模型内部自行做二值化、归一化、离散化或特征提取，但必须在报告中说明。**”
> **§4.1**：“根据你选择的朴素贝叶斯变体，对像素进行合适的预处理，例如：二值化：`pixel > 0` 或 `pixel > 128`；
> 归一化…；离散化…。自行实现先验概率、条件概率/似然的估计。至少实现一种平滑方式…
> 使用对数概率计算预测结果。不允许直接调用 `sklearn.naive_bayes.*` 或其他现成朴素贝叶斯实现。”

**逐条判定**：

**(a) 是否仍是精确的 NB 分解式 `P(y)·∏_j P(z_j|y)`？** 是。
候选的 `_joint_log_likelihood` 累加 `log P(y) + Σ_{j=1}^{484} log P(z_j|y)`
（`builder_a_candidate.py:211-230`），与 §1.1 的连乘式**逐字同构**，只是把索引从像素 i=1..784
换成 patch 变量 z_j。**每个 patch 恰出现一次**（`scores += N_PATCH_FEATURES * default_log`），
没有重复计入也没有截断——独立暴力实现验证一致即证。
CONSTRAINTS.md 亦明文认定该模式合法：
> “✅ 2×2 / 3×3 patch 内像素联合编码为一个状态、patch 之间条件独立：**合法**（特征被重新定义，NB 分解式不变）。”

**(b) 16-bit patch（膨胀 4×4 二值）是不是“简单离散化特征”？** 是，属“二值化 + 联合编码”。
§4.1 明确列举“二值化：pixel > 0 或 pixel > 128”为合法预处理；候选用 `pixel ≥ 72` 二值化
（阈值在示例区间内），再把 16 个二值像素的位模式**联合编码**为一个 2^16 状态类别变量——
这与现行 2×2 灰度 patch 把 4 个档位联合编码为 625 状态**是同一手法**，只是 patch 更大、更二值。
“膨胀”（内部像素间距=2，感受野 7×7）不改变概率形式：它仍是从原图按固定几何采样出的
一个离散特征，属 §2.3 允许的“特征提取”。

**评分者可能提出的两点反对（须在报告中先发制人）**：
1. **“§1.1 的字面公式是 ∏_{i=1}^{784}，你的下标是 484，不是 784 个像素。”**
   反制：§2.3/§4.1 明确授权预处理与特征提取；现行 729-patch 模型已同理偏离且被接受；
   §5 强调“理解条件独立假设”而非机械照抄下标。**必须在报告写明“特征被重新定义为 patch 变量，分解式不变”。**
2. **“膨胀采样让 patch 内 16 个像素覆盖 7×7 区域，是否把长程相关也吸收进‘一个状态’，从而弱化条件独立？”**
   反制：这恰恰是 patch-NB 的标准释放——patch 内联合、patch 间独立；与 2×2 情形同族。
   候选 `docstring` 已声明该设计，报告需复述并给出验证集消融（builder_a.md 已含 α/几何分解）。

**判定：PASS**。合法，但**报告必须（i）说明特征被重定义为膨胀二值 patch，
（ii）给出 α 与几何的消融分解（净几何增益 +3.4~3.6 pt，平滑 ~+1.0 pt）**，
否则超字面派评分者仍有扣分口实。

---

## 项 4 — 贝叶斯网络结构一致性（PDF §1.2、§4.2）

**PDF §4.2 原文逐条**：
> “从原始图像中提取 **6‒12 个离散特征**。”
> “每个特征变量应有明确的语义，并离散成 **2‒5 个取值**。”
> “节点数：类别节点 + 6‒12 个特征节点；类别节点 Y 应作为**根节点**；
> **至少包含 2 条特征之间的有向边**；每个特征节点的**父节点数量不超过 2**；
> **不引入隐藏变量**；**不使用自动结构学习算法**；不使用 pgmpy、bnlearn 或其他现成贝叶斯网络库。”
> “使用训练数据估计每个节点的 CPT。对所有 CPT 使用平滑处理。”“报告中必须画出贝叶斯网络结构图，并解释每条边的含义。”

**M6 实测（`FEATURE_PARENTS = (7, 0, 1, 8, 3, 8, 7, 5, -1, 8, 9, 9)`）**：

| §4.2 规则 | 实测 | 结果 |
|---|---|---|
| 特征数 6–12 | 12 | ✔ |
| 每特征状态数 2–5 | `[5,5,5,5,5,5,5,5,4,5,4,4]` | ✔ |
| Y 是根节点、是**每个**特征的父 | 代码中 `1 + (parent>=0)` 恒≥1，Y 是全体父 | ✔ |
| ≥2 条特征间有向边 | **11 条**（见下） | ✔ |
| 每特征父节点≤2（**含 Y**） | 无节点 >2 | ✔ |
| 无隐藏变量 | 节点 = Y + 12 观测特征 | ✔ |
| 无环 | Kahn 拓扑排序访问 12/12 | ✔ |
| 手工设计（非自动结构学习） | 见下 | ✔ |
| `v4_common.check_bn_bounds` | `PASS (12 features, 11 feature edges)` | ✔ |

**精确边表（11 条，注意含 Y 的父节点计数）**：
```
br→tl, tl→tc, tc→tr, euler→ml, ml→mc, euler→mr, br→bl, mr→br,
euler→vruns, vruns→aspect, vruns→vdom
```
**无环证明**：对特征-特征子图跑 Kahn。入度零初始集 `{euler}`；
拓扑序 = `euler, vruns, vdom, aspect, mr, br, bl, tl, tc, tr, ml, mc`（12/12 全访问）→ 无环。
（直观：语义链 euler→vruns→{aspect,vdom} 与网格链均为有向链，唯一“回指”边 br→tl 指向链首，
不构成回路。）

**手工设计证据（§8/§4.2 “不使用自动结构学习”）**：
- M6 来自 `candidates/_manual_dags.py` 中 **9 个手写候选族**（M0_control_v3 … M8），
  每个结构带一句书面假设；`_manual_dags.py` 的 `docstring` 明写
  “curated design list … it is NOT an automatic structure search”。
- 候选文件 `docstring` 亦声明：“This is a hand-authored structure, NOT automatic structure learning”。
- **反面对照**：同一 builder 的 `ILLEGAL 坐标上升 DAG`（builder_c.md §2）达 0.8791 但被标注
  **✗ 违规（自动结构搜索）**并弃用——说明该组确实在主动规避自动学习，选择合规项。
- 每节点父数≤2：`ml←{Y,euler}`、`mc←{Y,ml}`、`mr←{Y,euler}`、`br←{Y,mr}`、`tl←{Y,br}` ——均为 2。

**判定：PASS**。M6 逐条满足 §4.2，含“手工设计”这一非机械条款；
唯一需报告落实的是 §4.2 末句“画出结构图并解释每条边”，见项 8（`plot_dag` 布局需修）。

---

## 项 5 — 学术诚信 / 依赖审计（PDF §4.1、§8）

**PDF §8 原文**：
> “本作业须独立完成…**不允许直接复制现成的朴素贝叶斯或贝叶斯网络实现**。
> 不允许使用自动结构学习算法完成贝叶斯网络部分。引用任何外部资料…须在报告中注明出处。”

**实测 import 清单**：

| 文件 | imports | 禁用记号（sklearn/scipy/pgmpy/bnlearn/pomegranate/statsmodels/torch/tf/keras/t10k/urllib/requests/http/subprocess/eval/exec/pip install） |
|---|---|---|
| `builder_a_candidate.py` | `['__future__', 'numpy']` | **NONE** |
| `builder_c_dag_m6.py` | `['__future__', 'numpy']` | **NONE** |
| `naive_bayes.py`（生产） | `['__future__', 'numpy']` | **NONE** |
| `bayes_net.py`（生产） | `['__future__', 'numpy']` | **NONE** |

- **建模只用 NumPy**：两候选全文无第三方分类器/BN 库、无网络访问、无 `pip install`、无子进程。
  `requirements.txt` 仅 `numpy>=1.24` + `matplotlib>=3.7`（matplotlib 仅 `main.py` 绘图用）。
- **原创性**：两候选的 CPT 估计、平滑、对数联合、推理均为手写 NumPy 向量化；
  与生产文件同族（NB 候选复用同一 `_validate_X` 语义；BN 候选特征/离散化与生产**逐字节一致**）。
  未发现任何外部实现痕迹（无 Welford 库式命名、无 sklearn 内部 API 形迹）。
- **报告出处义务**：候选代码本身无外部代码片段，故 §8 的“注明出处”主要落在
  report.md 对 MNIST 数据来源的声明（已具备，report.md:142）。

**判定：PASS**。

---

## 项 6 — 测试集泄漏审计

**约束**（CONSTRAINTS.md:54；PDF §2.3）：“不得读取 `t10k-*.gz`（测试集）。”

**实测**：
- 两个候选文件：**0 处** `t10k` / `load_mnist` / 测试文件引用（`grep` 确认）。
- 生产 `naive_bayes.py` / `bayes_net.py` / `tests/`：**0 处** `t10k`。
- 整个 `experiments_v4/`：`t10k` 仅出现 8 次，**全部是文档/约束文本**——
  `CONSTRAINTS.md:54`（禁止条款）、`builder_c.md:21,250`（“测试集全程未读取”声明），
  以及本审计脚本自身。**无任何代码读取该文件**。
- 协议链路正确：候选脚本依赖 `v4_common.load_train()`（只读 `train-images`/`train-labels`）
  与 `load_split(seed=1322)`（60,000 内部 49,999/10,001 分层划分）。
- 报告的验证数字口径：`builder_a_summary.json.winner.val_accuracy_seed1322 = 0.9737026…`（seed 1322）、
  BN M6 `0.8686`（seed 1322），均由训练内部划分产生；跨 seed 仅用额外 **训练**划分（7/2024），
  仍属训练集内部。测试集未参与任何选择。

**判定：PASS**。

---

## 项 7 — 单元测试兼容性

**现状**：从仓库根 `assignment2/` 运行 `python -m unittest discover -s tests -v` → **9/9 通过**。
（注：AGENTS.md 记载的命令 `-s assignment2/tests` 从仓库根运行会因测试以顶层名
`import bayes_net` 而 `ModuleNotFoundError`——约定实际是 `cd assignment2 && -s tests`。
这是文档/路径注记，非本次候选引入。）

**候选注入后实测**（把候选以生产名注入 `sys.modules`，不改任何文件，跑真实 9 项测试）：

```
Ran 9 tests — errors=1 failures=0
ERROR: test_naive_bayes_probabilities
  File "tests/test_models.py", line 63, in test_naive_bayes_probabilities
    np.exp(model.feature_log_prob_).sum(axis=-1), 1.0
AttributeError: 'NaiveBayesClassifier' object has no attribute 'feature_log_prob_'.
```

**精确破坏点**：`tests/test_models.py:62-64`
```python
np.testing.assert_allclose(
    np.exp(model.feature_log_prob_).sum(axis=-1), 1.0
)
```
NB 候选**不物化** `feature_log_prob_`——密集表 10×484×65536×8 B = 2.5 GB，不可行；
它改存 `observed_keys_` / `observed_log_prob_`（稀疏）+ `default_log_prob_`，
并提供 `check_probability_mass()` 在压缩表示下校验归一化（实测返回 `[484.×10]`，和=1/每 patch）。

**其余 8 项测试对两候选均通过**（含 `test_dag_constraints`、`test_bayes_net_features_and_cpts`、
`test_empty_prediction`、`test_invalid_alpha`、`test_input_validation` 等）。

**最小、忠实的测试适配**（建议，非建模变更）：
```python
# tests/test_models.py::test_naive_bayes_probabilities
model = NaiveBayesClassifier(alpha=0.1).fit(self.X, self.y)
probabilities = model.predict_proba(self.X[:7])
self.assertEqual(probabilities.shape, (7, 10))
np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
self.assertEqual(model.predict(self.X[:7]).dtype, np.int64)
# --- replace the dense-table assertion with the compressed-representation check ---
if hasattr(model, "check_probability_mass"):
    np.testing.assert_allclose(model.check_probability_mass(), N_PATCH_FEATURES)
else:
    np.testing.assert_allclose(np.exp(model.feature_log_prob_).sum(axis=-1), 1.0)
```
即用候选提供的 `check_probability_mass()` 断言“每个 patch 的条件分布和为 1”（等价命题），
并保留 `feature_log_prob_` 分支以兼容旧模型。

> ⚠ **性质声明**：这是**测试断言**的调整（换一种等价方式验证同一概率归一化性质），
> **不是建模变更**，也不放宽任何规则。**必须由主会话在集成前审阅确认**，
> 且应保持“概率质量守恒”这一被测性质不变。

**判定：RISK（可预见、单点）** — 集成 NB 候选必须同步做上述最小测试适配。

---

## 项 8 — 报告/`main.py` 事实与声明核对

当前 `report.md` 描述的是 **v2/v3 模型**；集成两候选后以下事实将变为**假**。
（“present”= 该声明当前在文中存在，因此集成后必须改。）

`report.md` 需更新的行 / 声明（file:line）：

| # | report.md 位置 | 现声明（集成后为假） | 集成后应改为 |
|---|---|---|---|
| 1 | `report.md:1` | 标题“（**v2 迭代版**）” | v4 / 版本号更新 |
| 2 | `report.md:21,24-29` | NB = **729 个 2×2 重叠 patch** | **484 个膨胀二值 4×4 patch**（间距 2、感受野 7×7、`pixel≥72` 二值化、2^16=65536 状态） |
| 3 | `report.md:29,35,126` | patch 状态 **5^4=625**、CPT `10×729×625` | 状态 **2^16=65536**；CPT 稀疏存储（观测键 ~1.47M，非密集） |
| 4 | `report.md:38,48,103` | NB 训练 2.4 s / 预测 0.7 s；**α=0.5**；**92.86%** | α=**0.001**；训练 ~7 s / 预测 ~2.4 s；测试 Acc=**待重测**（验证 0.9737） |
| 5 | `report.md:44-51` | 模型选择表全为 2×2 patch 档位 | 增补 4×4-s2 档位与几何/平滑消融（builder_a.md 数据） |
| 6 | `report.md:71-75` | DAG = **三条列链**、**共 10 条**特征边 | M6：**11 条**边，含 `euler→ml`、`euler→mr`、`tl←br`（列链描述本身也与代码不符，见下） |
| 7 | `report.md:95,97` | BN 验证 **84.92%**（列链 DAG） | M6 验证 **0.8686**；跨 seed 0.8553/0.8662 |
| 8 | `report.md:104,106` | BN 测试 **85.33%**、+2.0 pt | 待重测（验证 0.8686）；相对初版的增量需重算 |
| 9 | `report.md:114` | BN 混淆对 `5→3 183`、`2→3 107`… | M6 新混淆矩阵（`main.py` 重跑后替换） |
| 10 | `report.md:15,51,141` | 迭代数据在 **experiments_v2** | 增补/更新为 **experiments_v4**（或说明 v4 为最终迭代） |
| 11 | `report.md:139` | “**9 项测试**” | 若按项 7 改动测试，需同步说明断言调整 |

**顺带发现（非候选引入，但严格评分者会看出来）**：`report.md:71-73` 把 v3 DAG 描述为
“**三条列链（自上而下）** `top_left→middle_left→bottom_left` …”，并称“**共 10 条**特征间边”。
从生产代码重构的实际 v3 DAG 是 **9 条边、行链**（`tl→tc→tr`、`ml→mc→mr`、`br→bl`、`mr→br`），
**与报告文字不符**。集成时须一并修正（否则报告与 DAG 图/代码三处不自洽）。

`main.py` 必须改动（file:line）：

| # | 位置 | 现内容 | 必须改为 |
|---|---|---|---|
| M1 | `main.py:368` | `"selected_hyperparameters": {"naive_bayes_alpha": 0.5, "bayes_net_alpha": 1.0}` | NB α=**0.001**（否则 metrics.json 报的是 α=0.5 的 0.9540，而非候选宣称的 0.9737）；BN α=1.0 不变 |
| M2 | `main.py:376` | `NaiveBayesClassifier(alpha=0.5)` 硬编码 | `alpha=0.001` |
| M3 | `main.py:172-199,254` | `plot_dag` docstring“final 12-feature **hand-designed DAG (v3)**”、标题“**(v3, 12 features)**”、布局射影假设“每条特征父边向下/向右、无交叉” | 更新为 M6：新增 `euler→ml`、`euler→mr` 会**向左/斜穿**（euler 位于右上 (0.78,0.76)，ml 位于左下 (0.11,0.44)），现有坐标会**边交叉**、图面误导；标题/docstring/坐标须改。实测 `plot_dag(M6)` **不崩溃但仍可渲染**（无断言保护），故属“图错但不报错”的静默风险 |
| M4 | `main.py:254` | 标题字符串 `"Hand-designed Bayesian network (v3, 12 features)"` | 改为 M6/更新版本号 |

**判定：RISK（必然触发）** — report.md 至少 11 处、main.py 4 处必须更新，否则
交付物自相矛盾（报告与代码、图、metrics 不一致），是严格评分者最直接的扣分点。

---

## 汇总：主会话集成前必须改动的清单

**A. 生产模型文件**（`naive_bayes.py` / `bayes_net.py`）
- [ ] A1. 用 `builder_a_candidate.py` 替换 `naive_bayes.py`，**删除 `dense_slot_limit` kwarg**
      （项 1），使 `__init__` 精确为 `(self, alpha: float = 1.0)`。
- [ ] A2. 用 `builder_c_dag_m6.py` 替换 `bayes_net.py`（其 `FEATURE_NAMES/CARDINALITIES/
      extract_features/离散化/平滑` 与生产逐字节一致，仅 `FEATURE_PARENTS` 变化，改动面最小）。
- [ ] A3. 保留 `from __future__ import annotations` 与既有中文注释风格；确认 `predict` 返回 `int64`。

**B. 测试**（`tests/test_models.py`）
- [ ] B1. 按项 7 最小适配 `test_naive_bayes_probabilities`：
      用 `model.check_probability_mass()`（或带 `hasattr` 回退）替换
      `np.exp(model.feature_log_prob_).sum(...)` 断言。**须主会话审阅，声明为测试断言调整、非建模变更。**
- [ ] B2. 测试通过后确认仍是 9 项、全绿。

**C. `main.py`**
- [ ] C1. `selected_hyperparameters` 与 `NaiveBayesClassifier(alpha=…)` 改 α=**0.001**（M1/M2）。
- [ ] C2. `plot_dag` 的坐标布局、docstring、标题适配 M6（M3/M4）；确认新边不交叉、语义标注正确。
- [ ] C3. 重跑 `python main.py --validation-study`，用新混淆矩阵/误分类样例/指标覆盖 `figures/`。

**D. 报告**（`report.md`）
- [ ] D1. 更新项 8 表中 #1–#11 全部事实（2×2→4×4 膨胀二值、625→65536、α、accuracy、
      DAG 11 边与三条新边语义、混淆对、experiments_v4、测试说明）。
- [ ] D2. **新增合法性的先发制人表述**（项 3）：明写“特征被重新定义为膨胀二值 4×4 patch 变量，
      `P(y)·∏_j P(z_j|y)` 分解式不变；patch 内联合、patch 间独立”；给出几何 vs 平滑的消融分解。
- [ ] D3. **修正 v3 DAG 文字描述**（项 8 末）：现文“三条列链 / 10 条边”与代码（行链 / 9 条）不符。
- [ ] D4. 画 M6 结构图（由 C2 的 `plot_dag` 产出）并**逐条解释 11 条边的语义**（§4.2 强制）。

---

## 最终 INTEGRATION VERDICT

| 候选 | 能否原样集成？ | 结论 |
|---|---|---|
| **NB 候选（Builder A，4×4 膨胀二值 patch）** | **否，需 1 处生产改动 + 1 处测试适配** | **条件性 PASS**。协议/合法性/依赖/泄漏全部通过；唯一硬阻塞是 `test_naive_bayes_probabilities` 会因 `feature_log_prob_` 缺失而 ERROR，须做最小等价断言适配；另须删除死参数 `dense_slot_limit`。**模型本身可发布**（60k fit 7 s / 871 MB / predict 10k 无崩溃；独立暴力实现一致）。 |
| **BN 候选（Builder C，M6 DAG）** | **是（代码层面）** | **PASS**。逐条满足 §4.2，`check_bn_bounds` 通过，无环已证，手工设计有据（并主动弃用违规的自动搜索方案）；无测试破坏。**但**主会话若不同步更新 report.md/main.py（尤其 `plot_dag` 与混淆矩阵），交付物将自相矛盾。 |

**两候选均可集成，前提是执行上方 A–D 清单。** 其中**唯一会被评分者立刻察觉的硬失败**
是测试项 7（不修则 `unittest` 直接 ERROR）与报告项 8（不修则报告与代码互相打脸）。
两者都是**最小、机械化**的改动，不触及建模逻辑。

**Top risks（按严重度）**：
1. **测试 ERROR**（项 7，NB）：`feature_log_prob_` 属性缺失 → `test_naive_bayes_probabilities` 必失败。**必修**。
2. **报告/代码不自洽**（项 8）：report.md 11 处 + main.py 4 处过时声明；`plot_dag` 对 M6 静默画错边。
3. **合法性的文字缺口**（项 3）：若报告不解释“特征重定义+膨胀”，超字面派评分者可能质疑 §1.1 的 ∏_{784}。
4. **签名记号偏差**（项 1，NB）：多余 kwarg `dense_slot_limit`（无功能，建议删）。
5. **工程注记**：BN `fit` 重复调用 `_continuous_features` 两次，峰值 RSS ~1.6 GB（非违规，可优化）。
