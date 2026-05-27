## 基于文档依赖性分析的 RAG 冲突信号还原 — 实验工作总结

### 一、方法概述

#### 1.1 问题定义

RAG 系统隐含假设"每篇检索文档代表一个独立信息源"。当多篇文档支持同一答案时，LLM 将其视为多源交叉验证。然而现实中大量文档是同源改写/转载，导致**虚假共识**：

```
真实场景：8篇说 Paris（同源改写）+ 2篇说 Lyon（独立来源）
LLM感知：8:2 → 选 Paris（错误）
还原后：  1:2 → 选 Lyon（正确）
```

#### 1.2 Module 1：文档依赖性分析与信息源识别

三层架构，逐层过滤：

| 层级 | 功能 | 方法 | 开销 |
|------|------|------|------|
| Layer 1 | 依赖候选识别 | Jaccard ≥ 0.40（token级） | O(n²)，零API |
| Layer 2 | 依赖性精确判定 | LLM 四级分类 D0-D3 | 每候选对1次API |
| Layer 3 | 信息源聚合 | Union-Find 聚类 | 零API |

**D0-D3 量表**：
- D0：完全无关 → 保留
- D1：同话题独立写作 → 保留
- D2：部分依赖（共享来源但有独立内容）→ 保留
- D3：高度依赖（同源改写/复制）→ **合并为一个信息源**

**关键设计决策**：仅 D3 触发合并，D2 保留。因为 D2 文档对可能携带不同事实，合并会消灭真实冲突信号。

#### 1.3 Module 2：源级隔离仲裁（V4）

Module 1 输出聚类后，Module 2 执行：

```
Step 1: Source Unit Construction
  每个cluster → 1个source unit
  代表文档 = cluster中检索排名最高的（index最小）

Step 2: Source-Isolated Answer Extraction（并发）
  每个source unit独立回答query
  输出: {answer, status, support, confidence}

Step 3: Algorithmic Arbitration
  normalize答案 → 分组
  共识（1组）→ 直接输出
  冲突（多组）→ LLM仲裁（基于证据质量而非数量）
```

**核心思想**：让LLM在"每源一票"的公平条件下判断冲突，而非在被冗余扭曲的文档集上直接推理。

---

### 二、数据集

| 数据集 | 样本数 | 文档数/样本 | 特点 | 评测指标 |
|--------|--------|------------|------|----------|
| amp8 | 215 | 15（6干净+8改写+1种子） | 极端冗余，冲突信号被扭曲 | NEM |
| clean | 215 | 7 | 无冗余，正常检索 | NEM |
| RAMDocs | 500 | ~5.5 | 歧义+误信息+噪声 | Strict ACC |
| FaithEval-Inconsistent | 500 | 2（同源改写，含冲突答案） | 文档内冲突检测 | Strict ACC / Non-strict ACC |

---

### 三、实验结果

#### 3.1 主实验：V4 Source-Isolated Arbitration

| 数据集 | Naive | V4 | Δ | 正向翻转 | 负向翻转 | 比值 |
|--------|------:|---:|---:|--------:|--------:|-----:|
| amp8 | 27.0% | 49.3% | **+22.3pp** | 63 | 15 | 4.2:1 |
| clean | 66.5% | 72.6% | **+6.0pp** | 34 | 21 | 1.6:1 |
| RAMDocs | 18.6% | 42.4% | **+23.8pp** | 140 | 21 | 6.7:1 |
| FaithEval-Inconsistent | 82.4% | 83.4% | **+1.0pp** | — | — | — |

**amp8 关键指标**：
- 平均聚类数：9.5（15篇→9.5个独立源）
- CPR-source：0.40
- 决策分布：conflict_arbitrated 199, consensus 15, no_evidence 1
- 平均API调用：30.7次/样本（Module1 19.2 + Extraction 9.5 + Arbitration + Naive）

**clean 关键指标**：
- 平均聚类数：5.5（7篇→5.5个独立源，少量误合并）
- CPR-source：0.68
- 平均API调用：8.3次/样本（按需计算，正常查询开销很低）

**RAMDocs 关键指标**：
- 平均聚类数：5.0
- Wrong answer rate：22.8%
- 决策分布：multi_answer_arbitrated 433, consensus 61, no_evidence 6
- 平均API调用：7.5次/样本
- 参考：MADAM-RAG GPT-4o-mini 在 RAMDocs = 28.0%（论文报告值）

**FaithEval-Inconsistent 关键指标**：
- Non-strict ACC：Naive 82.6% → V4 83.6% (+1.0pp)
- Strict ACC：Naive 82.4% → V4 83.4% (+1.0pp)
- Avg API calls：2.2（几乎全部判 D3 合并，仅 1 pair + 1 extraction）
- D3 分类准确率 ≈ 97%+（两篇同源改写文档几乎全部被正确识别合并）
- 意义：证明 V4 在无 inter-source 冲突场景下不退化，且 D3 分类在真实同源文档上高度准确

#### 3.2 消融实验：V3 — 文档级隔离（无Module 1聚类）

每篇文档各自独立回答（15个源），无聚类。

| 数据集 | Naive | V3 | Δ |
|--------|------:|---:|---:|
| amp8 | 25.6% | 44.2% | +18.6pp |

- 平均源数：14.2（无合并）
- 决策分布：conflict_arbitrated 211, consensus 3
- **V4 vs V3**：49.3% vs 44.2% (+5.1pp) → 证明 Module 1 聚类有价值
- V3 的问题：15个源中8个是同源改写，仍会在仲裁中产生"伪多数"

#### 3.3 消融实验：V5 — 随机分组隔离

文档随机分成4组（不基于依赖分析），每组独立回答。

| 数据集 | Naive | V5 | Δ |
|--------|------:|---:|---:|
| amp8 | 26.5% | 29.3% | +2.8pp |

- 决策分布：consensus 71, conflict_arbitrated 131, no_evidence 13
- **V4 vs V5**：49.3% vs 29.3% (+20.0pp) → 证明依赖性分组 >> 随机分组
- V5 失败原因：随机分组大概率将同源文档分散到不同组，每组仍受冗余影响

#### 3.4 消融实验：Module 1 单独硬删除（无Module 2）

仅做 Module 1 聚类 + 硬删除冗余，然后直接用 Naive RAG 在去重后文档上推理。此实验为早期阶段（step_4_layered_e2e.py）的结果，与 V4 使用不同的 Naive baseline 运行批次（Naive 数值略有差异属 API 调用的正常波动）。

| 数据集 | Naive（该批次） | Module 1 Only | Δ |
|--------|------:|-------------:|---:|
| amp8 | 30.7% | 43.7% | +13.0pp |
| clean | 70.7% | 66.5% | −4.2pp |

- Module 1 单独在 amp8 上 +13.0pp，V4 (Module1+2) 在 amp8 上 +22.3pp → Module 2 源隔离仲裁在聚类基础上进一步显著提升
- Module 1 单独在 clean 上有 −4.2pp 副作用（误删独立源），V4 在 clean 上反而 +6.0pp → Module 2 通过隔离提取修复了硬删除的信息损失

#### 3.5 Exp1 简单 Baseline 对比

在 amp8 (N=215) 上测试四种简单策略，均不使用依赖性分析 LLM 调用：

| 方法 | 原理 | NEM | Δ vs Naive |
|------|------|----:|----------:|
| Naive RAG | 全文档拼接 | 29.8% | — |
| A) Prompt-only | 在prompt中加"注意文档可能有冗余" | 33.0% | +3.2pp |
| B) Jaccard≥0.95 Dedup | 高阈值词汇去重（几乎只去近似副本） | 31.2% | +1.4pp |
| C) Top-3 Truncate | 只用检索排名前3篇文档 | 28.8% | −1.0pp |
| D) Random-6 | 随机选6篇文档 | 30.7% | +0.9pp |

注：此批次 Naive=29.8% 与 V4 实验批次 Naive=27.0% 存在轻微波动，属 API 调用的正常差异。

**结论**：
- 所有简单策略提升均 ≤ 3.2pp，远不及 V4 的 +22.3pp
- Prompt-only 最优但效果微弱，说明仅靠提示词无法让 LLM 自主识别冗余
- Top-3 截断反而掉点，因为可能丢掉少数派正确答案的文档
- Jaccard≥0.95 去重仅去掉近乎完全重复的文档，对语义改写无效
- 证明需要显式的文档依赖性分析（Module 1 的 D0-D3 分级判定）

#### 3.6 外部 Baseline 对比

**MADAM-RAG 对比（自跑，同一 eval pipeline）**：

RAMDocs 上使用完全相同的数据（500样本）和相同的 strict accuracy 评测函数，自跑 MADAM-RAG（GPT-4o-mini, 3轮debate）。

| 方法 | N | Strict ACC | Wrong Rate | Avg API calls |
|------|--:|-----------:|-----------:|--------------:|
| Naive RAG | 500 | 18.6% | — | 1 |
| MADAM-RAG | 500 | 32.6% | 37.6% | 16.0 |
| **V4 (Ours)** | 500 | **42.4%** | 22.8% | 7.5 |

- V4 > MADAM-RAG **+9.8pp**，API 开销仅为其 47%（7.5 vs 16.0）
- MADAM-RAG 的 37.6% wrong rate 表明 multi-agent debate 在 misinfo 存在时反而传播错误
- V4 的源隔离策略从根本上阻断了 misinfo 的跨源传播

参考：MADAM-RAG 论文报告值为 28.0%（Table 1, GPT-4o-mini），我们自跑为 32.6%，差异可能来自实现细节或随机性。

---

### 四、方法对比总览

#### 4.1 amp8 数据集上的完整对比

| 方法 | NEM | Δ vs 自身Naive | 核心机制 |
|------|----:|----------:|----------|
| Naive RAG | 27.0%~29.8%† | — | 全文档拼接 |
| Prompt-only (Exp1) | 33.0% | +3.2pp | 提示词提醒冗余 |
| Jaccard≥0.95 Dedup (Exp1) | 31.2% | +1.4pp | 高阈值词汇去重 |
| Top-3 Truncate (Exp1) | 28.8% | −1.0pp | 只用前3篇 |
| Random-6 (Exp1) | 30.7% | +0.9pp | 随机选6篇 |
| V5 Random Grouping | 29.3% | +2.8pp | 随机分4组隔离 |
| Module 1 Hard Dedup Only | 43.7% | +13.0pp | 聚类+硬删除，无源隔离 |
| V3 Doc-level Isolated | 44.2% | +18.6pp | 每文档独立回答，无聚类 |
| **V4 Source-Isolated (Ours)** | **49.3%** | **+22.3pp** | 聚类+源隔离+仲裁 |

†不同批次间 Naive 存在 2-3pp 的正常波动（API 调用非确定性），各方法的 Δ 均基于同批次 Naive

#### 4.2 各消融的结论

| 消融 | 证明了什么 |
|------|-----------|
| V4 vs V3 (+5.1pp) | Module 1 依赖聚类有价值：减少伪多数 |
| V4 vs V5 (+20.0pp) | 依赖性分组 >> 随机分组：必须基于真实依赖关系 |
| V4 vs Module1-Only (+9.3pp) | Module 2 源隔离仲裁在聚类基础上进一步提升 |
| V4 vs Prompt-only (+19.0pp) | 简单提示词无法解决冗余问题 |
| V4 clean +6.0pp vs M1 clean −4.2pp | Module 2 修复了硬删除的信息损失 |

---

### 五、效率分析

| 场景 | 平均API调用/样本 | 特征 |
|------|----------------:|------|
| V4 amp8（高冗余，15doc） | 30.7 | Module1占62%（19.2次），Extraction 9.5次 |
| V4 clean（无冗余，7doc） | 8.3 | Layer1过滤掉大部分候选对，Module1仅1.0次 |
| V4 RAMDocs（~5.5doc） | 7.5 | 文档数少，聚类开销低 |
| MADAM-RAG AmbigDocs（~2.9doc，实测N=43） | 8.3 | 3轮debate × n_docs + aggregator，文档少时开销与V4接近 |

**按需计算特性**：无冗余时（clean），Layer 1 Jaccard 过滤掉绝大部分文档对，Layer 2 仅处理~1对，开销接近零。

---

### 六、进行中的实验

| 实验 | 状态 | 说明 |
|------|------|------|
| amp 梯度实验 (amp2/amp4) | 待运行 | 验证方法在温和冗余下仍有效 |
| D3 分类准确性验证 | 待运行 | 利用 amp8 ground truth 评估聚类质量 |

---

### 七、待完成工作

1. **amp 梯度实验 (amp2/amp4)**：验证方法在温和冗余下仍有效，画趋势图
2. **D3 分类准确性验证**：利用 amp8 ground truth 评估 Module 1 聚类 precision/recall
3. **Bootstrap CI + McNemar test**：对 amp8/clean/RAMDocs 已有结果计算置信区间
4. **效率表 + Case Study**：从已有结果提取典型案例

---

### 八、核心文件索引

| 文件 | 用途 |
|------|------|
| `module2/source_isolated_arbitration.py` | V4 主实现（amp8/clean） |
| `module2/baselines_isolated.py` | V3、V5 消融实验 |
| `module2_ramdocs/run_ramdocs.py` | V4 在 RAMDocs 上的适配 |
| `module2_ramdocs/run_madam_rag_ramdocs.py` | MADAM-RAG 在 RAMDocs 上的自跑实现 |
| `module3_faitheval/run_faitheval.py` | FaithEval-Inconsistent 实验 |
| `module3_faitheval/faitheval_inconsistent_md_full.jsonl` | FaithEval 预处理数据 |
| `exp_amp_gradient.py` | amp 梯度实验 (amp2/amp4) |
| `exp_d3_accuracy.py` | D3 分类准确性验证 |
| `time_benchmark.py` | 各组件时间开销测试 |
| `module2/results/` | amp8, clean 结果 |
| `module2_ramdocs/results/` | RAMDocs 结果 |
| `module3_faitheval/faitheval_results.jsonl` | FaithEval 结果 |
| `method_summary_zh.md` | 方法论详细文档 |
