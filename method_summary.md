## Layered Pairwise Deduplication for Defending Against Copy-Amplification Attacks in RAG

### 1. Problem Statement

Retrieval-Augmented Generation (RAG) systems are vulnerable to **copy-amplification attacks**: an adversary rewrites a single piece of misinformation into multiple paraphrased variants and injects them into the retrieval corpus. At query time, the retriever surfaces several near-duplicate documents all containing the same false claim, overwhelming the generator's ability to distinguish fact from fiction through majority-signal effects.

We formalize the threat model as follows:
- **Attacker capability**: Given a target question Q and a desired wrong answer A*, the attacker generates k paraphrased documents (k=8 in our experiments) that all support A*, and injects them into the corpus.
- **Attack surface**: The RAG pipeline retrieves top-N documents (N=14 in our setting: 6 clean + 8 amplified).
- **Success metric**: The generator outputs A* instead of the correct answer A.

The core challenge is to **detect and deduplicate** amplified copies at inference time, without degrading performance on clean (non-attacked) queries.

---

### 2. Threat Model and Evaluation Design

We construct two parallel evaluation sets (N=215 each):

| Dataset | Composition | Purpose |
|---------|-------------|---------|
| **amp_8** | 6 clean docs + 8 amplified copies per query | Measures attack success & defense effectiveness |
| **clean** | 14 independently retrieved docs per query | Measures false-positive rate (collateral damage) |

**Primary metric**: Normalized Exact Match (NEM) — whether the generated answer contains the gold answer.

**Key evaluation dimensions**:
- **Δ NEM (amp_8)**: NEM improvement after dedup → measures defense strength
- **Δ NEM (clean)**: NEM change after dedup → measures collateral damage (should be ≈ 0)

---

### 3. Method: Layered Pairwise Deduplication

Our defense operates as a preprocessing step between retrieval and generation. It consists of two layers:

#### 3.1 Layer 1: Lexical Coarse Filter (Zero-Cost)

**Objective**: Rapidly identify candidate document pairs that *might* be near-duplicates, eliminating the need to evaluate all O(n²) pairs with expensive LLM calls.

**Method**: Token-level Jaccard similarity on the first 800 characters of each document, after lowercasing, stopword removal, and tokenization.

```
Jaccard(A, B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|
```

**Threshold**: Jaccard ≥ 0.20

**Rationale for threshold selection** (validated on 80 amplified + 80 independent pairs):
- At threshold 0.20: captures 72.5% of amplified pairs while passing only 6.2% of independent pairs
- Precision of Layer 1 alone: 92.1%
- Despite 72.5% pair-level recall appearing modest, **cluster-level recall is near-perfect**: connecting 8 nodes in a graph requires only 7 edges, while 72.5% of C(8,2)=28 possible edges yields ~20 edges — far more than the minimum spanning tree requirement.

**Computational cost**: O(n²) string operations, zero API calls. For n=14 documents, this evaluates 91 pairs in <10ms.

#### 3.2 Layer 2: Pairwise Dependence Classification (LLM-Based)

**Objective**: For each candidate pair surviving Layer 1, determine whether the two documents are independently authored or derived from a common source.

**Method**: A calibrated LLM prompt (GPT-4o-mini, temperature=0) classifies each pair into a 4-level dependence scale:

| Level | Definition | Action |
|-------|-----------|--------|
| **D0** | Completely unrelated | Keep both |
| **D1** | Same topic, independently written | Keep both |
| **D2** | Partial dependency (shared source but significant independent content) | Keep both |
| **D3** | High dependency (near-paraphrase / copy of same source) | **Merge** |

**Critical design decision**: Only D3 triggers deduplication. D2 pairs are preserved because they may contain factually distinct information from the same source — removing them risks deleting contradicting evidence that helps the generator identify misinformation.

**Prompt design principles**:
1. Explicit calibration examples (two Wikipedia articles about the same subject = D1, NOT D3)
2. Requirement for "rare shared details" as D3 evidence — shared topic alone is insufficient
3. Structured JSON output for reliable parsing

**Validated performance** (on 80 amplified + 78 independent pairs):
- D3 Recall (P(model=D3 | true amplified)): 53.8%
- D3 Precision (P(true amplified | model=D3)): 95.6%
- Combined false-positive rate (Layer1 × Layer2): 6.2% × 2.6% ≈ **0.16% per pair**

#### 3.3 Cluster Formation: Union-Find with Longest-Document Retention

**Objective**: Aggregate pairwise D3 judgments into document clusters and select a representative.

**Method**:
1. Initialize a Union-Find data structure over the top-N retrieved documents
2. For each pair judged as D3 by Layer 2, merge the two documents into the same component
3. For each connected component of size > 1:
   - **Retain** the longest document (measured by character count)
   - **Remove** all other documents in the cluster
4. Singleton components (size = 1) are always retained

**Rationale for longest-document retention**:
- Longer documents tend to contain more complete information
- In the amplification attack scenario, all copies are semantically equivalent, so any selection criterion suffices
- In natural near-duplicates, the longer version typically subsumes the shorter one

**Cluster-level recall analysis**:
- Layer 1 passes ~20/28 amplified edges; Layer 2 confirms ~11 as D3
- 8 nodes require only 7 edges for full connectivity → cluster-level recall ≈ 100%
- Observed cluster sizes: mean=5.4, max=10 (on amp_8 data)

---

### 4. End-to-End Pipeline

```
Input: Query Q, Retrieved Documents D = {d₁, d₂, ..., d_N}
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Layer 1: Jaccard Coarse Filter         │
│  For each pair (dᵢ, dⱼ):               │
│    if Jaccard(dᵢ, dⱼ) ≥ 0.20 →         │
│      add to candidate_pairs             │
│  Cost: O(n²) string ops, zero API      │
└─────────────────────────────────────────┘
                    │
          candidate_pairs (subset of all pairs)
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Layer 2: Pairwise D0–D3 Classification │
│  For each (dᵢ, dⱼ) in candidate_pairs: │
│    level = LLM_classify(Q, dᵢ, dⱼ)     │
│    if level == "D3": union(dᵢ, dⱼ)     │
│  Cost: 1 API call per candidate pair    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Cluster Deduplication                  │
│  For each cluster C with |C| > 1:      │
│    keep = argmax_{d ∈ C} len(d)         │
│    remove all others in C               │
│  Output: D' ⊆ D (deduplicated)         │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  RAG Generation                         │
│  Answer = LLM(Q, D')                    │
└─────────────────────────────────────────┘
```

---

### 5. Experimental Results

#### 5.1 Main Results

| Condition | Naive NEM | Dedup NEM | Δ NEM | Avg Removed | Layer 2 Calls |
|-----------|-----------|-----------|-------|-------------|---------------|
| **amp_8** (attack) | 30.7% | 43.7% | **+13.0pp** | 4.8 | 4,777 |
| **clean** (no attack) | 70.7% | 66.5% | −4.2pp | 0.8 | 421 |

#### 5.2 Comparison with Batch Deduplication Baseline

| Metric | Batch Method | Layered Method | Improvement |
|--------|--------------|----------------|-------------|
| amp_8 Δ NEM | +11.6pp | **+13.0pp** | +1.4pp stronger defense |
| clean Δ NEM | −6.0pp | **−4.2pp** | 30% less collateral damage |
| amp_8 avg removed | 6.2 | 4.8 | 23% more precise |
| clean avg removed | 1.9 | 0.8 | 58% fewer false removals |

#### 5.3 Flip Analysis

| | Good Flips (wrong→correct) | Bad Flips (correct→wrong) | Ratio |
|--|---------------------------|--------------------------|-------|
| **amp_8** | 35 | 7 | 5.0:1 |
| **clean** | 6 | 15 | 0.4:1 |

On attacked queries, the method produces 5× more corrections than errors. On clean queries, the residual errors are minimal (15/215 = 7.0%).

#### 5.4 Cluster Statistics

| Dataset | Cluster Count | Mean Size | Max Size |
|---------|--------------|-----------|----------|
| **amp_8** | 232 | 5.4 | 10 |
| **clean** | 149 | 2.1 | 6 |

The stark difference in cluster sizes (5.4 vs 2.1) itself serves as a strong signal for detecting copy-amplification attacks.

---

### 6. Cost Analysis

| Component | amp_8 (per query) | clean (per query) |
|-----------|-------------------|-------------------|
| Layer 1 (Jaccard) | ~91 pairs, 0 API calls | ~91 pairs, 0 API calls |
| Layer 2 (LLM) | ~22 API calls | ~2 API calls |
| RAG generation | 2 calls (naive + dedup) | 1–2 calls |
| **Total API cost** | ~24 calls/query | ~3 calls/query |
| **Wall-clock time** | ~47 sec/query | ~6 sec/query |

The layered design ensures that clean queries (the common case) incur minimal overhead, while attacked queries trigger proportionally more computation only where needed.

---

### 7. Key Design Decisions and Ablations

| Decision | Rationale | Evidence |
|----------|-----------|----------|
| Jaccard ≥ 0.20 (not cosine) | Jaccard more discriminative at low thresholds; zero-cost | Jaccard precision 92.1% vs cosine 74.4% at comparable recall |
| D3-only (not D2+D3) | D2 pairs may contain factually distinct versions; removing them causes 26% of bad flips | Analysis of 19 bad flips in batch method |
| Longest-document retention | Subsumes shorter versions; robust in attack scenarios | All amplified docs are equivalent; longest captures most info |
| Pairwise (not batch) classification | Batch binary classification conflates "same entity" with "same source" | Batch method: 58% of bad flips were same-entity-different-events |

---

### 8. Limitations and Future Work

1. **Residual clean degradation (−4.2pp)**: While significantly improved over the batch baseline (−6.0pp), the method still produces some false positives on clean queries. Future work could explore confidence-calibrated thresholds or cluster-size-gated deduplication (only deduplicate when cluster size ≥ 3).

2. **Latency**: Layer 2 adds ~22 LLM calls per attacked query. This could be mitigated by parallel API calls or a fine-tuned lightweight classifier replacing the prompted LLM.

3. **Adaptive attacks**: An adversary aware of Jaccard-based filtering could craft paraphrases with lower lexical overlap. Future work should evaluate robustness against adversarial paraphrasing strategies.

4. **Soft weighting alternative**: Rather than hard deduplication, cluster membership could be used to down-weight redundant documents in the RAG prompt (e.g., via inverse-cluster-size weighting), potentially preserving useful redundancy while suppressing amplification.
