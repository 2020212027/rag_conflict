"""
Experiment: D3 Classification Accuracy Validation
Uses amp8 ground truth (document type field) to evaluate Module 1 clustering quality.

Ground truth logic:
- misinfo + misinfo_amplified = same source cluster (should be merged)
- correct / noise = independent sources (should NOT be merged with misinfo cluster)

This script re-runs Module 1 only (no RAG call) and records per-doc cluster assignments,
then compares against ground truth.
"""
import json, time, re, os
from itertools import combinations
from openai import OpenAI

MODEL = "gpt-4o-mini"
client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

JACCARD_THRESHOLD = 0.20
STOPWORDS = set("a an the is was were be been being have has had do does did will would shall should may might can could and or but if then else when at by for with about against between through during before after above below to from up down in out on off over under again further once here there all each every both few more most other some such no nor not only own same so than too very".split())

DEPENDENCE_PROMPT = """You judge whether two document excerpts are INDEPENDENT or DEPENDENT (from same source).

RULES:
- Same topic + different wording/details = D0/D1 (INDEPENDENT)
- Shared RARE details, near-identical structure, or shared errors = D3 (DEPENDENT)
- D3 requires: content that could NOT be independently produced

CALIBRATION:
- Two Wikipedia articles about the same subject = D1, NOT D3
- Two news reports covering the same event = D1, NOT D3  
- Near-identical text with minor paraphrasing = D3
- Same rare statistics, same unusual phrasing, same errors = D3

Question: {question}

Document A (first 300 chars): {doc_a}

Document B (first 300 chars): {doc_b}

Rate dependency:
D0=Completely unrelated
D1=Same topic, independently written
D2=Partial dependency (shared source but significant independent content)
D3=High dependency (paraphrase/copy of same source)

Output ONLY valid JSON:
{{"level":"D0/D1/D2/D3","reasoning":"one sentence"}}"""


def tokenize(text):
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in STOPWORDS and len(w) > 1]


def jaccard(text_a, text_b):
    tokens_a = set(tokenize(text_a[:800]))
    tokens_b = set(tokenize(text_b[:800]))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def call_llm(prompt, max_tokens=200, expect_json=False):
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content.strip()
            if expect_json:
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(content)
            return content
        except Exception as e:
            wait = 2 * (attempt + 1)
            if attempt < 4:
                time.sleep(wait)
    return {} if expect_json else ""


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def clusters(self, n):
        groups = {}
        for i in range(n):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return list(groups.values())


def run_module1(question, documents):
    """Run Module 1 clustering only, return cluster assignments and pair judgments."""
    n = min(len(documents), 10)
    texts = [d["text"] for d in documents[:n]]

    candidate_pairs = []
    for i, j in combinations(range(n), 2):
        jac = jaccard(texts[i], texts[j])
        if jac >= JACCARD_THRESHOLD:
            candidate_pairs.append((i, j, jac))

    if not candidate_pairs:
        return list(range(n)), [], 0  # each doc is its own cluster

    uf = UnionFind(n)
    pair_judgments = []
    api_calls = 0

    for i, j, jac in candidate_pairs:
        prompt = DEPENDENCE_PROMPT.format(
            question=question,
            doc_a=texts[i][:300],
            doc_b=texts[j][:300]
        )
        result = call_llm(prompt, max_tokens=150, expect_json=True)
        api_calls += 1
        time.sleep(0.3)

        level = result.get("level", "D0") if result else "D0"
        pair_judgments.append({
            "doc_i": i, "doc_j": j,
            "type_i": documents[i]["type"], "type_j": documents[j]["type"],
            "jaccard": round(jac, 3),
            "level": level,
            "reasoning": result.get("reasoning", "") if result else ""
        })

        if level == "D3":
            uf.union(i, j)

    # Get cluster assignments
    clusters = uf.clusters(n)
    cluster_assignment = [0] * n
    for cluster_id, members in enumerate(clusters):
        for m in members:
            cluster_assignment[m] = cluster_id

    return cluster_assignment, pair_judgments, api_calls


def evaluate_clustering(documents, cluster_assignment):
    """
    Evaluate clustering against ground truth.
    Ground truth: misinfo + misinfo_amplified should be in same cluster;
                  correct + noise should NOT be in that cluster.
    """
    n = min(len(documents), 10)
    types = [documents[i]["type"] for i in range(n)]

    # Identify misinfo source docs (misinfo + misinfo_amplified)
    misinfo_indices = [i for i in range(n) if types[i] in ("misinfo", "misinfo_amplified")]
    clean_indices = [i for i in range(n) if types[i] in ("correct", "noise")]

    if not misinfo_indices:
        return {"no_misinfo": True}

    # Find the dominant cluster for misinfo docs
    misinfo_clusters = [cluster_assignment[i] for i in misinfo_indices]
    from collections import Counter
    cluster_counts = Counter(misinfo_clusters)
    dominant_cluster = cluster_counts.most_common(1)[0][0]

    # Metrics
    # 1. Rewrite Recall: fraction of misinfo docs in the dominant cluster
    in_dominant = sum(1 for i in misinfo_indices if cluster_assignment[i] == dominant_cluster)
    rewrite_recall = in_dominant / len(misinfo_indices)

    # 2. False Merge Rate: fraction of clean docs incorrectly in the dominant cluster
    falsely_merged = sum(1 for i in clean_indices if cluster_assignment[i] == dominant_cluster)
    false_merge_rate = falsely_merged / len(clean_indices) if clean_indices else 0.0

    # 3. Number of clusters
    num_clusters = len(set(cluster_assignment))

    # 4. Ideal: all misinfo in 1 cluster, all clean separate
    #    Ideal clusters = len(clean_indices) + 1
    ideal_clusters = len(clean_indices) + 1

    return {
        "num_misinfo_docs": len(misinfo_indices),
        "num_clean_docs": len(clean_indices),
        "rewrite_recall": rewrite_recall,
        "false_merge_rate": false_merge_rate,
        "num_clusters": num_clusters,
        "ideal_clusters": ideal_clusters,
        "misinfo_in_dominant": in_dominant,
        "clean_falsely_merged": falsely_merged,
    }


def main():
    data_path = r"d:\pythonProject\dataset_amp_8.jsonl"
    ckpt_path = r"d:\pythonProject\d3_accuracy_checkpoint.jsonl"
    output_path = r"d:\pythonProject\d3_accuracy_results.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]

    print(f"D3 Classification Accuracy Experiment")
    print(f"Dataset: amp8, {len(data)} samples")
    print(f"{'='*60}", flush=True)

    # Load checkpoint
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["idx"]] = r
        print(f"Checkpoint: {len(done)} done, {len(data)-len(done)} remaining")

    t0 = time.time()

    for idx, sample in enumerate(data):
        if idx in done:
            continue

        cluster_assignment, pair_judgments, api_calls = run_module1(
            sample["question"], sample["documents"]
        )

        eval_result = evaluate_clustering(sample["documents"], cluster_assignment)

        result = {
            "idx": idx,
            "question": sample["question"],
            "doc_types": [d["type"] for d in sample["documents"][:10]],
            "cluster_assignment": cluster_assignment,
            "pair_judgments": pair_judgments,
            "api_calls": api_calls,
            **eval_result
        }
        done[idx] = result

        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if len(done) % 10 == 0:
            elapsed = time.time() - t0
            valid = [r for r in done.values() if not r.get("no_misinfo")]
            if valid:
                avg_recall = sum(r["rewrite_recall"] for r in valid) / len(valid)
                avg_fmr = sum(r["false_merge_rate"] for r in valid) / len(valid)
                avg_clusters = sum(r["num_clusters"] for r in valid) / len(valid)
                print(f"  [{len(done)}/{len(data)}] "
                      f"RewriteRecall={avg_recall:.3f} FalseMergeRate={avg_fmr:.3f} "
                      f"AvgClusters={avg_clusters:.1f} | {elapsed/60:.1f}min", flush=True)

    # Final summary
    all_r = [r for r in done.values() if not r.get("no_misinfo")]
    avg_recall = sum(r["rewrite_recall"] for r in all_r) / len(all_r)
    avg_fmr = sum(r["false_merge_rate"] for r in all_r) / len(all_r)
    avg_clusters = sum(r["num_clusters"] for r in all_r) / len(all_r)
    avg_ideal = sum(r["ideal_clusters"] for r in all_r) / len(all_r)

    # Perfect recall (all misinfo merged)
    perfect_recall = sum(1 for r in all_r if r["rewrite_recall"] == 1.0) / len(all_r)
    # Zero false merge
    zero_fm = sum(1 for r in all_r if r["false_merge_rate"] == 0.0) / len(all_r)

    # Pair-level accuracy
    all_pairs = []
    for r in done.values():
        for pj in r.get("pair_judgments", []):
            # Ground truth: same-source pair if both are misinfo/misinfo_amplified
            is_same_source = (pj["type_i"] in ("misinfo", "misinfo_amplified") and
                             pj["type_j"] in ("misinfo", "misinfo_amplified"))
            predicted_same = (pj["level"] == "D3")
            all_pairs.append({
                "gt_same_source": is_same_source,
                "predicted_d3": predicted_same,
                "level": pj["level"]
            })

    # Pair-level metrics
    tp = sum(1 for p in all_pairs if p["gt_same_source"] and p["predicted_d3"])
    fp = sum(1 for p in all_pairs if not p["gt_same_source"] and p["predicted_d3"])
    fn = sum(1 for p in all_pairs if p["gt_same_source"] and not p["predicted_d3"])
    tn = sum(1 for p in all_pairs if not p["gt_same_source"] and not p["predicted_d3"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'='*60}")
    print(f"FINAL D3 ACCURACY RESULTS (N={len(all_r)})")
    print(f"{'='*60}")
    print(f"\n--- Cluster-Level Metrics ---")
    print(f"  Avg Rewrite Recall:     {avg_recall:.3f}  (fraction of misinfo docs correctly merged)")
    print(f"  Avg False Merge Rate:   {avg_fmr:.3f}  (fraction of clean docs wrongly merged)")
    print(f"  Perfect Recall Rate:    {perfect_recall:.1%}  (samples where ALL misinfo merged)")
    print(f"  Zero False-Merge Rate:  {zero_fm:.1%}  (samples with NO wrong merges)")
    print(f"  Avg Predicted Clusters: {avg_clusters:.1f}  (ideal: {avg_ideal:.1f})")
    print(f"\n--- Pair-Level Metrics (D3 as positive) ---")
    print(f"  Total pairs judged:     {len(all_pairs)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision: {precision:.3f}  (D3 predictions that are truly same-source)")
    print(f"  Recall:    {recall:.3f}  (same-source pairs correctly identified as D3)")
    print(f"  F1:        {f1:.3f}")
    print(f"\n--- Interpretation ---")
    print(f"  High Precision = low false merges (safe dedup)")
    print(f"  High Recall = catches most redundancy")
    print(f"  FP={fp} means {fp} cross-source pairs wrongly merged (information loss risk)")

    # Save full results
    summary = {
        "dataset": "amp8",
        "num_samples": len(all_r),
        "cluster_metrics": {
            "avg_rewrite_recall": round(avg_recall, 4),
            "avg_false_merge_rate": round(avg_fmr, 4),
            "perfect_recall_rate": round(perfect_recall, 4),
            "zero_false_merge_rate": round(zero_fm, 4),
            "avg_predicted_clusters": round(avg_clusters, 2),
            "avg_ideal_clusters": round(avg_ideal, 2),
        },
        "pair_metrics": {
            "total_pairs": len(all_pairs),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved summary: {output_path}")


if __name__ == "__main__":
    main()
