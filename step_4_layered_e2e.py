"""
Step 4: Layered E2E Pipeline
Layer 1: Jaccard >= 0.20 coarse filter (zero API cost)
Layer 2: Pairwise D0-D3 judgment (only for high-Jaccard pairs)
Dedup: D3-only Union-Find, keep longest doc per cluster
"""
import json, time, os, re, sys
from itertools import combinations
from openai import OpenAI

MODEL = "gpt-4o-mini"
LOG_PATH = None
CKPT_PATH = None
OUTPUT_PATH = None

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

RAG_PROMPT = """Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

Question: {question}

{doc_texts}

Answer:"""


def log(msg):
    print(msg, flush=True)
    if LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


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


def normalized_exact_match(prediction, gold_answers):
    pred_lower = prediction.lower().strip().rstrip(".")
    for gold in gold_answers:
        gold_lower = gold.lower().strip()
        if gold_lower in pred_lower or pred_lower in gold_lower:
            return True
    return False


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


def layered_dedup(question, documents):
    """
    Layer 1: Jaccard filter
    Layer 2: Pairwise D3 judgment
    Returns: (kept_indices, num_removed, cluster_sizes, api_calls)
    """
    n = min(len(documents), 10)
    texts = [d["text"] for d in documents[:n]]

    # Layer 1: Jaccard coarse filter
    candidate_pairs = []
    for i, j in combinations(range(n), 2):
        jac = jaccard(texts[i], texts[j])
        if jac >= JACCARD_THRESHOLD:
            candidate_pairs.append((i, j, jac))

    if not candidate_pairs:
        # No high-similarity pairs, skip Layer 2
        return list(range(len(documents))), 0, [], 0

    # Layer 2: Pairwise D0-D3 judgment
    uf = UnionFind(n)
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
        if level == "D3":
            uf.union(i, j)

    # Get clusters
    all_clusters = uf.clusters(n)
    cluster_sizes = [len(c) for c in all_clusters if len(c) > 1]

    # Keep longest doc per cluster
    kept = set()
    for cluster in all_clusters:
        if len(cluster) == 1:
            kept.add(cluster[0])
        else:
            # Pick longest doc in cluster
            best = max(cluster, key=lambda idx: len(texts[idx]))
            kept.add(best)

    # Add docs beyond top-10
    for i in range(n, len(documents)):
        kept.add(i)

    kept_sorted = sorted(kept)
    num_removed = len(documents) - len(kept_sorted)
    return kept_sorted, num_removed, cluster_sizes, api_calls


def process_sample(sample, idx):
    question = sample["question"]
    documents = sample["documents"]
    gold_answers = sample["gold_answers"]
    wrong_answer = sample.get("wrong_answer", "")

    # Naive RAG (all docs)
    doc_texts = "\n\n".join(f"[Document {k+1}]\n{d['text'][:500]}" for k, d in enumerate(documents))
    naive_answer = call_llm(RAG_PROMPT.format(question=question, doc_texts=doc_texts), max_tokens=64)
    time.sleep(0.3)

    # Layered dedup
    kept_indices, num_removed, cluster_sizes, layer2_calls = layered_dedup(question, documents)

    if num_removed > 0:
        sel = [documents[i] for i in kept_indices]
        dt = "\n\n".join(f"[Document {k+1}]\n{d['text'][:500]}" for k, d in enumerate(sel))
        dedup_answer = call_llm(RAG_PROMPT.format(question=question, doc_texts=dt), max_tokens=64)
        time.sleep(0.3)
    else:
        dedup_answer = naive_answer

    return {
        "idx": idx,
        "question": question,
        "gold_answers": gold_answers,
        "wrong_answer": wrong_answer,
        "num_docs": len(documents),
        "num_kept": len(kept_indices),
        "num_removed": num_removed,
        "cluster_sizes": cluster_sizes,
        "layer2_calls": layer2_calls,
        "naive_answer": naive_answer,
        "naive_correct": normalized_exact_match(naive_answer, gold_answers),
        "naive_wrong": normalized_exact_match(naive_answer, [wrong_answer]) if wrong_answer else False,
        "dedup_answer": dedup_answer,
        "dedup_correct": normalized_exact_match(dedup_answer, gold_answers),
        "dedup_wrong": normalized_exact_match(dedup_answer, [wrong_answer]) if wrong_answer else False,
    }


def main():
    global LOG_PATH, CKPT_PATH, OUTPUT_PATH

    if len(sys.argv) < 2 or sys.argv[1] not in ("amp8", "clean"):
        print("Usage: py step_4_layered_e2e.py [amp8|clean]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "amp8":
        data_path = r"d:\pythonProject\dataset_amp_8.jsonl"
        LOG_PATH = r"d:\pythonProject\layered_amp8.log"
        CKPT_PATH = r"d:\pythonProject\layered_amp8_checkpoint.jsonl"
        OUTPUT_PATH = r"d:\pythonProject\results_layered_amp8.jsonl"
    else:
        data_path = r"d:\pythonProject\dataset_clean.jsonl"
        LOG_PATH = r"d:\pythonProject\layered_clean.log"
        CKPT_PATH = r"d:\pythonProject\layered_clean_checkpoint.jsonl"
        OUTPUT_PATH = r"d:\pythonProject\results_layered_clean.jsonl"

    if not os.path.exists(CKPT_PATH):
        open(LOG_PATH, "w", encoding="utf-8").close()

    log("=" * 60)
    log(f"Step 4: Layered E2E - {mode.upper()}")
    log(f"Model: {MODEL} | Jaccard threshold: {JACCARD_THRESHOLD}")
    log("=" * 60)

    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    log(f"Loaded {len(data)} samples")

    done = {}
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["idx"]] = r
    log(f"Checkpoint: {len(done)} done, {len(data)-len(done)} remaining")

    t0 = time.time()
    total_l2_calls = sum(r.get("layer2_calls", 0) for r in done.values())

    for idx, sample in enumerate(data):
        if idx in done:
            continue
        result = process_sample(sample, idx)
        done[idx] = result
        total_l2_calls += result["layer2_calls"]

        with open(CKPT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if len(done) % 10 == 0 or idx == len(data) - 1:
            elapsed = time.time() - t0
            n_em = sum(1 for r in done.values() if r["naive_correct"]) / len(done) * 100
            d_em = sum(1 for r in done.values() if r["dedup_correct"]) / len(done) * 100
            avg_rm = sum(r["num_removed"] for r in done.values()) / len(done)
            remaining = len(data) - len(done)
            speed = elapsed / max(1, len(done) - sum(1 for r in done.values() if r.get("_preloaded")))
            eta_min = speed * remaining / 60 if speed > 0 else 0
            log(f"  [{len(done)}/{len(data)}] NEM={n_em:.1f}->{d_em:.1f}% "
                f"rm={avg_rm:.1f} L2={total_l2_calls} | {elapsed/60:.1f}min ETA~{eta_min:.0f}min")

    # Final summary
    all_r = list(done.values())
    n_em = sum(1 for r in all_r if r["naive_correct"]) / len(all_r) * 100
    d_em = sum(1 for r in all_r if r["dedup_correct"]) / len(all_r) * 100
    avg_rm = sum(r["num_removed"] for r in all_r) / len(all_r)
    flipped_good = sum(1 for r in all_r if not r["naive_correct"] and r["dedup_correct"])
    flipped_bad = sum(1 for r in all_r if r["naive_correct"] and not r["dedup_correct"])
    all_cluster_sizes = []
    for r in all_r:
        all_cluster_sizes.extend(r.get("cluster_sizes", []))

    log(f"\n{'='*60}")
    log(f"FINAL {mode.upper()} (N={len(all_r)})")
    log(f"{'='*60}")
    log(f"  Naive NEM:  {n_em:.1f}%")
    log(f"  Dedup NEM:  {d_em:.1f}%")
    log(f"  Delta NEM:  {d_em-n_em:+.1f}pp")
    log(f"  Avg removed: {avg_rm:.1f}")
    log(f"  Flipped good: {flipped_good} | Flipped bad: {flipped_bad}")
    log(f"  Total Layer2 calls: {total_l2_calls}")
    if all_cluster_sizes:
        log(f"  Cluster sizes: mean={sum(all_cluster_sizes)/len(all_cluster_sizes):.1f} "
            f"max={max(all_cluster_sizes)} count={len(all_cluster_sizes)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in sorted(all_r, key=lambda x: x["idx"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
