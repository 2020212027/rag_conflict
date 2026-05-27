"""
Experiment: Amplification Gradient (amp2/amp4/amp8)
Runs V4 (Layered Dedup) on amp2 and amp4 datasets.
amp8 results already exist in results_layered_amp8.jsonl.
"""
import json, time, os, re, sys
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

RAG_PROMPT = """Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

Question: {question}

{doc_texts}

Answer:"""


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
    n = min(len(documents), 10)
    texts = [d["text"] for d in documents[:n]]

    candidate_pairs = []
    for i, j in combinations(range(n), 2):
        jac = jaccard(texts[i], texts[j])
        if jac >= JACCARD_THRESHOLD:
            candidate_pairs.append((i, j, jac))

    if not candidate_pairs:
        return list(range(len(documents))), 0, [], 0

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

    all_clusters = uf.clusters(n)
    cluster_sizes = [len(c) for c in all_clusters if len(c) > 1]

    kept = set()
    for cluster in all_clusters:
        if len(cluster) == 1:
            kept.add(cluster[0])
        else:
            best = max(cluster, key=lambda idx: len(texts[idx]))
            kept.add(best)

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

    # Naive RAG
    doc_texts = "\n\n".join(f"[Document {k+1}]\n{d['text'][:500]}" for k, d in enumerate(documents))
    naive_answer = call_llm(RAG_PROMPT.format(question=question, doc_texts=doc_texts), max_tokens=64)
    time.sleep(0.3)

    # V4 dedup
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
    if len(sys.argv) < 2 or sys.argv[1] not in ("amp2", "amp4"):
        print("Usage: python exp_amp_gradient.py [amp2|amp4]")
        sys.exit(1)

    mode = sys.argv[1]
    data_path = rf"d:\pythonProject\dataset_{mode.replace('amp','amp_')}.jsonl"
    ckpt_path = rf"d:\pythonProject\layered_{mode}_checkpoint.jsonl"
    output_path = rf"d:\pythonProject\results_layered_{mode}.jsonl"
    log_path = rf"d:\pythonProject\layered_{mode}.log"

    print(f"{'='*60}")
    print(f"Amp Gradient Experiment: {mode.upper()}")
    print(f"Data: {data_path}")
    print(f"{'='*60}", flush=True)

    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(data)} samples")

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
        result = process_sample(sample, idx)
        done[idx] = result

        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if len(done) % 10 == 0 or idx == len(data) - 1:
            elapsed = time.time() - t0
            n_acc = sum(1 for r in done.values() if r["naive_correct"]) / len(done) * 100
            d_acc = sum(1 for r in done.values() if r["dedup_correct"]) / len(done) * 100
            n_wrong = sum(1 for r in done.values() if r["naive_wrong"]) / len(done) * 100
            d_wrong = sum(1 for r in done.values() if r["dedup_wrong"]) / len(done) * 100
            avg_rm = sum(r["num_removed"] for r in done.values()) / len(done)
            print(f"  [{len(done)}/{len(data)}] Naive={n_acc:.1f}% V4={d_acc:.1f}% "
                  f"NaiveWrong={n_wrong:.1f}% V4Wrong={d_wrong:.1f}% AvgRm={avg_rm:.1f} "
                  f"| {elapsed/60:.1f}min", flush=True)

    # Final
    all_r = list(done.values())
    n_acc = sum(1 for r in all_r if r["naive_correct"]) / len(all_r) * 100
    d_acc = sum(1 for r in all_r if r["dedup_correct"]) / len(all_r) * 100
    n_wrong = sum(1 for r in all_r if r["naive_wrong"]) / len(all_r) * 100
    d_wrong = sum(1 for r in all_r if r["dedup_wrong"]) / len(all_r) * 100

    print(f"\n{'='*60}")
    print(f"FINAL {mode.upper()} (N={len(all_r)})")
    print(f"{'='*60}")
    print(f"  Naive ACC:    {n_acc:.1f}%")
    print(f"  V4 ACC:       {d_acc:.1f}%")
    print(f"  Delta:        {d_acc-n_acc:+.1f}pp")
    print(f"  Naive Wrong:  {n_wrong:.1f}%")
    print(f"  V4 Wrong:     {d_wrong:.1f}%")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in sorted(all_r, key=lambda x: x["idx"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {output_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Mode: {mode}\nN: {len(all_r)}\n")
        f.write(f"Naive ACC: {n_acc:.1f}%\nV4 ACC: {d_acc:.1f}%\nDelta: {d_acc-n_acc:+.1f}pp\n")
        f.write(f"Naive Wrong: {n_wrong:.1f}%\nV4 Wrong: {d_wrong:.1f}%\n")


if __name__ == "__main__":
    main()
