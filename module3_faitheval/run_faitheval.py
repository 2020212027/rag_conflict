"""
FaithEval-Inconsistent-MD-Full Experiment
Methods: Naive RAG, V4 (Source-Isolated Arbitration)

Evaluation follows official FaithEval protocol:
- Non-strict ACC: prediction contains any conflict phrase
- Strict ACC: normalized prediction == "conflict"
"""
import json, time, os, re, sys
from itertools import combinations
from openai import OpenAI

MODEL = "gpt-4o-mini"
client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

DATA_PATH = r"d:\pythonProject\module3_faitheval\faitheval_inconsistent_md_full.jsonl"
CKPT_PATH = r"d:\pythonProject\module3_faitheval\faitheval_checkpoint.jsonl"
OUTPUT_PATH = r"d:\pythonProject\module3_faitheval\faitheval_results.jsonl"

# FaithEval official conflict phrases (non-strict)
CONFLICT_PHRASES = [
    "conflict", "multiple answers", "disagreement", "inconsistent",
    "contradictory", "contradiction", "inconsistency", "two answers", "conflicting"
]

JACCARD_THRESHOLD = 0.20
STOPWORDS = set("a an the is was were be been being have has had do does did will would shall should may might can could and or but if then else when at by for with about against between through during before after above below to from up down in out on off over under again further once here there all each every both few more most other some such no nor not only own same so than too very".split())

# Official FaithEval-style prompt
FAITHEVAL_PROMPT = """You are an expert in retrieval question answering.
Please respond with the exact answer only. Do not be verbose or provide extra information.
If there is conflict information or multiple answers from the context, the answer should be "conflict".

Context:
{context}

Question:
{question}

Answer:"""

# Dependency judgment prompt (for V4 Module 1)
DEPENDENCE_PROMPT = """You judge whether two document excerpts are INDEPENDENT or DEPENDENT (from same source).

RULES:
- Same topic + different wording/details = D0/D1 (INDEPENDENT)
- Shared RARE details, near-identical structure, or shared errors = D3 (DEPENDENT)
- D3 requires: content that could NOT be independently produced

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

# V4 Module 2: Source-isolated extraction per cluster
ISOLATED_EXTRACT_PROMPT = """You are an expert in retrieval question answering.
Please respond with the exact answer only. Do not be verbose or provide extra information.
If there is conflict information or multiple answers from the context, the answer should be "conflict".

Context:
{context}

Question:
{question}

Answer:"""

# V4 Module 2: Arbitration across cluster answers
ARBITRATION_PROMPT = """You are an expert in retrieval question answering.
Multiple independent source groups provide different answers to the same question.
If the sources give conflicting/different answers, respond with "conflict".
If they agree, give the agreed answer.

Question: {question}

Source answers:
{source_answers}

Final answer:"""


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


def eval_nonstrict(prediction: str) -> bool:
    """FaithEval non-strict: contains any conflict phrase."""
    pred_lower = prediction.lower().strip()
    return any(phrase in pred_lower for phrase in CONFLICT_PHRASES)


def eval_strict(prediction: str) -> bool:
    """FaithEval strict: normalized answer == 'conflict'."""
    pred_normalized = prediction.lower().strip().rstrip(".!").strip()
    return pred_normalized == "conflict"


def run_naive(question: str, documents: list) -> str:
    """Naive RAG: concatenate all docs, ask with FaithEval prompt."""
    context = "\n\n".join(f"[Document {i+1}]\n{doc}" for i, doc in enumerate(documents))
    prompt = FAITHEVAL_PROMPT.format(context=context, question=question)
    return call_llm(prompt, max_tokens=64)


def run_v4(question: str, documents: list) -> tuple:
    """
    V4: Source-Isolated Arbitration.
    Module 1: Cluster documents by dependency.
    Module 2: Extract answer per cluster, then arbitrate.
    Returns (final_answer, api_calls)
    """
    n = len(documents)
    api_calls = 0

    # Module 1: Pairwise dependency (Jaccard filter + D3 judgment)
    candidate_pairs = []
    for i, j in combinations(range(n), 2):
        jac = jaccard(documents[i], documents[j])
        if jac >= JACCARD_THRESHOLD:
            candidate_pairs.append((i, j, jac))

    uf = UnionFind(n)

    if candidate_pairs:
        for i, j, jac in candidate_pairs:
            prompt = DEPENDENCE_PROMPT.format(
                question=question,
                doc_a=documents[i][:300],
                doc_b=documents[j][:300]
            )
            result = call_llm(prompt, max_tokens=150, expect_json=True)
            api_calls += 1
            time.sleep(0.3)

            level = result.get("level", "D0") if result else "D0"
            if level == "D3":
                uf.union(i, j)

    clusters = uf.clusters(n)

    # Module 2: Source-isolated extraction
    cluster_answers = []
    for cluster in clusters:
        cluster_docs = [documents[idx] for idx in cluster]
        context = "\n\n".join(f"[Document {k+1}]\n{doc}" for k, doc in enumerate(cluster_docs))
        prompt = ISOLATED_EXTRACT_PROMPT.format(context=context, question=question)
        answer = call_llm(prompt, max_tokens=64)
        api_calls += 1
        time.sleep(0.3)
        cluster_answers.append(answer)

    # If only one cluster, return its answer directly
    if len(cluster_answers) == 1:
        return cluster_answers[0], api_calls

    # Arbitration: compare answers from different clusters
    source_text = "\n".join(f"- Source {i+1}: {ans}" for i, ans in enumerate(cluster_answers))
    prompt = ARBITRATION_PROMPT.format(question=question, source_answers=source_text)
    final_answer = call_llm(prompt, max_tokens=64)
    api_calls += 1
    time.sleep(0.3)

    return final_answer, api_calls


def process_sample(sample, idx):
    question = sample["question"]
    documents = sample["documents"]

    # Naive RAG
    naive_answer = run_naive(question, documents)
    time.sleep(0.3)

    # V4
    v4_answer, v4_api_calls = run_v4(question, documents)

    return {
        "idx": idx,
        "qid": sample["qid"],
        "question": question,
        "subset": sample["subset"],
        "gold": "conflict",
        "naive_answer": naive_answer,
        "naive_nonstrict": eval_nonstrict(naive_answer),
        "naive_strict": eval_strict(naive_answer),
        "v4_answer": v4_answer,
        "v4_nonstrict": eval_nonstrict(v4_answer),
        "v4_strict": eval_strict(v4_answer),
        "v4_api_calls": v4_api_calls,
    }


def main():
    num_samples = 500  # Run first 500 by default
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])

    print(f"{'='*60}")
    print(f"FaithEval-Inconsistent-MD-Full Experiment")
    print(f"Samples: {num_samples} | Model: {MODEL}")
    print(f"{'='*60}", flush=True)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()][:num_samples]
    print(f"Loaded {len(data)} samples")

    # Load checkpoint
    done = {}
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
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

        with open(CKPT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if len(done) % 10 == 0 or idx == len(data) - 1:
            elapsed = time.time() - t0
            n_ns = sum(1 for r in done.values() if r["naive_nonstrict"]) / len(done) * 100
            n_s = sum(1 for r in done.values() if r["naive_strict"]) / len(done) * 100
            v_ns = sum(1 for r in done.values() if r["v4_nonstrict"]) / len(done) * 100
            v_s = sum(1 for r in done.values() if r["v4_strict"]) / len(done) * 100
            avg_calls = sum(r["v4_api_calls"] for r in done.values()) / len(done)
            print(f"  [{len(done)}/{len(data)}] "
                  f"Naive(ns={n_ns:.1f}% s={n_s:.1f}%) "
                  f"V4(ns={v_ns:.1f}% s={v_s:.1f}%) "
                  f"AvgCalls={avg_calls:.1f} | {elapsed/60:.1f}min", flush=True)

    # Final summary
    all_r = list(done.values())
    naive_ns = sum(1 for r in all_r if r["naive_nonstrict"]) / len(all_r) * 100
    naive_s = sum(1 for r in all_r if r["naive_strict"]) / len(all_r) * 100
    v4_ns = sum(1 for r in all_r if r["v4_nonstrict"]) / len(all_r) * 100
    v4_s = sum(1 for r in all_r if r["v4_strict"]) / len(all_r) * 100
    avg_calls = sum(r["v4_api_calls"] for r in all_r) / len(all_r)

    # Per-subset breakdown
    subsets = sorted(set(r["subset"] for r in all_r))

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (N={len(all_r)})")
    print(f"{'='*60}")
    print(f"\n--- Overall ---")
    print(f"  {'Method':<15} {'Non-strict ACC':>15} {'Strict ACC':>12}")
    print(f"  {'Naive RAG':<15} {naive_ns:>14.1f}% {naive_s:>11.1f}%")
    print(f"  {'V4 (Ours)':<15} {v4_ns:>14.1f}% {v4_s:>11.1f}%")
    print(f"  V4 Avg API calls: {avg_calls:.1f}")

    print(f"\n--- Per-Subset Non-strict ACC ---")
    print(f"  {'Subset':<25} {'Naive':>8} {'V4':>8}")
    for subset in subsets:
        sub_r = [r for r in all_r if r["subset"] == subset]
        sub_naive = sum(1 for r in sub_r if r["naive_nonstrict"]) / len(sub_r) * 100
        sub_v4 = sum(1 for r in sub_r if r["v4_nonstrict"]) / len(sub_r) * 100
        print(f"  {subset:<25} {sub_naive:>7.1f}% {sub_v4:>7.1f}%")

    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in sorted(all_r, key=lambda x: x["idx"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
