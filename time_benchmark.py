"""
Time benchmark: measure actual wall-clock time for each component of V4 pipeline.
Runs 10 samples from amp8 dataset.
"""
import json, time, os, re, sys
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

MODEL = "gpt-4o-mini"
client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

JACCARD_THRESHOLD = 0.40
MAX_WORKERS = 5
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

SOURCE_EXTRACTION_PROMPT = """You are answering a question based ONLY on the provided source document. 
If the document does not contain enough information to answer, say so.

Question: {question}

Source Document:
{source_text}

Respond in strict JSON format:
{{"answer": "your short answer or 'unknown'", "status": "supported|unsupported|ambiguous", "support": "one key sentence from the document that supports your answer", "confidence": "high|medium|low"}}"""

ARBITRATION_PROMPT = """You are resolving a factual disagreement between independent information sources.

Question: {question}

{positions_text}

Instructions:
- Each position comes from an independently verified source
- Evaluate based on evidence QUALITY and SPECIFICITY, not quantity
- Prefer answers with direct, specific supporting evidence

Provide your final answer (short, direct):"""

NAIVE_RAG_PROMPT = """Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

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
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                return json.loads(content)
            return content
        except json.JSONDecodeError:
            if attempt < 4:
                time.sleep(1)
            continue
        except Exception as e:
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
    return {} if expect_json else ""


def normalize_answer(s):
    if not s:
        return ""
    s = s.lower().strip().rstrip(".")
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch.isalnum() or ch == ' ')
    return ' '.join(s.split())


def answers_match(a, b):
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and shorter in longer:
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


def benchmark_sample(sample):
    question = sample["question"]
    documents = sample["documents"]
    timings = {}

    # --- Naive RAG ---
    t0 = time.time()
    doc_texts = "\n\n".join(f"[Document {k+1}]\n{d['text'][:500]}" for k, d in enumerate(documents))
    naive_answer = call_llm(NAIVE_RAG_PROMPT.format(question=question, doc_texts=doc_texts), max_tokens=64)
    timings["naive_rag"] = time.time() - t0

    # --- Module 1: Jaccard filter (no API) ---
    t0 = time.time()
    n = min(len(documents), 10)
    texts = [d["text"] for d in documents[:n]]
    candidate_pairs = []
    for i, j in combinations(range(n), 2):
        jac = jaccard(texts[i], texts[j])
        if jac >= JACCARD_THRESHOLD:
            candidate_pairs.append((i, j, jac))
    timings["m1_jaccard"] = time.time() - t0
    timings["m1_jaccard_pairs"] = len(candidate_pairs)

    # --- Module 1: D3 judgment (API, concurrent) ---
    t0 = time.time()
    uf = UnionFind(n)
    m1_api_calls = 0
    if candidate_pairs:
        def judge_pair(pair):
            i, j, jac = pair
            prompt = DEPENDENCE_PROMPT.format(question=question, doc_a=texts[i][:300], doc_b=texts[j][:300])
            result = call_llm(prompt, max_tokens=150, expect_json=True)
            level = result.get("level", "D0") if result else "D0"
            return i, j, level

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(judge_pair, p) for p in candidate_pairs]
            for future in as_completed(futures):
                i, j, level = future.result()
                m1_api_calls += 1
                if level == "D3":
                    uf.union(i, j)
    timings["m1_d3_judgment"] = time.time() - t0
    timings["m1_api_calls"] = m1_api_calls

    clusters = uf.clusters(n)
    for i in range(n, len(documents)):
        clusters.append([i])

    # --- Step 1: Source Unit Construction (no API) ---
    t0 = time.time()
    source_units = []
    for cluster in clusters:
        rep_idx = min(cluster)
        source_units.append({
            "representative_text": documents[rep_idx]["text"],
            "cluster_size": len(cluster),
        })
    timings["step1_construction"] = time.time() - t0
    timings["num_clusters"] = len(clusters)
    timings["num_source_units"] = len(source_units)

    # --- Step 2: Source-Isolated Extraction (API, concurrent) ---
    t0 = time.time()
    extractions = [None] * len(source_units)

    def extract_one(idx, unit):
        prompt = SOURCE_EXTRACTION_PROMPT.format(question=question, source_text=unit["representative_text"])
        return idx, call_llm(prompt, max_tokens=150, expect_json=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(extract_one, i, u) for i, u in enumerate(source_units)]
        for future in as_completed(futures):
            idx, result = future.result()
            if not result:
                result = {"answer": "unknown", "status": "unsupported", "support": "", "confidence": "low"}
            extractions[idx] = result
    timings["step2_extraction"] = time.time() - t0
    timings["step2_api_calls"] = len(source_units)

    # --- Step 3: Algorithmic Arbitration ---
    t0 = time.time()
    supported = []
    for i, ext in enumerate(extractions):
        if ext and ext.get("status") == "supported" and ext.get("answer", "").lower() not in ("unknown", "i don't know", ""):
            supported.append((i, ext))

    decision_type = "consensus"
    arb_calls = 0
    if supported:
        groups = {}
        for source_idx, ext in supported:
            norm = normalize_answer(ext["answer"])
            matched_key = None
            for existing_key in groups:
                if answers_match(norm, existing_key):
                    matched_key = existing_key
                    break
            if matched_key:
                groups[matched_key].append((source_idx, ext))
            else:
                groups[norm] = [(source_idx, ext)]

        if len(groups) > 1:
            decision_type = "conflict_arbitrated"
            sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]))
            lines = []
            for pos_idx, (answer_key, members) in enumerate(sorted_groups, 1):
                lines.append(f"== Position {pos_idx}: \"{members[0][1]['answer']}\" ({len(members)} source(s)) ==")
                for si, ext in members:
                    lines.append(f"  Evidence: \"{ext.get('support', 'N/A')}\"")
            positions_text = "\n".join(lines)
            final = call_llm(ARBITRATION_PROMPT.format(question=question, positions_text=positions_text), max_tokens=100)
            arb_calls = 1

    timings["step3_arbitration"] = time.time() - t0
    timings["step3_api_calls"] = arb_calls
    timings["decision_type"] = decision_type

    # --- Total V4 time ---
    timings["v4_total"] = (timings["m1_jaccard"] + timings["m1_d3_judgment"] +
                           timings["step1_construction"] + timings["step2_extraction"] +
                           timings["step3_arbitration"])
    timings["total_api_calls"] = m1_api_calls + len(source_units) + arb_calls + 1  # +1 for naive

    return timings


def main():
    data_path = r"d:\pythonProject\dataset_amp_8.jsonl"
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()][:10]

    print(f"Time Benchmark: V4 Pipeline on amp8 ({len(data)} samples)")
    print(f"Model: {MODEL} | Concurrency: {MAX_WORKERS}")
    print("=" * 70)

    all_timings = []
    for idx, sample in enumerate(data):
        print(f"\n--- Sample {idx} ---")
        t = benchmark_sample(sample)
        all_timings.append(t)
        print(f"  Naive RAG:          {t['naive_rag']:.2f}s (1 call)")
        print(f"  M1 Jaccard filter:  {t['m1_jaccard']:.3f}s (0 calls, {t['m1_jaccard_pairs']} pairs)")
        print(f"  M1 D3 judgment:     {t['m1_d3_judgment']:.2f}s ({t['m1_api_calls']} calls, concurrent)")
        print(f"  Step1 construction: {t['step1_construction']:.4f}s (0 calls)")
        print(f"  Step2 extraction:   {t['step2_extraction']:.2f}s ({t['step2_api_calls']} calls, concurrent)")
        print(f"  Step3 arbitration:  {t['step3_arbitration']:.2f}s ({t['step3_api_calls']} calls) [{t['decision_type']}]")
        print(f"  ---")
        print(f"  V4 total wall-time: {t['v4_total']:.2f}s")
        print(f"  Total API calls:    {t['total_api_calls']}")
        print(f"  Clusters: {t['num_clusters']} | Sources: {t['num_source_units']}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({len(all_timings)} samples)")
    print(f"{'='*70}")

    def avg(key):
        return sum(t[key] for t in all_timings) / len(all_timings)

    print(f"\n{'Component':<25} {'Avg Time':>10} {'Avg Calls':>10} {'% of V4':>8}")
    print(f"{'-'*55}")

    v4_avg = avg("v4_total")
    components = [
        ("Naive RAG", "naive_rag", 1),
        ("M1 Jaccard (local)", "m1_jaccard", 0),
        ("M1 D3 judgment", "m1_d3_judgment", avg("m1_api_calls")),
        ("Step1 construction", "step1_construction", 0),
        ("Step2 extraction", "step2_extraction", avg("step2_api_calls")),
        ("Step3 arbitration", "step3_arbitration", avg("step3_api_calls")),
    ]
    for name, key, calls in components:
        t_avg = avg(key)
        pct = t_avg / v4_avg * 100 if v4_avg > 0 else 0
        calls_str = f"{calls:.1f}" if isinstance(calls, float) else str(calls)
        print(f"  {name:<23} {t_avg:>8.2f}s {calls_str:>10} {pct:>7.1f}%")

    print(f"\n  {'V4 Total':<23} {v4_avg:>8.2f}s {avg('total_api_calls'):>10.1f}")
    print(f"  {'Naive RAG only':<23} {avg('naive_rag'):>8.2f}s {'1':>10}")
    print(f"\n  Speedup vs serial (est): {avg('total_api_calls')*1.0:.0f} calls serial ~{avg('total_api_calls')*1.0:.0f}s vs {v4_avg:.1f}s concurrent")


if __name__ == "__main__":
    main()
