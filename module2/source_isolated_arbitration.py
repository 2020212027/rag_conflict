"""
Module 2: Source-Isolated Arbitration (V4)
Pipeline:
  Module 1: Jaccard coarse filter → D3 LLM judgment → Union-Find clustering
  Step 1: Source Unit Construction (representative per cluster)
  Step 2: Source-Isolated Answer Extraction (each source answers independently)
  Step 3: Algorithmic Arbitration (normalize + group answers)
  Step 4: Conditional Final Generation (consensus → direct / conflict → LLM arbitration)
"""
import json, time, os, re, sys
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

# ─── Module 1: Independence Detection (reused from step_4) ───────────────────
JACCARD_THRESHOLD = 0.40
MAX_WORKERS = 5  # concurrent API calls
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

# ─── Module 2 Prompts ────────────────────────────────────────────────────────
SOURCE_EXTRACTION_PROMPT = """You are answering a question based ONLY on the provided source document. 
If the document does not contain enough information to answer, say so.

Question: {question}

Source Document:
{source_text}

Respond in strict JSON format:
{{"answer": "your short answer or 'unknown'", "status": "supported|unsupported|ambiguous", "support": "one key sentence from the document that supports your answer", "confidence": "high|medium|low"}}

Rules:
- "supported": the document clearly contains the answer
- "unsupported": the document does not mention anything relevant
- "ambiguous": the document is vague or contradictory about the answer
- Keep answer short (a few words)
- support must be a direct quote or close paraphrase from the document"""

ARBITRATION_PROMPT = """You are resolving a factual disagreement between independent information sources.

Question: {question}

{positions_text}

Instructions:
- Each position comes from an independently verified source
- The number of sources is already calibrated (duplicates removed)
- Evaluate based on evidence QUALITY and SPECIFICITY, not quantity
- Prefer answers with direct, specific supporting evidence
- If one position cites vague or tangential evidence, weigh it less

Provide your final answer (short, direct):"""


# ─── Utilities ────────────────────────────────────────────────────────────────
def log(msg, log_path=None):
    print(msg, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
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
                timeout=30,
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
            wait = 2 * (attempt + 1)
            if attempt < 4:
                log(f"    [retry {attempt+1}] {type(e).__name__}: {str(e)[:80]}")
                time.sleep(wait)
    return {} if expect_json else ""


def normalize_answer(s):
    """Normalize for comparison: lowercase, remove articles/punctuation, strip."""
    if not s:
        return ""
    s = s.lower().strip().rstrip(".")
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch.isalnum() or ch == ' ')
    return ' '.join(s.split())


def answers_match(a, b):
    """Check if two normalized answers are equivalent (substring match)."""
    if not a or not b:
        return False
    if a == b:
        return True
    # Substring match only if shorter answer has >= 4 chars
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 4 and shorter in longer:
        return True
    return False


def normalized_exact_match(prediction, gold_answers):
    pred_norm = normalize_answer(prediction)
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if answers_match(pred_norm, gold_norm):
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


# ─── Module 1: Independence Detection ────────────────────────────────────────
def module1_cluster(question, documents):
    """
    Layer 1: Jaccard coarse filter
    Layer 2: Pairwise D3 judgment
    Layer 3: Union-Find clustering
    Returns: list of clusters (each cluster = list of doc indices), api_calls
    """
    n = min(len(documents), 10)
    texts = [d["text"] for d in documents[:n]]

    # Layer 1: Jaccard
    candidate_pairs = []
    for i, j in combinations(range(n), 2):
        jac = jaccard(texts[i], texts[j])
        if jac >= JACCARD_THRESHOLD:
            candidate_pairs.append((i, j, jac))

    uf = UnionFind(n)
    api_calls = 0

    if candidate_pairs:
        # Layer 2: D3 judgment (concurrent)
        def judge_pair(pair):
            i, j, jac = pair
            prompt = DEPENDENCE_PROMPT.format(
                question=question,
                doc_a=texts[i][:300],
                doc_b=texts[j][:300]
            )
            result = call_llm(prompt, max_tokens=150, expect_json=True)
            level = result.get("level", "D0") if result else "D0"
            return i, j, level

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(judge_pair, p) for p in candidate_pairs]
            for future in as_completed(futures):
                i, j, level = future.result()
                api_calls += 1
                if level == "D3":
                    uf.union(i, j)

    # Get clusters from top-n
    clusters = uf.clusters(n)

    # Add unchecked docs (index >= n) as individual clusters
    for i in range(n, len(documents)):
        clusters.append([i])

    return clusters, api_calls


# ─── Module 2 Step 1: Source Unit Construction ────────────────────────────────
def build_source_units(question, documents, clusters):
    """
    Each cluster → 1 source unit.
    Representative = first doc in cluster by retrieval rank (lowest index).
    """
    source_units = []
    for cluster in clusters:
        # Representative: lowest index (best retrieval rank)
        representative_idx = min(cluster)
        representative_doc = documents[representative_idx]

        source_units.append({
            "representative_idx": representative_idx,
            "representative_text": representative_doc["text"],
            "cluster_indices": cluster,
            "cluster_size": len(cluster),
        })

    return source_units


# ─── Module 2 Step 2: Source-Isolated Answer Extraction ───────────────────────
def extract_source_answers(question, source_units):
    """
    Each source unit independently answers the question (concurrent).
    Returns list of extraction results in order.
    """
    def extract_one(idx, unit):
        prompt = SOURCE_EXTRACTION_PROMPT.format(
            question=question,
            source_text=unit["representative_text"]
        )
        result = call_llm(prompt, max_tokens=150, expect_json=True)
        if not result:
            result = {"answer": "unknown", "status": "unsupported", "support": "", "confidence": "low"}
        return idx, result

    extractions = [None] * len(source_units)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(extract_one, i, u) for i, u in enumerate(source_units)]
        for future in as_completed(futures):
            idx, result = future.result()
            extractions[idx] = result

    # Ensure required fields for all
    for i, ext in enumerate(extractions):
        if ext is None:
            extractions[i] = {"answer": "unknown", "status": "unsupported", "support": "", "confidence": "low"}
        else:
            ext.setdefault("answer", "unknown")
            ext.setdefault("status", "ambiguous")
            ext.setdefault("support", "")
            ext.setdefault("confidence", "low")

    return extractions


# ─── Module 2 Step 3: Algorithmic Arbitration ─────────────────────────────────
def arbitrate(question, source_units, extractions):
    """
    Compare source-level answers:
    - Group supported answers by normalized match
    - Consensus: only one answer group → return it
    - Conflict: multiple groups → build position summary for LLM arbitration
    - Insufficient: no supported answers → fallback

    Returns: (final_answer, decision_type, details)
    """
    # Filter to supported answers only
    supported = []
    for i, ext in enumerate(extractions):
        if ext["status"] == "supported" and ext["answer"].lower() not in ("unknown", "i don't know", ""):
            supported.append((i, ext))

    if not supported:
        # Fallback: use any non-unknown answer, or return unknown
        any_answer = [ext for ext in extractions if ext["answer"].lower() not in ("unknown", "i don't know", "")]
        if any_answer:
            return any_answer[0]["answer"], "insufficient_fallback", {}
        return "unknown", "no_evidence", {}

    # Group by normalized answer
    groups = {}  # normalized_answer → [(source_idx, extraction)]
    for source_idx, ext in supported:
        norm = normalize_answer(ext["answer"])
        # Check if this matches any existing group
        matched_key = None
        for existing_key in groups:
            if answers_match(norm, existing_key):
                matched_key = existing_key
                break
        if matched_key:
            groups[matched_key].append((source_idx, ext))
        else:
            groups[norm] = [(source_idx, ext)]

    # Consensus check
    if len(groups) == 1:
        key = list(groups.keys())[0]
        best = groups[key][0][1]  # first source's extraction
        return best["answer"], "consensus", {"num_sources": len(groups[key])}

    # Conflict: build position summary and call LLM for arbitration
    positions_text = build_positions_text(groups, source_units)
    final_answer = call_llm(
        ARBITRATION_PROMPT.format(question=question, positions_text=positions_text),
        max_tokens=100
    )
    time.sleep(0.05)

    return final_answer, "conflict_arbitrated", {
        "num_positions": len(groups),
        "position_sizes": {k: len(v) for k, v in groups.items()}
    }


def build_positions_text(groups, source_units):
    """Build structured position text for arbitration prompt."""
    lines = []
    # Sort: fewer sources first (minority position first to counter majority bias)
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]))

    for pos_idx, (answer_key, members) in enumerate(sorted_groups, 1):
        num_sources = len(members)
        lines.append(f"== Position {pos_idx}: \"{members[0][1]['answer']}\" ({num_sources} independent source{'s' if num_sources > 1 else ''}) ==")
        for source_idx, ext in members:
            unit = source_units[source_idx]
            conf = ext.get("confidence", "unknown")
            support = ext.get("support", "N/A")
            lines.append(f"  Source {source_idx+1} (confidence: {conf}, based on {unit['cluster_size']} original documents):")
            lines.append(f"    Evidence: \"{support}\"")
        lines.append("")

    return "\n".join(lines)


# ─── Naive RAG baseline (for comparison) ──────────────────────────────────────
NAIVE_RAG_PROMPT = """Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

Question: {question}

{doc_texts}

Answer:"""


def naive_rag(question, documents):
    doc_texts = "\n\n".join(f"[Document {k+1}]\n{d['text']}" for k, d in enumerate(documents))
    answer = call_llm(NAIVE_RAG_PROMPT.format(question=question, doc_texts=doc_texts), max_tokens=64)
    time.sleep(0.05)
    return answer


# ─── Full Pipeline ────────────────────────────────────────────────────────────
def process_sample(sample, idx, log_path=None):
    """Run full Module 1 + Module 2 pipeline on a single sample."""
    question = sample["question"]
    documents = sample["documents"]
    gold_answers = sample["gold_answers"]

    # Naive RAG
    naive_answer = naive_rag(question, documents)
    naive_correct = normalized_exact_match(naive_answer, gold_answers)

    # Module 1: Clustering
    clusters, m1_api_calls = module1_cluster(question, documents)

    # Module 2 Step 1: Build source units
    source_units = build_source_units(question, documents, clusters)

    # Module 2 Step 2: Source-isolated extraction
    extractions = extract_source_answers(question, source_units)
    m2_extraction_calls = len(source_units)

    # Module 2 Step 3+4: Arbitration
    final_answer, decision_type, details = arbitrate(question, source_units, extractions)
    m2_arbitration_calls = 1 if decision_type == "conflict_arbitrated" else 0

    final_correct = normalized_exact_match(final_answer, gold_answers)

    # CPR-source: proportion of supported sources whose answer matches gold
    cpr_source = compute_cpr_source(extractions, gold_answers)

    result = {
        "idx": idx,
        "question": question,
        "gold_answers": gold_answers,
        "naive_answer": naive_answer,
        "naive_correct": naive_correct,
        "num_clusters": len(clusters),
        "cluster_sizes": [len(c) for c in clusters],
        "num_source_units": len(source_units),
        "extractions": extractions,
        "decision_type": decision_type,
        "final_answer": final_answer,
        "final_correct": final_correct,
        "cpr_source": cpr_source,
        "api_calls": {
            "module1": m1_api_calls,
            "extraction": m2_extraction_calls,
            "arbitration": m2_arbitration_calls,
            "naive": 1,
            "total": m1_api_calls + m2_extraction_calls + m2_arbitration_calls + 1
        }
    }

    log(f"  [{idx}] naive={naive_answer[:30]}({'✓' if naive_correct else '✗'}) "
        f"| m2={final_answer[:30]}({'✓' if final_correct else '✗'}) "
        f"| type={decision_type} | clusters={len(clusters)} | cpr_src={cpr_source:.2f}",
        log_path)

    return result


def compute_cpr_source(extractions, gold_answers):
    """CPR-source: fraction of supported sources that match gold answer."""
    supported = [ext for ext in extractions if ext["status"] == "supported"
                 and ext["answer"].lower() not in ("unknown", "i don't know", "")]
    if not supported:
        return 0.0
    correct_sources = sum(1 for ext in supported
                         if normalized_exact_match(ext["answer"], gold_answers))
    return correct_sources / len(supported)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Module 2: Source-Isolated Arbitration")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset jsonl")
    parser.add_argument("--tag", type=str, default="v4", help="Experiment tag for output naming")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples (0=all)")
    args = parser.parse_args()

    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    output_path = os.path.join(RESULTS_DIR, f"results_{args.tag}_{dataset_name}.jsonl")
    ckpt_path = os.path.join(RESULTS_DIR, f"ckpt_{args.tag}_{dataset_name}.jsonl")
    log_path = os.path.join(LOGS_DIR, f"{args.tag}_{dataset_name}.log")

    # Load data
    with open(args.dataset, "r", encoding="utf-8") as f:
        all_data = [json.loads(line.strip()) for line in f]

    if args.limit > 0:
        all_data = all_data[:args.limit]

    # Resume from checkpoint
    completed = set()
    results = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                completed.add(r["idx"])
                results.append(r)
        log(f"Resumed from checkpoint: {len(completed)} completed", log_path)

    log(f"=== Module 2 (V4) Source-Isolated Arbitration ===", log_path)
    log(f"Dataset: {args.dataset} ({len(all_data)} samples)", log_path)
    log(f"Output: {output_path}", log_path)
    log("", log_path)

    # Process
    for idx, sample in enumerate(all_data):
        if idx in completed:
            continue

        result = process_sample(sample, idx, log_path)
        results.append(result)

        # Save checkpoint
        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Progress stats every 10 samples
        if (idx + 1) % 10 == 0 or idx == len(all_data) - 1:
            done = [r for r in results]
            naive_acc = sum(r["naive_correct"] for r in done) / len(done) * 100
            m2_acc = sum(r["final_correct"] for r in done) / len(done) * 100
            avg_cpr = sum(r["cpr_source"] for r in done) / len(done) * 100
            log(f"\n--- Progress: {len(done)}/{len(all_data)} ---", log_path)
            log(f"  Naive ACC: {naive_acc:.1f}%", log_path)
            log(f"  M2 ACC:    {m2_acc:.1f}% (Δ={m2_acc-naive_acc:+.1f}pp)", log_path)
            log(f"  CPR-src:   {avg_cpr:.1f}%", log_path)
            consensus_count = sum(1 for r in done if r["decision_type"] == "consensus")
            conflict_count = sum(1 for r in done if r["decision_type"] == "conflict_arbitrated")
            log(f"  Decisions: consensus={consensus_count}, conflict={conflict_count}, other={len(done)-consensus_count-conflict_count}\n", log_path)

    # Final output
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final summary
    naive_acc = sum(r["naive_correct"] for r in results) / len(results) * 100
    m2_acc = sum(r["final_correct"] for r in results) / len(results) * 100
    avg_cpr = sum(r["cpr_source"] for r in results) / len(results) * 100
    avg_calls = sum(r["api_calls"]["total"] for r in results) / len(results)

    log(f"\n{'='*60}", log_path)
    log(f"FINAL RESULTS — {args.tag} on {dataset_name} (N={len(results)})", log_path)
    log(f"{'='*60}", log_path)
    log(f"  Naive ACC:     {naive_acc:.1f}%", log_path)
    log(f"  Module 2 ACC:  {m2_acc:.1f}% (Δ={m2_acc-naive_acc:+.1f}pp)", log_path)
    log(f"  CPR-source:    {avg_cpr:.1f}%", log_path)
    log(f"  Avg API calls: {avg_calls:.1f}", log_path)
    log(f"{'='*60}", log_path)


if __name__ == "__main__":
    main()
