"""
Module 2 V4 adapted for RAMDocs benchmark.
Key differences from module2/source_isolated_arbitration.py:
  1. Multi-answer output: outputs ALL well-supported answers (for ambiguity)
  2. Misinfo filtering: excludes answers with weak/suspicious evidence
  3. Strict accuracy: must contain all gold_answers AND not contain wrong_answers
  4. Field adaptation: wrong_answers (plural), disambig_entity
"""
import json, time, os, re, sys
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
MAX_WORKERS = 5
JACCARD_THRESHOLD = 0.40

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

STOPWORDS = set("a an the is was were be been being have has had do does did will would shall should may might can could and or but if then else when at by for with about against between through during before after above below to from up down in out on off over under again further once here there all each every both few more most other some such no nor not only own same so than too very".split())

# ─── Prompts ──────────────────────────────────────────────────────────────────
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
D2=Partial dependency
D3=High dependency (paraphrase/copy of same source)

Output ONLY valid JSON:
{{"level":"D0/D1/D2/D3","reasoning":"one sentence"}}"""

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
- "ambiguous": the document is vague or contradictory
- Keep answer short (a few words)"""

MULTI_ANSWER_ARBITRATION_PROMPT = """You are analyzing answers from multiple independent sources to a question.
Some sources may provide DIFFERENT VALID answers (because the question is ambiguous).
Some sources may provide WRONG answers (misinformation).

Question: {question}

{positions_text}

Instructions:
- If multiple positions have strong, specific evidence, they may ALL be correct (ambiguous question)
- Exclude answers that seem factually wrong, vague, or poorly supported
- List ALL answers that appear well-supported by their evidence
- Do NOT include answers that seem like misinformation

Output ALL correct answers as a comma-separated list. If only one answer is correct, output just that one.
Answer:"""

NAIVE_RAG_PROMPT = """Answer the following question based ONLY on the provided documents. Give a short, direct answer. If the question is ambiguous and multiple answers are valid, list all valid answers separated by commas.

Question: {question}

{doc_texts}

Answer:"""


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
                time.sleep(wait)
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
    if len(shorter) >= 3 and shorter in longer:
        return True
    return False


# ─── Evaluation: MADAM-RAG strict accuracy ────────────────────────────────────
def strict_accuracy(prediction, gold_answers, wrong_answers):
    """
    MADAM-RAG accuracy: prediction must contain ALL gold answers
    and must NOT contain any wrong answer.
    """
    pred_norm = normalize_answer(prediction)

    # Check all gold answers are present
    all_gold_present = True
    for gold in gold_answers:
        gold_norm = normalize_answer(gold)
        if not answers_match(pred_norm, gold_norm) and not answers_match(gold_norm, pred_norm):
            all_gold_present = False
            break

    # Check no wrong answer is present
    no_wrong_present = True
    for wrong in (wrong_answers or []):
        wrong_norm = normalize_answer(wrong)
        if wrong_norm and (answers_match(pred_norm, wrong_norm) or answers_match(wrong_norm, pred_norm)):
            no_wrong_present = False
            break

    return all_gold_present and no_wrong_present


# ─── Module 1: Independence Detection ────────────────────────────────────────
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


def module1_cluster(question, documents):
    n = len(documents)
    texts = [d["text"] for d in documents]

    candidate_pairs = []
    for i, j in combinations(range(n), 2):
        jac = jaccard(texts[i], texts[j])
        if jac >= JACCARD_THRESHOLD:
            candidate_pairs.append((i, j, jac))

    uf = UnionFind(n)
    api_calls = 0

    if candidate_pairs:
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

    clusters = uf.clusters(n)
    return clusters, api_calls


# ─── Module 2: Source-Isolated Extraction + Multi-Answer Arbitration ──────────
def build_source_units(documents, clusters):
    source_units = []
    for cluster in clusters:
        representative_idx = min(cluster)
        source_units.append({
            "representative_idx": representative_idx,
            "representative_text": documents[representative_idx]["text"],
            "cluster_indices": cluster,
            "cluster_size": len(cluster),
        })
    return source_units


def extract_source_answers(question, source_units):
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

    for i, ext in enumerate(extractions):
        if ext is None:
            extractions[i] = {"answer": "unknown", "status": "unsupported", "support": "", "confidence": "low"}
        else:
            ext.setdefault("answer", "unknown")
            ext.setdefault("status", "ambiguous")
            ext.setdefault("support", "")
            ext.setdefault("confidence", "low")

    return extractions


def arbitrate_multi_answer(question, source_units, extractions):
    """
    Multi-answer arbitration for RAMDocs:
    - If all supported sources agree → single answer
    - If different sources give different well-supported answers → output ALL (ambiguity)
    - Filter out poorly supported answers (potential misinfo)
    """
    supported = []
    for i, ext in enumerate(extractions):
        if ext["status"] == "supported" and ext["answer"].lower() not in ("unknown", "i don't know", ""):
            supported.append((i, ext))

    if not supported:
        any_answer = [ext for ext in extractions if ext["answer"].lower() not in ("unknown", "i don't know", "")]
        if any_answer:
            return any_answer[0]["answer"], "insufficient_fallback", {}
        return "unknown", "no_evidence", {}

    # Group by normalized answer
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

    # Single answer group → consensus
    if len(groups) == 1:
        key = list(groups.keys())[0]
        best = groups[key][0][1]
        return best["answer"], "consensus", {"num_sources": len(groups[key])}

    # Multiple answer groups → use LLM to determine which are valid vs misinfo
    positions_text = build_positions_text(groups, source_units)
    final_answer = call_llm(
        MULTI_ANSWER_ARBITRATION_PROMPT.format(question=question, positions_text=positions_text),
        max_tokens=150
    )
    return final_answer, "multi_answer_arbitrated", {
        "num_positions": len(groups),
        "position_sizes": {k: len(v) for k, v in groups.items()}
    }


def build_positions_text(groups, source_units):
    lines = []
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))  # most supported first
    for pos_idx, (answer_key, members) in enumerate(sorted_groups, 1):
        num_sources = len(members)
        lines.append(f"== Position {pos_idx}: \"{members[0][1]['answer']}\" ({num_sources} independent source{'s' if num_sources > 1 else ''}) ==")
        for source_idx, ext in members:
            unit = source_units[source_idx]
            conf = ext.get("confidence", "unknown")
            support = ext.get("support", "N/A")
            lines.append(f"  Source {source_idx+1} (confidence: {conf}, based on {unit['cluster_size']} documents):")
            lines.append(f"    Evidence: \"{support}\"")
        lines.append("")
    return "\n".join(lines)


def naive_rag(question, documents):
    doc_texts = "\n\n".join(f"[Document {k+1}]\n{d['text']}" for k, d in enumerate(documents))
    answer = call_llm(NAIVE_RAG_PROMPT.format(question=question, doc_texts=doc_texts), max_tokens=100)
    return answer


# ─── Process Sample ───────────────────────────────────────────────────────────
def process_sample(sample, idx, log_path=None):
    question = sample["question"]
    documents = sample["documents"]
    gold_answers = sample["gold_answers"]
    wrong_answers = sample.get("wrong_answers", [])

    # Naive RAG
    naive_answer = naive_rag(question, documents)
    naive_correct = strict_accuracy(naive_answer, gold_answers, wrong_answers)

    # Module 1
    clusters, m1_calls = module1_cluster(question, documents)

    # Module 2
    source_units = build_source_units(documents, clusters)
    extractions = extract_source_answers(question, source_units)
    final_answer, decision_type, details = arbitrate_multi_answer(question, source_units, extractions)

    final_correct = strict_accuracy(final_answer, gold_answers, wrong_answers)

    # Check if wrong answer is in output
    contains_wrong = False
    for w in wrong_answers:
        w_norm = normalize_answer(w)
        f_norm = normalize_answer(final_answer)
        if w_norm and (answers_match(f_norm, w_norm) or answers_match(w_norm, f_norm)):
            contains_wrong = True
            break

    result = {
        "idx": idx,
        "question": question,
        "gold_answers": gold_answers,
        "wrong_answers": wrong_answers,
        "naive_answer": naive_answer,
        "naive_correct": naive_correct,
        "num_clusters": len(clusters),
        "num_source_units": len(source_units),
        "decision_type": decision_type,
        "final_answer": final_answer,
        "final_correct": final_correct,
        "contains_wrong": contains_wrong,
        "api_calls": m1_calls + len(source_units) + (1 if "arbitrated" in decision_type else 0) + 1,
    }

    log(f"  [{idx}] naive={'✓' if naive_correct else '✗'} | v4={'✓' if final_correct else '✗'} "
        f"| type={decision_type} | clusters={len(clusters)} | wrong_in_output={contains_wrong}",
        log_path)

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Module 2 V4 on RAMDocs")
    parser.add_argument("--dataset", type=str, default="RAMDocs/RAMDocs_test.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset_name = "ramdocs"
    output_path = os.path.join(RESULTS_DIR, f"results_v4_{dataset_name}.jsonl")
    ckpt_path = os.path.join(RESULTS_DIR, f"ckpt_v4_{dataset_name}.jsonl")
    log_path = os.path.join(LOGS_DIR, f"v4_{dataset_name}.log")

    with open(args.dataset, "r", encoding="utf-8") as f:
        all_data = [json.loads(line.strip()) for line in f]

    if args.limit > 0:
        all_data = all_data[:args.limit]

    # Resume
    completed = set()
    results = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                completed.add(r["idx"])
                results.append(r)
        log(f"Resumed: {len(completed)} completed", log_path)

    log(f"=== Module 2 V4 on RAMDocs (N={len(all_data)}) ===", log_path)
    log(f"Output: {output_path}\n", log_path)

    for idx, sample in enumerate(all_data):
        if idx in completed:
            continue
        result = process_sample(sample, idx, log_path)
        results.append(result)

        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if (idx + 1) % 10 == 0 or idx == len(all_data) - 1:
            done = results
            naive_acc = sum(r["naive_correct"] for r in done) / len(done) * 100
            v4_acc = sum(r["final_correct"] for r in done) / len(done) * 100
            wrong_rate = sum(r["contains_wrong"] for r in done) / len(done) * 100
            log(f"\n--- Progress: {len(done)}/{len(all_data)} ---", log_path)
            log(f"  Naive ACC (strict): {naive_acc:.1f}%", log_path)
            log(f"  V4 ACC (strict):    {v4_acc:.1f}% (Δ={v4_acc-naive_acc:+.1f}pp)", log_path)
            log(f"  Wrong answer rate:  {wrong_rate:.1f}%\n", log_path)

    # Final
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    naive_acc = sum(r["naive_correct"] for r in results) / len(results) * 100
    v4_acc = sum(r["final_correct"] for r in results) / len(results) * 100
    wrong_rate = sum(r["contains_wrong"] for r in results) / len(results) * 100
    avg_calls = sum(r["api_calls"] for r in results) / len(results)

    log(f"\n{'='*60}", log_path)
    log(f"FINAL RESULTS — V4 on RAMDocs (N={len(results)})", log_path)
    log(f"{'='*60}", log_path)
    log(f"  Naive ACC (strict):  {naive_acc:.1f}%", log_path)
    log(f"  V4 ACC (strict):     {v4_acc:.1f}% (Δ={v4_acc-naive_acc:+.1f}pp)", log_path)
    log(f"  Wrong answer rate:   {wrong_rate:.1f}%", log_path)
    log(f"  Avg API calls:       {avg_calls:.1f}", log_path)
    log(f"{'='*60}", log_path)
    log(f"\nReference: MADAM-RAG GPT-4o-mini on RAMDocs = 28.0%", log_path)


if __name__ == "__main__":
    main()
