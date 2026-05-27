"""
Baseline experiments for Module 2 comparison:
  V3: Document-level isolated answers (no Module 1 clustering)
      → Each document answers independently → majority vote / arbitration
  V5: Random grouping isolated answers (random clusters, no dependency detection)
      → Documents randomly grouped → each group answers → arbitration

Both use the same arbitration logic as V4, but differ in how "sources" are defined:
  V3: each document = 1 source (15 sources for amp8)
  V5: random groups of ~3-4 docs = 1 source (~4-5 sources)
  V4: dependency clusters = 1 source (Module 1 output)
"""
import json, time, os, re, sys, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL = "gpt-4o-mini"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
MAX_WORKERS = 5

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

# ─── Prompts ──────────────────────────────────────────────────────────────────
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
- Evaluate based on evidence QUALITY and SPECIFICITY, not quantity
- Prefer answers with direct, specific supporting evidence
- If one position cites vague or tangential evidence, weigh it less

Provide your final answer (short, direct):"""

NAIVE_RAG_PROMPT = """Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

Question: {question}

{doc_texts}

Answer:"""


# ─── Utilities ────────────────────────────────────────────────────────────────
def log(msg, log_path=None):
    print(msg, flush=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


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


# ─── Source Construction ──────────────────────────────────────────────────────
def build_sources_v3(documents):
    """V3: Each document is its own source."""
    sources = []
    for i, doc in enumerate(documents):
        sources.append({
            "representative_text": doc["text"],
            "cluster_indices": [i],
            "cluster_size": 1,
        })
    return sources


def build_sources_v5(documents, num_groups=4, seed=42):
    """V5: Random grouping into num_groups clusters."""
    rng = random.Random(seed)
    indices = list(range(len(documents)))
    rng.shuffle(indices)

    # Split into roughly equal groups
    groups = [[] for _ in range(num_groups)]
    for i, idx in enumerate(indices):
        groups[i % num_groups].append(idx)

    sources = []
    for group in groups:
        if not group:
            continue
        # Representative: first doc in group (random, since shuffled)
        rep_idx = group[0]
        sources.append({
            "representative_text": documents[rep_idx]["text"],
            "cluster_indices": group,
            "cluster_size": len(group),
        })
    return sources


# ─── Extraction + Arbitration (shared with V4) ───────────────────────────────
def extract_source_answers(question, sources):
    """Each source independently answers the question (concurrent)."""
    def extract_one(idx, source):
        prompt = SOURCE_EXTRACTION_PROMPT.format(
            question=question,
            source_text=source["representative_text"]
        )
        result = call_llm(prompt, max_tokens=150, expect_json=True)
        if not result:
            result = {"answer": "unknown", "status": "unsupported", "support": "", "confidence": "low"}
        return idx, result

    extractions = [None] * len(sources)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(extract_one, i, s) for i, s in enumerate(sources)]
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


def arbitrate(question, sources, extractions):
    """Compare source-level answers and resolve conflicts."""
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

    if len(groups) == 1:
        key = list(groups.keys())[0]
        best = groups[key][0][1]
        return best["answer"], "consensus", {"num_sources": len(groups[key])}

    # Conflict: LLM arbitration
    positions_text = build_positions_text(groups, sources)
    final_answer = call_llm(
        ARBITRATION_PROMPT.format(question=question, positions_text=positions_text),
        max_tokens=100
    )
    return final_answer, "conflict_arbitrated", {
        "num_positions": len(groups),
        "position_sizes": {k: len(v) for k, v in groups.items()}
    }


def build_positions_text(groups, sources):
    lines = []
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]))
    for pos_idx, (answer_key, members) in enumerate(sorted_groups, 1):
        num_sources = len(members)
        lines.append(f"== Position {pos_idx}: \"{members[0][1]['answer']}\" ({num_sources} independent source{'s' if num_sources > 1 else ''}) ==")
        for source_idx, ext in members:
            unit = sources[source_idx]
            conf = ext.get("confidence", "unknown")
            support = ext.get("support", "N/A")
            lines.append(f"  Source {source_idx+1} (confidence: {conf}, based on {unit['cluster_size']} original documents):")
            lines.append(f"    Evidence: \"{support}\"")
        lines.append("")
    return "\n".join(lines)


def naive_rag(question, documents):
    doc_texts = "\n\n".join(f"[Document {k+1}]\n{d['text']}" for k, d in enumerate(documents))
    answer = call_llm(NAIVE_RAG_PROMPT.format(question=question, doc_texts=doc_texts), max_tokens=64)
    return answer


# ─── Process Sample ───────────────────────────────────────────────────────────
def process_sample(sample, idx, mode, log_path=None):
    question = sample["question"]
    documents = sample["documents"]
    gold_answers = sample["gold_answers"]

    # Naive RAG
    naive_answer = naive_rag(question, documents)
    naive_correct = normalized_exact_match(naive_answer, gold_answers)

    # Build sources based on mode
    if mode == "v3":
        sources = build_sources_v3(documents)
    elif mode == "v5":
        sources = build_sources_v5(documents, num_groups=4)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Extract + Arbitrate
    extractions = extract_source_answers(question, sources)
    final_answer, decision_type, details = arbitrate(question, sources, extractions)
    final_correct = normalized_exact_match(final_answer, gold_answers)

    # CPR-source
    supported_ext = [ext for ext in extractions if ext["status"] == "supported"
                     and ext["answer"].lower() not in ("unknown", "i don't know", "")]
    cpr_source = 0.0
    if supported_ext:
        correct_sources = sum(1 for ext in supported_ext
                             if normalized_exact_match(ext["answer"], gold_answers))
        cpr_source = correct_sources / len(supported_ext)

    result = {
        "idx": idx,
        "question": question,
        "gold_answers": gold_answers,
        "naive_answer": naive_answer,
        "naive_correct": naive_correct,
        "num_sources": len(sources),
        "decision_type": decision_type,
        "final_answer": final_answer,
        "final_correct": final_correct,
        "cpr_source": cpr_source,
        "api_calls": len(sources) + (1 if decision_type == "conflict_arbitrated" else 0) + 1,
    }

    log(f"  [{idx}] naive={naive_answer[:30]}({'✓' if naive_correct else '✗'}) "
        f"| {mode}={final_answer[:30]}({'✓' if final_correct else '✗'}) "
        f"| type={decision_type} | sources={len(sources)} | cpr={cpr_source:.2f}",
        log_path)

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Baselines: V3 (doc-level) / V5 (random grouping)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["v3", "v5"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    tag = args.mode
    output_path = os.path.join(RESULTS_DIR, f"results_{tag}_{dataset_name}.jsonl")
    ckpt_path = os.path.join(RESULTS_DIR, f"ckpt_{tag}_{dataset_name}.jsonl")
    log_path = os.path.join(LOGS_DIR, f"{tag}_{dataset_name}.log")

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

    log(f"=== Baseline {tag.upper()} ===", log_path)
    log(f"Dataset: {args.dataset} ({len(all_data)} samples)", log_path)
    log(f"Output: {output_path}\n", log_path)

    for idx, sample in enumerate(all_data):
        if idx in completed:
            continue
        result = process_sample(sample, idx, args.mode, log_path)
        results.append(result)

        with open(ckpt_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if (idx + 1) % 10 == 0 or idx == len(all_data) - 1:
            done = results
            naive_acc = sum(r["naive_correct"] for r in done) / len(done) * 100
            m_acc = sum(r["final_correct"] for r in done) / len(done) * 100
            avg_cpr = sum(r["cpr_source"] for r in done) / len(done) * 100
            log(f"\n--- Progress: {len(done)}/{len(all_data)} ---", log_path)
            log(f"  Naive ACC: {naive_acc:.1f}%", log_path)
            log(f"  {tag.upper()} ACC: {m_acc:.1f}% (Δ={m_acc-naive_acc:+.1f}pp)", log_path)
            log(f"  CPR-src: {avg_cpr:.1f}%\n", log_path)

    # Final
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    naive_acc = sum(r["naive_correct"] for r in results) / len(results) * 100
    m_acc = sum(r["final_correct"] for r in results) / len(results) * 100
    avg_cpr = sum(r["cpr_source"] for r in results) / len(results) * 100
    avg_calls = sum(r["api_calls"] for r in results) / len(results)

    log(f"\n{'='*60}", log_path)
    log(f"FINAL RESULTS — {tag.upper()} on {dataset_name} (N={len(results)})", log_path)
    log(f"{'='*60}", log_path)
    log(f"  Naive ACC:     {naive_acc:.1f}%", log_path)
    log(f"  {tag.upper()} ACC:  {m_acc:.1f}% (Δ={m_acc-naive_acc:+.1f}pp)", log_path)
    log(f"  CPR-source:    {avg_cpr:.1f}%", log_path)
    log(f"  Avg API calls: {avg_calls:.1f}", log_path)
    log(f"{'='*60}", log_path)


if __name__ == "__main__":
    main()
