"""
Step 3 E2E - Clean dataset (confirm no false positive harm).
Same batch clustering as amp_8 run.
"""
import json, time, os
from openai import OpenAI

MODEL = "gpt-4o-mini"
CLEAN_PATH = r"d:\pythonProject\dataset_clean.jsonl"
LOG_PATH = r"d:\pythonProject\e2e_clean.log"
CKPT_PATH = r"d:\pythonProject\e2e_clean_checkpoint.jsonl"
OUTPUT_PATH = r"d:\pythonProject\results_e2e_full_clean.jsonl"

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)


def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def call_llm(prompt, max_tokens=300, expect_json=False):
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


CLUSTER_PROMPT = """Given a question and {n} document excerpts, identify which documents are PARAPHRASES of each other (from the same original source).

Question: {question}

{doc_list}

Group documents that appear to be paraphrases/copies of the same source into clusters.
Documents with genuinely different information should be in separate clusters.

RULES:
- Same topic + different wording/details = SEPARATE clusters
- Near-identical content, shared rare details, same structure = SAME cluster
- When in doubt, keep them separate

Output ONLY JSON array of clusters (each cluster is array of doc indices):
{{"clusters": [[0,3,5], [1,4], [2], [6], ...]}}"""

RAG_PROMPT = """Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

Question: {question}

{doc_texts}

Answer:"""


def normalized_exact_match(prediction, gold_answers):
    pred_lower = prediction.lower().strip().rstrip(".")
    for gold in gold_answers:
        gold_lower = gold.lower().strip()
        if gold_lower in pred_lower or pred_lower in gold_lower:
            return True
    return False


def cluster_documents(question, documents):
    n = min(len(documents), 10)
    doc_list = "\n".join(
        f"Doc {i}: {documents[i]['text'][:200]}" for i in range(n)
    )
    prompt = CLUSTER_PROMPT.format(question=question, n=n, doc_list=doc_list)
    result = call_llm(prompt, max_tokens=300, expect_json=True)
    if not result or "clusters" not in result:
        return list(range(len(documents))), 0
    clusters = result["clusters"]
    kept = set()
    for cluster in clusters:
        if cluster:
            kept.add(cluster[0])
    for i in range(n, len(documents)):
        kept.add(i)
    return sorted(kept), len(documents) - len(kept)


def process_sample(sample, idx):
    question = sample["question"]
    documents = sample["documents"]
    gold_answers = sample["gold_answers"]
    wrong_answer = sample.get("wrong_answer", "")

    # Naive
    all_idx = list(range(len(documents)))
    doc_texts = "\n\n".join(f"[Document {k+1}]\n{d['text'][:500]}" for k, d in enumerate(documents))
    naive_answer = call_llm(RAG_PROMPT.format(question=question, doc_texts=doc_texts), max_tokens=64)
    time.sleep(0.3)

    # Cluster + dedup
    kept_indices, num_removed = cluster_documents(question, documents)
    time.sleep(0.3)

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
        "naive_answer": naive_answer,
        "naive_correct": normalized_exact_match(naive_answer, gold_answers),
        "naive_wrong": normalized_exact_match(naive_answer, [wrong_answer]) if wrong_answer else False,
        "dedup_answer": dedup_answer,
        "dedup_correct": normalized_exact_match(dedup_answer, gold_answers),
        "dedup_wrong": normalized_exact_match(dedup_answer, [wrong_answer]) if wrong_answer else False,
    }


def main():
    if not os.path.exists(CKPT_PATH):
        open(LOG_PATH, "w", encoding="utf-8").close()
    log("=" * 60)
    log("Step 3 E2E - CLEAN dataset (false positive check)")
    log(f"Model: {MODEL}")
    log("=" * 60)

    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
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
    for idx, sample in enumerate(data):
        if idx in done:
            continue
        result = process_sample(sample, idx)
        done[idx] = result
        with open(CKPT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if len(done) % 10 == 0 or idx == len(data) - 1:
            elapsed = time.time() - t0
            n_em = sum(1 for r in done.values() if r["naive_correct"]) / len(done) * 100
            d_em = sum(1 for r in done.values() if r["dedup_correct"]) / len(done) * 100
            avg_rm = sum(r["num_removed"] for r in done.values()) / len(done)
            remaining = len(data) - len(done)
            speed = elapsed / max(1, len(done))
            eta_min = speed * remaining / 60
            log(f"  [{len(done)}/{len(data)}] NEM={n_em:.1f}->{d_em:.1f}% "
                f"rm={avg_rm:.1f} | {elapsed/60:.1f}min ETA~{eta_min:.0f}min")

    all_r = list(done.values())
    n_em = sum(1 for r in all_r if r["naive_correct"]) / len(all_r) * 100
    d_em = sum(1 for r in all_r if r["dedup_correct"]) / len(all_r) * 100
    avg_rm = sum(r["num_removed"] for r in all_r) / len(all_r)
    flipped_good = sum(1 for r in all_r if not r["naive_correct"] and r["dedup_correct"])
    flipped_bad = sum(1 for r in all_r if r["naive_correct"] and not r["dedup_correct"])

    log(f"\n{'='*60}")
    log(f"FINAL CLEAN (N={len(all_r)})")
    log(f"{'='*60}")
    log(f"  Naive NEM:  {n_em:.1f}%")
    log(f"  Dedup NEM:  {d_em:.1f}%")
    log(f"  Delta NEM:  {d_em-n_em:+.1f}pp")
    log(f"  Avg removed: {avg_rm:.1f}")
    log(f"  Flipped good: {flipped_good} | Flipped bad: {flipped_bad}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in sorted(all_r, key=lambda x: x["idx"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
