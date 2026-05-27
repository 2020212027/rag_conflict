"""
Experiment 1: How strong are simple baselines?
Compare 4 simple baselines against our layered method (+13.0pp on amp_8).

Baselines:
A) Prompt-only: Add "beware of duplicates" instruction to RAG prompt
B) High-Jaccard dedup: Jaccard >= 0.95 trivial dedup (zero LLM cost)
C) Top-3 truncation: Only use first 3 documents
D) Random-to-6: Randomly keep 6 documents (simulate "no redundancy")

Runs on both amp_8 and clean datasets.
"""
import json, time, os, re, sys, random
from openai import OpenAI

MODEL = "gpt-4o-mini"
random.seed(42)

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

STOPWORDS = set("a an the is was were be been being have has had do does did will would shall should may might can could and or but if then else when at by for with about against between through during before after above below to from up down in out on off over under again further once here there all each every both few more most other some such no nor not only own same so than too very".split())

RAG_PROMPT = """Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

Question: {question}

{doc_texts}

Answer:"""

RAG_PROMPT_AWARE = """Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

IMPORTANT: Some retrieved documents may be near-duplicates or paraphrases from the same source. Do NOT treat repetition as stronger evidence. Judge the answer based on the quality and diversity of independent sources, not the number of documents supporting a claim.

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


def call_llm(prompt, max_tokens=64):
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 * (attempt + 1)
            if attempt < 4:
                time.sleep(wait)
    return ""


def normalized_exact_match(prediction, gold_answers):
    pred_lower = prediction.lower().strip().rstrip(".")
    for gold in gold_answers:
        gold_lower = gold.lower().strip()
        if gold_lower in pred_lower or pred_lower in gold_lower:
            return True
    return False


def format_docs(documents, max_chars=500):
    return "\n\n".join(f"[Document {k+1}]\n{d['text'][:max_chars]}" for k, d in enumerate(documents))


def baseline_a_prompt_only(question, documents):
    """Baseline A: Same docs, but prompt warns about duplicates."""
    doc_texts = format_docs(documents)
    answer = call_llm(RAG_PROMPT_AWARE.format(question=question, doc_texts=doc_texts))
    return answer, len(documents), "prompt_only"


def baseline_b_jaccard_dedup(question, documents):
    """Baseline B: Remove docs with Jaccard >= 0.95 (trivial near-exact dedup)."""
    kept = [0]
    for i in range(1, len(documents)):
        is_dup = False
        for j in kept:
            if jaccard(documents[i]["text"], documents[j]["text"]) >= 0.95:
                is_dup = True
                break
        if not is_dup:
            kept.append(i)
    sel = [documents[i] for i in kept]
    doc_texts = format_docs(sel)
    answer = call_llm(RAG_PROMPT.format(question=question, doc_texts=doc_texts))
    return answer, len(sel), "jaccard_095"


def baseline_c_top3(question, documents):
    """Baseline C: Only use top-3 documents."""
    sel = documents[:3]
    doc_texts = format_docs(sel)
    answer = call_llm(RAG_PROMPT.format(question=question, doc_texts=doc_texts))
    return answer, 3, "top3"


def baseline_d_random6(question, documents):
    """Baseline D: Randomly keep 6 documents."""
    if len(documents) <= 6:
        sel = documents
    else:
        indices = random.sample(range(len(documents)), 6)
        sel = [documents[i] for i in sorted(indices)]
    doc_texts = format_docs(sel)
    answer = call_llm(RAG_PROMPT.format(question=question, doc_texts=doc_texts))
    return answer, len(sel), "random6"


def process_sample(sample, idx):
    question = sample["question"]
    documents = sample["documents"]
    gold_answers = sample["gold_answers"]
    wrong_answer = sample.get("wrong_answer", "")

    # Naive RAG (all docs, standard prompt) - as reference
    doc_texts = format_docs(documents)
    naive_answer = call_llm(RAG_PROMPT.format(question=question, doc_texts=doc_texts))
    time.sleep(0.3)

    # Baseline A: Prompt-only
    ans_a, kept_a, _ = baseline_a_prompt_only(question, documents)
    time.sleep(0.3)

    # Baseline B: Jaccard >= 0.95 dedup
    ans_b, kept_b, _ = baseline_b_jaccard_dedup(question, documents)
    time.sleep(0.3)

    # Baseline C: Top-3
    ans_c, kept_c, _ = baseline_c_top3(question, documents)
    time.sleep(0.3)

    # Baseline D: Random-6
    ans_d, kept_d, _ = baseline_d_random6(question, documents)
    time.sleep(0.3)

    return {
        "idx": idx,
        "question": question,
        "gold_answers": gold_answers,
        "wrong_answer": wrong_answer,
        "num_docs": len(documents),
        "naive_answer": naive_answer,
        "naive_correct": normalized_exact_match(naive_answer, gold_answers),
        "baseline_a_answer": ans_a,
        "baseline_a_correct": normalized_exact_match(ans_a, gold_answers),
        "baseline_a_kept": kept_a,
        "baseline_b_answer": ans_b,
        "baseline_b_correct": normalized_exact_match(ans_b, gold_answers),
        "baseline_b_kept": kept_b,
        "baseline_c_answer": ans_c,
        "baseline_c_correct": normalized_exact_match(ans_c, gold_answers),
        "baseline_c_kept": kept_c,
        "baseline_d_answer": ans_d,
        "baseline_d_correct": normalized_exact_match(ans_d, gold_answers),
        "baseline_d_kept": kept_d,
    }


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("amp8", "clean"):
        print("Usage: py exp1_baselines.py [amp8|clean]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "amp8":
        data_path = r"d:\pythonProject\dataset_amp_8.jsonl"
    else:
        data_path = r"d:\pythonProject\dataset_clean.jsonl"

    log_path = f"d:\\pythonProject\\exp1_{mode}_run.log"
    ckpt_path = f"d:\\pythonProject\\exp1_{mode}_checkpoint.jsonl"
    output_path = f"d:\\pythonProject\\exp1_{mode}_results.jsonl"

    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f if l.strip()]

    # Load checkpoint
    done = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["idx"]] = r

    print(f"Exp1 Baselines - {mode.upper()}", flush=True)
    print(f"Loaded {len(data)} samples, {len(done)} done, {len(data)-len(done)} remaining", flush=True)

    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'='*60}\nExp1 Baselines - {mode.upper()}\n{'='*60}\n")

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
            n_done = len(done)
            nem_naive = sum(1 for r in done.values() if r["naive_correct"]) / n_done * 100
            nem_a = sum(1 for r in done.values() if r["baseline_a_correct"]) / n_done * 100
            nem_b = sum(1 for r in done.values() if r["baseline_b_correct"]) / n_done * 100
            nem_c = sum(1 for r in done.values() if r["baseline_c_correct"]) / n_done * 100
            nem_d = sum(1 for r in done.values() if r["baseline_d_correct"]) / n_done * 100
            remaining = len(data) - n_done
            eta = elapsed / max(1, n_done) * remaining / 60

            msg = (f"  [{n_done}/{len(data)}] "
                   f"Naive={nem_naive:.1f}% A={nem_a:.1f}% B={nem_b:.1f}% "
                   f"C={nem_c:.1f}% D={nem_d:.1f}% | "
                   f"{elapsed/60:.1f}min ETA~{eta:.0f}min")
            print(msg, flush=True)
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(msg + "\n")

    # Final summary
    all_r = list(done.values())
    n = len(all_r)
    nem_naive = sum(1 for r in all_r if r["naive_correct"]) / n * 100
    nem_a = sum(1 for r in all_r if r["baseline_a_correct"]) / n * 100
    nem_b = sum(1 for r in all_r if r["baseline_b_correct"]) / n * 100
    nem_c = sum(1 for r in all_r if r["baseline_c_correct"]) / n * 100
    nem_d = sum(1 for r in all_r if r["baseline_d_correct"]) / n * 100

    avg_kept_b = sum(r["baseline_b_kept"] for r in all_r) / n
    avg_kept_d = sum(r["baseline_d_kept"] for r in all_r) / n

    summary = f"""
{'='*60}
FINAL EXP1 BASELINES - {mode.upper()} (N={n})
{'='*60}
  Naive (all docs):         {nem_naive:.1f}%
  A) Prompt-only:           {nem_a:.1f}%  (Δ={nem_a-nem_naive:+.1f}pp)
  B) Jaccard>=0.95 dedup:   {nem_b:.1f}%  (Δ={nem_b-nem_naive:+.1f}pp, avg_kept={avg_kept_b:.1f})
  C) Top-3 only:            {nem_c:.1f}%  (Δ={nem_c-nem_naive:+.1f}pp)
  D) Random-6:              {nem_d:.1f}%  (Δ={nem_d-nem_naive:+.1f}pp, avg_kept={avg_kept_d:.1f})
  [Reference] Layered:      Δ=+13.0pp (amp8) / -4.2pp (clean)
"""
    print(summary, flush=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(summary)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in sorted(all_r, key=lambda x: x["idx"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
