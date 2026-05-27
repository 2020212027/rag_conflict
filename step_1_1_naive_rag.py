"""
Step 1.1 - Naive RAG Baseline
Concatenate all documents, ask LLM to answer the question.
Run on clean + amp_8 only.
"""
import json, time
from openai import OpenAI

MODEL = "gpt-4o-mini"
CONDITIONS = ["clean", "amp_8"]
OUTPUT_PREFIX = "results_naive_rag"
LOG_PATH = r"d:\pythonProject\naive_rag_progress.log"


def log(message: str) -> None:
    print(message, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


open(LOG_PATH, "w", encoding="utf-8").close()

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)


def build_prompt(question: str, documents: list) -> str:
    doc_texts = "\n\n".join(
        f"[Document {i+1}]\n{d['text']}" for i, d in enumerate(documents)
    )
    return f"""Answer the following question based ONLY on the provided documents. Give a short, direct answer (a few words or a short phrase). If you cannot determine the answer, say "I don't know".

Question: {question}

{doc_texts}

Answer:"""


def call_llm(prompt: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    return ""


def normalized_exact_match(prediction: str, gold_answers: list) -> bool:
    pred_lower = prediction.lower().strip().rstrip(".")
    for gold in gold_answers:
        gold_lower = gold.lower().strip()
        if gold_lower in pred_lower or pred_lower in gold_lower:
            return True
    return False


for condition in CONDITIONS:
    input_path = rf"d:\pythonProject\dataset_{condition}.jsonl"
    output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"

    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    log(f"\n{'='*50}")
    log(f"Condition: {condition} | Samples: {len(samples)}")
    log(f"{'='*50}")

    results = []
    correct_count = 0
    wrong_answer_count = 0
    start = time.time()

    for idx, s in enumerate(samples):
        prompt = build_prompt(s["question"], s["documents"])
        prediction = call_llm(prompt)

        is_correct = normalized_exact_match(prediction, s["gold_answers"])
        matches_wrong = normalized_exact_match(prediction, [s["wrong_answer"]]) if s["wrong_answer"] else False

        if is_correct:
            correct_count += 1
        if matches_wrong:
            wrong_answer_count += 1

        results.append({
            "pilot_id": s["pilot_id"],
            "question": s["question"],
            "gold_answers": s["gold_answers"],
            "wrong_answer": s["wrong_answer"],
            "prediction": prediction,
            "is_correct": is_correct,
            "matches_wrong_answer": matches_wrong,
            "num_docs": s["num_total_docs"],
        })

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start
            em = correct_count / (idx + 1) * 100
            wr = wrong_answer_count / (idx + 1) * 100
            eta = elapsed / (idx + 1) * (len(samples) - idx - 1) / 60
            log(f"  [{idx+1}/{len(samples)}] EM={em:.1f}% | Wrong={wr:.1f}% | ETA={eta:.1f}min")

        time.sleep(0.1)

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    em_score = correct_count / len(samples) * 100
    wrong_rate = wrong_answer_count / len(samples) * 100
    elapsed = time.time() - start

    log(f"\n  FINAL: EM={em_score:.1f}% | Wrong Answer Rate={wrong_rate:.1f}%")
    log(f"  Time: {elapsed/60:.1f}min | Saved: {output_path}")

log(f"\n\n{'='*50}")
log("SUMMARY")
log(f"{'='*50}")
for condition in CONDITIONS:
    output_path = rf"d:\pythonProject\{OUTPUT_PREFIX}_{condition}.jsonl"
    results = [json.loads(l) for l in open(output_path, "r", encoding="utf-8") if l.strip()]
    em = sum(1 for r in results if r["is_correct"]) / len(results) * 100
    wr = sum(1 for r in results if r["matches_wrong_answer"]) / len(results) * 100
    log(f"  {condition:8s}: EM={em:.1f}% | Wrong={wr:.1f}% | N={len(results)}")
