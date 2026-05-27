"""
LLM-as-Judge evaluation for AmbigDocs results.
Reads existing checkpoint file, uses LLM to judge if prediction covers each gold answer.
Does NOT re-run inference — only re-evaluates existing predictions.
Supports resume via separate judge checkpoint.
"""
import json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

MODEL = "gpt-4o-mini"
MAX_WORKERS = 5

client = OpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)

JUDGE_PROMPT = """You are judging whether a prediction answers a question correctly.

Question: {question}
Gold answer: {gold_answer}
Prediction: {prediction}

Does the prediction contain or convey the same meaning as the gold answer?
- The prediction may list multiple answers separated by "|" or commas
- The gold answer may be expressed differently but mean the same thing
- Partial matches count if the core fact is present (e.g., "Bishop" matches "Bishop of the Methodist Episcopal Church")
- Numbers must match (e.g., "4,657" matches "4657" but not "4,419")

Reply ONLY "yes" or "no"."""


def call_llm(prompt, max_tokens=5):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                timeout=15,
            )
            return resp.choices[0].message.content.strip().lower()
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "no"


def judge_one_answer(question, gold_answer, prediction):
    """Judge if prediction covers one gold answer."""
    result = call_llm(JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        prediction=prediction
    ))
    return result.startswith("yes")


def judge_sample(record):
    """Judge all gold answers for one sample, return recall and strict."""
    question = record["question"]
    gold_answers = record["gold_answers"]

    # Judge naive
    naive_hits = 0
    for gold in gold_answers:
        if judge_one_answer(question, gold, record["naive_answer"]):
            naive_hits += 1
    naive_recall = naive_hits / len(gold_answers) if gold_answers else 1.0
    naive_strict = (naive_hits == len(gold_answers))

    # Judge v4
    v4_hits = 0
    for gold in gold_answers:
        if judge_one_answer(question, gold, record["final_answer"]):
            v4_hits += 1
    v4_recall = v4_hits / len(gold_answers) if gold_answers else 1.0
    v4_strict = (v4_hits == len(gold_answers))

    return {
        "idx": record["idx"],
        "naive_recall_llm": naive_recall,
        "naive_strict_llm": naive_strict,
        "v4_recall_llm": v4_recall,
        "v4_strict_llm": v4_strict,
        "naive_hits": naive_hits,
        "v4_hits": v4_hits,
        "num_gold": len(gold_answers),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="module2_ambigdocs/results/ckpt_v4_ambigdocs.jsonl")
    args = parser.parse_args()

    results_dir = os.path.dirname(args.ckpt)
    judge_ckpt = os.path.join(results_dir, "judge_ckpt_ambigdocs.jsonl")
    judge_output = os.path.join(results_dir, "judge_results_ambigdocs.jsonl")

    # Load predictions
    with open(args.ckpt, "r", encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f]
    print(f"Loaded {len(records)} predictions")

    # Resume
    completed = set()
    judge_results = []
    if os.path.exists(judge_ckpt):
        with open(judge_ckpt, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                completed.add(r["idx"])
                judge_results.append(r)
        print(f"Resumed: {len(completed)} judged")

    for record in records:
        if record["idx"] in completed:
            continue

        jr = judge_sample(record)
        judge_results.append(jr)

        with open(judge_ckpt, "a", encoding="utf-8") as f:
            f.write(json.dumps(jr) + "\n")

        if (len(judge_results)) % 20 == 0:
            naive_r = sum(r["naive_recall_llm"] for r in judge_results) / len(judge_results) * 100
            v4_r = sum(r["v4_recall_llm"] for r in judge_results) / len(judge_results) * 100
            naive_s = sum(r["naive_strict_llm"] for r in judge_results) / len(judge_results) * 100
            v4_s = sum(r["v4_strict_llm"] for r in judge_results) / len(judge_results) * 100
            print(f"[{len(judge_results)}] Naive R={naive_r:.1f}% V4 R={v4_r:.1f}% | "
                  f"Naive S={naive_s:.1f}% V4 S={v4_s:.1f}%")

    # Final
    with open(judge_output, "w", encoding="utf-8") as f:
        for r in judge_results:
            f.write(json.dumps(r) + "\n")

    naive_r = sum(r["naive_recall_llm"] for r in judge_results) / len(judge_results) * 100
    v4_r = sum(r["v4_recall_llm"] for r in judge_results) / len(judge_results) * 100
    naive_s = sum(r["naive_strict_llm"] for r in judge_results) / len(judge_results) * 100
    v4_s = sum(r["v4_strict_llm"] for r in judge_results) / len(judge_results) * 100

    print(f"\n{'='*60}")
    print(f"LLM Judge Results (N={len(judge_results)})")
    print(f"{'='*60}")
    print(f"  Naive Answer Recall (LLM): {naive_r:.1f}%")
    print(f"  V4 Answer Recall (LLM):    {v4_r:.1f}% (Δ={v4_r-naive_r:+.1f}pp)")
    print(f"  Naive Strict ACC (LLM):    {naive_s:.1f}%")
    print(f"  V4 Strict ACC (LLM):       {v4_s:.1f}% (Δ={v4_s-naive_s:+.1f}pp)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
