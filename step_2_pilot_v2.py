"""
Step 2 - Plan B Pilot v2: Expanded samples + Calibrated prompt
Changes from v1:
1. Use dataset_clean.jsonl to get 50-80 independent pairs (correct docs, same answer)
2. Add CALIBRATION section to dependence prompt
3. Keep amplified pairs from dataset_amp_8.jsonl for comparison
"""
import json
import time
import asyncio
from itertools import combinations
from openai import AsyncOpenAI

MODEL = "gpt-4o-mini"
CLEAN_PATH = r"d:\pythonProject\dataset_clean.jsonl"
AMP8_PATH = r"d:\pythonProject\dataset_amp_8.jsonl"
OUTPUT_EVIDENCE = r"d:\pythonProject\pilot_v2b_evidence.jsonl"
OUTPUT_PAIRS = r"d:\pythonProject\pilot_v2b_pairs.jsonl"
LOG_PATH = r"d:\pythonProject\pilot_v2b.log"
TARGET_INDEPENDENT_PAIRS = 80
TARGET_AMPLIFIED_PAIRS = 80
CONCURRENCY = 3

client = AsyncOpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1"
)
semaphore = asyncio.Semaphore(CONCURRENCY)


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ─── Prompts ───────────────────────────────────────────────────────────────────

EVIDENCE_PROMPT = """Given a question and ONE document, extract a structured evidence unit.

Question: {question}
Document: {doc_text}

Extract:
1. answer: What answer does this document suggest? (short, factual)
2. supporting_facts: Why does it support this answer? List specific reasons, NOT the answer itself.
3. specific_details: Unique numbers, dates, names, quotes, unusual phrases in this document.
4. evidence_span: The single most relevant sentence from the document.

Output ONLY valid JSON:
{{"answer": "...", "supporting_facts": ["...", "..."], "specific_details": ["...", "..."], "evidence_span": "..."}}"""

DEPENDENCE_PROMPT = """You are judging whether two evidence units are INDEPENDENT or DEPENDENT.

Both evidence units support the same answer to the same question.
Your job is NOT to judge whether the answer is correct.
Your job is to judge whether these two units provide independent information,
or whether they are likely copied, paraphrased, or derived from the same source.

IMPORTANT RULES:
- Two units supporting the same answer does NOT make them dependent.
- Focus on: Do they share the same supporting reasoning, the same rare details,
  the same wording, or the same errors?
- Generic facts shared by both are weak evidence of dependence.
- Rare details (specific numbers, dates, quotes, unusual names) shared by both
  are strong evidence of dependence.

CALIBRATION:
- If both documents describe the same well-known fact (e.g. "the capital of France is Paris")
  using DIFFERENT wording, DIFFERENT examples, and DIFFERENT supporting details,
  this is D0 or D1, NOT D3 or D4.
- D3/D4 requires: shared RARE details (unusual numbers, specific dates, same quotes,
  same named entities that are NOT part of the question/answer itself),
  OR near-identical sentence structure, OR shared errors.
- Ask yourself: "Could two people independently write these two passages?"
  If yes -> D0/D1. If unlikely -> D3/D4.

Question: {question}
Candidate answer: {answer}

Evidence Unit A:
  supporting_facts: {facts_a}
  specific_details: {details_a}
  evidence_span: {span_a}

Evidence Unit B:
  supporting_facts: {facts_b}
  specific_details: {details_b}
  evidence_span: {span_b}

Rate the dependence level:
- D0: Independent. Different information paths, no shared rare details.
- D1: Minimal overlap. Same answer, maybe one shared common fact.
- D2: Moderate overlap. Several shared supporting facts, but different details.
- D3: High overlap. Shared rare details or highly similar reasoning chain. Likely same source.
- D4: Near-duplicate. Same wording, same details, same errors. Almost certainly same source.

Output ONLY valid JSON:
{{"level": "D0/D1/D2/D3/D4", "score": 0.0, "key_evidence": "one sentence explanation"}}"""


# ─── API Calls ─────────────────────────────────────────────────────────────────

async def extract_evidence(question: str, doc_text: str) -> dict:
    async with semaphore:
        prompt = EVIDENCE_PROMPT.format(question=question, doc_text=doc_text[:2000])
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0, max_tokens=500
                    ),
                    timeout=30
                )
                content = resp.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(content)
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return {"answer": "", "supporting_facts": [], "specific_details": [], "evidence_span": "", "error": str(e)}
        return {"answer": "", "supporting_facts": [], "specific_details": [], "evidence_span": "", "error": "timeout"}


async def judge_dependence(question: str, answer: str, unit_a: dict, unit_b: dict) -> dict:
    async with semaphore:
        prompt = DEPENDENCE_PROMPT.format(
            question=question, answer=answer,
            facts_a=json.dumps(unit_a.get("supporting_facts", []), ensure_ascii=False),
            details_a=json.dumps(unit_a.get("specific_details", []), ensure_ascii=False),
            span_a=unit_a.get("evidence_span", ""),
            facts_b=json.dumps(unit_b.get("supporting_facts", []), ensure_ascii=False),
            details_b=json.dumps(unit_b.get("specific_details", []), ensure_ascii=False),
            span_b=unit_b.get("evidence_span", "")
        )
        level_to_score = {"D0": 0.0, "D1": 0.25, "D2": 0.5, "D3": 0.75, "D4": 1.0}
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0, max_tokens=200
                    ),
                    timeout=30
                )
                content = resp.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                result = json.loads(content)
                level = result.get("level", "D2")
                if result.get("score", 0.0) == 0.0:
                    result["score"] = level_to_score.get(level, 0.5)
                return result
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return {"level": "ERROR", "score": -1, "key_evidence": str(e)}
        return {"level": "ERROR", "score": -1, "key_evidence": "timeout after 3 retries"}


# ─── Data Selection ────────────────────────────────────────────────────────────

def select_independent_queries(clean_data: list, target_pairs: int) -> list:
    """Select queries from clean dataset that have enough correct docs for same-answer pairing."""
    candidates = []
    for idx, item in enumerate(clean_data):
        correct_docs = [d for d in item["documents"] if d["type"] == "correct"]
        # Estimate same-answer pairs: group by the doc's answer field
        answer_groups = {}
        for d in correct_docs:
            ans = d["answer"].lower().strip()
            answer_groups[ans] = answer_groups.get(ans, 0) + 1
        same_answer_pairs = sum(
            len(list(combinations(range(cnt), 2))) for cnt in answer_groups.values()
        )
        if same_answer_pairs >= 1:
            candidates.append((idx, same_answer_pairs, correct_docs))

    # Sort by same-answer pair count descending
    candidates.sort(key=lambda x: -x[1])
    selected = []
    total_pairs = 0
    for idx, num_pairs, docs in candidates:
        selected.append((idx, docs))
        total_pairs += num_pairs
        if total_pairs >= target_pairs * 2:  # over-select to account for extraction variance
            break
    return selected, total_pairs


def select_amplified_queries(amp8_data: list, target_pairs: int) -> list:
    """Select queries from amp_8 dataset for amplified pairs."""
    candidates = []
    for idx, item in enumerate(amp8_data):
        amp_docs = [d for d in item["documents"] if d["type"] in ("misinfo", "misinfo_amplified")]
        num_pairs = len(list(combinations(range(len(amp_docs)), 2)))
        if num_pairs >= 3:
            candidates.append((idx, num_pairs, amp_docs))

    candidates.sort(key=lambda x: -x[1])
    selected = []
    total_pairs = 0
    for idx, num_pairs, docs in candidates:
        selected.append((idx, docs))
        total_pairs += num_pairs
        if total_pairs >= target_pairs:
            break
    return selected, total_pairs


# ─── Main ──────────────────────────────────────────────────────────────────────

async def run_pilot_v2():
    open(LOG_PATH, "w", encoding="utf-8").close()
    log("=" * 60)
    log("Plan B Pilot v2: Expanded Samples + Calibrated Prompt")
    log(f"Model: {MODEL}, Concurrency: {CONCURRENCY}")
    log("=" * 60)

    # Load datasets
    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        clean_data = [json.loads(l) for l in f]
    with open(AMP8_PATH, "r", encoding="utf-8") as f:
        amp8_data = [json.loads(l) for l in f]

    # Select queries
    ind_selected, ind_total = select_independent_queries(clean_data, TARGET_INDEPENDENT_PAIRS)
    amp_selected, amp_total = select_amplified_queries(amp8_data, TARGET_AMPLIFIED_PAIRS)
    log(f"\nIndependent: {len(ind_selected)} queries, ~{ind_total} potential pairs")
    log(f"Amplified:   {len(amp_selected)} queries, ~{amp_total} potential pairs")

    all_evidence = []
    all_pairs = []

    # ─── Phase 1: Extract evidence for independent pairs ───
    log("\n--- Phase 1a: Evidence Extraction (Independent / Clean) ---")
    ind_evidence_by_query = {}

    for sel_idx, (data_idx, correct_docs) in enumerate(ind_selected):
        question = clean_data[data_idx]["question"]
        log(f"  [{sel_idx+1}/{len(ind_selected)}] Q: {question[:50]}... ({len(correct_docs)} correct docs)")

        tasks = [extract_evidence(question, doc["text"]) for doc in correct_docs]
        results = await asyncio.gather(*tasks)

        query_evidence = []
        for doc, ev in zip(correct_docs, results):
            rec = {"source": "clean", "data_idx": data_idx, "doc_type": "correct",
                   "question": question, "evidence_unit": ev}
            all_evidence.append(rec)
            query_evidence.append(rec)
        ind_evidence_by_query[data_idx] = query_evidence

    # ─── Phase 1b: Extract evidence for amplified pairs ───
    log("\n--- Phase 1b: Evidence Extraction (Amplified / Amp8) ---")
    amp_evidence_by_query = {}

    for sel_idx, (data_idx, amp_docs) in enumerate(amp_selected):
        question = amp8_data[data_idx]["question"]
        log(f"  [{sel_idx+1}/{len(amp_selected)}] Q: {question[:50]}... ({len(amp_docs)} misinfo docs)")

        tasks = [extract_evidence(question, doc["text"]) for doc in amp_docs]
        results = await asyncio.gather(*tasks)

        query_evidence = []
        for doc, ev in zip(amp_docs, results):
            rec = {"source": "amp8", "data_idx": data_idx, "doc_type": doc["type"],
                   "question": question, "evidence_unit": ev}
            all_evidence.append(rec)
            query_evidence.append(rec)
        amp_evidence_by_query[data_idx] = query_evidence

    # Save evidence
    with open(OUTPUT_EVIDENCE, "w", encoding="utf-8") as f:
        for rec in all_evidence:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"\nSaved {len(all_evidence)} evidence units")

    # ─── Phase 2: Pairwise Dependence Judgment ───
    log("\n--- Phase 2a: Independent Pairs ---")
    ind_pair_count = 0
    ind_tasks = []

    for data_idx, query_evidence in ind_evidence_by_query.items():
        question = query_evidence[0]["question"]
        # Group by extracted answer, only pair docs with SAME answer
        answer_groups = {}
        for ev_rec in query_evidence:
            eu = ev_rec["evidence_unit"]
            ans = eu.get("answer", "").lower().strip()
            # Skip empty/failed extractions and "not provided" answers
            if not ans or "not provide" in ans or "does not" in ans or not eu.get("evidence_span"):
                continue
            if ans not in answer_groups:
                answer_groups[ans] = []
            answer_groups[ans].append(eu)

        for answer, group in answer_groups.items():
            if len(group) < 2:
                continue
            for i, j in combinations(range(len(group)), 2):
                if ind_pair_count >= TARGET_INDEPENDENT_PAIRS:
                    break
                ind_tasks.append({
                    "question": question, "answer": answer,
                    "unit_a": group[i], "unit_b": group[j],
                    "category": "independent_pair"
                })
                ind_pair_count += 1
            if ind_pair_count >= TARGET_INDEPENDENT_PAIRS:
                break
        if ind_pair_count >= TARGET_INDEPENDENT_PAIRS:
            break

    log(f"  Judging {len(ind_tasks)} independent pairs...")
    ind_results = await asyncio.gather(*[
        judge_dependence(t["question"], t["answer"], t["unit_a"], t["unit_b"])
        for t in ind_tasks
    ])
    for task, result in zip(ind_tasks, ind_results):
        all_pairs.append({"category": "independent_pair", "judgment": result})

    log("\n--- Phase 2b: Amplified Pairs ---")
    amp_pair_count = 0
    amp_tasks = []

    for data_idx, query_evidence in amp_evidence_by_query.items():
        question = query_evidence[0]["question"]
        # Group amplified docs by answer too
        answer_groups = {}
        for ev_rec in query_evidence:
            eu = ev_rec["evidence_unit"]
            ans = eu.get("answer", "").lower().strip()
            if not ans or "not provide" in ans or "does not" in ans or not eu.get("evidence_span"):
                continue
            if ans not in answer_groups:
                answer_groups[ans] = []
            answer_groups[ans].append(eu)

        for answer, group in answer_groups.items():
            if len(group) < 2:
                continue
            for i, j in combinations(range(len(group)), 2):
                if amp_pair_count >= TARGET_AMPLIFIED_PAIRS:
                    break
                amp_tasks.append({
                    "question": question, "answer": answer,
                    "unit_a": group[i], "unit_b": group[j],
                    "category": "amplified_pair"
                })
                amp_pair_count += 1
            if amp_pair_count >= TARGET_AMPLIFIED_PAIRS:
                break
        if amp_pair_count >= TARGET_AMPLIFIED_PAIRS:
            break

    log(f"  Judging {len(amp_tasks)} amplified pairs...")
    amp_results = await asyncio.gather(*[
        judge_dependence(t["question"], t["answer"], t["unit_a"], t["unit_b"])
        for t in amp_tasks
    ])
    for task, result in zip(amp_tasks, amp_results):
        all_pairs.append({"category": "amplified_pair", "judgment": result})

    # Save pairs
    with open(OUTPUT_PAIRS, "w", encoding="utf-8") as f:
        for rec in all_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"\nSaved {len(all_pairs)} pair judgments")

    # ─── Phase 3: Analysis ───
    analyze(all_pairs)


def analyze(all_pairs: list) -> None:
    log("\n" + "=" * 60)
    log("PILOT v2 RESULTS")
    log("=" * 60)

    categories = {}
    for pair in all_pairs:
        cat = pair["category"]
        score = pair["judgment"].get("score", -1)
        level = pair["judgment"].get("level", "ERROR")
        if score < 0:
            continue
        if cat not in categories:
            categories[cat] = {"scores": [], "levels": []}
        categories[cat]["scores"].append(score)
        categories[cat]["levels"].append(level)

    for cat in ["amplified_pair", "independent_pair"]:
        if cat not in categories:
            continue
        scores = categories[cat]["scores"]
        levels = categories[cat]["levels"]
        avg = sum(scores) / len(scores) if scores else 0
        level_dist = {}
        for lv in levels:
            level_dist[lv] = level_dist.get(lv, 0) + 1

        log(f"\n[{cat}] ({len(scores)} pairs)")
        log(f"  Avg score: {avg:.3f}")
        log(f"  Levels: {dict(sorted(level_dist.items()))}")
        log(f"  Range: [{min(scores):.2f}, {max(scores):.2f}]")

    # AUROC
    if "amplified_pair" in categories and "independent_pair" in categories:
        amp_scores = categories["amplified_pair"]["scores"]
        ind_scores = categories["independent_pair"]["scores"]
        amp_mean = sum(amp_scores) / len(amp_scores)
        ind_mean = sum(ind_scores) / len(ind_scores)

        correct = 0
        total = 0
        for a in amp_scores:
            for i in ind_scores:
                total += 1
                if a > i:
                    correct += 1
                elif a == i:
                    correct += 0.5
        auroc = correct / total if total > 0 else 0

        log(f"\n--- SEPARABILITY ---")
        log(f"  Amplified mean:   {amp_mean:.3f} (n={len(amp_scores)})")
        log(f"  Independent mean: {ind_mean:.3f} (n={len(ind_scores)})")
        log(f"  Gap:              {amp_mean - ind_mean:.3f}")
        log(f"  AUROC:            {auroc:.3f}")

        if auroc > 0.85:
            log(f"\n  >>> GOOD: Method is viable with {MODEL}. Consider deployment.")
        elif auroc > 0.7:
            log(f"\n  >>> MODERATE: Has discriminative power. Try stronger model or tune prompt.")
        else:
            log(f"\n  >>> POOR: Insufficient separation. Reconsider approach.")

    log("\n" + "=" * 60)


if __name__ == "__main__":
    start = time.time()
    asyncio.run(run_pilot_v2())
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
