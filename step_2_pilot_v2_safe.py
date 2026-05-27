"""
Step 2 - Plan B Pilot v2 (safe version): Sequential batches to avoid API throttling.
"""
import json, time, asyncio
from itertools import combinations
from openai import AsyncOpenAI

MODEL = "gpt-4o-mini"
CLEAN_PATH = r"d:\pythonProject\dataset_clean.jsonl"
AMP8_PATH = r"d:\pythonProject\dataset_amp_8.jsonl"
OUTPUT_PAIRS = r"d:\pythonProject\pilot_v2b_pairs.jsonl"
LOG_PATH = r"d:\pythonProject\pilot_v2b.log"
TARGET_INDEPENDENT_PAIRS = 80
TARGET_AMPLIFIED_PAIRS = 80
BATCH_SIZE = 5
BATCH_DELAY = 1.0

client = AsyncOpenAI(
    api_key="sk-32wSMR4Uqs101YZ7EgsgsvRJYu03bNEADZrvIMjLYhvwQCRD",
    base_url="https://api.chatanywhere.tech/v1",
    timeout=30.0
)

def log(msg):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

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
- If both documents describe the same well-known fact using DIFFERENT wording,
  DIFFERENT examples, and DIFFERENT supporting details, this is D0 or D1, NOT D3/D4.
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


async def call_llm(prompt, max_tokens=500):
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=max_tokens
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(content)
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))
            else:
                return None
    return None


async def run_batch(prompts, max_tokens=500):
    """Run prompts in small batches with delays."""
    results = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[i:i+BATCH_SIZE]
        batch_results = await asyncio.gather(*[call_llm(p, max_tokens) for p in batch])
        results.extend(batch_results)
        if i + BATCH_SIZE < len(prompts):
            await asyncio.sleep(BATCH_DELAY)
    return results


async def main():
    open(LOG_PATH, "w", encoding="utf-8").close()
    log("=" * 60)
    log("Plan B Pilot v2b: Safe batch mode")
    log(f"Model: {MODEL}, Batch: {BATCH_SIZE}, Delay: {BATCH_DELAY}s")
    log("=" * 60)

    with open(CLEAN_PATH, "r", encoding="utf-8") as f:
        clean_data = [json.loads(l) for l in f]
    with open(AMP8_PATH, "r", encoding="utf-8") as f:
        amp8_data = [json.loads(l) for l in f]

    # ─── Select data ───
    # Independent: from clean, correct docs grouped by answer
    ind_candidates = []
    for idx, item in enumerate(clean_data):
        correct_docs = [d for d in item["documents"] if d["type"] == "correct"]
        groups = {}
        for d in correct_docs:
            ans = d["answer"].lower().strip()
            groups.setdefault(ans, []).append(d)
        same_pairs = sum(len(list(combinations(range(len(g)), 2))) for g in groups.values() if len(g) >= 2)
        if same_pairs >= 1:
            ind_candidates.append((idx, same_pairs, groups))
    ind_candidates.sort(key=lambda x: -x[1])

    # Amplified: from amp8, misinfo docs
    amp_candidates = []
    for idx, item in enumerate(amp8_data):
        amp_docs = [d for d in item["documents"] if d["type"] in ("misinfo", "misinfo_amplified")]
        if len(amp_docs) >= 3:
            amp_candidates.append((idx, len(amp_docs), amp_docs))
    amp_candidates.sort(key=lambda x: -x[1])

    # ─── Phase 1: Evidence extraction ───
    log("\n--- Phase 1: Evidence Extraction ---")

    # Collect all docs to extract
    ind_extract_tasks = []  # (query_idx, question, doc_text, group_answer)
    ind_query_count = 0
    for idx, same_pairs, groups in ind_candidates:
        question = clean_data[idx]["question"]
        for ans, docs in groups.items():
            if len(docs) >= 2:
                for d in docs:
                    ind_extract_tasks.append((idx, question, d["text"], ans))
        ind_query_count += 1
        if len(ind_extract_tasks) > TARGET_INDEPENDENT_PAIRS * 4:
            break

    amp_extract_tasks = []
    for idx, num_docs, docs in amp_candidates[:5]:
        question = amp8_data[idx]["question"]
        for d in docs:
            amp_extract_tasks.append((idx, question, d["text"]))
        if len(amp_extract_tasks) > TARGET_AMPLIFIED_PAIRS * 3:
            break

    log(f"  Independent: {len(ind_extract_tasks)} docs from {ind_query_count} queries")
    log(f"  Amplified:   {len(amp_extract_tasks)} docs")

    # Extract independent evidence
    log("  Extracting independent evidence...")
    ind_prompts = [EVIDENCE_PROMPT.format(question=q, doc_text=t[:2000]) for _, q, t, _ in ind_extract_tasks]
    ind_evidence_raw = await run_batch(ind_prompts)
    log(f"  Done. {sum(1 for r in ind_evidence_raw if r)} / {len(ind_prompts)} succeeded")

    # Extract amplified evidence
    log("  Extracting amplified evidence...")
    amp_prompts = [EVIDENCE_PROMPT.format(question=q, doc_text=t[:2000]) for _, q, t in amp_extract_tasks]
    amp_evidence_raw = await run_batch(amp_prompts)
    log(f"  Done. {sum(1 for r in amp_evidence_raw if r)} / {len(amp_prompts)} succeeded")

    # ─── Phase 2: Build pairs and judge ───
    log("\n--- Phase 2: Pair Construction & Judgment ---")

    # Build independent pairs (same query + same answer)
    from collections import defaultdict
    ind_groups = defaultdict(list)  # (query_idx, answer) -> [evidence_unit]
    for task, ev in zip(ind_extract_tasks, ind_evidence_raw):
        if ev is None or not ev.get("answer") or not ev.get("evidence_span"):
            continue
        query_idx, question, _, group_ans = task
        ind_groups[(query_idx, group_ans)].append({"question": question, "evidence": ev})

    ind_pair_inputs = []
    for (qi, ans), units in ind_groups.items():
        if len(units) < 2:
            continue
        for i, j in combinations(range(len(units)), 2):
            ind_pair_inputs.append(("independent_pair", units[i], units[j], ans))
            if len(ind_pair_inputs) >= TARGET_INDEPENDENT_PAIRS:
                break
        if len(ind_pair_inputs) >= TARGET_INDEPENDENT_PAIRS:
            break

    # Build amplified pairs (same query, all misinfo share wrong answer)
    amp_groups = defaultdict(list)
    for task, ev in zip(amp_extract_tasks, amp_evidence_raw):
        if ev is None or not ev.get("answer") or not ev.get("evidence_span"):
            continue
        query_idx, question, _ = task
        amp_groups[query_idx].append({"question": question, "evidence": ev})

    amp_pair_inputs = []
    for qi, units in amp_groups.items():
        if len(units) < 2:
            continue
        # Group by answer
        ans_groups = defaultdict(list)
        for u in units:
            ans_groups[u["evidence"].get("answer", "").lower().strip()].append(u)
        for ans, group in ans_groups.items():
            if len(group) < 2:
                continue
            for i, j in combinations(range(len(group)), 2):
                amp_pair_inputs.append(("amplified_pair", group[i], group[j], ans))
                if len(amp_pair_inputs) >= TARGET_AMPLIFIED_PAIRS:
                    break
            if len(amp_pair_inputs) >= TARGET_AMPLIFIED_PAIRS:
                break
        if len(amp_pair_inputs) >= TARGET_AMPLIFIED_PAIRS:
            break

    log(f"  Independent pairs: {len(ind_pair_inputs)}")
    log(f"  Amplified pairs:   {len(amp_pair_inputs)}")

    # Judge all pairs
    all_pair_inputs = ind_pair_inputs + amp_pair_inputs
    log(f"  Judging {len(all_pair_inputs)} pairs total...")

    judge_prompts = []
    for category, unit_a, unit_b, answer in all_pair_inputs:
        ea = unit_a["evidence"]
        eb = unit_b["evidence"]
        p = DEPENDENCE_PROMPT.format(
            question=unit_a["question"], answer=answer,
            facts_a=json.dumps(ea.get("supporting_facts", []), ensure_ascii=False),
            details_a=json.dumps(ea.get("specific_details", []), ensure_ascii=False),
            span_a=ea.get("evidence_span", ""),
            facts_b=json.dumps(eb.get("supporting_facts", []), ensure_ascii=False),
            details_b=json.dumps(eb.get("specific_details", []), ensure_ascii=False),
            span_b=eb.get("evidence_span", "")
        )
        judge_prompts.append(p)

    judgments = await run_batch(judge_prompts, max_tokens=200)
    log(f"  Done. {sum(1 for j in judgments if j)} / {len(judgments)} succeeded")

    # ─── Save & Analyze ───
    level_to_score = {"D0": 0.0, "D1": 0.25, "D2": 0.5, "D3": 0.75, "D4": 1.0}
    all_pairs = []
    for (category, ua, ub, ans), judgment in zip(all_pair_inputs, judgments):
        if judgment is None:
            judgment = {"level": "ERROR", "score": -1, "key_evidence": "failed"}
        else:
            level = judgment.get("level", "D2")
            if judgment.get("score", 0.0) == 0.0:
                judgment["score"] = level_to_score.get(level, 0.5)
        all_pairs.append({"category": category, "judgment": judgment})

    with open(OUTPUT_PAIRS, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Analysis
    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)

    cats = defaultdict(list)
    for p in all_pairs:
        s = p["judgment"].get("score", -1)
        if s >= 0:
            cats[p["category"]].append((s, p["judgment"].get("level", "?")))

    for cat in ["amplified_pair", "independent_pair"]:
        if cat not in cats:
            continue
        scores = [x[0] for x in cats[cat]]
        levels = [x[1] for x in cats[cat]]
        avg = sum(scores) / len(scores)
        from collections import Counter
        dist = dict(Counter(levels).most_common())
        log(f"\n[{cat}] n={len(scores)}")
        log(f"  Avg: {avg:.3f}, Range: [{min(scores):.2f}, {max(scores):.2f}]")
        log(f"  Levels: {dist}")

    if "amplified_pair" in cats and "independent_pair" in cats:
        amp_s = [x[0] for x in cats["amplified_pair"]]
        ind_s = [x[0] for x in cats["independent_pair"]]
        correct = sum(1 for a in amp_s for i in ind_s if a > i) + 0.5 * sum(1 for a in amp_s for i in ind_s if a == i)
        total = len(amp_s) * len(ind_s)
        auroc = correct / total if total else 0
        log(f"\n  Amp mean: {sum(amp_s)/len(amp_s):.3f} | Ind mean: {sum(ind_s)/len(ind_s):.3f}")
        log(f"  Gap: {sum(amp_s)/len(amp_s) - sum(ind_s)/len(ind_s):.3f}")
        log(f"  AUROC: {auroc:.3f}")
        if auroc > 0.85:
            log("  >>> GOOD: Method viable.")
        elif auroc > 0.7:
            log("  >>> MODERATE: Try stronger model.")
        else:
            log("  >>> POOR: Reconsider approach.")

    log("\n" + "=" * 60)


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    print(f"\nTotal: {time.time()-start:.1f}s")
