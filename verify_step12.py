import json

for name in ["clean", "amp_8"]:
    path = f"results_majority_vote_{name}.jsonl"
    rs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    em = sum(r["is_correct"] for r in rs) / len(rs) * 100
    wr = sum(r["matches_wrong_answer"] for r in rs) / len(rs) * 100
    print(f"Step 1.2 {name}: N={len(rs)} EM={em:.1f}% Wrong={wr:.1f}%")

print("\n=== MADAM-RAG Cost Estimation ===")
# MADAM-RAG: per sample, per round:
#   - N agents (one per doc), each makes 1 call
#   - 1 aggregation call
# Rounds: up to 3 (may early-stop at 2)
# Total calls per sample per round = N_docs + 1
# Total calls per sample (3 rounds) = 3 * (N_docs + 1)

for name in ["clean", "amp_8"]:
    path = f"dataset_{name}.jsonl"
    samples = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    avg_docs = sum(s["num_total_docs"] for s in samples) / len(samples)
    # Round 1: N calls (agents) + 1 aggregation = N+1
    # Round 2: N calls + 1 aggregation = N+1
    # Round 3: N calls + 1 aggregation = N+1
    # Total (worst case 3 rounds): 3*(N+1)
    # But round 2+ prompts are MUCH longer (include all other agents' responses)
    calls_per_sample_3rounds = 3 * (avg_docs + 1)
    total_calls = calls_per_sample_3rounds * len(samples)
    
    # Token estimation:
    # Round 1 agent: ~300 input + 128 output = ~428 tok/call
    # Round 2+ agent: ~300 + N*200 (history) + 128 output
    # Aggregation: ~N*200 + 200 + 128 output
    avg_input_r1 = 350
    avg_input_r2 = 350 + avg_docs * 200  # includes history
    avg_input_agg = avg_docs * 200 + 400  # example + responses
    avg_output = 128
    
    r1_tokens = avg_docs * (avg_input_r1 + avg_output) + (avg_input_agg + avg_output)
    r2_tokens = avg_docs * (avg_input_r2 + avg_output) + (avg_input_agg + avg_output)
    total_tokens_per_sample = r1_tokens + 2 * r2_tokens  # 3 rounds
    total_tokens = total_tokens_per_sample * len(samples)
    
    # Time: ~1s per call with concurrency=3
    time_serial = total_calls * 1.0 / 60  # minutes
    time_concurrent3 = time_serial / 3
    
    # Cost (gpt-4o-mini): input $0.15/1M, output $0.60/1M
    input_tokens = total_tokens * 0.7  # rough split
    output_tokens = total_tokens * 0.3
    cost = input_tokens * 0.15 / 1e6 + output_tokens * 0.60 / 1e6
    
    print(f"\n{name}:")
    print(f"  Avg docs/sample: {avg_docs:.1f}")
    print(f"  API calls (3 rounds): {total_calls:.0f}")
    print(f"  Est. tokens: {total_tokens/1e6:.2f}M")
    print(f"  Est. cost: ${cost:.2f}")
    print(f"  Est. time (serial): {time_serial:.0f} min")
    print(f"  Est. time (3x concurrent): {time_concurrent3:.0f} min")

print("\n=== COMBINED ===")
clean_samples = [json.loads(l) for l in open("dataset_clean.jsonl", encoding="utf-8") if l.strip()]
amp8_samples = [json.loads(l) for l in open("dataset_amp_8.jsonl", encoding="utf-8") if l.strip()]
avg_clean = sum(s["num_total_docs"] for s in clean_samples) / len(clean_samples)
avg_amp8 = sum(s["num_total_docs"] for s in amp8_samples) / len(amp8_samples)
total_calls = 3 * (avg_clean + 1) * 215 + 3 * (avg_amp8 + 1) * 215
print(f"  Total API calls: {total_calls:.0f}")
print(f"  Est. time (3x concurrent): {total_calls / 3 / 60:.0f} min")
print(f"  Est. time (5x concurrent): {total_calls / 5 / 60:.0f} min")
