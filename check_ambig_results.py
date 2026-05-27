import json

with open("d:/pythonProject/module2_ambigdocs/results/ckpt_v4_ambigdocs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        print(f'[{r["idx"]}] Q={r["question"]}')
        print(f'  gold={r["gold_answers"]}')
        print(f'  naive: {r["naive_answer"]}  recall={r["naive_recall"]}')
        print(f'  v4:    {r["final_answer"]}  recall={r["final_recall"]}')
        print(f'  type={r["decision_type"]} clusters={r["num_clusters"]}/{r["num_docs"]}')
        print()
