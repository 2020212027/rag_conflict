"""
Prepare FaithEval-Inconsistent-MD-Full dataset.
Splits original context by 'Document:' into a document list.
No filtering, no text modification.
"""
import pyarrow.parquet as pq
import json
import os

INPUT_PATH = r"d:\pythonProject\FaithEval_inconsistent\data\test-00000-of-00001.parquet"
OUTPUT_PATH = r"d:\pythonProject\module3_faitheval\faitheval_inconsistent_md_full.jsonl"


def split_context_to_documents(context: str) -> list:
    """Split context by 'Document:' marker into separate documents."""
    parts = context.split("Document:")
    documents = []
    for part in parts:
        text = part.strip()
        if text:
            documents.append(text)
    return documents


def main():
    table = pq.read_table(INPUT_PATH)
    print(f"Loaded {table.num_rows} samples from FaithEval-Inconsistent")

    samples = []
    for i in range(table.num_rows):
        context = table.column("context")[i].as_py()
        documents = split_context_to_documents(context)

        sample = {
            "qid": table.column("qid")[i].as_py(),
            "question": table.column("question")[i].as_py(),
            "documents": documents,
            "context_original": context,
            "answers": table.column("answers")[i].as_py(),
            "subset": table.column("subset")[i].as_py(),
            "gold": "conflict",
        }
        samples.append(sample)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} samples to {OUTPUT_PATH}")

    # Stats
    doc_counts = [len(s["documents"]) for s in samples]
    print(f"Documents per sample: min={min(doc_counts)} max={max(doc_counts)} avg={sum(doc_counts)/len(doc_counts):.1f}")

    from collections import Counter
    print(f"Subset distribution: {dict(Counter(s['subset'] for s in samples))}")


if __name__ == "__main__":
    main()
