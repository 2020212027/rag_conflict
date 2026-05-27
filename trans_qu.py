import os
import json
import numpy as np
import pandas as pd
import argparse

np.random.seed(31415)


def build_user_content(category, aspect_schema, review):
    """
    将 category / aspect_schema / review 拼装为用户侧输入文本。
    格式与 instruction 模板中占位符对应。
    """
    return (
        f"Category:\n{category}\n\n"
        f"Aspect Schema:\n{aspect_schema}\n\n"
        f"Review:\n{review}\n\n"
        f"Return only a JSON array with one object for each aspect in the aspect schema."
    )


def process_fn(example, idx, split):
    """
    将单条 JSON 条目转换为 parquet 所需格式。
    """
    # system 侧：使用记录自带的完整 instruction（含标注规范、示例等）
    instruction = example["instruction"]

    # user 侧：由 category + aspect_schema + review 拼装
    input_text = build_user_content(
        category      = example["category"],
        aspect_schema = example["aspect_schema"],
        review        = example["review"],
    )

    # ground_truth：第一个标注结果（字符串形式）
    output = example["ground_truth"]

    data = {
        "data_source": "absa_ecommerce",
        "prompt": [
            {"role": "system", "content": instruction},
            {"role": "user",   "content": input_text},
        ],
        "ability": "shopping",
        "reward_model": {
            "style":        "rule",
            "ground_truth": output,
        },
        "extra_info": {
            "split":       split,
            "index":       idx,
            "id":          example["id"],
            "category":    example["category"],
            "instruction": instruction,
            "input":       input_text,
            "output":      output,
            # 保留所有标注结果以备后用
            "all_results": json.dumps(
                example.get("all_results", []),
                ensure_ascii=False
            ),
        },
    }
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir",    default='./')
    parser.add_argument("--dataset_file", default='output.json')   # 上一步生成的 JSON
    parser.add_argument("--val_size",     default=0.1, type=float)
    args = parser.parse_args()

    # ── 读取 JSON ──────────────────────────────────────────────────
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Total entries loaded: {len(dataset)}")

    # ── 划分 train / test ──────────────────────────────────────────
    train_num    = int(len(dataset) * (1 - args.val_size))
    dataset_train = dataset[:train_num]
    dataset_test  = dataset[train_num:]

    # ── Shuffle 训练集 ─────────────────────────────────────────────
    np.random.shuffle(dataset_train)

    # ── 处理为 parquet 格式 ────────────────────────────────────────
    train_dataset = [process_fn(d, idx, "train") for idx, d in enumerate(dataset_train)]
    test_dataset  = [process_fn(d, idx, "test")  for idx, d in enumerate(dataset_test)]

    # ── 转为 DataFrame ─────────────────────────────────────────────
    train_df = pd.DataFrame(train_dataset)
    test_df  = pd.DataFrame(test_dataset)

    # ── 保存为 Parquet ─────────────────────────────────────────────
    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)

    train_df.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(local_dir,  "test.parquet"))

    print(f"\n{'='*50}")
    print(f"  Train size : {len(train_dataset)}")
    print(f"  Test  size : {len(test_dataset)}")
    print(f"  Saved to   : {local_dir}")
    print(f"{'='*50}")
