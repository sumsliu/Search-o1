#!/usr/bin/env python3
"""Download and preprocess benchmark data for Search-o1."""
import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def preprocess_gpqa(split: str = "diamond") -> Path:
    """Download GPQA from HuggingFace and convert to project JSON format."""
    from datasets import load_dataset

    config_map = {
        "diamond": "gpqa_diamond",
        "main": "gpqa_main",
        "extended": "gpqa_extended",
    }
    if split not in config_map:
        raise ValueError(f"Unsupported GPQA split: {split}")

    print(f"Downloading GPQA ({split}) from HuggingFace...")
    ds = load_dataset("Idavidrein/gpqa", config_map[split], split="train")

    output_dir = DATA_ROOT / "GPQA"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.json"

    filtered_data = []
    for idx, row in enumerate(tqdm(ds, desc="Processing GPQA")):
        question = row["Question"]
        answers = [
            ("Correct Answer", row["Correct Answer"]),
            ("Incorrect Answer 1", row["Incorrect Answer 1"]),
            ("Incorrect Answer 2", row["Incorrect Answer 2"]),
            ("Incorrect Answer 3", row["Incorrect Answer 3"]),
        ]
        random.seed(idx)
        random.shuffle(answers)

        choices = ["A", "B", "C", "D"]
        formatted_answers = []
        correct_choice = None
        for i, (label, answer) in enumerate(answers):
            choice = choices[i]
            formatted_answers.append((choice, answer))
            if label == "Correct Answer":
                correct_choice = choice

        formatted_choices = "\n".join(
            [f"({choice}) {answer}" for choice, answer in formatted_answers]
        )
        filtered_data.append(
            {
                "id": idx,
                "Question": f"{question} Choices:\n{formatted_choices}\n",
                "Subdomain": row.get("Subdomain", ""),
                "High-level domain": row.get("High-level domain", ""),
                "Correct Choice": correct_choice,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(filtered_data)} examples to {output_path}")
    return output_path


def preprocess_nq(limit: int = 500) -> Path:
    """Download NQ from FlashRAG datasets and convert to project JSON format."""
    from datasets import load_dataset

    print("Downloading NQ from HuggingFace (FlashRAG)...")
    ds = load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split="test")

    output_dir = DATA_ROOT / "QA_Datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nq.json"

    data_list = []
    for idx, row in enumerate(tqdm(ds, desc="Processing NQ")):
        data_list.append(
            {
                "id": idx,
                "Question": row["question"],
                "answer": row["golden_answers"],
            }
        )
        if len(data_list) >= limit:
            break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)

    print(f"Saved {len(data_list)} examples to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download Search-o1 benchmark data.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gpqa",
        choices=["gpqa", "nq"],
        help="Dataset to download.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="diamond",
        choices=["diamond", "main", "extended"],
        help="GPQA split (ignored for nq).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Max examples for nq.",
    )
    args = parser.parse_args()

    if args.dataset == "gpqa":
        preprocess_gpqa(args.split)
    elif args.dataset == "nq":
        preprocess_nq(args.limit)


if __name__ == "__main__":
    main()
