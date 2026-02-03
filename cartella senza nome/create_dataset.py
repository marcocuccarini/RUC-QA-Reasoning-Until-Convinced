# ======================================================
# file: prepare_datasets_and_facts.py
# ======================================================
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Local imports
from classes.utils import *
from configuration.hyperparameters import *

OUTPUT_DIR = Path("prepared_datasets")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def main():
    print("🚀 Preparing datasets and facts files...")

    datasets_dict = {name: safe_load_dataset(cfg) for name, cfg in datasets_config.items()}

    for dataset_name, dataset in datasets_dict.items():
        print(f"\n📂 Processing {dataset_name} (sample limit = {SAMPLE_LIMIT})")

        file_base = dataset_aliases.get(dataset_name, dataset_name.lower())
        combined_data = []
        facts_data = {}

        for i, example in enumerate(tqdm(dataset, desc=f"{dataset_name}")):
            if SAMPLE_LIMIT and i >= SAMPLE_LIMIT:
                print(f"⏹️ Stopping at {SAMPLE_LIMIT} samples for {dataset_name}")
                break

            # --------------------------
            # Detect choices and answer
            # --------------------------
            choices, answer_key = detect_dataset_format(example, {}, i)

            # Save facts data (choice text mapping)
            facts_data[str(i)] = {c: c for c in choices}

            # Save combined dataset entry
            combined_data.append({
                "index": i,
                "question": example.get("question", ""),
                "choices": choices,
                "answerKey": answer_key
            })

        # Save combined dataset
        dataset_file = OUTPUT_DIR / f"{file_base}_dataset.json"
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        # Save facts file
        facts_file = OUTPUT_DIR / f"{file_base}_facts.json"
        with open(facts_file, "w", encoding="utf-8") as f:
            json.dump(facts_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {dataset_name}: dataset -> {dataset_file}, facts -> {facts_file}")

    print("\n🎉 All datasets prepared successfully!")

if __name__ == "__main__":
    main()
