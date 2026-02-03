# ======================================================
# merge_dataset_with_facts.py
# ======================================================

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_dataset

# -----------------------------
# Configuration
# -----------------------------
RESULTS_DIR = Path("merged_datasets")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN")

# Dataset configuration
datasets_config = {
    "ARC-Easy": {"flag": True, "hf_name": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test"},
    "SciQ": {"flag": True, "hf_name": "allenai/sciq", "split": "test"},
    "PlausibleQA": {"flag": False, "hf_name": "data/plausibleQA.json"},
}

dataset_aliases = {
    "PlausibleQA": "plausibleqa",
    "SciQ": "sciq",
    "ARC-Easy": "arc-easy"
}

SUMMARY_SECTION_ALIASES = ["summary", "overview", "abstract"]

# -----------------------------
# Utility functions
# -----------------------------
def sanitize_for_windows(name: str) -> str:
    return re.sub(r'[<>:\\"/\\|?*]', "_", name)

def safe_load_dataset(cfg: Dict[str, Any]):
    """
    Load dataset from HuggingFace or local JSON.
    """
    if cfg["flag"]:
        ds = load_dataset(cfg["hf_name"], cfg.get("config"))
        if cfg["split"] in ds:
            ds = ds[cfg["split"]]
        else:
            ds = next(iter(ds.values()))
        print(f"✅ Loaded {cfg['hf_name']} ({len(ds)} samples)")
        return ds
    else:
        with open(cfg["hf_name"], "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Loaded local dataset ({len(data)} samples)")
        return data

def load_or_aggregate_facts(dataset_name: str) -> Dict[str, Any]:
    """
    Load pre-aggregated facts if available.
    """
    agg_dir = Path("aggregated_fact")
    agg_dir.mkdir(exist_ok=True)
    file_base = dataset_aliases.get(dataset_name, dataset_name.lower())
    file_path = agg_dir / f"{file_base}_preprocessed_fact.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def detect_dataset_format(example: Dict[str, Any], facts_data: Dict[str, Any], index: int):
    """
    Return choices and answer_key based on dataset format.
    """
    choices: Dict[str, str] = {}
    answer_key = ""

    # SciQ
    if all(k in example for k in ["correct_answer", "distractor1", "distractor2", "distractor3"]):
        choice_texts = [
            example["correct_answer"],
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
        ]
        facts_for_example = facts_data.get(str(index), {})
        choices = {text: facts_for_example.get(text, text) for text in choice_texts}
        answer_key = example["correct_answer"]

    # ARC
    elif "choices" in example and "answerKey" in example:
        choice_texts = example["choices"]["text"]
        choice_labels = example["choices"]["label"]
        facts_for_example = facts_data.get(str(index), {})
        choices = {text: facts_for_example.get(text, text) for text in choice_texts}
        label_to_text = dict(zip(choice_labels, choice_texts))
        answer_key = label_to_text.get(example["answerKey"], "")

    # MedMCQA
    elif all(k in example for k in ["opa", "opb", "opc", "opd", "cop"]):
        choice_texts = [example["opa"], example["opb"], example["opc"], example["opd"]]
        facts_for_example = facts_data.get(str(index), {})
        choices = {text: facts_for_example.get(text, text) for text in choice_texts}
        try:
            answer_key = choice_texts[int(example["cop"]) - 1]
        except Exception:
            answer_key = ""

    # PlausibleQA
    elif "question_type" in example and "candidate_answers" in example and "answer" in example:
        choice_texts = list(example["candidate_answers"].keys())
        facts_for_example = facts_data.get(str(index), {})
        choices = {text: facts_for_example.get(text, str(example["candidate_answers"][text])) for text in choice_texts}
        if isinstance(example["answer"], dict) and "text" in example["answer"]:
            answer_key = example["answer"]["text"]
        else:
            answer_key = str(example["answer"])

    # Fallback generic
    else:
        choices = example.get("options", {})
        answer_key = example.get("answer", "")

    # Final safety
    if not answer_key and choices:
        answer_key = list(choices.keys())[0]

    return choices, answer_key

# -----------------------------
# Main merging function
# -----------------------------
def merge_datasets_with_facts(datasets_config, dataset_aliases):
    merged_datasets = {}

    for dataset_name, cfg in datasets_config.items():
        dataset = safe_load_dataset(cfg)
        facts_data = load_or_aggregate_facts(dataset_name)
        file_base = dataset_aliases.get(dataset_name, dataset_name.lower())
        merged_list = []

        for i, example in enumerate(dataset):
            choices, answer_key = detect_dataset_format(example, facts_data, i)
            choice_facts = facts_data.get(str(i), {c: c for c in choices})

            merged_example = {
                "index": i,
                "question": example.get("question", ""),
                "choices": choices,
                "answerKey": answer_key,
                "facts": choice_facts
            }
            merged_list.append(merged_example)

        merged_datasets[dataset_name] = merged_list

        # Save merged dataset
        save_path = RESULTS_DIR / f"{file_base}_with_facts.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved merged dataset: {save_path} ({len(merged_list)} examples)")

    return merged_datasets

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    merged_datasets = merge_datasets_with_facts(datasets_config, dataset_aliases)
    print("✅ All datasets merged successfully.")
