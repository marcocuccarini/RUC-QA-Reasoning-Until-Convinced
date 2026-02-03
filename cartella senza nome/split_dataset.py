import json
import random
from pathlib import Path

MERGED_DATA_DIR = Path("merged_datasets")
SPLIT_DIR = Path("split_datasets")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def split_dataset_fixed_validation(dataset_name: str, val_samples: int = 40, seed: int = 42):
    """
    Split merged dataset into a fixed-size validation set and the remaining as test set.
    
    Args:
        dataset_name: name of the dataset (matching merged JSON file)
        val_samples: number of examples to use for validation
        seed: random seed for reproducibility
    """
    file_path = MERGED_DATA_DIR / f"{dataset_name.lower()}_with_facts.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if val_samples >= len(data):
        raise ValueError(f"Validation size {val_samples} is larger than dataset size {len(data)}")

    random.seed(seed)
    random.shuffle(data)

    val_set = data[:val_samples]
    test_set = data[val_samples:]

    # Save splits
    val_path = SPLIT_DIR / f"{dataset_name.lower()}_val.json"
    test_path = SPLIT_DIR / f"{dataset_name.lower()}_test.json"

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_set, f, indent=2, ensure_ascii=False)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"✅ {dataset_name}: {len(val_set)} validation, {len(test_set)} test samples saved.")
    return val_set, test_set

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    datasets = ["ARC-Easy", "SciQ", "PlausibleQA"]
    for ds in datasets:
        split_dataset_fixed_validation(ds, val_samples=50)
