import json
import unicodedata
from pathlib import Path

# ----------------------------
# Functions
# ----------------------------
def is_icon(value: str) -> bool:
    """
    Returns True if the value is an emoji or symbol.
    """
    if not isinstance(value, str) or not value:
        return False

    for char in value:
        try:
            # Unicode name exists for emojis/icons
            name = unicodedata.name(char).lower()
            if any(word in name for word in ["sign", "symbol", "musical", "face", "heart", "star"]):
                return True
        except ValueError:
            # Non-character or unknown, ignore
            continue

    # If string contains non-alphanumeric characters and is short, treat as icon
    if len(value) <= 2 and not value.isalnum():
        return True

    return False

def filter_dataset_icons(input_file: Path, output_file: Path):
    """
    Removes examples where the answerKey is an emoji or icon.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_data = []
    skipped_count = 0
    for example in data:
        answer_key = example.get("answerKey")
        if is_icon(answer_key):
            skipped_count += 1
            continue  # skip this example
        filtered_data.append(example)

    # Save filtered dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"Filtered dataset saved to {output_file}")
    print(f"Skipped {skipped_count} examples with emoji/icon answerKeys")

# ----------------------------
# Usage example
# ----------------------------
if __name__ == "__main__":
    input_path = Path("split_datasets/plausibleqa_test.json")  # input JSON file
    output_path = Path("split_datasets/plausibleqa_test_filtered.json")  # output JSON file

    filter_dataset_icons(input_path, output_path)
