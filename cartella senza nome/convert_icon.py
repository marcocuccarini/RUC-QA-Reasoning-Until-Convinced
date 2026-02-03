import json
import unicodedata
from pathlib import Path

# ----------------------------
# Functions
# ----------------------------
def convert_icon_to_word(key: str) -> str:
    """
    Converts a single emoji or icon into a descriptive word.
    If the string is normal text, returns it unchanged.
    """
    if not isinstance(key, str) or not key:
        return key

    # Check if the string is a single emoji or icon
    try:
        # Some emojis are surrogate pairs, so handle the whole string
        return unicodedata.name(key).lower().replace("_", " ")
    except ValueError:
        return key  # normal text, leave as is

def convert_dataset_icons(input_file: Path, output_file: Path):
    """
    Converts only emoji/icon keys in answerKey, choices, and facts
    into descriptive words, keeping the rest of the dataset unchanged.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted_data = []
    for example in data:
        # Convert answerKey if it's an icon
        example["answerKey"] = convert_icon_to_word(example.get("answerKey"))

        # Convert choices keys if they are icons
        if "choices" in example:
            new_choices = {}
            for k, v in example["choices"].items():
                new_key = convert_icon_to_word(k)
                new_choices[new_key] = v
            example["choices"] = new_choices

        # Convert facts keys if they are icons
        if "facts" in example:
            new_facts = {}
            for k, v in example["facts"].items():
                new_key = convert_icon_to_word(k)
                new_facts[new_key] = v
            example["facts"] = new_facts

        converted_data.append(example)

    # Save the updated dataset
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)

    print(f"Converted dataset saved to {output_file}")

# ----------------------------
# Usage example
# ----------------------------
if __name__ == "__main__":
    input_path = Path("split_datasets/plausibleqa_test.json")  # input JSON file
    output_path = Path("split_datasets/plausibleqa_test_converted.json")  # output JSON file

    convert_dataset_icons(input_path, output_path)
