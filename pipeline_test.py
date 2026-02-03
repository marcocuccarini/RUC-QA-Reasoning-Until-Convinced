
from classes.running import *


SPLIT_DIR = Path("split_datasets")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS = ["PlausibleQA", "SciQ", "arc-easy"]

model="gpt-oss:20b"

BEST_PARAMS = {
    "SciQ": {"TOP_PARAGRAPHS": 10, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
    "PlausibleQA": {"TOP_PARAGRAPHS": 10, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
    "arc-easy": {"TOP_PARAGRAPHS": 10, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1}
}

for name in DATASETS:
    with open(SPLIT_DIR / f"{name.lower()}_test.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    preds, acc = run_on_test(
        BEST_PARAMS[name],
        dataset,
        graph_dir=f"graphs/{model.replace(":","-")}/{name}",
        use_intro_only=True,
        dataset_sample=900
    )

    out = RESULTS_DIR / f"{name.lower()}_test_predictions.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

    print(f"\n🏆 {name} accuracy: {acc:.2f}%")
