import json
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
RESULTS_DIR = Path("results_retrieval")

EXCLUDE_INVALID = True
EXCLUDE_EMPTY_CONTEXT = True   # only applies to with_context
EXCLUDE_NO_WIKI = True         # uses wiki_pages length

# =====================================================
# HELPERS
# =====================================================
def is_valid_result(ex, *, require_context=False):
    if EXCLUDE_INVALID and ex["model_answer"] == "INVALID":
        return False

    if require_context:
        if EXCLUDE_EMPTY_CONTEXT and not ex.get("context_used"):
            return False
        if EXCLUDE_NO_WIKI and not ex.get("wiki_pages"):
            return False

    return True


def accuracy(filtered):
    if not filtered:
        return 0.0
    return 100.0 * sum(e["correct"] for e in filtered) / len(filtered)


# =====================================================
# MAIN EVALUATION
# =====================================================
def evaluate_filtered(results_dir):
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue

        print(f"\n=== MODEL: {model_dir.name} ===")

        for pred_file in model_dir.glob("*_predictions.json"):
            dataset = pred_file.stem.replace("_predictions", "")

            with open(pred_file, "r", encoding="utf-8") as f:
                results = json.load(f)["results"]

            # -------- NO CONTEXT --------
            no_ctx_valid = [
                ex for ex in results["no_context"]
                if is_valid_result(ex)
            ]

            print(f"\n📚 Dataset: {dataset}")
            print(
                f" No context: "
                f"{accuracy(no_ctx_valid):.2f}% "
                f"(kept {len(no_ctx_valid)}/{len(results['no_context'])})"
            )

            # -------- WITH CONTEXT --------
            for k, examples in sorted(results["with_context"].items(), key=lambda x: int(x[0])):
                valid = [
                    ex for ex in examples
                    if is_valid_result(ex, require_context=True)
                ]

                print(
                    f" Top {k:>2}: "
                    f"{accuracy(valid):.2f}% "
                    f"(kept {len(valid)}/{len(examples)})"
                )


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    evaluate_filtered(RESULTS_DIR)
