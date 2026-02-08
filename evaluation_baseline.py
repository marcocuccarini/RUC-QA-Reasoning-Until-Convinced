import json
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
RESULTS_DIR = Path("results_baseline")  # Where predictions are stored
TOP_PARAGRAPHS = [2, 5, 10, 15]

# =====================================================
# HELPERS
# =====================================================
def compute_accuracy(lst):
    """Compute percentage of correct predictions."""
    if not lst:
        return 0.0
    return 100.0 * sum(x["correct"] for x in lst) / len(lst)

def print_step_accuracy(results, label="FINAL"):
    print(f"\n📊 ACCURACY ({label})")
    print(f" No context: {compute_accuracy(results['no_context']):.2f}%")
    for n in TOP_PARAGRAPHS:
        print(f" Top {n}: {compute_accuracy(results['with_context'][n]):.2f}%")

# =====================================================
# MAIN LOOP
# =====================================================
for model_dir in RESULTS_DIR.iterdir():
    if not model_dir.is_dir():
        continue
    
    print(f"\n=== EVALUATING MODEL: {model_dir.name} ===")
    
    # Process all prediction files
    for pred_file in model_dir.glob("*_predictions.json"):
        print(f"\nDataset: {pred_file.name}")
        with open(pred_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        results = data.get("results")
        if not results:
            print(" No results found in file.")
            continue
        
        # Ensure 'with_context' keys are integers
        results["with_context"] = {int(k): v for k, v in results["with_context"].items()}
        
        # Print accuracy
        print_step_accuracy(results, label=pred_file.stem)
