import json
import time
from pathlib import Path
from classes.running import *

# Define the base directory for results
BASE_SEARCH_DIR = Path("hyperparameters_search")

datasets = ["SciQ", "PlausibleQA", "arc-easy"]
# Ensure 'models' is defined in your classes.running or here
# models = ["gpt-oss:20b", "another-model:latest"] 

for model_name in models:
    # 1. Create a model-specific subdirectory
    # We sanitize the model name to replace colons or slashes for filesystem safety
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    model_results_dir = BASE_SEARCH_DIR / safe_model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "###" * 20)
    print(f"STARTING SEARCH FOR MODEL: {model_name}")
    print(f"###" * 20)

    for dataset_name in datasets:
        val_file = SPLIT_DIR / f"{dataset_name.lower()}_val.json"
        
        # Determine output file path based on your requested structure
        out_file = model_results_dir / f"{dataset_name.lower()}_hyperparam_search.json"

        if not val_file.exists():
            print(f"[WARN] Validation file {val_file} not found. Skipping.")
            continue

        with open(val_file, "r", encoding="utf-8") as f:
            val_dataset = json.load(f)

        # Resume logic
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                results_log = json.load(f)
            print(f"[INFO] Loaded {len(results_log)} results for {dataset_name} from {out_file}")
        else:
            results_log = []

        best_acc = max([r["accuracy"] for r in results_log], default=0.0)
        best_hparams = next((r["hparams"] for r in results_log if r["accuracy"] == best_acc), None)

        custom_hparams_list = [
            {"TOP_PARAGRAPHS": 5, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
            {"TOP_PARAGRAPHS": 10, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
            {"TOP_PARAGRAPHS": 15, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
            {"TOP_PARAGRAPHS": 10, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.2}
        ]

        evaluated_hparams = [r["hparams"] for r in results_log]

        start_time = time.time()
        for hparams in custom_hparams_list:
            if hparams in evaluated_hparams:
                print(f"[SKIP] Already evaluated: {hparams}")
                continue

            print(f"\n--- Testing {model_name} | {dataset_name} | {hparams} ---")
            
            # Pass the current model_name to the validation function
            acc = run_on_validation(hparams, val_dataset, model=model_name)
            
            print(f"➡️ Accuracy: {acc:.2f}%")
            results_log.append({"hparams": hparams, "accuracy": acc})

            if acc > best_acc:
                best_acc = acc
                best_hparams = hparams

            # Save incrementally
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results_log, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"\n🏆 Best for {dataset_name} ({model_name}): {best_acc:.2f}%")
        print(f"Elapsed: {elapsed:.1f}s")