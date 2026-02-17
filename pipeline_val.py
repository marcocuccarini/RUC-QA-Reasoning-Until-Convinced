import json
import time
from pathlib import Path
import random

# This import pulls in run_on_validation_generator from your running.py
from classes.running import * # Define the base directory for results
BASE_SEARCH_DIR = Path("hyperparameters_search")
datasets = ["SciQ", "PlausibleQA", "arc-easy"]
models = ["gpt-oss:20b"]  # Define your model list here

for model_name in models:
    safe_model_name = model_name.replace(":", "_").replace("/", "_")
    model_results_dir = BASE_SEARCH_DIR / safe_model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "###" * 20)
    print(f"STARTING SEARCH FOR MODEL: {model_name}")
    print(f"###" * 20)

    for dataset_name in datasets:
        val_file = SPLIT_DIR / f"{dataset_name.lower()}_val.json"
        out_file = model_results_dir / f"{dataset_name.lower()}_hyperparam_search.json"

        if not val_file.exists():
            print(f"[WARN] Validation file {val_file} not found. Skipping.")
            continue

        with open(val_file, "r", encoding="utf-8") as f:
            val_dataset = json.load(f)

        # Resume logic
        results_log = []
        if out_file.exists():
            with open(out_file, "r", encoding="utf-8") as f:
                results_log = json.load(f)
            print(f"[INFO] Loaded {len(results_log)} results for {dataset_name}")

        best_acc = max([r["accuracy"] for r in results_log], default=0.0)
        evaluated_hparams = [r["hparams"] for r in results_log]

        custom_hparams_list = [
            {"TOP_PARAGRAPHS": 5, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
            {"TOP_PARAGRAPHS": 10, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
            {"TOP_PARAGRAPHS": 15, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
            {"TOP_PARAGRAPHS": 10, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.7, "DEFAULT_CONFIDENCE_MARGIN": 0.2}
        ]

        for hparams in custom_hparams_list:
            if hparams in evaluated_hparams:
                print(f"[SKIP] Already evaluated: {hparams}")
                continue

            print(f"\n--- Testing {model_name} | {dataset_name} | {hparams} ---")
            
            correct_count = 0
            total_count = 0
            detailed_results = []

            # CALLING THE GENERATOR FROM running.py
            for result in run_on_validation_generator(hparams, val_dataset, model=model_name):
                total_count += 1
                if result["is_correct"]:
                    correct_count += 1
                
                detailed_results.append(result)
                print(f"  Example {result['example_idx']}/{len(val_dataset)}: {'✅' if result['is_correct'] else '❌'}")

            acc = (correct_count / total_count) * 100 if total_count > 0 else 0
            print(f"➡️ Accuracy: {acc:.2f}%")
            
            results_log.append({"hparams": hparams, "accuracy": acc})

            # Save incrementally
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results_log, f, indent=2, ensure_ascii=False)
            
            # Use the helper from running.py to save the full details of this specific run
            save_detailed_predictions(model_results_dir, dataset_name, hparams, detailed_results)

        print(f"\n🏆 Finished search for {dataset_name}")