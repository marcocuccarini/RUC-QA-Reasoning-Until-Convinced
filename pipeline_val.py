# ======================================================
# hyperparam_search_fixed.py  (REVISED with oriented debug & fixes)
# ======================================================

from classes.running import *


datasets=["SciQ","PlausibleQA","arc-easy"]

for dataset_name in datasets:
    val_file = SPLIT_DIR / f"{dataset_name.lower()}_val.json"
    with open(val_file, "r", encoding="utf-8") as f:
        val_dataset = json.load(f)

    # -----------------------------
    # Grid search
    # -----------------------------
    # -----------------------------
    # Grid search with resume support
    # -----------------------------
    out_file = RESULTS_DIR / f"{dataset_name.lower()}_hyperparam_search.json"

    # Load previous results if exist
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as f:
            results_log = json.load(f)
        print(f"[INFO] Loaded {len(results_log)} previous results from {out_file}")
    else:
        results_log = []

    # Determine best so far
    best_acc = max([r["accuracy"] for r in results_log], default=0.0)
    best_hparams = None
    for r in results_log:
        if r["accuracy"] == best_acc:
            best_hparams = r["hparams"]
            break

    # List of hyperparameters to evaluate
    custom_hparams_list = [
        {"TOP_PARAGRAPHS": 5, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
        {"TOP_PARAGRAPHS": 10, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
        {"TOP_PARAGRAPHS": 15, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.1},
        {"TOP_PARAGRAPHS": 10, "MAX_ARGUMENTS": 2, "DEFAULT_CONFIDENCE_THRESHOLD": 0.6, "DEFAULT_CONFIDENCE_MARGIN": 0.2}
    ]

    # Skip already evaluated hparams
    evaluated_hparams = [r["hparams"] for r in results_log]

    start_time = time.time()
    for hparams in custom_hparams_list:
        if hparams in evaluated_hparams:
            print(f"[INFO] Skipping already evaluated hyperparameters: {hparams}")
            continue

        print(f"\n=== Testing hyperparameters: {hparams} ===")
        acc = run_on_validation(hparams, val_dataset)
        print(f"➡️ Accuracy: {acc:.2f}%")

        results_log.append({"hparams": hparams, "accuracy": acc})

        if acc > best_acc:
            best_acc = acc
            best_hparams = hparams

        # Save after each evaluation
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Results saved to {out_file}")

    elapsed = time.time() - start_time
    print(f"\n🏆 Best hyperparameters: {best_hparams} with validation accuracy {best_acc:.2f}%")
    print(f"Elapsed time: {elapsed:.1f} seconds")
