import json
from pathlib import Path
import re
import matplotlib.pyplot as plt
import random

# --------------------------------------------------
# Normalize helper
# --------------------------------------------------
def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

# --------------------------------------------------
# Compute prediction from a graph
# --------------------------------------------------
def compute_prediction_from_graph(graph_json, choice_facts):
    strengths = {}
    nodes = graph_json.get("nodes", {})

    for label, fact_text in choice_facts.items():
        fact_norm = normalize(fact_text)
        for node_id, node_data in nodes.items():
            node_text = normalize(node_data.get("text", ""))
            if node_text.endswith(fact_norm):
                strengths[label] = float(node_data.get("strength", 0.0))

    if not strengths:
        return None, {}
    best_value = max(strengths.values())
    best_labels = [k for k, v in strengths.items() if abs(v - best_value) < 1e-9]
    predicted = random.choice(best_labels)
    return predicted, strengths

# --------------------------------------------------
# Evaluate graphs for one model and dataset
# Compute both accuracies: all graphs vs modified-only
# Also return percentage of graphs not considered
# --------------------------------------------------
def evaluate_graphs_for_model_dataset(model_dir, dataset_name, dataset_file):
    dataset_path = dataset_file
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return [], 0.0, 0.0, 0.0

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    graph_dir = model_dir / dataset_name
    results = []
    correct_all = 0
    correct_modified = 0
    count_all = 0
    count_modified = 0
    count_unconsidered = 0

    if not graph_dir.exists() or not any(graph_dir.iterdir()):
        print(f"⚠️ No graphs found for {model_dir.name} - {dataset_name}. Accuracy = 0%")
        return [], 0.0, 0.0, 100.0

    for i, example in enumerate(dataset, start=1):
        graph_path = graph_dir / f"graph_{i}.json"
        if not graph_path.exists():
            count_unconsidered += 1
            continue

        with open(graph_path, "r", encoding="utf-8") as f:
            graph_json = json.load(f)

        choice_facts = example["facts"]
        gold = example.get("answerKey")

        predicted, _ = compute_prediction_from_graph(graph_json, choice_facts)

        nodes = graph_json.get("nodes", {})
        hypothesis_nodes = {nid: n for nid, n in nodes.items() if n.get("type") == "hypothesis"}
        graph_changed = any(abs(n.get("strength", 0.5) - 0.5) > 1e-9 for n in hypothesis_nodes.values())

        # Count for all-graphs accuracy
        count_all += 1
        if graph_changed:
            count_modified += 1
            if predicted.lower() == gold.lower():
                correct_modified += 1
        else:
            # Random choice for unmodified graphs
            predicted = random.choice(list(choice_facts.keys()))
            count_unconsidered += 1

        if predicted.lower() == gold.lower():
            correct_all += 1

        results.append({"correct": predicted.lower() == gold.lower(), "graph_changed": graph_changed})

    # Compute accuracies
    accuracy_all = (correct_all / count_all * 100) if count_all else 0.0
    accuracy_modified = (correct_modified / count_modified * 100) if count_modified else 0.0
    percent_unconsidered = (count_unconsidered / len(dataset) * 100) if dataset else 0.0

    print(f"🎯 {model_dir.name} - {dataset_name}: "
          f"Accuracy (All Graphs) = {accuracy_all:.2f}%, "
          f"Accuracy (Modified Only) = {accuracy_modified:.2f}%, "
          f"Unconsidered Graphs = {percent_unconsidered:.2f}%")
    return results, accuracy_all, accuracy_modified, percent_unconsidered

# --------------------------------------------------
# Plot statistics for changed vs unchanged graphs
# --------------------------------------------------
def plot_graph_change_statistics(results, title):
    changed_correct = 0
    changed_total = 0
    unchanged_correct = 0
    unchanged_total = 0

    for r in results:
        if r['correct'] is None:
            continue
        if r['graph_changed']:
            changed_total += 1
            if r['correct']:
                changed_correct += 1
        else:
            unchanged_total += 1
            if r['correct']:
                unchanged_correct += 1

    changed_acc = (changed_correct / changed_total * 100) if changed_total else 0
    unchanged_acc = (unchanged_correct / unchanged_total * 100) if unchanged_total else 0

    categories = ['Graph Changed', 'Graph Unchanged']
    accuracies = [changed_acc, unchanged_acc]
    counts = [changed_total, unchanged_total]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, accuracies, color=['skyblue', 'salmon'])
    for bar, count, acc in zip(bars, counts, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.1f}%\n(n={count})", ha='center', va='bottom', fontsize=11)

    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    plt.show()

# --------------------------------------------------
# Main: iterate over all models and datasets
# --------------------------------------------------
if __name__ == "__main__":
    BASE_PATH = Path("/Users/marco/Documents/GitHub/ArgMultipleChoiseQuestion/src")
    GRAPHS_ROOT = BASE_PATH / "graphs"
    SPLIT_DATASETS = BASE_PATH / "split_datasets"

    DATASET_NAMES = ["PlausibleQA", "SciQ", "arc-easy"]
    model_dirs = [d for d in GRAPHS_ROOT.iterdir() if d.is_dir()]


    print("!!! The performance at each run can slightly change because, in cases where hypotheses have the same strength, the system will select randomly between the candidates.")
    # Store accuracies for final table
    all_acc_all = {}
    all_acc_modified = {}
    all_percent_unconsidered = {}

    for model_dir in model_dirs:
        all_acc_all[model_dir.name] = {}
        all_acc_modified[model_dir.name] = {}
        all_percent_unconsidered[model_dir.name] = {}
        for dataset_name in DATASET_NAMES:
            dataset_file = SPLIT_DATASETS / f"{dataset_name.lower()}_test.json"
            results, acc_all, acc_mod, percent_un = evaluate_graphs_for_model_dataset(
                model_dir, dataset_name, dataset_file)
            all_acc_all[model_dir.name][dataset_name] = acc_all
            all_acc_modified[model_dir.name][dataset_name] = acc_mod
            all_percent_unconsidered[model_dir.name][dataset_name] = percent_un

            # Plot only if modified graphs exist
            if results:
                plot_graph_change_statistics(results, f"{dataset_name} - {model_dir.name}")

    # Print final table
    print("\n================= FINAL ACCURACY TABLE =================")
    header = "Model".ljust(15) + "".join([f"{d.ljust(35)}" for d in DATASET_NAMES])
    print(header)
    print("-" * len(header))
    for model_name in all_acc_all:
        row_all = model_name.ljust(15) + "".join([f"{all_acc_all[model_name].get(d,0.0):<35.2f}" for d in DATASET_NAMES])
        row_mod = " " * 15 + "".join([f"{all_acc_modified[model_name].get(d,0.0):<35.2f}" for d in DATASET_NAMES])
        row_un = " " * 15 + "".join([f"{all_percent_unconsidered[model_name].get(d,0.0):<35.2f}" for d in DATASET_NAMES])
        print(row_all)
        print(row_mod)
        print(row_un)
        print()
