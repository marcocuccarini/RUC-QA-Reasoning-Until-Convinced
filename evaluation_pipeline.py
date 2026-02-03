import json
from pathlib import Path
import re
import matplotlib.pyplot as plt

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

            # relaxed match: node text ends with the choice
            if node_text.endswith(fact_norm):
                strengths[label] = float(node_data.get("strength", 0.0))

    if not strengths:
        return None, {}

    best_value = max(strengths.values())

    # collect all equally-best labels
    best_labels = [k for k, v in strengths.items() if abs(v - best_value) < 1e-9]

    # NEW: random tie-breaking
    import random
    predicted = random.choice(best_labels)

    return predicted, strengths


# --------------------------------------------------
# Evaluate graphs and print if changed
# --------------------------------------------------
import random

def evaluate_graphs_terminal(dataset_name):
    base_path = Path("/Users/marco/Documents/GitHub/ArgMultipleChoiseQuestion/src")
    graph_dir = base_path / "graphs" / dataset_name
    dataset_file = base_path / "split_datasets" / f"{dataset_name.lower()}_test.json"

    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # Load dataset
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    correct = 0
    predicted_count = 0

    print(f"\n===== Evaluating dataset: {dataset_name} =====\n")

    for i, example in enumerate(dataset, start=1):
        graph_path = graph_dir / f"graph_{i}.json"

        if not graph_path.exists():
            # Skip if graph is missing
            continue

        # Load graph
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_json = json.load(f)

        choice_facts = example["facts"]
        gold = example.get("answerKey")

        predicted, _ = compute_prediction_from_graph(graph_json, choice_facts)

        # Check if graph changed
        nodes = graph_json.get("nodes", {})
        hypothesis_nodes = {nid: n for nid, n in nodes.items() if n.get("type") == "hypothesis"}
        graph_changed = any(abs(n.get("strength", 0.5) - 0.5) > 1e-9 for n in hypothesis_nodes.values())

        if not graph_changed:
            # Graph exists but uncaged → random choice
            predicted = random.choice(list(choice_facts.keys()))

        is_correct = predicted.lower() == gold.lower()
        predicted_count += 1
        if is_correct:
            correct += 1

        results.append({"correct": is_correct, "graph_changed": graph_changed})
        changed_text = "✅ Graph Changed" if graph_changed else "❌ Graph Unchanged → Random Choice"
        print(f"[Q{i}] Predicted={predicted} | Gold={gold} | Correct={is_correct} | {changed_text}")

    if predicted_count == 0:
        print("\n❌ No predictions were made. Accuracy cannot be computed.\n")
        return [], 0.0

    accuracy = (correct / predicted_count) * 100
    print(f"\n🎯 Final Accuracy (only actual predictions): {accuracy:.2f}%")
    print(f"ℹ️ Predictions made: {predicted_count} / {len(dataset)}\n")

    return results, accuracy


# --------------------------------------------------
# Plot statistics for changed vs unchanged graphs
# --------------------------------------------------
def plot_graph_change_statistics(results, dataset_name):
    changed_correct = 0
    changed_total = 0
    unchanged_correct = 0
    unchanged_total = 0

    for r in results:
        if r['correct'] is None:
            continue  # skip if no prediction
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

    print(f"Graphs Changed: {changed_total} | Accuracy: {changed_acc:.2f}%")
    print(f"Graphs Unchanged: {unchanged_total} | Accuracy: {unchanged_acc:.2f}%")

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
    ax.set_title(f"Accuracy by Graph Change Status - {dataset_name}")
    plt.show()

# --------------------------------------------------
# Main 
# --------------------------------------------------
if __name__ == "__main__":
    DATASET_NAMES = ["PlausibleQA", "SciQ", "arc-easy"]

    for dataset_name in DATASET_NAMES:
        results, accuracy = evaluate_graphs_terminal(dataset_name)
        plot_graph_change_statistics(results, dataset_name)
