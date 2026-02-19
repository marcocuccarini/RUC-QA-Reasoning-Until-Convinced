# ======================================================
# hyperparam_search_fixed.py  (REVISED with oriented debug & fixes)
# ======================================================

import json
from pathlib import Path
from itertools import product
import math
import time
import random
# ---- Utilities & local classes ----
from classes.ServerOllama import OllamaServer, OllamaChat
from classes.LLMUser import LLMUser
from classes.ArgumentationGraph import ArgumentationGraph
from classes.utils import fetch_wikipedia_pages, rank_paragraphs_from_text


# -----------------------------
# Configuration / defaults
# -----------------------------
SPLIT_DIR = Path("split_datasets")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)




def safe_parse_llm_json(raw_text):
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return {"arguments": [], "relations": [], "targets": []}

    # Normalize alias keys
    if "relation" in data and "relations" not in data:
        data["relations"] = data["relation"]

    for key in ["arguments", "relations", "targets"]:
        if key not in data or not isinstance(data[key], list):
            data[key] = []

    return data



# -----------------------------
# Compute strengths from graph (robust)
# -----------------------------
def compute_strengths_from_graph(G: ArgumentationGraph, choice_facts: dict):
    """
    Returns a dict mapping choice_key -> float(strength)
    Defensive: ensures returned values are floats and logs conversions.
    """
    raw_node_strengths = G.compute_strengths()
    strengths = {}
    for nid, val in raw_node_strengths.items():
        node_txt = (G.get_text_from_id(nid) or "").strip()
        if not node_txt:
            continue
        try:
            val_f = float(val)
        except Exception:
            try:
                val_f = float(str(val).strip())
            except Exception:
                print(f"[WARN] Unparsable node strength for node {nid}: {val!r} -> using 0.0")
                val_f = 0.0

        for choice_key, choice_text in choice_facts.items():
            if isinstance(choice_text, str) and node_txt.lower() == choice_text.strip().lower():
                strengths[choice_key] = max(strengths.get(choice_key, 0.0), float(val_f))
    strengths = {k: float(v) for k, v in strengths.items()}
    return strengths


# -----------------------------
# Run pipeline with hyperparams
# -----------------------------
def run_on_validation(hparams: dict, dataset: list):
    global TOP_PARAGRAPHS, MAX_ARGUMENTS, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DOMINANCE_MARGIN
    TOP_PARAGRAPHS = hparams["TOP_PARAGRAPHS"]
    MAX_ARGUMENTS = hparams["MAX_ARGUMENTS"]
    DEFAULT_CONFIDENCE_THRESHOLD = hparams["DEFAULT_CONFIDENCE_THRESHOLD"]
    DEFAULT_DOMINANCE_MARGIN = hparams["DEFAULT_CONFIDENCE_MARGIN"]

    correct = 0
    total = len(dataset)

    # LLM initialization
    server = OllamaServer()
    wiki_llm = OllamaChat(server, model="gpt-oss:20b")
    reason_llm = OllamaChat(server, model="gpt-oss:20b")
    wiki_user = LLMUser(wiki_llm)
    reason_user = LLMUser(reason_llm)

    for example_idx, example in enumerate(dataset, 1):
        question_text = example["question"]
        choices = example["choices"]
        answer_key = example.get("answerKey")
        choice_facts = example["facts"]

        # -----------------------------
        # Example header
        # -----------------------------
        print("\n" + "="*60)
        print(f"[EXAMPLE {example_idx}] Question: {question_text}")
        print(f"Choices: {choices}")
        print(f"Answer key: {answer_key}")
        print(f"[HYPERPARAMS] TOP_PARAGRAPHS={TOP_PARAGRAPHS}, MAX_ARGUMENTS={MAX_ARGUMENTS}, "
              f"THRESHOLD={DEFAULT_CONFIDENCE_THRESHOLD}, DOM_MARGIN={DEFAULT_DOMINANCE_MARGIN}")
        print("="*60)

        # --- Robust candidate-pages normalization ---
        raw_pages = wiki_user.get_candidate_pages(question_text, choices, max_pages=10)

        # Case 1: Already a dict → OK
        if isinstance(raw_pages, dict):
            pages = raw_pages

        # Case 2: A list of tuples → convert to dict
        elif isinstance(raw_pages, list) and all(
            isinstance(x, (list, tuple)) and len(x) == 2 for x in raw_pages
        ):
            pages = {k: v for k, v in raw_pages}

        # Case 3: String or garbage → fallback to empty dict
        else:
            print("[WARN] get_candidate_pages() returned invalid structure, using empty {}")
            pages = {}

        wiki_pages = fetch_wikipedia_pages(pages)

        print(pages)

        # Argumentation graph
        G = ArgumentationGraph(debug=True)
        for label, fact_text in choice_facts.items():
            G.add_argument(fact_text, node_type="hypothesis", initial_strength=0.5)

        predicted_label = None
        last_strengths = {}
        stop = False

        # -----------------------------
        # Loop over pages
        # -----------------------------
        for page_title, page in wiki_pages.items():
            if not page:
                continue

            print(f"\n[EXAMPLE {example_idx}] Processing page: '{page_title}'")

            for page_section in wiki_pages[page_title]:
                if page_section == "Introduction":
                    intro_text = "\n".join(page.get("Introduction", [])) if isinstance(page.get("Introduction"), list) else str(page.get("Introduction", ""))
                    if intro_text.strip():
                        paras = rank_paragraphs_from_text(question_text, intro_text, TOP_PARAGRAPHS)
                        G.extend_from_text("\n".join(paras), reason_user, list(choice_facts.values()), max_arguments=MAX_ARGUMENTS)
                        print(f"  Introduction processed.")

                else:
                    
                    sections = page.get("Sections", {})
                    for sec_name, sec_content in sections.items():
                        section_text = "\n".join(sec_content) if isinstance(sec_content, list) else str(sec_content)
                        if not section_text.strip():
                            continue
                    paras = rank_paragraphs_from_text(question_text, section_text, TOP_PARAGRAPHS)
                    G.extend_from_text("\n".join(paras), reason_user, list(choice_facts.values()), max_arguments=MAX_ARGUMENTS)
                    print(f"   Section '{sec_name}' processed.")

                # Compute strengths after page
                strengths = compute_strengths_from_graph(G, choice_facts)
                strengths = {k: float(v) for k, v in strengths.items()}
                last_strengths = strengths.copy()
                #print(f"    [Strengths] {strengths}")

                # Early stopping
                if strengths:
                    sorted_strengths = sorted(strengths.items(), key=lambda x: float(x[1]), reverse=True)

                    print(f"    [Strengths] {sorted_strengths}")
                    top_label, top_val = sorted_strengths[0]
                    second_val = sorted_strengths[1][1] if len(sorted_strengths) > 1 else 0.0
                    margin = top_val - second_val

                    cond_confident = (top_val > (DEFAULT_CONFIDENCE_THRESHOLD - EPS))
                    cond_margin = (margin + EPS >= DEFAULT_DOMINANCE_MARGIN)
                    cond_absolute = (top_val > 0.95)

                    print(f"    [DEBUG] cond_confident={cond_confident}, cond_margin={cond_margin}, cond_absolute={cond_absolute}")

                    if cond_absolute or (cond_confident and cond_margin):
                        predicted_label = top_label
                        print(f"--> EARLY STOP: Predicted '{predicted_label}' with value={top_val:.6f} (margin={margin:.6f})")
                        stop = True

                if stop:
                    break

            if stop:
                break

        # -----------------------------
        # Fallback if no early stop
        # -----------------------------
        if predicted_label is None:
            sorted_strengths = sorted(last_strengths.items(), key=lambda x: float(x[1]), reverse=True)

            if not sorted_strengths:
                predicted_label = random.choice(list(choice_facts.keys()))
                print(f"--> FINAL CHOICE fallback: no strengths at all, choosing random '{predicted_label}'")
            else:
                best_value = sorted_strengths[0][1]
                best_candidates = [label for label, value in sorted_strengths if abs(value - best_value) < 1e-9]
                predicted_label = random.choice(best_candidates)
                print(f"--> FINAL CHOICE (no early stop): {predicted_label} (tied: {best_candidates}, value={best_value})")

        # -----------------------------
        # Accuracy logging
        # -----------------------------
        if predicted_label and answer_key and predicted_label.strip().lower() == answer_key.strip().lower():
            correct += 1
            print(f"[EXAMPLE {example_idx}] ✅ Correct (predicted {predicted_label} == gold {answer_key})")
        else:
            print(f"[EXAMPLE {example_idx}] ❌ Incorrect (predicted {predicted_label} != gold {answer_key})")

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy

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
