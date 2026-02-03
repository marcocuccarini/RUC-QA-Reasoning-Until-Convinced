import json
from pathlib import Path
import random
import re
import matplotlib.pyplot as plt
from itertools import product
import math
import time


# ---- Local imports ----
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

TOP_PARAGRAPHS = 5
MAX_ARGUMENTS = 4
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_DOMINANCE_MARGIN = 0.2  # can be tuned
EPS = 1e-9  # small epsilon for float comparisons
ABSOLUTE_STOP = 0.95

def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def export_graph(G: ArgumentationGraph, path):
    data = {"nodes": {}, "edges": []}
    for nid, attrs in G.G.nodes(data=True):
        data["nodes"][nid] = {
            "type": attrs.get("type", ""),
            "text": attrs.get("text", ""),
            "strength": float(attrs.get("strength", 0.5))
        }
    for src, tgt, attrs in G.G.edges(data=True):
        data["edges"].append({
            "source": src,
            "target": tgt,
            "relation": attrs.get("relation", "")
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

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

def compute_prediction_from_graph(graph_json, choice_facts):
    strengths = {}
    for label, fact in choice_facts.items():
        fn = normalize(fact)
        for node in graph_json.get("nodes", {}).values():
            if normalize(node.get("text", "")).endswith(fn):
                strengths[label] = float(node.get("strength", 0.0))
    if not strengths:
        return None, {}
    best = max(strengths.values())
    return random.choice([k for k, v in strengths.items() if abs(v - best) < EPS]), strengths

# -----------------------------
# Compute strengths from graph (robust)
# -----------------------------

def compute_strengths_from_graph(G: ArgumentationGraph, choice_facts: dict):
    raw = G.compute_strengths()
    strengths = {}
    for nid, val in raw.items():
        node_txt = (G.get_text_from_id(nid) or "").strip().lower()
        if not node_txt:
            continue
        try:
            val = float(val)
        except Exception:
            val = 0.0
        for k, fact in choice_facts.items():
            if node_txt == fact.strip().lower():
                strengths[k] = max(strengths.get(k, 0.0), val)
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


# =====================================================
# Main test pipeline WITH EARLY STOP
# =====================================================
def run_on_test(best_hparams, dataset, graph_dir, use_intro_only=True, dataset_sample=None):

    TOP_PARAGRAPHS = best_hparams["TOP_PARAGRAPHS"]
    MAX_ARGUMENTS = best_hparams["MAX_ARGUMENTS"]
    CONF_THR = best_hparams["DEFAULT_CONFIDENCE_THRESHOLD"]
    MARGIN = best_hparams["DEFAULT_CONFIDENCE_MARGIN"]

    if dataset_sample:
        dataset = dataset[:dataset_sample]

    server = OllamaServer()
    wiki_user = LLMUser(OllamaChat(server, model=model))
    reason_user = LLMUser(OllamaChat(server, model=model))

    graph_dir = Path(graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    predictions = []

    for idx, ex in enumerate(dataset, 1):
        print(f"\n[Q{idx}] {ex['question']}")
        choice_facts = ex["facts"]
        answer_key = ex.get("answerKey")
        graph_path = graph_dir / f"graph_{idx}.json"

        # ---------------- LOAD EXISTING GRAPH ----------------
        if graph_path.exists():
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_json = json.load(f)
            predicted, _ = compute_prediction_from_graph(graph_json, choice_facts)
            graph_changed = any(
                abs(n.get("strength", 0.5) - 0.5) > EPS
                for n in graph_json.get("nodes", {}).values()
                if n.get("type") == "hypothesis"
            )
            if not graph_changed:
                predicted = random.choice(list(choice_facts.keys()))

        # ---------------- BUILD GRAPH WITH EARLY STOP ----------------
        else:
            G = ArgumentationGraph(debug=True)
            for fact in choice_facts.values():
                G.add_argument(fact, node_type="hypothesis", initial_strength=0.5)

            pages = wiki_user.get_candidate_pages(ex["question"], ex["choices"], max_pages=10)
            if isinstance(pages, list):
                pages = dict(pages)
            if not isinstance(pages, dict):
                pages = {}

            wiki_pages = fetch_wikipedia_pages(pages)
            predicted = None
            stop = False

            print("Pages", pages)

            for page_title, page in wiki_pages.items():
                if not page:
                    continue

                print(f"\n[EXAMPLE {idx}] Processing page: '{page_title}'")

                for page_section in wiki_pages[page_title]:
                    if page_section == "Introduction":
                        intro_text = "\n".join(page.get("Introduction", [])) if isinstance(page.get("Introduction"), list) else str(page.get("Introduction", ""))
                        if intro_text.strip():
                            paras = rank_paragraphs_from_text(ex['question'], intro_text, TOP_PARAGRAPHS)
                            G.extend_from_text("\n".join(paras), reason_user, list(choice_facts.values()), max_arguments=MAX_ARGUMENTS)
                            print(f"  Introduction processed.")

                    else:
                        
                        sections = page.get("Sections", {})
                        for sec_name, sec_content in sections.items():
                            section_text = "\n".join(sec_content) if isinstance(sec_content, list) else str(sec_content)
                            if not section_text.strip():
                                continue
                        paras = rank_paragraphs_from_text(ex['question'], section_text, TOP_PARAGRAPHS)
                        G.extend_from_text("\n".join(paras), reason_user, list(choice_facts.values()), max_arguments=MAX_ARGUMENTS)
                        print(f"   Section processed.")

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

                        cond_confident = (top_val > (CONF_THR - EPS))
                        cond_margin = (margin + EPS >= MARGIN)
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

            if predicted is None:
                strengths = compute_strengths_from_graph(G, choice_facts)
                predicted = max(strengths, key=strengths.get) if strengths else random.choice(list(choice_facts.keys()))

            export_graph(G, graph_path)
            graph_changed = True

        correct = predicted.lower() == (answer_key or "").lower()
        predictions.append({
            "predicted": predicted,
            "gold": answer_key,
            "correct": correct,
            "graph_changed": graph_changed
        })

        print(f"→ Predicted={predicted} | Gold={answer_key} | Correct={correct}")

    accuracy = 100 * sum(p["correct"] for p in predictions) / len(predictions)
    return predictions, accuracy