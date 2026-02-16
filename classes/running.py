import json
from pathlib import Path
import random
import re
import matplotlib.pyplot as plt
from itertools import product
import math
import time
from configuration.hyperparameters import *


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
def save_detailed_predictions(results_dir, dataset_name, hparams, predictions):
    """Saves a JSON file containing every prediction for a specific hparam set."""
    # Create a unique filename based on hparams
    hparam_str = (
        f"P{hparams['TOP_PARAGRAPHS']}_"
        f"A{hparams['MAX_ARGUMENTS']}_"
        f"T{hparams['DEFAULT_CONFIDENCE_THRESHOLD']}_"
        f"M{hparams['DEFAULT_CONFIDENCE_MARGIN']}"
    )
    
    # Path: RESULTS_DIR / detailed_logs / sciq / preds_P5_A2...json
    log_dir = results_dir / "detailed_logs" / dataset_name.lower()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = log_dir / f"preds_{hparam_str}.json"
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Detailed logs saved to {file_path}")



def run_on_validation(hparams: dict, dataset: list, model="gpt-oss:20b"):
    global TOP_PARAGRAPHS, MAX_ARGUMENTS, DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_DOMINANCE_MARGIN
    TOP_PARAGRAPHS = hparams["TOP_PARAGRAPHS"]
    MAX_ARGUMENTS = hparams["MAX_ARGUMENTS"]
    DEFAULT_CONFIDENCE_THRESHOLD = hparams["DEFAULT_CONFIDENCE_THRESHOLD"]
    DEFAULT_DOMINANCE_MARGIN = hparams["DEFAULT_CONFIDENCE_MARGIN"]

    correct = 0
    total = len(dataset)
    detailed_results = []  # List to store the detailed log for each example

    # LLM initialization
    server = OllamaServer()
    wiki_llm = OllamaChat(server, model=model)
    reason_llm = OllamaChat(server, model=model)
    wiki_user = LLMUser(wiki_llm)
    reason_user = LLMUser(reason_llm)

    for example_idx, example in enumerate(dataset, 1):
        question_text = example["question"]
        choices = example["choices"]
        answer_key = example.get("answerKey")
        choice_facts = example["facts"]

        # --- CRITICAL: Initialize variables to avoid NameError ---
        predicted_label = None 
        last_strengths = {}
        stop = False
        # -------------------------------------------------------

        print("\n" + "="*60)
        print(f"[EXAMPLE {example_idx}] Question: {question_text}")
        print(f"Choices: {choices}")
        print(f"Answer key: {answer_key}")
        print("="*60)

        # Candidate pages
        raw_pages = wiki_user.get_candidate_pages(question_text, choices, max_pages=10)
        
        # Robust normalization of pages
        if isinstance(raw_pages, dict):
            pages = raw_pages
        elif isinstance(raw_pages, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw_pages):
            pages = {k: v for k, v in raw_pages}
        else:
            pages = {}

        wiki_pages = fetch_wikipedia_pages(pages)

        # Argumentation graph
        G = ArgumentationGraph(debug=True)
        for label, fact_text in choice_facts.items():
            G.add_argument(fact_text, node_type="hypothesis", initial_strength=0.5)

        # Processing pages
        for page_title, page in wiki_pages.items():
            if not page: continue
            
            # Process sections (Introduction + others)
            all_sections = [("Introduction", page.get("Introduction", []))] + list(page.get("Sections", {}).items())
            
            for sec_name, sec_content in all_sections:
                section_text = "\n".join(sec_content) if isinstance(sec_content, list) else str(sec_content)
                if not section_text.strip(): continue
                
                paras = rank_paragraphs_from_text(question_text, section_text, TOP_PARAGRAPHS)
                G.extend_from_text("\n".join(paras), reason_user, list(choice_facts.values()), max_arguments=MAX_ARGUMENTS)
                
                # Update strengths
                strengths = compute_strengths_from_graph(G, choice_facts)
                last_strengths = {k: float(v) for k, v in strengths.items()}

                # Check for Early Stopping
                if last_strengths:
                    sorted_s = sorted(last_strengths.items(), key=lambda x: x[1], reverse=True)
                    top_label, top_val = sorted_s[0]
                    second_val = sorted_s[1][1] if len(sorted_s) > 1 else 0.0
                    margin = top_val - second_val

                    cond_confident = (top_val > (DEFAULT_CONFIDENCE_THRESHOLD - 1e-9))
                    cond_margin = (margin + 1e-9 >= DEFAULT_DOMINANCE_MARGIN)
                    cond_absolute = (top_val > 0.95)

                    if cond_absolute or (cond_confident and cond_margin):
                        predicted_label = top_label
                        print(f"--> EARLY STOP: {predicted_label} (val={top_val:.4f})")
                        stop = True
                        break
            if stop: break

        # --- Fallback Logic if no early stop occurred ---
        if predicted_label is None:
            if not last_strengths:
                predicted_label = random.choice(list(choice_facts.keys()))
                print(f"--> FALLBACK: Random choice '{predicted_label}'")
            else:
                sorted_s = sorted(last_strengths.items(), key=lambda x: x[1], reverse=True)
                best_val = sorted_s[0][1]
                candidates = [l for l, v in sorted_s if abs(v - best_val) < 1e-9]
                predicted_label = random.choice(candidates)
                print(f"--> FALLBACK: Best strength '{predicted_label}'")

        # --- Final Scoring and Logging ---
        is_correct = False
        if predicted_label and answer_key and str(predicted_label).strip().lower() == str(answer_key).strip().lower():
            correct += 1
            is_correct = True
            print(f"✅ CORRECT")
        else:
            print(f"❌ INCORRECT (Gold: {answer_key})")

        detailed_results.append({
            "predicted": choices.get(predicted_label, "None"),
            "gold": choices.get(answer_key, "None"),
            "correct": is_correct,
            "label_predicted": predicted_label,
            "label_gold": answer_key,
            "graph_changed": True
        })

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, detailed_results

# =====================================================
# Main test pipeline WITH EARLY STOP
# =====================================================
def run_on_test(best_hparams, dataset, graph_dir, use_intro_only=True, dataset_sample=None, model="gpt-oss:20b"):

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