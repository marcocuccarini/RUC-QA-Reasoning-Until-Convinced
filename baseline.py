import json
from pathlib import Path
import re

from classes.ServerOllama import OllamaServer, OllamaChat
from classes.utils import rank_paragraphs_from_text

# =====================================================
# CONFIG
# =====================================================
MODELS = ["gpt-oss:20b"]

DATASETS = [
    "plausibleqa_test.json",
    #"sciq_test.json",
    #"arc-easy_test.json"
]

TOP_PARAGRAPHS = [2, 5, 10, 15]
SAVE_EVERY = 100
DATASET_SAMPLE = 10000  
SPLIT_DIR = Path("split_datasets")
RESULTS_DIR = Path("results_retrieval")
RESULTS_DIR.mkdir(exist_ok=True)

# =====================================================
# HELPERS
# =====================================================
def normalize(text):
    """Standardizes text for comparison (lowercase, stripped)."""
    if not text: return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())

def ask_question_with_context(chat, ex, context_paragraphs):
    """
    Maps original keys to A, B, C, D for the LLM, 
    then maps the model's letter choice back to the original key.
    """
    question = ex["question"]
    choices = ex["choices"]
    
    # Mapping Layer: {'A': 'the Sun', 'B': 'the Little Dipper', ...}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    mapping = {letters[i]: k for i, k in enumerate(choices.keys())}
    
    prompt = ""
    if context_paragraphs:
        prompt += "Use the following context to answer the question:\n"
        prompt += "\n".join(context_paragraphs) + "\n\n"
    
    prompt += f"Question: {question}\nChoices:\n"
    for letter, original_key in mapping.items():
        # This shows the model "A) the Sun: The Sun remains..."
        prompt += f"{letter}) {original_key}: {choices[original_key]}\n"
    
    prompt += "\nAnswer with the letter (A, B, C, or D) of the correct choice. Provide ONLY the letter."
    
    resp = chat.send_prompt(prompt)
    ans_raw = resp.raw_text.strip().upper()
    
    # Robust Regex: Look for the first isolated letter A, B, C, or D
    match = re.search(r'\b([A-D])\b', ans_raw)
    
    if match:
        letter_picked = match.group(1)
        if letter_picked in mapping:
            return mapping[letter_picked]
    
    return "INVALID"

def compute_accuracy(lst):
    if not lst: return 0.0
    return 100.0 * sum(x["correct"] for x in lst) / len(lst)

def print_step_accuracy(results, step):
    print(f"\n📊 ACCURACY @ STEP {step}")
    print(f" No context: {compute_accuracy(results['no_context']):.2f}%")
    for n in TOP_PARAGRAPHS:
        print(f" Top {n}: {compute_accuracy(results['with_context'][n]):.2f}%")

def load_wiki_pages(wiki_file, question_index):
    if not wiki_file.exists(): return {}
    try:
        with open(wiki_file, "r", encoding="utf-8") as f:
            all_wiki_data = json.load(f)
        return all_wiki_data.get(str(question_index), {})
    except:
        return {}

def extract_paragraphs_from_wiki_data(wiki_data, question, top_k):
    if not wiki_data: return []
    all_paragraphs = []
    for page_title, content in wiki_data.items():
        if not isinstance(content, dict): continue
        
        intro = content.get("Introduction", [])
        intro_text = "\n".join(intro) if isinstance(intro, list) else str(intro)
        
        sections = content.get("Sections", {})
        section_texts = [
            ("\n".join(c) if isinstance(c, list) else str(c)) 
            for c in sections.values()
        ]
        
        full_text = intro_text + "\n" + "\n".join(section_texts)
        if full_text.strip():
            ranked = rank_paragraphs_from_text(question, full_text, top_k=top_k)
            all_paragraphs.extend(ranked)
    
    return all_paragraphs[:top_k]

# =====================================================
# MAIN LOOP
# =====================================================
for model_name in MODELS:
    print(f"\n=== MODEL: {model_name} ===")
    server = OllamaServer()
    chat = OllamaChat(server, model=model_name)
    
    model_dir = RESULTS_DIR / model_name.replace(":", "_")
    model_dir.mkdir(exist_ok=True)
    
    for dataset_file in DATASETS:
        dataset_path = SPLIT_DIR / dataset_file
        if not dataset_path.exists():
            print(f"Skipping {dataset_file}, file not found.")
            continue

        dataset_name = Path(dataset_file).stem
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        if DATASET_SAMPLE:
            dataset = dataset[:DATASET_SAMPLE]
        
        pred_file = model_dir / f"{dataset_name}_predictions.json"
        wiki_file = model_dir / f"{dataset_name}_wiki_pages.json"
        
        # Resume Logic
        if pred_file.exists():
            with open(pred_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results = saved["results"]
                results["with_context"] = {int(k): v for k, v in results["with_context"].items()}
                start_idx = len(results["no_context"]) + 1
        else:
            results = {"no_context": [], "with_context": {n: [] for n in TOP_PARAGRAPHS}}
            start_idx = 1
        
        for idx in range(start_idx, len(dataset) + 1):
            ex = dataset[idx - 1]
            print(f"\n[Q{idx}] {ex['question']}")
            
            # --- 1. BASELINE (NO CONTEXT) ---
            # This now uses the A,B,C,D layer automatically
            base_ans = ask_question_with_context(chat, ex, [])
            base_correct = normalize(base_ans) == normalize(ex["answerKey"])
            
            results["no_context"].append({
                "index": ex["index"],
                "model_answer": base_ans,
                "correct_answer": ex["answerKey"],
                "correct": base_correct
            })
            print(f" [NO CONTEXT] {base_ans} | Correct: {base_correct}")
            
            # --- 2. RAG (WITH CONTEXT) ---
            wiki_data = load_wiki_pages(wiki_file, ex["index"])
            
            for n in TOP_PARAGRAPHS:
                if not wiki_data:
                    # Fallback if no wiki info exists
                    ans, correct, paras = base_ans, base_correct, []
                else:
                    paras = extract_paragraphs_from_wiki_data(wiki_data, ex["question"], n)
                    ans = ask_question_with_context(chat, ex, paras)
                    correct = normalize(ans) == normalize(ex["answerKey"])
                
                print(f" [TOP {n}] {ans} | Correct: {correct}")
                results["with_context"][n].append({
                    "index": ex["index"],
                    "model_answer": ans,
                    "correct_answer": ex["answerKey"],
                    "correct": correct,
                    "context_used": paras
                })
            
            # Auto-save
            if idx % SAVE_EVERY == 0:
                with open(pred_file, "w", encoding="utf-8") as f:
                    json.dump({"results": results}, f, indent=2, ensure_ascii=False)
                print_step_accuracy(results, idx)
        
        # Final Save per Dataset
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print_step_accuracy(results, len(dataset))