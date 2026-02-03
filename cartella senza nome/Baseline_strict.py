import json
import re
from pathlib import Path
import random

from classes.ServerOllama import OllamaServer, OllamaChat
from classes.LLMUser import LLMUser
from classes.utils import fetch_wikipedia_pages, rank_paragraphs_from_text

# =====================================================
# CONFIG
# =====================================================
MODELS = ["qwen3:14b"]

DATASETS = [
    "plausibleqa_test.json",
    "sciq_test.json",
    "arc-easy_test.json"
]

import ollama


class OllamaServer:
    """
    Thin wrapper. Keeps compatibility with existing code.
    """
    def __init__(self):
        pass


class OllamaChat:
    def __init__(self, server, model):
        self.server = server
        self.model = model

    def send_prompt(
        self,
        prompt,
        temperature=0.0,
        top_p=0.05,
        max_tokens=1,
        stop=None,
    ):
        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                **({"stop": stop} if stop else {})
            }
        )

        # Normalize response object
        if isinstance(response, dict):
            content = response["message"]["content"]
        else:
            content = response.message.content

        class Resp:
            def __init__(self, text):
                self.raw_text = text

        return Resp(content)


TOP_PARAGRAPHS = [2, 5, 10, 15]
SAVE_EVERY = 100
DATASET_SAMPLE = 400  # None for full dataset

SPLIT_DIR = Path("split_datasets")
RESULTS_DIR = Path("results_retrieval")
RESULTS_DIR.mkdir(exist_ok=True)

# =====================================================
# UTILS
# =====================================================
def normalize(text):
    """Normalize string for comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())

def extract_choice(raw_text, choices):
    """Extract A/B/C/D from raw model output."""
    if not raw_text:
        return "INVALID"
    text = raw_text.strip().upper()
    if text in choices:
        return text
    # Check if any choice letter appears in the text
    pattern = r"\b(" + "|".join(map(re.escape, choices.keys())) + r")\b"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    # fallback: first letter of text
    if text and text[0] in choices:
        return text[0]
    return "INVALID"

def get_correct_letter(ex):
    """Return the letter corresponding to the correct answer, safely."""
    answer_norm = normalize(ex["answerKey"])
    for k, v in ex["choices"].items():
        if normalize(v) == answer_norm:
            return k
    # fallback if not found
    return list(ex["choices"].keys())[0]

def ask_question_with_context(chat, ex, context_paragraphs):
    """Strict MCQ prompt with context."""
    question = ex["question"]
    choices = ex["choices"]

    prompt = ""
    if context_paragraphs:
        prompt += "Use the following context to answer the question:\n"
        prompt += "\n".join(context_paragraphs) + "\n\n"

    prompt += f"Question: {question}\nChoices:\n"
    for k, v in choices.items():
        prompt += f"- {k}: {v}\n"

    prompt += (
        "\nAnswer with ONLY ONE LETTER.\n"
        f"Valid answers: {', '.join(choices.keys())}.\n"
        "Do not explain. Do not add text.\n"
    )

    resp = chat.send_prompt(
        prompt,
        temperature=0.0,
        top_p=0.05,
        max_tokens=1,
        stop=["\n", " ", ".", ")", ":"]
    )

    return extract_choice(resp.raw_text, choices)

def ask_with_retry(chat, ex, context, retries=1):
    """Retry until a valid choice is returned."""
    for _ in range(retries + 1):
        ans = ask_question_with_context(chat, ex, context)
        if ans != "INVALID":
            return ans
    # fallback: first choice
    return list(ex["choices"].keys())[0]

def compute_accuracy(lst):
    if not lst:
        return 0.0
    return 100.0 * sum(x["correct"] for x in lst) / len(lst)

def print_step_accuracy(results, step):
    print(f"\n📊 ACCURACY @ STEP {step}")
    print(f" No context: {compute_accuracy(results['no_context']):.2f}%")
    for n in TOP_PARAGRAPHS:
        print(f" Top {n}: {compute_accuracy(results['with_context'][n]):.2f}%")

# =====================================================
# MCQ CONVERSION
# =====================================================
def to_mcq(ex):
    """Convert any question into A/B/C/D MCQ format if not already."""
    if "choices" in ex:
        return ex
    distractors = ["Option1", "Option2", "Option3", "Option4"]
    random.shuffle(distractors)
    choices = {"A": ex["answerKey"]}
    letters = ["B", "C", "D"]
    for i, l in enumerate(letters):
        distractor = distractors[i]
        if distractor == ex["answerKey"]:
            distractor += " X"
        choices[l] = distractor
    ex["choices"] = choices
    return ex

# =====================================================
# MAIN LOOP
# =====================================================
for model_name in MODELS:
    print(f"\n=== MODEL: {model_name} ===")

    server = OllamaServer()
    chat = OllamaChat(server, model=model_name)
    wiki_user = LLMUser(chat)

    model_dir = RESULTS_DIR / model_name.replace(":", "_")
    model_dir.mkdir(exist_ok=True)

    for dataset_file in DATASETS:
        dataset_name = Path(dataset_file).stem
        print(f"\n=== DATASET: {dataset_name} ===")

        with open(SPLIT_DIR / dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if DATASET_SAMPLE:
            dataset = dataset[:DATASET_SAMPLE]

        pred_file = model_dir / f"{dataset_name}_predictions.json"
        wiki_file = model_dir / f"{dataset_name}_wiki_pages.json"

        # ---------- RESUME ----------
        if pred_file.exists():
            with open(pred_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                results = saved["results"]
                results["with_context"] = {int(k): v for k, v in results["with_context"].items()}
                for n in TOP_PARAGRAPHS:
                    results["with_context"].setdefault(n, [])
                start_idx = len(results["no_context"]) + 1
                print(f"🔁 Resuming from question {start_idx}")
        else:
            results = {"no_context": [], "with_context": {n: [] for n in TOP_PARAGRAPHS}}
            start_idx = 1

        if wiki_file.exists():
            with open(wiki_file, "r", encoding="utf-8") as f:
                wiki_pages = json.load(f)
        else:
            wiki_pages = {}

        # ---------- PROCESS ----------
        for idx in range(start_idx, len(dataset) + 1):
            ex = to_mcq(dataset[idx - 1])
            qid = str(ex["index"])
            print(f"\n==============================")
            print(f"[Q{idx}] {ex['question']}")
            print(f"Gold: {ex['answerKey']}")

            # ===== BASELINE =====
            base_ans = ask_with_retry(chat, ex, [])
            correct_letter = get_correct_letter(ex)
            base_correct = base_ans.upper() == correct_letter
            print(f" [NO CONTEXT] {base_ans} | {base_correct}")

            results["no_context"].append({
                "index": ex["index"],
                "question": ex["question"],
                "model_answer": base_ans,
                "correct_answer": correct_letter,
                "correct": base_correct
            })

            # ===== RETRIEVAL =====
            if qid in wiki_pages:
                fetched = wiki_pages[qid]
                pages = {k: None for k in fetched.keys()}
                print(" Wikipedia pages: (cached)")
            else:
                pages = wiki_user.get_candidate_pages(ex["question"], ex["choices"], max_pages=5)
                if isinstance(pages, list):
                    pages = dict(pages) if pages else {}
                fetched = fetch_wikipedia_pages(pages)
                wiki_pages[qid] = fetched
                print(f" Wikipedia pages: {list(pages.keys()) if pages else 'None'}")

            if not fetched:
                for n in TOP_PARAGRAPHS:
                    results["with_context"][n].append({
                        "index": ex["index"],
                        "question": ex["question"],
                        "model_answer": base_ans,
                        "correct_answer": correct_letter,
                        "correct": base_correct,
                        "context_used": [],
                        "wiki_pages": []
                    })
                continue

            # ===== RANK PARAGRAPHS =====
            all_paragraphs = []
            for content in fetched.values():
                intro = "\n".join(content.get("Introduction", []))
                sections = content.get("Sections", {})
                section_text = "\n".join(
                    "\n".join(v) if isinstance(v, list) else str(v) for v in sections.values()
                )
                text = intro + "\n" + section_text
                ranked = rank_paragraphs_from_text(ex["question"], text, top_k=max(TOP_PARAGRAPHS))
                all_paragraphs.extend(ranked)

            # ===== ANSWER WITH CONTEXT =====
            for n in TOP_PARAGRAPHS:
                ctx = all_paragraphs[:n]
                ans = ask_with_retry(chat, ex, ctx)
                correct_letter = get_correct_letter(ex)
                correct = ans.upper() == correct_letter
                print(f" [TOP {n}] {ans} | {correct}")

                results["with_context"][n].append({
                    "index": ex["index"],
                    "question": ex["question"],
                    "model_answer": ans,
                    "correct_answer": correct_letter,
                    "correct": correct,
                    "context_used": ctx,
                    "wiki_pages": list(fetched.keys())
                })

            # ===== SAVE =====
            if idx % SAVE_EVERY == 0:
                with open(pred_file, "w", encoding="utf-8") as f:
                    json.dump({"results": results}, f, indent=2, ensure_ascii=False)
                with open(wiki_file, "w", encoding="utf-8") as f:
                    json.dump(wiki_pages, f, indent=2, ensure_ascii=False)
                print_step_accuracy(results, idx)

        # ---------- FINAL SAVE ----------
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        with open(wiki_file, "w", encoding="utf-8") as f:
            json.dump(wiki_pages, f, indent=2, ensure_ascii=False)

        print_step_accuracy(results, len(dataset))
        print(f"✅ Finished {dataset_name}")
