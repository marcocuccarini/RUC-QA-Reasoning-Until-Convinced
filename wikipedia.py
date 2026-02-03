import json
from pathlib import Path

from classes.ServerOllama import OllamaServer, OllamaChat
from classes.LLMUser import LLMUser
from classes.utils import fetch_wikipedia_pages

# =====================================================
# CONFIG
# =====================================================
MODELS = ["qwen3:14b"]

DATASETS = [
    "plausibleqa_test.json",
    "sciq_test.json",
    "arc-easy_test.json"
]

DATASET_SAMPLE = 10000  # set None for full dataset

SPLIT_DIR = Path("split_datasets")
RESULTS_DIR = Path("results_wikipedia_only")
RESULTS_DIR.mkdir(exist_ok=True)

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

        wiki_file = model_dir / f"{dataset_name}_wiki_pages.json"

        # ---------- RESUME ----------
        if wiki_file.exists():
            with open(wiki_file, "r", encoding="utf-8") as f:
                wiki_pages = json.load(f)
            print(f"🔁 Resuming, already have {len(wiki_pages)} questions")
        else:
            wiki_pages = {}

        # ---------- PROCESS ----------
        for idx, ex in enumerate(dataset, start=1):
            qid = str(ex["index"])

            if qid in wiki_pages:
                continue  # already processed

            print(f"\n[Q{idx}] {ex['question']}")

            pages = wiki_user.get_candidate_pages(
                ex["question"],
                ex["choices"],
                max_pages=5
            )

            # Normalize pages → dict
            if isinstance(pages, list):
                try:
                    pages = dict(pages)
                except ValueError:
                    pages = {}

            if not isinstance(pages, dict) or not pages:
                print(" ⚠️ No Wikipedia pages found")
                wiki_pages[qid] = {}
                continue

            print(f" Wikipedia pages: {list(pages.keys())}")

            fetched = fetch_wikipedia_pages(pages)
            wiki_pages[qid] = fetched

            # ---------- SAVE (incremental) ----------
            with open(wiki_file, "w", encoding="utf-8") as f:
                json.dump(wiki_pages, f, indent=2, ensure_ascii=False)

        print(f"✅ Finished Wikipedia generation for {dataset_name}")
