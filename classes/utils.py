import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import json
from configuration.hyperparameters import *

# ======================================================
# Configurazione modello embedding
# ======================================================
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Ranking model device: {device}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

# ======================================================
# Funzioni di utilità di base
# ======================================================
def vprint(*args, **kwargs):
    """Print only when VERBOSE=True."""
    if VERBOSE:
        print(*args, **kwargs)


def sanitize_for_windows(name: str) -> str:
    """Rimuove caratteri non validi per nomi file Windows."""
    return re.sub(r'[<>:\\"/\\|?*]', "_", name)


def _ensure_ordered_items(sections: Dict[str, List[str]]) -> List[Tuple[str, List[str]]]:
    """Assicura un ordine coerente per le sezioni."""
    if isinstance(sections, list):
        return sections
    return list(sections.items())


def is_summary_section_name(name: str) -> bool:
    """Verifica se un nome di sezione è considerato 'summary'."""
    if name is None:
        return True
    return name.strip().lower() in SUMMARY_SECTION_ALIASES


# ======================================================
# Dataset
# ======================================================
def safe_load_dataset(cfg):
    """Carica dataset HuggingFace o JSON locale in base alla configurazione."""
    print(cfg)
    if cfg["flag"]:
        ds = load_dataset(cfg["hf_name"], cfg.get("config"))
        if cfg["split"] in ds:
            ds = ds[cfg["split"]]
        else:
            ds = next(iter(ds.values()))
        print(f"✅ Loaded {cfg['hf_name']} ({len(ds)} samples)")
        return ds
    else:
        with open("data/PlausibleQA.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✅ Loaded {cfg['hf_name']} ({len(data)} samples)")
        return data


def load_or_aggregate_facts(dataset_name: str) -> Dict[str, Any]:
    """Carica i fatti pre-aggregati se presenti."""
    agg_dir = Path("aggregated_fact")
    agg_dir.mkdir(exist_ok=True)
    file_base = dataset_aliases.get(dataset_name, dataset_name.lower())
    file_path = agg_dir / f"{file_base}_preprocessed_fact.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ======================================================
# Dataset format detection
# ======================================================
def detect_dataset_format(example: Dict[str, Any], facts_data: Dict[str, Any], index: int) -> Tuple[Dict[str, str], str]:
    """Restituisce choices e answer_key basati sul dataset."""
    choices: Dict[str, str] = {}
    answer_key = ""

    # SciQ style
    if all(k in example for k in ["correct_answer", "distractor1", "distractor2", "distractor3"]):
        choice_texts = [
            example["correct_answer"],
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
        ]
        facts_for_example = facts_data.get(str(index), {})
        choices = {text: facts_for_example.get(text, text) for text in choice_texts}
        answer_key = example["correct_answer"]

    # ARC style
    elif "choices" in example and "answerKey" in example:
        choice_texts = example["choices"]["text"]
        choice_labels = example["choices"]["label"]
        facts_for_example = facts_data.get(str(index), {})
        choices = {text: facts_for_example.get(text, text) for text in choice_texts}
        label_to_text = dict(zip(choice_labels, choice_texts))
        answer_key = label_to_text.get(example["answerKey"], "")

    # MedMCQA style
    elif all(k in example for k in ["opa", "opb", "opc", "opd", "cop"]):
        choice_texts = [example["opa"], example["opb"], example["opc"], example["opd"]]
        facts_for_example = facts_data.get(str(index), {})
        choices = {text: facts_for_example.get(text, text) for text in choice_texts}
        try:
            answer_key = choice_texts[int(example["cop"]) - 1]
        except Exception:
            answer_key = ""

    # PlausibleQA style
    elif "question_type" in example and "candidate_answers" in example and "answer" in example:
        choice_texts = list(example["candidate_answers"].keys())
        facts_for_example = facts_data.get(str(index), {})
        choices = {text: facts_for_example.get(text, str(example["candidate_answers"][text])) for text in choice_texts}
        if isinstance(example["answer"], dict) and "text" in example["answer"]:
            answer_key = example["answer"]["text"]
        else:
            answer_key = str(example["answer"])

    # Fallback generico
    else:
        choices = example.get("options", {})
        answer_key = example.get("answer", "")

    # Sicurezza finale
    if not answer_key and choices:
        answer_key = list(choices.keys())[0]

    return choices, answer_key


# ======================================================
# Paragraph ranking
# ======================================================
def rank_paragraphs_from_text(question: str, text: str, top_k: int = RANK_TOP_K, chunk_size: int = 5) -> List[str]:
    """
    Suddivide il testo in chunk di ~chunk_size frasi, calcola similitudine con la domanda,
    e restituisce i top-k paragrafi.
    """
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    # Merge sentences into chunks
    paras = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    paras = [p for p in paras if len(p.split()) > 10]
    if not paras:
        return []

    # Embedding
    q_emb = embedding_model.encode(question, convert_to_tensor=True, device=device)
    para_emb = embedding_model.encode(paras, convert_to_tensor=True, device=device)

    # Cosine similarity
    sims = util.cos_sim(q_emb, para_emb)[0]
    scores = sims.cpu().tolist()

    # Rank top-k
    sorted_idx = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    top_idx = sorted_idx[:min(top_k, len(scores))]

    return [paras[i] for i in top_idx]


# ======================================================
# Wikipedia fetching
# ======================================================
import wikipediaapi
from requests.exceptions import RequestException
import time

def fetch_wikipedia_pages(page_titles, language='en', max_retries=3, sleep_between=2):
    """
    Fetch Wikipedia pages, restituisce:
    {
        "Title": {
            "Introduction": [...],
            "Sections": {...}
        }
    }
    """
    wiki = wikipediaapi.Wikipedia(
        language=language,
        user_agent='MyWikipediaFetcher/1.0 (https://github.com/yourname; your_email@example.com)'
    )
    pages_content = {}

    def recurse_sections(sections):
        content = {}
        for s in sections:
            content[s.title] = [p for p in s.text.split("\n\n") if p.strip()]
            content.update(recurse_sections(s.sections))
        return content

    for title in page_titles:
        page_title = title[0] if isinstance(title, (list, tuple)) else title
        for attempt in range(max_retries):
            try:
                page = wiki.page(page_title)
                if not page.exists():
                    pages_content[page_title] = {}
                    break

                # Introduction
                lines = page.text.split("\n")
                intro_lines = []
                for line in lines:
                    if line.strip().startswith("==") and line.strip().endswith("=="):
                        break
                    if line.strip():
                        intro_lines.append(line.strip())
                introduction_paragraphs = [p for p in "\n".join(intro_lines).split("\n\n") if p.strip()]

                # Sections
                sections = recurse_sections(page.sections)
                if not sections:
                    sections = {"Body": [p for p in page.text.split("\n\n") if p.strip()]}

                pages_content[page_title] = {
                    "Introduction": introduction_paragraphs,
                    "Sections": sections
                }
                break

            except RequestException:
                time.sleep(sleep_between)
            except Exception:
                pages_content[page_title] = {}
                break

        else:
            pages_content[page_title] = {}

    return pages_content
