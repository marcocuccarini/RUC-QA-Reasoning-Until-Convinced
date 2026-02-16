# ======================================================
# Configuration
# ======================================================
import os
from pathlib import Path
import re

HF_TOKEN = os.getenv("HF_TOKEN")
WIKI_MODEL = os.getenv("WIKI_MODEL", "qwen3:8b")
REASON_MODEL = os.getenv("REASON_MODEL", "qwen3:8b")

def sanitize_for_windows(name: str) -> str:
    return re.sub(r'[<>:\\"/\\|?*]', "_", name)

# Toggle verbose per-sample printing
VERBOSE = True
safe_reason_model = sanitize_for_windows(REASON_MODEL)




SECTION_EVAL_MODE = "aggregate"  # options: "single" or "aggregate"




# Se True, rimuove i nodi che non hanno archi (né attacco né supporto)
PRUNE_ISOLATED_ARGUMENTS = True


datasets_config = {
    "ARC-Easy": {"flag": True, "hf_name": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test"},
    "SciQ": {"flag": True, "hf_name": "allenai/sciq", "split": "test"},
    "PlausibleQA": {"flag": False, "hf_name": "data/plausibleQA.json"},
}

models=["gpt-oss:20b","qwen3:14b"]


# Maps dataset names to shorter / filesystem-safe identifiers
dataset_aliases = {
    "PlausibleQA": "plausibleqa",
    "SciQ": "sciq",
    "ARC-Easy": "arc-easy"}

MAX_PAGE=10

# Hyperparameters & constants
TOP_PARAGRAPHS = 10
RANK_TOP_K = 10
MAX_ARGUMENTS = 4
SAVE_EVERY = 10

SAMPLE_LIMIT = 500  # set to None to run all samples
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_DOMINANCE_MARGIN = 0.20


DIVERSITY_LAMBDA = 0.7
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

from classes.utils import *

RUN_DIR = Path("runs") / f"{safe_reason_model}" / f"args_{MAX_ARGUMENTS}" / f"paras_{RANK_TOP_K}"
RESULTS_DIR = RUN_DIR / "results"
GRAPH_DIR = RUN_DIR / "graph"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# Save hyperparameters
hyperparams = {
    "TOP_PARAGRAPHS": TOP_PARAGRAPHS,
    "MAX_ARGUMENTS": MAX_ARGUMENTS,
    "SAVE_EVERY": SAVE_EVERY,
    "SAMPLE_LIMIT": SAMPLE_LIMIT,
    "DEFAULT_CONFIDENCE_THRESHOLD": DEFAULT_CONFIDENCE_THRESHOLD,
    "DEFAULT_DOMINANCE_MARGIN": DEFAULT_DOMINANCE_MARGIN,
    "RANK_TOP_K": RANK_TOP_K,
    "DIVERSITY_LAMBDA": DIVERSITY_LAMBDA,
    "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
    "WIKI_MODEL": WIKI_MODEL,
    "REASON_MODEL": REASON_MODEL,
}
