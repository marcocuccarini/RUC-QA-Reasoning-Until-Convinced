# ==========================================
# FILE 1: README.md
# ==========================================

# 🤖 Bipolar Argumentation-based RAG System

This project implements a sophisticated **Reasoning-based Question Answering** pipeline. Unlike standard RAG systems that provide a single text response, this system builds a **formal argumentation graph** to evaluate multiple hypotheses (answers). It uses local LLMs (via Ollama) and Argumentation Theory to determine the most logically sound answer based on evidence retrieved from Wikipedia.



---

## 📂 Project Structure & Components

### 🧠 Core Engine (`classes/`)
* **`ServerOllama.py`**: Manages local Ollama server connections and chat sessions.
* **`LLMUser.py`**: High-level interface for argument extraction and **JSON Self-Healing**.
* **`ArgumentationGraph.py`**: The mathematical core. Builds the graph and solves the Quadratic Energy Model.
* **`utils.py`**: Infrastructure for semantic ranking (Sentence-BERT) and Wikipedia fetching.

### 🧪 Pipeline & Evaluation Scripts
1. **`pipeline_test.py` & `pipeline_val.py`**: Orchestrate full runs on Test/Val sets using optimized parameters.
2. **`wikipedia.py`**: Fetches and caches Wikipedia content to `results_wikipedia_only/` for offline use.
3. **`baseline.py`**: Runs **Zero-Shot** and **Standard RAG** to establish a performance floor.
4. **`evaluation_baseline.py` & `evaluation_pipeline.py`**: Calculate accuracy and generate statistical charts comparing Graph logic vs. Baseline.

---

## 💾 Understanding the Output Files

When you run the system, it generates structured data to document its reasoning. Here is how to interpret the results:

### 1. Argumentation Graphs (`graphs/`)
For every question, the system generates:
* **`graph_N.png`**: A visual map of the logic. **Green dashed arrows** represent support, and **red solid arrows** represent attacks.
* **`graph_N.json`**: The raw data for the graph, including the final **strength** (probability of truth) for each node after the energy model resolution.


### 2. Predictions & Logs (`results/` & `results_retrieval/`)
* **`*_test_predictions.json`**: A list of all questions processed, showing the model's choice vs. the ground truth.
* **`*_wiki_pages.json`**: A local cache of retrieved Wikipedia paragraphs. This ensures that repeated experiments do not require new internet requests.

### 3. Analytics (`results_retrieval/`)
* **Baseline Logs**: These files compare the accuracy of the LLM in "Zero-shot" mode (no help) versus "Standard RAG" (plain text help). This is the key to proving that your Argumentation Graph actually improves performance.

---

## 📊 Experimental Workflow
To replicate the study, follow this order:
1. **Harvest**: `python wikipedia.py` (Cache Wikipedia content).
2. **Tune**: `python pipeline_val.py` (Find best hyperparameters).
3. **Benchmark**: `python baseline.py` (Get baseline accuracy).
4. **Test**: `python pipeline_test.py` (Run the Argumentation logic).
5. **Analyze**: `python evaluation_pipeline.py` (Generate final reports and charts).


---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Ollama** installed and the following models pulled:
```bash
ollama pull qwen3:14b
ollama pull gpt-oss:20b
```



### 1. Prerequisites


git clone [https://github.com/tuo-username/arg-rag-system.git](https://github.com/tuo-username/arg-rag-system.git)
cd arg-rag-system
pip install -r requirements.txt


### 2. Installation

git clone [https://github.com/marcocuccarini/RUC-QA-Reasoning-Until-Convinced.git](https://github.com/marcocuccarini/RUC-QA-Reasoning-Until-Convinced.git)
cd RUC-QA-Reasoning-Until-Convinced
pip install -r requirements.txt

