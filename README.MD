# 🤖 Bipolar Argumentation-based RAG System

This project implements a sophisticated **Reasoning-based Question Answering** pipeline. Unlike standard RAG systems that provide a single text response, this system builds a **formal argumentation graph** to evaluate multiple hypotheses (answers). It uses local LLMs (via Ollama) and Argumentation Theory to determine the most logically sound answer based on evidence retrieved from Wikipedia.



## 📂 Project Structure

### 🧠 Core Engine (`classes/`)
* **`ServerOllama.py`**: Manages the local connection to the Ollama server.
* **`LLMUser.py`**: High-level interface for argument extraction and **JSON Self-Healing**.
* **`ArgumentationGraph.py`**: The mathematical core. Builds the graph and solves the Quadratic Energy Model.
* **`PromptBuilder.py`**: Centralized prompt engineering for all LLM tasks.
* **`utils.py`**: Utility functions for Wikipedia API fetching and Sentence-BERT ranking.
* **`running.py`**: Main orchestrator for single-run experiments.


### 🚀 Execution Pipelines
* **`pipeline_test.py`**: Runs final tests on the `test` split using optimized parameters.
* **`pipeline_val.py`**: Evaluation engine for **Hyperparameter Grid Search**.
* **`wikipedia.py`**: Knowledge harvester to pre-fetch and cache Wikipedia content.
* **`baseline.py`**: Runs control group tests (Zero-shot vs Standard RAG).

### 📊 Evaluation & Analysis
* **`evaluation_baseline.py`**: Calculates accuracy for baseline and standard RAG tests.
* **`evaluation_pipeline.py`**: Analyzes graph performance and generates statistical charts.

---

## ⚙️ How It Works

1.  **Semantic Retrieval**: Paragraphs are ranked using `all-mpnet-base-v2` embeddings.
2.  **Argument Mining**: The LLM extracts factual evidence and assigns a **Support** or **Attack** relation.
3.  **Graph Resolution**: A Bipolar Argumentation Graph is solved using a **Quadratic Energy Model** (via RK4) to determine the winning hypothesis.



## 🛠 Self-Healing JSON Mechanism
One of the key features of the `LLMUser` class is its resilience. If a local model produces malformed JSON during extraction, the system automatically triggers a **Repair Agent** (e.g., Qwen3) to fix the formatting, ensuring the pipeline continues without interruption.

---

## 🚀 Getting Started

### . Prerequisites
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

