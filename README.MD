🧪 Pipeline & Evaluation Scripts
While the classes/ folder contains the engine, these scripts manage the execution of large-scale experiments and the analysis of results.

1. pipeline_test.py & pipeline_val.py (The Executioners)

These scripts are responsible for running the full argumentation pipeline on test and validation sets.

pipeline_test.py: Executes the final evaluation using the BEST_PARAMS identified during search. It handles sampling (up to 900 questions) and saves detailed predictions to results/.

pipeline_val.py: Implements Grid Search logic. It iterates through different combinations of TOP_PARAGRAPHS and MAX_ARGUMENTS to find the configuration that yields the highest accuracy on the validation split.

2. wikipedia.py (The Knowledge Harvester)

This script is a specialized utility to build a local knowledge base.

It iterates through datasets and uses LLMUser.get_candidate_pages to find relevant Wikipedia titles.

It fetches the full content and stores it in results_wikipedia_only/.

Benefit: This allows you to run multiple LLM experiments without hitting the Wikipedia API repeatedly, significantly increasing speed and reliability.

3. baseline.py (The Control Group)

This script establishes a performance baseline without the argumentation graph.

It asks the LLM to answer questions using two methods:

Zero-Shot: No context provided.

Standard RAG: Context provided as plain text, but without building a logical graph.

This allows you to quantify exactly how much "accuracy boost" the Argumentation Graph provides compared to standard methods.

4. evaluation_baseline.py & evaluation_pipeline.py (The Analysts)

These scripts transform raw JSON predictions into readable metrics and visualizations.

evaluation_baseline.py: Calculates accuracy for the baseline tests, including filters for "INVALID" responses.

evaluation_pipeline.py: A specialized diagnostic tool. It compares questions where the Graph Changed (logic was successfully applied) vs. questions where the Graph remained Unchanged (leading to a random choice). It generates Matplotlib charts to visualize these performance deltas.

📊 Experimental Workflow Summary
To replicate the full study, follow this order:

Harvest Knowledge: Run wikipedia.py to cache Wikipedia content.

Tune: Run pipeline_val.py to find the best hyperparameters.

Benchmark: Run baseline.py to see how the LLM performs normally.

Test: Run pipeline_test.py to execute the Bipolar Argumentation RAG.

Analyze: Use the evaluation_ scripts to generate final accuracy reports and charts.

📂 Updated Directory Map

Plaintext
src/
├── classes/               # Core Engine
├── split_datasets/        # Train/Val/Test JSON splits
├── results_retrieval/     # Logs for Baseline & Standard RAG
├── results/               # Final Predictions (Argumentation)
├── graphs/                # Visual Export of BAG logic
├── pipeline_test.py       # Final Test Runner
├── pipeline_val.py        # Hyperparameter Optimizer
├── wikipedia.py           # Content Pre-fetcher
└── evaluation_pipeline.py # Statistical Plotting
Ready for GitHub?

Everything is now documented! Would you like me to generate a requirements.txt file based on all these scripts, or perhaps a summary of the results format so you can fill in your final accuracy numbers?