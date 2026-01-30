# Agent Handover & Project Status

## Project Overview
Benchmarks SLMs on a 95-query banking dataset using `transformers`, `rouge-score`, and `sentence-transformers`.

## Current Status (As of Jan 30, 2026)
- **Dataset:** Expanded from 5 to 95 queries in `data/bank_queries.json`.
- **Metrics Implemented:**
    - **Performance:** Latency (s), Tokens Generated, Tokens Per Second (TPS).
    - **Quality:** ROUGE-L (F-measure) and Semantic Similarity (Cosine similarity using `sentence-transformers`).
- **Models Benchmarked:** 5 models ranging from 0.5B to 1.7B parameters.
- **Key Findings:**
    - `TinyLlama/TinyLlama-1.1B-Chat-v1.0` currently leads in Semantic Similarity (0.725).
    - `Qwen/Qwen1.5-0.5B-Chat` is the fastest at ~30 Tokens/Sec.
- **Quantization:** A `src/benchmark_quantized.py` script was added using `bitsandbytes`. It is verified for CUDA environments but currently incompatible with the local Mac MPS backend for 4-bit loading.

## Technical Decisions
- **Evaluation Workflow:** Separated inference (`benchmark.py`) from scoring (`evaluate_models.py`) to allow re-evaluation without expensive re-generation.
- **Metrics Backfilling:** Created a temporary script to update existing results with TPS and token counts by re-tokenizing decoded responses.
- **Dependencies:** Added `rouge-score`, `sentence-transformers`, `evaluate`, `scikit-learn`, `tabulate`, and `bitsandbytes`.

## Next Steps
1. **Mac Quantization:** Implement GGUF support via `llama-cpp-python` to test quantization on Apple Silicon.
2. **Visualizations:** Create a script to generate charts (e.g., Latency vs. Accuracy scatter plot).
3. **Dataset Refinement:** Increase the number of reference answers per query to improve ROUGE reliability.
4. **API Integration:** Wrap the top-performing model (`TinyLlama`) in a FastAPI service.

## How to Continue
- Run `python3 src/evaluate_models.py` to refresh the `BENCHMARK_REPORT.md` after any result changes.
- Check `results/all_models_benchmark.csv` for the most up-to-date raw statistics.