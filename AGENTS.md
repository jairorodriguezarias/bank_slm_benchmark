# Agent Handover & Project Status

## Project Overview
Benchmarks SLMs on a 95-query banking dataset using `transformers`, `rouge-score`, and `sentence-transformers`.

## Current Status (As of Jan 30, 2026)
- **Dataset Expansion:**
    - **Primary Set:** Expanded `data/bank_queries.json` from 95 to **195 high-quality queries** with detailed reference answers.
    - **Secondary Set:** Downloaded the **full Banking77 dataset** (13,069 queries) via `src/utils/download_banking77.py` and saved as `data/banking77_full.json`.
    - **Dataset Synthesis:** Added `src/utils/generate_dataset.py` to convert raw PDFs into high-quality instruction-output pairs for SFT using local SLMs (e.g., Qwen2-1.5B) with Metal/CUDA acceleration.
- **Inference Pipeline:**
    - Unified benchmarking in `src/benchmark.py`, supporting standard HF models, 4-bit quantization, and GGUF models via `llama-cpp-python`.
    - Fixed multi-line f-string syntax in GGUF prompt generation.
- **Evaluation & Metrics:**
    - Implemented `src/evaluate.py` to calculate ROUGE-L and Semantic Similarity (Cosine similarity) metrics.
    - Automatically generates consolidated charts, updates `BENCHMARK_REPORT.md`, and produces a professional PDF report (`benchmark_report.pdf`).
- **Refactoring:**
    - Moved one-time utility scripts (`download_gguf.py`, `download_banking77.py`) to `src/utils/`.
    - Implemented project-relative absolute path handling in all utilities for improved robustness.

## Technical Decisions
- **Consolidation:** Merged legacy scripts (`benchmark_gguf.py`, `visualize_results.py`, `evaluate_models.py`) into unified `src/benchmark.py` and `src/evaluate.py` to reduce complexity.
- **Evaluation Workflow:** Separated inference from scoring to allow re-evaluation without expensive re-generation.
- **GGUF Support:** Chose `llama-cpp-python` for Mac optimization (Metal) as `bitsandbytes` 4-bit loading is CUDA-only.

## Next Steps
1. **API Integration:** Wrap the top-performing model (TinyLlama GGUF or SFT) in a FastAPI service.
2. **Dataset Refinement:** Increase the number of reference answers per query to improve ROUGE reliability.
3. **Automated CI/CD:** Integrate evaluation into a single command or workflow for continuous benchmarking.
4. **Fine-Tuning GGUF:** Explore if SFT adapters can be merged and converted to GGUF for the best of both worlds.

## How to Continue
- Run `./venv/bin/python src/benchmark.py` to execute benchmarks.
- Run `./venv/bin/python src/evaluate.py` to refresh the `BENCHMARK_REPORT.md` and plots after any result changes.
- Check `results/latest/all_models_benchmark.csv` for the most up-to-date raw statistics.
