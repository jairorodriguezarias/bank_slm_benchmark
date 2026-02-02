# Agent Handover & Project Status

## Project Overview
Benchmarks SLMs on a 95-query banking dataset using `transformers`, `rouge-score`, and `sentence-transformers`. **Includes a Distillation pipeline to improve SLM performance using DeepSeek-V3 as a teacher.**

## Current Status (As of Feb 02, 2026)
- **Project Unification:**
    - Merged `distillation_SLM` into this project.
    - Added `src/utils/generate_distillation_data.py` for generating synthetic training data from DeepSeek-V3.
    - Added `src/benchmark_distillation.py` (formerly `benchmark_comparison.py`) for specific teacher-student comparisons.
    - Archived legacy reports to `DISTILLATION_REPORT_ARCHIVE.md`.
- **Dataset Expansion:**
    - **Primary Set:** Expanded `data/bank_queries.json` from 95 to **195 high-quality queries** with detailed reference answers.
    - **Secondary Set:** Downloaded the **full Banking77 dataset** (13,069 queries) via `src/utils/download_banking77.py` and saved as `data/banking77_full.json`.
    - **Distilled Data:** Added `data/distilled_training_data.json` containing Chain-of-Thought (CoT) reasoning distilled from DeepSeek-R1/V3.
    - **Dataset Synthesis:** Added `src/utils/generate_dataset.py` to convert raw PDFs into high-quality instruction-output pairs for SFT.
- **Inference Pipeline:**
    - Unified benchmarking in `src/benchmark.py`, supporting standard HF models, 4-bit quantization, and GGUF models via `llama-cpp-python`.
    - Fixed multi-line f-string syntax in GGUF prompt generation.
- **Evaluation & Metrics:**
    - Implemented `src/evaluate.py` to calculate ROUGE-L and Semantic Similarity (Cosine similarity) metrics.
    - Automatically generates consolidated charts, updates `BENCHMARK_REPORT.md`, and produces a professional PDF report (`benchmark_report.pdf`).

## Key Files
- `src/benchmark.py`: Main unified benchmark script for all models (HF, GGUF).
- `src/evaluate.py`: Scoring and reporting (ROUGE, Similarity, PDF generation).
- `src/utils/generate_distillation_data.py`: Generates distilled training data from Teacher models.
- `src/train.py`: SFT/LoRA training script for model fine-tuning.
- `src/utils/`: Contains data downloaders and dataset generators.

## Technical Decisions
- **Consolidation:** Merged legacy scripts (`benchmark_gguf.py`, `visualize_results.py`, `evaluate_models.py`) into unified `src/benchmark.py` and `src/evaluate.py` to reduce complexity.
- **Evaluation Workflow:** Separated inference from scoring to allow re-evaluation without expensive re-generation.
- **GGUF Support:** Chose `llama-cpp-python` for Mac optimization (Metal) as `bitsandbytes` 4-bit loading is CUDA-only.

## Next Steps
1. **API Integration:** Wrap the top-performing model (TinyLlama GGUF or SFT) in a FastAPI service.
2. **Distillation Fine-Tuning:** Use `data/distilled_training_data.json` to fine-tune a raw Qwen2/Phi-3 model and evaluate if it matches the "Distilled" variants.
3. **Dataset Refinement:** Increase the number of reference answers per query to improve ROUGE reliability.
4. **Automated CI/CD:** Integrate evaluation into a single command or workflow for continuous benchmarking.

## How to Continue
- Run `./venv/bin/python src/benchmark.py` to execute standard benchmarks.
- Run `./venv/bin/python src/utils/generate_distillation_data.py` to generate new training data.
- Run `./venv/bin/python src/evaluate.py` to refresh reports.
- Check `results/latest/all_models_benchmark.csv` for the most up-to-date raw statistics.