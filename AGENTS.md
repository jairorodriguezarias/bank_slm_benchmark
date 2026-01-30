# Agent Handover & Project Status

## Project Overview
This project benchmarks 5 Small Language Models (SLMs) using the Hugging Face `transformers` library on a specific Bank Customer Support use case.

## Current Status
- **Environment:** Python 3.11+ with `torch`, `transformers`, `accelerate`, and `pandas`.
- **Infrastructure:** Configured to run on `mps` (Apple Silicon) if available, falling back to `cpu`.
- **Dataset:** 5 synthetic banking queries covering Security, Accounts, Loans, Digital Banking, and Fees.
- **Models Benchmarked:**
    - `Qwen/Qwen2-1.5B-Instruct`
    - `Qwen/Qwen1.5-0.5B-Chat`
    - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
    - `HuggingFaceTB/SmolLM-1.7B-Instruct`
    - `facebook/opt-1.3b`
- **Output:** Individual CSVs per model and a consolidated `all_models_benchmark.csv` in the `results/` folder. A `BENCHMARK_REPORT.md` is generated with performance stats.

## Technical Decisions
- **Path Handling:** All scripts use `os.path` relative to `__file__` to ensure portability.
- **Memory Management:** Explicit `gc.collect()` and cache clearing (`torch.mps.empty_cache()`) are implemented between model swaps to prevent OOM errors on local hardware.
- **Padding:** Since many SLMs lack a `pad_token`, the scripts automatically patch the tokenizer to use `eos_token` as `pad_token`.

## Next Steps / Future Work
1. **Model Expansion:** Test even smaller models like `google/shieldgemma-2b` or `microsoft/phi-3-mini` (requires fixing the `rope_scaling` config issue noted in earlier iterations).
2. **Evaluation Metrics:** Implement automated scoring (e.g., ROUGE, BLEU, or LLM-as-a-judge) to compare model responses against the `reference_answer` in `bank_queries.json`.
3. **Quantization:** Add support for `bitsandbytes` (4-bit/8-bit) to test larger SLMs or improve speed.
4. **API Deployment:** Create a small FastAPI wrapper for the best-performing SLM found in the benchmark.

## How to Continue
- Run `python3 src/benchmark.py` to add new models (update the `MODELS_TO_TEST` list).
- Run `python3 src/analyze_results.py` to refresh the report after changes.
