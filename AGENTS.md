# Agent Handover & Project Status

## Project Overview
Benchmarks SLMs on a 95-query banking dataset using `transformers`, `rouge-score`, and `sentence-transformers`. **Includes a Distillation pipeline to improve SLM performance using DeepSeek-V3 as a teacher.**

## Current Status (As of March 8, 2026)
- **Zero-Shot Knowledge Injection (Doc-to-LoRA):**
    - Implemented `src/8_doc_to_lora_injection.py` utilizing Sakana AI's Hypernetwork methodology.
    - Enables near-instant model adaptation by predicting LoRA weights directly from raw documents, bypassing backpropagation.
    - Integrated automatic checkpoint downloading from HuggingFace and a disk cleanup utility for large (~5GB) model binaries.
    - *Note:* This component is engineered for NVIDIA/Colab environments due to `triton` hardware requirements.
- **Academic Thesis Completion:**
    - Finalized a comprehensive PhD-level LaTeX document (`ACADEMIC/bank_slm_benchmark_ACADEMIC.tex`).
    - Added a novel chapter on "Zero-Shot Knowledge Injection," detailing the transition from batch SFT to real-time weight prediction.
    - Formalized mathematical paradigms for LoRA, DPO, ORPO, and TIES-merging.
- **Pipeline Parameterization:**
    - Refactored `run_pipeline.sh` to extract all hardcoded configurations into top-level variables (`BASE_MODEL`, `MODEL_PREFIX`, `DATASET`).
    - Standardized end-to-end automation to allow seamless switching between SLM architectures (e.g., TinyLlama, Qwen, Gemma).
- **Preference Tuning (DPO) Integration:**
    - Added Direct Preference Optimization (DPO) to the training pipeline to align SLMs with desired response formats.
    - Created `src/utils/generate_dpo_data.py` to synthetically generate "rejected" answers using the Gemini API.
    - Implemented `src/dpo_train.py` using `trl` to perform DPO tuning on top of SFT models.
    - Updated `run_pipeline.sh` to dynamically detect DPO data, run preference tuning, and benchmark SFT vs DPO models side-by-side.
- **Data Pipeline Unification:**
    - Consolidated various data sources (PDFs, Search, Banking77) into a single, high-density training file: `data/train_final_5500.jsonl`.
- Added an automated 90/10 train/test split within `src/sft_train.py`, which automatically saves the test set to `data/train_final_5500_test.jsonl` for unbiased benchmarking.
    - Successfully executed full benchmark runs.
    - Updated `src/benchmark.py` to include **MobileLLaMA-1.4B** and **Gemma-2b** in the default model list for future runs.
    - Generated a comprehensive PDF report in `results/latest/benchmark_report.pdf`.
- **Project Unification:**
    - Created `run_pipeline.sh` to provide a single-command execution for the entire workflow (Training -> DPO -> Benchmarking -> Evaluation).
    - Merged `distillation_SLM` into this project.
    - Added `src/utils/generate_distillation_data.py` for generating synthetic training data from DeepSeek-V3/Gemini.
    - Refactored `src/sft_train.py` to support both original and distilled dataset schemas.
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
- `src/8_doc_to_lora_injection.py`: Zero-shot weight injection via Doc-to-LoRA hypernetworks.
- `src/sft_train.py`: SFT/LoRA training script for model fine-tuning.
- `src/dpo_train.py`: Direct Preference Optimization (DPO) script for alignment.
- `run_pipeline.sh`: Shell script to execute the end-to-end process (SFT -> DPO -> Eval).
- `src/utils/generate_distillation_data.py`: Generates distilled training data from Teacher models.
- `src/utils/generate_dpo_data.py`: Synthesizes 'rejected' answers via Gemini for DPO training.
- `src/utils/`: Contains data downloaders and dataset generators.

## Technical Decisions
- **Zero-Shot Adaptation:** Chose Doc-to-LoRA hypernetworks over RAG or standard SFT for real-time document internalization, allowing the model to project factual knowledge into weights without iterative training.
- **Unified Dataset:** Standardized all training on `train_final_5500.jsonl` to ensure consistency and prevent data leakage during benchmarking.
- **Preference Tuning over PPO:** Chose Direct Preference Optimization (DPO) over RLHF/PPO as it does not require a separate reward model, making it feasible for local SLM workflows.
- **Consolidation:** Merged legacy scripts (`benchmark_gguf.py`, `visualize_results.py`, `evaluate_models.py`) into unified `src/benchmark.py` and `src/evaluate.py` to reduce complexity.
- **Evaluation Workflow:** Separated inference from scoring to allow re-evaluation without expensive re-generation.
- **GGUF Support:** Chose `llama-cpp-python` for Mac optimization (Metal) as `bitsandbytes` 4-bit loading is CUDA-only.

## Next Steps
1. **API Integration:** Wrap the top-performing model (TinyLlama GGUF or SFT/DPO) in a FastAPI service.
2. **DPO Tuning Execution:** Run `src/utils/generate_dpo_data.py` to generate the preference dataset, and execute the full pipeline to evaluate SFT vs DPO outputs.
3. **Dataset Refinement:** Increase the number of reference answers per query to improve ROUGE reliability.
4. **Automated CI/CD:** Integrate evaluation into a single command or workflow for continuous benchmarking.

## How to Continue
- Run `./run_pipeline.sh` to execute the full training, benchmarking, and evaluation workflow.
- Run `./venv/bin/python src/8_doc_to_lora_injection.py` on a GPU-enabled environment (e.g., Colab) to test zero-shot injection.
- Run `./venv/bin/python src/benchmark.py` to execute standard benchmarks manually.
- Run `./venv/bin/python src/utils/generate_distillation_data.py` to generate new training data.
- Run `./venv/bin/python src/evaluate.py` to refresh reports.
- Check `results/latest/all_models_benchmark.csv` for the most up-to-date raw statistics.